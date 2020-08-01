import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.nn.utils import spectral_norm as snorm


'''
Classifier for categorizing the adversarial examples, (similar to the classifier in MagNet: https://arxiv.org/pdf/1705.09064.pdf)
MNIST accuracy: 0.993100
FashionMNIST accuracy: 0.926600
'''
class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.name = 'Classifier'
        self.image_size = args.image_size
        self.image_channels = args.image_channels
        self.conv1 = nn.Conv2d(self.image_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(3136, 200)
        self.fc2 = nn.Linear(200, 10)
    
    def main(self, x):
        self.batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(self.batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        self.batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(self.batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class OPVAE(nn.Module):
    def __init__(self, leaky=True):
        super(OPVAE, self).__init__()

        # Encode
        self.fc1 = nn.Linear(784, 500)
        self.mu = nn.Linear(500, 20)
        self.sigma = nn.Linear(500, 20)

        # Decode
        self.fc3 = nn.Linear(20, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 784)

        self.nl = nn.LeakyReLU(0.1) if leaky else nn.ReLU()

    def encode(self, x):
        x = self.nl(self.fc1(x))
        mean, var = self.mu(x), self.sigma(x)
        return Normal(mean, var)

    def decode(self, z):
        x = self.nl(self.fc3(z))
        x = self.nl(self.fc4(x))
        x = F.sigmoid(self.fc5(x))

        return x

    def forward(self, x):
        dist = self.encode(x)
        z = dist.rsample()
        output = self.decode(z)

        return output, dist.mean, dist.stddev, z


# Class for convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=0, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim,
                              out_dim,
                              kernel_size,
                              stride,
                              padding,
                              bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(False)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


# Class for de-convolution block
class DeConvBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel_size=0,
                 stride=1,
                 padding=0,
                 out_padding=0):
        super(DeConvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_dim,
                                       out_dim,
                                       kernel_size,
                                       stride,
                                       padding,
                                       out_padding,
                                       bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(False)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


# Main class for MAD-VAE
class MADVAE(nn.Module):
    def __init__(self, args):
        super(MADVAE, self).__init__()
        self.model_name = 'DAD-VAE'
        self.image_size = args.image_size
        self.image_channels = args.image_channels
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim
        # module for encoder
        self.c1 = ConvBlock(self.image_channels, 64, 5, 1, 2)
        self.c2 = ConvBlock(64, 64, 4, 2, 3)
        self.c3 = ConvBlock(64, 128, 4, 2, 1)
        self.c4 = ConvBlock(128, 256, 4, 2, 1)
        self.e_module = nn.Sequential(self.c1, self.c2, self.c3, self.c4)
        self.mu = nn.Linear(self.h_dim, self.z_dim)
        self.sigma = nn.Linear(self.h_dim, self.z_dim)
        # module for image decoder
        self.linear = nn.Linear(self.z_dim, self.h_dim)
        self.d1 = DeConvBlock(256, 128, 4, 2, 1)
        self.d2 = DeConvBlock(128, 64, 4, 2, 1)
        self.d3 = DeConvBlock(64, 64, 4, 2, 3)
        self.d4 = nn.ConvTranspose2d(64,
                                     self.image_channels,
                                     5,
                                     1,
                                     2,
                                     bias=False)
        self.img_module = nn.Sequential(self.d1, self.d2, self.d3, self.d4)

    # Encoder
    def encode(self, x):
        self.batch_size = x.size(0)
        x = self.e_module(x)
        x = x.view(self.batch_size, -1)
        mean = self.mu(x)
        var = self.sigma(x)
        distribution = Normal(mean, var)

        return distribution

    # Decoder for image denoising
    def img_decode(self, z):
        self.batch_size = z.size(0)
        x = F.relu(self.linear(z), inplace=False)
        x = x.view(self.batch_size, 256, 4, 4)

        return F.sigmoid(self.img_module(x))

    # Forward function
    def forward(self, x):
        dist = self.encode(x)
        if self.training == True:
            z = dist.rsample()
        else:
            z = dist.mean
        output = self.img_decode(z)

        return output, dist.mean, dist.stddev, z


'''
Reference to the paper Adversarial Defense by Restricting the Hidden 
Space of Deep Neural Networks at https://github.com/aamir-mustafa/pcl-adversarial-defense
'''
# proximity loss for the z
class Proximity(nn.Module):
    def __init__(self, args):
        super(Proximity, self).__init__()
        self.num_classes = args.num_classes
        self.z_dim = args.z_dim
        self.use_gpu = args.use_gpu
        self.centers = torch.randn(self.num_classes, self.z_dim)
        self.classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            self.centers = self.centers.to(device)
            self.classes = self.classes.to(device)
        self.centers = nn.Parameter(self.centers)

    def forward(self, x, labels):
        # calculate the distance between x and centers
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).T
        distmat.addmm_(1, -2, x, self.centers.T)

        # get matrix masks
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(self.classes.expand(batch_size, self.num_classes))

        # calculate distance for each batch
        dist = list()
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = torch.sqrt(value)
            value = value.clamp(min=1e-12, max=1e+12)
            dist.append(value)
        losses = torch.cat(dist)
        loss = losses.mean()

        return loss


# distance loss for the z
class Distance(nn.Module):
    def __init__(self, args):
        super(Distance, self).__init__()
        self.num_classes = args.num_classes
        self.z_dim = args.z_dim
        self.use_gpu = args.use_gpu
        self.centers = torch.randn(self.num_classes, self.z_dim)
        self.classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            self.centers = self.centers.to(device)
            self.classes = self.classes.to(device)
        self.centers = nn.Parameter(self.centers)

    def forward(self, x, labels):
        # calculate the distance between x and centers
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).T
        distmat.addmm_(1, -2, x, self.centers.T)

        # get matrix masks
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(self.classes.expand(batch_size, self.num_classes))

        # calculate loss
        dist = list()
        for i in range(batch_size):
            k = mask[i].clone().to(dtype=torch.int8)
            k = -1 * k + 1
            kk = k.clone().to(dtype=torch.uint8)
            value = distmat[i][kk]
            value = torch.sqrt(value)
            value = value.clamp(min=1e-8, max=1e+8)  # for numerical stability
            dist.append(value)
        losses = torch.cat(dist)
        loss = losses.mean()

        return loss