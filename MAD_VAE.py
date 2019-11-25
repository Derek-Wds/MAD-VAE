import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

# Class for convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=0, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x

# Class for de-convolution block
class DeConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=0, stride=1, padding=0, out_padding=0):
        super(DeConvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, out_padding)
        self.norm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x

# Class for residual conv block
class ResidualConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=0, stride=1, padding=0):
        super(ResidualConvBlock, self).__init__()
        self.pad1 = nn.ReplicationPad2d(padding)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, stride)
        self.norm1 = nn.InstanceNorm2d(dim)
        self.relu1 = nn.ReLU(True)
        self.pad2 = nn.ReplicationPad2d(padding)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, stride)
        self.norm2 = nn.InstanceNorm2d(dim)
        self.relu2 = nn.ReLU(True)
        self.module = nn.Sequential(self.pad1, self.conv1, self.norm1, self.relu1, self.pad2, self.conv2, self.norm2, self.relu2)
    
    def forward(self, x):
        return x + self.module(x)

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
        self.c4 = ConvBlock(128, 128, 4, 2, 1)
        self.e_module = nn.Sequential(self.c1, self.c2, self.c3, self.c4)
        self.mu =nn.Linear(self.h_dim, self.z_dim)
        self.sigma = nn.Linear(self.h_dim, self.z_dim)
        # module for image decoder
        self.linear = nn.Linear(self.z_dim, self.h_dim)
        self.d1 = DeConvBlock(128, 128, 4, 2, 1)
        self.d2 = DeConvBlock(128, 64, 4, 2, 1)
        self.d3 = DeConvBlock(64, 64, 4, 2, 3)
        self.d4 = DeConvBlock(64, self.image_channels, 5, 1, 2)
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
        x = self.linear(z)
        x = x.view(self.batch_size, 128, 4, 4)

        return F.sigmoid(self.img_module(x))
    
    # Forward function
    def forward(self, x):
        dist = self.encode(x)
        z = dist.rsample()
        output = self.img_decode(z)
        
        return output, dist.mean, dist.stddev, z