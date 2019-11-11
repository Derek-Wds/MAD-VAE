import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal

use_cuda=True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# recon loss function with 2 reconstruction loss and two KL-divergence loss
def recon_loss_function(recon, target, distribution, step, beta=1):
    CE = F.binary_cross_entropy(
        recon.view(-1, recon.size(-1)),
        target.view(-1, recon.size(-1)),
        reduction='mean')
    normal = Normal(
        torch.zeros(distribution.mean.size()).to(device),
        torch.ones(distribution.stddev.size()).to(device))
    KLD = kl_divergence(distribution, normal).mean()
    
    return CE + beta * KLD, CE, KLD

# classification loss function
def classification_loss(recon, label, classifier):
    criterion = nn.CrossEntropyLoss()
    # get output
    output = classifier(recon)
    # calculate loss
    loss = criterion(output, label)

    return loss

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
            k = -1 * k +1
            kk = k.clone().to(dtype=torch.uint8)
            value = distmat[i][kk]
            value = torch.sqrt(value)
            value = value.clamp(min=1e-8, max=1e+8) # for numerical stability
            dist.append(value)
        losses = torch.cat(dist)
        loss = losses.mean()

        return loss