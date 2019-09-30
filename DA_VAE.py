import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.lrelu1 = nn.LeakyReLU(0.2, True)
        self.pad2 = nn.ReplicationPad2d(padding)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, stride)
        self.norm2 = nn.InstanceNorm2d(dim)
        self.lrelu2 = nn.LeakyReLU(0.2, True)
        self.module = nn.Sequential(self.pad1, self.conv1, self.norm1, self.lrelu1, self.pad2, self.conv2, self.norm2, self.lrelu2)
    
    def forward(self, x):
        return x + self.module(x)

# Main class for DA-VAE
class DAVAE(nn.Module):
    def __init__(self, args):
        super(DAVAE, self).__init__()
        self.model_name = 'DA-VAE'
        self.image_size = args.image_size
        self.image_channels = args.image_channels
        self.h_dim = args.h_dim
        self.z1_dim = args.z1_dim
        self.z2_dim = args.z2_dim
        # module for encoder
        self.c1 = ConvBlock(self.image_channels, 32, 7, 4, 3) # 256*256*3 -> 64*64*32
        self.c2 = ConvBlock(32, 64, 3, 2, 1) # 64*64*32 -> 32*32*64
        self.c3 = ConvBlock(64, 128, 3, 2, 1) # 32*32*64 -> 16*16*128
        self.r1 = ResidualConvBlock(128, 3, 1, 1) # 16*16*128 -> 16*16*128
        self.c4 = ConvBlock(128, 64, 3, 2, 1) # 16*16*128 -> 8*8*64
        self.c5 = ConvBlock(64, 32, 3, 2, 1) # 8*8*64 -> 4*4*32
        self.mu =nn.Linear(self.h_dim, self.z1_dim + self.z2_dim)
        self.sigma = nn.Linear(self.h_dim, self.z1_dim + self.z2_dim)
        # module for image decoder
        self.linear1 = nn.Linear(self.z1_dim, self.h_dim)
        self.d1 = DeConvBlock(32, 64, 7, 4, 3)
        self.d2 = DeConvBlock(64, 128, 3, 2, 1)
        self.r2 = ResidualConvBlock(128, 3, 1, 1)
        self.r3 = ResidualConvBlock(128, 3, 1, 1)
        self.r4 = ResidualConvBlock(128, 3, 1, 1)
        self.d3 = DeConvBlock(128, 64, 3, 2, 1)
        self.d4 = DeConvBlock(64, 32, 3, 2, 1)
        self.d5 = DeConvBlock(32, self.image_channels, 3, 2, 1)
        # module for adversarial decoder
        self.linear2 = nn.Linear(self.z2_dim, self.h_dim)
        self.d6 = DeConvBlock(128, 64, 7, 4, 3)
        self.d7 = DeConvBlock(64, 64, 3, 2, 1)
        self.r5 = ResidualConvBlock(64, 3, 3, 1)
        self.d8 = DeConvBlock(64, 32, 7, 4, 3)
        self.d9 = DeConvBlock(32, self.image_channels, 3, 2, 1)

    # Encoder
    def encode(self, x):
        self.e_module = nn.Sequential(self.c1, self.c2, self.c3, self.r1, self.c4, self.c5)
        x = self.e_module(x)
        x = x.view(-1, 2*2*128)
        mean = self.mu(x)
        var = self.sigma(x)
        distribution_1 = Normal(mean[:, :self.z1_dims], var[:, :self.z1_dims])
        distribution_2 = Normal(mean[:, self.z1_dims:], var[:, self.z1_dims:])

        return distribution_1, distribution_2
    
    # Decoder for adversarial features
    def adv_decode(self, z):
        x = self.linear2(z)
        x = x.view(-1, 32, 4, 4)
        self.adv_module = nn.Sequential(self.d6, self.d7, self.r5, self.d8, self.d9)

        return self.adv_module(x)
    
    # Decoder for image denoising
    def img_decode(self, z):
        x = self.linear1(z)
        x = x.view(-1, 32, 4, 4)
        self.img_module = nn.Sequential(self.d1, self.d2, self.r2, self.r3, self.r4, self.d3, self.d4, self.d5)

        return self.img_module(x)
    
    # Forward function
    def forward(self, x):
        dist1, dist22 = self.encode(x)
        z1 = dist1.rsample()
        z2 = dist2.rsample()
        output = self.img_decode(z1)
        adv_output = self.adv_decode(z2)
        
        return output, adv_output, dist1.mean, dist1.stddev, dist2.mean, dist2.stddev