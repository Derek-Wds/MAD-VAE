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
        self.c1 = ConvBlock(self.image_channels, 32, 3, 2, 1) # 32*32*3 -> 16*16*32
        self.c2 = ConvBlock(32, 64, 3, 2, 1) # 16*16*32 -> 8*8*64
        self.r1 = ResidualConvBlock(64, 3, 1, 1) # 8*8*64 -> 8*8*64
        self.c3 = ConvBlock(64, 128, 3, 2, 1) # 8*8*64 -> 4*4*128
        self.c4 = ConvBlock(128, 128, 3, 2, 1) # 4*4*128 -> 2*2*128
        self.mu =nn.Linear(self.h_dim, self.z1_dim + self.z2_dim)
        self.sigma = nn.Linear(self.h_dim, self.z1_dim + self.z2_dim)
        # module for image decoder
        self.linear1 = nn.Linear(self.z1_dim, self.h_dim)
        self.d1 = DeConvBlock(128, 128, 3, 2, 1)
        self.d2 = DeconvBlock(128, 64, 3, 2, 1)
        self.r2 = ResidualConvBlock(64, 3, 1, 1)
        self.r3 = ResidualConvBlock(64, 3, 1, 1)
        self.r4 = ResidualConvBlock(64, 3, 1, 1)
        self.d3 = DeConvBlock(64, 32, 3, 2, 1)
        self.d4 = DeconvBlock(32, self.image_channels, 3, 2, 1)
        # module for adversarial decoder
        self.linear2 = nn.Linear(self.z2_dim, self.h_dim)
        self.d5 = DeConvBlock(128, 64, 7, 4, 3)
        self.r5 = ResidualConvBlock(64, 3, 3, 1)
        self.d6 = DeConvBlock(64, self.image_channels, 7, 4, 3)

    # Encoder
    def encode(self, x):
        self.e_module = nn.Sequential(self.c1, self.c2, self.r1, self.c3, self.c4)
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
        x = x.view(-1, 128, 2, 2)
        self.adv_module = nn.Sequential(self.d5, self.r5, self.d6)
        return self.adv_module(x)
    
    # Decoder for image denoising
    def img_decode(self, z):
        x = self.linear1(z)
        x = x.view(-1, 128, 2, 2)
        self.img_module = nn.Sequential(self.d1, self.d2, self.r2, self.r3, self.r4, self.d3, self.d4)
        return self.img_module(x)
    
    # Forward function
    def forward(self, x):
        z1, z2 = self.encode(x)
        z1 = z1.rsample()
        z2 = z2.rsample()
        output = self.img_decode(z1)
        adv_output = self.adv_decode(z2)
        return output, adv_output