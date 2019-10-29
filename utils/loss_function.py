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
        reduction='elementwise_mean')
    normal = Normal(
        torch.zeros(distribution.mean.size()).to(device),
        torch.ones(distribution.stddev.size()).to(device))
    KLD = kl_divergence(distribution, normal).mean()
    
    return CE + beta * KLD, CE, KLD