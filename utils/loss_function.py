import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal

use_cuda=True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# recon loss function with 2 reconstruction loss and two KL-divergence loss
def recon_loss_function(recon, target, adv_recon, adv_target,
                  distribution_1, distribution_2, step, beta=1):
    adv_target = adv_target - target
    CE1 = F.mse_loss(
        recon.view(-1, recon.size(-1)),
        target,
        reduction='elementwise_mean')
    CE2 = F.mse_loss(
        adv_recon,
        adv_target.detach(),
        reduction='elementwise_mean')
    normal1 = Normal(
        torch.zeros(distribution_1.mean.size()).to(device),
        torch.ones(distribution_1.stddev.size()).to(device))
    normal2 = Normal(
        torch.zeros(distribution_2.mean.size()).to(device),
        torch.ones(distribution_2.stddev.size()).to(device))
    KLD1 = kl_divergence(distribution_1, normal1).mean()
    KLD2 = kl_divergence(distribution_2, normal2).mean()
    
    return CE1 + CE2 + beta * (KLD1 + KLD2), CE1, CE2