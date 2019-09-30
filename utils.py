import foolbox
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils import data
from test.attacks import simba_attack

# scheduler to decay the learning rate
class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

# function for construct adversarial images
def add_adv(image, adv):
    return image, adv

# loss function with 2 reconstruction loss and two KL-divergence loss
def loss_function(recon, target, adv_recon, adv_target,
                  distribution_1, distribution_2, step, beta=1):
    CE1 = F.cross_entropy(
        recon.view(-1, recon.size(-1)),
        target,
        reduction='elementwise_mean')
    CE2 = F.cross_entropy(
        adv_recon.view(-1, adv_recon.size(-1)),
        adv_target,
        reduction='elementwise_mean')
    normal1 = Normal(
        torch.zeros(distribution_1.mean.size()).cuda(),
        torch.ones(distribution_1.stddev.size()).cuda())
    normal2 = Normal(
        torch.zeros(distribution_2.mean.size()).cuda(),
        torch.ones(distribution_2.stddev.size()).cuda())
    KLD1 = kl_divergence(distribution_1, normal1).mean()
    KLD2 = kl_divergence(distribution_2, normal2).mean()
    
    return CE1 + CE2 + beta * (KLD1 + KLD2)
