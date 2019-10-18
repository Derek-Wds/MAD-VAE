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
    CE1 = F.binary_cross_entropy(
        recon.view(-1, recon.size(-1)),
        target,
        reduction='elementwise_mean')
    CE2 = F.binary_cross_entropy(
        adv_recon.view(-1, adv_recon.size(-1)),
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

# classfication loss function with classification on output and adv_output
def classification_loss_function(model, adv_data, output, adv_output):
    criterion = nn.CrossEntropyLoss()
    batch_size = adv_data.size(0)
    # get classifier output
    adv_label = model(adv_data)
    pure_output = model(output)
    all_output = model(output + adv_output)
    # get loss
    target = torch.zeros((adv_data.size(0),)).long()
    adv_target = torch.ones((adv_data.size(0),)).long()
    if torch.cuda.is_available():
        target = target.cuda()
        adv_target = adv_target.cuda()
    adv_class_loss = criterion(all_output, adv_target) + criterion(adv_label, adv_target)
    recon_class_loss = criterion(pure_output, target)

    return recon_class_loss + adv_class_loss