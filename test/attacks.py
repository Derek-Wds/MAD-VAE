import torch
import torch.nn.functional as F
import numpy as np

use_cuda=True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# FGSM attack code (https://arxiv.org/pdf/1412.6572.pdf)
def fgsm_attack(model, data, target, epsilon):
    # Send the data and label to the device
    data, target = data.to(device), target.to(device)
    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True
    # Forward pass the data through the model
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    # If the initial prediction is wrong, dont bother attacking, just move on
    if init_pred.item() != target.item():
        return 0, 0
    # Calculate the loss
    loss = F.nll_loss(output, target)
    # Zero all existing gradients
    model.zero_grad()
    # Calculate gradients of model in backward pass
    loss.backward()
    # Collect datagrad
    data_grad = data.grad.data
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = data + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return init_pred, perturbed_image

# i-FGSM attack code (https://arxiv.org/pdf/1607.02533.pdf)
def ifgsm_attck(model, data, target, epsilon):
    # Calculate the iterative number
    iter_num = int(min(epsilon + 4, epsilon * 1.25))
    for i in range(iter_num):
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        # Calculate the loss
        loss = F.nll_loss(output, target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        data = data - sign_data_grad
        # Adding clipping to maintain [0,1] range
        data = torch.clamp(data, 0, 1)

    return init_pred, data

# Iterative least likely attack code (https://arxiv.org/pdf/1607.02533.pdf)
def iterll_attack(model, data, target, epsilon):
    # Calculate the iterative number
    iter_num = int(min(epsilon + 4, epsilon * 1.25))
    # Send the data to the device
    data = data.to(device)
    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True
    # Forward pass the data through the model
    output = model(data)
    # Get init output
    init_pred = output.max(1, keepdim=True)[1]
    # Get the label with smallest prediction probability
    target = output.min(1, keepdim=True)[1]
    target = target.detach_().to(device)
    for i in range(iter_num):
        data.requires_grad = True
        output = model(data)
        # Calculate the loss
        loss = F.nll_loss(output, target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        data = data - sign_data_grad
        # Adding clipping to maintain [0,1] range
        data = torch.clamp(data, 0, 1)

    return init_pred, data

# r-FGSM attack code
def rfgsm_attck(model, data, target, epsilon=16/255, alpha=8/255):
    # Add random noise
    data = data + alpha*torch.randn_like(data).sign()
    # Send the data and label to the device
    data, target = data.to(device), target.to(device)
    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True
    # Forward pass the data through the model
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    # If the initial prediction is wrong, dont bother attacking, just move on
    if init_pred.item() != target.item():
        return 0, 0
    # Calculate the loss
    loss = F.nll_loss(output, target)
    # Zero all existing gradients
    model.zero_grad()
    # Calculate gradients of model in backward pass
    loss.backward()
    # Collect datagrad
    data_grad = data.grad.data
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = data + (epsilon-alpha)*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(data, 0, 1)

    return init_pred, perturbed_image
