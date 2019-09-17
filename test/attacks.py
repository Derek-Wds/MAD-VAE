import torch
import torch.nn.functional as F
import numpy as np
import copy

use_cuda=True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# -------------------------- WHITE-BOX ATTACK --------------------------

# FGSM attack code (https://arxiv.org/pdf/1412.6572.pdf)
def fgsm_attack(model, data, target, epsilon):
    # Send the data and label to the device
    data, target = data.to(device), target.to(device)
    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True
    # Forward pass the data through the model
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # Get the index of the max log-probability
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
        # Check if the prediction is correct
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
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
    # Process data
    data = data.to(device)
    data.requires_grad = True
    output = model(data)
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

# r-FGSM attack code (https://arxiv.org/pdf/1705.07204.pdf)
def rfgsm_attck(model, data, target, epsilon=16/255, alpha=8/255):
    # Add random noise
    data = data + alpha*torch.randn_like(data).sign()
    # Check if the prediction is correct
    data, target = data.to(device), target.to(device)
    data.requires_grad = True
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1]
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

# MI-FGSM attack code (https://arxiv.org/pdf/1710.06081.pdf)
def mifgsm_attack(model, data, target, epsilon=16, decay_rate=1, iter_num=10):
    # Initialize variables
    momentum = torch.zeros_like(data)
    alpha = epsilon / iter_num
    # Check if the prediction is correct
    data, target = data.to(device), target.to(device)
    data.requires_grad = True
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1]
    if init_pred.item() != target.item():
        return 0, 0
    # Perform iterative loop
    for i in range(iter_num):
        data.requires_grad = True
        output = model(data)
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        # Calculate gradient of data
        data_grad = data.grad.data
        data_grad = data_grad / (torch.mean(torch.abs(data_grad)) + 1e-10)
        # Update momentum
        momentum = decay_rate * momentum + data_grad
        # Add adversarial features
        data = torch.clamp(data + alpha * momentum.sign(), 0, 1)
    
    return init_pred, data

# PGD attack code (https://arxiv.org/pdf/1706.06083.pdf)
def pgd_attack(model, data, target, epsilon=0.3, alpha=0.01, iter_num=40):
    # Copy original image
    original_image = copy.deepcopy(data)
    for i in range(iter_num):
        # Check if the prediction is correct
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
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
        data = data + alpha * sign_data_grad
        # Crop delta_data based on epsilon
        delta_data = torch.clamp(data - original_image, -epsilon, epsilon)
        # Adding clipping to maintain [0,1] range
        data = torch.clamp(original_image + delta_data, 0, 1)

    return init_pred, data

# DeepFool attack code (https://arxiv.org/pdf/1511.04599.pdf)
def deepfool_attack(model, data, target, num_classes=10, overshoot=0.02, iter_num=50):
    # Check if the prediction is correct
    data = data.to(device)
    data.requires_grad = True
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1]
    if init_pred.item() != target.item():
        return 0, 0
    else:
        current = init_pred # Set the current class of data
    # Calculate the loss
    i = 0 # Track iterations
    input_shape = data.numpy().shape # Get the input shape
    w = torch.zeros(input_shape) # Set weight
    r_out = torch.zeros(input_shape) # Set return value
    I = (np.array(output)).flatten().argsort()[::-1] # Get the index for the classes in a descending order

    # Start loop
    while (current.item() == target.item()) or (i <= iter_num):
        pert = np.inf # Set the initial perturbation to infinite
        # Calculate gradient for correct class
        original_loss = F.nll_loss(output, target)
        model.zero_grad()
        original_loss.backward()
        original_grad = data.grad.data.numpy()
        
        # Loop for num_classes
        for k in range(1, num_classes):
            # Calculate gradient
            loss = F.nll_loss(output, output[:, I[k]])
            model.zero_grad()
            loss.backward()
            current_grad = data.grad.data.numpy()
            # Get w_k and f_k
            w_k = current_grad - original_grad
            f_k = (output[:, I[k]] - output[:, I[0]]).data.numpy()
            # Calculate pertubation for class k
            pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k.flatten())
            if pert_k < pert:
                pert = pert_k
                w = w_k
        
        # Return value for each time step
        r_i = pert * w / np.linalg.norm(w)
        r_out = r_out + r_i

        # Apply new data to see if attack successful
        data = torch.clamp(data + r_out, 0, 1)
        data.requires_grad = True
        output = model(data)
        current = output.max(1, keepdim=True)[1]

        i += 1 # Add iterative number
    
    data = torch.clamp(data + (1 + overshoot) * r_out, 0, 1) # Prepare output
    
    return init_pred, data


# -------------------------- BLACK-BOX ATTACK --------------------------

# SimBA attack code (https://arxiv.org/pdf/1905.07121.pdf)
# This is modified from the original code here: https://github.com/cg563/simple-blackbox-attack/blob/master/simba_single.py
def simba_attack(model, x, y, num_iters=10000, epsilon=0.2):
    # Get the classification probability of certain class
    def get_probs(model, x, y):
        output = model(x)
        probs = torch.nn.Softmax()(output)[:, y]
        return torch.diag(probs.data)
    # Get the total dimension of input x
    n_dims = x.view(1, -1).size(1)
    # Generate random permutation from 0 to n_dims
    perm = torch.randperm(n_dims)
    # The probability to track current lowest probability
    last_prob = get_probs(model, x, y)
    # Start loop
    for i in range(num_iters):
        # Initialize noise
        diff = torch.zeros(n_dims)
        # At specific pixel, make one step
        diff[perm[i % n_dims]] = epsilon
        # Try to determine which direction to step
        left_prob = get_probs(model, (x - diff.view(x.size())).clamp(0, 1), y)
        if left_prob < last_prob:
            x = (x - diff.view(x.size())).clamp(0, 1)
            last_prob = left_prob
        else:
            right_prob = get_probs(model, (x + diff.view(x.size())).clamp(0, 1), y)
            if right_prob < last_prob:
                x = (x + diff.view(x.size())).clamp(0, 1)
                last_prob = right_prob
    return x