import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import numpy as np
import copy

use_cuda=True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# -------------------------- WHITE-BOX ATTACK --------------------------

# FGSM attack code (https://arxiv.org/pdf/1412.6572.pdf)
def fgsm_attack(model, data, target, epsilon=0.3):
    # Send the data and label to the device
    data, target = data.to(device), target.to(device)
    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True
    # Forward pass the data through the model
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # Get the index of the max log-probability
    # If the initial prediction is wrong, dont bother attacking, just move on
    if torch.all(torch.eq(init_pred, target)):
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
def ifgsm_attck(model, data, target, epsilon=16):
    # Calculate the iterative number
    iter_num = int(min(epsilon + 4, epsilon * 1.25))
    data, target = data.to(device), target.to(device) # Move tensors to device
    data = Variable(data, requires_grad = True)
    for i in range(iter_num):
        # Check if the prediction is correct
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
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
        # Adding clipping to maintain [0,1] range
        data.data = torch.clamp(data - sign_data_grad, 0, 1)

    return init_pred, data

# Iterative least likely attack code (https://arxiv.org/pdf/1607.02533.pdf)
def iterll_attack(model, data, target, epsilon=16):
    # Calculate the iterative number
    iter_num = int(min(epsilon + 4, epsilon * 1.25))
    # Process data
    data = data.to(device)
    data = Variable(data, requires_grad = True)
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1]
    # Get the label with smallest prediction probability
    target = output.min(1, keepdim=True)[1]
    target = target.detach_().to(device).reshape(-1)
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
        # Adding clipping to maintain [0,1] range
        data.data = torch.clamp(data - sign_data_grad, 0, 1)

    return init_pred, data

# r-FGSM attack code (https://arxiv.org/pdf/1705.07204.pdf)
def rfgsm_attck(model, data, target, epsilon=16/255, alpha=8/255):
    # Add random noise
    data = data + alpha*torch.randn_like(data).sign()
    # Check if the prediction is correct
    data, target = data.to(device), target.to(device)
    data = Variable(data, requires_grad = True)
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1]
    if torch.all(torch.eq(init_pred, target)):
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
    momentum = torch.zeros_like(data, requires_grad=True)
    alpha = epsilon / iter_num
    # Check if the prediction is correct
    data, target = data.to(device), target.to(device)
    data = Variable(data, requires_grad = True)
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1]
    if torch.all(torch.eq(init_pred, target)):
        return 0, 0
    # Perform iterative loop
    for i in range(iter_num):
        output = model(data)
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        # Calculate gradient of data
        data_grad = data.grad.data
        data_grad = data_grad / torch.max(torch.ones_like(data_grad) * 1e-10, torch.mean(torch.abs(data_grad)))
        # Update momentum
        momentum = decay_rate * momentum + data_grad
        # Add adversarial features
        data.data = torch.clamp(data + alpha * momentum.sign(), 0, 1)
    
    return init_pred, data

# PGD attack code (https://arxiv.org/pdf/1706.06083.pdf)
def pgd_attack(model, data, target, epsilon=0.3, alpha=0.01, iter_num=40):
    # Copy original image
    original_image = copy.deepcopy(data)
    for i in range(iter_num):
        # Check if the prediction is correct
        data, target = data.to(device), target.to(device)
        data = Variable(data, requires_grad = True)
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if torch.all(torch.eq(init_pred, target)):
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
        # Crop delta_data based on epsilon
        delta_data = torch.clamp(data + alpha * sign_data_grad - original_image, -epsilon, epsilon)
        # Adding clipping to maintain [0,1] range
        data.data = torch.clamp(original_image + delta_data, 0, 1)

    return init_pred, data

# DeepFool attack code (https://arxiv.org/pdf/1511.04599.pdf)
def deepfool_attack(model, data, target, num_classes=10, overshoot=0.02, iter_num=50):
    # Check if the prediction is correct
    batch_size = data.size(0)
    data = data.to(device)
    data = Variable(data, requires_grad = True)
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1]
    if torch.all(torch.eq(init_pred, target)):
        return 0, 0
    else:
        current = init_pred # Set the current class of data
    # Calculate the loss
    i = 0 # Track iterations
    input_shape = data.shape # Get the input shape
    w = torch.zeros(input_shape) # Set weight
    r_out = torch.zeros(input_shape) # Set return value
    I = torch.argsort(output, dim=1, descending=True) # Get the index for the classes in a descending order
     
    # Start loop
    while (torch.all(torch.eq(init_pred, target))) or (i <= iter_num):
        pert = torch.tensor([np.inf for b in range(batch_size)]).to(device) # Set the initial perturbation to infinite
        # Calculate gradient for correct class
        output[list(range(batch_size)), list(I[:, 0])].sum().backward(retain_graph=True)
        original_grad = copy.deepcopy(data.grad.data)

        # Loop for num_classes
        for k in range(1, num_classes):
            # Calculate gradient
            zero_gradients(data)
            output[list(range(batch_size)), list(I[:, k])].sum().backward(retain_graph=True)
            current_grad = copy.deepcopy(data.grad)
            # Get w_k and f_k
            w_k = current_grad - original_grad
            f_k = (output[list(range(batch_size)), list(I[:, k])] - output[list(range(batch_size)), list(I[:, 0])]).data
            # Calculate pertubation for class k
            pert_k = abs(f_k) / (w_k.flatten().norm() + 0.001)
            ci = torch.where(pert_k < pert)
            pert[ci] = pert_k[ci]
            w[ci] = w_k[ci]
        
        # Return value for each time step
        r_i = pert[:, None, None, None].float() * w.float() / (w.norm() + 0.001)
        r_out = r_out + r_i

        # Apply new data to see if attack successful
        data.data = torch.clamp(data + r_out, 0, 1)
        output = model(data)
        current = output.max(1, keepdim=True)[1]

        i += 1 # Add iterative number
    
    data = torch.clamp(data + (1 + overshoot) * r_out, 0, 1) # Prepare output
    
    return init_pred, data

# CW attack code (https://arxiv.org/abs/1608.04644)
# This code is modified from the one here: http://places.csail.mit.edu/deepscene/small-projects/network_adversarial/pytorch-nips2017-attack-example/attacks/attack_carlini_wagner_l2.py
def cw_attack(model, data, target, targeted=False, num_classes=10, max_steps=100, lr=0.001, \
    confidence=10, binary_search_steps=5, abort_early = True, clip_min=0, clip_max=1, clamp_fn='tanh', init_rand=False):
    def _compare(output, target):
        if not isinstance(output, (float, int, np.int64)) and len(output.shape) > 0:
            output = np.copy(output)
            if targeted:
                output[target] -= confidence
            else:
                output[target] += confidence
            output = np.argmax(output)
        if targeted:
            return output == target
        else:
            return output != target

    def _loss(output, target, dist, scale_const):
        # compute the probability of the label class versus the maximum other
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if targeted:
            # if targeted, optimize for making the other class most likely
            loss1 = torch.clamp(other - real + confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            loss1 = torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
        loss1 = torch.sum(scale_const * loss1)

        loss2 = dist.sum()

        loss = loss1 + loss2
        return loss

    def _optimize(optimizer, model, input_var, modifier_var, target_var, scale_const_var, input_orig=None):
        # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max
        if clamp_fn == 'tanh':
            input_adv = tanh_rescale(modifier_var + input_var, clip_min, clip_max)
        else:
            input_adv = torch.clamp(modifier_var + input_var, clip_min, clip_max)

        output = model(input_adv)

        # distance to the original input data
        if input_orig is None:
            dist = l2_dist(input_adv, input_var, keepdim=False)
        else:
            dist = l2_dist(input_adv, input_orig, keepdim=False)

        loss = _loss(output, target_var, dist, scale_const_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_np = loss.item()
        dist_np = dist.data
        output_np = output.data
        input_adv_np = input_adv.data.permute(0, 2, 3, 1)  # back to BHWC for numpy consumption
        return loss_np, dist_np, output_np, input_adv_np
    
    def torch_arctanh(x, eps=1e-6):
        x = x * (1. - eps)
        return (torch.log((1 + x) / (1 - x))) * 0.5
    
    def tanh_rescale(x, x_min=-1., x_max=1.):
        return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min
    
    def l2_dist(x, y, keepdim=True):
        d = (x - y)**2
        return reduce_sum(d, keepdim=keepdim)
    
    def reduce_sum(x, keepdim=True):
        for a in reversed(range(1, x.dim())):
            x = x.sum(a, keepdim=keepdim)
        return x

    repeat = binary_search_steps >= 10
    data = data.to(device)
    data = Variable(data, requires_grad = True)
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1]
    if torch.all(torch.eq(init_pred, target)):
        return 0, 0
    target = target.to(device)
    batch_size = data.size(0)

    # set the lower and upper bounds accordingly
    lower_bound = np.zeros(batch_size)
    scale_const = np.ones(batch_size) * 0.1
    upper_bound = np.ones(batch_size) * 1e10

    # python/numpy placeholders for the overall best l2, label score, and adversarial image
    o_best_l2 = [1e10] * batch_size
    o_best_score = [-1] * batch_size
    o_best_attack = data.permute(0, 2, 3, 1)

    # setup input (image) variable, clamp/scale as necessary
    if clamp_fn == 'tanh':
        # convert to tanh-space, input already int -1 to 1 range, does it make sense to do
        # this as per the reference implementation or can we skip the arctanh?
        input_var = Variable(torch_arctanh(data), requires_grad=False)
        input_orig = tanh_rescale(input_var, clip_min, clip_max)
    else:
        input_var = Variable(data, requires_grad=False)
        input_orig = None

    # setup the target variable, we need it to be in one-hot form for the loss function
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.to(device)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)

    # setup the modifier variable, this is the variable we are optimizing over
    modifier = torch.zeros(input_var.size()).float()
    if init_rand:
        # Experiment with a non-zero starting point...
        modifier = torch.normal(means=modifier, std=0.001)
    modifier = modifier.to(device)
    modifier_var = Variable(modifier, requires_grad=True)

    optimizer = optim.Adam([modifier_var], lr=0.0005)

    for search_step in range(binary_search_steps):
        best_l2 = [1e10] * batch_size
        best_score = [-1] * batch_size

        # The last iteration (if we run many steps) repeat the search once.
        if repeat and search_step == binary_search_steps - 1:
            scale_const = upper_bound

        scale_const_tensor = torch.from_numpy(scale_const).float()
        scale_const_tensor = scale_const_tensor.to(device)
        scale_const_var = Variable(scale_const_tensor, requires_grad=False)

        prev_loss = 1e6
        for step in range(max_steps):
            # perform the attack
            loss, dist, output, adv_img = _optimize(
                optimizer, model, input_var, modifier_var,
                target_var, scale_const_var, input_orig)

            if abort_early and step % (max_steps // 10) == 0:
                if loss > prev_loss * .9999:
                    break
                prev_loss = loss

            # update best result found
            for i in range(batch_size):
                target_label = target[i]
                output_logits = output[i]
                output_label = np.argmax(output_logits)
                di = dist[i]
                if di < best_l2[i] and _compare(output_logits, target_label):
                    best_l2[i] = di
                    best_score[i] = output_label
                if di < o_best_l2[i] and _compare(output_logits, target_label):
                    o_best_l2[i] = di
                    o_best_score[i] = output_label
                    o_best_attack[i] = adv_img[i]
            # end inner step loop

        # adjust the constants
        batch_failure = 0
        batch_success = 0
        for i in range(batch_size):
            if _compare(best_score[i], target[i]) and best_score[i] != -1:
                # successful, do binary search and divide const by two
                upper_bound[i] = min(upper_bound[i], scale_const[i])
                if upper_bound[i] < 1e9:
                    scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
            else:
                # failure, multiply by 10 if no solution found
                # or do binary search with the known upper bound
                lower_bound[i] = max(lower_bound[i], scale_const[i])
                if upper_bound[i] < 1e9:
                    scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    scale_const[i] *= 10
            if _compare(o_best_score[i], target[i]) and o_best_score[i] != -1:
                batch_success += 1
            else:
                batch_failure += 1

    return init_pred, o_best_attack.permute(0, 3, 2, 1)

# -------------------------- BLACK-BOX ATTACK --------------------------

# SimBA attack code (https://arxiv.org/pdf/1905.07121.pdf)
# This is modified from the original code here: https://github.com/cg563/simple-blackbox-attack/blob/master/simba_single.py
def simba_attack(model, x, y, num_iters=10000, epsilon=0.2):
    # Get the classification probability of certain class
    def get_probs(model, x, y):
        output = model(x)
        probs = F.softmax(output, dim=1)[:, y]
        return torch.diag(probs.data)
    # Get inital prediction output
    x = x.to(device)
    output = model(x)
    init_pred = output.max(1, keepdim=True)[1]
    if torch.all(torch.eq(init_pred, y)):
        return 0, 0
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
        ci = torch.where(left_prob < last_prob)
        if len(ci[0].shape) > 0:
            x = (x - diff.view(x.size())).clamp(0, 1)
            last_prob[ci] = left_prob[ci]
        
        right_prob = get_probs(model, (x + diff.view(x.size())).clamp(0, 1), y)
        cj = torch.where(right_prob < last_prob)
        if len(cj[0].shape) > 0:
            x = (x + diff.view(x.size())).clamp(0, 1)
            last_prob[cj] = right_prob[cj]

    return init_pred, x