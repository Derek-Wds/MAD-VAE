from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import foolbox
import numpy as np
import matplotlib.pyplot as plt
from attacks import *
from test_models import *
from torchvision import datasets, transforms
from foolbox.attacks import *
from foolbox.models import PyTorchModel
from utils.preprocess import Classifier


# function for construct adversarial one single image
def attack(model, image, label, adv):
    # fast gradient sign method, https://arxiv.org/abs/1412.6572
    if adv == 'fgsm':
        image, label = image.numpy(), label.numpy()
        model = PyTorchModel(model, bounds=(0, 1), num_classes=10)
        fgsm_attack = FGSM(model)
        adv_image = fgsm_attack(image, label)
    # interative fast gradient sign method, https://arxiv.org/pdf/1607.02533.pdf
    elif adv == 'i-fgsm':
        image, label = image.numpy(), label.numpy()
        model = PyTorchModel(model, bounds=(0, 1), num_classes=10)
        ifgsm_attack = LinfinityBasicIterativeAttack(model, distance=foolbox.distances.Linfinity)
        adv_image = ifgsm_attack(image, label, epsilon=0.3, iterations=10, random_start=True)
    # momentum iterative fast gradient sign method, https://arxiv.org/pdf/1710.06081.pdf
    elif adv == 'mi-fgsm':
        image, label = image.numpy(), label.numpy()
        model = PyTorchModel(model, bounds=(0, 1), num_classes=10)
        mifgsm_attack = MomentumIterativeAttack(model, distance=foolbox.distances.Linfinity)
        adv_image = mifgsm_attack(image, label, epsilon=0.3, iterations=10, random_start=True)
    # projected gradient descent, https://arxiv.org/pdf/1706.06083.pdf
    elif adv == 'pgd':
        image, label = image.numpy(), label.numpy()
        model = PyTorchModel(model, bounds=(0, 1), num_classes=10)
        pgd_attack = RandomPGD(model, distance=foolbox.distances.Linfinity)
        adv_image = pgd_attack(image, label, epsilon=0.3, iterations=40, random_start=True)
    # deepfool attack, https://arxiv.org/pdf/1511.04599.pdf
    elif adv == 'deepfool':
        image, label = image.numpy(), label.numpy()
        model = PyTorchModel(model, bounds=(0, 1), num_classes=10)
        deepfool_attack = DeepFoolL2Attack(model)
        adv_image = deepfool_attack(image, label, steps=50)
    # ADef attack, https://arxiv.org/abs/1804.07729
    elif adv == 'adef':
        image, label = image.numpy(), label.numpy()
        model = PyTorchModel(model, bounds=(0, 1), num_classes=10)
        adef_attack = ADefAttack(model)
        adv_image = adef_attack(image, label)
    # Carlini-Wagner attack, https://arxiv.org/abs/1608.04644
    elif adv == 'cw-attack':
        image, label = image.numpy(), label.numpy()
        model = PyTorchModel(model, bounds=(0, 1), num_classes=10)
        cw_attack = CarliniWagnerL2Attack(model)
        adv_image = cw_attack(image, label, max_iterations=200)
    # spatial attack, http://arxiv.org/abs/1712.02779
    elif adv == 'spatial':
        image, label = image.numpy(), label.numpy()
        model = PyTorchModel(model, bounds=(0, 1), num_classes=10)
        spatial_attack = SpatialAttack(model)
        adv_image = spatial_attack(image, label)
    else:
        adv_image = image
        print('Did not perform attack on the images!')
    # if attack fails, return original
    if adv_image is None:
        adv_image = image

    return image, adv_image


# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = model_a().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

def test( model, device, test_loader):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Call FGSM Attack
        init_pred, perturbed_data = simba_attack(model, data, target)

        # Re-classify the perturbed image
        if type(perturbed_data) != int and 'torch' in str(perturbed_data.dtype):
            output = model(perturbed_data)
        else:
            continue # No bother to attack since the prediciton is wrong

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Test Accuracy = {} / {} = {}".format(correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []

# Run test
acc, ex = test(model, device, test_loader)
accuracies.append(acc)
examples.append(ex)

# Plotting
plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # get arguments
    from main import parse_args
    args = parse_args()
    # init and load model
    classifier = Classifier(args)
    classifier.load_state_dict(torch.load('../pretrained_model/classifier_mnist.pt'))
    classifier.eval()
    # init dataset
    transform  = transforms.Compose([transforms.CenterCrop(args.image_size), transforms.ToTensor()])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    image, label = next(iter(dataloader))
    # adversarial methods
    adv_list = ['fgsm', 'i-fgsm', 'mi-fgsm', 'pgd', 'deepfool', 'adef', 'cw-attack', 'spatial']
    # test for accuracy
    for adv in adv_list:
        output, adv_out = attack(classifier, image[22], label[22], adv)
        output = classifier(torch.from_numpy(output).unsqueeze(0))
        adv_out = classifier(torch.from_numpy(adv_out).unsqueeze(0))
        print('attack method {}'.format(adv))
        print('actual class ', np.argmax(output.detach().numpy()))
        print('adversarial class ', np.argmax(adv_out.detach().numpy()))
        print('========================')