import sys, os, argparse
sys.path.insert(0, os.path.abspath('..'))
import warnings
warnings.filterwarnings("ignore")
import foolbox
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST, FashionMNIST
from test.attacks import *
from advertorch.attacks import *
from MAD_VAE import *

# argument parser
def parse_args():
    desc = "MAD-VAE for adversarial defense"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Training epoch numbers')
    parser.add_argument('--h_dim', type=int, default=4096, help='Hidden dimensions')
    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimensions for images')
    parser.add_argument('--image_channels', type=int, default=1, help='Image channels')
    parser.add_argument('--image_size', type=int, default=28, help='Image size (default to be squared images)')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of image classes')
    parser.add_argument('--log_dir', type=str, default='v_logs', help='Logs directory')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the Adam optimizer')
    parser.add_argument('--data_root', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='pretrained_model', help='Pretrained model directory')
    parser.add_argument('--use_gpu', type=bool, default=True, help='If use GPU for training')
    parser.add_argument('--gpu_num', type=int, default=1, choices=range(0,5), help='GPU numbers available for parallel training')
    parser.add_argument('--experiment', type=int, default=0, choices=range(0,4))
    return parser.parse_args()


'''
Classifier for generating the adversarial examples, (similar to the one in MagNet: https://arxiv.org/pdf/1705.09064.pdf)
MNIST accuracy: 0.993100
FashionMNIST accuracy: 0.926600
'''
class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.name = 'Classifier'
        self.image_size = args.image_size
        self.image_channels = args.image_channels
        self.conv1 = nn.Conv2d(self.image_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(3136, 200)
        self.fc2 = nn.Linear(200, 10)
    
    def forward(self, x):
        self.batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(self.batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


# function for construct adversarial images
def add_adv(model, image, label, adv):
    # fast gradient sign method
    if adv == 'fgsm':
        # _, adv_image = fgsm_attack(model, image, label)
        fgsm = GradientSignAttack(model)
        adv_image = fgsm(image, label)
    # iterative fast gradient sign method
    elif adv == 'i-fgsm':
        # _, adv_image = ifgsm_attck(model, image, label)
        ifgsm = LinfBasicIterativeAttack(model)
        adv_image = ifgsm(image, label)
    # iterative least likely sign method
    elif adv == 'iterll':
        _, adv_image = iterll_attack(model, image, label)
    # random fast gradient sign method
    elif adv == 'r-fgsm':
        #_, adv_image = rfgsm_attck(model, image, label)
        alpha = 0.05
        data = torch.clamp(image + alpha * torch.empty(image.shape).normal_(mean=0,std=1).cuda(), min=0, max=1)
        fgsm = GradientSignAttack(model, eps=0.3-alpha)
        adv_image = fgsm(data, label)
    # momentum iterative fast gradient sign method
    elif adv == 'mi-fgsm':
        mifgsm = MomentumIterativeAttack(model)
        adv_image = mifgsm(image, label)
    # projected gradient sign method
    elif adv == 'pgd':
        #_, adv_image = pgd_attack(model, image, label)
        pgd = L2PGDAttack(model)
        adv_image = pgd(image, label)
    # deepfool attack method
    elif adv == 'deepfool':
        _, adv_image = deepfool_attack(model, image, label)
    # Carlini-Wagner attack
    elif adv == 'cw':
        # _, adv_image = cw_attack(model, image, label)
        cw = CarliniWagnerL2Attack(model, 10, confidence=10, max_iterations=1500)
        adv_image = cw(image, label)
    # simba attack
    elif adv == 'simba':
        _, adv_image = simba_attack(model, image, label)
    else:
        _, adv_image = image
        print('Did not perform attack on the images!')
    # if attack fails, return original
    if adv_image is None:
        adv_image = image
    if torch.cuda.is_available():
        image = image.cuda()
        adv_image = adv_image.cuda()

    return image, adv_image

if __name__ == "__main__":
    # get arguments
    args = parse_args()

    # init and load classifier
    classifier = Classifier(args)
    classifier.load_state_dict(torch.load('../pretrained_model/classifier_mnist.pt'))
    classifier.eval()
    classifier = classifier.cuda()

    defense = MADVAE(args)

    # for running parallel experiments
    experiments = [
        'vanilla',
        'combined',
        'proxi_dist',
        'classification'
    ]

    defense_pt = torch.load(f'../pretrained_model/{experiments[args.experiment]}/params.pt')
    # for key in list(defense_pt.keys()):
        # defense_pt[key.replace('module.', '')] = defense_pt.pop(key)
    defense.load_state_dict(defense_pt, strict=False)
    defense.eval()
    defense = defense.cuda()

    # init dataset for TESTING
    transform  = transforms.Compose([transforms.CenterCrop(args.image_size), transforms.ToTensor()])
    dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=1)

    # adversarial methods
    adv_accuracy = {'fgsm': 0, 'r-fgsm': 0, 'cw': 0}

    # test for accuracy
    for adv in adv_accuracy:
        true = 0
        total = len(dataset)
        for image, label in dataloader:
            image = image.cuda()
            label = label.cuda()
            
            output, adv_out = add_adv(classifier, image, label, adv)
            output = classifier(output)
            def_out, _, _, _ = defense(adv_out)
            adv_out = classifier(def_out)

            true_class = torch.argmax(output, 1)
            adversarial_class = torch.argmax(adv_out, 1)

            print(f'attack method {adv}')
            print(f'actual class {true_class}')
            print(f'adversarial class {adversarial_class}')

            true += torch.sum(torch.eq(true_class, adversarial_class))

        adv_accuracy[adv] = int(true) / total

    
    with open(f'./accuracy_{experiments[args.experiment]}.txt', 'w') as f:
        json.dump(adv_accuracy, f)


