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
import matplotlib.pyplot as plt
import umap
import seaborn as sns
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


if __name__ == "__main__":
    # get arguments
    args = parse_args()
    sns.set(style='white', context='paper', rc={'figure.figsize':(14,10)})

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

    exp_name = ['A', 'B', 'C', 'D']

    defense_pt = torch.load(f'../pretrained_model/{experiments[args.experiment]}/params.pt')
    # for key in list(defense_pt.keys()):
        # defense_pt[key.replace('module.', '')] = defense_pt.pop(key)
    defense.load_state_dict(defense_pt, strict=False)
    defense.eval()
    defense = defense.cuda()

    # adversarial methods
    adv_accuracy = ['cw'] # needs CW attack

    # test for accuracy
    for adv in adv_accuracy:
        adv_examples = np.load('../data/' + adv + '_advs_mnist.npy')
        adv_examples = adv_examples.reshape(adv_examples.shape[0],28*28)
        sample = np.random.randint(adv_examples.shape[0], size=3000)
        adv_examples = adv_examples[sample,:]

        adv_images = torch.from_numpy(adv_examples.reshape(adv_examples.shape[0], 1, 28, 28))
        adv_images = adv_images.cuda()

        _, mean, _, _ = defense.forward(adv_images)
        mean = mean.detach().cpu().numpy()

        labels = np.load('../data/' + adv + '_ys_mnist.npy')
        labels = labels[sample]

        fit = umap.UMAP(n_components=2, random_state=42)
        u = fit.fit_transform(mean)

        plt.scatter(u[:,0], u[:,1], c=labels, cmap='Spectral',s=14)
        plt.gca().set_aspect('equal', 'datalim')
        clb = plt.colorbar(boundaries=np.arange(11)-0.5)
        clb.set_ticks(np.arange(10))
        clb.ax.tick_params(labelsize=18)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Latent clustering under {adv.upper()} with {exp_name[args.experiment]}', fontsize=24);
        plt.savefig(f'img/latent_{adv}_{experiments[args.experiment]}.png', dpi=300, pad_inches=0)
        plt.clf()


