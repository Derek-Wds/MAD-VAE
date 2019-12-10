import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
Classifier for generating the adversarial examples, (similar to the classifier in MagNet: https://arxiv.org/pdf/1705.09064.pdf)
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