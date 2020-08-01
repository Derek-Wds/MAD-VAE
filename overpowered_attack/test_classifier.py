from torch.utils.tensorboard import SummaryWriter
from models import Classifier
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", "--epochs", type=int, default=50)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-save", "--save_path", type=str, default="classifier.pt")
    parser.add_argument("-batch", "--batch_size", type=int, default=128)
    parser.add_argument("-tb", "--tensorboard_path", type=str, default=None)
    parser.add_argument('--image_channels',
                        type=int,
                        default=1,
                        help='Image channels')
    parser.add_argument('--image_size',
                        type=int,
                        default=28,
                        help='Image size (default to be squared images)')
    args = parser.parse_args()

    # load data
    transform = transforms.Compose(
        [transforms.CenterCrop(args.image_size),
         transforms.ToTensor()])
    testset = datasets.MNIST('./data',
                             train=False,
                             download=True,
                             transform=transform)
    dataloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=1)

    model = Classifier(args)
    model_pt = torch.load(args.save_path)
    model.load_state_dict(model_pt)
    model.eval()

    correct = 0
    total = 0
    for data, label in dataloader:
        if torch.cuda.is_available():
            model = model.cuda()
            data = data.cuda()
            label = label.cuda()

        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))
