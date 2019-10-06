import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST, FashionMNIST
from test_models import *

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

# init dataset
transform  = transforms.Compose([transforms.CenterCrop(28), transforms.ToTensor()])
train_data = FashionMNIST('../data', train=True, download=True, transform=transform)
test_data = FashionMNIST('../data', train=False, download=True, transform=transform)

# init dataloader
train_data_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=8)
test_data_loader = DataLoader(test_data, batch_size=1024, num_workers=8)

# init models and optimizers
models = [model_a(), model_b(), model_c(), model_d(), model_e()]
criterion = nn.CrossEntropyLoss()
optimizers = [optim.Adam(model.parameters(), lr=1e-3) for model in models]
schedulers = [MinExponentialLR(optimizer, gamma=0.95, minimum=1e-5) for optimizer in optimizers]

# train function
def train(epoch):
    for model in models:
        model.train()
        if torch.cuda.is_available():
            model.cuda()
    loss_list = [[] for i in range(len(models))]
    for i in range(len(models)):
        for j, (images, labels) in enumerate(train_data_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            optimizers[i].zero_grad()
            output = models[i](images)
            loss = criterion(output, labels)
            loss_list[i].append(loss.detach().cpu().item())
            if j % 10 == 0:
                print(models[i].name)
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, j, loss.detach().cpu().item()))
            loss.backward()
            optimizers[i].step()
        schedulers[i].step()

# test function
def test():
    for model in models:
        model.eval()
    total_corrects = [0 for i in range(len(models))]
    avg_losses = [0.0 for i in range(len(models))]
    for i in range(len(models)):
        for j, (images, labels) in enumerate(test_data_loader):
            output = models[i].cpu()(images)
            avg_losses[i] += criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            total_corrects[i] += pred.eq(labels.view_as(pred)).sum()
        avg_losses[i] /= len(data_test)
        print('%s test Avg. Loss: %f, Accuracy: %f' % (models[i].name, avg_losses[i].detach().cpu().item(), float(total_corrects[i]) / len(data_test)))

# main function
def main():
    for e in range(1, 30):
        train(e)
        test()
    # save pretrained models
    for i in range(len(models)):
        torch.save(models[i].state_dict(), "pretrained/%s_fmnist_params.pt" % models[i].name)

if __name__ == '__main__':
    main()