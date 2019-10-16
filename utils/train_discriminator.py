import sys, os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST, FashionMNIST
from DAD_VAE import Discriminator
from scheduler import MinExponentialLR
from preprocess import *
from main import parse_args

# init dataset
transform  = transforms.Compose([transforms.CenterCrop(28), transforms.ToTensor()])
train_data = MNIST('../data', train=True, download=True, transform=transform)
test_data = MNIST('../data', train=False, download=True, transform=transform)

# init dataloader
train_data_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=8)
test_data_loader = DataLoader(test_data, batch_size=256, num_workers=8)

# init models and optimizers
args = parse_args()
model = Discriminator(args)
classifier = Classifier(args)
classifier.load_state_dict(torch.load('../pretrained_model/classifier_mnist.pt'))
classifier.eval()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = MinExponentialLR(optimizer, gamma=0.95, minimum=1e-5)

# train function
def train(epoch, adv):
    model.train()
    if torch.cuda.is_available():
        model.cuda()
    loss_list = []
    for i, (images, labels) in enumerate(train_data_loader):
        real = Variable(torch.zeros((images.size(0),)).long())
        fake = Variable(torch.ones((images.size(0),)).long())
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
            real = real.cuda()
            fake = fake.cuda()
        optimizer.zero_grad()
        images, adv_images = add_adv(classifier, images, labels, adv)
        if 'torch' in str(type(images)):
            output = model(images)
            adv_output = model(adv_images)
            loss = criterion(output, real) + criterion(adv_output, fake)
            loss_list.append(loss.detach().cpu().item())
            if i % 50 == 0:
                print(model.name)
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))
            loss.backward()
            optimizer.step()
    scheduler.step()

# test function
def test(adv):
    model.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(test_data_loader):
        images, adv_images = add_adv(classifier, images, labels, adv)
        if 'torch' in str(type(images)):
            real = Variable(torch.zeros((images.size(0),)).long())
            fake = Variable(torch.ones((images.size(0),)).long())
            output = model.cpu()(images.cpu())
            adv_output = model.cpu()(adv_images.cpu())
            avg_loss += criterion(output, real).sum() + criterion(adv_output, fake).sum()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
    avg_loss /= len(test_data)
    print('%s test Avg. Loss: %f, Accuracy: %f' % (model.name, avg_loss.detach().cpu().item(), float(total_correct) / len(test_data)))

# main function
def main():
    adv_list = ['fgsm', 'iterll', 'mi-fgsm', 'pgd', 'cw']
    for e in range(1, 30):
        for adv in adv_list:
            train(e, adv)
            test(adv)
    # save pretrained model
    torch.save(model.state_dict(), "pretrained/%s_mnist_params.pt" % model.name)

if __name__ == '__main__':
    main()