from torch.utils.tensorboard import SummaryWriter
from models import Classifier
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np


def train(args, dataloader, classifier, optimizer, crit, step):
    total_loss = []
    for i, (data, label) in enumerate(dataloader):
        if torch.cuda.is_available():
            classifier = classifier.cuda()
            data = data.cuda()
            label = label.cuda()
        
        optimizer.zero_grad()
        output = classifier(data)
        loss = crit(output, label)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    return total_loss, step + len(total_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", "--epochs", type=int, default=50)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-save", "--save_path", type=str, default="classifier.pt")
    parser.add_argument("-batch", "--batch_size", type=int, default=128)
    parser.add_argument("-tb", "--tensorboard_path", type=str, default=None)
    parser.add_argument('--image_channels', type=int, default=1, help='Image channels')
    parser.add_argument('--image_size', type=int, default=28, help='Image size (default to be squared images)')
    args = parser.parse_args()

    # load data
    transform  = transforms.Compose([transforms.CenterCrop(args.image_size), transforms.ToTensor()])
    trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    model = Classifier(args)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.learning_rate)
    writer = SummaryWriter(args.tensorboard_path)
    print(f"Training for {args.epochs} epochs")
    step = 0
    for epoch in range(args.epochs):
        loss, step = train(args, dataloader, model, opt, crit, step)
        avg_loss = np.average(loss)
        print(f"Epoch {epoch} loss = {avg_loss}")
        writer.add_scalar('class_loss', avg_loss, step)
        
    
    torch.save(model.state_dict(), args.save_path)