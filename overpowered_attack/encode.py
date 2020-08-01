from torch.utils.tensorboard import SummaryWriter
from models import MADVAE
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np
import os

R = 20
L = 50000
BATCH_SIZE = 50


def encode(model, data, labels, num_steps=L, suffix=""):
    os.makedirs(f"encodings{suffix}", exist_ok=True)
    f_name = f"encodings{suffix}/saved_latents_{num_steps}"
    # _, zh = latent_space_opt(model, data, labels, BATCH_SIZE)
    _, mean, _, _ = model(data)
    torch.save(mean.detach(), f_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Training epoch numbers')
    parser.add_argument('--h_dim',
                        type=int,
                        default=4096,
                        help='Hidden dimensions')
    parser.add_argument('--z_dim',
                        type=int,
                        default=128,
                        help='Latent dimensions for images')
    parser.add_argument('--image_channels',
                        type=int,
                        default=1,
                        help='Image channels')
    parser.add_argument('--image_size',
                        type=int,
                        default=28,
                        help='Image size (default to be squared images)')
    parser.add_argument('--num_classes',
                        type=int,
                        default=10,
                        help='Number of image classes')
    parser.add_argument('--log_dir',
                        type=str,
                        default='c_logs',
                        help='Logs directory')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate for the Adam optimizer')
    parser.add_argument('--closs_weight',
                        type=float,
                        default=0.1,
                        help='Weight for classification loss functions')
    parser.add_argument('--data_root',
                        type=str,
                        default='data',
                        help='Data directory')
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_model',
                        help='Pretrained model directory')
    parser.add_argument('--use_gpu',
                        type=bool,
                        default=True,
                        help='If use GPU for training')
    parser.add_argument('--gpu_num',
                        type=int,
                        default=2,
                        choices=range(0, 5),
                        help='GPU numbers available for parallel training')
    parser.add_argument('--model', type=int, default=0, choices=range(5))
    args = parser.parse_args()

    methods = ['vanilla', 'classification', 'proxi_dist', 'combined', 'identity']
    METHOD = methods[args.model]

    model = MADVAE(args)
    model_pt = torch.load(
        f'../MAD-VAE/MAD-VAE/pretrained_model/{METHOD}/params.pt')
    model.load_state_dict(model_pt)
    model.eval()

    transform = transforms.Compose(
        [transforms.CenterCrop(args.image_size),
         transforms.ToTensor()])
    testset = datasets.MNIST('./data',
                             train=False,
                             download=True,
                             transform=transform)
    dataloader = torch.utils.data.DataLoader(testset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=1)

    if torch.cuda.is_available():
        model = model.cuda()

    for i, (data, label) in enumerate(dataloader):
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()

        encode(model, data, label, num_steps=i, suffix=f"_{METHOD}")
