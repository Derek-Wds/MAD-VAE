import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.distributions import kl_divergence, Normal
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from DA_VAE import *
from utils import *

# argument parser
def parse_args():
    desc = "DA-VAE for adversarial defense and attack"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Training epoch numbers')
    parser.add_argument('--h_dim', type=int, default=512, help='Hidden dimensions')
    parser.add_argument('--z1_dim', type=int, default=256, help='Latent dimensions for images')
    parser.add_argument('--z2_dim', type=int, default=128, help='Latent dimensions for adversarial noise')
    parser.add_argument('--image_channels', type=int, default=3, help='Image channels')
    parser.add_argument('--image_size', type=int, default=256, help='Image size (default to be squared images)')
    parser.add_argument('--log_dir', type=str, default='logs', help='Logs directory')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for the Adam optimizer')
    parser.add_argument('--model_dir', type=str, default='pretrained_model', help='Pretrained model directory')
    parser.add_argument('--gpu_num', type=int, default=1, choices=range(0,5), help='GPU numbers available for parallel training')

    return parser.parse_args()

# main function
def main():
    args = parse_args()
    # make directories for pretrained models
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    # prepare dataset
    transform  = transforms.Compose([transforms.CenterCrop(args.image_size), transforms.ToTensor()])
    trainSet = datasets.ImageNet(root, split='train', download=True, transform = transform)
    dataloader = data.DataLoader(trainSet, batch_size=args.batch_size, shuffle=True, num_workers=1)
    adv_list = ['fgsm', 'i-fgsm', 'mi-fgsm', 'pgd', 'deepfool', 'saliecy-map', 'cw-attack', 'single-pixel', 'simba']

    # construct model
    model = DAVAE(args)
    if torch.cuda.is_available():
        model = model.cuda()
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.train()

    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = MinExponentialLR(optimizer, gamma=0.95, minimum=1e-5)

    # summary writer for tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # tratinig steps
    step = 0
    for epoch in range(1, args.epochs+1):
        print('Epoch: {}'.format(epoch))
        for data, target in dataloader:
            step += 1
            losses = list()
            outputs = list()
            adv_outputs = list()
            for adv in adv_list:
                data, adv_data = add_adv(data, adv)
                optimizer.zero_grad()
                output, adv_output, ds1m, ds1s, ds2m, ds2s = model(data)
                distribution1, distribution2 = Normal(ds1m, ds1s), Normal(ds2m, ds2s)
                loss = loss_function(output, data, adv_output, adv_data, distribution1, distribution2, step, 0.5)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                losses.append(loss.item())
                outputs.append(output)
                adv_outputs.append(adv_output)
            # write to tensorboard
            writer.add_scalar('loss', np.sum(losses)/len(losses), step)
            if step % 50 == 0:
                writer.add_image('original data', data[0], step)
                for i in range(len(outputs)):
                    writer.add_image("reconstruct data {}".format(i), outputs[i][0], step)
                    writer.add_image("adversarial noise {}".format(i), adv_outputs[i][0], step)
            # print out loss
            if step % 10 == 0:
                print("batch {}'s loss: {:.5f}".format(step, np.sum(losses)/len(losses)))
            # step scheduler
            if step % 50 == 0:
                scheduler.step()
            # save model parameters
            if epoch % 20 == 0:
                torch.save(model.cpu().state_dict(), 'params_{}.pt'.format(epoch))
        
    torch.save(model.cpu().state_dict(), 'params.pt')

if __name__ == '__main__':
    main()