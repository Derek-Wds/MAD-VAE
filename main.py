import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.distributions import kl_divergence, Normal
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from MAD_VAE import *
from utils.loss_function import *
from utils.preprocess import *
from utils.scheduler import *
from utils.dataset import *

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
    parser.add_argument('--log_dir', type=str, default='logs', help='Logs directory')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the Adam optimizer')
    parser.add_argument('--loss_weight', type=float, default=0.5, help='Weight for two different loss functions')
    parser.add_argument('--data_root', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='pretrained_model', help='Pretrained model directory')
    parser.add_argument('--gpu_num', type=int, default=2, choices=range(0,5), help='GPU numbers available for parallel training')

    return parser.parse_args()

# main function
def main():
    args = parse_args()
    # make directories for pretrained models
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    # prepare dataset
    data = np.load('xs_mnist.npy') # image data in npy file
    adv_data = np.load('advs_mnist.npy') # adversarial image data in npy file
    dataset = Dataset(data, adv_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # construct model and classifier
    model = MADVAE(args)
    classifier = Classifier(args)
    if args.gpu_num > 1:
        model = torch.nn.DataParallel(model, device_ids=range(args.gpu_num))
        classifier = torch.nn.DataParallel(classifier, device_ids=range(args.gpu_num))
        model = model.module
        classifier = classifier.module
    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.train()
    classifier.load_state_dict(torch.load('pretrained_model/classifier_mnist.pt'))
    classifier.eval()

    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MinExponentialLR(optimizer, gamma=0.99999, minimum=1e-5)

    # summary writer for tensorboard
    writer1 = SummaryWriter(args.log_dir+'/reon_loss')
    writer2 = SummaryWriter(args.log_dir+'/img_loss')
    writer3 = SummaryWriter(args.log_dir+'/kl_loss')
    
    # tratinig steps
    step = 0
    for epoch in range(1, args.epochs+1):
        print('Epoch: {}'.format(epoch))
        recon_losses = list()
        img_losses = list()
        kl_losses = list()
        outputs = list()
        # loop for each data pairs
        for data, adv_data in dataloader:
            # initialize
            step += 1
            if torch.cuda.is_available():
                model = model.cuda()
                data = data.cuda()
                adv_data = adv_data.cuda()

            # zero grad for optimizer
            optimizer.zero_grad()

            # get data and run model
            output, dsm, dss = model(data)
            distribution = Normal(dsm, dss)

            # calculate losses
            loss, img_recon, kld = recon_loss_function(output, data, distribution, step, 0.1)
            loss.backward()

            # clip for gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # step optimizer
            optimizer.step()

            # record results
            recon_losses.append(loss.item())
            img_losses.append(img_recon.item())
            kl_losses.append(kld.item())
            outputs.append(output)
        
        # write to tensorboard
        writer1.add_scalar('recon_loss', np.sum(recon_losses)/len(recon_losses), step)
        writer2.add_scalar('img_loss', np.sum(img_losses)/len(img_losses), step)
        writer3.add_scalar('kl_loss', np.sum(kl_losses)/len(kl_losses), step)
        if step % 1000 == 0:
            writer1.add_image('original data', data[0], step)
            for i in range(len(outputs)):
                writer1.add_image("reconstruct data {}".format(i), outputs[i][0], step)

        # print out loss
        if step % 50 == 0:
            print("batch {}'s img_recon loss: {:.5f}, recon loss: {:.5f}, kl loss: {:.5f}"\
                .format(step, np.sum(img_losses)/len(img_losses), np.sum(recon_losses)/len(recon_losses),\
                     np.sum(kl_losses)/len(kl_losses)))

        # step scheduler
        if step % 1000 == 0:
            scheduler.step()

        # save model parameters
        if epoch % 5 == 0:
            torch.save(model.cpu().state_dict(), 'params_{}.pt'.format(epoch))
        
    torch.save(model.cpu().state_dict(), 'params.pt')

if __name__ == '__main__':
    main()