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
from utils.classifier import *
from utils.scheduler import *
from utils.dataset import *

# argument parser
def parse_args():
    desc = "MAD-VAE for adversarial defense"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Training epoch numbers')
    parser.add_argument('--h_dim', type=int, default=4096, help='Hidden dimensions')
    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimensions for images')
    parser.add_argument('--image_channels', type=int, default=1, help='Image channels')
    parser.add_argument('--image_size', type=int, default=28, help='Image size (default to be squared images)')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of image classes')
    parser.add_argument('--log_dir', type=str, default='logs', help='Logs directory')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the Adam optimizer')
    parser.add_argument('--closs_weight', type=float, default=0.1, help='Weight for classification loss functions')
    parser.add_argument('--ploss_weight', type=float, default=0.01, help='Weight for proximity loss functions')
    parser.add_argument('--dloss_weight', type=float, default=0.00001, help='Weight for distance loss functions')
    parser.add_argument('--data_root', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='pretrained_model', help='Pretrained model directory')
    parser.add_argument('--use_gpu', type=bool, default=True, help='If use GPU for training')
    parser.add_argument('--gpu_num', type=int, default=2, choices=range(0,5), help='GPU numbers available for parallel training')

    return parser.parse_args()

# main function
def main():
    args = parse_args()
    # make directories for pretrained models
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    # prepare dataset
    data = np.load('data/xs_mnist.npy') # image data in npy file
    labels = np.load('data/ys_mnist.npy') # labels data in npy file
    adv_data = np.load('data/advs_mnist.npy') # adversarial image data in npy file
    dataset = Dataset(data, labels, adv_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)

    # summary writer for tensorboard
    writer1 = SummaryWriter(args.log_dir+'/recon_loss')
    writer2 = SummaryWriter(args.log_dir+'/img_loss')
    writer3 = SummaryWriter(args.log_dir+'/kl_loss')
    writer4 = SummaryWriter(args.log_dir+'/c_loss')
    writer5 = SummaryWriter(args.log_dir+'/pd_loss')

    # create modules needed
    model, proximity, distance, classifier, optimizer, scheduler,\
         optimizer1, scheduler1, optimizer2, scheduler2 = init_models(args)
    
    # tratinig steps
    step = 0
    for epoch in range(1, args.epochs+1):
        print('Epoch: {}'.format(epoch))
        recon_losses, img_losses, kl_losses, c_losses, pd_losses, datas, adv_datas, outputs, step = \
            train(args, dataloader, model, classifier, proximity, distance, optimizer, optimizer1, optimizer2, step, epoch)
        
        # write to tensorboard
        writer1.add_scalar('recon_loss', np.sum(recon_losses)/len(recon_losses), step)
        writer2.add_scalar('img_loss', np.sum(img_losses)/len(img_losses), step)
        writer3.add_scalar('kl_loss', np.sum(kl_losses)/len(kl_losses), step)
        writer4.add_scalar('c_loss', np.sum(c_losses)/len(c_losses), step)
        writer5.add_scalar('pd_loss', np.sum(pd_losses)/len(pd_losses), step)
        for i in range(len(datas)):
            writer1.add_image('original data', datas[i][0], step)
            writer1.add_image('adv data', adv_datas[i][0], step)
            writer1.add_image("reconstruct data", outputs[i][0], step)

        # print out loss
        print("batch {}'s img_recon loss: {:.5f}, recon loss: {:.5f}, kl loss: {:.5f}"\
            .format(step, np.sum(img_losses)/len(img_losses), np.sum(recon_losses)/len(recon_losses),\
                    np.sum(kl_losses)/len(kl_losses)))

        # scheduler step
        scheduler.step()
        scheduler1.step()
        scheduler2.step()

        # save model parameters
        if epoch % 5 == 0:
            torch.save(model.state_dict(), '{}/combined/params_{}.pt'.format(args.model_dir, epoch))
        
    torch.save(model.state_dict(), '{}/combined/params.pt'.format(args.model_dir))

# training function
def train(args, dataloader, model, classifier, proximity, distance, optimizer, optimizer1, optimizer2, step, epoch):
    # init output lists
    recon_losses = list()
    img_losses = list()
    kl_losses = list()
    c_losses = list()
    pd_losses = list()
    datas = list()
    adv_datas = list()
    outputs = list()
    # loop for each data pairs
    for data, label, adv_data in dataloader:
        # initialize
        step += 1
        if torch.cuda.is_available():
            model = model.cuda()
            data = data.cuda()
            label = label.cuda()
            adv_data = adv_data.cuda()

        # zero grad for optimizer
        optimizer.zero_grad()
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # get data and run model
        output, dsm, dss, z = model(adv_data)
        distribution = Normal(dsm, dss)

        # calculate losses
        r_loss, img_recon, kld = recon_loss_function(output, data, distribution, step, epoch/100)
        c_loss = classification_loss(output, label, classifier)
        p_loss = proximity(z, label)
        d_loss = distance(z, label)
        loss = r_loss + args.closs_weight * c_loss + args.ploss_weight * p_loss - args.dloss_weight * d_loss
        loss.backward()

        # clip for gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(proximity.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(distance.parameters(), 1)

        # step optimizer
        optimizer.step()
        for param in proximity.parameters():
            param.grad.data *= (1. / args.ploss_weight)
        optimizer1.step()
        for param in distance.parameters():
            param.grad.data *= (1. / args.dloss_weight)
        optimizer2.step()

        # record results
        recon_losses.append(loss.cpu().item())
        img_losses.append(img_recon.cpu().item())
        kl_losses.append(kld.cpu().item())
        c_losses.append(c_loss.cpu().item())
        pd_losses.append(p_loss.cpu().item() - d_loss.cpu().item())
        outputs.append(output.cpu())
        datas.append(data.cpu())
        adv_datas.append(adv_data.cpu())
    
    return recon_losses, img_losses, kl_losses, c_losses, pd_losses, datas, adv_datas, outputs, step

# init models to be used
def init_models(args):
    # construct model, classifier and loss module
    model = MADVAE(args)
    classifier = Classifier(args)
    proximity = Proximity(args)
    distance = Distance(args)
    if args.use_gpu and torch.cuda.is_available():
        # multi gpu training
        if args.gpu_num > 1:
            model = torch.nn.DataParallel(model, device_ids=range(args.gpu_num))
            classifier = torch.nn.DataParallel(classifier, device_ids=range(args.gpu_num))
            proximity = torch.nn.DataParallel(proximity, device_ids=range(args.gpu_num))
            distance = torch.nn.DataParallel(distance, device_ids=range(args.gpu_num))
            model = model.module
            classifier = classifier.module
            proximity = proximity.module
            distance = distance.module
        # move to cuda
        model.apply(weights_init)
        model = model.cuda()
        classifier = classifier.cuda()
        proximity = proximity.cuda()
        distance = distance.cuda()
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    
    # training module
    model.train()
    proximity = proximity.train()
    distance = distance.train()
    # evaluation module
    classifier.load_state_dict(torch.load('pretrained_model/classifier_mnist.pt'))
    classifier.eval()

    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MinExponentialLR(optimizer, gamma=0.998, minimum=1e-5)
    optimizer1 = optim.Adam(proximity.parameters(), lr=args.lr*50)
    scheduler1 = MinExponentialLR(optimizer1, gamma=0.998, minimum=1e-5)
    optimizer2 = optim.Adam(distance.parameters(), lr=args.lr/100)
    scheduler2 = MinExponentialLR(optimizer2, gamma=0.998, minimum=1e-5)

    return model, proximity, distance, classifier, optimizer, scheduler,\
         optimizer1, scheduler1, optimizer2, scheduler2

# initialize model weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Block') == -1:
        m.weight.data.normal_(0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':
    main()