from models import MADVAE, Classifier
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np
import os
import sys
import json

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
parser.add_argument('--norm', type=int, default=0, choices=range(0, 4))
args = parser.parse_args([])

BIG_BATCH_SIZE = 50
BATCH_SIZE = 5

transform = transforms.Compose(
    [transforms.CenterCrop(args.image_size),
     transforms.ToTensor()])
testset = datasets.MNIST('./data',
                         train=False,
                         download=True,
                         transform=transform)
dataloader = torch.utils.data.DataLoader(testset,
                                         batch_size=BIG_BATCH_SIZE,
                                         shuffle=False)

def accuracy(cla, model, suffix="", threshold = 28 * 28 * 0.005, norms=[1.0, 2.5, 3.0, 3.5]):
    total_correct = 0
    total_correct_inbds = 0
    total_inbds = 0
    total = 0
    i = 0
    try:
        for k, (_, labels) in enumerate(dataloader):
            labels = labels.cuda()
            all_ints = []
            all_orig = []
            
            for norm in norms:
                for j in range(BIG_BATCH_SIZE // BATCH_SIZE):
                    i = k * (BIG_BATCH_SIZE // BATCH_SIZE) + j
                    intermediates = torch.load(f"intermediates{suffix}_{norm}/batch_{i}_attack")
                    originals = torch.load(f"intermediates{suffix}_{norm}/batch_{i}_orig")
                    
#                     plt.imshow(intermediates.view(-1, 1, 28,28).cpu().numpy()[2,0,:,:], cmap='gray')
#                     return
#                     return
                    
                    all_ints.append(intermediates)
                    all_orig.append(originals)
                    
            intermediates = torch.cat(all_ints, 0)
            originals = torch.cat(all_orig, 0)
            originals = originals.view(-1, 784)

            clean_images, _, _, _ = model(intermediates.view(-1, 1, 28,28))
            with torch.no_grad():
#                 plt.imshow(intermediates.view(-1, 1, 28,28).cpu().numpy()[0,0,:,:], cmap='gray')
                out = cla(clean_images)
                preds = out.argmax(1)
            
                preds_repeat = torch.stack(preds.chunk(len(norms)), 1)
                labels_repeat = torch.stack(labels.repeat(len(norms)).chunk(len(norms)), 1)
#                 print("pred", preds_repeat)
#                 print("labels", labels_repeat)
#                 return
                
                correct = preds_repeat == labels_repeat
                total_correct += torch.sum(correct).float()
                total += preds_repeat.shape[0] * preds_repeat.shape[1]
                

#                 print(intermediates.shape)
#                 print(originals.shape)
                
                adv_norms = (intermediates-originals).float().pow(2).sum(-1).pow(0.5)
                adv_norms = torch.stack(adv_norms.chunk(BIG_BATCH_SIZE),0)
                
                inbds = adv_norms < threshold
                total_inbds += torch.sum(inbds).float()
                total_correct_inbds += torch.sum(correct * inbds).float()
#                 total_incorrect_inbds += torch.any((torch.logical_not(correct))*(adv_norms<4),1).float().sum()
#             print(total, total_correct, total_inbds, total_correct_inbds)
#             print("Classifier Accuracy %f | Classifier Accuracy in-bounds %f" % (total_correct/total, total_correct_inbds/total_inbds))
    except OSError as e:
#         print(e)
        return total, total_inbds, total_correct/total, total_correct_inbds/total_inbds
    except Exception as e:
        print("Error", e)

def accuracy_paper(cla, model, suffix="", threshold=28 * 28 * 0.005, norms=[1.0, 2.5, 3.0, 3.5]):
    total_correct = 0
    total_correct_inbds = 0
    total_incorrect_inbds = 0
    total_inbds = 0
    total = 0
    i = 0
    try:
        for k, (_, labels) in enumerate(dataloader):
            labels = labels.cuda()
            all_ints = []
            all_orig = []
            
            for norm in norms:
                for j in range(BIG_BATCH_SIZE // BATCH_SIZE):
                    i = k * (BIG_BATCH_SIZE // BATCH_SIZE) + j
                    intermediates = torch.load(f"intermediates{suffix}_{norm}/batch_{i}_attack")
                    originals = torch.load(f"intermediates{suffix}_{norm}/batch_{i}_orig")
                    
#                     plt.imshow(intermediates.view(-1, 1, 28,28).cpu().numpy()[2,0,:,:], cmap='gray')
#                     return
#                     return
                    
                    all_ints.append(intermediates)
                    all_orig.append(originals)
                    
            intermediates = torch.cat(all_ints, 0)
            originals = torch.cat(all_orig, 0)
            originals = originals.view(-1, 784)

            clean_images, _, _, _ = model(intermediates.view(-1, 1, 28,28))
            with torch.no_grad():
#                 plt.imshow(intermediates.view(-1, 1, 28,28).cpu().numpy()[0,0,:,:], cmap='gray')
                out = cla(clean_images.detach())
                preds = out.argmax(1)
            
                preds_repeat = torch.stack(preds.chunk(len(norms)), 1)
                labels_repeat = torch.stack(labels.repeat(len(norms)).chunk(len(norms)), 1)
#                 print(preds_repeat.shape)
#                 print(labels_repeat.shape)
#                 return
                
                correct = (preds_repeat == labels_repeat)
#                 total_correct += torch.sum(correct).float()
#                 total += preds_repeat.shape[0] * preds_repeat.shape[1]
                total_correct += torch.all(correct, 1).float().sum()
                total += preds_repeat.shape[0]
    
#                 print(correct)
#                 return

#                 print(intermediates.shape)
#                 print(originals.shape)
                
                adv_norms = (intermediates-originals).float().pow(2).sum(-1).pow(0.5)
                adv_norms = torch.stack(adv_norms.chunk(len(norms)),1)
            
                total_incorrect_inbds += torch.any(torch.logical_not(correct)*(adv_norms<4),1).float().sum()
#                 total_incorrect_inbds += torch.any((torch.logical_not(correct))*(adv_norms<4),1).float().sum()
#             print(total, total_correct, total_inbds, total_correct_inbds)
            print("Classifier Accuracy %f | Adversarial in-bounds accuracy %f" % (total_correct/total, total_incorrect_inbds/total))
    except OSError as e:
#         print(e)
#         print(total, total_correct, total_inbds, total_correct_inbds)
        return total, total_correct/total, total_incorrect_inbds/total
    except Exception as e:
        print("Error", e)


if __name__ == "__main__":    
    norms = [1.0, 2.5, 3.0, 3.5]
    methods = ['vanilla', 'classification', 'proxi_dist', 'combined', 'identity']

    cla = Classifier(args)
    classifier_pt = torch.load('classifier.pt')
    cla.load_state_dict(classifier_pt)
    cla.eval()

    for method in methods:
        print(method)
        model = MADVAE(args)
        model_pt = torch.load(
            f'../pretrained_model/{method}/params.pt')
        model.load_state_dict(model_pt)
        model.eval()

        if torch.cuda.is_available():
            print("Using CUDA")
            model = model.cuda()
            cla = cla.cuda()

        results = {}

        for norm in norms:
            total, total_inbds, adv, adv_inb = accuracy(cla, model, norms=[norm], suffix=f"_{method}")
            _, adv_old, _ = accuracy_paper(cla, model, norms=[norm], suffix=f"_{method}")
            results[f'{norm}'] = [adv.item(), adv_inb.item(), adv_old.item()]

        total, total_inbds, adv, adv_inb = accuracy(cla, model, suffix=f"_{method}")
        _, adv_old, _ = accuracy_paper(cla, model, suffix=f"_{method}")
        results['all'] = [adv.item(), adv_inb.item(), adv_old.item()]

        with open(f'./results/accuracy_{method}.txt', 'w') as f:
            json.dump(results, f)
