import torch, json
import torchvision
import numpy as np
import argparse, sys, os
from torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath('..'))
from MAD_VAE import *
from classifier import *
from adversarial import *
from test.plotting import *
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
    parser.add_argument('--num_classes', type=int, default=10, help='Number of image classes')
    parser.add_argument('--log_dir', type=str, default='v_logs', help='Logs directory')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the Adam optimizer')
    parser.add_argument('--data_root', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='pretrained_model', help='Pretrained model directory')
    parser.add_argument('--use_gpu', type=bool, default=True, help='If use GPU for training')
    parser.add_argument('--gpu_num', type=int, default=1, choices=range(0,5), help='GPU numbers available for parallel training')
    parser.add_argument('--experiment', type=int, default=1, choices=range(0,4))
    return parser.parse_args()



if __name__ == "__main__":    
    models = ['vanilla', 'classification', 'proxi_dist', 'combined']
    for i in range(4):
        args = parse_args()
        model = MADVAE(args)
        dic = torch.load('old_pretrained/{}/params.pt'.format(models[i]))
        model.load_state_dict(dic)
        model.eval()
        model = model.cuda()

        # init and load classifier
        classifier = Classifier(args)
        classifier.load_state_dict(torch.load('pretrained_model/classifier_mnist.pt'))
        classifier.eval()
        classifier = classifier.cuda()

        # init dataset for TESTING
        transform  = transforms.Compose([transforms.CenterCrop(args.image_size), transforms.ToTensor()])
        dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=1)

        # adversarial methods
        adv_accuracy = {'fgsm': 0, 'r-fgsm': 0, 'cw': 0, 'mi-fgsm': 0, 'pgd': 0, 'single': 0}

        # test for accuracy
        for adv in adv_accuracy:
            true = 0
            total = len(dataset)
            for image, label in dataloader:
                image = image.cuda()
                label = label.cuda()

                output, adv_out = add_adv(classifier, image, label, adv)
                output_class = classifier(output)
                def_out, _, _, _ = model(adv_out)
                adv_out_class = classifier(def_out)

                true_class = torch.argmax(output_class, 1)
                adversarial_class = torch.argmax(adv_out_class, 1)

                print(f'attack method {adv}')
                print(f'actual class {true_class}')
                print(f'adversarial class {adversarial_class}')

                true += torch.sum(torch.eq(true_class, adversarial_class))

                print(int(true) / total)
            adv_accuracy[adv] = int(true) / total
            print(int(true) / total)
            print('=================================')
        print()


        with open(f'./accuracy_{models[i]}.txt', 'w') as f:
            json.dump(adv_accuracy, f)
