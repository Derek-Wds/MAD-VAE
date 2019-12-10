import sys, os
sys.path.insert(0, os.path.abspath('..'))
import warnings
warnings.filterwarnings("ignore")
import foolbox
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST, FashionMNIST
from classifier import *
from adversarial import *
from utils.dataset import *
from utils.adversarial import *
from utils.classifier import *
from MAD_VAE import *

# argument parser
def parse_args():
    desc = "MAD-VAE for adversarial defense"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Training epoch numbers')
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
    parser.add_argument('--gpu_num', type=int, default=2, choices=range(0,5), help='GPU numbers available for parallel training')

    return parser.parse_args()

# generate latex table
def generate(values):
    output = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{"
    for i in range(10):
        output += "c|"
    output += "c}\n"
    output += "\\textbf{Labels}"
    for i in range(10):
        output += " & "
        output += "\\textbf{%d}" % i
    output += " \\\\ \hline \n"
    for i in range(10):
        output += "\\textbf{%d}" % i
        for j in range(10):
            output += " & "
            output += str(values[i*10+j])
        output += " \\\\ \n"
    output += "\\textbf{Accuracy}"
    values = np.array(values).reshape(10,10)
    sums = np.sum(values, axis=0)
    for i in range(10):
        output += " & "
        output += "{0:.2f}".format((values[i,i]/sums[i]).item())
    output += " \n"
    output += "\\end{tabular} \n\
\\caption{} \n\
\\label{table:confusion} \n\
\\end{table}"
    
    print(output)

if __name__ == "__main__":
    args = parse_args()
    # load models
    model = MADVAE(args)
    dic = torch.load('old_pretrained/combined/params.pt')
    model.load_state_dict(dic)
    model.eval()
    model = model.cuda()

    # init and load classifier
    classifier = Classifier(args)
    classifier.load_state_dict(torch.load('pretrained_model/classifier_mnist.pt'))
    classifier.eval()
    classifier = classifier.cuda()

    # adversarial methods
    adv_accuracy = {'fgsm': np.zeros(100), 'r-fgsm': np.zeros(100), 'cw': np.zeros(100)}

    total = len(dataset)

    for adv in adv_accuracy:
        # init dataset for TESTING
        data = np.load('../testdata/{}_xs_mnist.npy'.format(adv))
        adv_data = np.load('../testdata/{}_advs_mnist.npy'.format(adv))
        labels = np.load('../testdata/{}_ys_mnist.npy'.format(adv))
        dataset = Dataset(data, adv_data, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=1)
        
        for image, adv_img, label in dataloader:
            image = image.cuda()
            adv_img = adv_img.cuda()
            label = label.cuda()

            # get model output
            def_out, _, _, _ = model(adv_img)
            adv_out_class = classifier(def_out)

            # get model predicted class
            adversarial_class = torch.argmax(adv_out_class, 1)
            
            # update confusion matrix
            adv_accuracy[adv][(adversarial_class*10+label).astype(int)] += 1

    output = np.zeros(100)
    for adv in adv_accuracy:
        output += adv_accuracy[adv]
    generate(output.tolist())