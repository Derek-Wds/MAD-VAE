import torch
import torchvision
import numpy as np
import argparse, sys, os
from torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath('..'))
from MAD_VAE import *

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

# plot images
def plot(images, labels, preds=None):
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all')
    ax = ax.flatten()
    for i in range(25):
        img = images[i].reshape(28, 28)
        ax[i].set_title(str(labels[i]))
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()

    plt.show()

# datset class for dataloader
class Dataset(data.Dataset):
    def __init__(self, data, adv_data, labels):
        self.data = torch.from_numpy(data)
        self.adv_data = torch.from_numpy(adv_data)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = self.data[index]
        y = self.adv_data[index]
        l = self.labels[index]

        return X, y, l


'''
========================
Plot the reuslt of model
========================
'''
data = np.load('../data/xs_mnist.npy')
adv_data = np.load('../data/advs_mnist.npy')
labels = np.load('../data/ys_mnist.npy')
dataset = Dataset(data, adv_data, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=250, shuffle=False, num_workers=1)

args = parse_args()
# load models
model = MADVAE(args)
dic = torch.load('../pretrained_model/vanilla/params.pt')
# for name in list(dic.keys()):
    # dic[name.replace('module.', '')] = dic.pop(name)
# model_dict = model.state_dict()
# dic = {k: v for k, v in dic.items() if k in model_dict}
# model_dict.update(dic)
model.load_state_dict(dic)
model.eval()
# get data
d, ad, l = iter(dataloader).next()
output, dsm, dss, _ = model(d)

plot(d, l)
plot(ad, l)
plot(output.detach().numpy(), l)