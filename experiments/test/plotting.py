import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils import data
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils.preprocess import *
import itertools

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
    parser.add_argument('--gpu_num', type=int, default=2, choices=range(0,5), help='GPU numbers available for parallel training')

    return parser.parse_args()

# plot images
def plot(images, labels, preds=None):
    assert images.shape[0] == 25 and images.shape[0] == 25
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


# plot t-SNE
'''
data: (n_samples, n_features)
label: (n_samples, )
'''
def plot_tsne(data, label, title, path):
    # doing transformation
    pca = PCA(n_components=50)
    pca_data = pca.fit_transform(data, label)
    tsne = TSNE(n_components=2, verbose=1, init='pca',  perplexity=22, n_iter=500)
    tsne_data = tsne.fit_transform(pca_data, label)

    # plotting
    x_min, x_max = np.min(tsne_data, 0), np.max(tsne_data, 0)
    X = (tsne_data - x_min) / (x_max - x_min)

    # get random picked data ids
    np.random.seed(0)
    ids = np.random.randint(0, X.shape[0], size=1000)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(ids):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])

    if title is not None:
        plt.title(title)
    plt.savefig(path, dpi=100)



if __name__ == "__main__":

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
    ================
    Plot the dataset
    ================
    '''
    m_loader = torch.utils.data.DataLoader(datasets.MNIST(root='../data', train=False,\
        download=False, transform=transforms.ToTensor()), batch_size=25)
    f_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(root='../data', train=False,\
        download=False, transform=transforms.ToTensor()), batch_size=25)


    m_data, m_label = iter(m_loader).next()
    plot(m_data, m_label)

    f_data, f_label = iter(f_loader).next()
    plot(f_data, f_label)


    '''
    ========================
    Plot the reuslt of model
    ========================
    '''
    data = np.load('../data/xs_mnist.npy')
    adv_data = np.load('../data/advs_mnist.npy')
    labels = np.load('../data/ys_mnist.npy')
    dataset = Dataset(data, adv_data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=True, num_workers=1)

    args = parse_args()
    # load models
    model = MADVAE(args)
    dic = torch.load('vanilla_params.pt')
    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    model_dict = model.state_dict()
    dic = {k: v for k, v in dic.items() if k in model_dict}
    model_dict.update(dic)
    model.load_state_dict(dic)
    model.eval()
    # get data
    d, ad, l = iter(dataloader).next()
    output, dsm, dss = model(d)

    plot(d, l)
    plot(ad, l)
    plot(output.detach().numpy(), l)


    '''
    ======================================================
    Plot the t-SNE clustering for output data and latent z
    ======================================================
    '''
    args = parse_args()
    # load models
    model = MADVAE(args)
    dic = torch.load('vanilla_params.pt')
    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    model_dict = model.state_dict()
    dic = {k: v for k, v in dic.items() if k in model_dict}
    model_dict.update(dic)
    model.load_state_dict(dic)
    model.eval()

    # prepare data
    transform  = transforms.Compose([transforms.CenterCrop(args.image_size), transforms.ToTensor()])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=1)
    image, label = next(iter(dataloader))

    # adversarial methods
    adv_list = ['fgsm', 'r-fgsm', 'cw']

    # test for accuracy
    images = list()
    labels = list()
    adv_images = [[] for i in range(len(adv_list))]
    outputs = [[] for i in range(len(adv_list))]
    zs = [[] for i in range(len(adv_list))]
    batch = 0
    for image, label in dataloader:
        image = image.cuda()
        label = label.cuda()
        batch += 1
        print(batch)
        images.append(image)
        labels.append(label)
        for i in range(len(adv_list)):
            _, adv_out = add_adv(classifier, image, label, adv_list[i])
            output, dsm, dss, z = model(adv_out)
            adv_images[i].append(adv_out)
            outputs[i].append(output)
            zs[i].append(z)
    for i in range(len(adv_list)):
        adv_images[i] = list(itertools.chain(*adv_images[i]))
        outputs[i] = list(itertools.chain(*outputs[i]))
        zs[i] = list(itertools.chain(*zs[i]))

    # plot original data
    plot_tsne(images, labels, 'Original data t-SNE', '../plots/orginal_test_data_tsne.png')

    # plot attacked data
    plot_tsne(adv_images[0], labels, 'Original data t-SNE', '../plots/attack_fgsm_test_data_tsne.png')
    plot_tsne(adv_images[1], labels, 'Original data t-SNE', '../plots/attack_rfgsm_test_data_tsne.png')
    plot_tsne(adv_images[2], labels, 'Original data t-SNE', '../plots/attack_cw_test_data_tsne.png')

    # plot output data
    plot_tsne(outputs[0], labels, 'Original data t-SNE', '../plots/output_fgsm_test_data_tsne.png')
    plot_tsne(outputs[1], labels, 'Original data t-SNE', '../plots/output_rfgsm_test_data_tsne.png')
    plot_tsne(outputs[2], labels, 'Original data t-SNE', '../plots/output_cw_test_data_tsne.png')

    # plot z
    plot_tsne(zs[0], labels, 'Original data t-SNE', '../plots/z_fgsm_test_data_tsne.png')
    plot_tsne(zs[1], labels, 'Original data t-SNE', '../plots/z_rfgsm_test_data_tsne.png')
    plot_tsne(zs[2], labels, 'Original data t-SNE', '../plots/z_cw_test_data_tsne.png')