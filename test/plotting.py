import torch
import torchvision
from torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from .main import parse_args

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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=False, num_workers=1)

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