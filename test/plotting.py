import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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


m_loader = torch.utils.data.DataLoader(datasets.MNIST(root='../data', train=False,\
     download=False, transform=transforms.ToTensor()), batch_size=25)
f_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(root='../data', train=False,\
     download=False, transform=transforms.ToTensor()), batch_size=25)


m_data, m_label = iter(m_loader).next()
plot(m_data, m_label)

f_data, f_label = iter(f_loader).next()
plot(f_data, f_label)