import torch
import numpy as np
from torch.nn import functional as F
from torch.utils import data

# datset class for dataloader
class Dataset(data.Dataset):
    def __init__(self, data, adv_data):
        self.data = torch.from_numpy(data)
        self.adv_data = torch.from_numpy(adv_data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = self.data[index]
        y = self.adv_data[index]

        return X, y