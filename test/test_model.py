import torch
import torch.nn as nn
import torch.nn.functional as F

# model a
class model_a(nn.Module):
    def __init__(self):
        super(model_a, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# model b
class model_b(nn.Module):
    def __init__(self):
        super(model_b, self).__init__()
    
    def forward(self, x):

        return x

# model c
class model_c(nn.Module):
    def __init__(self):
        super(model_c, self).__init__()
    
    def forward(self, x):

        return x

# model d
class model_d(nn.Module):
    def __init__(self):
        super(model_d, self).__init__()
    
    def forward(self, x):

        return x