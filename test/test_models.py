import torch
import torch.nn as nn
import torch.nn.functional as F

# model a
class model_a(nn.Module):
    def __init__(self):
        super(model_a, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, 1, 2)
        self.conv2 = nn.Conv2d(10, 20, 5, 2, 0)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(144, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv2_drop(x)
        x = x.view(-1, 144)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

# model b
class model_b(nn.Module):
    def __init__(self):
        super(model_b, self).__init__()
        self.dropout =nn.Dropout2d(p=0.2)
        self.conv1 = nn.Conv2d(1, 64, 8, 2, 5)
        self.conv2 = nn.Conv2d(64, 128, 6, 2, 0)
        self.conv3 = nn.Conv2d(128, 128, 5, 1, 0)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64)
        x = F.dropout(x, training = self.training)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

# model c
class model_c(nn.Module):
    def __init__(self):
        super(model_c, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 64, 5, 2, 0)
        self.dropout =nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(144, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(-1, 144)
        x = F.relu(self.fc1(x))
        x = F.dropout(self.fc2(x), training = self.training)

        return F.log_softmax(x, dim=1)

# model d
class model_d(nn.Module):
    def __init__(self):
        super(model_d, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        self.dropout = nn.Dropout2d(p=0.25)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.dropout(F.relu(self.fc1(x)), training = self.training)
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

# model e
class model_e(nn.Module):
    def __init__(self):
        super(model_e, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)