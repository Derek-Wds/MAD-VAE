import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Model A
MNIST accuracy: 0.993900
Fashion-MNIST accuracy: 0.923600
'''
class model_a(nn.Module):
    def __init__(self):
        super(model_a, self).__init__()
        self.name = 'Model_A'
        self.conv1 = nn.Conv2d(1, 64, 5, 1, 2)
        self.conv2 = nn.Conv2d(64, 64, 5, 2, 0)
        self.dropout = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        self.batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(self.batch_size, -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

'''
Model B
MNIST accuracy: 0.992500
Fashion-MNIST accuracy: 0.912000
'''
class model_b(nn.Module):
    def __init__(self):
        super(model_b, self).__init__()
        self.name = 'Model_B'
        self.dropout = nn.Dropout2d(p=0.2)
        self.conv1 = nn.Conv2d(1, 64, 8, 2, 5)
        self.conv2 = nn.Conv2d(64, 128, 6, 2, 0)
        self.conv3 = nn.Conv2d(128, 128, 5, 1, 0)
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        self.batch_size = x.size(0)
        x = self.dropout(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.dropout(x, training = self.training)
        x = x.view(self.batch_size, -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

'''
Model C
MNIST accuracy: 0.993800
Fashion-MNIST accuracy: 0.923000
'''
class model_c(nn.Module):
    def __init__(self):
        super(model_c, self).__init__()
        self.name = 'Model_C'
        self.conv1 = nn.Conv2d(1, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 64, 5, 2, 0)
        self.dropout = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        self.batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(self.batch_size, -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

'''
Model D
MNIST accuracy: 0.981200
Fashion-MNIST accuracy: 0.889300
'''
class model_d(nn.Module):
    def __init__(self):
        super(model_d, self).__init__()
        self.name = 'Model_D'
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        self.dropout = nn.Dropout2d(p=0.25)
    
    def forward(self, x):
        self.batch_size = x.size(0)
        x = x.view(self.batch_size, -1)
        x = F.dropout(F.relu(self.fc1(x)), training = self.training)
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

'''
Model E
MNIST accuracy: 0.980700
Fashion-MNIST accuracy: 0.890500
'''
class model_e(nn.Module):
    def __init__(self):
        super(model_e, self).__init__()
        self.name = 'Model_E'
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
    
    def forward(self, x):
        self.batch_size = x.size(0)
        x = x.view(self.batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(self.batch_size, -1)

        return F.log_softmax(x, dim=1)