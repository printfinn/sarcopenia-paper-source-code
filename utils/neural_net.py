import os

import torch
import numpy as np

import torch.utils.data
import torch.nn.functional as F
from torch import autograd
from torch import optim
from torch import nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
class Net(nn.Module):
    def __init__(self, D_in):
        super(Net, self).__init__()
        self.H1, self.H2 = 128, 128
        self.fc1 = nn.Linear(D_in, self.H1)
        self.fc2 = nn.Linear(self.H1, self.H2)
        self.fc3 = nn.Linear(self.H2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = F.relu(x)
        return x

