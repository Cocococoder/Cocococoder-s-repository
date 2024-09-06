import torch
from torch import nn


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_features=15, out_features=15)
        self.fc2 = nn.Linear(in_features=15, out_features=15)
        self.fc3 = nn.Linear(in_features=15, out_features=1) # 注意

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
