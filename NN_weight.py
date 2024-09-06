import torch
from torch import nn


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_features=15, out_features=15)
        self.fc2 = nn.Linear(in_features=15, out_features=1)

    def sigmoid_activation(self, x):
        return 1 / (1 + torch.exp(-4 * x))

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.sigmoid_activation(self.fc2(x))

        return x
