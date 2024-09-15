import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, dims):
        '''
        dims: list of dimensions (including input and output) for linear layers
        '''
        super(MLP, self).__init__()
        self.module = nn.Sequential(
            *[
                layer 
                for i in range(1, len(dims)-1)
                for layer in (nn.Linear(dims[i-1], dims[i]), nn.ReLU())
            ],
            nn.Linear(dims[-2], dims[-1])
        )

    def forward(self, x):
        return self.module(x)