import numpy as np

import torch
from torch import nn


def get_activation_by_string(key):
    if key == 'swish':
        activation = nn.SiLU()
    elif key == 'relu':
        activation = nn.ReLU()
    elif key == 'elu':
        activation = nn.ELU()
    elif key == 'leaky_relu':
        activation = nn.LeakyReLU()
    elif key == 'tanh':
        activation = nn.Tanh()
    elif key == 'sigmoid':
        activation = nn.Sigmoid()
    elif key == 'softplus':
        activation = nn.Softplus()
    elif key == 'gelu':
        activation = nn.GELU()
    elif key == 'ssp':
        activation = ShiftedSoftplus()
    else:
        raise NotImplementedError("The activation function '%s' is unknown."%str(key))
    return activation


class ShiftedSoftplus(nn.Module):
    '''
    Compute shifted soft-plus activation function.
    Copied from: https://github.com/atomistic-machine-learning/schnetpack under the MIT License.

    Notes:
        y = ln(1 + e^(-x)) - ln(2)
    '''
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.softplus = nn.Softplus()
        self.shift = torch.log(torch.tensor(2.0))

    def forward(self, x):
        return self.softplus(x) - self.shift