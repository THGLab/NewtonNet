import torch
from torch import nn
import numpy as np


def get_activation_by_string(key):
    if key == "swish":
        activation = swish
    elif key == 'relu':
        activation = nn.ReLU()
    elif key == 'ssp':
        activation = shifted_softplus
    elif key == 'gelu':
        activation = gelu
    else:
        raise NotImplementedError("The activation function '%s' is unknown."%str(key))
    return activation

def swish(x):
    r"""Compute the self-gated Swish activation function.

    .. math::
       y = x * sigmoid(x)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: Swish activation of input.

    """
    return x * torch.sigmoid(x)

def shifted_softplus(x):
    r"""Compute shifted soft-plus activation function.
    As it is used in the https://github.com/atomistic-machine-learning/schnetpack

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return nn.functional.softplus(x) - np.log(2.0)

def gelu(x):
    r"""Compute the gaussian error linear unit (GELU) activation function.

    .. math::
       y = x * sigmoid(1.702x)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: Swish activation of input.

    """
    return x * torch.sigmoid(1.702*x)