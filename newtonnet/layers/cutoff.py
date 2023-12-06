import numpy as np

import torch
from torch import nn


def get_cutoff_by_string(key, cutoff=5.0, p=9):
    if key == 'poly':
        cutoff = PolynomialCutoff(cutoff=cutoff, p=p)
    elif key == 'cos':
        cutoff = CosineCutoff(cutoff=cutoff)
    else:
        raise NotImplementedError(f'The cutoff function {key} is unknown.')
    return cutoff


class PolynomialCutoff(nn.Module):
    '''
    Compute polynomial cutoff function.
    Based on Johannes Klicpera, Janek Grob, Stephan Gunnemann. Directional Message Passing for Molecular Graphs. ICLR 2020.

    Parameters:
        cutoff (float): cutoff radius. Default: 5.0.
        p (int): degree of polynomial. Default: 9.

    Notes:
        x = r / r_cutoff
        y = 1 - 0.5 * (p + 1) * (p + 2) * x^p + p * (p + 2) * x^(p + 1) - 0.5 * p * (p + 1) * x^(p + 2)
        y(0) = 1
        y(x) = 0 for x >= 1
    '''
    def __init__(self, cutoff=5.0, p=9):
        super(PolynomialCutoff, self).__init__()
        self.cutoff = cutoff
        self.p = p

    def forward(self, distances):
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Scale distances by cutoff radius
        distances = distances / self.cutoff

        # Compute values of cutoff function
        cutoffs = 1 \
            - 0.5 * (self.p + 1) * (self.p + 2) * distances.pow(self.p) \
            + self.p * (self.p + 2) * distances.pow(self.p + 1) \
            - 0.5 * self.p * (self.p + 1) * distances.pow(self.p + 2)

        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff)

        return cutoffs


class CosineCutoff(nn.Module):
    '''
    Compute Behler cosine cutoff function.
    Copied from: https://github.com/atomistic-machine-learning/schnetpack under the MIT License.

    Parameters:
        cutoff (float): cutoff radius. Default: 5.0.

    Notes:
        x = r / r_cutoff
        y = 0.5 * (1 + cos(pi * x))
        y(0) = 1
        y(x) = 0 for x >= 1
    '''
    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff = cutoff

    def forward(self, distances):
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Scale distances by cutoff radius
        distances = distances / self.cutoff

        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff)

        return cutoffs
