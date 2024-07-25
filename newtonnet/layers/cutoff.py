import numpy as np

import torch
from torch import nn


def get_cutoff_by_string(key, **kwargs):
    if key == 'scalednorm':
        cutoff = ScaledNorm(**kwargs)
    elif key == 'poly':
        cutoff = PolynomialCutoff(**kwargs)
    elif key == 'cos':
        cutoff = CosineCutoff(**kwargs)
    else:
        raise NotImplementedError(f'The cutoff function {key} is unknown.')
    return cutoff

class ScaledNorm(nn.Module):
    '''
    Compute scaled norm of interatomic distances.
    Based on Johannes Klicpera, Janek Grob, Stephan Gunnemann. Directional Message Passing for Molecular Graphs. ICLR 2020.

    Parameters:
        r (float): cutoff radius.
    '''
    def __init__(self, r=5.0, **kwargs):
        super(ScaledNorm, self).__init__()
        self.r = r

    def forward(self, disp):
        """Compute scaled norm.

        Args:
            disp (torch.Tensor): values of interatomic distance vectors.

        Returns:
            torch.Tensor: values of scaled norm.

        """
        # Compute values of scaled norm
        dist = torch.norm(disp, dim=-1, keepdim=True)
        dir = disp / dist
        dist = dist / self.r

        return dist, dir

class PolynomialCutoff(nn.Module):
    '''
    Compute polynomial cutoff function.
    Based on Johannes Klicpera, Janek Grob, Stephan Gunnemann. Directional Message Passing for Molecular Graphs. ICLR 2020.

    Parameters:
        p (int): degree of polynomial. Default: 9.

    Notes:
        y = 1 - 0.5 * (p + 1) * (p + 2) * x^p + p * (p + 2) * x^(p + 1) - 0.5 * p * (p + 1) * x^(p + 2)
        y(0) = 1
        y(1) = 0
    '''
    def __init__(self, p=9, **kwargs):
        super(PolynomialCutoff, self).__init__()
        self.p = p

    def forward(self, dist):
        """Compute cutoff.

        Args:
            dist (torch.Tensor): values of scaled interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = 1 \
            - 0.5 * (self.p + 1) * (self.p + 2) * dist.pow(self.p) \
            + self.p * (self.p + 2) * dist.pow(self.p + 1) \
            - 0.5 * self.p * (self.p + 1) * dist.pow(self.p + 2)

        return cutoffs


class CosineCutoff(nn.Module):
    '''
    Compute Behler cosine cutoff function.
    Copied from: https://github.com/atomistic-machine-learning/schnetpack under the MIT License.

    Notes:
        y = 0.5 * (1 + cos(pi * x))
        y(0) = 1
        y(1) = 0
    '''
    def __init__(self, **kwargs):
        super(CosineCutoff, self).__init__()

    def forward(self, dist):
        """Compute cutoff.

        Args:
            dist (torch.Tensor): values of scaled interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(dist * np.pi) + 1.0)

        return cutoffs
