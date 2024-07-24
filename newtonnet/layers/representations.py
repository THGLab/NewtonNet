import torch
from torch import nn


def get_representation_by_string(key, **kwargs):
    if key == 'bessel':
        representation = RadialBesselLayer(**kwargs)
    else:
        raise NotImplementedError(f'The representation function {key} is unknown.')
    return representation


class RadialBesselLayer(nn.Module):
    '''
    Radial Bessel functions based on the work by DimeNet: https://github.com/klicperajo/dimenet

    Parameters:
        n_basis (int): Total number of radial functions. Default: 16.

    Notes:
        y = sin(pi * r) / (pi * r)
    '''

    def __init__(self, n_basis=16):
        super(RadialBesselLayer, self).__init__()
        self.n_basis = n_basis
        self.frequencies = nn.Parameter(torch.arange(1, self.n_basis + 1) * torch.pi, requires_grad=False)
        self.epsilon = 1.0e-8

    def forward(self, dist):
        '''
        Compute smeared-gaussian distance values.

        Arguments:
            dist (torch.Tensor): interatomic distance values of shape (batch_size, n_atoms, n_atoms).

        Returns:
            torch.Tensor: edge embedding of shape (batch_size, n_atoms, n_atoms, n_radials).
        '''
        out = torch.sin(self.frequencies * dist) / (dist + self.epsilon) / self.frequencies

        return out
