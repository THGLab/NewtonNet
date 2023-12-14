import torch
from torch import nn


class RadialBesselLayer(nn.Module):
    '''
    Radial Bessel functions based on the work by DimeNet: https://github.com/klicperajo/dimenet

    Parameters:
        n_radials (int): Total number of radial functions. Default: 16.
        cutoff (float): Cutoff. Default: 5.0.
        device (torch.device): Device. Default: torch.device('cpu').
    '''

    def __init__(
        self, 
        n_basis: int = 16,
        cutoff: float = 5.0,
        device: torch.device = torch.device('cpu'),
    ):
        super(RadialBesselLayer, self).__init__()
        self.inv_cutoff = 1 / cutoff
        self.frequencies = nn.Parameter(torch.arange(1, n_basis + 1, requires_grad=False, device=device) * torch.pi)
        self.epsilon = 1.0e-8

    def forward(self, distances):
        '''
        Compute smeared-gaussian distance values.

        Arguments:
            distances (torch.Tensor): interatomic distance values of shape (batch_size, n_atoms, n_atoms).

        Returns:
            torch.Tensor: edge embedding of shape (batch_size, n_atoms, n_atoms, n_radials).
        '''
        d_scaled = distances * self.inv_cutoff
        d_scaled = d_scaled.unsqueeze(-1)
        out = torch.sin(self.frequencies * d_scaled) / (d_scaled + self.epsilon) / self.frequencies

        return out
