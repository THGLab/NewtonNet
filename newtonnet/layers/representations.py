import numpy as np

import torch
from torch import nn


class RadialBesselLayer(nn.Module):
    r"""Radial Bessel functions based on the work by DimeNet: https://github.com/klicperajo/dimenet

    Args:
        n_radials (int, optional): total number of radial functions, :math:`N_g`.
        cutoff (float, optional): cutoff, :math:`\mu_{r_c}`

    """

    def __init__(
        self, n_radial=16, cutoff=5.0, device=None
    ):
        super(RadialBesselLayer, self).__init__()
        self.inv_cutoff = 1/cutoff
        self.frequencies = nn.Parameter(torch.tensor(np.arange(1,n_radial+1) * np.pi, device=device), requires_grad=False)

    def forward(self, distances):
        """Compute smeared-gaussian distance values.

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.

        """
        d_scaled = distances * self.inv_cutoff
        d_scaled = d_scaled.unsqueeze(-1)
        out = torch.sin(self.frequencies * d_scaled)

        return out
