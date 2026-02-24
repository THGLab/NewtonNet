from ase.data import covalent_radii

import torch
from torch import nn
from torch_geometric.utils import scatter

from newtonnet.layers.representations import PolynomialCutoff


class ZBLBasis(nn.Module):
    '''
    Implementation of the Ziegler-Biersack-Littmark (ZBL) potential
    with a polynomial cutoff envelope.
    Based on Ilyes Batatia et al. A foundation model for atomistic materials chemistry. J Chem Phys 2025.
    '''

    def __init__(self, p=6, trainable=False, **kwargs):
        super().__init__()
        # Pre-calculate the p coefficients for the ZBL potential
        self.register_buffer('c', torch.tensor([0.1818, 0.5099, 0.2802, 0.02817]))
        self.register_buffer('p', torch.tensor(p, dtype=torch.int))
        self.register_buffer('covalent_radii', torch.tensor(covalent_radii))
        if trainable:
            self.a_exp = torch.nn.Parameter(torch.tensor(0.300, requires_grad=True))
            self.a_prefactor = torch.nn.Parameter(
                torch.tensor(0.4543, requires_grad=True)
            )
        else:
            self.register_buffer('a_exp', torch.tensor(0.300))
            self.register_buffer('a_prefactor', torch.tensor(0.4543))
        self.cutoff = PolynomialCutoff(p=9)

    def forward(
        self,
        z: torch.Tensor,
        disp: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        z_i = z[edge_index[0]]
        z_j = z[edge_index[1]]
        a = (
            self.a_prefactor
            * 0.529
            / (torch.pow(z_i, self.a_exp) + torch.pow(z_j, self.a_exp))
        )
        dist = torch.linalg.norm(disp, dim=-1)
        dist_scaled = dist / a
        phi = (
            self.c[0] * torch.exp(-3.2 * dist_scaled)
            + self.c[1] * torch.exp(-0.9423 * dist_scaled)
            + self.c[2] * torch.exp(-0.4028 * dist_scaled)
            + self.c[3] * torch.exp(-0.2016 * dist_scaled)
        )
        v_edge = (14.3996 * z_i * z_j) / dist * phi
        dist_max = self.covalent_radii[z_i] + self.covalent_radii[z_j]
        envelope = self.cutoff(dist / dist_max) * ( dist <= dist_max ).float()
        v_edge = 0.5 * v_edge * envelope
        v_node = scatter(v_edge, edge_index[0], dim=0, dim_size=z.size(0))
        return v_node.unsqueeze(-1)