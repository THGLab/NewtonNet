import numpy as np
import torch
from torch import nn


class PolynomialCutoff(nn.Module):
    r"""Class of Polynomial cutoff.

    credit : Directional Message Passing for Molecular Graphs, Johannes Klicpera, Janek Grob, Stephan Gunnemann
             Published at ICLR 2020


    Parameters
    ----------
    cutoff (float, optional): cutoff radius.

    """
    def __init__(self, cutoff=5.0, p=9):
        super(PolynomialCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        self.register_buffer("p", torch.FloatTensor([p]))

    def forward(self, distances):
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        d = distances / self.cutoff

        # Compute values of cutoff function
        cutoffs = 1 - 0.5*(self.p+1)*(self.p+2)*d.pow(self.p) + self.p*(self.p+2)*d.pow(self.p+1) - 0.5*self.p*(self.p+1)*d.pow(self.p+2)

        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()

        return cutoffs


class CosineCutoff(nn.Module):
    r"""Class of Behler cosine cutoff.

    credit : https://github.com/atomistic-machine-learning/schnetpack under the MIT License

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): cutoff radius.

    """
    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs
