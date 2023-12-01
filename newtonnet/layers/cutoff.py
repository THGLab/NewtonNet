import numpy as np
import torch
from torch import nn

def get_cutoff_by_string(key, cutoff=5.0, degree=9):
    if key == "poly":
        cutoff = PolynomialCutoff(cutoff=cutoff, degree=degree)
    elif key == 'cos':
        cutoff = CosineCutoff(cutoff=cutoff)
    else:
        raise NotImplementedError(f'The cutoff function {key} is unknown.')
    return cutoff


class PolynomialCutoff(nn.Module):
    r"""Class of Polynomial cutoff.

    credit : Directional Message Passing for Molecular Graphs, Johannes Klicpera, Janek Grob, Stephan Gunnemann
             Published at ICLR 2020


    Parameters
    ----------
    cutoff (float, optional): cutoff radius.

    """
    def __init__(self, cutoff=5.0, degree=9):
        super(PolynomialCutoff, self).__init__()
        self.register_buffer('cutoff', torch.tensor([cutoff], dtype=torch.float))
        self.register_buffer('degree', torch.tensor([degree], dtype=torch.float))

    def forward(self, distances):
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        distances_scaled = distances / self.cutoff

        # Compute values of cutoff function
        cutoffs = 1 \
            - 0.5 * (self.degree + 1) * (self.degree+2) * distances_scaled.pow(self.degree) \
            + self.degree * (self.degree + 2) * distances_scaled.pow(self.degree + 1) \
            - 0.5 * self.degree * (self.degree + 1) * distances_scaled.pow(self.degree + 2)

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
        self.register_buffer('cutoff', torch.tensor([cutoff], dtype=torch.float))

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
