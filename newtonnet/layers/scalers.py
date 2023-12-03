import torch
from torch import nn


class ScaleShift(nn.Module):
    r"""Trainable scale and shift layer for standardization. Each atom type uses its dedicated scale and shift values

    Credit : https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py under the MIT License.

    .. math::
       y = x \times \sigma + \mu

    Parameters
    ----------
    max_z: the maximum atomic number for the molecule
    mean: torch.Tensor
        mean value :math:`\mu`.

    stddev: torch.Tensor
        standard deviation value :math:`\sigma`.

    Copyright: https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py

    """
    def __init__(self, max_z, mean=0.0, stddev=1.0, trainable=True):

        super(ScaleShift, self).__init__()

        self.mean = nn.Parameter(torch.tensor([mean] * max_z, dtype=torch.float), requires_grad=trainable)
        self.std = nn.Parameter(torch.tensor([stddev] * max_z, dtype=torch.float), requires_grad=trainable)


    def forward(self, inputs, atomic_numbers, scale=True, shift=True):
        """Compute layer output.

        Parameters
        ----------
        input_energies: torch.Tensor
            input atomic energies.

        z: torch.Tensor
            input atomic numbers

        Returns
        -------
        torch.Tensor: layer output.

        """
        dim = inputs.dim()
        if dim == 2:    # inputs in batch_size, 1
            selected_mean = self.mean[atomic_numbers]
            selected_std = self.std[atomic_numbers]
        elif dim == 3:  # inputs in batch_size, n_atoms, 1
            selected_mean = self.mean[atomic_numbers][:, :, None]
            selected_std = self.std[atomic_numbers][:, :, None]
        else:
            raise ValueError(f'inputs dimension {dim} is not supported')
        outputs = inputs
        if shift:
            outputs = outputs - selected_mean
        if scale:
            outputs = outputs / selected_std
        return outputs
    
    def reverse(self, inputs, atomic_numbers, scale=True, shift=True):
        """Compute layer output.

        Parameters
        ----------
        input_energies: torch.Tensor
            input atomic energies.

        z: torch.Tensor
            input atomic numbers

        Returns
        -------
        torch.Tensor: layer output.

        """
        dim = inputs.dim()
        if dim == 2:    # inputs in batch_size, 1
            selected_mean = self.mean[atomic_numbers]
            selected_stddev = self.std[atomic_numbers]
        elif dim == 3:
            selected_mean = self.mean[atomic_numbers][:, :, None]
            selected_stddev = self.std[atomic_numbers][:, :, None]
        else:
            raise ValueError(f'inputs dimension {dim} is not supported')
        outputs = inputs
        if scale:
            outputs = outputs * selected_stddev
        if shift:
            outputs = outputs + selected_mean
        return outputs