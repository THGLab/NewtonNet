import torch
from torch import nn


class Normalizer(nn.Module):
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
    def __init__(self, data=None, atomic_numbers=None, trainable=False):
        super(Normalizer, self).__init__()
        if data.dim() == 2:    # graph property  # inputs in data_size, _
            mean = data.mean()
            std = data.std()
        elif data.dim() == 3:    # node property  # inputs in data_size, n_atoms, _
            max_z = atomic_numbers.max()
            mean = torch.tensor([data[atomic_numbers == z].mean() for z in range(max_z + 1)])
            std = torch.tensor([data[atomic_numbers == z].std() for z in range(max_z + 1)])
        self.mean = nn.Parameter(mean, requires_grad=trainable)
        self.std = nn.Parameter(std, requires_grad=trainable)


    def forward(self, inputs, atomic_numbers):
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
        if dim == 2:    # graph property  # inputs in data_size, _
            selected_mean = self.mean
            selected_std = self.std
        elif dim == 3:  # node property  # inputs in data_size, n_atoms, _
            selected_mean = self.mean[atomic_numbers][:, :, None]
            selected_std = self.std[atomic_numbers][:, :, None]
        else:
            raise ValueError(f'inputs dimension {dim} is not supported')
        outputs = (inputs - selected_mean) / selected_std
        return outputs
    
    def reverse(self, inputs, atomic_numbers):
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
        if dim == 2:    # graph property  # inputs in data_size, _
            selected_mean = self.mean
            selected_stddev = self.std
        elif dim == 3:    # node property  # inputs in data_size, n_atoms, _
            selected_mean = self.mean[atomic_numbers][:, :, None]
            selected_stddev = self.std[atomic_numbers][:, :, None]
        else:
            raise ValueError(f'inputs dimension {dim} is not supported')
        outputs = inputs * selected_stddev + selected_mean
        return outputs