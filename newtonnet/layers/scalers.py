import torch
from torch import nn


class ScaleShift(nn.Module):
    r"""Scale and shift layer for standardization.

    Credit : https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py under the MIT License.

    .. math::
       y = x \times \sigma + \mu

    Parameters
    ----------
    mean: torch.Tensor
        mean value :math:`\mu`.

    stddev: torch.Tensor
        standard deviation value :math:`\sigma`.

    Copyright: https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py
    """
    def __init__(self, mean, stddev):
        super(ScaleShift, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

    def forward(self, input):
        """Compute layer output.

        Parameters
        ----------
        input: torch.Tensor
            input data.

        Returns
        -------
        torch.Tensor: layer output.

        """
        y = input * self.stddev + self.mean
        return y


class TrainableScaleShift(nn.Module):
    r"""Trainable scale and shift layer for standardization. Each atom type uses its dedicated scale and shift values

    .. math::
       y = x \times \sigma + \mu

    Parameters
    ----------
    max_z: the maximum atomic number for the molecule
    mean: torch.Tensor
        mean value :math:`\mu`.

    stddev: torch.Tensor
        standard deviation value :math:`\sigma`.

    """
    def __init__(self, max_z, initial_mean=None, initial_stddev=None):
        super(TrainableScaleShift, self).__init__()
        if initial_mean is not None:
            mean = nn.Parameter(torch.zeros(max_z) + initial_mean, requires_grad=True)
        else:
            mean = nn.Parameter(torch.zeros(max_z), requires_grad=True)
        if initial_stddev is not None:
            stddev = nn.Parameter(torch.ones(max_z) * initial_stddev, requires_grad=True)
        else:
            stddev = nn.Parameter(torch.ones(max_z), requires_grad=True)
        self.register_parameter('mean', mean)
        self.register_parameter('stddev', stddev)


    def forward(self, input_energies, z):
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
        selected_mean = self.mean[z][...,None]
        selected_stddev = self.stddev[z][...,None]
        y = input_energies * selected_stddev + selected_mean
        return y