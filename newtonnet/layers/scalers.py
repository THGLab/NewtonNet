import torch
from torch import nn


class Normalizer(nn.Module):
    '''
    Normalizer layer for property standardization. 
    Copied from: https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py under the MIT License.

    Parameters:
        data (torch.Tensor): The training data to be used for standardization.
        numbers (torch.Tensor): The atomic numbers of the atoms in the molecule.
        trainable (bool): Whether the layer is trainable. Default: False.
    
    Note:
        For graph properties, the same scale and shift values are used for all atom types
        For node properties, each atom type uses its dedicated scale and shift values
    '''
    def __init__(self, data=None, numbers=None, trainable=False):
        super(Normalizer, self).__init__()
        if data.dim() == 2:    # graph property  # inputs in data_size, _
            mean = data.mean()
            std = data.std()
        elif data.dim() == 3:    # node property  # inputs in data_size, n_atoms, _
            max_z = numbers.max()
            mean = torch.tensor([data[numbers == z].mean() for z in range(max_z + 1)])
            std = torch.tensor([data[numbers == z].std() for z in range(max_z + 1)])
        self.mean = nn.Parameter(mean, requires_grad=trainable)
        self.std = nn.Parameter(std, requires_grad=trainable)

    def forward(self, inputs, numbers):
        '''
        Normalize inputs.

        Args:
            inputs (torch.Tensor): The input values.
            numbers (torch.Tensor): The atomic numbers of the atoms in the molecule.

        Returns:
            torch.Tensor: The normalized inputs.
        '''
        dim = inputs.dim()
        if dim == 2:    # graph property  # inputs in data_size, _
            selected_mean = self.mean
            selected_std = self.std
        elif dim == 3:  # node property  # inputs in data_size, n_atoms, _
            selected_mean = self.mean[numbers][:, :, None]
            selected_std = self.std[numbers][:, :, None]
        else:
            raise ValueError(f'inputs dimension {dim} is not supported')
        outputs = (inputs - selected_mean) / selected_std
        return outputs
    
    def reverse(self, inputs, atomic_numbers):
        '''
        Denormalize inputs.

        Args:
            inputs (torch.Tensor): The normalized input values.
            numbers (torch.Tensor): The atomic numbers of the atoms in the molecule.

        Returns:
            torch.Tensor: The denormalized inputs.
        '''
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