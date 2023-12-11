import torch
from torch import nn


def get_normalizer_by_string(key, **kwargs):
    if key == 'energy':
        normalizer = GraphPropertyNormalizer(**kwargs)
    elif key == 'forces':
        normalizer = NodePropertyNormalizer(**kwargs)
    else:
        raise ValueError(f'normalizer {key} is not supported')
    return normalizer


class Normalizer(nn.Module):
    '''
    Normalizer layer for graph property standardization. 
    Copied from: https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py under the MIT License.

    Parameters:
        data (torch.Tensor): The training data to be used for standardization.
        atomic_numbers (torch.Tensor): The atomic numbers of the atoms in the molecule.
    
    Note:
        For graph properties, the same scale and shift values are used for all atom types
        For node properties, each atom type uses its dedicated scale and shift values
    '''
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def select(self, atomic_numbers):
        raise NotImplementedError('This is an abstract class. Use GraphPropertyNormalizer or NodePropertyNormalizer instead.')

    def forward(self, inputs: torch.Tensor, atomic_numbers: torch.Tensor):
        '''
        Normalize inputs.

        Args:
            inputs (torch.Tensor): The input values.
            atomic_numbers (torch.Tensor): The atomic numbers of the atoms in the molecule.

        Returns:
            torch.Tensor: The normalized inputs.
        '''
        selected_mean, selected_std = self.select(atomic_numbers)
        while selected_mean.ndim < inputs.ndim:
            selected_mean = selected_mean.unsqueeze(-1)
            selected_std = selected_std.unsqueeze(-1)
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
        selected_mean, selected_std = self.select(atomic_numbers)
        while selected_mean.ndim < inputs.ndim:
            selected_mean = selected_mean.unsqueeze(-1)
            selected_std = selected_std.unsqueeze(-1)
        outputs = inputs * selected_std + selected_mean
        return outputs
    

class GraphPropertyNormalizer(Normalizer):
    '''
    Normalizer layer for graph properties. 
    
    Parameters:
        data (torch.Tensor): The training data to be used for standardization.
        atomic_numbers (torch.Tensor): The atomic numbers of the atoms in the molecule.
    '''
    def __init__(self, data, atomic_numbers):
        mean = data.mean(dim=0)
        std = data.std(dim=0)
        super(GraphPropertyNormalizer, self).__init__(mean, std)

    def select(self, atomic_numbers):
        return self.mean, self.std
    

class NodePropertyNormalizer(Normalizer):
    '''
    Normalizer layer for node properties. 
    
    Parameters:
        data (torch.Tensor): The training data to be used for standardization.
        atomic_numbers (torch.Tensor): The atomic numbers of the atoms in the molecule.
    '''
    def __init__(self, data, atomic_numbers):
        max_atomic_number = atomic_numbers.max()
        mean = torch.tensor([data[atomic_numbers == z].mean() for z in range(max_atomic_number + 1)])
        std = torch.tensor([data[atomic_numbers == z].std() for z in range(max_atomic_number + 1)])
        super(NodePropertyNormalizer, self).__init__(mean, std)

    def select(self, atomic_numbers):
        return self.mean[atomic_numbers], self.std[atomic_numbers]
    
class NullNormalizer(Normalizer):
    '''
    Null normalizer for untrained properties. Identity function.
    '''
    def __init__(self):
        super(NullNormalizer, self).__init__(torch.tensor(0.), torch.tensor(1.))

    def select(self, atomic_numbers):
        return self.mean, self.std