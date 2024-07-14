import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SumAggregation, MeanAggregation, StdAggregation
from torch_geometric.utils import one_hot


def get_scaler_by_string(key, stats):
    if key == 'energy':
        scaler = ScaleShift(stats['z'], stats['energy_shift'], stats['energy_scale'])
    elif key == 'forces':
        scaler = NullScaleShift()
    elif key == 'hessian':
        scaler = NullScaleShift()
    else:
        print(f'scaler {key} is not supported, use NullScaleShift.')
        scaler = NullScaleShift()
    return scaler


class ScaleShift(nn.Module):
    '''
    Node-level scale and shift layer.
    
    Parameters:
        z (torch.Tensor): The atomic numbers of the atoms in the molecule.
        shift (torch.Tensor): The shift values for the properties.
        scale (torch.Tensor): The scale values for the properties.
    '''
    def __init__(self, z, shift, scale):
        super(GraphPropertyScaleShift, self).__init__()
        self.z_max = z.max().item()
        shift_dense = torch.zeros(self.z_max + 1)
        shift_dense[z] = shift
        self.shift = nn.Embedding.from_pretrained(shift_dense.reshape(-1, 1), requires_grad=True)
        if len(scale) == 1:
            self.scale = nn.Embedding.from_pretrained(torch.full((self.z_max + 1, 1), scale), requires_grad=True)
        else:
            scale_dense = torch.zeros(self.z_max + 1)
            scale_dense[z] = scale
            self.scale = nn.Embedding.from_pretrained(scale_dense.reshape(-1, 1), requires_grad=True)

    def forward(self, inputs, z):
        '''
        Scale and shift inputs.

        Args:
            inputs (torch.Tensor): The input values.
            z (torch.Tensor): The atomic numbers of the atoms in the molecule.

        Returns:
            torch.Tensor: The normalized inputs.
        '''
        outputs = input * self.scale(z) + self.shift(z)
        return outputs

    def __repr__(self):
        return f'{self.__class__.__name__}(shift={self.shift.weight.flatten().tolist()}, scale={self.scale.data.flatten().mean().item()})'
    

class NullScaleShift(nn.Module):
    '''
    Null scale and shift layer for untrained properties. Identity function.
    '''
    def __init__(self):
        super(NullScaleShift, self).__init__()

    def forward(self, inputs, z):
        return inputs