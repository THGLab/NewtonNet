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
        super(ScaleShift, self).__init__()
        self.z_max = z.max().item()
        shift_dense = torch.zeros(self.z_max + 1)
        shift_dense[z] = shift
        self.shift = nn.Embedding.from_pretrained(shift_dense.reshape(-1, 1), freeze=False)
        # if scale.ndim == 0:
        #     self.scale = nn.Embedding.from_pretrained(torch.full((self.z_max + 1, 1), scale), freeze=False)
        #     # self.scale = nn.Parameter(scale, requires_grad=False)
        #     # self.single_scale = True
        # else:
        #     scale_dense = torch.zeros(self.z_max + 1)
        #     scale_dense[z] = scale
        #     self.scale = nn.Embedding.from_pretrained(scale_dense.reshape(-1, 1), freeze=True)
        #     # self.single_scale = False

    def forward(self, inputs, z):
        '''
        Scale and shift inputs.

        Args:
            inputs (torch.Tensor): The input values.
            z (torch.Tensor): The atomic numbers of the atoms in the molecule.

        Returns:
            torch.Tensor: The normalized inputs.
        '''
        # if self.single_scale:
        #     outputs = inputs * self.scale + self.shift(z)
        # else:
        #     outputs = inputs * self.scale(z) + self.shift(z)
        # outputs = inputs * self.scale(z) + self.shift(z)
        outputs = inputs + self.shift(z)
        return outputs
    

class NullScaleShift(nn.Module):
    '''
    Null scale and shift layer for untrained properties. Identity function.
    '''
    def __init__(self):
        super(NullScaleShift, self).__init__()
        self.z_max = 0

    def forward(self, inputs, z):
        return inputs