import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SumAggregation, MeanAggregation, StdAggregation
from torch_geometric.utils import one_hot


def get_scaler_by_string(key, dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for data in loader:
        break
    if key == 'energy':
        scaler = GraphPropertyScaleShift(data.energy, data.z, data.batch)
    elif key == 'forces':
        scaler = NullScaleShift()
    else:
        raise ValueError(f'scaler {key} is not supported')
    return scaler


class GraphPropertyScaleShift(nn.Module):
    '''
    Scale and shift layer for graph properties.
    
    Parameters:
        data (torch.Tensor): The training data to be used for standardization.
        z (torch.Tensor): The atomic numbers of the atoms in the molecule.
        batch (torch.Tensor): The batch indices.
        freeze (bool): Whether to freeze the scale parameter.
    '''
    def __init__(self, data, z, batch):
        super(GraphPropertyScaleShift, self).__init__()
        sum_aggr = SumAggregation()
        formula = sum_aggr(one_hot(z.long()), batch)    # z count for each graph
        shift = torch.linalg.lstsq(formula, data).solution
        scale = ((data - torch.matmul(formula, shift)).square().sum() / len(z)).sqrt()
        scale = torch.ones_like(shift) * scale
        
        self.shift = nn.Embedding.from_pretrained(shift.reshape(-1, 1))
        self.scale = nn.Parameter(scale.reshape(-1, 1))

    def forward(self, inputs, z):
        '''
        Scale and shift inputs.

        Args:
            inputs (torch.Tensor): The input values.
            z (torch.Tensor): The atomic numbers of the atoms in the molecule.

        Returns:
            torch.Tensor: The normalized inputs.
        '''
        outputs = input * self.scale + self.shift(z)
        return outputs

    def __repr__(self):
        return f'{self.__class__.__name__}(shift={self.shift.weight.flatten().tolist()}, scale={self.scale.data.flatten().mean().item()})'
    

class NodePropertyScaleShift(nn.Module):
    '''
    Scale and shift layer for node properties.
    
    Parameters:
        data (torch.Tensor): The training data to be used for standardization.
        z (torch.Tensor): The atomic numbers of the atoms in the molecule.
        freeze (bool): Whether to freeze the scale parameter.
    '''
    def __init__(self, data, z, freeze=False):
        super(NodePropertyNormalizer, self).__init__(mean, std)
        mean_aggr = MeanAggregation()
        shift = mean_aggr(data, z)
        std_aggr = StdAggregation()
        scale = std_aggr(data, z) 

        self.shift = nn.Embedding.from_pretrained(shift, freeze=freeze)
        self.scale = nn.Embedding.from_pretrained(scale, freeze=freeze)

    def forward(self, inputs, z):
        '''
        Scale and shift inputs.

        Args:
            inputs (torch.Tensor): The input values.
            z (torch.Tensor): The atomic numbers of the atoms in the molecule.

        Returns:
            torch.Tensor: The normalized inputs.
        '''
        outputs = inputs * self.scale(z) + self.shift(z)
        return outputs

    def __repr__(self):
        return f'{self.__class__.__name__}(shift={self.shift.weight.flatten().tolist()}, scale={self.scale.weight.flatten().tolist()})'
    

class NullScaleShift(nn.Module):
    '''
    Null scale and shift layer for untrained properties. Identity function.
    '''
    def __init__(self):
        super(NullScaleShift, self).__init__()

    def forward(self, inputs, z):
        return inputs