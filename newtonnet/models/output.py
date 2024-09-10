import torch
from torch import nn
from torch.autograd import grad
from torch_geometric.utils import scatter

from newtonnet.layers.scalers import ScaleShift



def get_output_by_string(key, n_features, activation):
    if key == 'energy':
        output_layer = EnergyOutput(n_features, activation)
    elif key == 'gradient_force':
        output_layer = GradientForceOutput()
    elif key == 'direct_force':
        output_layer = DirectForceOutput(n_features, activation)
    # elif key == 'hessian':
    #     output_layer = SecondDerivativeProperty(
    #         dependent_property='forces',
    #         independent_property='positions',
    #         negate=True,
    #         scaler=scaler,
    #         **kwargs,
    #         )
    else:
        raise NotImplementedError(f'Output type {key} is not implemented yet')
    return output_layer

def get_aggregator_by_string(key):
    if key == 'energy':
        aggregator = SumAggregator()
    elif key == 'gradient_force':
        aggregator = NullAggregator()
    elif key == 'direct_force':
        aggregator = NullAggregator()
    # elif key == 'hessian':
    #     aggregator = NullAggregator()
    else:
        raise NotImplementedError(f'Aggregate type {key} is not implemented yet')
    return aggregator


class CustomOutputSet:
    def __init__(self, **outputs):
        for key, value in outputs.items():
            setattr(self, key, value)


class DirectProperty(nn.Module):
    def __init__(self):
        super(DirectProperty, self).__init__()
class FirstDerivativeProperty(nn.Module):
    def __init__(self):
        super(FirstDerivativeProperty, self).__init__()
class SecondDerivativeProperty(nn.Module):
    def __init__(self):
        super(SecondDerivativeProperty, self).__init__()


class EnergyOutput(DirectProperty):
    '''
    Energy prediction

    Parameters:
        n_features (int): Number of features in the hidden layer.
        activation (nn.Module): Activation function.
    '''
    def __init__(self, n_features, activation):
        super(EnergyOutput, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, 1),
            )

    def forward(self, outputs):
        energy = self.layers(outputs.atom_node)
        # energy = scatter(energy, outputs.batch, dim=0, reduce='sum').reshape(-1)
        # outputs.energy = energy
        return energy

class GradientForceOutput(FirstDerivativeProperty):
    '''
    Gradient force prediction
    '''
    def __init__(self):
        super(GradientForceOutput, self).__init__()

    def forward(self, outputs):
        force = -grad(
            outputs.energy, 
            outputs.disp, 
            grad_outputs=torch.ones_like(outputs.energy),
            create_graph=True, 
            retain_graph=True,
            )[0]
        force = scatter(force, outputs.edge_index[0], dim=0, reduce='sum', dim_size=outputs.atom_node.size(0)) - \
            scatter(force, outputs.edge_index[1], dim=0, reduce='sum', dim_size=outputs.atom_node.size(0))
        # outputs.gradient_force = force
        return force
    
class DirectForceOutput(DirectProperty):
    '''
    Direct force prediction
    '''
    def __init__(self, n_features, activation):
        super(DirectForceOutput, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
            )

    def forward(self, outputs):
        force = self.layers(outputs.atom_node).unsqueeze(1) * outputs.force_node  # n_nodes, 3, n_features
        force = force.sum(dim=-1)  # n_nodes, 3
        # outputs.direct_force = force
        return force
    

class SumAggregator(nn.Module):
    def __init__(self):
        super(SumAggregator, self).__init__()

    def forward(self, output, outputs):
        return scatter(output, outputs.batch, dim=0, reduce='sum').reshape(-1)

class NullAggregator(nn.Module):
    def __init__(self):
        super(NullAggregator, self).__init__()

    def forward(self, output, outputs):
        return output