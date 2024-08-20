import torch
from torch import nn
from torch.autograd import grad
from torch_geometric.utils import scatter

from newtonnet.layers.scalers import NullScaleShift


def get_output_by_string(key, n_features, activation, scalers):
    if key == 'energy':
        output_layer = EnergyOutput(n_features, activation, scalers['energy'])
    elif key == 'gradient_force':
        output_layer = GradientForceOutput()
    elif key == 'direct_force':
        output_layer = DirectForceOutput(n_features, activation, scalers['force'])
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


class CustomOutputs:
    def __init__(self, z, disp, atom_node, force_node, edge_index, batch):
        self.z = z
        self.disp = disp
        self.atom_node = atom_node
        self.force_node = force_node
        self.edge_index = edge_index
        self.batch = batch


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
        scaler (nn.Module): The normalizer for the atomic properties.
    '''
    def __init__(self, n_features, activation, scaler):
        super(EnergyOutput, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, 1),
            )
        self.scaler = scaler

    def forward(self, outputs):
        energy = self.layers(outputs.atom_node)
        energy = self.scaler(energy, outputs.z)
        energy = scatter(energy, outputs.batch, dim=0, reduce='sum').reshape(-1)
        outputs.energy = energy
        return outputs

class GradientForceOutput(FirstDerivativeProperty):
    '''
    Gradient force prediction
    '''
    def __init__(self):
        super(GradientForceOutput, self).__init__()
        self.scaler = NullScaleShift()

    def forward(self, outputs):
        force = -grad(
            outputs.energy, 
            outputs.disp, 
            grad_outputs=torch.ones_like(outputs.energy),
            create_graph=True, 
            retain_graph=True,
            )[0]
        force = scatter(force, outputs.edge_index[0], dim=0, reduce='sum') - scatter(force, outputs.edge_index[1], dim=0, reduce='sum')
        outputs.gradient_force = force
        return outputs
    
class DirectForceOutput(DirectProperty):
    '''
    Direct force prediction
    '''
    def __init__(self, n_features, activation, scaler):
        super(DirectForceOutput, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
            )
        self.scaler = scaler

    def forward(self, outputs):
        coeff = self.layers(outputs.atom_node)  # n_nodes, n_features
        force = coeff.unsqueeze(1) * outputs.force_node  # n_nodes, 3, n_features
        force = force.sum(dim=-1)  # n_nodes, 3
        force = self.scaler(force, outputs.z)
        outputs.direct_force = force
        return outputs