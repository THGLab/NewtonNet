import torch
from torch import nn
from torch.autograd import grad
from torch_geometric.utils import scatter

from newtonnet.layers.scalers import ScaleShift



def get_output_by_string(key, n_features=None, activation=None):
    if key == 'energy':
        output_layer = EnergyOutput(n_features, activation)
    elif key == 'gradient_force':
        output_layer = GradientForceOutput()
    elif key == 'direct_force':
        output_layer = DirectForceOutput(n_features, activation)
    elif key == 'hessian':
        output_layer = HessianOutput()
    elif key == 'virial':
        output_layer = VirialOutput()
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
    elif key == 'hessian':
        aggregator = NullAggregator()
    elif key == 'virial':
        aggregator = NullAggregator()
    else:
        raise NotImplementedError(f'Aggregate type {key} is not implemented yet')
    return aggregator


class CustomOutputSet:
    def __init__(self, **outputs):
        for key, value in outputs.items():
            setattr(self, key, value)


class DirectProperty(nn.Module):
    def __init__(self):
        super().__init__()

class DerivativeProperty(nn.Module):
    def __init__(self):
        super().__init__()
        self.create_graph = False  # Set by the model with train() or eval()

    def get_pairwise_force(self, outputs):
        if not hasattr(outputs, 'pairwise_force'):
            outputs.pairwise_force = -grad(
                outputs.energy, 
                outputs.disp, 
                grad_outputs=torch.ones_like(outputs.energy),
                create_graph=self.create_graph, 
                retain_graph=self.create_graph,
                )[0]
        return outputs.pairwise_force
    
class SecondDerivativeProperty(DerivativeProperty):
    def __init__(self):
        super().__init__()


class EnergyOutput(DirectProperty):
    '''
    Energy prediction

    Parameters:
        n_features (int): Number of features in the hidden layer.
        activation (nn.Module): Activation function.
    '''
    def __init__(self, n_features, activation):
        super().__init__()
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

class GradientForceOutput(DerivativeProperty):
    '''
    Gradient force prediction
    '''
    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        pairwise_force = self.get_pairwise_force(outputs)
        force = scatter(pairwise_force, outputs.edge_index[0], dim=0, reduce='sum', dim_size=outputs.atom_node.size(0)) - \
            scatter(pairwise_force, outputs.edge_index[1], dim=0, reduce='sum', dim_size=outputs.atom_node.size(0))
        # outputs.gradient_force = force
        return force
    
class DirectForceOutput(DirectProperty):
    '''
    Direct force prediction
    '''
    def __init__(self, n_features, activation):
        super().__init__()
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
    
class HessianOutput(SecondDerivativeProperty):
    '''
    Hessian prediction
    '''
    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        hessian = torch.vmap(
            lambda vec: grad(
                -outputs.gradient_force.flatten(), 
                outputs.disp, 
                grad_outputs=vec, 
                create_graph=self.create_graph,
                retain_graph=True,
                )[0],
            )(torch.eye(outputs.gradient_force.numel()))
        hessian = hessian.reshape(*outputs.gradient_force.shape, *outputs.disp.shape)
        hessian = scatter(hessian, outputs.edge_index[0], dim=2, reduce='sum', dim_size=outputs.atom_node.size(0)) - \
            scatter(hessian, outputs.edge_index[1], dim=2, reduce='sum', dim_size=outputs.atom_node.size(0))
        # outputs.hessian = hessian
        return hessian
    
class VirialOutput(DerivativeProperty):
    '''
    Virial prediction
    '''
    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        pairwise_force = self.get_pairwise_force(outputs)
        virial = outputs.disp[:, :, None] * pairwise_force[:, None, :]
        virial = scatter(virial, outputs.edge_index[0], dim=0, reduce='sum', dim_size=outputs.atom_node.size(0)) + \
            scatter(virial, outputs.edge_index[1], dim=0, reduce='sum', dim_size=outputs.atom_node.size(0))
        virial = scatter(virial, outputs.batch, dim=0, reduce='sum')
        return virial
    

class SumAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, outputs):
        return scatter(output, outputs.batch, dim=0, reduce='sum').reshape(-1)

class NullAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, outputs):
        return output