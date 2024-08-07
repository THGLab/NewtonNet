import torch
from torch import nn
from torch.autograd import grad
from torch_geometric.utils import scatter


def get_output_by_string(key, **kwargs):
    if key == 'energy':
        output_layer = EnergyOutput(**kwargs)
    elif key == 'forces':
        output_layer = GradientForceOutput(**kwargs)
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
    def __init__(self, n_features=None, activation=None, scaler=None, **kwargs):
        super(EnergyOutput, self).__init__()
        self.key = 'energy'
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, 1),
            )
        self.scaler = scaler

    def forward(self, inputs):
        output = self.layers(inputs.atom_node)
        # output = self.scaler(output, inputs.z)
        output = scatter(output, inputs.batch, dim=0, reduce='sum').reshape(-1)
        output = output * 5.992277830325989 + -406274.63784969115
        return output

class GradientForceOutput(FirstDerivativeProperty):
    '''
    Gradient force prediction
    '''
    def __init__(self, **kwargs):
        super(GradientForceOutput, self).__init__()
        self.key = 'force'
        self.requires_dr = False

    def forward(self, inputs):
        output = grad(
            inputs.energy.sum(), 
            inputs.pos, 
            # grad_outputs=torch.ones_like(inputs.pos), 
            create_graph=self.requires_dr, 
            retain_graph=True,
            )[0]
        return -output


# class InvariantNodeProperty(nn.Module):
#     '''
#     Invariant node property prediction

#     Parameters:
#         n_features (int): Number of features in the hidden layer.
#         activation (nn.Module): Activation function.
#         dropout (float): Dropout rate.
#         scaler (nn.Module): The normalizer for the atomic properties.
#         train_normalizer (bool): Whether the normalizers are trainable.
#     '''
#     def __init__(self, key=None, n_features=None, activation=None, scaler=None, **kwargs):
#         super(InvariantNodeProperty, self).__init__()
#         self.key = key
#         self.output = nn.Sequential(
#             nn.Linear(n_features, n_features),
#             activation,
#             nn.Linear(n_features, n_features),
#             activation,
#             nn.Linear(n_features, 1),
#             )
#         self.scaler = scaler

#     def forward(self, outputs):
#         output = self.output(outputs.atom_node)
#         output = self.scaler(output, outputs.z)
#         return output


# class InvariantGraphProperty(nn.Module):
#     '''
#     Invariant graph property prediction

#     Parameters:
#         n_features (int): Number of features in the hidden layer.
#         activation (nn.Module): Activation function.
#         dropout (float): Dropout rate.
#         aggregration (str): Aggregration method.
#         normalizer (nn.Module): The normalizer for the atomic properties.
#         train_normalizer (bool): Whether the normalizers are trainable.
#     '''
#     def __init__(self, key=None, aggregration=None, **kwargs):
#         super(InvariantGraphProperty, self).__init__()
#         self.key = key
#         self.output = InvariantNodeProperty(**kwargs)
#         self.aggregration = aggregration

#     def forward(self, atom_node, z, batch, **inputs):
#         output = self.output(atom_node, z)
#         output = scatter(output, batch, dim=0, reduce=self.aggregration)
#         return output
    

# class FirstDerivativeProperty(nn.Module):
#     '''
#     First derivative property prediction

#     Parameters:
#         dependent_property (str): The dependent property.
#         independent_property (str): The independent property.
#         negate (bool): Whether to negate the output.
#     '''
#     def __init__(self, key=None, dependent_property=None, independent_property=None, negate=None, **kwargs):
#         super(FirstDerivativeProperty, self).__init__()
#         self.key = key
#         self.dependent_property = dependent_property
#         self.independent_property = independent_property
#         self.negate = negate
        
#         self.requires_dr = False

#     def forward(self, **inputs):
#         dependent_property = inputs.__getattribute__(self.dependent_property)
#         independent_property = inputs.__getattribute__(self.independent_property)
#         grad_outputs = torch.ones_like(dependent_property)
#         output = grad(
#             dependent_property, 
#             independent_property, 
#             grad_outputs=grad_outputs, 
#             create_graph=self.requires_dr, 
#             retain_graph=True,
#             )[0]
#         if self.negate:
#             return -output
#         else:
#             return output

# class SecondDerivativeProperty(nn.Module):
#     '''
#     Second derivative property prediction

#     Parameters:
#         dependent_property (str): The dependent property.
#         independent_property (str): The independent property.
#         negate (bool): Whether to negate the output.
#         normalizer (nn.Module): The normalizer for the atomic properties.
#         train_normalizer (bool): Whether the normalizers are trainable.
#     '''
#     def __init__(self, key=None, dependent_property=None, independent_property=None, negate=None, **kwargs):
#         super(SecondDerivativeProperty, self).__init__()
#         self.key = key
#         self.dependent_property = dependent_property
#         self.independent_property = independent_property
#         self.negate = negate

#     def forward(self, **inputs):
#         dependent_property = inputs[self.dependent_property]
#         independent_property = inputs[self.independent_property]
#         n_data, n_atoms, n_dim = independent_property.shape
#         grad_outputs = torch.eye(n_atoms * n_dim, device=independent_property.device)    # n_atoms * n_dim, n_atoms * n_dim
#         grad_outputs = grad_outputs.view((n_atoms * n_dim, 1, n_atoms, n_dim)).repeat(1, n_data, 1, 1)    # n_atoms * n_dim, n_data, n_atoms, n_dim
#         output = torch.vmap(
#             lambda V: grad(
#                 dependent_property, 
#                 independent_property, 
#                 grad_outputs=V, 
#                 create_graph=False, 
#                 retain_graph=True,
#                 )[0]
#             )(grad_outputs)
#         output = output.view((n_atoms, n_dim, n_data, n_atoms, n_dim)).permute((2, 0, 1, 3, 4))    # n_data, n_atoms, grad_outputs, n_atoms, grad_outputs
#         # output = torch.zeros(n_data, n_atoms, n_dim, n_atoms, n_dim, device=independent_property.device)
#         # for atom in range(n_atoms):
#         #     for dim in range(n_dim):
#         #         output[:, atom, dim, :, :] = grad(
#         #             dependent_property[:, atom, dim], 
#         #             independent_property, 
#         #             grad_outputs=torch.ones(n_data, device=independent_property.device), 
#         #             create_graph=False, 
#         #             retain_graph=True,
#         #             )[0]

#         output_normalized = self.normalizer.forward(output, atomic_numbers)
#         if self.negate:
#             return -output, -output_normalized
#         else:
#             return output, output_normalized