import torch
from torch import nn
from torch.autograd import grad

from newtonnet.layers.aggregation import get_aggregation_by_string
from newtonnet.layers.scalers import NullNormalizer


def get_output_by_string(key, normalizer=None, **kwargs):
    if normalizer is None:
        print(f'no normalizer is provided for {key}, use null normalizer instead')
        normalizer = NullNormalizer()
    if key == 'energy':
        output_layer = InvariantGraphProperty(
            aggregration='sum',
            normalizer=normalizer,
            **kwargs,
            )
    elif key == 'forces':
        output_layer = FirstDerivativeProperty(
            dependent_property='energy',
            independent_property='positions',
            negate=True,
            normalizer=normalizer,
            **kwargs,
            )
    elif key == 'hessian':
        output_layer = SecondDerivativeProperty(
            dependent_property='forces',
            independent_property='positions',
            negate=True,
            normalizer=normalizer,
            **kwargs,
            )
    else:
        raise NotImplementedError(f'Output type {key} is not implemented yet')
    return output_layer


class InvariantNodeProperty(nn.Module):
    '''
    Invariant node property prediction

    Parameters:
        n_features (int): Number of features in the hidden layer.
        activation (nn.Module): Activation function.
        dropout (float): Dropout rate.
        normalizer (nn.Module): The normalizer for the atomic properties.
        train_normalizer (bool): Whether the normalizers are trainable.
    '''
    def __init__(self, n_features, activation, dropout, normalizer, train_normalizer, **kwargs):

        super(InvariantNodeProperty, self).__init__()
        if dropout > 0.0:
            self.invariant_node_prediction = nn.Sequential(
                nn.Linear(n_features, 128),
                activation,
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                activation,
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                )
        else:
            self.invariant_node_prediction = nn.Sequential(
                nn.Linear(n_features, 128),
                activation,
                nn.Linear(128, 64),
                activation,
                nn.Linear(64, 1),
                )
        
        self.normalizer = normalizer
        if train_normalizer:
            self.normalizer.requires_grad_(True)

    def forward(self, invariant_node, atom_mask, **inputs):
        output_normalized = self.invariant_node_prediction(invariant_node) * atom_mask.unsqueeze(-1)
        output = self.normalizer.reverse(output_normalized)
        return output, output_normalized


class InvariantGraphProperty(nn.Module):
    '''
    Invariant graph property prediction

    Parameters:
        n_features (int): Number of features in the hidden layer.
        activation (nn.Module): Activation function.
        dropout (float): Dropout rate.
        aggregration (str): Aggregration method.
        normalizer (nn.Module): The normalizer for the atomic properties.
        train_normalizer (bool): Whether the normalizers are trainable.
    '''
    def __init__(self, n_features, activation, dropout, aggregration, normalizer, train_normalizer=False, **kwargs):

        super(InvariantGraphProperty, self).__init__()
        if dropout > 0.0:
            self.invariant_node_prediction = nn.Sequential(
                nn.Linear(n_features, n_features),
                activation,
                nn.Dropout(dropout),
                nn.Linear(n_features, n_features),
                activation,
                nn.Dropout(dropout),
                nn.Linear(n_features, 1),
                )
        else:
            self.invariant_node_prediction = nn.Sequential(
                nn.Linear(n_features, 128),
                activation,
                nn.Linear(128, 64),
                activation,
                nn.Linear(64, 1),
                )

        self.aggregration = get_aggregation_by_string(aggregration)
        
        self.normalizer = normalizer
        if train_normalizer:
            self.normalizer.requires_grad_(True)

    def forward(self, invariant_node, atomic_numbers, atom_mask, **inputs):
        output_normalized = self.invariant_node_prediction(invariant_node) * atom_mask.unsqueeze(-1)
        output_normalized = self.aggregration(output_normalized, dim=1)
        output = self.normalizer.reverse(output_normalized, atomic_numbers)
        return output, output_normalized
    

class FirstDerivativeProperty(nn.Module):
    '''
    First derivative property prediction

    Parameters:
        dependent_property (str): The dependent property.
        independent_property (str): The independent property.
        negate (bool): Whether to negate the output.
        normalizer (nn.Module): The normalizer for the atomic properties.
        train_normalizer (bool): Whether the normalizers are trainable.
    '''
    def __init__(self, dependent_property, independent_property, negate, normalizer, train_normalizer=False, **kwargs):

        super(FirstDerivativeProperty, self).__init__()
        self.dependent_property = dependent_property
        self.independent_property = independent_property
        self.negate = negate

        self.normalizer = normalizer
        if train_normalizer:
            self.normalizer.requires_grad_(True)
        self.requires_dr = False

    def forward(self, atomic_numbers, **inputs):
        dependent_property = inputs[self.dependent_property]
        independent_property = inputs[self.independent_property]
        grad_outputs = torch.ones_like(dependent_property)
        output = grad(
            dependent_property, 
            independent_property, 
            grad_outputs=grad_outputs, 
            create_graph=self.requires_dr, 
            retain_graph=True,
            )[0]
        output_normalized = self.normalizer.forward(output, atomic_numbers)
        if self.negate:
            return -output, -output_normalized
        else:
            return output, output_normalized

class SecondDerivativeProperty(nn.Module):
    '''
    Second derivative property prediction

    Parameters:
        dependent_property (str): The dependent property.
        independent_property (str): The independent property.
        negate (bool): Whether to negate the output.
        normalizer (nn.Module): The normalizer for the atomic properties.
        train_normalizer (bool): Whether the normalizers are trainable.
    '''
    def __init__(self, dependent_property, independent_property, negate, normalizer, train_normalizer=False, **kwargs):

        super(SecondDerivativeProperty, self).__init__()
        self.dependent_property = dependent_property
        self.independent_property = independent_property
        self.negate = negate

        self.normalizer = normalizer
        if train_normalizer:
            self.normalizer.requires_grad_(True)

    def forward(self, atomic_numbers, **inputs):
        dependent_property = inputs[self.dependent_property]
        independent_property = inputs[self.independent_property]
        n_data, n_atoms, n_dim = independent_property.shape
        grad_outputs = torch.eye(n_atoms * n_dim, device=independent_property.device)    # n_atoms * n_dim, n_atoms * n_dim
        grad_outputs = grad_outputs.view((n_atoms * n_dim, 1, n_atoms, n_dim)).repeat(1, n_data, 1, 1)    # n_atoms * n_dim, n_data, n_atoms, n_dim
        output = torch.vmap(
            lambda V: grad(
                dependent_property, 
                independent_property, 
                grad_outputs=V, 
                create_graph=False, 
                retain_graph=True,
                )[0]
            )(grad_outputs)
        output = output.view((n_atoms, n_dim, n_data, n_atoms, n_dim)).permute((2, 0, 1, 3, 4))    # n_data, n_atoms, grad_outputs, n_atoms, grad_outputs
        # output = torch.zeros(n_data, n_atoms, n_dim, n_atoms, n_dim, device=independent_property.device)
        # for atom in range(n_atoms):
        #     for dim in range(n_dim):
        #         output[:, atom, dim, :, :] = grad(
        #             dependent_property[:, atom, dim], 
        #             independent_property, 
        #             grad_outputs=torch.ones(n_data, device=independent_property.device), 
        #             create_graph=False, 
        #             retain_graph=True,
        #             )[0]

        output_normalized = self.normalizer.forward(output, atomic_numbers)
        if self.negate:
            return -output, -output_normalized
        else:
            return output, output_normalized