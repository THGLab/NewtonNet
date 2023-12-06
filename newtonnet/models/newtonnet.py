import copy

import torch
from torch import nn
from torch.autograd import grad

from newtonnet.layers.shells import ShellProvider
from newtonnet.layers.cutoff import PolynomialCutoff
from newtonnet.layers.representations import RadialBesselLayer
from newtonnet.layers.aggregation import get_aggregation_by_string


def get_output_by_string(key, normalizer, kwargs):
    if key == 'energy':
        output_layer = InvariantGraphProperty(
            aggregration='sum',
            normalizer=normalizer,
            **kwargs,
            )
    elif key == 'forces':
        output_layer = DerivativeProperty(
            dependent_property='energy',
            independent_property='positions',
            negate=True,
            normalizer=normalizer,
            **kwargs,
            )
    elif key == 'hessian':
        # output_layer = DerivativeProperty(
        #     dependent_property='forces',
        #     independent_property='positions',
        #     normalizer=normalizers['hessian'],
        #     train_normalizer=kwargs['train_normalizer'],
        #     )
        raise NotImplementedError('Hessian is not implemented yet')
    else:
        raise NotImplementedError(f'Output type {key} is not implemented yet')
    return output_layer


class NewtonNet(nn.Module):
    '''
    Molecular Newtonian Message Passing

    Parameters:
        n_basis (int): Number of radial basis functions. Default: 20.
        n_features (int): Number of features in the latent layer. Default: 128.
        activation (nn.Module): Activation function. Default: nn.SiLU().
        n_interactions (int): Number of message passing layers. Default: 3.
        dropout (float): Dropout rate. Default: 0.0.
        embedded_atomic_numbers (torch.Tensor): The atomic numbers of the atoms in the molecule.
        shell (nn.Module): The shell module. Default: ShellProvider().
        cutoff_network (nn.Module): The cutoff layer. Default: PolynomialCutoff(shell.cutoff).
        normalizers (nn.ModuleDict): The normalizers for the atomic properties. Default: {}.
        train_normalizer (bool): Whether the normalizers are trainable. Default: False.
        requires_dr (bool): Whether to create gradient graph of the positions. Default: False.
        device (torch.device): The device to run the network. Default: None.
        share_layers (bool): Whether to share the weights of the message passing layers. Default: False.
        double_update_node (bool): Whether to update the invariant node twice in each message passing layer. Default: False.
        layer_norm (bool): Whether to use layer normalization after message passing. Default: False.
        predictions (list): The properties to predict. Default: [].
    '''
    def __init__(
            self,
            n_basis: int = 20,
            n_features: int = 128,
            activation: nn.Module = nn.SiLU(),
            n_interactions: int = 3,
            dropout: float = 0.0,
            embedded_atomic_numbers: torch.Tensor = torch.tensor([1, 6, 7, 8]),
            shell: nn.Module = None,
            cutoff_network: nn.Module = None,
            normalizers: nn.ModuleDict = {},
            train_normalizer: bool = False,
            device: torch.device = torch.device('cpu'),
            share_layers: bool = False,
            double_update_node: bool = False,
            layer_norm: bool = False,
            predictions: list = [],
    ) -> None:

        super(NewtonNet, self).__init__()

        # embedding layer
        self.requires_dr = False
        if shell is None:
            print('use default shell')
            shell = ShellProvider()
        if cutoff_network is None:
            print('use default cutoff network')
            cutoff_network = PolynomialCutoff(shell.cutoff)
        self.embedding_layer = EmbeddingNet(
            n_features=n_features,
            embedded_atomic_numbers=embedded_atomic_numbers,
            n_basis=n_basis,
            shell=shell,
            device=device,
            )

        # message passing
        if share_layers:
            # use the same message instance (hence the same weights)
            self.interaction_layers = nn.ModuleList([
                InteractionNet(
                    n_features=n_features,
                    n_basis=n_basis,
                    activation=activation,
                    shell=shell,
                    cutoff_network=cutoff_network,
                    double_update_node=double_update_node,
                    layer_norm=layer_norm,
                    )
                ] * n_interactions)
        else:
            # use one SchNetInteraction instance for each interaction
            self.interaction_layers = nn.ModuleList([
                InteractionNet(
                    n_features=n_features,
                    n_basis=n_basis,
                    activation=activation,
                    shell=shell,
                    cutoff_network=cutoff_network,
                    double_update_node=double_update_node,
                    layer_norm=layer_norm,
                    ) for _ in range(n_interactions)
                ])

        # final output layer
        output_kwargs = {
            'n_features': n_features,
            'activation': activation,
            'dropout': dropout,
            'train_normalizer': train_normalizer,
            }
        self.output_layers = nn.ModuleDict({
            property: get_output_by_string(property, normalizers[property], kwargs=output_kwargs) for property in predictions
            })
        for output_layer in self.output_layers.values():
            if isinstance(output_layer, DerivativeProperty):
                self.requires_dr = True
                self.embedding_layer.requires_dr = True
                break

    def forward(
            self,
            atomic_numbers: torch.Tensor,
            positions: torch.Tensor,
            atom_mask: torch.Tensor,
            neighbor_mask: torch.Tensor,
            distances: torch.Tensor,
            distance_vectors: torch.Tensor,
            ):
        # initialize node and edge representations
        invariant_node, equivariant_node_F, equivariant_node_f, equivariant_node_dr, invariant_edge, distances, distance_vectors = \
            self.embedding_layer(atomic_numbers, positions, neighbor_mask, distances, distance_vectors)

        # compute interaction block and update atomic embeddings
        for interaction_layer in self.interaction_layers:
            # messages
            invariant_node, equivariant_node_F, equivariant_node_f, equivariant_node_dr = \
                interaction_layer(invariant_node, equivariant_node_F, equivariant_node_f, equivariant_node_dr, invariant_edge, neighbor_mask, distances, distance_vectors)

        # output net
        outputs = {
            'invariant_node': invariant_node,
            'equivariant_node_F': equivariant_node_F,
            'equivariant_node_f': equivariant_node_f,
            'equivariant_node_dr': equivariant_node_dr,
            'atomic_numbers': atomic_numbers,
            'positions': positions,
            'atom_mask': atom_mask,
            }
        for property, output_layer in self.output_layers.items():
            output, output_normalized = output_layer(**outputs)
            outputs[property] = output
            outputs[property + '_normalized'] = output_normalized

        return outputs
        

class EmbeddingNet(nn.Module):
    '''
    Embedding layer of the network

    Parameters:
        n_features (int): Number of features in the hidden layer.
        embedded_atomic_numbers (torch.Tensor): The atomic numbers of the atoms in the molecule.
        n_basis (int): Number of radial basis functions.
        shell (nn.Module): The shell module.
        device (torch.device): The device to run the network.
    '''
    def __init__(self, n_features, embedded_atomic_numbers, n_basis, shell, device):

        super(EmbeddingNet, self).__init__()

        # atomic embedding
        self.n_features = n_features
        self.node_embedding = nn.Embedding(embedded_atomic_numbers.max() + 1, n_features, padding_idx=0, device=device)
        for z in range(1, embedded_atomic_numbers.max() + 1):
            if z not in embedded_atomic_numbers:
                self.node_embedding.weight.data[z] = torch.nan
        self.requires_dr = False

        # edge embedding
        self.shell = shell
        self.edge_embedding = RadialBesselLayer(n_basis, shell.cutoff, device=device)
        self.epsilon = 1.0e-8

    def forward(self, atomic_numbers, positions, neighbor_mask, distances, distance_vectors):

        # initialize node representations
        invariant_node = self.node_embedding(atomic_numbers)  # batch_size, n_atoms, n_features
        batch_size, n_atoms, n_features = invariant_node.shape
        equivariant_node_F = torch.zeros((batch_size, n_atoms, 3), device=positions.device)  # batch_size, n_atoms, 3
        equivariant_node_f = torch.zeros((batch_size, n_atoms, 3, n_features), device=positions.device)  # batch_size, n_atoms, 3, n_features
        equivariant_node_dr = torch.zeros((batch_size, n_atoms, 3, n_features), device=positions.device)  # batch_size, n_atoms, 3, n_features

        # recompute distances and distance vectors
        if self.requires_dr:
            positions.requires_grad_()
            distances, distance_vectors, neighbor_mask = self.shell(positions, neighbor_mask)
        distance_vectors = distance_vectors / (distances.unsqueeze(-1) + self.epsilon)

        # initialize edge representations
        invariant_edge = self.edge_embedding(distances)


        return invariant_node, equivariant_node_F, equivariant_node_f, equivariant_node_dr, invariant_edge, distances, distance_vectors


class InteractionNet(nn.Module):
    '''
    Message passing layer of the network

    Parameters:
        n_features (int): Number of features in the hidden layer.
        n_basis (int): Number of radial basis functions.
        activation (nn.Module): Activation function.
        shell (nn.Module): The shell module to handle neighbors.
        cutoff_network (nn.Module): The cutoff layer.
        double_update_node (bool): Whether to update the invariant node twice in each message passing layer.
        layer_norm (bool): Whether to use layer normalization after message passing.
    '''
    def __init__(self, n_features, n_basis, activation, shell, cutoff_network, double_update_node, layer_norm):
        super(InteractionNet, self).__init__()

        self.n_features = n_features

        # invariant message passing
        self.invariant_message_edge = nn.Linear(n_basis, n_features)
        nn.init.xavier_uniform_(self.invariant_message_edge.weight)
        nn.init.zeros_(self.invariant_message_edge.bias)
        self.invariant_message_node = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )

        # cutoff layer used in interaction block
        self.shell = shell
        self.cutoff_network = cutoff_network

        # equivariant message passing
        self.equivariant_message_coefficient = nn.Linear(n_features, 1, bias=False)
        nn.init.xavier_uniform_(self.equivariant_message_coefficient.weight)
        self.equivariant_message_feature = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )
        self.equivariant_selfupdate_coefficient = nn.Sequential(
            nn.Linear(n_features, n_features),
        )
        self.equivariant_selfupdate_coefficient = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )
        self.equivariant_message_edge = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            activation,
            nn.Linear(n_features, n_features, bias=False),
        )

        self.invariant_selfupdate_coefficient = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )

        self.double_update_node = double_update_node

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm = nn.LayerNorm(n_features)
    
    def forward(self, invariant_node, equivariant_node_F, equivariant_node_f, equivariant_node_dr, invariant_edge, neighbor_mask, distances, distance_vectors):
        # map decomposed distances
        invariant_message_edge = self.invariant_message_edge(invariant_edge)    # batch_size, n_atoms, n_neighbors, n_features

        # cutoff
        invariant_message_edge = invariant_message_edge * self.cutoff_network(distances).unsqueeze(-1)    # batch_size, n_atoms, n_neighbors, n_features

        # map atomic features
        invariant_message_node = self.invariant_message_node(invariant_node)    # batch_size, n_atoms, n_features

        # # symmetric feature multiplication
        invariant_message_node_i, invariant_message_node_f = self.shell.gather_neighbors(invariant_message_node, neighbor_mask)    # batch_size, n_atoms, n_neighbors, n_features
        invariant_message = invariant_message_edge * invariant_message_node_i * invariant_message_node_f    # batch_size, n_atoms, n_neighbors, n_features

        # update a with invariance
        if self.double_update_node:
            invariant_update = (invariant_message * neighbor_mask.unsqueeze(-1)).sum(dim=2)    # batch_size, n_atoms, n_features
            invariant_node = invariant_node + invariant_update    # batch_size, n_atoms, n_features

        # F
        equivariant_message_F = self.equivariant_message_coefficient(invariant_message) * distance_vectors  # batch_size, n_atoms, n_neighbors, 3
        equivariant_update_F = (equivariant_message_F * neighbor_mask.unsqueeze(-1)).sum(dim=2)  # batch_size, n_atoms, 3
        equivariant_node_F = equivariant_node_F + equivariant_update_F

        # f
        equivariant_message_f = self.equivariant_message_feature(invariant_message).unsqueeze(-2) * equivariant_message_F.unsqueeze(-1)  # batch_size, n_atoms, n_neighbors, 3, n_features
        equivariant_update_f = (equivariant_message_f * neighbor_mask.unsqueeze(-1).unsqueeze(-2)).sum(dim=2)  # batch_size, n_atoms, 3, n_features
        equivariant_node_f = equivariant_node_f + equivariant_update_f
        
        # dr
        equivariant_message_dr_edge = self.equivariant_message_edge(invariant_message).unsqueeze(-2)    # batch_size, n_atoms, n_neighbors, 3, n_features
        equivariant_message_dr_node_i, equivariant_message_dr_node_f = self.shell.gather_neighbors(equivariant_node_dr, neighbor_mask)    # batch_size, n_atoms, 3, n_features
        equivariant_message_dr = equivariant_message_dr_edge * equivariant_message_dr_node_f    # batch_size, n_atoms, n_neighbors, 3, n_features
        equivariant_update_dr = (equivariant_message_dr * neighbor_mask.unsqueeze(-1).unsqueeze(-2)).sum(dim=2)  # batch_size, n_atoms, 3, n_features
        equivariant_node_dr = equivariant_node_dr + equivariant_update_dr    # batch_size, n_atoms, 3, n_features

        equivariant_update_dr = self.equivariant_selfupdate_coefficient(invariant_node).unsqueeze(-2) * equivariant_update_f    # batch_size, n_atoms, 3, n_features
        equivariant_node_dr = equivariant_node_dr + equivariant_update_dr    # batch_size, n_atoms, 3, n_features

        # update energy
        invariant_update = -self.invariant_selfupdate_coefficient(invariant_node) * torch.sum(equivariant_node_f * equivariant_node_dr, dim=-2)  # batch_size, n_atoms, n_features
        invariant_node = invariant_node + invariant_update

        # layer norm
        if self.layer_norm:
            invariant_node = self.norm(invariant_node)

        return invariant_node, equivariant_node_F, equivariant_node_f, equivariant_node_dr


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
        if train_normalizer:
            self.normalizer = copy.deepcopy(normalizer).requires_grad_(True)
        else:
            self.normalizer = normalizer

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
    def __init__(self, n_features, activation, dropout, aggregration, normalizer, train_normalizer, **kwargs):

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
        
        if train_normalizer:
            self.normalizer = copy.deepcopy(normalizer).requires_grad_(True)
        else:
            self.normalizer = normalizer

    def forward(self, invariant_node, atomic_numbers, atom_mask, **inputs):
        output_normalized = self.invariant_node_prediction(invariant_node) * atom_mask.unsqueeze(-1)
        output_normalized = self.aggregration(output_normalized, dim=1)
        output = self.normalizer.reverse(output_normalized, atomic_numbers)
        return output, output_normalized
    

class DerivativeProperty(nn.Module):
    '''
    First derivative property prediction

    Parameters:
        dependent_property (str): The dependent property.
        independent_property (str): The independent property.
        negate (bool): Whether to negate the output.
        normalizer (nn.Module): The normalizer for the atomic properties.
        train_normalizer (bool): Whether the normalizers are trainable.
    '''
    def __init__(self, dependent_property, independent_property, negate, normalizer, train_normalizer, **kwargs):

        super(DerivativeProperty, self).__init__()
        self.dependent_property = dependent_property
        self.independent_property = independent_property
        self.negate = negate
        if train_normalizer:
            self.normalizer = copy.deepcopy(normalizer).requires_grad_(True)
        else:
            self.normalizer = normalizer

    def forward(self, atomic_numbers, **inputs):
        dependent_property = inputs[self.dependent_property]
        independent_property = inputs[self.independent_property]
        output = grad(
            dependent_property, 
            independent_property, 
            grad_outputs=torch.ones_like(dependent_property), 
            create_graph=True, 
            retain_graph=True,
            )[0]
        output_normalized = self.normalizer.forward(output, atomic_numbers)
        if self.negate:
            output = -output
            output_normalized = -output_normalized
        return output, output_normalized
