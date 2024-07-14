import torch
from torch import nn

from newtonnet.layers.shells import ShellProvider
from newtonnet.layers.cutoff import get_cutoff_by_string
from newtonnet.layers.representations import get_representation_by_string
from newtonnet.models.output import get_output_by_string, FirstDerivativeProperty, SecondDerivativeProperty


class NewtonNet(nn.Module):
    '''
    Molecular Newtonian Message Passing

    Parameters:
        n_features (int): Number of features in the latent layer. Default: 128.
        n_basis (int): Number of radial basis functions for edge description. Default: 20.
        distance_network (nn.Module): The distance transformation function.
        n_interactions (int): Number of message passing layers. Default: 3.
        activation (nn.Module): Activation function. Default: nn.SiLU().
        infer_properties (list): The properties to predict. Default: [].
        scalers (nn.ModuleDict): The scalers for the atomic properties. Default: {}.
        train_scalers (bool): Whether the normalizers are trainable. Default: False.
        device (torch.device): The device to run the network. Default: torch.device('cpu').
    '''
    def __init__(
            self,
            n_features: int = 128,
            distance_network: nn.Module = None,
            n_interactions: int = 3,
            infer_properties: list = [],
            scalers: nn.ModuleDict = {},
            device: torch.device = torch.device('cpu'),
    ) -> None:

        super(NewtonNet, self).__init__()

        # embedding layer
        if distance_network is None:
            distance_network = nn.Sequential(
                get_cutoff_by_string(
                    settings['model'].get('cutoff_network', 'poly'), 
                    cutoff=5.0,
                    ),
                get_representation_by_string(
                    settings['model'].get('representation', 'bessel'), 
                    n_basis=20,
                    cutoff=5.0,
                    ),
                )
        z_max = max([scaler.z_max for scaler in scalers.values()])
        self.embedding_layer = EmbeddingNet(
            n_features=n_features,
            z_max=z_max,
            distance_network=distance_network,
            )

        # message passing
        if share_interactions:
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
        self.output_layers = nn.ModuleDict({})
        for key in infer_properties:
            normalizer = normalizers[key] if key in normalizers else None
            output_layer = get_output_by_string(key, normalizer, **output_kwargs)
            self.output_layers.update({key: output_layer})
            if isinstance(output_layer, FirstDerivativeProperty):
                self.embedding_layer.requires_dr = True
            if isinstance(output_layer, SecondDerivativeProperty):
                assert output_layer.dependent_property in self.output_layers.keys(), f'cannot find dependent property {output_layer.dependent_property}'
                self.output_layers[output_layer.dependent_property].requires_dr = True
        
        # device
        self.to(device)

    def forward(self, z, pos, batch):
        '''
        Network forward pass

        Parameters:
            z (torch.Tensor): The atomic numbers of the atoms in the molecule. Shape: (n_atoms, ).
            pos (torch.Tensor): The positions of the atoms in the molecule. Shape: (n_atoms, 3).
            batch (torch.Tensor): The batch of the atoms in the molecule. Shape: (n_atoms, ).

        Returns:
            outputs (dict): The outputs of the network.
        '''
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
        for key, output_layer in self.output_layers.items():
            output, output_normalized = output_layer(**outputs)
            outputs[key] = output
            outputs[key + '_normalized'] = output_normalized

        return outputs
        

class EmbeddingNet(nn.Module):
    '''
    Embedding layer of the network

    Parameters:
        n_features (int): Number of features in the hidden layer.
        z_max (int): Maximum atomic number.
        distance_network (nn.Module): The distance transformation function.
    '''
    def __init__(self, n_features, z_max, distance_network):

        super(EmbeddingNet, self).__init__()

        # atomic embedding
        self.n_features = n_features
        self.node_embedding = nn.Embedding(z_max + 1, n_features)

        # edge embedding
        self.norm = distance_network['scalednorm']
        self.cutoff = distance_network['cutoff']
        self.edge_embedding = distance_network['representation']

    def forward(self, z, pos, edge_index, batch):

        # initialize node representations
        atom_node = self.node_embedding(z)  # n_nodes, n_features
        force_node = torch.zeros(*pos.shape, n_features)  # n_nodes, 3, n_features
        disp_node = torch.zeros(*pos.shape, n_features)  # n_nodes, 3, n_features

        # recompute distances and distance vectors
        if self.requires_dr:
            pos.requires_grad_()
            disp = pos[edge_index[0]] - pos[edge_index[1]]  # n_edges, 3

        # initialize edge representations
        dist = self.norm(disp)  # n_edges, 1
        dist_edge = self.cutoff(dist) * self.edge_embedding(dist)  # n_edges, n_basis

        return atom_node, force_node, disp_node, dist_edge


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
