import torch
from torch import nn
from torch_geometric.utils import scatter

from newtonnet.layers.activations import get_activation_by_string
from newtonnet.layers.scalers import get_scaler_by_string
from newtonnet.layers.representations import EdgeEmbedding
from newtonnet.models.output import get_output_by_string, get_aggregator_by_string
from newtonnet.models.output import CustomOutputSet, DerivativeProperty


class NewtonNet(nn.Module):
    '''
    Molecular Newtonian Message Passing

    Parameters:
        cutoff (float): Cutoff radius for the edge embedding. Default: 5.0.
        n_features (int): Number of features in the latent layer. Default: 128.
        n_basis (int): Number of radial basis functions. Default: 20.
        n_interactions (int): Number of message passing layers. Default: 3.
        activation (str): Activation function. Default: 'swish'.
        layer_norm (bool): Whether to use layer normalization. Default: False.
        output_properties (list): The properties to predict. Default: [].
        representations (dict): The distance transformation functions.
    '''
    def __init__(
            self,
            cutoff: float = 5.0,
            n_features: int = 128,
            n_basis: int = 20,
            n_interactions: int = 3,
            activation: str = 'swish',
            layer_norm: bool = False,
            output_properties: list = [],
    ) -> None:

        super().__init__()
        activation = get_activation_by_string(activation)

        # embedding layer
        self.embedding_layers = EmbeddingNet(
            cutoff=cutoff,
            n_features=n_features,
            n_basis=n_basis,
            )

        # message passing
        self.interaction_layers = nn.ModuleList([
            InteractionNet(
                n_features=n_features,
                n_basis=n_basis,
                activation=activation,
                layer_norm=layer_norm,
                ) for _ in range(n_interactions)
            ])

        # final output layer
        self.output_properties = output_properties
        self.output_layers = nn.ModuleList()
        self.scalers = nn.ModuleList()
        self.aggregators = nn.ModuleList()
        for key in self.output_properties:
            output_layer = get_output_by_string(key, n_features, activation)
            self.output_layers.append(output_layer)
            if isinstance(output_layer, DerivativeProperty):
                self.embedding_layers.requires_dr = True
            scaler = get_scaler_by_string(key)
            self.scalers.append(scaler)
            aggregator = get_aggregator_by_string(key)
            self.aggregators.append(aggregator)


    # def forward(self, batch):
    def forward(self, z, pos, cell, batch):
        '''
        Network forward pass

        Parameters:
            batch: The input data.
                z (torch.Tensor): The atomic numbers of the atoms in the molecule. Shape: (n_nodes, ).
                disp (torch.Tensor): The displacement vectors of the atoms in the molecule. Shape: (n_edges, 3).
                edge_index (torch.Tensor): The edge index of the atoms in the molecule. Shape: (2, n_edges).
                batch (torch.Tensor): The batch of the atoms in the molecule. Shape: (n_nodes, ).

        Returns:
            outputs (dict): The outputs of the network.
        '''

        # initialize node and edge representations
        atom_node, force_node, dir_edge, dist_edge, edge_index, displacement = self.embedding_layers(z, pos, cell, batch)

        # compute interaction block and update atomic embeddings
        for interaction_layer in self.interaction_layers:
            atom_node, force_node = interaction_layer(atom_node, force_node, dir_edge, dist_edge, edge_index)

        # output net
        outputs = CustomOutputSet(z=z, pos=pos, atom_node=atom_node, force_node=force_node, edge_index=edge_index, cell=cell, displacement=displacement, batch=batch)
        for key, output_layer, scaler, aggregator in zip(self.output_properties, self.output_layers, self.scalers, self.aggregators):
            output = output_layer(outputs)
            output = scaler(output, outputs)
            output = aggregator(output, outputs)
            setattr(outputs, key, output)

        return outputs
    
    def train(self, mode=True):
        '''
        Set the network to training mode
        '''
        super().train(mode)
        for output_layer in self.output_layers:
            if isinstance(output_layer, DerivativeProperty):
                output_layer.create_graph = mode


class EmbeddingNet(nn.Module):
    '''
    Embedding layer of the network

    Parameters:
        cutoff (float): Cutoff radius for the edge embedding.
        n_features (int): Number of features in the hidden layer.
        n_basis (int): Number of radial basis functions.
    '''
    def __init__(self, cutoff, n_features, n_basis):

        super().__init__()

        # atomic embedding
        self.n_features = n_features
        self.node_embedding = nn.Embedding(118 + 1, n_features, padding_idx=0)

        # edge embedding
        self.edge_embedding = EdgeEmbedding(cutoff=cutoff, n_basis=n_basis)

        # requires dr
        self.requires_dr = False

    def forward(self, z, pos, cell, batch):

        # initialize node representations
        atom_node = self.node_embedding(z)  # n_nodes, n_features
        force_node = torch.zeros(z.size(0), 3, self.n_features, dtype=pos.dtype, device=pos.device)  # n_nodes, 3, n_features

        # prepare for gradient calculation
        displacement = torch.zeros_like(cell)
        displacement[:, 0, 0] = 1.0
        displacement[:, 1, 1] = 1.0
        displacement[:, 2, 2] = 1.0
        if self.requires_dr:
            pos.requires_grad = True
            displacement.requires_grad = True
        symmetric_displacement = (displacement + displacement.transpose(-1, -2)) / 2
        pos_displaced = torch.bmm(pos.unsqueeze(1), symmetric_displacement[batch]).squeeze(1)  # n_nodes, 3
        cell_displaced = torch.bmm(cell, symmetric_displacement).squeeze(1)  # n_nodes, 3, 3


        # initialize edge representations
        dist_edge, dir_edge, edge_index = self.edge_embedding(pos_displaced, cell_displaced, batch)  # n_edges, n_basis; n_edges, 3; 2, n_edges

        return atom_node, force_node, dir_edge, dist_edge, edge_index, displacement
        # return atom_node, force_node, disp_node, dir_edge, cutoff_edge, rbf_edge


class InteractionNet(nn.Module):
    '''
    Message passing layer of the network

    Parameters:
        n_features (int): Number of features in the hidden layer.
        n_basis (int): Number of radial basis functions.
        activation (nn.Module): Activation function.
        layer_norm (bool): Whether to use layer normalization.
    '''
    def __init__(self, n_features, n_basis, activation, layer_norm):
        super().__init__()

        self.n_features = n_features

        # invariant message passing
        self.message_nodepart = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )
        self.message_edgepart = nn.Linear(n_basis, n_features, bias=False)
        
        self.equiv_message1 = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            activation,
            nn.Linear(n_features, n_features, bias=False),
        )
        self.equiv_message2 = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            activation,
            nn.Linear(n_features, n_features, bias=False),
        )
        
        self.equiv_update = nn.Linear(n_features, n_features, bias=False)

        # layer norm
        if layer_norm:
            self.layer_norm = nn.LayerNorm(n_features)
        else:
            self.layer_norm = None
    
    def forward(self, atom_node, force_node, dir_edge, dist_edge, edge_index):
        # a
        message_nodepart = self.message_nodepart(atom_node)    # n_nodes, n_features
        message_edgepart = self.message_edgepart(dist_edge)    # n_edges, n_features
        message = message_edgepart * message_nodepart[edge_index[0]] * message_nodepart[edge_index[1]]    # n_edges, n_features

        inv_message1 = message    # n_nodes, n_features
        inv_update1 = scatter(inv_message1, edge_index[0], dim=0, dim_size=atom_node.size(0))    # n_nodes, n_features
        atom_node = atom_node + inv_update1    # n_nodes, n_features

        # f
        equiv_message1_invpart = self.equiv_message1(message).unsqueeze(1)    # n_edges, 1, n_features
        equiv_message1_equivpart = dir_edge.unsqueeze(2)    # n_edges, 3, 1
        equiv_message1 = equiv_message1_invpart * equiv_message1_equivpart    # n_edges, 3, n_features

        equiv_message2_invpart = self.equiv_message2(message).unsqueeze(1)    # n_edges, 1, n_features
        equiv_message2_equivpart = force_node[edge_index[1]]    # n_edges, 3, n_features
        equiv_message2 = equiv_message2_invpart * equiv_message2_equivpart    # n_edges, 3, n_features

        force_update = scatter(equiv_message1 + equiv_message2, edge_index[0], dim=0, dim_size=force_node.size(0))    # n_nodes, 3, n_features
        force_node = force_node + force_update    # n_nodes, 3, n_features

        # update energy
        inv_update2 = torch.sum(force_node * self.equiv_update(force_node), dim=1)    # n_nodes, n_features
        atom_node = atom_node + inv_update2

        # layer norm
        if self.layer_norm is not None:
            atom_node = self.layer_norm(atom_node)

        return atom_node, force_node
