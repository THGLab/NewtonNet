import torch
from torch import nn
from torch_geometric.utils import scatter

from newtonnet.layers.activations import get_activation_by_string
from newtonnet.layers.scalers import get_scaler_by_string
from newtonnet.models.output import get_output_by_string, get_aggregator_by_string
from newtonnet.models.output import CustomOutputSet, FirstDerivativeProperty, SecondDerivativeProperty


class NewtonNet(nn.Module):
    '''
    Molecular Newtonian Message Passing

    Parameters:
        n_features (int): Number of features in the latent layer. Default: 128.
        n_interactions (int): Number of message passing layers. Default: 3.
        activation (str): Activation function. Default: 'swish'.
        layer_norm (bool): Whether to use layer normalization. Default: False.
        infer_properties (list): The properties to predict. Default: [].
        representations (dict): The distance transformation functions.
    '''
    def __init__(
            self,
            n_features: int = 128,
            n_interactions: int = 3,
            activation: str = 'swish',
            layer_norm: bool = False,
            infer_properties: list = [],
            representations: nn.Module = None,
    ) -> None:

        super(NewtonNet, self).__init__()
        activation = get_activation_by_string(activation)

        # embedding layer
        self.embedding_layer = EmbeddingNet(
            n_features=n_features,
            representations=representations,
            )

        # message passing
        self.interaction_layers = nn.ModuleList([
            InteractionNet(
                n_features=n_features,
                n_basis=self.embedding_layer.n_basis,
                activation=activation,
                layer_norm=layer_norm,
                ) for _ in range(n_interactions)
            ])

        # final output layer
        self.infer_properties = infer_properties
        self.output_layers = nn.ModuleList()
        self.scalers = nn.ModuleList()
        self.aggregators = nn.ModuleList()
        for key in self.infer_properties:
            output_layer = get_output_by_string(key, n_features, activation)
            self.output_layers.append(output_layer)
            if isinstance(output_layer, FirstDerivativeProperty):
                self.embedding_layer.requires_dr = True
            # if isinstance(output_layer, SecondDerivativeProperty):
            #     dependent_property = output_layer.dependent_property
            #     assert dependent_property in self.output_layers.keys(), f'cannot find dependent property {dependent_property}'
            #     self.output_layers[dependent_property].requires_dr = True
            scaler = get_scaler_by_string(key)
            self.scalers.append(scaler)
            aggregator = get_aggregator_by_string(key)
            self.aggregators.append(aggregator)


    # def forward(self, batch):
    def forward(self, z, disp, edge_index, batch):
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
        atom_node, force_node, disp_node, dir_edge, dist_edge = self.embedding_layer(z, disp)

        # compute interaction block and update atomic embeddings
        for interaction_layer in self.interaction_layers:
            atom_node, force_node, disp_node = interaction_layer(atom_node, force_node, disp_node, dir_edge, dist_edge, edge_index)

        # output net
        outputs = CustomOutputSet(z=z, disp=disp, atom_node=atom_node, force_node=force_node, edge_index=edge_index, batch=batch)
        for key, output_layer, scaler, aggregator in zip(self.infer_properties, self.output_layers, self.scalers, self.aggregators):
            output = output_layer(outputs)
            output = scaler(output, outputs)
            output = aggregator(output, outputs)
            setattr(outputs, key, output)

        return outputs



class EmbeddingNet(nn.Module):
    '''
    Embedding layer of the network

    Parameters:
        n_features (int): Number of features in the hidden layer.
        representations (dict): The distance transformation functions.
    '''
    def __init__(self, n_features, representations):

        super(EmbeddingNet, self).__init__()

        # atomic embedding
        self.n_features = n_features
        self.node_embedding = nn.Embedding(118 + 1, n_features, padding_idx=0)

        # edge embedding
        self.norm = representations['norm']
        self.cutoff = representations['cutoff']
        self.edge_embedding = representations['radial']
        self.n_basis = self.edge_embedding.n_basis

        # requires dr
        self.requires_dr = False

    def forward(self, z, disp):

        # initialize node representations
        atom_node = self.node_embedding(z)  # n_nodes, n_features
        force_node = torch.zeros(z.size(0), 3, self.n_features, dtype=disp.dtype, device=disp.device)  # n_nodes, 3, n_features
        disp_node = torch.zeros(z.size(0), 3, self.n_features, dtype=disp.dtype, device=disp.device)  # n_nodes, 3, n_features

        # recompute distances and distance vectors
        if self.requires_dr:
            disp.requires_grad = True

        # initialize edge representations
        dist_edge, dir_edge = self.norm(disp)  # n_edges, 1; n_edges, 3
        dist_edge = self.cutoff(dist_edge) * self.edge_embedding(dist_edge)  # n_edges, n_basis

        return atom_node, force_node, disp_node, dir_edge, dist_edge
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
        super(InteractionNet, self).__init__()

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
    
    def forward(self, atom_node, force_node, disp_node, dir_edge, dist_edge, edge_index):
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

        return atom_node, force_node, disp_node
