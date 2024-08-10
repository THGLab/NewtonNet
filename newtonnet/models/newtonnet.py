import torch
from torch import nn
from torch_geometric.utils import scatter

from newtonnet.layers.activations import get_activation_by_string
from newtonnet.layers.scalers import NullScaleShift
from newtonnet.layers.representations import get_representation_by_string
from newtonnet.models.output import get_output_by_string, CustomOutputs, FirstDerivativeProperty, SecondDerivativeProperty


class NewtonNet(nn.Module):
    '''
    Molecular Newtonian Message Passing

    Parameters:
        n_features (int): Number of features in the latent layer. Default: 128.
        n_interactions (int): Number of message passing layers. Default: 3.
        activation (str): Activation function. Default: 'swish'.
        infer_properties (list): The properties to predict. Default: [].
        representations (dict): The distance transformation functions.
        scalers (nn.ModuleDict): The scalers for the chemical properties. Default: None.
    '''
    def __init__(
            self,
            n_features: int = 128,
            n_interactions: int = 3,
            activation: str = 'swish',
            infer_properties: list = [],
            representations: nn.Module = None,
            scalers: dict = None,
    ) -> None:

        super(NewtonNet, self).__init__()
        activation = get_activation_by_string(activation)

        # embedding layer
        self.embedding_layer = EmbeddingNet(
            n_features=n_features,
            z_max=max([scaler.z_max for scaler in scalers.values()]) if scalers else 128,
            representations=representations,
            )

        # message passing
        self.interaction_layers = nn.ModuleList([
            InteractionNet(
                n_features=n_features,
                n_basis=self.embedding_layer.n_basis,
                activation=activation,
                ) for _ in range(n_interactions)
            ])

        # final output layer
        self.output_layers = nn.ModuleList()
        for key in infer_properties:
            output_layer = get_output_by_string(key, n_features, activation, scalers)
            self.output_layers.append(output_layer)
            if isinstance(output_layer, FirstDerivativeProperty):
                self.embedding_layer.requires_dr = True
            # if isinstance(output_layer, SecondDerivativeProperty):
            #     dependent_property = output_layer.dependent_property
            #     assert dependent_property in self.output_layers.keys(), f'cannot find dependent property {dependent_property}'
            #     self.output_layers[dependent_property].requires_dr = True
        
        # device
        # self.to(device)

    def forward(self, z, pos, edge_index, batch):
        '''
        Network forward pass

        Parameters:
            z (torch.Tensor): The atomic numbers of the atoms in the molecule. Shape: (n_nodes, ).
            pos (torch.Tensor): The positions of the atoms in the molecule. Shape: (n_nodes, 3).
            edge_index (torch.Tensor): The edge index of the atoms in the molecule. Shape: (2, n_edges).
            batch (torch.Tensor): The batch of the atoms in the molecule. Shape: (n_nodes, ).

        Returns:
            outputs (dict): The outputs of the network.
        '''
        # initialize node and edge representations
        atom_node, force_node, disp_node, dir_edge, dist_edge = self.embedding_layer(z, pos, edge_index)
        # atom_node, force_node, disp_node, dir_edge, cutoff_edge, rbf_edge = self.embedding_layer(z, pos, edge_index)

        # compute interaction block and update atomic embeddings
        for interaction_layer in self.interaction_layers:
            atom_node, force_node, disp_node = interaction_layer(atom_node, force_node, disp_node, dir_edge, dist_edge, edge_index)
            # atom_node, force_node, disp_node = interaction_layer(atom_node, force_node, disp_node, dir_edge, cutoff_edge, rbf_edge, edge_index)

        # output net
        outputs = CustomOutputs(z, pos, batch, atom_node, force_node)
        for output_layer in self.output_layers:
            outputs = output_layer(outputs)

        return outputs



class EmbeddingNet(nn.Module):
    '''
    Embedding layer of the network

    Parameters:
        n_features (int): Number of features in the hidden layer.
        z_max (int): Maximum atomic number.
        representations (dict): The distance transformation functions.
    '''
    def __init__(self, n_features, z_max, representations):

        super(EmbeddingNet, self).__init__()

        # atomic embedding
        self.n_features = n_features
        self.node_embedding = nn.Embedding(z_max + 1, n_features)

        # edge embedding
        self.norm = representations['scale']
        self.cutoff = representations['cutoff']
        self.edge_embedding = representations['radial']
        self.n_basis = self.edge_embedding.n_basis

        # requires dr
        self.requires_dr = False

    def forward(self, z, pos, edge_index):

        # initialize node representations
        atom_node = self.node_embedding(z)  # n_nodes, n_features
        force_node = torch.zeros(*pos.shape, self.n_features, dtype=pos.dtype, device=pos.device)  # n_nodes, 3, n_features
        disp_node = torch.zeros(*pos.shape, self.n_features, dtype=pos.dtype, device=pos.device)  # n_nodes, 3, n_features

        # recompute distances and distance vectors
        if self.requires_dr:
            pos.requires_grad_()
        disp_edge = pos[edge_index[0]] - pos[edge_index[1]]  # n_edges, 3

        # initialize edge representations
        dist_edge, dir_edge = self.norm(disp_edge)  # n_edges, 1; n_edges, 3
        dist_edge = self.cutoff(dist_edge) * self.edge_embedding(dist_edge)  # n_edges, n_basis
        # cutoff_edge = self.cutoff(dist_edge)  # n_edges, 1
        # rbf_edge = self.edge_embedding(dist_edge)  # n_edges, n_basis

        return atom_node, force_node, disp_node, dir_edge, dist_edge
        # return atom_node, force_node, disp_node, dir_edge, cutoff_edge, rbf_edge


class InteractionNet(nn.Module):
    '''
    Message passing layer of the network

    Parameters:
        n_features (int): Number of features in the hidden layer.
        n_basis (int): Number of radial basis functions.
        activation (nn.Module): Activation function.
    '''
    def __init__(self, n_features, n_basis, activation):
        super(InteractionNet, self).__init__()

        self.n_features = n_features

        # invariant message passing
        self.message_nodepart = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )
        self.message_edgepart = nn.Linear(n_basis, n_features, bias=False)
        # nn.init.xavier_uniform_(self.message_edgepart.weight)
        # nn.init.zeros_(self.inv_message_edgepart.bias)
        # self.message_edgepart = nn.Sequential(
        #     nn.Linear(n_basis, n_features, bias=False),
        #     activation,
        #     nn.Linear(n_features, n_features, bias=False),
        # )

        # self.inv_message = nn.Sequential(
        #     nn.Linear(n_features, n_features, bias=False),
        #     activation,
        #     nn.Linear(n_features, n_features, bias=False),
        # )

        # equivariant message passing
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
        self.equiv_message3 = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )

        # self.inv_update1 = nn.Sequential(
        #     nn.Linear(n_features, n_features),
        #     activation,
        #     nn.Linear(n_features, n_features),
        # )
        self.inv_update2 = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )

        # self.norm = nn.LayerNorm(n_features)
    
    def forward(self, atom_node, force_node, disp_node, dir_edge, dist_edge, edge_index):
    # def forward(self, atom_node, force_node, disp_node, dir_edge, cutoff_edge, rbf_edge, edge_index):
        # a
        message_nodepart = self.message_nodepart(atom_node)    # n_nodes, n_features
        message_edgepart = self.message_edgepart(dist_edge)    # n_edges, n_features
        # message_edgepart = self.message_edgepart(rbf_edge) * cutoff_edge    # n_edges, n_features
        message = message_edgepart * message_nodepart[edge_index[0]] * message_nodepart[edge_index[1]]    # n_edges, n_features

        # inv_update1 = self.inv_update1(message)    # n_edges, n_features
        inv_update1 = message    # n_nodes, n_features
        inv_update1 = scatter(inv_update1, edge_index[0], dim=0, dim_size=atom_node.size(0))    # n_nodes, n_features
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
        
        # # dr
        # equiv_message2_nodepart = disp_node    # n_nodes, 3, n_features
        # equiv_message2_edgepart = self.equiv_message2(inv_message).unsqueeze(1)    # n_edges, 1, n_features
        # equiv_message2 = equiv_message2_edgepart * equiv_message2_nodepart[edge_index[1]]    # n_edges, 3, n_features
        # disp_node = disp_node + scatter(equiv_message2, edge_index[0], dim=0, dim_size=disp_node.size(0))    # n_nodes, 3, n_features

        # equiv_message3_invnodepart = self.equiv_message3(atom_node).unsqueeze(1)    # n_nodes, 1, n_features
        # equiv_message3_equivnodepart = force_update    # n_nodes, 3, n_features
        # equiv_message3 = equiv_message3_invnodepart * equiv_message3_equivnodepart    # n_nodes, 3, n_features
        # disp_update = scatter(equiv_message3[edge_index[1]], edge_index[0], dim=0, dim_size=disp_node.size(0))    # n_nodes, 3, n_features
        # disp_node = disp_node + disp_update    # n_nodes, 3, n_features

        # update energy
        # inv_update2 = self.inv_update2(atom_node) * torch.sum(- force_node * disp_node, dim=1)  # n_nodes, n_features
        inv_update2 = self.inv_update2(atom_node) * torch.sum(force_node * force_node, dim=1)    # n_nodes, n_features
        atom_node = atom_node + inv_update2

        # # layer norm
        # atom_node = self.norm(atom_node)

        return atom_node, force_node, disp_node
