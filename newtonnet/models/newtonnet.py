import torch
from torch import nn
from torch_geometric.utils import scatter

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
            activation: nn.Module = nn.SiLU(),
            infer_properties: list = [],
            scalers: nn.ModuleDict = {},
            device: torch.device = torch.device('cpu'),
    ) -> None:

        super(NewtonNet, self).__init__()

        # embedding layer
        if distance_network is None:
            distance_network = nn.ModuleDict({
                'scalednorm': get_cutoff_by_string('scalednorm'),
                'cutoff': get_cutoff_by_string('poly'), 
                'representation': get_representation_by_string('bessel'),
                })
        self.embedding_layer = EmbeddingNet(
            n_features=n_features,
            z_max=max([scaler.z_max for scaler in scalers.values()]) if scalers else 128,
            distance_network=distance_network,
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
        self.output_layers = nn.ModuleDict({})
        for key in infer_properties:
            scaler = scalers[key] if key in scalers else None
            output_layer = get_output_by_string(key, scaler=scaler, n_features=n_features, activation=activation)
            self.output_layers.update({key: output_layer})
            if isinstance(output_layer, FirstDerivativeProperty):
                self.embedding_layer.requires_dr = True
            if isinstance(output_layer, SecondDerivativeProperty):
                dependent_property = output_layer.dependent_property
                assert dependent_property in self.output_layers.keys(), f'cannot find dependent property {dependent_property}'
                self.output_layers[dependent_property].requires_dr = True
        
        # device
        self.to(device)

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
        atom_node, force_node, disp_node, disp_edge, dist_edge = self.embedding_layer(z, pos, edge_index)

        # compute interaction block and update atomic embeddings
        for interaction_layer in self.interaction_layers:
            atom_node, force_node, disp_node = interaction_layer(atom_node, force_node, disp_node, disp_edge, dist_edge, edge_index)

        # output net
        outputs = {
            'z': z,
            'pos': pos,
            'batch': batch,
            'atom_node': atom_node,
            'force_node': force_node,
            }
        for key, output_layer in self.output_layers.items():
            output = output_layer(**outputs)
            outputs.update({key: output})

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
        self.n_basis = self.edge_embedding.n_basis

        # requires dr
        self.requires_dr = False

    def forward(self, z, pos, edge_index):

        # initialize node representations
        atom_node = self.node_embedding(z)  # n_nodes, n_features
        force_node = torch.zeros(*pos.shape, self.n_features)  # n_nodes, 3, n_features
        disp_node = torch.zeros(*pos.shape, self.n_features)  # n_nodes, 3, n_features

        # recompute distances and distance vectors
        if self.requires_dr:
            pos.requires_grad_()
            disp_edge = pos[edge_index[0]] - pos[edge_index[1]]  # n_edges, 3

        # initialize edge representations
        dist = self.norm(disp_edge)  # n_edges, 1
        dist_edge = self.cutoff(dist) * self.edge_embedding(dist)  # n_edges, n_basis

        return atom_node, force_node, disp_node, disp_edge, dist_edge


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
        self.inv_message_nodepart = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )
        self.inv_message_edgepart = nn.Linear(n_basis, n_features)
        nn.init.xavier_uniform_(self.inv_message_edgepart.weight)
        nn.init.zeros_(self.inv_message_edgepart.bias)

        # equivariant message passing
        self.equiv_message1 = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )
        self.equiv_message2 = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )
        self.equiv_message3 = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            activation,
            nn.Linear(n_features, n_features, bias=False),
        )

        self.inv_update = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )

        self.norm = nn.LayerNorm(n_features)
    
    def forward(self, atom_node, force_node, disp_node, disp_edge, dist_edge, edge_index):
        # a
        inv_message_nodepart = self.inv_message_nodepart(atom_node)    # n_nodes, n_features
        inv_message_edgepart = self.inv_message_edgepart(dist_edge)    # n_edges, n_features
        inv_message = inv_message_edgepart * inv_message_nodepart[edge_index[0]] * inv_message_nodepart[edge_index[1]]    # n_edges, n_features
        atom_node = atom_node + scatter(inv_message, edge_index[0], dim=0)    # n_nodes, n_features

        # f
        equiv_message1_invedgepart = self.equiv_message1(inv_message).unsqueeze(1)    # n_edges, 1, n_features
        equiv_message1_equivedgepart = disp_edge.unsqueeze(2)    # n_edges, 3, 1
        equiv_message1 = equiv_message1_invedgepart * equiv_message1_equivedgepart    # n_edges, 3, n_features
        force_node = force_node + scatter(equiv_message1, edge_index[0], dim=0)    # n_nodes, 3, n_features
        
        # dr
        equiv_message2_nodepart = disp_node    # n_nodes, 3, n_features
        equiv_message2_edgepart = self.equiv_message2(inv_message).unsqueeze(1)    # n_edges, 1, n_features
        equiv_message2 = equiv_message2_edgepart * equiv_message2_nodepart[edge_index[1]]    # n_edges, 3, n_features
        disp_node = disp_node + scatter(equiv_message2, edge_index[0], dim=0)    # n_nodes, 3, n_features

        equiv_message3_invnodepart = self.equiv_message3(atom_node).unsqueeze(1)    # n_nodes, 1, n_features
        equiv_message3_equivnodepart = scatter(equiv_message1, edge_index[0], dim=0)    # n_nodes, 3, n_features
        equiv_message3 = equiv_message3_invnodepart[edge_index[1]] * equiv_message3_equivnodepart[edge_index[1]]    # n_edges, 3, n_features
        disp_node = disp_node + scatter(equiv_message3, edge_index[0], dim=0)    # n_nodes, 3, n_features

        # update energy
        inv_update = self.inv_update(atom_node) * torch.sum(- force_node * disp_node, dim=-2)  # n_nodes, n_features
        atom_node = atom_node + inv_update

        # layer norm
        atom_node = self.norm(atom_node)

        return atom_node, force_node, disp_node
