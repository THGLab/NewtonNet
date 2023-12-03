import torch
from torch import nn
from torch.autograd import grad
from torch.jit.annotations import Optional

from typing import Tuple, Union, List, Dict, Callable

# from newtonnet.layers import Dense
from newtonnet.layers.shells import ShellProvider
from newtonnet.layers.scalers import ScaleShift
from newtonnet.layers.cutoff import PolynomialCutoff
from newtonnet.layers.representations import RadialBesselLayer


class NewtonNet(nn.Module):
    """
    Molecular Newtonian Message Passing

    Parameters
    ----------
    resolution: int
        number of radial functions to describe interatomic distances

    n_features: int
        number of neurons in the latent layer. This number will remain fixed in the entire network except
        for the last fully connected network that predicts atomic energies.

    activation: function
        activation function from newtonnet.layers.activations
        you can aslo get it by string from newtonnet.layers.activations.get_activation_by_string

    n_interactions: int, default: 3
        number of interaction blocks

    dropout: float, default: 0.0
        dropout rate
    
    max_z: int, default: 10
        maximum atomic number Z in the dataset

    cutoff: float, default: 5.0
        cutoff radius in Angstrom

    cutoff_network: str, default: 'poly'
        cutoff function, can be 'poly' or 'cosine'

    normalizer: tuple, default: (0.0, 1.0)
        mean and standard deviation of the target property. If you have a dictionary of normalizers for each atomic type,
        you can pass it as a dictionary. For example, {'1': (0.0, 1.0), '6': (0.0, 1.0), '7': (0.0, 1.0), '8': (0.0, 1.0)}

    normalize_atomic: bool, default: False
        whether to normalize the atomic energies

    requires_dr: bool, default: False
        whether to compute the forces

    device: torch.device, default: None
        device to run the network

    create_graph: bool, default: False
        whether to create the graph for the gradient computation

    shared_interactions: bool, default: False
        whether to share the interaction block weights

    return_latent: bool, default: False
        whether to return the latent forces
    """

    def __init__(
            self,
            n_basis: int = 20,
            n_features: int = 128,
            activation: nn.Module = nn.SiLU(),
            n_layers: int = 3,
            dropout: float = 0.0,
            max_z: int = 10,
            cutoff: float = 5.0,
            cutoff_network: nn.Module = PolynomialCutoff(),
            normalizer: tuple = (0.0, 1.0),
            train_normalizer: bool = False,
            requires_dr: bool = False,
            device: torch.device = None,
            create_graph: bool = False,
            share_layers: bool = False,
            return_hessian: bool = False,
            layer_norm: bool = False,
            atomic_properties_only: bool = False,
            double_update_latent: bool = True,
            period_boundary: bool = False,
            aggregration: Callable = torch.sum,
    ) -> None:

        super(NewtonNet, self).__init__()

        self.requires_dr = requires_dr
        self.create_graph = create_graph
        self.train_normalizer = train_normalizer
        self.return_hessian = return_hessian
        self.period_boundary = period_boundary
        self.epsilon = 1.0e-8

        # atomic embedding
        self.n_features = n_features
        self.node_embedding = nn.Embedding(max_z, n_features, padding_idx=0, device=device)

        # edge embedding
        shell_cutoff = cutoff
        if period_boundary:
            # make the cutoff here a little bit larger so that it can be handled with differentiable cutoff layer in interaction block
            shell_cutoff = cutoff * 1.1
        self.shell = ShellProvider(periodic_boundary=period_boundary, cutoff=shell_cutoff)
        self.edge_embedding = RadialBesselLayer(n_basis, cutoff, device=device)
        self.epsilon = 1.0e-8

        # message passing
        self.n_interactions = n_layers
        if share_layers:
            # use the same message instance (hence the same weights)
            self.message_passing_layers = nn.ModuleList([
                MessagePassingLayer(
                    n_features=n_features,
                    n_basis=n_basis,
                    activation=activation,
                    cutoff_network=cutoff_network,
                    double_update_latent=double_update_latent,
                    layer_norm=layer_norm,
                    )
                ] * n_layers)
        else:
            # use one SchNetInteraction instance for each interaction
            self.message_passing_layers = nn.ModuleList([
                MessagePassingLayer(
                    n_features=n_features,
                    n_basis=n_basis,
                    activation=activation,
                    cutoff_network=cutoff_network,
                    double_update_latent=double_update_latent,
                    layer_norm=layer_norm,
                    ) for _ in range(n_layers)
                ])

        # final output layer
        self.invariant_node_property = InvariantNodeProperty(n_features, activation, dropout)

        if self.train_normalizer:
            self.inverse_normalize = ScaleShift(max_z)
        else:
            if type(normalizer) is dict:
                normalizer = torch.tensor([normalizer.get(i, (0, 1)) for i in range(max_z)], device=device)
                self.inverse_normalize = ScaleShift(
                    max_z=max_z,
                    mean=normalizer[:, 0],
                    stddev=normalizer[:, 1],
                    trainable=False,
                    )
            else:
                self.inverse_normalize = ScaleShift(
                    max_z=max_z,
                    mean=torch.tensor(normalizer[0], device=device),
                    stddev=torch.tensor(normalizer[1], device=device),
                    trainable=False,
                    )

        self.atomic_properties_only = atomic_properties_only
        self.aggregration = aggregration


    def forward(
            self,
            atomic_numbers: torch.Tensor,
            positions: torch.Tensor,
            atom_mask: torch.Tensor,
            neighbors: torch.Tensor,
            neighbor_mask: torch.Tensor,
            distances: torch.Tensor,
            distance_vectors: torch.Tensor,
            lattice: torch.Tensor = torch.eye(3),
            ):

        # initiate main containers
        invariant_node = self.node_embedding(atomic_numbers)  # batch_size, n_atoms, n_features
        batch_size, n_atoms, n_features = invariant_node.shape
        equivariant_node_F = torch.zeros((batch_size, n_atoms, 3), dtype=torch.float, device=positions.device)  # batch_size, n_atoms, 3
        equivariant_node_f = torch.zeros((batch_size, n_atoms, 3, n_features), device=positions.device)  # B,A,3,nf
        equivariant_node_dr = torch.zeros((batch_size, n_atoms, 3, n_features), device=positions.device)  # B,A,3,nf

        # recompute distances (B,A,N) and distance vectors (B,A,N,3)
        if self.requires_dr:
            positions.requires_grad_()
            distances, distance_vectors, neighbors, neighbor_mask = self.shell(positions, neighbors, neighbor_mask, lattice)
            distance_vectors = distance_vectors / (distances[:, :, :, None] + self.epsilon)

        # comput d1 representation (B, A, N, G)
        invariant_edges = self.edge_embedding(distances)

        # compute interaction block and update atomic embeddings
        for message_passing_layer in self.message_passing_layers:
            # messages
            invariant_node, equivariant_node_F, equivariant_node_f, equivariant_node_dr = message_passing_layer(
                invariant_node, 
                invariant_edges, 
                distances, 
                distance_vectors, 
                neighbors, 
                neighbor_mask, 
                equivariant_node_F, 
                equivariant_node_f, 
                equivariant_node_dr,
                )

        # When using the network to obtain atomic properties only
        if self.atomic_properties_only:
            invariant_node_output = self.invariant_node_property(invariant_node)
            invariant_node_output = self.inverse_normalize(invariant_node_output, atomic_numbers)
            return {'Ai': invariant_node_output}

        # output net
        invariant_node_output = self.invariant_node_property(invariant_node)
        invariant_node_output = self.inverse_normalize(invariant_node_output, atomic_numbers)
        # if self.train_normalizer:
        #     Ei = self.inverse_normalize(Ei, atomic_numbers)

        # inverse normalize
        invariant_node_output = invariant_node_output * atom_mask[..., None]  # (B,A,1)
        invariant_graph_output = self.aggregration(invariant_node_output, dim=1)  # (B,1)
        # if not self.train_normalizer:
        #     E = self.inverse_normalize(E, torch.zeros_like(E, dtype=torch.long))

        if self.requires_dr:
            if self.return_hessian:
                invariant_graph_output_derivative = grad([invariant_graph_output], [positions], grad_outputs=[torch.ones(invariant_graph_output.shape[0], 1, device=positions.device) if invariant_graph_output.shape[0]==positions.shape[0] else None], create_graph=True, retain_graph=True)[0]
                # TODO: make Hessian calculations work
                # ddE = torch.zeros(E.shape[0], R.shape[1], R.shape[2], R.shape[1], R.shape[2], device=R.device)
                # for A_ in range(R.shape[1]):
                #     for X_ in range(R.shape[2]):
                #         dE[:, A_, X_]
                #         ddE[:, A_, X_, :, :] = grad(dE[:, A_, X_], R, grad_outputs=torch.ones(E.shape[0], device=R.device), create_graph=False, retain_graph=True)[0]
                # ddE = torch.stack([grad(dE, R, grad_outputs=V, create_graph=True, retain_graph=True, allow_unused=True)[0] for V in torch.eye(R.shape[1] * R.shape[2], device=R.device).reshape((-1, 1, R.shape[1], R.shape[2])).repeat(1, R.shape[0], 1, 1)])
                # ddE = torch.vmap(lambda V: grad(dE, R, grad_outputs=V, create_graph=True, retain_graph=True))(torch.eye(R.shape[1] * R.shape[2], device=R.device).reshape((-1, 1, R.shape[1], R.shape[2])).repeat(1, R.shape[0], 1, 1))
                # ddE = ddE.permute(1,2,3,0).unflatten(dim=3, sizes=(-1, 3))
            else:
                invariant_graph_output_derivative = grad([invariant_graph_output], [positions], grad_outputs=[torch.ones(invariant_graph_output.shape[0], 1, device=positions.device) if invariant_graph_output.shape[0]==positions.shape[0] else None], create_graph=True, retain_graph=True)[0]
        else:
            invariant_graph_output_derivative = torch.zeros_like(positions)
        assert invariant_graph_output_derivative is not None
        invariant_graph_output_derivative = -invariant_graph_output_derivative

        if self.return_hessian:
            # TODO: make Hessian calculations work
            # return {'R': R, 'E': E, 'F': dE, 'H': ddE, 'Ei': Ei, 'F_latent': f_dir}
            return {'R': positions, 'E': invariant_graph_output, 'F': invariant_graph_output_derivative, 'Ei': invariant_node_output, 'F_latent': equivariant_node_F}
        else:
            return {'R': positions, 'E': invariant_graph_output, 'F': invariant_graph_output_derivative, 'Ei': invariant_node_output, 'F_latent': equivariant_node_F}


class MessagePassingLayer(nn.Module):

    def __init__(
            self,
            n_features: int = 128,
            n_basis: int = 20,
            activation: nn.Module = nn.SiLU(),
            cutoff_network: nn.Module = PolynomialCutoff(),
            double_update_latent: bool = True,
            layer_norm: bool = False,
            ):
        super(MessagePassingLayer, self).__init__()

        self.n_features = n_features

        # non-directional message passing
        self.invariant_message_edge = nn.Linear(n_basis, n_features)
        self.invariant_message_node = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )

        # cutoff layer used in interaction block
        self.cutoff_network = cutoff_network

        # directional message passing
        self.equivariant_message_coefficient = nn.Linear(n_features, 1, bias=False)
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

        self.double_update_latent = double_update_latent

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm = nn.LayerNorm(n_features)

    def gather_neighbors(self, inputs, neighbors):
        n_features = inputs.shape[-1]
        n_dim = inputs.dim()
        batch_size, n_atoms, n_neighbors = neighbors.size()  # batch, atoms, neighbors size

        if n_dim == 3:    # inputs: batch_size, n_atoms, n_features
            neighbors = neighbors[:, :, :, None].expand(-1, -1, -1, n_features)    # batch_size, n_atoms, n_neighbors, n_features
            inputs = inputs[:, :, None, :].expand(-1, -1, n_neighbors, -1)    # batch_size, n_atoms, n_neighbors, n_features
            outputs = torch.gather(inputs, dim=1, index=neighbors)    # batch_size, n_atoms, n_neighbors, n_features
            return outputs
        elif n_dim == 4:    # inputs: batch_size, n_atoms, 3, n_features
            neighbors = neighbors[:, :, :, None, None].expand(-1, -1, -1, 3, n_features)    # batch_size, n_atoms, n_neighbors, 3, n_features
            inputs = inputs[:, :, None, :, :].expand(-1, -1, n_neighbors, -1, -1)    # batch_size, n_atoms, n_neighbors, 3, n_features
            outputs = torch.gather(inputs, dim=1, index=neighbors)    # batch_size, n_atoms, n_neighbors, 3, n_features
            return outputs
        else:
            raise ValueError(f'Unknown input dimension: {n_dim}')

    def sum_neighbors(self, inputs, mask):
        n_dim = inputs.dim()

        if n_dim == 3:    # inputs: batch_size, n_atoms, n_neighbors
            outputs = torch.sum(inputs * mask, dim=2)
        elif n_dim == 4:    # inputs: batch_size, n_atoms, n_neighbors, n_features
            mask = mask[:, :, :, None]
            outputs = torch.sum(inputs * mask, dim=2)
        elif n_dim == 5:    # inputs: batch_size, n_atoms, n_neighbors, 3, n_features
            mask = mask[:, :, :, None, None]
            outputs = torch.sum(inputs * mask, dim=2)
        else:
            raise ValueError(f'Unknown input dimension: {n_dim}')
        return outputs
    
    def forward(
            self, 
            invariant_node, 
            invariant_edge, 
            distances, 
            distance_vector, 
            neighbors, 
            neighbor_mask,
            equivariant_node_F, 
            equivariant_node_f, 
            equivariant_node_dr, 
            ):

        # map decomposed distances
        invariant_message_edge = self.invariant_message_edge(invariant_edge)    # batch_size, n_atoms, n_neighbors, n_features

        # cutoff
        invariant_message_edge = invariant_message_edge * self.cutoff_network(distances)[:, :, :, None]    # batch_size, n_atoms, n_neighbors, n_features

        # map atomic features
        invariant_message_node = self.invariant_message_node(invariant_node)    # batch_size, n_atoms, n_features

        # # symmetric feature multiplication
        invariant_message = invariant_message_edge * self.gather_neighbors(invariant_message_node, neighbors) * invariant_message_node[:, :, None, :]    # batch_size, n_atoms, n_neighbors, n_features

        # update a with invariance
        if self.double_update_latent:
            invariant_update = self.sum_neighbors(invariant_message, neighbor_mask)    # batch_size, n_atoms, n_features
            invariant_node = invariant_node + invariant_update    # batch_size, n_atoms, n_features

        # F
        equivariant_message_F = self.equivariant_message_coefficient(invariant_message) * distance_vector  # batch_size, n_atoms, n_neighbors, 3
        equivariant_update_F = self.sum_neighbors(equivariant_message_F, neighbor_mask)  # batch_size, n_atoms, 3
        equivariant_node_F = equivariant_node_F + equivariant_update_F

        # f
        equivariant_message_f = self.equivariant_message_feature(invariant_message)[:, :, :, None, :] * equivariant_message_F[:, :, :, :, None]  # batch_size, n_atoms, n_neighbors, 3, n_features
        equivariant_update_f = self.sum_neighbors(equivariant_message_f, neighbor_mask)  # batch_size, n_atoms, 3, n_features
        equivariant_node_f = equivariant_node_f + equivariant_update_f
        
        # dr
        equivariant_message_dr_edge = self.equivariant_message_edge(invariant_message)[:, :, :, None, :]    # batch_size, n_atoms, n_neighbors, 3, n_features
        equivariant_message_dr_node = equivariant_node_dr    # batch_size, n_atoms, 3, n_features
        equivariant_message_dr = equivariant_message_dr_edge * self.gather_neighbors(equivariant_message_dr_node, neighbors)    # batch_size, n_atoms, n_neighbors, 3, n_features
        equivariant_update_dr = self.sum_neighbors(equivariant_message_dr, neighbor_mask)  # batch_size, n_atoms, 3, n_features
        equivariant_node_dr = equivariant_node_dr + equivariant_update_dr    # batch_size, n_atoms, 3, n_features

        equivariant_update_dr = self.equivariant_selfupdate_coefficient(invariant_node)[:, :, None, :] * equivariant_update_f    # batch_size, n_atoms, 3, n_features
        equivariant_node_dr = equivariant_node_dr + equivariant_update_dr    # batch_size, n_atoms, 3, n_features

        # update energy
        invariant_update = -self.invariant_selfupdate_coefficient(invariant_node) * torch.sum(equivariant_node_f * equivariant_node_dr, dim=-2)  # batch_size, n_atoms, n_features
        invariant_node = invariant_node + invariant_update

        # layer norm
        if self.layer_norm:
            invariant_node = self.norm(invariant_node)

        return invariant_node, equivariant_node_F, equivariant_node_f, equivariant_node_dr


class InvariantNodeProperty(nn.Module):

    def __init__(self, n_features, activation, dropout):

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

    def forward(self, invariant_node):

        output = self.invariant_node_prediction(invariant_node)

        return output

