import torch
from torch import nn
from torch_geometric.utils import scatter
from newtonnet.models.output import DirectProperty


class BondOutput(DirectProperty):
    def __init__(
        self,
        n_features: int = 128,
        activation: str = "swish",
        representations: nn.Module = None,
    ):
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, 1),
        )
        self.embedding_layer = EdgeEmbeddingNet(representations=representations)
        self.message_layer = EdgeInteractionNet(
            n_features=n_features,
            n_basis=self.embedding_layer.n_basis,
            activation=activation,
        )

    def forward(self, outputs):
        _, dist_edge = self.embedding_layer(outputs.disp)
        atom_edge = self.message_layer(outputs.atom_node, dist_edge, outputs.edge_index)
        bond_order = self.layers(atom_edge)
        return torch.sparse_coo_tensor(
            outputs.edge_index,
            bond_order,
            dtype=bond_order.dtype,
            device=bond_order.device,
        )


class EdgeInteractionNet(nn.Module):
    """
    Parameters:
        n_features (int): Number of features in the hidden layer.
        n_basis (int): Number of radial basis functions.
        activation (nn.Module): Activation function.
    """

    def __init__(self, n_features, n_basis, activation):
        super().__init__()

        self.n_features = n_features

        # invariant message passing
        self.message_nodepart = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )
        self.message_edgepart = nn.Linear(n_basis, n_features, bias=False)

    def forward(self, atom_node, dist_edge, edge_index):
        message_nodepart = self.message_nodepart(atom_node)  # n_nodes, n_features
        message_edgepart = self.message_edgepart(dist_edge)  # n_edges, n_features
        message = (
            message_edgepart
            * message_nodepart[edge_index[0]]
            * message_nodepart[edge_index[1]]
        )  # n_edges, n_features
        return message


class EdgeEmbeddingNet(nn.Module):
    """
    Parameters:
        representations (dict): The distance transformation functions.
    """

    def __init__(self, representations):
        super().__init__()

        # edge embedding
        self.norm = representations["norm"]
        self.cutoff = representations["cutoff"]
        self.edge_embedding = representations["radial"]
        self.n_basis = self.edge_embedding.n_basis

        # requires dr
        self.requires_dr = False

    def forward(self, disp):
        # recompute distances and distance vectors
        if self.requires_dr:
            disp.requires_grad = True

        # initialize edge representations
        dist_edge, dir_edge = self.norm(disp)  # n_edges, 1; n_edges, 3
        dist_edge = self.cutoff(dist_edge) * self.edge_embedding(
            dist_edge
        )  # n_edges, n_basis

        return dir_edge, dist_edge
