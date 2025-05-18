import torch
from torch import nn


class EdgeEmbedding(nn.Module):
    '''
    Edge embedding layer of the network

    Parameters:
        cutoff (float): cutoff radius.
        n_basis (int): number of radial basis functions. Default: 20.
    '''
    def __init__(self, cutoff, n_basis=20):
        super().__init__()
        self.radius_graph = RadiusGraph(r=cutoff)
        self.norm = ScaledNorm(r=cutoff)
        self.envelope = PolynomialCutoff(p=9)
        self.embedding = RadialBesselLayer(n_basis=n_basis)

    def forward(self, pos, cell=None, batch=None):
        """Compute edge embedding.

        Args:
            pos (torch.Tensor): positions of atoms.
            cell (torch.Tensor, optional): cell vectors. Default: None.
            batch (torch.Tensor, optional): batch indices. Default: None.

        Returns:
            edge_index (torch.Tensor): edge indices of the graph.
            dist_edge (torch.Tensor): distance values of the edges.
            dir_edge (torch.Tensor): direction values of the edges.

        """
        # Compute radius graph
        edge_index, dist = self.radius_graph(pos, cell=cell, batch=batch)

        # Compute distance and direction
        dist_edge, dir_edge = self.norm(dist)

        # Compute radial embedding
        dist_edge = self.envelope(dist_edge) * self.embedding(dist_edge)

        return dist_edge, dir_edge, edge_index
    

class RadiusGraph(nn.Module):
    '''
    Create a radius graph based on the interatomic distances.

    Parameters:
        r (float): cutoff radius.
    '''
    def __init__(self, r):
        super().__init__()
        self.r = r

    def forward(self, pos, cell=None, batch=None):
        """Compute radius graph.

        Args:
            pos (torch.Tensor): positions of atoms.
            cell (torch.Tensor, optional): cell vectors. Default: None.
            batch (torch.Tensor, optional): batch indices. Default: None.

        Returns:
            edge_index (torch.Tensor): edge indices of the graph.
            disp (torch.Tensor): displacement vectors of the edges.

        """
        # Create full graph
        n_node = pos.shape[0]
        if batch is not None:
            edge_index = []
            for b in batch.unique():
                nodes = (batch == b).nonzero().flatten()
                row, col = torch.meshgrid(nodes, nodes, indexing='ij')
                edge_index.append(torch.stack([row.flatten(), col.flatten()], dim=0))
            edge_index = torch.cat(edge_index, dim=1)
        else:
            row, col = torch.meshgrid(torch.arange(n_node, dtype=pos.dtype), torch.arange(n_node, dtype=pos.dtype), indexing='ij')
            edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
            
        # Compute distances
        dist = pos[edge_index[0]] - pos[edge_index[1]]
        if cell is not None and not (cell == 0).all():
            if batch is not None:
                cell = cell[batch]
            else:
                cell = cell.repeat(n_node, 1, 1)
            cell = cell[edge_index[0]]
            scaled_dist = torch.linalg.solve(cell.transpose(1, 2), dist)
            dist = dist - torch.bmm(cell, torch.round(scaled_dist).unsqueeze(-1)).squeeze(-1)

        # Filter edges based on distance
        mask = (dist.norm(dim=1) < self.r)
        edge_index = edge_index[:, mask]
        dist = dist[mask]

        return edge_index, dist

    def __repr__(self):
        return f'{self.__class__.__name__}(r={self.r})'


class ScaledNorm(nn.Module):
    '''
    Compute scaled norm of interatomic distances.
    Based on Johannes Klicpera, Janek Grob, Stephan Gunnemann. Directional Message Passing for Molecular Graphs. ICLR 2020.

    Parameters:
        r (float): cutoff radius.
    '''
    def __init__(self, r):
        super().__init__()
        self.r = r

    def forward(self, disp):
        """Compute scaled norm.

        Args:
            disp (torch.Tensor): values of interatomic distance vectors.

        Returns:
            dist (torch.Tensor): values of scaled interatomic distances.
            dir (torch.Tensor): values of normalized interatomic distance vectors.
        """
        # Compute values of scaled norm
        dist = torch.norm(disp, dim=-1, keepdim=True)
        dir = disp / dist
        dist = dist / self.r

        return dist, dir
    
    def __repr__(self):
        return f'{self.__class__.__name__}(r={self.r})'

class PolynomialCutoff(nn.Module):
    '''
    Compute polynomial cutoff function.
    Based on Johannes Klicpera, Janek Grob, Stephan Gunnemann. Directional Message Passing for Molecular Graphs. ICLR 2020.

    Parameters:
        p (int): degree of polynomial. Default: 9.

    Notes:
        y = 1 - 0.5 * (p + 1) * (p + 2) * x^p + p * (p + 2) * x^(p + 1) - 0.5 * p * (p + 1) * x^(p + 2)
        y(0) = 1
        y(1) = 0
    '''
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, dist):
        """Compute cutoff.

        Args:
            dist (torch.Tensor): values of scaled interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = 1 \
            - 0.5 * (self.p + 1) * (self.p + 2) * dist.pow(self.p) \
            + self.p * (self.p + 2) * dist.pow(self.p + 1) \
            - 0.5 * self.p * (self.p + 1) * dist.pow(self.p + 2)

        return cutoffs
    
    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class CosineCutoff(nn.Module):
    '''
    Compute Behler cosine cutoff function.
    Copied from: https://github.com/atomistic-machine-learning/schnetpack under the MIT License.

    Notes:
        y = 0.5 * (1 + cos(pi * x))
        y(0) = 1
        y(1) = 0
    '''
    def __init__(self):
        super().__init__()

    def forward(self, dist):
        """Compute cutoff.

        Args:
            dist (torch.Tensor): values of scaled interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(dist * torch.pi) + 1.0)

        return cutoffs


class RadialBesselLayer(nn.Module):
    '''
    Radial Bessel functions based on the work by DimeNet: https://github.com/klicperajo/dimenet

    Parameters:
        n_basis (int): Total number of radial functions. Default: 16.

    Notes:
        y = sin(pi * r) / (pi * r)
    '''

    def __init__(self, n_basis):
        super().__init__()
        self.n_basis = n_basis
        self.frequencies = nn.Parameter(torch.arange(1, self.n_basis + 1) * torch.pi, requires_grad=False)
        self.epsilon = 1.0e-8

    def forward(self, dist):
        '''
        Compute smeared-gaussian distance values.

        Arguments:
            dist (torch.Tensor): interatomic distance values of shape (batch_size, n_atoms, n_atoms).

        Returns:
            torch.Tensor: edge embedding of shape (batch_size, n_atoms, n_atoms, n_radials).
        '''
        out = torch.sin(self.frequencies * dist) / dist #/ self.frequencies

        return out
    
    def __repr__(self):
        return f'{self.__class__.__name__}(basis={self.n_basis})'
