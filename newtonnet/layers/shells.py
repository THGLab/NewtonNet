import torch
from torch import nn


class ShellProvider(nn.Module):
    '''
    This layer calculates distance of each atom in a molecule to its closest neighbouring atoms.
    Based on the SchnetPack AtomDistances: https://github.com/atomistic-machine-learning/schnetpack under the MIT License.

    Parameters:
        cutoff (float): The cutoff radius. Default: 5.0.
        pbc (bool): Whether to use periodic boundary conditions. Default: False.
        cell (torch.tensor): Unit cell size. Default: [[0, 0, 0], [0, 0, 0], [0, 0, 0]], i.e. no periodic boundary.

    Notes:
        Sparse tensors are not yet supported.
    '''
    def __init__(
            self,
            cutoff: float = 5.0,
            pbc: bool = False,
            cell: torch.Tensor = torch.zeros(3, 3),
            ):
        super(ShellProvider, self).__init__()
        self.cutoff = cutoff
        self.pbc = pbc
        if self.pbc:
            self.cell = cell
            cell_shift_vectors = torch.tensor(
                [[[[i, j, k] for i in (-1, 0, 1)] for j in (-1, 0, 1)] for k in (-1, 0, 1)],
                device=self.cell.device,
                ).reshape((27, 3))    # 27, 3
            self.shift_vectors = cell_shift_vectors @ self.cell    # 27, 3
            self.orthorhombic = (cell[0] @ cell[1]) == (cell[1] @ cell[2]) == (cell[2] @ cell[0]) == 0.0    # cubic, tetragonal, or orthorhombic
        self.epsilon = 1.0e-8

    # def gather_neighbors_sparse(self, input, neighbor_mask):
        
    #     sparse_indices = neighbor_mask.indices()    # 3, n_atoms * n_neighbors
    #     sparse_shape = neighbor_mask.shape    # 3 (i.e. n_data, n_atoms, and n_neighbors)
    #     sparse_indices_extand = (sparse_indices.repeat_interleave(3, dim=1), torch.arange(3).repeat(1, sparse_indices.shape[1]))
    #     sparse_indices_extand = torch.cat(sparse_indices_extand, dim=0)    # 4, n_atoms * n_neighbors
    #     input_i = torch.sparse_coo_tensor(
    #         indices=sparse_indices_extand,
    #         values=inputs[sparse_indices[[0, 1], :].tolist()].flatten(),
    #         size=(*sparse_shape, 3),
    #         )
    #     input_f = torch.sparse_coo_tensor(
    #         indices=sparse_indices_extand,
    #         values=inputs[sparse_indices[[0, 2], :].tolist()].flatten(),
    #         size=(*sparse_shape, 3),
    #         )

    #     return input_i, input_f

    def gather_neighbors(self, input, neighbor_mask):
        '''
        Gather neighbor properties for each atom in a molecule.

        Args:
            input (torch.tensor): The properties of atoms in molecules with shape (n_data, n_atoms, *n_properties).
            neighbor_mask (torch.tensor): The mask for neighbors with shape (n_data, n_atoms, n_atoms).

        Returns:
            input_i (torch.tensor): The properties of central atoms with shape (n_data, n_atoms, n_atoms, *n_properties).
            input_f (torch.tensor): The properties of neighboring atoms with shape (n_data, n_atoms, n_atoms, *n_properties).
        '''
        # Expand neighbor mask to match the shape of input properties
        while input.dim() + 1 > neighbor_mask.dim():
            neighbor_mask = neighbor_mask.unsqueeze(-1)

        # Expand inputs for neighbor dimension
        input_i = input.unsqueeze(2)    # data_size, n_atoms, 1, ...
        input_i = input_i * neighbor_mask    # data_size, n_atoms, n_atoms, ...
        input_f = input.unsqueeze(1)    # data_size, 1, n_atoms, ...
        input_f = input_f * neighbor_mask    # data_size, n_atoms, n_atoms, ...

        return input_i, input_f

    def forward(
            self,
            positions: torch.Tensor,
            neighbor_mask: torch.Tensor,
            ):
        '''
        The main driver to calculate distances of atoms in a shell from center atom.

        Args:
            positions (torch.Tensor): The atomic positions with shape (n_data, n_atoms, 3).
            neighbor_mask (torch.Tensor): The mask for neighbors with shape (n_data, n_atoms, n_atoms).

        Returns:
            distances (torch.Tensor): The distances between atoms with shape (n_data, n_atoms, n_atoms).
            distance_vectors (torch.Tensor): The distance vectors between atoms with shape (n_data, n_atoms, n_atoms, 3).
            neighbor_mask (torch.Tensor): The mask for neighbors with shape (n_data, n_atoms, n_atoms).
        '''
        if neighbor_mask.is_sparse:    # TODO: implement sparse tensors
            # # Get atomic positions of all neighboring indices
            # neighbor_mask = neighbor_mask.coalesce()
            # positions_i, positions_f = self.gather_neighbors(positions, neighbor_mask)    # data_size, n_atoms, n_atoms, 3

            # # Subtract positions of central atoms to get distance vectors
            # distance_vectors = positions_f - positions_i    # data_size, n_atoms, n_atoms, 3

            # # Calculate distances
            # if self.periodic_boundary:
            #     distance_vectors = distance_vectors.unsqueeze(-2) + self.shift_vectors.unsqueeze(0).unsqueeze(1).unsqueeze(2)    # data_size, n_atoms, n_neighbors, 27, 3
            #     if self.orthorhombic:    # cubic, tetragonal, or orthorhombic
            #         distance_vectors = distance_vectors.min(dim=-2)    # data_size, n_atoms, n_neighbors, 3
            #         distances = distance_vectors.square().sum(dim=-1).sqrt()    # data_size, n_atoms, n_neighbors
            #     else:    # monoclinic, triclinic, hexagonal, or rhombohedral
            #         distances = distance_vectors.square().sum(dim=-1).sqrt()    # data_size, n_atoms, n_neighbors, 27
            #         distances_min_index = torch.argmin(distances, dim=-1)    # data_size, n_atoms, n_neighbors    # not yet implemented for sparse tensors
            #         distance_vectors = torch.gather(distance_vectors, dim=-2, index=distances_min_index.unsqueeze(-1).unsqueeze(-1)).squeeze(-2)    # data_size, n_atoms, n_atoms, 3
            #         distances = torch.gather(distances, dim=-1, index=distances_min_index.unsqueeze(-1)).squeeze(-1)    # data_size, n_atoms, n_atoms
            # else:
            #     distances = distance_vectors.square().sum(dim=-1).sqrt()    # data_size, n_atoms, n_atoms
            raise NotImplementedError('sparse tensors not yet implemented')
        else:
            # Get atomic positions of all indices
            positions_i, positions_f = self.gather_neighbors(positions, neighbor_mask)    # data_size, n_atoms, n_atoms, 3

            # Subtract positions of central atoms to get distance vectors
            distance_vectors = positions_f - positions_i    # data_size, n_atoms, n_atoms, 3

            # Calculate distances
            if self.pbc:
                distance_vectors = distance_vectors.unsqueeze(-2) + self.shift_vectors.unsqueeze(0).unsqueeze(1).unsqueeze(2)    # data_size, n_atoms, n_atoms, 27, 3
                distances = (distance_vectors + self.epsilon).norm(dim=-1)    # data_size, n_atoms, n_atoms, 27
                distances_min_index = torch.argmin(distances, dim=-1)    # data_size, n_atoms, n_atoms
                distance_vectors = torch.gather(distance_vectors, dim=-2, index=distances_min_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, 3)).squeeze(-2)    # data_size, n_atoms, n_atoms, 3
                distances = torch.gather(distances, dim=-1, index=distances_min_index.unsqueeze(-1)).squeeze(-1)    # data_size, n_atoms, n_atoms
            else:
                distances = (distance_vectors + self.epsilon).norm(dim=-1)    # data_size, n_atoms, n_atoms    # TODO: remove epsilon and replace with mask

            # mask based on cutoff
            if self.cutoff is not None:
                neighbor_mask = neighbor_mask * (distances < self.cutoff)    # data_size, n_atoms, n_atoms
            
            # sparsify matrices
            # neighbor_mask_sparse = neighbor_mask.to_sparse()
            # indices = neighbor_mask_sparse.indices()    # 3, n_atoms * n_neighbors
            # distances_sparse = torch.sparse_coo_tensor(values=distances[neighbor_mask].flatten(), indices=indices, size=distances.shape)
            # indices_expanded = (indices.repeat_interleave(3, dim=1), torch.arange(3).repeat(1, indices.shape[1]))
            # indices_expanded = torch.cat(indices_expanded, dim=0)    # 4, n_atoms * n_neighbors * 3
            # distance_vectors_sparse = torch.sparse_coo_tensor(values=distance_vectors[neighbor_mask].flatten(), indices=indices_expanded, size=distance_vectors.shape)

            # neighbor_mask = neighbor_mask_sparse
            # distances = distances_sparse
            # distance_vectors = distance_vectors_sparse
            
        return distances, distance_vectors, neighbor_mask
