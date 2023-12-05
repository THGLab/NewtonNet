import torch
from torch import nn

class ShellProvider(nn.Module):
    """
    This layer calculates distance of each atom in a molecule to its
    closest neighbouring atoms.

    Based on the SchnetPack AtomDistances: https://github.com/atomistic-machine-learning/schnetpack under the MIT License.


    Parameters
    ----------

    """

    def __init__(
            self,
            cutoff: float = 5.0,
            periodic_boundary: bool = False,
            lattice: torch.Tensor = torch.eye(3) * 10.0,
            ):
        super(ShellProvider, self).__init__()
        self.cutoff = cutoff
        self.periodic_boundary = periodic_boundary
        if self.periodic_boundary:
            self.lattice = lattice
            lattice_shift_vectors = torch.tensor(
                [[[[i, j, k] for i in (-1, 0, 1)] for j in (-1, 0, 1)] for k in (-1, 0, 1)],
                dtype=self.lattice.dtype,
                device=self.lattice.device,
                ).reshape((27, 3))    # 27, 3
            self.shift_vectors = lattice_shift_vectors @ self.lattice    # 27, 3
            self.orthorhombic = (lattice[0] @ lattice[1]) == (lattice[1] @ lattice[2]) == (lattice[2] @ lattice[0]) == 0.0    # cubic, tetragonal, or orthorhombic

    # def gather_neighbors_sparse(self, inputs, neighbor_mask):
        
    #     sparse_indices = neighbor_mask.indices()    # 3, n_atoms * n_neighbors
    #     sparse_shape = neighbor_mask.shape    # 3 (i.e. n_data, n_atoms, and n_neighbors)
    #     sparse_indices_extand = (sparse_indices.repeat_interleave(3, dim=1), torch.arange(3).repeat(1, sparse_indices.shape[1]))
    #     sparse_indices_extand = torch.cat(sparse_indices_extand, dim=0)    # 4, n_atoms * n_neighbors
    #     inputs_i = torch.sparse_coo_tensor(
    #         indices=sparse_indices_extand,
    #         values=inputs[sparse_indices[[0, 1], :].tolist()].flatten(),
    #         size=(*sparse_shape, 3),
    #         )
    #     inputs_f = torch.sparse_coo_tensor(
    #         indices=sparse_indices_extand,
    #         values=inputs[sparse_indices[[0, 2], :].tolist()].flatten(),
    #         size=(*sparse_shape, 3),
    #         )

    #     return inputs_i, inputs_f

    def gather_neighbors(self, inputs, neighbor_mask):
        """
        Gather neighbor positions for each atom in a molecule.

        Parameters
        ----------
        inputs: torch.Tensor
            XYZ coordinates of atoms in molecules.
            shape: (B, A, 3)

        neighbor_mask: torch.Tensor
            boolean mask for neighbor positions.
            shape: (B, A, N), for small systems N=A-1

        Returns
        -------
        torch.Tensor: positions of central atoms with shape: (B, A, 1, 3)
        torch.Tensor: positions of neighboring atoms with shape: (B, A, N, 3)

        Notes
        -----
        shape of tensors are specified with following symbols throughout the documentation:
            - B: batch size
            - A: max number of atoms
            - N: max number of neighbors (upper limit is A-1)

        """
        # Get atomic positions of all neighboring indices
        while inputs.dim() + 1 > neighbor_mask.dim():
            neighbor_mask = neighbor_mask.unsqueeze(-1)
        inputs_i = inputs.unsqueeze(2)    # data_size, n_atoms, 1, ...
        inputs_i = inputs_i * neighbor_mask    # data_size, n_atoms, n_atoms, ...
        inputs_f = inputs.unsqueeze(1)    # data_size, 1, n_atoms, ...
        inputs_f = inputs_f * neighbor_mask    # data_size, n_atoms, n_atoms, ...

        return inputs_i, inputs_f

    def forward(
            self,
            positions: torch.Tensor,
            neighbor_mask: torch.Tensor,
            ):
        """
        The main driver to calculate distances of atoms in a shell from center atom.

        Parameters
        ----------
        positions: torch.Tensor
            XYZ coordinates of atoms in molecules.
            shape: (B, A, 3)

        neighbors: torch.Tensor
            indices of adjacent atoms.
            shape: (B, A, N), for small systems N=A-1

        neighbor_mask: torch.tensor
            boolean mask for neighbor positions.

        lattice: torch.Tensor or None
            the pbc lattice array for current batch of data
            shape: (9,)


        Returns
        -------
        torch.Tensor: distances with shape: (B, A, N)
        torch.Tensor: distance vector with shape: (B, A, N, 3)

        Notes
        -----
        shape of tensors are specified with following symbols throughout the documentation:
            - B: batch size
            - A: max number of atoms
            - N: max number of neighbors (upper limit is A-1)

        """
        if neighbor_mask.is_sparse:
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
            if self.periodic_boundary:
                distance_vectors = distance_vectors.unsqueeze(-2) + self.shift_vectors.unsqueeze(0).unsqueeze(1).unsqueeze(2)    # data_size, n_atoms, n_atoms, 27, 3
                distances = distance_vectors.norm(dim=-1)    # data_size, n_atoms, n_atoms, 27
                distances_min_index = torch.argmin(distances, dim=-1)    # data_size, n_atoms, n_atoms
                distance_vectors = torch.gather(distance_vectors, dim=-2, index=distances_min_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, 3)).squeeze(-2)    # data_size, n_atoms, n_atoms, 3
                distances = torch.gather(distances, dim=-1, index=distances_min_index.unsqueeze(-1)).squeeze(-1)    # data_size, n_atoms, n_atoms
            else:
                distances = distance_vectors.norm(dim=-1)

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
