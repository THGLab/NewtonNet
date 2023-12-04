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
            ):
        super(ShellProvider, self).__init__()
        self.epsilon = 1.0e-8
        self.cutoff = cutoff
        self.periodic_boundary = periodic_boundary

    def gather_neighbors(self, inputs, neighbor_mask):
        
        sparse_indices = neighbor_mask.indices()    # 3, n_atoms * n_neighbors
        sparse_shape = neighbor_mask.shape    # 3 (i.e. n_data, n_atoms, and n_neighbors)
        sparse_indices_extand = (sparse_indices.repeat_interleave(3, dim=1), torch.arange(3).repeat(1, sparse_indices.shape[1]))
        sparse_indices_extand = torch.cat(sparse_indices_extand, dim=0)    # 4, n_atoms * n_neighbors
        inputs_i = torch.sparse.FloatTensor(
            indices=sparse_indices_extand,
            values=inputs[sparse_indices[[0, 1], :].tolist()].flatten(),
            size=(*sparse_shape, 3),
            )
        inputs_f = torch.sparse.FloatTensor(
            indices=sparse_indices_extand,
            values=inputs[sparse_indices[[0, 2], :].tolist()].flatten(),
            size=(*sparse_shape, 3),
            )

        return inputs_i, inputs_f

    def forward(
            self,
            positions: torch.Tensor,
            neighbor_mask: torch.Tensor,
            lattice: torch.Tensor = torch.eye(3),
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
        # Get atomic positions of all neighboring indices
        positions_i, positions_f = self.gather_neighbors(positions, neighbor_mask)    # data_size, n_atoms, n_atoms, 3

        # Subtract positions of central atoms to get distance vectors
        distance_vectors = positions_f - positions_i    # data_size, n_atoms, n_atoms, 3

        if self.periodic_boundary:
            lattice_shift_vectors = torch.tensor(
                [[[[i, j, k] for i in (-1, 0, 1)] for j in (-1, 0, 1)] for k in (-1, 0, 1)],
                dtype=lattice.dtype,
                device=lattice.device,
                ).reshape((27, 3))    # 27, 3
            distance_shift_vectors = lattice_shift_vectors @ lattice[None, :, :]    # 1, 27, 3
            distance_vectors = distance_vectors[:, :, :, None, :] + distance_shift_vectors[:, None, None, :, :]    # data_size, n_atoms, n_neighbors, 27, 3
            distances = torch.norm(distance_vectors, dim=-1, keepdim=True)    # data_size, n_atoms, n_atoms, 27, 1
            distance_min_index = torch.argmin(distances, dim=-2, keepdim=True)    # data_size, n_atoms, n_atoms, 1, 1
            distances = torch.gather(distances, dim=-2, index=distance_min_index).squeeze(-2)    # data_size, n_atoms, n_atoms, 1
            distance_vectors = torch.gather(distance_vectors, dim=-2, index=distance_min_index.expand(-1, -1, -1, -1, 3)).squeeze(-2)    # data_size, n_atoms, n_atoms, 3
        else:
            distances = distance_vectors.square().sum(dim=3).sqrt()    # data_size, n_atoms, n_atoms
            
        return distances, distance_vectors
