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

    def gather_neighbors(self, inputs, neighbors):
        data_size, n_atoms, n_neighbors = neighbors.size()  # data_size, n_atoms, n_neighbors

        neighbors = neighbors[:, :, :, None].expand(-1, -1, -1, 3)    # data_size, n_atoms, n_neighbors, 3
        inputs = inputs[:, :, None, :].expand(-1, -1, n_neighbors, -1)    # data_size, n_atoms, n_neighbors, 3
        outputs = torch.gather(inputs, dim=1, index=neighbors)    # data_size, n_atoms, n_neighbors, 3

        return outputs

    def forward(
            self,
            positions: torch.Tensor,
            neighbors: torch.Tensor,
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
        # Construct auxiliary index vector
        data_size, n_atoms, _ = positions.size()

        # Get atomic positions of all neighboring indices
        ngh_atoms_xyz = self.gather_neighbors(positions, neighbors)    # data_size, n_atoms, n_neighbors, 3

        # Subtract positions of central atoms to get distance vectors
        distance_vectors = ngh_atoms_xyz - positions[:, :, None, :]    # data_size, n_atoms, n_neighbors, 3

        if self.periodic_boundary:
            lattice_shift_vectors = torch.tensor(
                [[[[i, j, k] for i in (-1, 0, 1)] for j in (-1, 0, 1)] for k in (-1, 0, 1)],
                dtype=lattice.dtype,
                device=lattice.device,
                ).reshape((27, 3))    # 27, 3
            distance_shift_vectors = lattice_shift_vectors @ lattice[None, :, :]    # data_size, 27, 3
            distance_vectors = distance_vectors[:, :, :, None, :] + distance_shift_vectors[:, None, None, :, :]    # data_size, n_atoms, n_neighbors, 27, 3
            distances = torch.norm(distance_vectors, dim=-1, keepdim=True)    # data_size, n_atoms, n_atoms, 27, 1

            # expand neighbor (and neighbor mask)
            # neighbors = neighbors[:, :, :, None].tile((1, 1, 1, 27)).flatten(start_dim=2)  # B x A x Nx27
            # if neighbor_mask is not None:
            #     neighbor_mask = neighbor_mask[:, :, :, None].tile((1, 1, 1, 27)).flatten(start_dim=2)

            distance_min_index = torch.argmin(distances, dim=-2, keepdim=True)    # data_size, n_atoms, n_atoms, 1, 1
            distances = torch.gather(distances, dim=-2, index=distance_min_index).squeeze(-2)    # data_size, n_atoms, n_atoms, 1
            distance_vectors = torch.gather(distance_vectors, dim=-2, index=distance_min_index.expand(-1, -1, -1, -1, 3)).squeeze(-2)    # data_size, n_atoms, n_atoms, 3
        else:
            distances = torch.norm(distance_vectors, dim=-1, keepdim=False)   # data_size, n_atoms, n_atoms

        # if neighbor_mask is not None:
        #     # Avoid problems with zero distances in forces (instability of square
        #     # root derivative at 0) This way is neccessary, as gradients do not
        #     # work with inplace operations, such as e.g.
        #     # -> distances[mask==0] = 0.0
        #     tmp_dist = torch.zeros_like(distances)
        #     tmp_dist[neighbor_mask != 0] = distances[neighbor_mask != 0]
        #     distances = tmp_dist
        # # print(distances)
        distance_vectors = distance_vectors * neighbor_mask[:, :, :, None]

        if self.cutoff is not None:
            # # remove all neighbors beyond cutoff to save computation
            # within_cutoff = distances < self.cutoff
            # if neighbor_mask is not None:
            #     within_cutoff[neighbor_mask == 0] = False
            # neighbor_counts = torch.zeros((data_size, n_atoms), dtype=int)
            # temporal_distances = [[[] for _ in range(n_atoms)] for _ in range(data_size)]
            # temporal_distance_vec = [[[] for _ in range(n_atoms)] for _ in range(data_size)]
            # temporal_neighbor = [[[] for _ in range(n_atoms)] for _ in range(data_size)]
            # temporal_neighbor_mask = [[[] for _ in range(n_atoms)] for _ in range(data_size)]
            # for i in range(data_size):
            #     for j in range(n_atoms):
            #         neighbor_count = within_cutoff[i, j].sum()
            #         neighbor_counts[i, j] = neighbor_count
            #         temporal_distances[i][j] = distances[i, j, within_cutoff[i, j]]
            #         temporal_distance_vec[i][j] = distance_vectors[i, j, within_cutoff[i, j]]
            #         temporal_neighbor[i][j] = neighbors[i, j, within_cutoff[i, j]]
            #         temporal_neighbor_mask[i][j] = torch.tensor([1] * neighbor_count)
            # N = neighbor_counts.max()
            # distances = torch.zeros((data_size, n_atoms, N), device=positions.device)
            # distance_vectors = torch.zeros((data_size, n_atoms, N, 3), device=positions.device)
            # neighbors = torch.zeros((data_size, n_atoms, N), device=positions.device, dtype=torch.int64)
            # neighbor_mask = torch.zeros((data_size, n_atoms, N), device=positions.device)
            # for i in range(data_size):
            #     for j in range(n_atoms):
            #         distances[i, j, :neighbor_counts[i, j]] = temporal_distances[i][j]
            #         distance_vectors[i, j, :neighbor_counts[i, j]] = temporal_distance_vec[i][j]
            #         neighbors[i, j, :neighbor_counts[i, j]] = temporal_neighbor[i][j]
            #         neighbor_mask[i, j, :neighbor_counts[i, j]] = temporal_neighbor_mask[i][j]
            neighbor_mask = neighbor_mask * (distances < self.cutoff).long()    # data_size, n_atoms, n_atoms    # TODO: sparsify neighbors and neighbor_mask
            neighbors = neighbors * neighbor_mask
            distances = distances * neighbor_mask
            distance_vectors = distance_vectors * neighbor_mask[:, :, :, None]

            
        return distances, distance_vectors, neighbors, neighbor_mask
