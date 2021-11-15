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
    def __init__(self,
                 return_vecs=False,
                 normalize_vecs=False,
                 pbc=False,
                 cutoff=None):
        super(ShellProvider, self).__init__()
        self.return_vecs = return_vecs
        self.normalize_vecs = normalize_vecs
        self.epsilon = 1e-8
        self.pbc = pbc
        self.cutoff = cutoff

    def forward(self,
                atoms,
                neighbors,
                neighbor_mask=None,
                lattice=None):
        """
        The main driver to calculate distances of atoms in a shell from center atom.

        Parameters
        ----------
        atoms: torch.Tensor
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
        B, A, _ = atoms.size()
        idx_m = torch.arange(B, device=atoms.device, dtype=torch.long)[
                :, None, None
                ]

        # Get atomic positions of all neighboring indices
        ngh_atoms_xyz = atoms[idx_m, neighbors[:, :, :], :]

        # Subtract positions of central atoms to get distance vectors
        distance_vector = ngh_atoms_xyz - atoms[:, :, None, :]


        # pbc: for distance in a direction (d) and boxlength (L), d = (d + L/2) % L - L/2
        if self.pbc:
            lattice_shift_arr = torch.tensor([[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]],
                                             dtype=lattice.dtype,
                                             device=lattice.device)  # 27 x 3
            lattice_batchlast = lattice.view((-1, 3, 3)).moveaxis(0, 2) # 3 x 3 x B
            distance_shift_arr = torch.tensordot(lattice_shift_arr, lattice_batchlast, 1).moveaxis(2, 1) # 27 x B x 3
            distance_vector_pbc = distance_vector[None] + distance_shift_arr[:, :, None, None] # 27 x B x A x N x 3
            distance_vector = distance_vector_pbc.moveaxis(0, -2).flatten(start_dim=2, end_dim=3) # B x A x N*27 x 3
            distances = torch.linalg.norm(distance_vector, dim=-1)

            # expand neighbor (and neighbor mask)
            neighbors = neighbors[:, :, :, None].tile((1, 1, 1, 27)).flatten(start_dim=2)  # B x A x Nx27
            if neighbor_mask is not None:
                neighbor_mask = neighbor_mask[:, :, :, None].tile((1, 1, 1, 27)).flatten(start_dim=2)
            # distance_min_idx = torch.argmin(distances_pbc, dim=0)
            # distance_min_idx_tiled = distance_min_idx[None, ..., None].tile((1, 1, 1, 1, 3))
            # distance_vector = torch.gather(distance_vector_pbc, 0, distance_min_idx_tiled).squeeze(0)
            # distances = torch.gather(distances_pbc, 0, distance_min_idx[None]).squeeze(0)


        #     x_vec = distance_vector[:, :, :, 0]
        #     x_vec = (x_vec + 0.5 * self.box[0]) % self.box[0] - 0.5 * self.box[0]
        #     y_vec = distance_vector[:, :, :, 1]
        #     y_vec = (y_vec + 0.5 * self.box[1]) % self.box[1] - 0.5 * self.box[1]
        #     z_vec = distance_vector[:, :, :, 2]
        #     z_vec = (z_vec + 0.5 * self.box[2]) % self.box[2] - 0.5 * self.box[2]
        #     distance_vector[:, :, :, 0] = x_vec
        #     distance_vector[:, :, :, 1] = y_vec
        #     distance_vector[:, :, :, 2] = z_vec
        # # print(distance_vector)
        else:
            distances = torch.norm(distance_vector, 2, 3)   # B, A, N

        if neighbor_mask is not None:
            # Avoid problems with zero distances in forces (instability of square
            # root derivative at 0) This way is neccessary, as gradients do not
            # work with inplace operations, such as e.g.
            # -> distances[mask==0] = 0.0
            tmp_dist = torch.zeros_like(distances)
            tmp_dist[neighbor_mask != 0] = distances[neighbor_mask != 0]
            distances = tmp_dist
        # print(distances)

        if self.cutoff is not None:
            # remove all neighbors beyond cutoff to save computation
            within_cutoff = distances < self.cutoff
            if neighbor_mask is not None:
                within_cutoff[neighbor_mask == 0] = False
            neighbor_counts = torch.zeros((B, A), dtype=int)
            temporal_distances = [[[] for _ in range(A)] for _ in range(B)]
            temporal_distance_vec = [[[] for _ in range(A)] for _ in range(B)]
            temporal_neighbor = [[[] for _ in range(A)] for _ in range(B)]
            temporal_neighbor_mask = [[[] for _ in range(A)] for _ in range(B)]
            for i in range(B):
                for j in range(A):
                    neighbor_count = within_cutoff[i, j].sum()
                    neighbor_counts[i, j] = neighbor_count
                    temporal_distances[i][j] = distances[i, j, within_cutoff[i, j]]
                    temporal_distance_vec[i][j] = distance_vector[i, j, within_cutoff[i, j]]
                    temporal_neighbor[i][j] = neighbors[i, j, within_cutoff[i, j]]
                    temporal_neighbor_mask[i][j] = torch.tensor([1] * neighbor_count)
            N = neighbor_counts.max()
            distances = torch.zeros((B, A, N), device=atoms.device)
            distance_vector = torch.zeros((B, A, N, 3), device=atoms.device)
            neighbors = torch.zeros((B, A, N), device=atoms.device, dtype=torch.int64)
            neighbor_mask = torch.zeros((B, A, N), device=atoms.device)
            for i in range(B):
                for j in range(A):
                    distances[i, j, :neighbor_counts[i, j]] = temporal_distances[i][j]
                    distance_vector[i, j, :neighbor_counts[i, j]] = temporal_distance_vec[i][j]
                    neighbors[i, j, :neighbor_counts[i, j]] = temporal_neighbor[i][j]
                    neighbor_mask[i, j, :neighbor_counts[i, j]] = temporal_neighbor_mask[i][j]

        if self.return_vecs:
            tmp_distances = torch.ones_like(distances)
            tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0] + self.epsilon

            if self.normalize_vecs:
                distance_vector = distance_vector / tmp_distances[:, :, :, None]
            return distances, distance_vector, neighbors, neighbor_mask

        return distances, neighbors, neighbor_mask