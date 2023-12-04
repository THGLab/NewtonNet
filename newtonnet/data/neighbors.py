"""
A compilation of modules that help to find closest neighbours of each atom in a molecule.
Each Molecule is represented as a dictionary with following keys:
    - atoms: atomic positions with shape (n_atom, 3)
    - z: atomic numbers with shape (n_atoms, 1)
    - cell: unit cell with shape (3,3)
    - atom_prop: atomic property with shape (n_atoms, n_atomic_prop)
    - mol_prop: molecular property with shape (1, n_mol_prop)

"""
import torch
from newtonnet.layers.shells import ShellProvider

class NeighborEnvironment(object):
    """
    Provide atomic environment of an array of atoms and their atomic numbers.

    """
    def __init__(
            self, 
            cutoff: float = 5.0,
            periodic_boundary: bool = False,
            ):
        self.cutoff = cutoff
        self.periodic_boundary = periodic_boundary
        self.shell = ShellProvider(cutoff=cutoff, periodic_boundary=periodic_boundary)

    def _check_shapes(self, Rshape, Zshape):
        if Rshape[0] != Zshape[0]:
            msg = "@ExtensiveEnvironment: atoms and atomic_numbers must have same dimension 0 (n_data)."
            raise ValueError(msg)

        if Rshape[2] != 3:
            msg = "@ExtensiveEnvironment: atoms must have 3 coordinates at dimension 2."
            raise ValueError(msg)

        if Rshape[1] != Zshape[1]:
            msg = "@ExtensiveEnvironment: atoms and atomic_numbers must have same dimension 1 (n_atoms)."
            raise ValueError(msg)

    def get_environment(self, positions, atomic_numbers, lattice=torch.eye(3)):
        """
        This function finds atomic environments extensively for each atom in the frame.

        Parameters
        ----------
        positions: ndarray
            A 3D array of atomic positions in XYZ coordinates with shape (D, A, 3), where
            D is number of snapshots of data and A is number of atoms per data point

        atomic_numbers: ndarray
            A 2D array of atomic numbers with shape (D, A)

        Returns
        -------
        ndarray: 3D array of neighbors with shape (D, A, A)
        ndarray: 3D array of neighbor mask with shape (D, A, A)
        ndarray: 2D array of atomic mask for atomic energies (D, A)

        """
    
        _, n_atoms, _ = positions.shape

        self._check_shapes(positions.shape, atomic_numbers.shape)

        # mask based on zero atomic_numbers
        atom_mask = (atomic_numbers > 0)  # data_size, n_atoms

        neighbor_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)  # data_size, n_atoms, n_atoms
        neighbor_mask[:, torch.arange(n_atoms), torch.arange(n_atoms)] = 0

        # subtract positions of central atoms to get distance vectors
        distance_vectors = positions[:, None, :, :] - positions[:, :, None, :]    # data_size, n_atoms, n_atoms, 3
        if self.periodic_boundary:
            lattice_shift_vectors = torch.tensor(
                [[[[i, j, k] for i in (-1, 0, 1)] for j in (-1, 0, 1)] for k in (-1, 0, 1)],
                dtype=lattice.dtype,
                device=lattice.device,
                ).reshape((27, 3))    # 27, 3
            distance_shift_vectors = lattice_shift_vectors @ lattice[None, :, :]    # data_size, 27, 3
            distance_vectors = distance_vectors[:, :, :, None, :] + distance_shift_vectors[:, None, None, :, :]    # data_size, n_atoms, n_atoms, 27, 3
            distances = torch.norm(distance_vectors, dim=-1, keepdim=True)    # data_size, n_atoms, n_atoms, 27, 1
            distance_min_index = torch.argmin(distances, dim=-2, keepdim=True)    # data_size, n_atoms, n_atoms, 1, 1
            distances = torch.gather(distances, dim=-2, index=distance_min_index).squeeze(-2)    # data_size, n_atoms, n_atoms, 1
            distance_vectors = torch.gather(distance_vectors, dim=-2, index=distance_min_index.expand(-1, -1, -1, -1, 3)).squeeze(-2)    # data_size, n_atoms, n_atoms, 3
        else:
            distances = torch.norm(distance_vectors, dim=-1, keepdim=False)   # data_size, n_atoms, n_atoms
        
        # mask based on cutoff
        if self.cutoff is not None:
            neighbor_mask = neighbor_mask * (distances < self.cutoff)    # data_size, n_atoms, n_atoms
        
        # sparsify matrices
        atom_mask_sparse = atom_mask.to_sparse()
        neighbor_mask_sparse = neighbor_mask.to_sparse()
        indices = neighbor_mask_sparse.indices()
        distances_sparse = torch.sparse.FloatTensor(values=distances[neighbor_mask].flatten(), indices=indices, size=distances.shape)
        indices_expanded = (indices.repeat_interleave(3, dim=1), torch.arange(3).repeat(1, indices.shape[1]))
        indices_expanded = torch.cat(indices_expanded, dim=0)
        distance_vectors_sparse = torch.sparse.FloatTensor(values=distance_vectors[neighbor_mask].flatten(), indices=indices_expanded, size=distance_vectors.shape)

        return distances_sparse, distance_vectors_sparse, atom_mask_sparse, neighbor_mask_sparse
    
    