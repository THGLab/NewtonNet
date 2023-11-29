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
    
        n_data, n_atoms, _ = positions.shape

        self._check_shapes(positions.shape, atomic_numbers.shape)

        neighbors = torch.arange(n_atoms, dtype=torch.long)[None, None, :].expand(n_data, n_atoms, -1)  # n_data, n_atoms, n_atoms

        # mask based on zero atomic_numbers
        atom_mask = (atomic_numbers > 0).long()  # n_data, n_atoms

        neighbor_mask = atom_mask[:, :, None] * atom_mask[:, None, :]  # n_data, n_atoms, n_atoms
        neighbor_mask[:, torch.arange(n_atoms), torch.arange(n_atoms)] = 0
        neighbors = neighbors * neighbor_mask  # n_data, n_atoms, n_atoms

        distances, distance_vectors, neighbors, neighbor_mask = self.shell(positions, neighbors, neighbor_mask, lattice)

        return neighbors, neighbor_mask, atom_mask, distances, distance_vectors
    
    