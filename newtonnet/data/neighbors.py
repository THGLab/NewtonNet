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
            lattice: torch.Tensor = torch.eye(3) * 10.0,
            ):
        self.cutoff = cutoff
        self.shell = ShellProvider(cutoff=cutoff, periodic_boundary=periodic_boundary, lattice=lattice)

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

    def get_environment(self, positions, atomic_numbers):
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

        # distances
        distances, distance_vectors, neighbor_mask = self.shell(positions, neighbor_mask)
        # atom_mask = atom_mask.to_sparse()

        return distances, distance_vectors, atom_mask, neighbor_mask
    
    