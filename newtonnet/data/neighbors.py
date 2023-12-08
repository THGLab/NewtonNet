import torch

from newtonnet.layers.shells import ShellProvider


class NeighborEnvironment(object):
    '''
    This class finds atomic environments for each atom in the frame.

    Parameters:
        cutoff (float): The cutoff radius. Default: 5.0.
        pbc (bool): Whether to use periodic boundary conditions. Default: False.
        cell (torch.tensor): The unit cell size. Default: [[0, 0, 0], [0, 0, 0], [0, 0, 0]], i.e. no periodic boundary.

    Notes:
        Sparse tensors are not yet supported.
    '''
    def __init__(
            self, 
            cutoff: float = 5.0,
            pbc: bool = False,
            cell: torch.Tensor = torch.zeros(3, 3),
            ):
        self.cutoff = cutoff
        self.shell = ShellProvider(cutoff=cutoff, pbc=pbc, cell=cell)

    def _check_shapes(self, positions, numbers):
        assert positions.ndim == 3, 'positions must have 3 dimensions (n_data, n_atoms, 3).'
        assert numbers.ndim == 2, 'numbers must have 2 dimensions (n_data, n_atoms).'
        assert positions.shape[0] == numbers.shape[0], 'positions and numbers must have same dimension 0 (n_data).'
        assert positions.shape[1] == numbers.shape[1], 'positions and numbers must have same dimension 1 (n_atoms).'
        assert positions.shape[2] == 3, 'positions must have 3 coordinates at dimension 2 (x, y, z).'

    def get_environment(self, positions, numbers):
        '''
        This function finds atomic environments for each atom in the frame.

        Parameters:
            positions (torch.tensor): The atomic positions with shape (n_data, n_atoms, 3).
            numbers (torch.tensor): The atomic numbers with shape (n_data, n_atoms).

        Returns:
            distances (torch.tensor): The distances between atoms with shape (n_data, n_atoms, n_atoms).
            distance_vectors (torch.tensor): The distance vectors between atoms with shape (n_data, n_atoms, n_atoms, 3).
            atom_mask (torch.tensor): The mask for atoms with shape (n_data, n_atoms).
            neighbor_mask (torch.tensor): The mask for neighbors with shape (n_data, n_atoms, n_atoms).
        '''
    
        _, n_atoms, _ = positions.shape

        self._check_shapes(positions, numbers)

        # mask based on zero atomic_numbers
        atom_mask = (numbers > 0)  # data_size, n_atoms

        neighbor_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)  # data_size, n_atoms, n_atoms
        neighbor_mask[:, torch.arange(n_atoms), torch.arange(n_atoms)] = 0

        # distances
        distances, distance_vectors, neighbor_mask = self.shell(positions, neighbor_mask)
        # atom_mask = atom_mask.to_sparse()
        print(neighbor_mask.dtype)

        return distances, distance_vectors, atom_mask, neighbor_mask
    
    