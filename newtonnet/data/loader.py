import torch
from torch import nn
from torch.utils.data import Dataset

from newtonnet.data import NeighborEnvironment
from newtonnet.layers.scalers import Normalizer


class MolecularDataset(Dataset):
    '''
    This class is a dataset for molecular data.

    Parameters:
        input (dict): The input data.
        properties (list): The properties to be predicted. Default: ['E', 'F'].
        environment (NeighborEnvironment): The environment for the dataset. Default: NeighborEnvironment().
        device (torch.device): The device for the dataset. Default: torch.device('cpu').
    '''
    def __init__(self, input, properties=['energy', 'forces'], environment=NeighborEnvironment(), device=torch.device('cpu')):
        self.properties = properties
        self._check_properties(input)
        self._check_shapes(input)
        self.positions = torch.tensor(input['positions'], dtype=torch.float)
        self.atomic_numbers = torch.tensor(input['atomic_numbers'], dtype=torch.long)
        self.distances, self.distance_vectors, self.atom_mask, self.neighbor_mask = environment.get_environment(self.positions, self.atomic_numbers)
        for property in self.properties:
            setattr(self, property, torch.tensor(input[property], dtype=torch.float))
        self.device = device
        self.normalized = False

    def _check_properties(self, input):
        for property in ['positions', 'atomic_numbers'] + self.properties:
            assert property in input.keys(), f'property {property} is not in the input data'

    def _check_shapes(self, input):
        assert input['positions'].ndim == 3, 'positions must have 3 dimensions (n_data, n_atoms, 3).'
        assert input['atomic_numbers'].ndim == 2, 'atomic_numbers must have 2 dimensions (n_data, n_atoms).'
        assert input['positions'].shape[0] == input['atomic_numbers'].shape[0], 'positions and atomic_numbers must have same dimension 0 (n_data).'
        assert input['positions'].shape[1] == input['atomic_numbers'].shape[1], 'positions and atomic_numbers must have same dimension 1 (n_atoms).'
        assert input['positions'].shape[2] == 3, 'positions must have 3 coordinates at dimension 2 (x, y, z).'
        for property in self.properties:
            assert input[property].shape[0] == input['positions'].shape[0], f'{property} and positions must have same dimension 0 (n_data).'

    def __getitem__(self, index):
        output = dict()
        for property in ['positions', 'atomic_numbers', 'distances', 'distance_vectors', 'atom_mask', 'neighbor_mask']:
            output[property] = getattr(self, property)[index].to(self.device)
        for property in self.properties:
            output[property] = getattr(self, property)[index].to(self.device)
        if self.normalized:
            for property in self.properties:
                output[property + '_normalized'] = getattr(self, property + '_normalized')[index].to(self.device)
        return output

    def __len__(self):
        return self.atomic_numbers.shape[0]
    
    def get(self, property):
        return getattr(self, property)
    
    def normalize(self, normalizers: nn.ModuleDict):
        '''
        Normalize the properties.

        Args:
            normalizers (nn.ModuleDict): The normalizers for each property.
        '''
        for property, normalizer in normalizers.items():
            setattr(self, property + '_normalized', normalizer.forward(getattr(self, property), self.atomic_numbers))
        self.normalized = True