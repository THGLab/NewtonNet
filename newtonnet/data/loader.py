import torch
from torch.utils.data import Dataset, DataLoader
from newtonnet.data import NeighborEnvironment


class MolecularDataset(Dataset):
    """
    Parameters
    ----------
    input: dict
        The dictionary of batch data in ndarray format.

    """
    def __init__(self, input, properties=['E', 'F'], environment=NeighborEnvironment(), device: torch.device = torch.device('cpu')):
        self.R = torch.tensor(input['R'], dtype=torch.float)
        self.Z = torch.tensor(input['Z'], dtype=torch.long)
        self.D, self.V, self.AM, self.NM = environment.get_environment(self.R, self.Z)
        self.properties = properties
        for property in self.properties:
            if property in input.keys():
                setattr(self, property, torch.tensor(input[property], dtype=torch.float))
            else:
                raise ValueError(f'property {property} is not in the input data')
        self.device = device
        self.normalized = False

    def __getitem__(self, index):
        output = dict()
        if self.normalized:
            for property in ['R', 'Z', 'AM', 'NM', 'D', 'V'] + self.properties + [property + '_normalized' for property in self.properties]:
                output[property] = getattr(self, property)[index].to(self.device)
        else:
            for property in ['R', 'Z', 'AM', 'NM', 'D', 'V'] + self.properties:
                output[property] = getattr(self, property)[index].to(self.device)
        return output

    def __len__(self):
        return self.Z.shape[0]
    
    def get(self, property):
        return getattr(self, property)
    
    def normalize(self, normalizers):
        self.normalized = True
        for property, normalizer in normalizers.items():
            setattr(self, property + '_normalized', normalizer.forward(getattr(self, property), self.Z))