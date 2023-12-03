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
        self.N, self.NM, self.AM, self.D, self.V = environment.get_environment(self.R, self.Z)
        self.properties = properties
        for property in self.properties:
            if property in input.keys():
                setattr(self, property, torch.tensor(input[property], dtype=torch.float))
            else:
                raise ValueError(f'property {property} is not in the input data')
        self.device = device
        # self.normalizer = None

    def __getitem__(self, index):
        output = dict()
        for property in ['R', 'Z', 'AM', 'N', 'NM', 'D', 'V'] + self.properties:
            output[property] = getattr(self, property)[index].to(self.device)
        return output

    def __len__(self):
        return self.Z.shape[0]
    
    # def set_normalizer(self, normalizer):
    #     self.normalizer = normalizer
    #     self.E = self.normalizer.forward(self.E, self.Z, scale=True, shift=True)
    #     self.F = self.normalizer.forward(self.F, self.Z, scale=True, shift=False)