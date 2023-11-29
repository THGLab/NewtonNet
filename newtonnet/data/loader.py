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
    def __init__(self, input, environment=NeighborEnvironment()):
        self.R = torch.tensor(input['R'], dtype=torch.float)
        self.Z = torch.tensor(input['Z'], dtype=torch.long)
        self.E = torch.tensor(input['E'], dtype=torch.float)
        self.F = torch.tensor(input['F'], dtype=torch.float)
        self.N, self.NM, self.AM, self.D, self.V = environment.get_environment(self.R, self.Z)

    def __getitem__(self, index):
        output = dict()
        output['R'] = self.R[index]
        output['Z'] = self.Z[index]
        output['E'] = self.E[index]
        output['F'] = self.F[index]
        output['AM'] = self.AM[index]
        output['N'] = self.N[index]
        output['NM'] = self.NM[index]
        output['D'] = self.D[index]
        output['V'] = self.V[index]
        return output

    def __len__(self):
        return self.Z.shape[0]