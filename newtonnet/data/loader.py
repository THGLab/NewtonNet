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
    def __init__(self, input, environment=NeighborEnvironment(), device: torch.device = torch.device('cpu')):
        self.R = torch.tensor(input['R'], dtype=torch.float)
        self.Z = torch.tensor(input['Z'], dtype=torch.long)
        self.E = torch.tensor(input['E'], dtype=torch.float)
        self.F = torch.tensor(input['F'], dtype=torch.float)
        self.N, self.NM, self.AM, self.D, self.V = environment.get_environment(self.R, self.Z)
        self.device = device

    def __getitem__(self, index):
        output = dict()
        output['R'] = self.R[index].to(self.device)
        output['Z'] = self.Z[index].to(self.device)
        output['E'] = self.E[index].to(self.device)
        output['F'] = self.F[index].to(self.device)
        output['AM'] = self.AM[index].to(self.device)
        output['N'] = self.N[index].to(self.device)
        output['NM'] = self.NM[index].to(self.device)
        output['D'] = self.D[index].to(self.device)
        output['V'] = self.V[index].to(self.device)
        return output

    def __len__(self):
        return self.Z.shape[0]