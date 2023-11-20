import torch
from torch.utils.data import Dataset, DataLoader
from newtonnet.data import ExtensiveEnvironment


class BatchDataset(Dataset):
    """
    Parameters
    ----------
    input: dict
        The dictionary of batch data in ndarray format.

    """
    def __init__(self, input):
        self.R = torch.tensor(input['R'], dtype=torch.float)
        self.Z = torch.tensor(input['Z'], dtype=torch.long)
        self.E = torch.tensor(input['E'], dtype=torch.float)
        self.F = torch.tensor(input['F'], dtype=torch.float)
        N, NM, AM = ExtensiveEnvironment().get_environment(input['R'], input['Z'])
        self.AM = torch.tensor(AM, dtype=torch.long)
        self.N = torch.tensor(N, dtype=torch.long)
        self.NM = torch.tensor(NM, dtype=torch.long)

    def __getitem__(self, index):
        output = dict()
        output['R'] = self.R[index]
        output['Z'] = self.Z[index]
        output['E'] = self.E[index]
        output['F'] = self.F[index]
        output['AM'] = self.AM[index]
        output['N'] = self.N[index]
        output['NM'] = self.NM[index]
        return output

    def __len__(self):
        return self.Z.shape[0]


def extensive_train_loader(data,
                           batch_size=32,
                           shuffle=True,
                           drop_last=False):
    """
    The main function to load and iterate data based on the extensive environment provider.

    Parameters
    ----------
    data: dict
        Dictionary containing the following keys:
            - 'R' (3D array): positions
            - 'Z' (2D array): atomic_numbers
            - 'E' (2D array): energy
            - 'F' (3D array): forces

    Yields
    -------
    BatchDataset: instance of BatchDataset with the all batch data

    """
    gen = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return gen