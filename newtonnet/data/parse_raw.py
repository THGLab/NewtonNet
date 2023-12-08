import numpy as np

import torch
from torch.nn import ModuleDict
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from newtonnet.data import MolecularDataset, NeighborEnvironment
from newtonnet.layers.scalers import get_normalizer_by_string


def split(data, data_sizes):
    '''
    Split data into train, val, and test sets.

    Args:
        data (torch.utils.data.Dataset): The data to be split.
        data_sizes (tuple): The sizes of the train, val, and test sets.

    Returns:
        train_data (torch.utils.data.Subset): The training data.
        val_data (torch.utils.data.Subset): The validation data.
        test_data (torch.utils.data.Subset): The test data.
    '''
    train_size, val_size, test_size = data_sizes
    train_size = len(data) if train_size == -1 else train_size
    val_size = len(data) if val_size == -1 else val_size
    test_size = len(data) if test_size == -1 else test_size

    train_data, val_data, test_data, _ = random_split(data, [train_size, val_size, test_size, len(data)-train_size-val_size-test_size])

    return train_data, val_data, test_data

def parse_train_test(
        train_path: str = None,
        val_path: str = None,
        test_path: str = None,
        train_size: int = -1,
        val_size: int = -1,
        test_size: int = -1,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        properties: list = ['energy', 'forces'],
        pbc: bool = False,
        cell: torch.tensor = torch.zeros(3, 3),
        cutoff: float = 5.0,
        device: torch.device = torch.device('cpu'),
        precision: torch.dtype = torch.float32,
        ):
    '''
    Parse the training, validation, and test data.

    Args:
        train_path (str): The path to the training data.
        val_path (str): The path to the validation data. If None, use the training data. Default: None.
        test_path (str): The path to the test data. If None, use the validation data. Default: None.
        train_size (int): The size of the training set. If -1, use all data. Default: -1.
        val_size (int): The size of the validation set. If -1, use all data. Default: -1.
        test_size (int): The size of the test set. If -1, use all data. Default: -1.
        train_batch_size (int): The batch size for training. Default: 32.
        val_batch_size (int): The batch size for validation. Default: 32.
        test_batch_size (int): The batch size for testing. Default: 32.
        properties (list): The properties to be trained. Default: ['energy', 'forces'].
        pbc (bool): Whether to use periodic boundary. Default: False.
        cell (torch.tensor): The unit cell size. Default: [[0, 0, 0], [0, 0, 0], [0, 0, 0]].
        cutoff (float): The cutoff radius. Default: 5.0.
        device (torch.device): The device to use. Default: torch.device('cpu').
        precision (torch.dtype): The precision of the model. Default: torch.float32.

    Returns:
        train_gen (torch.utils.data.DataLoader): The training data loader.
        val_gen (torch.utils.data.DataLoader): The validation data loader.
        test_gen (torch.utils.data.DataLoader): The test data loader.
        embedded_atomic_numbers (torch.tensor): The embedded atomic numbers.
        normalizers (nn.ModuleDict): The normalizers for each property.
        shell (nn.Module): The neighbor environment.
    '''
    # environment
    print('Environment:')
    environment = NeighborEnvironment(cutoff=cutoff, pbc=pbc, cell=cell)
    shell = environment.shell
    print(f'distance cutoff: {environment.shell.cutoff}')
    print(f'periodic boundary: {environment.shell.pbc}')
    if environment.shell.pbc:
        print(f'  unit cell size: {environment.shell.cell.tolist()}')
    print()

    print('Data:')
    # meta data
    assert train_path is not None, 'train_path is required'
    train_path = train_path
    val_path = val_path or train_path
    test_path = test_path or val_path

    # load data
    if train_path == val_path == test_path:
        print('use training data for validation and test')
        train_data = MolecularDataset(np.load(train_path), properties=properties, environment=environment, device=device, precision=precision)
        train_data, val_data, test_data = split(train_data, (train_size, val_size, test_size))
    elif train_path == val_path:
        print('use training data for validation')
        train_data = MolecularDataset(np.load(train_path), properties=properties, environment=environment, device=device, precision=precision)
        train_data, val_data = split(train_data, (train_size, val_size, 0))
        test_data = MolecularDataset(np.load(test_path), properties=properties, environment=environment, device=device, precision=precision)
        _, _, test_data = split(test_data, (0, 0, test_size))
    elif train_path == test_path:
        print('use training data for test')
        train_data = MolecularDataset(np.load(train_path), properties=properties, environment=environment, device=device, precision=precision)
        train_data, _, test_data = split(train_data, (train_size, 0, test_size))
        val_data = MolecularDataset(np.load(val_path), properties=properties, environment=environment, device=device, precision=precision)
        _, val_data, _ = split(val_data, (0, val_size, 0))
    elif val_path == test_path:
        print('use validation data for test')
        train_data = MolecularDataset(np.load(train_path), properties=properties, environment=environment, device=device, precision=precision)
        train_data, _, _, = split(train_data, (train_size, 0, 0))
        val_data = MolecularDataset(np.load(val_path), properties=properties, environment=environment, device=device, precision=precision)
        _, val_data, test_data = split(val_data, (0, val_size, test_size))
    else:
        print('use separate training, validation, and test data')
        train_data = MolecularDataset(np.load(train_path), properties=properties, environment=environment, device=device, precision=precision)
        train_data, _, _ = split(train_data, (train_size, 0, 0))
        val_data = MolecularDataset(np.load(val_path), properties=properties, environment=environment, device=device, precision=precision)
        _, val_data, _ = split(val_data, (0, val_size, 0))
        test_data = MolecularDataset(np.load(test_path), properties=properties, environment=environment, device=device, precision=precision)
        _, _, test_data = split(test_data, (0, 0, test_size))
    print(f'data size (train, val, test): {len(train_data)}, {len(val_data)}, {len(test_data)}')

    # create data loader
    train_gen = DataLoader(
        dataset=train_data,
        batch_size=train_batch_size,
        shuffle=True,
        )
    val_gen = DataLoader(
        dataset=val_data,
        batch_size=val_batch_size,
        shuffle=False,
        )
    test_gen = DataLoader(
        dataset=test_data,
        batch_size=test_batch_size,
        shuffle=False,
        )
    print(f'batch size (train, val, test): {train_batch_size}, {val_batch_size}, {test_batch_size}')

    # extract data stats
    print('embedded atomic numbers:')
    embedded_atomic_numbers = torch.unique(train_data.dataset.atomic_numbers[train_data.indices])
    embedded_atomic_numbers = embedded_atomic_numbers[embedded_atomic_numbers > 0]
    print(f'  {embedded_atomic_numbers.tolist()}')
    print('normalizers:')
    normalizers = {}
    for property in properties:
        normalizer = get_normalizer_by_string(
            key=property,
            data=train_data.dataset.get(property)[train_data.indices],
            atomic_numbers=train_data.dataset.atomic_numbers[train_data.indices],
            )
        normalizers[property] = normalizer
        print(f'  {property} normalizer: mean {normalizer.mean.data.tolist()}, std {normalizer.std.data.tolist()}')
    normalizers = ModuleDict(normalizers)
    for data in [train_data, val_data, test_data]:
        data.dataset.normalize(normalizers)
    print()

    return train_gen, val_gen, test_gen, embedded_atomic_numbers, normalizers, shell
