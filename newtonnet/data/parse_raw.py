import numpy as np
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from newtonnet.data import MolecularDataset
from newtonnet.data import NeighborEnvironment


def split(data, data_sizes):
    train_size, val_size, test_size = data_sizes
    train_size = len(data) if train_size == -1 else train_size
    val_size = len(data) if val_size == -1 else val_size
    test_size = len(data) if test_size == -1 else test_size

    train_data, val_data, test_data, _ = random_split(data, [train_size, val_size, test_size, len(data)-train_size-val_size-test_size])

    return train_data, val_data, test_data

def parse_train_test(settings, device: torch.device = torch.device('cpu')):
    """
    Implementation based on pre-splitted training, validation, and test data.

    Parameters
    ----------
    settings: dict
        Dictionary containing the following keys:
            - 'train_path' (str, default=None): Path to the training data.
            - 'val_path' (str, default=train_path): Path to the validation data.
            - 'test_path' (str, default=val_path): Path to the test data.
            - 'train_size' (int, default=-1): Number of training data to use (-1 for all data).
            - 'val_size' (int, default=-1): Number of validation data to use (-1 for all data).
            - 'test_size' (int, default=-1): Number of test data to use (-1 for all data).
            - 'train_batch_size' (int, default=32): Batch size for training.
            - 'val_batch_size' (int, default=32): Batch size for validation.
            - 'test_batch_size' (int, default=32): Batch size for test.
            - 'random_states' (int, default=0): Random state for sampling data.

    device: list
        List of torch devices.
   
    Returns
    -------
    generator: train, val, test generators, respectively
    int: n_steps for train, val, test, respectively
    tuple: tuple of mean and standard deviation of energies in the training data

    """
    # environment
    cutoff = settings['data'].get('cutoff', 5.0)
    periodic_boundary = settings['data'].get('periodic_boundary', False)
    environment = NeighborEnvironment(cutoff=cutoff, periodic_boundary=periodic_boundary)

    # meta data
    train_path = settings['data'].get('train_path', None)
    val_path = settings['data'].get('val_path', train_path)
    test_path = settings['data'].get('test_path', val_path)
    train_size = settings['data'].get('train_size', -1)
    val_size = settings['data'].get('val_size', -1)
    test_size = settings['data'].get('test_size', -1)
    print('Data:')
    if train_path == val_path == test_path:
        print('use training data for validation and test')
        data = MolecularDataset(np.load(train_path), environment=environment, device=device)
        train_data, val_data, test_data = split(data, (train_size, val_size, test_size))
    elif train_path == val_path:
        print('use training data for validation')
        data = MolecularDataset(np.load(train_path), environment=environment, device=device)
        train_data, val_data = split(data, (train_size, val_size, 0))
        data = MolecularDataset(np.load(test_path), environment=environment, device=device)
        _, _, test_data = split(data, (0, 0, test_size))
    elif train_path == test_path:
        print('use training data for test')
        data = MolecularDataset(np.load(train_path), environment=environment, device=device)
        train_data, _, test_data = split(data, (train_size, 0, test_size))
        data = MolecularDataset(np.load(val_path), environment=environment, device=device)
        _, val_data, _ = split(data, (0, val_size, 0))
    elif val_path == test_path:
        print('use validation data for test')
        data = MolecularDataset(np.load(train_path), environment=environment, device=device)
        train_data, _, _, = split(data, (train_size, 0, 0))
        data = MolecularDataset(np.load(val_path), environment=environment, device=device)
        _, val_data, test_data = split(data, (0, val_size, test_size))
    else:
        print('use separate training, validation, and test data')
        data = MolecularDataset(np.load(train_path), environment=environment, device=device)
        train_data, _, _ = split(data, (train_size, 0, 0))
        data = MolecularDataset(np.load(val_path), environment=environment, device=device)
        _, val_data, _ = split(data, (0, val_size, 0))
        data = MolecularDataset(np.load(test_path), environment=environment, device=device)
        _, _, test_data = split(data, (0, 0, test_size))
    print(f'data size (train, val, test): {len(train_data)}, {len(val_data)}, {len(test_data)}')

    # extract data stats
    train_E = train_data.dataset.E[train_data.indices]
    normalizer = (train_E.mean(), train_E.std())
    print('normalizer: ', normalizer)
    print()

    train_batch_size = settings['training'].get('train_batch_size', 32)
    val_batch_size = settings['training'].get('val_batch_size', 32)
    test_batch_size = settings['training'].get('test_batch_size', 32)

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

    return train_gen, val_gen, test_gen, normalizer
