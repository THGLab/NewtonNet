import numpy as np
import torch
from torch.nn import ModuleDict
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from newtonnet.data import MolecularDataset, NeighborEnvironment
from newtonnet.layers.scalers import Normalizer


def split(data, data_sizes):
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
        properties: list = ['E', 'F'],
        periodic_boundary: bool = False,
        lattice: torch.tensor = torch.eye(3, dtype=torch.float) * 10.0,
        cutoff: float = 5.0,
        device: torch.device = torch.device('cpu'),
        ):
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
            - 'random_states' (int, default=42): Random state for sampling data.

    device: list
        List of torch devices.
   
    Returns
    -------
    generator: train, val, test generators, respectively
    int: n_steps for train, val, test, respectively
    tuple: tuple of mean and standard deviation of energies in the training data

    """
    # environment
    environment = NeighborEnvironment(cutoff=cutoff, periodic_boundary=periodic_boundary, lattice=lattice)
    shell = environment.shell

    # meta data
    if train_path is None:
        raise ValueError('train_path is required')
    if val_path is None:
        val_path = train_path
    if test_path is None:
        test_path = val_path

    # load data
    print('Data:')
    if train_path == val_path == test_path:
        print('  use training data for validation and test')
        train_data = MolecularDataset(np.load(train_path), properties=properties, environment=environment, device=device)
        train_data, val_data, test_data = split(train_data, (train_size, val_size, test_size))
    elif train_path == val_path:
        print('  use training data for validation')
        train_data = MolecularDataset(np.load(train_path), properties=properties, environment=environment, device=device)
        train_data, val_data = split(train_data, (train_size, val_size, 0))
        test_data = MolecularDataset(np.load(test_path), properties=properties, environment=environment, device=device)
        _, _, test_data = split(test_data, (0, 0, test_size))
    elif train_path == test_path:
        print('  use training data for test')
        train_data = MolecularDataset(np.load(train_path), properties=properties, environment=environment, device=device)
        train_data, _, test_data = split(train_data, (train_size, 0, test_size))
        val_data = MolecularDataset(np.load(val_path), properties=properties, environment=environment, device=device)
        _, val_data, _ = split(val_data, (0, val_size, 0))
    elif val_path == test_path:
        print('  use validation data for test')
        train_data = MolecularDataset(np.load(train_path), properties=properties, environment=environment, device=device)
        train_data, _, _, = split(train_data, (train_size, 0, 0))
        val_data = MolecularDataset(np.load(val_path), properties=properties, environment=environment, device=device)
        _, val_data, test_data = split(val_data, (0, val_size, test_size))
    else:
        print('  use separate training, validation, and test data')
        train_data = MolecularDataset(np.load(train_path), properties=properties, environment=environment, device=device)
        train_data, _, _ = split(train_data, (train_size, 0, 0))
        val_data = MolecularDataset(np.load(val_path), properties=properties, environment=environment, device=device)
        _, val_data, _ = split(val_data, (0, val_size, 0))
        test_data = MolecularDataset(np.load(test_path), properties=properties, environment=environment, device=device)
        _, _, test_data = split(test_data, (0, 0, test_size))
    print(f'  data size (train, val, test): {len(train_data)}, {len(val_data)}, {len(test_data)}')

    # extract data stats
    print('  embedded atomic numbers:')
    embedded_atomic_numbers = torch.unique(train_data.dataset.Z[train_data.indices])
    embedded_atomic_numbers = embedded_atomic_numbers[embedded_atomic_numbers > 0]
    print(f'    {embedded_atomic_numbers.tolist()}')
    print('  normalizers:')
    normalizers = {}
    for property in properties:
        normalizer = Normalizer(
            data=train_data.dataset.get(property)[train_data.indices],
            atomic_numbers=train_data.dataset.Z[train_data.indices],
            trainable=False,
            )
        normalizers[property] = normalizer
        print(f'    {property} normalizer: mean {normalizer.mean.data.tolist()}, std {normalizer.std.data.tolist()}')
    normalizers = ModuleDict(normalizers)
    for data in [train_data, val_data, test_data]:
        data.dataset.normalize(normalizers)
    print()

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

    return train_gen, val_gen, test_gen, embedded_atomic_numbers, normalizers, shell
