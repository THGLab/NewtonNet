import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from newtonnet.data import MolecularDataset
from newtonnet.layers.scalers import get_normalizer_by_string


def parse_train_test(
        train_path: str = None,
        val_path: str = None,
        test_path: str = None,
        train_size: int = None,
        val_size: int = None,
        test_size: int = None,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        transform: callable = None,
        pre_transform: callable = None,
        pre_filter: callable = None,
        force_reload: bool = False,
        ):
    '''
    Parse the training, validation, and test data.

    Args:
        train_path (str): The path to the training data.
        val_path (str): The path to the validation data. If None, split from the unused training data. Default: None.
        test_path (str): The path to the test data. If None, split from the unused validation data. Default: None.
        train_size (int): The size of the training set. If None, use all available data. Default: None.
        val_size (int): The size of the validation set. If None, use all available data. Default: None.
        test_size (int): The size of the test set. If None, use all available data. Default: None.
        train_batch_size (int): The batch size for training. Default: 32.
        val_batch_size (int): The batch size for validation. Default: 32.
        test_batch_size (int): The batch size for testing. Default: 32.
        transform (callable): A function/transform that takes in a data object and returns a transformed version. The data object will be transformed before every access. Default: None.
        pre_transform (callable): A function/transform that takes in a data object and returns a transformed version. The data object will be transformed before being saved to disk. Default: None.
        pre_filter (callable): A function that takes in a data object and returns a boolean value, indicating whether the data object should be included in the final dataset. Default: None.
        force_reload (bool): Whether to re-process the dataset. Default: False.

    Returns:
        train_gen (torch.utils.data.DataLoader): The training data loader.
        val_gen (torch.utils.data.DataLoader): The validation data loader.
        test_gen (torch.utils.data.DataLoader): The test data loader.
    '''

    # load data
    print('Data:')
    if train_path is not None:
        train_data = MolecularDataset(root=train_path, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter, force_reload=force_reload)
        print(f'load {len(train_data)} data from {train_path}')
    else:
        raise ValueError('train_path must be provided')
    train_size = len(train_data) if train_size is None else train_size
    train_data, left_data = random_split(train_data, [train_size, len(train_data) - train_size])
    if val_path is not None:
        val_data = MolecularDataset(root=val_path, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter, force_reload=force_reload)
        print(f'load {len(val_data)} data from {val_path}')
    else:
        val_data = left_data
    val_size = len(val_data) if val_size is None else val_size
    val_data, left_data = random_split(val_data, [val_size, len(val_data) - val_size])
    if test_path is not None:
        test_data = MolecularDataset(root=test_path, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter, force_reload=force_reload)
        print(f'load {len(test_data)} data from {test_path}')
    else:
        test_data = left_data
    test_size = len(test_data) if test_size is None else test_size
    test_data, left_data = random_split(test_data, [test_size, len(test_data) - test_size])
    print(f'data size (train, val, test): {len(train_data)}, {len(val_data)}, {len(test_data)}')

    # create data loader
    train_gen = DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True)
    val_gen = DataLoader(dataset=val_data, batch_size=val_batch_size, shuffle=False)
    test_gen = DataLoader(dataset=test_data, batch_size=test_batch_size, shuffle=False)
    print(f'batch size (train, val, test): {train_batch_size}, {val_batch_size}, {test_batch_size}')

    # # extract data stats
    # print('embedded atomic numbers:')
    # embedded_atomic_numbers = torch.unique(train_data.dataset.atomic_numbers[train_data.indices])
    # embedded_atomic_numbers = embedded_atomic_numbers[embedded_atomic_numbers > 0]
    # print(f'  {embedded_atomic_numbers.tolist()}')
    # print('normalizers:')
    # normalizers = {}
    # for key in train_properties:
    #     normalizer = get_normalizer_by_string(
    #         key=key,
    #         data=train_data.dataset.get(key)[train_data.indices],
    #         atomic_numbers=train_data.dataset.atomic_numbers[train_data.indices],
    #         )
    #     normalizers[key] = normalizer
    #     print(f'  {key} normalizer: mean {normalizer.mean.data.tolist()}, std {normalizer.std.data.tolist()}')
    # normalizers = ModuleDict(normalizers)
    # for data in [train_data, val_data, test_data]:
    #     data.dataset.normalize(normalizers)
    # print()

    return train_gen, val_gen, test_gen
