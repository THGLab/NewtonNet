import os.path as osp
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
# from torch_geometric.transforms import RadiusGraph

from newtonnet.data import MolecularDataset, MolecularStatistics


def parse_train_test(
        train_root: str = None,
        val_root: str = None,
        test_root: str = None,
        train_size: int = None,
        val_size: int = None,
        test_size: int = None,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        **dataset_kwargs,
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
        dataset_kwargs (dict): The keyword arguments for MolecularDataset.

    Returns:
        train_gen (torch.utils.data.DataLoader): The training data loader.
        val_gen (torch.utils.data.DataLoader): The validation data loader.
        test_gen (torch.utils.data.DataLoader): The test data loader.
    '''

    # load data
    print('Data:')
    if train_root is not None:
        train_data = MolecularDataset(root=train_root, **dataset_kwargs)
        print(f'load {len(train_data)} data from {train_root}')
    else:
        raise ValueError('train_root must be provided')
    train_size = len(train_data) if train_size is None else train_size
    train_data, left_data = random_split(train_data, [train_size, len(train_data) - train_size])
    if val_root is not None:
        val_data = MolecularDataset(root=val_root, **dataset_kwargs)
        print(f'load {len(val_data)} data from {val_root}')
    else:
        val_data = left_data
    val_size = len(val_data) if val_size is None else val_size
    val_data, left_data = random_split(val_data, [val_size, len(val_data) - val_size])
    if test_root is not None:
        test_data = MolecularDataset(root=test_root, **dataset_kwargs)
        print(f'load {len(test_data)} data from {test_root}')
    else:
        test_data = left_data
    test_size = len(test_data) if test_size is None else test_size
    test_data, left_data = random_split(test_data, [test_size, len(test_data) - test_size])
    print(f'data size (train, val, test): {len(train_data)}, {len(val_data)}, {len(test_data)}')

    # create data loader
    train_gen = DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True)
    val_gen = DataLoader(dataset=val_data, batch_size=val_batch_size, shuffle=(len(val_data) > 0))
    test_gen = DataLoader(dataset=test_data, batch_size=test_batch_size, shuffle=(len(test_data) > 0))
    print(f'batch size (train, val, test): {train_batch_size}, {val_batch_size}, {test_batch_size}')

    # extract data stats
    stats_calc = MolecularStatistics()
    for train_batch in train_gen:
        stats = stats_calc(train_batch)
        break
    # stats['cutoff'] = train_data.dataset.cutoff
    print('stats:')
    print_stats(stats)

    return train_gen, val_gen, test_gen, stats

def print_stats(stats, level=1):
    for key, value in stats.items():
        if isinstance(value, dict):
            print('  ' * level + f'{key}:')
            print_stats(value, level + 1)
        else:
            print('  ' * level + f'{key}: {value}')
    return stats