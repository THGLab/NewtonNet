import os
import numpy as np
import warnings
from collections import defaultdict
from numpy.lib.function_base import append
from sklearn.utils import random
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import train_test_split

from torch.utils.data import random_split
from newtonnet.data import BatchDataset
from newtonnet.data import extensive_train_loader

from ase.io import iread
import math
import pickle

def concat_listofdicts(listofdicts, axis=0):
    """

    Parameters
    ----------
    listofdicts: list
        values must be 2d arrays
    axis: int

    Returns
    -------
    dict

    """
    data = dict()
    for k in listofdicts[0].keys():
        data[k] = np.concatenate([d[k] for d in listofdicts], axis=axis)

    return data


def split(data, data_sizes):
    train_size, val_size, test_size = data_sizes
    train_size = len(data) if train_size == -1 else train_size
    val_size = len(data) if val_size == -1 else val_size
    test_size = len(data) if test_size == -1 else test_size

    train_data, val_data, test_data, _ = random_split(data, [train_size, val_size, test_size, len(data)-train_size-val_size-test_size])

    return train_data, val_data, test_data

def parse_train_test(settings, device):
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
    # meta data
    train_path = settings['data'].get('train_path', None)
    val_path = settings['data'].get('val_path', train_path)
    test_path = settings['data'].get('test_path', val_path)
    train_size = settings['data'].get('train_size', -1)
    val_size = settings['data'].get('val_size', -1)
    test_size = settings['data'].get('test_size', -1)
    if train_path == val_path == test_path:
        print('Use training data for validation and test.')
        data = BatchDataset(np.load(train_path))
        train_data, val_data, test_data = split(data, (train_size, val_size, test_size))
    elif train_path == val_path:
        print('Use training data for validation.')
        data = BatchDataset(np.load(train_path))
        train_data, val_data = split(data, (train_size, val_size, 0))
        data = BatchDataset(np.load(test_path))
        _, _, test_data = split(data, (0, 0, test_size))
    elif train_path == test_path:
        print('Use training data for test.')
        data = BatchDataset(np.load(train_path))
        train_data, _, test_data = split(data, (train_size, 0, test_size))
        data = BatchDataset(np.load(val_path))
        _, val_data, _ = split(data, (0, val_size, 0))
    elif val_path == test_path:
        print('Use validation data for test.')
        data = BatchDataset(np.load(train_path))
        train_data, _, _, = split(data, (train_size, 0, 0))
        data = BatchDataset(np.load(val_path))
        _, val_data, test_data = split(data, (0, val_size, test_size))
    else:
        print('Use separate training, validation, and test data.')
        data = BatchDataset(np.load(train_path))
        train_data, _, _ = split(data, (train_size, 0, 0))
        data = BatchDataset(np.load(val_path))
        _, val_data, _ = split(data, (0, val_size, 0))
        data = BatchDataset(np.load(test_path))
        _, _, test_data = split(data, (0, 0, test_size))

    # extract data stats
    train_E = train_data.dataset.E[train_data.indices]
    normalizer = (train_E.mean(), train_E.std())

    print(f'data size (train, val, test): {len(train_data)}, {len(val_data)}, {len(test_data)}')

    train_batch_size = settings['training'].get('train_batch_size', 32)
    val_batch_size = settings['training'].get('val_batch_size', 32)
    test_batch_size = settings['training'].get('test_batch_size', 32)

    train_gen = extensive_train_loader(
        data=train_data,
        batch_size=train_batch_size,
        shuffle=settings['training']['shuffle'],
        drop_last=settings['training']['drop_last'],
        )

    val_gen = extensive_train_loader(
        data=val_data,
        batch_size=val_batch_size,
        shuffle=settings['training']['shuffle'],
        drop_last=settings['training']['drop_last'],
        )

    test_gen = extensive_train_loader(
        data=test_data,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        )

    return train_gen, val_gen, test_gen, normalizer
