from tqdm import tqdm

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from newtonnet.data import MolecularStatistics


def parse_train_test(
        in_memory: bool = True,
        train_root: str = None,
        val_root: str = None,
        test_root: str = None,
        train_size: int = None,
        val_size: int = None,
        test_size: int = None,
        stats_size: int = None,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        **dataset_kwargs,
        ):
    '''
    Parse the training, validation, and test data.

    Args:
        in_memory (bool): Whether to load the data in memory. Default: True.
        train_path (str): The path to the training data.
        val_path (str): The path to the validation data. If None, split from the unused training data. Default: None.
        test_path (str): The path to the test data. If None, split from the unused validation data. Default: None.
        train_size (int): The size of the training set. If None, use all available data. Default: None.
        val_size (int): The size of the validation set. If None, use all available data. Default: None.
        test_size (int): The size of the test set. If None, use all available data. Default: None.
        train_batch_size (int): The batch size for training. Default: 32.
        val_batch_size (int): The batch size for validation. Default: 32.
        test_batch_size (int): The batch size for testing. Default: 32.
        stats_batch_size (int): The batch size for statistics calculation. Default: 32.
        dataset_kwargs (dict): The keyword arguments for MolecularDataset.

    Returns:
        train_gen (torch.utils.data.DataLoader): The training data loader.
        val_gen (torch.utils.data.DataLoader): The validation data loader.
        test_gen (torch.utils.data.DataLoader): The test data loader.
    '''
    # define dataset type
    if in_memory:
        from newtonnet.data import MolecularInMemoryDataset as MolecularDataset
    else:
        from newtonnet.data import MolecularDataset

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
    stats_gen = DataLoader(dataset=train_data, batch_size=stats_size if stats_size is not None else len(train_data), shuffle=True)
    print(f'batch size (train, val, test): {train_batch_size}, {val_batch_size}, {test_batch_size}')

    # extract data stats
    stats_calc = MolecularStatistics()
    for batch in stats_gen:
        stats = stats_calc(batch)
        break
    # print('stats:')
    # print_stats(stats)

    return train_gen, val_gen, test_gen, stats

# def process_stats(stats_raw):
#     stats = {'z': [], 'properties': {}}
#     for stat in stats_raw:
#         stats['z'].append(stat['z'])
#         for prop, prop_dict in stat['properties'].items():  # prop: 'energy', 'force', etc
#             if prop not in stats['properties']:
#                 stats['properties'][prop] = {}
#             for key, value in prop_dict.items():  # key: 'scale', 'shift'
#                 if key not in stats['properties'][prop]:
#                     stats['properties'][prop][key] = []
#                 if value.ndim > 0:
#                     value_dense = torch.full((129, ), torch.nan, dtype=value.dtype)
#                     value_dense[stat['z']] = value
#                     value = value_dense
#                 stats['properties'][prop][key].append(value)
#     stats['z'] = torch.cat(stats['z']).unique()
#     for prop, prop_dict in stats['properties'].items():
#         for key, value in prop_dict.items():
#             stats['properties'][prop][key] = torch.stack(value).nanmean(dim=0).nan_to_num()
#     return stats

# def print_stats(stats, level=1):
#     for key, value in stats.items():
#         if isinstance(value, dict):
#             print('  ' * level + f'{key}:')
#             print_stats(value, level + 1)
#         else:
#             print('  ' * level + f'{key}: {value[value != 0]}')
#     return stats