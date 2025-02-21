#! /usr/bin/env python

import argparse

from newtonnet.data import MolecularDataset, MolecularInMemoryDataset
from newtonnet.layers.precision import get_precision_by_string
# torch.autograd.set_detect_anomaly(True)

# argument parser description
parser = argparse.ArgumentParser(
    description='This is a pacakge to train NewtonNet on a given data.',
    )
parser.add_argument(
    '-r',
    '--root',
    type=str,
    help='The path to the raw data root directory.',
    )
parser.add_argument(
    '-p',
    '--precision',
    type=str,
    help='The precision of the model. Default: single.',
    default='single',
)
parser.add_argument(
    '--in-memory',
    action=argparse.BooleanOptionalAction,
    help='Whether to load the data in memory. Default: True.',
    default=True,
)

# define arguments
args = parser.parse_args()
root = args.root
precision = get_precision_by_string(args.precision)
in_memory = args.in_memory

# data
if in_memory:
    data = MolecularInMemoryDataset(root=root, precision=precision, force_reload=True)
else:
    data = MolecularDataset(root=root, precision=precision, force_reload=True)

print('done!')