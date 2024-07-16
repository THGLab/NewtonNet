#! /usr/bin/env python

import os
import os.path as osp
import argparse
import yaml
import json

import torch
from torch import nn

from newtonnet.models import NewtonNet
from newtonnet.train import Trainer
from newtonnet.data import RadiusGraph
from newtonnet.data import parse_train_test
from newtonnet.layers.precision import set_precison_by_string
from newtonnet.layers.activations import get_activation_by_string
from newtonnet.layers.cutoff import get_cutoff_by_string
from newtonnet.layers.representations import get_representation_by_string
from newtonnet.layers.scalers import get_scaler_by_string
from newtonnet.train.loss import get_loss_by_string
from newtonnet.train.optimizer import get_optimizer_by_string, get_scheduler_by_string
# torch.autograd.set_detect_anomaly(True)

# argument parser description
parser = argparse.ArgumentParser(
    description='This is a pacakge to train NewtonNet on a given data.',
    )
parser.add_argument(
    '-c',
    '--config',
    type=str,
    required=True,
    help='The path to the Yaml configuration file.',
    )

# define arguments
args = parser.parse_args()
config = args.config

# locate files
settings_path = os.path.abspath(config)
settings = yaml.safe_load(open(settings_path, 'r'))
script_path = os.path.abspath(__file__)
output_base_path = settings['general']['output']

# device
precision = set_precison_by_string(settings['general'].get('precision', 'float32'))
if type(settings['general'].get('device', 'cpu')) is list:
    device = [torch.device(item) for item in settings['general']['device']]
else:
    device = [torch.device(settings['general']['device'])]

# data
torch.manual_seed(settings['data'].get('random_states', 42))
transform = RadiusGraph(settings['data'].get('cutoff', 5.0))
train_gen, val_gen, test_gen, stats = parse_train_test(
    train_root=settings['data'].get('train_root', None),
    val_root=settings['data'].get('val_root', None),
    test_root=settings['data'].get('test_root', None),
    # train_properties=settings['data'].get('train_properties', ['energy', 'forces']),
    pre_transform=transform,
    transform=None,
    train_size=settings['data'].get('train_size', None),
    val_size=settings['data'].get('val_size', None),
    test_size=settings['data'].get('test_size', None),
    train_batch_size=settings['training'].get('train_batch_size', 32),
    val_batch_size=settings['training'].get('val_batch_size', 32),
    test_batch_size=settings['training'].get('test_batch_size', 32),
    )

# model
scalers = {}
for key in settings['data'].get('train_properties', ['energy', 'forces']):
    scalers[key] = get_scaler_by_string(key, stats)
distance_network = nn.ModuleDict({
    'scalednorm': get_cutoff_by_string(
        'scalednorm',
        cutoff=transform.r,
        ),
    'cutoff': get_cutoff_by_string(
        settings['model'].get('cutoff_network', 'poly'), 
        ),
    'representation': get_representation_by_string(
        settings['model'].get('representation', 'bessel'), 
        n_basis=settings['model'].get('n_basis', 20),
        ),
    })
if settings['model'].get('pretrained_model', None) is not None:
    model = torch.load(
        settings['model']['pretrained_model'], 
        map_location=device[0],
        )
else:
    model = NewtonNet(
        n_features=settings['model'].get('n_features', 128),
        distance_network=distance_network,
        n_interactions=settings['model'].get('n_interactions', 3),
        infer_properties=settings['model'].get('infer_properties', ['energy', 'forces']),
        scalers=scalers,
        device=device[0],
        )

# loss
main_loss, eval_loss = get_loss_by_string(
    settings['training'].get('loss', None),
    **settings['training'].get('loss_kwargs', {}),
    )

# optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = get_optimizer_by_string(
    settings['training'].get('optimizer', 'adam'),
    trainable_params,
    **settings['training'].get('optimizer_kwargs', {}),
    )
lr_scheduler = get_scheduler_by_string(
    settings['training'].get('lr_scheduler', 'plateau'),
    optimizer,
    **settings['training'].get('lr_scheduler_kwargs', {}),
    )

# training
trainer = Trainer(
    model=model,
    loss_fns=(main_loss, eval_loss),
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    output_base_path=output_base_path,
    script_path=script_path,
    settings_path=settings_path,
    resume_training=settings['checkpoint'].get('resume_training', None),
    checkpoint_log=settings['checkpoint'].get('log', 1),
    checkpoint_val=settings['checkpoint'].get('val', 1),
    checkpoint_test=settings['checkpoint'].get('test', 1),
    checkpoint_model=settings['checkpoint'].get('model', 1),
    verbose=settings['checkpoint'].get('verbose', False),
    device=device,
    )
trainer.train(
    train_generator=train_gen,
    val_generator=val_gen,
    test_generator=test_gen,
    epochs=settings['training'].get('epochs', 100),
    clip_grad=settings['training'].get('clip_grad', 0.0),
    )

print('done!')
