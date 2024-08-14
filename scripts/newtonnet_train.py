#! /usr/bin/env python

import os
import argparse
import yaml
import json
import wandb

import torch
from torch import nn
from torch_geometric.transforms import ToDevice, Compose 

from newtonnet.models import NewtonNet
from newtonnet.train import Trainer
from newtonnet.data import RadiusGraph
from newtonnet.data import parse_train_test
from newtonnet.layers.precision import get_precison_by_string
from newtonnet.layers.activations import get_activation_by_string
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
    help='The path to the Yaml configuration file.',
    )
parser.add_argument(
    '-r',
    '--resume',
    type=str,
    help='The path to the checkpoint to resume training.',
    )

# define arguments
args = parser.parse_args()
checkpoint = args.resume
if checkpoint is None:
    config = args.config
else:
    assert args.config is None, 'Cannot resume and train from scratch at the same time.'
    configs = [file for file in os.listdir(os.path.join(checkpoint, 'run_scripts')) if any(file.endswith(ext) for ext in ['.yaml', '.yml'])]
    assert len(configs) == 1, f'Found {len(configs)} config files in {checkpoint}.'
    config = os.path.join(checkpoint, 'run_scripts', configs[0])

# locate files
settings_path = os.path.abspath(config)
settings = yaml.safe_load(open(settings_path, 'r'))
script_path = os.path.abspath(__file__)
output_base_path = settings['general']['output']

# device
precision = get_precison_by_string(settings['general']['precision'])
if type(settings['general']['device']) is list:
    device = [torch.device(item) for item in settings['general']['device']]
else:
    device = [torch.device(settings['general']['device'])]

# data
torch.manual_seed(settings['general']['seed'])
# pre_transform = Compose([
#     RadiusGraph(settings['data'].get('cutoff', 5.0)),
#     ToDevice(device[0]),
#     ])
# transform = None
transform = ToDevice(device[0])
train_gen, val_gen, test_gen, stats = parse_train_test(
    transform=transform,
    **settings['data'],
    )

# model
scalers = {key: get_scaler_by_string(key, **stat) for key, stat in stats['properties'].items()}
represenations = get_representation_by_string(
    cutoff=stats['cutoff'], 
    **settings['model'].pop('representation', {}),
    )
pretrained_model = settings['model'].pop('pretrained_model', None)
if pretrained_model is not None:
    model = torch.load(pretrained_model, map_location=device[0])
else:
    model = NewtonNet(
        representations=represenations,
        scalers=scalers,
        **settings['model'],
        )
    model.to(device[0])
    model.to(precision)

# loss
main_loss, eval_loss = get_loss_by_string(settings['training'].pop('loss', None))

# optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer, optimizer_kwargs = settings['training'].pop('optimizer', {'adam': {}}).popitem()
optimizer = get_optimizer_by_string(optimizer, trainable_params, **optimizer_kwargs)
lr_scheduler = settings['training'].pop('lr_scheduler', None).items()
lr_scheduler = get_scheduler_by_string(lr_scheduler, optimizer)

# training
wandb_kwargs = settings['training'].pop('wandb', None)
if wandb_kwargs is not None:
    wandb.login()
    wandb.init(**wandb_kwargs, config=settings)
trainer = Trainer(
    model=model,
    loss_fns=(main_loss, eval_loss),
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    output_base_path=output_base_path,
    script_path=script_path,
    settings_path=settings_path,
    device=device,
    train_generator=train_gen,
    val_generator=val_gen,
    test_generator=test_gen,
    log_wandb=wandb_kwargs is not None,
    **settings['training'],
    )
if checkpoint is not None:
    trainer.resume(checkpoint)
trainer.train()

print('done!')
