import os
import pandas as pd
import torch
import time
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

import torch
from torch import nn, optim
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler

from newtonnet.train.loss import get_loss_by_string


class Trainer(object):
    '''
    Trainer class for NewtonNet.

    Parameters:
        model (nn.Module): The model to train.
        loss_fns (nn.Module, nn.Module): The loss functions to use for training and evaluation. Default: None.
        optimizer (optim.Optimizer): The optimizer to use for training. Default: Adam.
        lr_scheduler (optim.lr_scheduler._LRScheduler): The learning rate scheduler to use for training. Default: ReduceLROnPlateau.
        output_base_path (str): The base path for the output directory.
        script_path (str): The path to the script that was used to start the training.
        settings_path (str): The path to the settings file that was used to start the training.
        resume_from (str): The path to a checkpoint to resume training from. Default: False.
        check_log (int): The interval in epochs for logging the training progress. Default: 1.
        check_val (int): The interval in epochs for validation. Default: 1.
        check_test (int): The interval in epochs for testing. Default: 1.
        check_model (int): The interval in epochs for saving the model (must be validated first). Default: 1.
        verbose (bool): Whether to print the training progress. Default: False.
        device (torch.device): The device to use for training. Default: cpu
    '''
    def __init__(
            self,
            model: nn.Module,
            loss_fns: tuple[nn.Module] = None,
            optimizer: optim.Optimizer = None,
            lr_scheduler: optim.lr_scheduler._LRScheduler = None,
            output_base_path: str = None,
            script_path: str = None,
            settings_path: str = None,
            resume_from: str = None,
            check_log: int = 1,
            check_val: int = 1,
            check_test: int = 1,
            check_model: int = 1,
            verbose: bool = False,
            device: torch.device = torch.device('cpu'),
            ):
        super(Trainer, self).__init__()
        
        # training parameters
        self.model = model
        self.main_loss, self.eval_loss = loss_fns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.best_val_loss = torch.inf
        self.device = device
        self.multi_gpu = True if type(device) is list and len(device) > 1 else False

        # outputs
        self.make_subdirs(output_base_path, script_path, settings_path)
        self.verbose = verbose

        # checkpoints
        self.check_log = check_log
        self.check_val = check_val
        self.check_test = check_test
        self.check_model = check_model

        # checkpoints
        if resume_from is not None:
            checkpoint = torch.load(os.path.join(resume_from, 'models', 'train_state.tar'))
            self.start_epoch = checkpoint['epoch'] + 1
            self.start_step = checkpoint['step'] + 1
            self.model = checkpoint['model']
            self.optimizer = checkpoint['optimizer']
            self.lr_scheduler = checkpoint['scheduler']
            self.best_val_loss = checkpoint['best_val_loss']
            self.log = pd.read_csv(os.path.join(resume_from, 'log.csv'))
        else:
            self.start_epoch = 0
            self.start_step = 0
            self.log = pd.DataFrame()
        self.print_layers()

    def make_subdirs(self, output_base_path, script_path, settings_path):
        assert output_base_path is not None, 'output_base_path must be specified'
        assert script_path is not None, 'script_path must be specified'
        assert settings_path is not None, 'settings_path must be specified'

        # create output directory
        path_iter = 1
        output_path = os.path.join(output_base_path, f'training_{path_iter}')
        while os.path.exists(output_path):
            path_iter += 1
            output_path = os.path.join(output_base_path, f'training_{path_iter}')
        os.makedirs(output_path)
        self.output_path = output_path

        # create subdirectory for run scripts
        script_out = os.path.join(self.output_path, 'run_scripts')
        os.makedirs(script_out)
        shutil.copyfile(script_path, os.path.join(script_out,os.path.basename(script_path)))
        shutil.copyfile(settings_path, os.path.join(script_out,os.path.basename(settings_path)))

        # create subdirectory for computation graph
        self.graph_path = os.path.join(self.output_path, 'graph')
        os.makedirs(self.graph_path)

        # create subdirectory for models
        self.model_path = os.path.join(self.output_path, 'models')
        os.makedirs(self.model_path)

    def print_layers(self):
        print('Model:')
        print(self.model)
        print(f'total trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        print()

    def plot_grad_flow(self, epoch):
        grads = []
        layers = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is not None:
                layers.append(name)
                grads.append(parameter.grad.detach().abs().mean().cpu())
            else:
                # raise ValueError(f'parameter {name} has no gradient')
                pass

        plt.figure(figsize=(16, 3))
        plt.plot(grads, '-', color='tab:blue')
        plt.xticks(range(len(layers)), layers, rotation='vertical')
        plt.xlim(-1, len(layers))
        plt.yscale('log')
        plt.xlabel('Layers')
        plt.ylabel('Average gradients')
        plt.title(f'Gradient flow - Epoch {epoch}')
        plt.grid(True)
        plt.savefig(os.path.join(self.graph_path, f'grad_flow_{epoch}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def local_log(self, log):
        log = pd.DataFrame(log, index=[0])
        self.log = pd.concat([self.log, log], ignore_index=True)
        self.log.to_csv(os.path.join(self.output_path, 'log.csv'), index=False)

    def train(
            self,
            train_generator,
            val_generator,
            test_generator,
            epochs=100,
            clip_grad=0,
            ):
        
        self.model.to(self.device[0])
        if self.multi_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.device)                

        step = self.start_step
        for epoch in tqdm(range(self.start_epoch, epochs + 1)):
            train_log, val_log, test_log = {}, {}, {}

            # training
            self.model.train()

            for train_batch in train_generator:
                self.optimizer.zero_grad()
                preds = self.model(train_batch.z, train_batch.pos, train_batch.edge_index, train_batch.batch)
                main_loss = self.main_loss(preds, train_batch)
                main_loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                self.optimizer.step()
                main_loss = main_loss.detach().item()
                eval_loss = self.eval_loss(preds, train_batch)
                wandb.log({'epoch': epoch, 'step': step, 'train_loss': main_loss, 'lr': self.optimizer.param_groups[0]['lr']} | {f'train_{key}': value.detach().item() for key, value in eval_loss.items()})
                train_log['train_loss'] = train_log.get('train_loss', 0.0) + main_loss
                for key, value in eval_loss.items():
                    train_log[f'train_{key}'] = train_log.get(f'train_{key}', 0.0) + value.detach().item()
                step += 1
            for key, value in train_log.items():
                train_log[key] /= len(train_generator)

            # validation
            if epoch % self.check_val == 0:
                self.model.eval()

                for val_batch in val_generator:
                    preds = self.model(val_batch.z, val_batch.pos, val_batch.edge_index, val_batch.batch)
                    main_loss = self.main_loss(preds, val_batch)
                    val_log['val_loss'] = val_log.get('val_loss', 0.0) + main_loss.detach().item()
                    eval_loss = self.eval_loss(preds, val_batch)
                    for key, value in eval_loss.items():
                        val_log[f'val_{key}'] = val_log.get(f'val_{key}', 0.0) + value.detach().item()

                for key, value in val_log.items():
                    val_log[key] /= len(val_generator)
                wandb.log({'epoch': epoch, 'step': step} | val_log)

            # save test predictions
            if epoch % self.check_test == 0:
                self.model.eval()

                for test_batch in test_generator:
                    preds = self.model(test_batch.z, test_batch.pos, test_batch.edge_index, test_batch.batch)
                    main_loss = self.main_loss(preds, test_batch)
                    test_log['test_loss'] = test_log.get('test_loss', 0.0) + main_loss.detach().item()
                    eval_loss = self.eval_loss(preds, test_batch)
                    for key, value in eval_loss.items():
                        test_log[f'test_{key}'] = test_log.get(f'test_{key}' + key, 0.0) + value.detach().item()

                for key, value in test_log.items():
                    test_log[key] /= len(test_generator)
                wandb.log({'epoch': epoch, 'step': step} | test_log)

            # best model
            if epoch % self.check_model == 0:
                if val_log['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_log['val_loss']
                    if self.multi_gpu:
                        save_model = self.model.module
                    else:
                        save_model = self.model
                    torch.save(save_model, os.path.join(self.model_path, 'best_model.pt'))

            # plots
            if epoch % self.check_log == 0:
                self.plot_grad_flow(epoch)
                self.local_log({'epoch': epoch, 'step': step, 'lr': self.optimizer.param_groups[0]['lr']} | train_log | val_log | test_log)

            # learning rate decay
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                if 'val_loss' in val_log:
                    self.lr_scheduler.step(val_log['val_loss'])
            elif isinstance(self.lr_scheduler, LRScheduler):
                self.lr_scheduler.step()

            # # loss force weight decay
            # self.main_loss.force_loss_decay()

            # checkpoint
            if epoch % self.check_log == 0:
                torch.save({
                        'epoch': epoch,
                        'step': step,
                        'model': self.model,
                        'optimizer': self.optimizer,
                        'scheduler': self.lr_scheduler,
                        'best_val_loss': self.best_val_loss,
                    }, os.path.join(self.model_path, 'train_state.tar'))