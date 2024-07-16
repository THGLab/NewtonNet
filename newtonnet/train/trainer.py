import os
import pandas as pd
import torch
import time
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        resume_training (str): The path to a checkpoint to resume training from. Default: False.
        checkpoint_log (int): The interval in epochs for logging the training progress. Default: 1.
        checkpoint_val (int): The interval in epochs for validation. Default: 1.
        checkpoint_test (int): The interval in epochs for testing. Default: 1.
        checkpoint_model (int): The interval in epochs for saving the model (must be validated first). Default: 1.
        verbose (bool): Whether to print the training progress. Default: False.
        device (torch.device): The device to use for training. Default: cpu
    '''
    def __init__(
            self,
            model: nn.Module,
            loss_fns: (nn.Module, nn.Module) = None,
            optimizer: optim.Optimizer = None,
            lr_scheduler: optim.lr_scheduler._LRScheduler = None,
            output_base_path: str = None,
            script_path: str = None,
            settings_path: str = None,
            resume_training: str = None,
            checkpoint_log: int = 1,
            checkpoint_val: int = 1,
            checkpoint_test: int = 1,
            checkpoint_model: int = 1,
            verbose: bool = False,
            device: torch.device = torch.device('cpu'),
            ):
        super(Trainer, self).__init__()
        
        # training parameters
        self.model = model
        self.print_layers()
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.main_loss, self.eval_loss = loss_fns or get_loss_by_string()
        self.optimizer = optimizer or optim.Adam(trainable_params)
        self.lr_scheduler = lr_scheduler or ReduceLROnPlateau(self.optimizer)
        self.best_val_loss = torch.inf
        self.device = device
        self.multi_gpu = True if type(device) is list and len(device) > 1 else False

        # outputs
        self.make_subdirs(output_base_path, script_path, settings_path)
        self.verbose = verbose

        # checkpoints
        self.check_log = checkpoint_log
        self.check_val = checkpoint_val
        self.check_test = checkpoint_test
        self.check_model = checkpoint_model

        # checkpoints
        if resume_training is not None:
            checkpoint = torch.load(os.path.join(resume_training, 'models/train_state.tar'))
            self.epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_loss = checkpoint['best_val_loss']
            self.log = pd.read_csv(os.path.join(resume_training, 'log.csv'))
        else:
            self.epoch = -1
            self.log = pd.DataFrame()
            self.log['epoch'] = None
            for phase in ('train', 'val', 'test'):
                self.log[f'{phase}_loss'] = None
                for loss_fn in self.eval_loss.loss_fns:
                    self.log[f'{phase}_{loss_fn.name}'] = None
            self.log['lr'] = None
            self.log['time'] = None

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
            # # shorten names
            # name = name.replace('node_embedding', 'emb')
            # name = name.replace('message_passing_layers', 'mes')
            # name = name.replace('invariant', 'inv')
            # name = name.replace('equivariant', 'eq')
            # name = name.replace('message', 'mes')
            # name = name.replace('coefficient', 'coeff')
            # name = name.replace('feature', 'feat')
            # name = name.replace('selfupdate', 'upd')
            # name = name.replace('property', 'prop')
            # name = name.replace('prediction', 'pred')
            # name = name.replace('weight', 'w')
            # name = name.replace('bias', 'b')
            # name = name.replace('mean', 'm')
            # name = name.replace('std', 's')
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

        for epoch in tqdm(range(epochs + 1)):
            # skip epochs if resuming training
            if epoch <= self.epoch:
                for train_step, train_batch in enumerate(train_generator):
                    pass
                if epoch % self.check_val == 0:
                    for val_step, val_batch in enumerate(val_generator):
                        pass
                if epoch % self.check_test == 0:
                    for test_step, test_batch in enumerate(test_generator):
                        pass
                continue

            # training
            t0 = time.time()
            self.model.train()

            for train_step, train_batch in enumerate(train_generator):
                train_losses = {}
                self.optimizer.zero_grad()
                preds = self.model(train_batch.z, train_batch.pos, train_batch.edge_index, train_batch.batch)
                main_loss = self.main_loss(preds, train_batch)
                main_loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                self.optimizer.step()
                main_loss = main_loss.detach().item()
                train_losses['loss'] = main_loss

            # plots
            if epoch % self.check_log == 0:
                self.plot_grad_flow(epoch)

            # validation
            if epoch % self.check_val == 0:
                self.model.eval()
                val_losses = {}

                for val_step, val_batch in enumerate(val_generator):
                    preds = self.model(val_batch.z, val_batch.pos, val_batch.edge_index, val_batch.batch)
                    main_loss = self.main_loss(preds, val_batch)
                    val_losses['loss'] = val_losses.get('loss', 0.0) + main_loss.detach().item()
                    eval_loss = self.eval_loss(preds, val_batch)
                    for key, value in eval_loss.items():
                        val_losses[key] = val_losses.get(key, 0.0) + value.detach().item()

                for key, value in val_losses.items():
                    val_losses[key] /= len(val_generator)

                # best model
                if epoch % self.check_model == 0:
                    if val_losses['loss'] < self.best_val_loss:
                        self.best_val_loss = val_losses['loss']
                        if self.multi_gpu:
                            save_model = self.model.module
                        else:
                            save_model = self.model
                        torch.save(save_model, os.path.join(self.model_path, 'best_model.pt'))

                # learning rate decay
                self.lr_scheduler.step(val_losses['loss'])

            # save test predictions
            test_losses = {}
            if epoch % self.check_test == 0:
                self.model.eval()

                for test_step, test_batch in enumerate(test_generator):
                    preds = self.model(test_batch.z, test_batch.pos, test_batch.edge_index, test_batch.batch)
                    main_loss = self.main_loss(preds, test_batch)
                    test_losses['loss'] = test_losses.get('loss', 0.0) + main_loss.detach().item()
                    eval_loss = self.eval_loss(preds, test_batch)
                    for key, value in eval_loss.items():
                        test_losses[key] = test_losses.get(key, 0.0) + value.detach().item()

                for key, value in test_losses.items():
                    test_losses[key] /= len(test_generator)

            # # loss force weight decay
            # self.main_loss.force_loss_decay()

            # checkpoint
            if epoch % self.check_log == 0:
                checkpoint = {}
                checkpoint.update({'epoch': epoch})
                checkpoint.update({f'train_{key}': value for key, value in train_losses.items()})
                checkpoint.update({f'val_{key}': value for key, value in val_losses.items()})
                checkpoint.update({f'test_{key}': value for key, value in test_losses.items()})
                checkpoint.update({'lr': self.optimizer.param_groups[0]['lr']})
                checkpoint.update({'time': time.time() - t0})
                self.log.loc[epoch] = checkpoint
                self.log.to_csv(os.path.join(self.output_path, 'log.csv'), index=False)
                print(f'[{epoch}/{epochs}]', end=' ')
                print(*[f'{key}: {value:.5f}' for key, value in checkpoint.items() if key != 'epoch'], sep=' - ')
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.lr_scheduler.state_dict(),
                        'best_val_loss': self.best_val_loss,
                    }, os.path.join(self.model_path, 'train_state.tar'))