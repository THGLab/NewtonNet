import os
import pandas as pd
import torch
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel


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
        checkpoint (dict): The checkpoint settings.
            check_log (int): The interval in epochs for logging the training progress. Default: 1.
            check_val (int): The interval in epochs for validation. Default: 1.
            check_test (int): The interval in epochs for testing. Default: 1.
            check_model (int): The interval in epochs for saving the model (must be validated first). Default: 1.
        device (torch.device): The device to use for training. Default: cpu
        train_generator (DataLoader): The training data generator. Default: None.
        val_generator (DataLoader): The validation data generator. Default: None.
        test_generator (DataLoader): The test data generator. Default: None.
        epochs (int): The number of epochs to train. Default: 100.
        clip_grad (float): The gradient clipping value. Default: 0.0.
        log_wandb (bool): Whether to use wandb for logging. Default: False.
    '''
    def __init__(
            self,
            model: nn.Module,
            loss_fns: tuple[nn.Module] = None,
            optimizer: optim.Optimizer = None,
            lr_scheduler: LRScheduler = None,
            output_base_path: str = None,
            script_path: str = None,
            settings_path: str = None,
            checkpoint: dict = {},
            device: torch.device = torch.device('cpu'),
            train_generator: DataLoader = None,
            val_generator: DataLoader = None,
            test_generator: DataLoader = None,
            epochs: int = 100,
            clip_grad: float = 0.0,
            log_wandb: bool = False,
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
        self.model.to(self.device[0])
        if self.multi_gpu:
            self.model = DataParallel(self.model, device_ids=self.device)
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.start_epoch = 0
        self.start_step = 0
        self.epochs = epochs
        self.clip_grad = clip_grad
        self.log = pd.DataFrame()
        self.log_wandb = log_wandb

        # outputs
        if output_base_path is not None:
            self.make_subdirs(output_base_path, script_path, settings_path)
        else:
            self.output_path = None
            # self.graph_path = None
            self.model_path = None

        # checkpoints
        self.check_log = checkpoint.get('check_log', 1)
        self.check_val = checkpoint.get('check_val', 1)
        self.check_test = checkpoint.get('check_test', 1)
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
        print(f'Output directory: {output_path}')

        # create subdirectory for run scripts
        script_out = os.path.join(self.output_path, 'run_scripts')
        os.makedirs(script_out)
        shutil.copyfile(script_path, os.path.join(script_out, os.path.basename(script_path)))
        shutil.copyfile(settings_path, os.path.join(script_out, os.path.basename(settings_path)))

        # create subdirectory for computation graph
        # self.graph_path = os.path.join(self.output_path, 'graph')
        # os.makedirs(self.graph_path)

        # create subdirectory for models
        self.model_path = os.path.join(self.output_path, 'models')
        os.makedirs(self.model_path)

    def resume(self, checkpoint):
        shutil.copyfile(
            os.path.join(checkpoint, 'models', 'train_state.pt'),
            os.path.join(self.output_path, 'models', 'train_state.pt')
        )
        shutil.copyfile(
            os.path.join(checkpoint, 'models', 'best_model.pt'),
            os.path.join(self.output_path, 'models', 'best_model.pt')
        )
        shutil.copyfile(
            os.path.join(checkpoint, 'log.csv'),
            os.path.join(self.output_path, 'log.csv')
        )
        train_state = torch.load(os.path.join(self.output_path, 'models', 'train_state.pt'))
        self.start_epoch = train_state['epoch'] + 1
        self.start_step = train_state['step']
        self.model.load_state_dict(train_state['model_state_dict'])
        self.optimizer.load_state_dict(train_state['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(train_state['scheduler_state_dict'])
        self.best_val_loss = train_state['best_val_loss']
        torch.set_rng_state(train_state['rng_state'])
        self.log = pd.read_csv(os.path.join(self.output_path, 'log.csv'))

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

    def train(self):       
        step = self.start_step
        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            log_one_epoch = {'epoch': epoch, 'lr': self.optimizer.param_groups[0]['lr']}

            # training
            self.model.train()
            train_log = self.run_one_epoch(self.train_generator, step=True)
            step += len(self.train_generator)
            log_one_epoch['step'] = step
            log_one_epoch = log_one_epoch | {f'train_{key}': value for key, value in train_log.items()}

            # validation
            if epoch % self.check_val == 0 and self.val_generator is not None:
                self.model.eval()
                val_log = self.run_one_epoch(self.val_generator, step=False)
                log_one_epoch = log_one_epoch | {f'val_{key}': value for key, value in val_log.items()}

            # save test predictions
            if epoch % self.check_test == 0 and self.test_generator is not None:
                self.model.eval()
                test_log = self.run_one_epoch(self.test_generator, step=False)
                log_one_epoch = log_one_epoch | {f'test_{key}': value for key, value in test_log.items()}

            # best model
            if epoch % self.check_log == 0 and self.model_path is not None:
                if log_one_epoch['val_loss'] < self.best_val_loss:
                    self.best_val_loss = log_one_epoch['val_loss']
                    if self.multi_gpu:
                        save_model = self.model.module
                    else:
                        save_model = self.model
                    torch.save(save_model, os.path.join(self.model_path, 'best_model.pt'))
                    log_one_epoch['save_model'] = True

            # plots
            # if epoch % self.check_log == 0 and self.graph_path is not None:
            #     self.plot_grad_flow(epoch)
            if self.output_path is not None:
                self.local_log(log_one_epoch)
            if self.log_wandb:
                wandb.log(log_one_epoch)

            # learning rate decay
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                if 'val_loss' in log_one_epoch:
                    self.lr_scheduler.step(log_one_epoch['val_loss'])
            elif isinstance(self.lr_scheduler, LRScheduler):
                self.lr_scheduler.step()

            # # loss force weight decay
            # self.main_loss.force_loss_decay()

            # checkpoint
            if epoch % self.check_log == 0 and self.model_path is not None:
                torch.save({
                        'epoch': epoch,
                        'step': step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                        'best_val_loss': self.best_val_loss,
                        'rng_state': torch.get_rng_state(),
                    }, os.path.join(self.model_path, 'train_state.pt'))
                
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    if self.optimizer.param_groups[0]['lr'] <= self.lr_scheduler.min_lrs[0]:
                        break

        print('Training finished')
        
        # load best model
        self.model = torch.load(os.path.join(self.model_path, 'best_model.pt'))
        self.model.eval()
        log_one_epoch = {'epoch': 'final'}
        train_log = self.run_one_epoch(self.train_generator, step=False)
        log_one_epoch = log_one_epoch | {f'train_{key}': value for key, value in train_log.items()}
        if self.val_generator is not None:
            val_log = self.run_one_epoch(self.val_generator, step=False)
            log_one_epoch = log_one_epoch | {f'val_{key}': value for key, value in val_log.items()}
        if self.test_generator is not None:
            test_log = self.run_one_epoch(self.test_generator, step=False)
            log_one_epoch = log_one_epoch | {f'test_{key}': value for key, value in test_log.items()}
        if self.output_path is not None:
            self.local_log(log_one_epoch)
        if self.log_wandb:
            wandb.log(log_one_epoch)


    def run_one_epoch(self, generator, step=False):
        log_one_epoch = {}
        for batch in generator:
            if step:
                self.optimizer.zero_grad()
            # preds = self.model(batch)
            preds = self.model(batch.z, batch.disp, batch.edge_index, batch.batch)
            main_loss = self.main_loss(preds, batch)
            eval_loss = self.eval_loss(preds, batch)
            if step:
                main_loss.backward()
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
            log_one_epoch['loss'] = log_one_epoch.get('loss', 0.0) + main_loss.detach().item()
            for key, value in eval_loss.items():
                log_one_epoch[key] = log_one_epoch.get(key, 0.0) + value.detach().item()
        log_one_epoch = {key: value / len(generator) for key, value in log_one_epoch.items()}
        return log_one_epoch