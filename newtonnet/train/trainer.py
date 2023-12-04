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

from newtonnet.models.newtonnet import NewtonNet
from newtonnet.train.loss import get_loss_by_string


class Trainer:
    """
    Parameters
    ----------
    """
    def __init__(
            self,
            model: nn.Module = None,
            loss_fns: (nn.Module, nn.Module) = None,
            optimizer: optim.Optimizer = None,
            lr_scheduler: optim.lr_scheduler._LRScheduler = None,
            requires_dr: bool = False,
            device: torch.device = None,
            output_base_path: str = None,
            script_path: str = None,
            settings_path: str = None,
            checkpoint_log=1,
            checkpoint_val=1,
            checkpoint_test=20,
            checkpoint_model=1,
            verbose=False,
            ):
        # training parameters
        self.model = model or NewtonNet()
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.main_loss, self.eval_loss = loss_fns or get_loss_by_string('energy/force')
        self.optimizer = optimizer or optim.Adam(trainable_params)
        self.lr_scheduler = lr_scheduler or ReduceLROnPlateau(self.optimizer)
        self.requires_dr = requires_dr
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.multi_gpu = True if type(device) is list and len(device) > 1 else False
        self.verbose = verbose

        # outputs
        self.make_subdirs(output_base_path, script_path, settings_path)
        if self.verbose:
            self.print_layers()

        # checkpoints
        self.check_log = checkpoint_log
        self.check_val = checkpoint_val
        self.check_test = checkpoint_test
        self.check_model = checkpoint_model

        # checkpoints
        self.best_val_loss = torch.inf
        self.log = pd.DataFrame()
        self.log['epoch'] = None
        for phase in ('train', 'val', 'test'):
            self.log[f'{phase}_loss'] = None
            for loss_fn in self.eval_loss.loss_fns:
                self.log[f'{phase}_{loss_fn.name}'] = None
        self.log['lr'] = None
        self.log['time'] = None

    def make_subdirs(self, output_base_path, script_path, settings_path):
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
            # shorten names
            name = name.replace('node_embedding', 'emb')
            name = name.replace('message_passing_layers', 'mes')
            name = name.replace('invariant', 'inv')
            name = name.replace('equivariant', 'eq')
            name = name.replace('message', 'mes')
            name = name.replace('coefficient', 'coeff')
            name = name.replace('feature', 'feat')
            name = name.replace('selfupdate', 'upd')
            name = name.replace('property', 'prop')
            name = name.replace('prediction', 'pred')
            name = name.replace('weight', 'w')
            name = name.replace('bias', 'b')
            name = name.replace('mean', 'm')
            name = name.replace('std', 's')
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

    def resume_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return loss

    def train(
            self,
            train_generator,
            val_generator,
            test_generator,
            epochs,
            clip_grad=0,
            ):
        
        self.model.to(self.device[0])
        if self.multi_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.device)

        train_losses = {}
        for epoch in tqdm(range(epochs + 1)):
            t0 = time.time()

            # training
            train_losses['loss'] = 0.0
            self.model.train()
            # self.model.requires_dr = self.requires_dr
            # self.optimizer.zero_grad()

            for train_step, train_batch in enumerate(train_generator):
                batch_size = train_batch['Z'].shape[0]

                self.optimizer.zero_grad()
                preds = self.model(
                    atomic_numbers=train_batch['Z'], 
                    positions=train_batch['R'], 
                    atom_mask=train_batch['AM'], 
                    neighbors=train_batch['N'], 
                    neighbor_mask=train_batch['NM'],
                    distances=train_batch['D'],
                    distance_vectors=train_batch['V'],
                    )
                main_loss = self.main_loss(preds, train_batch)
                main_loss.backward()
                if clip_grad > 0:
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                    # if norm > clip_grad:
                    #     print(f'clipped gradients with norm {norm}')
                    #     for n, p in self.model.named_parameters():
                    #         if p.grad is not None:
                    #             print(f'{n}: {p.grad}')
                if main_loss.isnan():
                    raise ValueError('loss is nan')
                self.optimizer.step()

                main_loss = main_loss.detach().item()
                train_losses['loss'] += main_loss

                eval_loss = self.eval_loss(preds, train_batch)
                for key, value in eval_loss.items():
                    train_losses[key] = train_losses.get(key, 0.0) + value.detach().item() * batch_size

                if self.verbose:
                    print(f'Train: Epoch {epoch}/{epochs} - Batch {train_step}/{len(train_generator)} - loss: {main_loss:.5f} - ', end='')
                    print(*[f'{key}: {value:.5f}' for key, value in eval_loss.items()], sep=' - ')
                # self.plot_grad_flow(f'{epoch}_{train_step}')

            for key, value in train_losses.items():
                train_losses[key] /= len(train_generator) if key == 'loss' else len(train_generator.dataset)

            # plots
            # self.plot_grad_flow()

            # validation
            val_losses = {}
            if epoch % self.check_val == 0:
                val_losses['loss'] = 0.0
                self.model.eval()

                for val_step, val_batch in enumerate(val_generator):
                    batch_size = val_batch['Z'].shape[0]

                    if self.requires_dr:
                        preds = self.model(
                            atomic_numbers=val_batch['Z'], 
                            positions=val_batch['R'], 
                            atom_mask=val_batch['AM'], 
                            neighbors=val_batch['N'], 
                            neighbor_mask=val_batch['NM'],
                            distances=val_batch['D'],
                            distance_vectors=val_batch['V'],
                            )
                    else:
                        with torch.no_grad():
                            preds = self.model(
                                atomic_numbers=val_batch['Z'], 
                                positions=val_batch['R'], 
                                atom_mask=val_batch['AM'], 
                                neighbors=val_batch['N'], 
                                neighbor_mask=val_batch['NM'],
                                distances=val_batch['D'],
                                distance_vectors=val_batch['V'],
                                )
                    
                    main_loss = self.main_loss(preds, val_batch).detach().item()
                    val_losses['loss'] += main_loss

                    eval_loss = self.eval_loss(preds, val_batch)
                    for key, value in eval_loss.items():
                        val_losses[key] = val_losses.get(key, 0.0) + value.detach().item() * batch_size

                    if self.verbose:
                        print(f'Val: Epoch {epoch}/{epochs} - Batch {val_step}/{len(val_generator)} - loss: {main_loss:.5f} - ', end='')
                        print(*[f'{key}: {value:.5f}' for key, value in eval_loss.items()], sep=' - ')

                for key, value in val_losses.items():
                    val_losses[key] /= len(val_generator) if key == 'loss' else len(val_generator.dataset)

                # best model
                if self.best_val_loss > val_losses['loss']:
                    self.best_val_loss = val_losses['loss']
                    if self.multi_gpu:
                        save_model = self.model.module
                    else:
                        save_model = self.model
                    torch.save(save_model, os.path.join(self.model_path, 'best_model.pt'))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': save_model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': train_losses['loss'],
                        }, os.path.join(self.model_path, 'best_model_state.tar')
                    )

            # save test predictions
            test_losses = {}
            if epoch % self.check_test == 0:
                test_losses['loss'] = 0.0
                self.model.eval()

                for test_step, test_batch in enumerate(test_generator):
                    batch_size = test_batch['Z'].shape[0]

                    if self.requires_dr:
                        preds = self.model(
                            atomic_numbers=test_batch['Z'], 
                            positions=test_batch['R'], 
                            atom_mask=test_batch['AM'], 
                            neighbors=test_batch['N'], 
                            neighbor_mask=test_batch['NM'],
                            distances=test_batch['D'],
                            distance_vectors=test_batch['V'],
                            )
                    else:
                        with torch.no_grad():
                            preds = self.model(
                                atomic_numbers=test_batch['Z'], 
                                positions=test_batch['R'], 
                                atom_mask=test_batch['AM'], 
                                neighbors=test_batch['N'], 
                                neighbor_mask=test_batch['NM'],
                                distances=test_batch['D'],
                                distance_vectors=test_batch['V'],
                                )
                    
                    main_loss = self.main_loss(preds, test_batch).detach().item()
                    test_losses['loss'] += main_loss

                    eval_loss = self.eval_loss(preds, test_batch)
                    for key, value in eval_loss.items():
                        test_losses[key] = test_losses.get(key, 0.0) + value.detach().item() * batch_size

                    if self.verbose:
                        print(f'Test: Epoch {epoch}/{epochs} - Batch {test_step}/{len(test_generator)} - loss: {main_loss:.5f} - ', end='')
                        print(*[f'{key}: {value:.5f}' for key, value in eval_loss.items()], sep=' - ')

                for key, value in test_losses.items():
                    test_losses[key] /= len(test_generator) if key == 'loss' else len(test_generator.dataset)

            # checkpoint
            if epoch % self.check_log == 0:
                self.plot_grad_flow(epoch)
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

            # learning rate decay
            self.lr_scheduler.step(val_losses['loss'])

            # # loss force weight decay
            # self.main_loss.force_loss_decay()

    def log_statistics(self, n_train_data, n_val_data, n_test_data, normalizer, test_energy_hash):
        with open(os.path.join(self.output_path, "stats.txt"), "w") as f:
            f.write("Train data: %d\n" % n_train_data)
            f.write("Val data: %d\n" % n_val_data)
            f.write("Test data: %d\n" % n_test_data)
            f.write("Normalizer: %s\n" % str(normalizer))
            f.write("Test energy hash: %s\n" % test_energy_hash)
