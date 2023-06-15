import os
import numpy as np
import pandas as pd
import torch
import time
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch import nn
from newtonnet.utils.utility import standardize_batch
from itertools import chain


class Trainer:
    """
    Parameters
    ----------
    """
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 requires_dr,
                 device,
                 yml_path,
                 output_path,
                 script_name,
                 lr_scheduler,
                 energy_loss_w,
                 force_loss_w,
                 loss_wf_decay,
                 lambda_l1,
                 checkpoint_log=1,
                 checkpoint_val=1,
                 checkpoint_test=20,
                 checkpoint_model=1,
                 verbose=False,
                 training=True,
                 hooks=None,
                 mode="energy/force",
                 target_name=None,
                 force_latent=False):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.requires_dr = requires_dr
        self.device = device
        self.energy_loss_w = energy_loss_w
        self.force_loss_w = force_loss_w
        self.wf_lambda = lambda epoch: np.exp(-epoch * loss_wf_decay)
        self.lambda_l1 = lambda_l1

        if type(device) is list and len(device) > 1:
            self.multi_gpu = True
        else:
            self.multi_gpu = False

        # outputs
        self._subdirs(yml_path, output_path, script_name)
        if training:

            # hooks
            if hooks:
                self.hooks = None
                self._hooks(hooks)

            # learning rate scheduler
            self._handle_scheduler(lr_scheduler, optimizer)
            self.lr_scheduler = lr_scheduler

        # checkpoints
        self.check_log = checkpoint_log
        self.check_val = checkpoint_val
        self.check_test = checkpoint_test
        self.check_model = checkpoint_model
        self.verbose = verbose

        # checkpoints
        self.epoch = 0  # number of epochs of any steps that model has gone through so far
        self.log_loss = {
            'epoch': [],
            'loss(MSE)': [],
            'lr': [],
            'time': []
        }
        if mode in ["energy/force", "energy"]:
            self.log_loss.update({
                'tr_E(MAE)': [],
                'tr_F(MAE)': [],
                'val_E(MAE)': [],
                'val_F(MAE)': [],
                'irc_E(MAE)': [],
                'irc_F(MAE)': [],
                'test_E(MAE)': [],
                'test_F(MAE)': []
            })
        elif mode == "atomic_properties":
            self.log_loss.update({
                "tr_err(RMSE)": [],
                "val_err(RMSE)": [],
                "test_err(RMSE)": []
            })
        self.best_val_loss = float("inf")
        self.mode = mode
        self.target_name = target_name
        self.force_latent = force_latent

    def _handle_scheduler(self, lr_scheduler, optimizer):

        if lr_scheduler[0] == 'plateau':
            self.scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                               mode='min',
                                               patience=lr_scheduler[2],
                                               factor=lr_scheduler[3],
                                               min_lr=lr_scheduler[4])
        elif lr_scheduler[0] == 'decay':
            lambda1 = lambda epoch: np.exp(-epoch * lr_scheduler[1])
            self.scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda1)

        else:
            raise NotImplemented('scheduler "%s" is not implemented yet.'%lr_scheduler[0])

    def _subdirs(self, yml_path, output_path, script_name):

        # create output directory and subdirectories
        path_iter = output_path[1]
        out_path = os.path.join(output_path[0], 'training_%i'%path_iter)
        while os.path.exists(out_path):
            path_iter+=1
            out_path = os.path.join(output_path[0],'training_%i'%path_iter)
        os.makedirs(out_path)
        self.output_path = out_path

        # self.val_out_path = os.path.join(self.output_path, 'validation')
        # os.makedirs(self.val_out_path)

        # subdir for computation graph
        self.graph_path = os.path.join(self.output_path, 'graph')
        if not os.path.exists(self.graph_path):
            os.makedirs(self.graph_path)

        # saved models
        self.model_path = os.path.join(self.output_path, 'models')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        script_out = os.path.join(self.output_path, 'run_scripts')
        os.makedirs(script_out)
        shutil.copyfile(yml_path, os.path.join(script_out,os.path.basename(yml_path)))
        shutil.copyfile(script_name, os.path.join(script_out,os.path.basename(script_name)))

    def _hooks(self, hooks):
        hooks_list = []
        if 'vismolvector3d' in hooks and hooks['vismolvector3d']:
            from combust.train.hooks import VizMolVectors3D

            vis = VizMolVectors3D()
            vis.set_output(True, None)
            hooks_list.append(vis)

        if len(hooks_list) > 0:
            self.hooks = hooks_list


    def print_layers(self):
        total_n_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
                if len(param.shape) > 1:
                    total_n_params += param.shape[0] * param.shape[1]
                else:
                    total_n_params += param.shape[0]
        print('\n total trainable parameters: %i\n' % total_n_params)

    def plot_grad_flow(self):
        ave_grads = []
        layers = []
        for n, p in self.model.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                # shorten names
                layer_name = n.split('.')
                layer_name = [l[:3] for l in layer_name]
                layer_name = '.'.join(layer_name[:-1])
                layers.append(layer_name)
                # print(layer_name, p.grad)
                if p.grad is not None:
                    ave_grads.append(p.grad.abs().mean().detach().cpu())
                else:
                    ave_grads.append(0)

        fig, ax = plt.subplots(1, 1)
        ax.plot(ave_grads, alpha=0.3, color="b")
        ax.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow: epoch#%i" %self.epoch)
        plt.grid(True)
        ax.set_axisbelow(True)

        file_name= os.path.join(self.graph_path,"avg_grad.png")
        plt.savefig(file_name, dpi=300,bbox_inches='tight')
        plt.close(fig)

    def store_checkpoint(self, input, steps):
        self.log_loss['epoch'].append(self.epoch)
        for k in input:
            self.log_loss[k].append(input[k])


        df = pd.DataFrame(self.log_loss)
        df.applymap('{:.5f}'.format).to_csv(os.path.join(
            self.output_path, 'log.csv'),
                                           index=False)

        print("[%d, %3d]" % (self.epoch, steps), end="")
        for k in input:
            print("%s: %.5f; " % (k, input[k]), end="")
        print("\n")
        #  loss_mse: %.5f; "
        #     "tr_E(MAE): %.5f; tr_F(MAE): %.5f; "
        #     "val_E(MAE): %.5f; val_F(MAE): %.5f; "
        #     "irc_E(MAE): %.5f; irc_F(MAE): %.5f; "
        #     "test_E(MAE): %.5f; test_F(MAE): %.5f; "
        #     "lr: %.9f; epoch_time: %.3f\n"
        #     % (self.epoch, steps, np.sqrt(input[0]),
        #        input[1], input[2], input[3], input[4], input[5], input[6],
        #        input[7], input[8], input[9], input[10]))

    def _optimizer_to_device(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device[0])

    def metric_se(self, preds, data):
        """
        squared error

        Parameters
        ----------
        preds: B,A,...
        data: B,A,...

        """
        diff = preds - data
        se = diff**2

        # diff_forces = preds[1] - batch_data.forces
        # diff_forces = diff_forces ** 2

        # err_sq = torch.sum(diff_energy) + torch.sum(diff_forces)

        return se

    def metric_ae(self, preds, data, divider=None):
        """absolute error"""
        if data.ndim < preds.ndim:
            data = data[:, None]
        ae = np.abs(preds - data)
        if divider is not None:
            if divider.ndim < ae.ndim:
                divider = divider[:, None]
            ae /= divider
        # diff_forces = preds[1] - batch_data.forces
        # diff_forces = diff_forces ** 2

        # err_sq = torch.sum(diff_energy) + torch.sum(diff_forces)

        return ae

    def metric_rmse(self, preds, data, mask=None):
        """Root mean square error"""
        se = np.square(preds.squeeze() - data)
        if mask is not None:
            se *= mask
            return np.sqrt(np.sum(se) / (np.sum(mask) + 1e-7))
        else:
            return np.sqrt(np.mean(se))

    def masked_average(self, y, atom_mask):
        """

        Parameters
        ----------
        y: numpy array
        atom_mask: numpy array

        Returns
        -------

        """
        # handle rotation-wise loader batch size mismatch
        if atom_mask.shape[0] > y.shape[0]:
            # assert atom_mask.shape[1] == y.shape[1]
            atom_mask = atom_mask.reshape(y.shape[0], -1, y.shape[1])  # B, n_rot, A
            atom_mask = atom_mask.mean(axis=1)

        # size = np.sum(atom_mask, axis=1, keepdims=True)
        # size = np.maximum(size, np.ones_like(size))
        # y = np.sum(y, axis=1)
        # y = y / size

        y = y[atom_mask!=0]

        return y

    def resume_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return loss

    def validation(self, name, generator, steps):

        self.model.eval()
        self.model.requires_dr = self.requires_dr

        val_error_energy = []
        val_error_force = []
        energy_pred = []
        force_pred = []
        e = []
        f = []
        ei = []
        fi = []
        AM = []
        RM = []  # rotation angles/matrix

        for val_step in range(steps):
            val_batch = next(generator)

            if self.hooks is not None and val_step == steps-1:
                self.model.return_intermediate = True
                val_preds = self.model(val_batch)

                hs = val_preds['hs']
                for iter in range(1,4):
                    self.hooks[0].run(val_batch['R'][0], val_batch['Z'][0],val_batch['F'][0],
                                   hs[iter][1][0],hs[iter][2][0])
                    R_ = val_batch['R'].data.cpu().numpy()
                    np.save(os.path.join(self.val_out_path, 'hs_%i_R_'%iter), R_)
                    Z_ = val_batch['Z'].data.cpu().numpy()
                    np.save(os.path.join(self.val_out_path, 'hs_%i_Z_'%iter), Z_)
                    if self.force_latent:
                        F_ = val_batch['F_latent'].data.cpu().numpy()
                    else:
                        F_ = val_batch['F'].data.cpu().numpy()
                    np.save(os.path.join(self.val_out_path, 'hs_%i_F_'%iter), F_)
                    dF_ = hs[iter][1].data.cpu().numpy()
                    np.save(os.path.join(self.val_out_path, 'hs_%i_dF_'%iter), dF_)
                    dR_ = hs[iter][2].data.cpu().numpy()
                    np.save(os.path.join(self.val_out_path, 'hs_%i_dR_'%iter), dR_)

                # self.model.return_intermediate = False
            else:
                val_preds = self.model(val_batch)

            if val_preds['E'].ndim == 3:
                E = val_batch["E"].unsqueeze(1).repeat(1,val_batch["Z"].shape[1],1)
            else:
                E = val_batch["E"]

            val_error_energy.append(self.metric_ae(
                val_preds['E'].detach().cpu().numpy(), E.detach().cpu().numpy(),
                divider=val_batch['NA'].detach().cpu().numpy() if 'NA' in val_batch else None))
            energy_pred.append(val_preds['E'].detach().cpu().numpy())
            e.append(E.detach().cpu().numpy())
            ei.append(val_preds['Ei'].detach().cpu().numpy())
            if self.mode == 'energy/force':
                if self.force_latent:
                    predicted_force = val_preds['F_latent'].detach().cpu().numpy()
                else:
                    predicted_force = val_preds['F'].detach().cpu().numpy()
                target_force = val_batch["F"].detach().cpu().numpy()
                val_error_force.append(self.metric_ae(
                predicted_force, target_force))

                force_pred.append(predicted_force)
                f.append(target_force)

            if 'dEi' in val_preds and val_preds['dEi'] is not None:
                fi.append(val_preds['dEi'].detach().cpu().numpy())
            AM.append(val_batch["AM"].detach().cpu().numpy())
            RM.append(val_batch["RM"].detach().cpu().numpy())


            if self.verbose:
                val_error_force_report = standardize_batch(list(chain(*val_error_force)))
                AM_report = standardize_batch(list(chain(*AM)))
                val_mae_force_report = np.mean(self.masked_average(val_error_force_report, AM_report))

                print(
                    "%s: %i/%i - E_loss(MAE): %.5f - F_loss(MAE): %.5f"
                    % (name, val_step, steps,
                       np.mean(np.concatenate(val_error_energy, axis=0)),
                       val_mae_force_report
                       ))

            del val_batch

        outputs = dict()
        AM = standardize_batch(list(chain(*AM)))
        outputs['AM'] = AM
        outputs['RM'] = np.concatenate(RM, axis=0)
        outputs['E_ae'] = np.concatenate(val_error_energy, axis=0)
        outputs['E_pred'] = np.concatenate(energy_pred,axis=0)
        outputs['E'] = np.concatenate(e, axis=0)
        outputs['Ei'] = standardize_batch(list(chain(*ei)))
        if len(fi) > 0:
            outputs['dEi'] = standardize_batch(list(chain(*fi)))
        if self.mode == 'energy/force':
            F_ae = standardize_batch(list(chain(*val_error_force)))
            outputs['F_ae_masked'] = self.masked_average(F_ae, AM)
            outputs['F_ae'] = F_ae
            outputs['F_pred'] = standardize_batch(list(chain(*force_pred)))
            outputs['F'] = standardize_batch(list(chain(*f)))
            outputs['total_ae'] = np.mean(outputs['E_ae']) + np.mean(outputs['F_ae_masked'])
        else:
            outputs['F_ae_masked'] = 0
            outputs['F_ae'] = 0
            outputs['F_pred'] = []
            outputs['F'] = 0
            outputs['total_ae'] = np.mean(outputs['E_ae'])


        return outputs

    def validation_atomic_properties(self, name, target_name, generator, steps):
        target_name = "Ai"
        if self.target_name is not None:
            target_name = self.target_name
        self.model.eval()
        self.model.requires_dr = False

        val_rmse = []
        pred = []
        R = []
        Z = []
        CS = []
        M = []
        labels = []
        # e = []
        # f = []
        # ei = []
        # fi = []
        AM = []
        RM = []  # rotation angles/matrix

        for val_step in range(steps):
            val_batch = next(generator)
            with torch.no_grad():
                val_preds = self.model(val_batch)[target_name]
            batch_target = val_batch[target_name]
            batch_mask = val_batch["M"]

            val_rmse.append(self.metric_rmse(
                val_preds.detach().cpu().numpy(),
                batch_target.detach().cpu().numpy(),
                batch_mask.detach().cpu().numpy()
            ))
            pred.append(val_preds.detach().cpu().numpy())
            R.append(val_batch["R"].detach().cpu().numpy())
            Z.append(val_batch["Z"].detach().cpu().numpy())
            CS.append(val_batch["CS"].detach().cpu().numpy())
            M.append(batch_mask.detach().cpu().numpy())
            if "labels" in val_batch:
                labels.extend(val_batch["labels"])

            AM.append(val_batch["AM"].detach().cpu().numpy())
            RM.append(val_batch["RM"].detach().cpu().numpy())


            if self.verbose:

                print(
                    "%s: %i/%i - %s_RMSE: %.5f"
                    % (name, val_step, steps,
                    target_name, np.mean(np.array(val_rmse))
                       ))

            del val_batch

        outputs = dict()
        AM = standardize_batch(list(chain(*AM)))
        outputs['AM'] = AM
        outputs['RM'] = np.concatenate(RM, axis=0)
        outputs['pred'] = standardize_batch(list(chain(*pred)))
        outputs['RMSE'] = np.mean(val_rmse)
        outputs["R"] = R
        outputs["Z"] = Z
        outputs["CS"] = CS
        outputs["M"] = M
        outputs["labels"] = labels
        return outputs

    def train(self,
              train_generator,
              epochs,
              steps,
              val_generator=None,
              val_steps=None,
              irc_generator=None,
              irc_steps=None,
              test_generator=None,
              test_steps=None,
              clip_grad=0):
        """
        The main function to train model for the given number of epochs (and steps per epochs).
        The implementation allows for resuming the training with different data and number of epochs.

        Parameters
        ----------
        epochs: int
            number of training epochs

        steps: int
            number of steps to call it an epoch (designed for nonstop data generators)


        """
        self.model.to(self.device[0])
        if self.multi_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.device)
        self._optimizer_to_device()

        running_val_loss = []
        last_test_epoch = 0
        for _ in tqdm(range(epochs)):
            t0 = time.time()

            # record total number of epochs so far
            self.epoch += 1

            # loss force weight decay
            w_f = self.force_loss_w * self.wf_lambda(self.epoch)

            # training
            running_loss = 0.0
            ae_energy = 0.0
            ae_force = 0.0
            rmse_ai = []
            n_data = 0
            n_atoms = 0
            self.model.train()
            self.model.requires_dr = self.requires_dr
            self.optimizer.zero_grad()
            # step_iterator = range(steps)
            # if not self.verbose:
            #     step_iterator = tqdm(step_iterator)

            for s in range(steps):
                self.optimizer.zero_grad()

                train_batch = next(train_generator)
                # self.model.module(train_batch)
                # preds = self.model.forward(train_batch)
                preds = self.model(train_batch)
                loss = self.loss_fn(preds, train_batch, self.model.parameters(),
                                    w_e=self.energy_loss_w,
                                    w_f=w_f,
                                    lambda_l1=self.lambda_l1)
                loss.backward()
                if clip_grad>0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                self.optimizer.step()
                # if (s+1)%4==0 or s==steps-1:
                #     self.optimizer.step()          # if in, comment out the one in the loop
                #     self.optimizer.zero_grad()     # if in, comment out the one in the loop

                current_loss = loss.detach().item()
                running_loss += current_loss

                atom_mask = train_batch["AM"].detach().cpu().numpy()
                n_atoms += np.sum(atom_mask)

                if self.mode in ["energy/force", "energy"]:
                    ae_energy += np.sum(self.metric_ae(
                        preds['E'].detach().cpu().numpy(),
                        train_batch["E"].detach().cpu().numpy(),
                        divider=train_batch['NA'].detach().cpu().numpy() if 'NA' in train_batch else None))

                    if self.mode == "energy/force":
                        ae_f = self.metric_ae(
                            preds['F'].detach().cpu().numpy(),
                            train_batch["F"].detach().cpu().numpy())
                        actual_ae_f = self.masked_average(ae_f, atom_mask)
                        ae_force += np.sum(actual_ae_f)
                    n_data += train_batch["E"].size()[0]
                    
                # n_atoms = train_batch.R.size()[1]

                    if self.verbose:
                        print(
                            "Train: Epoch %i/%i - %i/%i - loss: %.5f - running_loss(RMSE): %.5f - E(MAE): %.5f - F(MAE): %.5f"
                            % (self.epoch, epochs, s, steps, current_loss,
                            np.sqrt(running_loss / (s + 1)),
                            (ae_energy / (n_data)),
                            (ae_force / (n_atoms*3))
                            ))
                elif self.mode == "atomic_properties":
                    target_name = 'Ai'
                    if self.target_name is not None:
                        target_name = self.target_name
                    rmse_ai.append(np.mean(self.metric_rmse(
                        preds[target_name].detach().cpu().numpy(),
                        train_batch["CS"].detach().cpu().numpy(),
                        train_batch["M"].detach().cpu().numpy()
                    )))

                    n_data += train_batch["CS"].size()[0]

                    if self.verbose:
                        print(
                            "Train: Epoch %i/%i - %i/%i - loss: %.5f - running_loss(RMSE): %.5f - RMSE: %.5f"
                            % (self.epoch, epochs, s, steps, current_loss,
                            np.sqrt(running_loss / (s + 1)),
                            (np.mean(rmse_ai[-100:]))
                            ))
                del train_batch

            running_loss /= steps
            if self.mode in ["energy/force", "energy"]:
                ae_energy /= n_data
                ae_force /= (n_atoms * 3)
            elif self.mode == "atomic_properties":
                rmse_ai = np.mean(rmse_ai[-100:])

            # plots
            # self.plot_grad_flow()

            # validation
            val_error = float("inf")
            if self.mode in ["energy/force", "energy"]:
                val_mae_E = val_mae_F = 0
                if val_generator is not None and \
                    self.epoch % self.check_val == 0:

                    outputs = self.validation('valid', val_generator, val_steps)
                    if self.requires_dr:
                        val_error = self.energy_loss_w * np.mean(outputs['E_ae']) + \
                                    self.force_loss_w * np.mean(outputs['F_ae_masked'])
                    else:
                        val_error = self.energy_loss_w * np.mean(outputs['E_ae'])

                    val_mae_E = np.mean(outputs['E_ae'])
                    val_mae_F = np.mean(outputs['F_ae_masked'])
            elif self.mode == "atomic_properties":
                if val_generator is not None and \
                    self.epoch % self.check_val == 0:

                    outputs = self.validation_atomic_properties('valid', "CS", val_generator, val_steps)
                    val_error = outputs["RMSE"]

            # best model
            irc_mae_E = 0; irc_mae_F = 0
            test_mae_E = 0; test_mae_F = 0
            test_error = 0
            if self.best_val_loss > val_error:
                self.best_val_loss = val_error
                if self.multi_gpu:
                    save_model = self.model.module
                else:
                    save_model = self.model
                torch.save(save_model,#.state_dict(),
                           os.path.join(self.model_path, 'best_model.pt'))
                torch.save({
                            'epoch': self.epoch,
                            'model_state_dict': save_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss
                            },
                    os.path.join(self.model_path, 'best_model_state.tar')
                )

                # save irc predictions
                irc_mae_E = irc_mae_F = 0.0
                if irc_generator is not None:
                    outputs = self.validation('irc', irc_generator, irc_steps)
                    irc_mae_E = np.mean(outputs['E_ae'])
                    irc_mae_F = np.mean(outputs['F_ae_masked'])
                    # np.save(os.path.join(self.val_out_path, 'irc_ae_E'), outputs['E_ae'])
                    # np.save(os.path.join(self.val_out_path, 'irc_ae_F'), outputs['F_ae'])
                    # np.save(os.path.join(self.val_out_path, 'irc_pred_E'), outputs['E_pred'])
                    # np.save(os.path.join(self.val_out_path, 'irc_pred_F'), outputs['F_pred'])
                    # np.save(os.path.join(self.val_out_path, 'irc_E'), outputs['E'])
                    # np.save(os.path.join(self.val_out_path, 'irc_F'), outputs['F'])
                    # np.save(os.path.join(self.val_out_path, 'irc_Ei_best'), outputs['Ei'])
                    # np.save(os.path.join(self.val_out_path, 'irc_AM'), outputs['AM'])
                    # np.save(os.path.join(self.val_out_path, 'irc_RM'), outputs['RM'])
                    # np.save(os.path.join(self.val_out_path, 'irc_Ei_epoch%i'%self.epoch), outputs['Ei'])

                # save test predictions
                if test_generator is not None and self.epoch - last_test_epoch >= self.check_test:
                    if self.mode in ["energy/force", "energy"]:
                        outputs = self.validation('test', test_generator, test_steps)
                        test_mae_E = np.mean(outputs['E_ae'])
                        test_mae_F = np.mean(outputs['F_ae_masked'])
                        # np.save(os.path.join(self.val_out_path, 'test_ae_E'), outputs['E_ae'])
                        # np.save(os.path.join(self.val_out_path, 'test_ae_F'), outputs['F_ae'])
                        # np.save(os.path.join(self.val_out_path, 'test_pred_E'), outputs['E_pred'])
                        # np.save(os.path.join(self.val_out_path, 'test_pred_F'), outputs['F_pred'])
                        # np.save(os.path.join(self.val_out_path, 'test_E'), outputs['E'])
                        # np.save(os.path.join(self.val_out_path, 'test_F'), outputs['F'])
                        # np.save(os.path.join(self.val_out_path, 'test_AM'), outputs['AM'])
                        # np.save(os.path.join(self.val_out_path, 'test_RM'), outputs['RM'])
                    elif self.mode == "atomic_properties":
                        outputs = self.validation_atomic_properties('test', "CS", test_generator, test_steps)
                        # torch.save(outputs, os.path.join(self.val_out_path, 'test_results.pkl'))
                        test_error = outputs["RMSE"]
                    last_test_epoch = self.epoch
                    # np.save(os.path.join(self.val_out_path, 'test_Ei_epoch%i'%self.epoch), outputs['Ei'])

            # learning rate decay
            if self.lr_scheduler[0] == 'plateau':
                running_val_loss.append(val_error)
                if len(running_val_loss) > self.lr_scheduler[1]:
                    running_val_loss.pop(0)
                accum_val_loss = np.mean(running_val_loss)
                self.scheduler.step(accum_val_loss)
            elif self.lr_scheduler[0] == 'decay':
                self.scheduler.step()
                accum_val_loss = 0.0

            # checkpoint
            if self.epoch % self.check_log == 0:

                for i, param_group in enumerate(
                        self.scheduler.optimizer.param_groups):
                    old_lr = float(param_group["lr"])

                if self.mode in ["energy/force", "energy"]:
                    self.store_checkpoint({
                        "loss(MSE)": running_loss,
                        'tr_E(MAE)': ae_energy,
                        'tr_F(MAE)': ae_force,
                        'val_E(MAE)': val_mae_E,
                        'val_F(MAE)': val_mae_F,
                        'irc_E(MAE)': irc_mae_E,
                        'irc_F(MAE)': irc_mae_F,
                        'test_E(MAE)': test_mae_E,
                        'test_F(MAE)': test_mae_F,
                        "lr": old_lr,
                        "time": time.time() - t0},
                        steps)
                elif self.mode == "atomic_properties":
                    self.store_checkpoint({
                        "loss(MSE)": running_loss,
                        "tr_err(RMSE)": rmse_ai,
                        "val_err(RMSE)": val_error,
                        "test_err(RMSE)": test_error,
                        "lr": old_lr,
                        "time": time.time() - t0
                    }, steps)

    def log_statistics(self, n_train_data, n_val_data, n_test_data, normalizer, test_energy_hash):
        with open(os.path.join(self.output_path, "stats.txt"), "w") as f:
            f.write("Train data: %d\n" % n_train_data)
            f.write("Val data: %d\n" % n_val_data)
            f.write("Test data: %d\n" % n_test_data)
            f.write("Normalizer: %s\n" % str(normalizer))
            f.write("Test energy hash: %s\n" % test_energy_hash)
