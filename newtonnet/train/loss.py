import torch
import torch.nn as nn

def get_loss_by_string(mode=None, **kwargs):
    '''
    Get loss function by string

    Parameters:
        mode (str): The loss function to use. Default: None.
        w_energy (float): The weight for the energy loss. Default: 0.0.
        w_force (float): The weight for the force loss. Default: 0.0.
        w_f_mag (float): The weight for the force magnitude loss. Default: 0.0.
        w_f_dir (float): The weight for the force direction loss. Default: 0.0.

    Returns:
        main_loss (nn.Module): The main loss function for model training (back propagation) and validation (learning rate scheduling).
        eval_loss (nn.Module): The evaluation loss function for task-specific model evaluation.
    '''
    if mode is None:
        main_loss = MultitaskLoss(loss_fns=[], sum=True)
        eval_loss = MultitaskLoss(loss_fns=[], sum=False)

    elif mode == 'energy':
        main_losses = []
        if kwargs.get('w_energy', 0.0) > 0.0:
            main_losses.append(ScalarLoss('energy_normalized', mode='mse', masked=False, weight=kwargs['w_energy']))
        main_loss = MultitaskLoss(loss_fns=main_losses, sum=True)

        eval_losses = []
        eval_losses.append(ScalarLoss('energy', mode='mae', masked=False, weight=1.0))
        eval_loss = MultitaskLoss(loss_fns=eval_losses, sum=False)

    elif mode == 'energy/forces':
        main_losses = []
        if kwargs.get('w_energy', 0.0) > 0.0:
            main_losses.append(ScalarLoss('energy_normalized', mode='mse', masked=False, weight=kwargs['w_energy']))
        if kwargs.get('w_force', 0.0) > 0.0:
            main_losses.append(VectorLoss('forces_normalized', mode='mse', masked=False, weight=kwargs['w_force']))
        if kwargs.get('w_f_mag', 0.0) > 0.0:
            main_losses.append(VectorNormLoss('forces_normalized', mode='mse', masked=False, weight=kwargs['w_f_mag']))
        if kwargs.get('w_f_dir', 0.0) > 0.0:
            main_losses.append(VectorCosLoss('forces_normalized', mode='mse', masked=False, weight=kwargs['w_f_dir']))
        main_loss = MultitaskLoss(loss_fns=main_losses, sum=True)

        eval_losses = []
        eval_losses.append(ScalarLoss('energy', mode='mae', masked=False, weight=1.0))
        eval_losses.append(VectorLoss('forces', mode='mae', masked=True, weight=1.0))
        eval_loss = MultitaskLoss(loss_fns=eval_losses, sum=False)
        
    else:
        raise ValueError(f'loss {mode} not implemented')
    
    return main_loss, eval_loss


class BaseLoss(nn.Module):
    '''
    Base loss class

    Parameters:
        key (str): The key of the data to be used.
        mode (str): The loss function to use. Default: 'mse'.
        masked (bool): Whether to mask the loss. Default: False.
        weight (float): The weight for the loss. Default: 1.0.
    '''
    def __init__(
            self, 
            key: str, 
            mode: str = 'mse',
            masked: bool = False,
            weight: float = 1.0,
            ):
        super(BaseLoss, self).__init__()
        self.key = key
        if mode == 'mse':
            self.loss_fn = nn.MSELoss()
        elif mode == 'mae':
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f'loss mode {mode} not implemented')
        self.masked = masked
        self.weight = weight

    def forward(self, pred, data):
        raise NotImplementedError('forward method not implemented')

class ScalarLoss(BaseLoss):
    def __init__(self, key, mode, masked, weight):
        super(ScalarLoss, self).__init__(key, mode, masked, weight)
        self.name = f'{key}_{mode}'

    def forward(self, pred, data):
        if self.masked:
            loss = self.loss_fn(pred[self.key] * data['atom_mask'], data[self.key] * data['atom_mask']) * data['atom_mask'].numel() / data['atom_mask'].sum()
            return self.weight * loss
        else:
            loss = self.loss_fn(pred[self.key], data[self.key])
            return self.weight * loss
        
class VectorLoss(BaseLoss):
    def __init__(self, key, mode, masked, weight):
        super(VectorLoss, self).__init__(key, mode, masked, weight)
        self.name = f'{key}_{mode}'

    def forward(self, pred, data):
        if self.masked:
            loss = self.loss_fn(pred[self.key] * data['atom_mask'][:, :, None], data[self.key] * data['atom_mask'][:, :, None]) * data['atom_mask'].numel() / data['atom_mask'].sum()
            return self.weight * loss
        else:
            loss = self.loss_fn(pred[self.key], data[self.key])
            return self.weight * loss
    
class VectorNormLoss(BaseLoss):
    def __init__(self, key, mode, masked, weight):
        super(VectorNormLoss, self).__init__(key, mode, masked, weight)
        self.name = f'{key}_norm_{mode}'

    def forward(self, pred, data):
        if self.masked:
            loss = self.loss_fn(pred[self.key].norm(dim=-1) * data['atom_mask'], data[self.key].norm(dim=-1) * data['atom_mask']) * data['atom_mask'].numel() / data['atom_mask'].sum()
            return self.weight * loss
        else:
            loss = self.loss_fn(pred[self.key], data[self.key])
            return self.weight * loss
    
class VectorCosLoss(BaseLoss):
    def __init__(self, key, mode, masked, weight):
        super(VectorCosLoss, self).__init__(key, mode, masked, weight)
        self.cos = nn.CosineSimilarity(dim=-1)
        self.name = f'{key}_cos_{mode}'

    def forward(self, pred, data):
        n_data, n_atoms, _ = data[self.key].shape
        if self.masked:
            loss = self.loss_fn(self.cos(pred[self.key], data[self.key]) * data['atom_mask'], torch.ones(n_data, n_atoms, device=pred[self.key].device)) * data['atom_mask'].numel() / data['atom_mask'].sum()
            return self.weight * loss
        else:
            loss = self.loss_fn(self.cos(pred[self.key], data[self.key]), torch.ones(n_data, n_atoms, device=pred[self.key].device))
            return self.weight * loss


class MultitaskLoss(nn.Module):
    '''
    Multitask loss class

    Parameters:
        loss_fns (list): The list of loss functions.
        sum (bool): Whether to sum the loss functions. Default: True.
    '''
    def __init__(self, loss_fns: list, sum: bool = True):
        super(MultitaskLoss, self).__init__()
        self.loss_fns = loss_fns
        self.sum = sum

    def forward(self, pred, data):
        if self.sum:
            return sum([loss_fn(pred, data) for loss_fn in self.loss_fns])
        else:
            return {loss_fn.name: loss_fn(pred, data) for loss_fn in self.loss_fns}