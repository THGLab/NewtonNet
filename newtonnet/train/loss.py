import torch
import torch.nn as nn

def get_loss_by_string(**kwargs):
    '''
    mode: str
        mode of the loss function
    kwargs: dict
        keyword arguments for the loss function

    Returns
    -------
    main_loss: nn.Module
        main loss function for model training (back propagation) and validation (learning rate scheduling)
    eval_loss: nn.Module
        evaluation loss function for task-specific model evaluation
    '''
    mode = kwargs.get('mode', 'energy/force')
    if mode == 'energy/force':
        main_losses = []
        if kwargs.get('w_energy', 0.0) > 0.0:
            main_losses.append(ScalarLoss('E', mode='mse', masked=False, weight=kwargs['w_energy']))
        if kwargs.get('w_force', 0.0) > 0.0:
            main_losses.append(VectorLoss('F', mode='mse', masked=False, weight=kwargs['w_force']))
        if kwargs.get('w_f_mag', 0.0) > 0.0:
            main_losses.append(VectorNormLoss('F', mode='mse', masked=False, weight=kwargs['w_f_mag']))
        if kwargs.get('w_f_dir', 0.0) > 0.0:
            main_losses.append(VectorCosLoss('F', mode='mse', masked=False, weight=kwargs['w_f_dir']))
        main_loss = MultitaskLoss(mode=mode, loss_fns=main_losses, sum=True)

        eval_losses = []
        eval_losses.append(ScalarLoss('E', mode='mae', masked=False, weight=1.0))
        eval_losses.append(VectorLoss('F', mode='mae', masked=True, weight=1.0))
        eval_loss = MultitaskLoss(mode=mode, loss_fns=eval_losses, sum=False)
    else:
        raise ValueError(f'loss {mode} not implemented')
    
    return main_loss, eval_loss


class BaseLoss(nn.Module):
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
            loss = self.loss_fn(pred[self.key] * data['AM'], data[self.key] * data['AM']) * data['AM'].numel() / data['AM'].sum()
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
            loss = self.loss_fn(pred[self.key] * data['AM'][:, :, None], data[self.key] * data['AM'][:, :, None]) * data['AM'].numel() / data['AM'].sum()
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
            loss = self.loss_fn(pred[self.key].norm(dim=-1) * data['AM'], data[self.key].norm(dim=-1) * data['AM']) * data['AM'].numel() / data['AM'].sum()
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
            loss = self.loss_fn(self.cos(pred[self.key], data[self.key]) * data['AM'], torch.ones(n_data, n_atoms, dtype=torch.float, device=pred[self.key].device)) * data['AM'].numel() / data['AM'].sum()
            return self.weight * loss
        else:
            loss = self.loss_fn(self.cos(pred[self.key], data[self.key]), torch.ones(n_data, n_atoms, dtype=torch.float, device=pred[self.key].device))
            return self.weight * loss


class MultitaskLoss(nn.Module):
    def __init__(self, mode: str, loss_fns: list, sum: bool = True):
        super(MultitaskLoss, self).__init__()
        self.mode = mode
        self.loss_fns = loss_fns
        self.sum = sum

    def forward(self, pred, data):
        if self.sum:
            return sum([loss_fn(pred, data) for loss_fn in self.loss_fns])
        else:
            return {loss_fn.name: loss_fn(pred, data) for loss_fn in self.loss_fns}