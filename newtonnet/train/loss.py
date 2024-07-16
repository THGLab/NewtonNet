import torch
import torch.nn as nn
from torch_geometric.utils import scatter

def get_loss_by_string(mode=None, **kwargs):
    '''
    Get loss function by string

    Parameters:
        mode (str): The loss function to use. Default: None.
        w_energy (float): The weight for the energy loss. Default: 0.0.
        w_force (float): The weight for the force loss. Default: 0.0.
        w_force_mag (float): The weight for the force magnitude loss. Default: 0.0.
        w_force_dir (float): The weight for the force direction loss. Default: 0.0.

    Returns:
        main_loss (nn.Module): The main loss function for model training (back propagation) and validation (learning rate scheduling).
        eval_loss (nn.Module): The evaluation loss function for task-specific model evaluation.
    '''
    if mode is None:
        main_loss = MultitaskLoss(loss_fns=[], sum=True)
        eval_loss = MultitaskLoss(loss_fns=[], sum=False)

    elif mode == 'energy':
        main_losses = []
        weights = []
        if kwargs.get('w_energy', 0.0) > 0.0:
            main_losses.append(EnergyLoss(mode='mse'))
            weights.append(kwargs['w_energy'])
        main_loss = MultitaskLoss(loss_fns=main_losses, sum=True, weights=weights)

        eval_losses = []
        eval_losses.append(EnergyLoss(mode='mae'))
        eval_losses.append(EnergyLoss(mode='mse'))
        eval_losses.append(EnergyPerAtomLoss(mode='mae'))
        eval_losses.append(EnergyPerAtomLoss(mode='mse'))
        eval_loss = MultitaskLoss(loss_fns=eval_losses, sum=False)

    elif mode == 'energy/forces':
        main_losses = []
        weights = []
        if kwargs.get('w_energy', 0.0) > 0.0:
            main_losses.append(EnergyLoss(mode='mse'))
            weights.append(kwargs['w_energy'])
        if kwargs.get('w_force', 0.0) > 0.0:
            main_losses.append(ForcesLoss(mode='mse'))
            weights.append(kwargs['w_force'])
        if kwargs.get('w_force_mag', 0.0) > 0.0:
            main_losses.append(ForcesNormLoss(mode='mse'))
            weights.append(kwargs['w_force_mag'])
        if kwargs.get('w_force_dir', 0.0) > 0.0:
            main_losses.append(ForcesCosLoss(mode='mse'))
            weights.append(kwargs['w_force_dir'])
        main_loss = MultitaskLoss(loss_fns=main_losses, sum=True, weights=weights)

        eval_losses = []
        eval_losses.append(EnergyLoss(mode='mae'))
        eval_losses.append(EnergyLoss(mode='mse'))
        eval_losses.append(EnergyPerAtomLoss(mode='mae'))
        eval_losses.append(EnergyPerAtomLoss(mode='mse'))
        eval_losses.append(ForcesLoss(mode='mae'))
        eval_losses.append(ForcesLoss(mode='mse'))
        eval_loss = MultitaskLoss(loss_fns=eval_losses, sum=False)
        
    else:
        raise ValueError(f'loss {mode} not implemented')
    
    return main_loss, eval_loss


class BaseLoss(nn.Module):
    def __init__(self, mode: str = 'mse'):
        super(BaseLoss, self).__init__()
        self.mode = mode
        if self.mode == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.mode == 'mae':
            self.loss_fn = nn.L1Loss()
        elif self.mode == 'mcs':
            self.loss_fn = lambda x, y: (1 - torch.cosine_similarity(x, y, dim=-1)).mean()
        else:
            raise ValueError(f'loss mode {mode} not implemented')

class EnergyLoss(BaseLoss):
    def __init__(self, mode='mse'):
        super(EnergyLoss, self).__init__(mode)
        self.name = f'energy_{self.mode}'

    def forward(self, pred, data):
        loss = self.loss_fn(pred.energy, data.energy)
        return loss

class EnergyPerAtomLoss(BaseLoss):
    def __init__(self, mode='mse'):
        super(EnergyPerAtomLoss, self).__init__(mode)
        self.name = f'energy_per_atom_{self.mode}'

    def forward(self, pred, data):
        n_atoms = scatter(data.z, data.batch)
        loss = self.loss_fn(pred.energy / n_atoms, data.energy / n_atoms)
        return loss

class ForcesLoss(BaseLoss):
    def __init__(self, mode='mse'):
        super(ForcesLoss, self).__init__(mode)
        self.name = f'forces_{self.mode}'

    def forward(self, pred, data):
        loss = self.loss_fn(pred.force, data.force)
        return loss

class ForcesNormLoss(BaseLoss):
    def __init__(self, mode='mse'):
        super(ForcesNormLoss, self).__init__(mode)
        self.name = f'forces_norm_{self.mode}'

    def forward(self, pred, data):
        loss = self.loss_fn(pred.force.norm(dim=-1), data.force.norm(dim=-1))
        return loss


# class ScalarLoss(BaseLoss):
#     def __init__(self, **kwargs):
#         super(ScalarLoss, self).__init__(**kwargs)
#         if self.per_atom:
#             self.name = f'{self.key}_per_atom_{self.mode}'
#         else:
#             self.name = f'{self.key}_{self.mode}'

#     def forward(self, pred, data):
#         if self.per_atom:
#             n_atoms = scatter(data.z, data.batch)
#             loss = self.loss_fn(
#                 pred.__getattribute__(self.key) / n_atoms, 
#                 data.__getattribute__(self.key) / n_atoms,
#                 )
#         else:
#             loss = self.loss_fn(
#                 pred.__getattribute__(self.key), 
#                 data.__getattribute__(self.key),
#                 )
#         return loss
        
# class VectorLoss(BaseLoss):
#     def __init__(self, **kwargs):
#         super(VectorLoss, self).__init__(**kwargs)
#         self.name = f'{self.key}_{self.mode}'

#     def forward(self, pred, data):
#         loss = self.loss_fn(
#             pred.__getattribute__(self.key),
#             data.__getattribute__(self.key),
#             )
#         return loss
    
# class VectorNormLoss(BaseLoss):
#     def __init__(self, **kwargs):
#         super(VectorNormLoss, self).__init__(**kwargs)
#         self.name = f'{self.key}_norm_{self.mode}'

#     def forward(self, pred, data):
#         loss = self.loss_fn(
#             pred.__getattribute__(self.key).norm(dim=-1),
#             data.__getattribute__(self.key).norm(dim=-1),
#             )
#         return loss
    
# class VectorCosLoss(BaseLoss):
#     def __init__(self, **kwargs):
#         super(VectorCosLoss, self).__init__(**kwargs)
#         self.cos = nn.CosineSimilarity(dim=-1)
#         self.name = f'{self.key}_cos_{self.mode}'

#     def forward(self, pred, data):
#         cos = self.cos(
#             pred.__getattribute__(self.key), 
#             data.__getattribute__(self.key),
#             )
#         loss = self.loss_fn(cos, torch.ones_like(cos))
#         return loss


class MultitaskLoss(nn.Module):
    '''
    Multitask loss class

    Parameters:
        loss_fns (list): The list of loss functions.
        sum (bool): Whether to sum the loss functions. Default: True.
        weights (list): The list of weights for the loss functions. Default: None.
    '''
    def __init__(self, loss_fns: list, sum: bool = True, weights: list = None):
        super(MultitaskLoss, self).__init__()
        self.loss_fns = loss_fns
        self.sum = sum
        self.weights = weights

    def forward(self, pred, data):
        if self.sum:
            return sum([loss_fn(pred, data) for loss_fn in self.loss_fns])
        else:
            return {loss_fn.name: loss_fn(pred, data) for loss_fn in self.loss_fns}