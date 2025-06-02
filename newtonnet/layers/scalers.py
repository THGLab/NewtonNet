import torch
from torch import nn


def get_scaler_by_string(key):
    if key == 'energy':
        scaler = ScaleShift(scale=1.0, shift=0.0)
    elif key == 'gradient_force':
        scaler = ScaleShift(scale=None, shift=None)
    elif key == 'direct_force':
        scaler = ScaleShift(scale=1.0, shift=None)
    elif key == 'hessian':
        scaler = ScaleShift(scale=None, shift=None)
    elif key == 'virial':
        scaler = ScaleShift(scale=None, shift=None)
    elif key == 'stress':
        scaler = ScaleShift(scale=None, shift=None)
    elif key == 'charge':
        scaler = ScaleShift(scale=0.1, shift=0.0)
    elif key == 'bec':
        scaler = ScaleShift(scale=None, shift=None)
    else:
        raise NotImplementedError(f'Scaler type {key} is not implemented yet')
    return scaler

def set_scaler_by_string(key, scaler, stats, fit_scale=True, fit_shift=True):
    if scaler.scale is not None and key in stats and fit_scale:
        scaler.set_scale(stats[key]['scale'])
    if scaler.shift is not None and key in stats and fit_shift:
        scaler.set_shift(stats[key]['shift'])
    return scaler

class ScaleShift(nn.Module):
    '''
    Node-level scale and shift layer.
    
    Parameters:
        key (str): The key for the scaler
        scale (bool): Whether to scale the output.
        shift (bool): Whether to shift the output.
    '''
    def __init__(self, scale=None, shift=None):
        super().__init__()
        self.scale = nn.Embedding.from_pretrained(torch.ones(118 + 1, 1), freeze=False, padding_idx=0) if scale is not None else None
        self.shift = nn.Embedding.from_pretrained(torch.zeros(118 + 1, 1), freeze=False, padding_idx=0) if shift is not None else None

    def forward(self, output, outputs):
        '''
        Scale and shift input.

        Args:
            output (torch.Tensor): The output values.
            outputs (Data): Other output data.
        '''
        if self.scale is not None:
            output = output * self.scale(outputs.z)
        if self.shift is not None:
            output = output + self.shift(outputs.z)
        return output
    
    def set_scale(self, scale):
        self.scale.weight.data = scale.reshape(-1, 1)

    def set_shift(self, shift):
        self.shift.weight.data = shift.reshape(-1, 1)

    
    def __repr__(self):
        return f'{self.__class__.__name__}(scale={self.scale is not None}, shift={self.shift is not None})'
    