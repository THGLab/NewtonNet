from torch import nn


def get_scaler_by_string(key, **stat):
    if key == 'energy':
        scaler = ScaleShift(**stat)
    elif key == 'force':
        scaler = ScaleShift(**stat)
    elif key == 'hessian':
        scaler = NullScaleShift()
    else:
        print(f'scaler {key} is not supported, use NullScaleShift.')
        scaler = NullScaleShift()
    return scaler


class ScaleShift(nn.Module):
    '''
    Node-level scale and shift layer.
    
    Parameters:
        shift (torch.Tensor): The shift values for the properties.
        scale (torch.Tensor): The scale values for the properties.
    '''
    def __init__(self, shift=None, scale=None):
        super(ScaleShift, self).__init__()
        self.z_max = 0
        if shift is None:
            self.shift = None
        elif shift.numel() == 1:
            self.shift = nn.Parameter(shift, requires_grad=True)
        else:
            self.shift = nn.Embedding.from_pretrained(shift.reshape(-1, 1), freeze=False)
            self.z_max = max(self.z_max, shift.size(0) - 1)
        if scale is None:
            self.scale = None
        elif scale.numel() == 1:
            self.scale = nn.Parameter(scale, requires_grad=True)
        else:
            self.scale = nn.Embedding.from_pretrained(scale.reshape(-1, 1), freeze=False)
            self.z_max = max(self.z_max, scale.size(0) - 1)

    def forward(self, input, z):
        '''
        Scale and shift input.

        Args:
            input (torch.Tensor): The input values.
            z (torch.Tensor): The atomic numbers of the atoms in the molecule.

        Returns:
            torch.Tensor: The normalized inputs.
        '''
        output = input
        if isinstance(self.scale, nn.Embedding):
            output = output * self.scale(z)
        elif isinstance(self.scale, nn.Parameter):
            output = output * self.scale
        if isinstance(self.shift, nn.Embedding):
            output = output + self.shift(z)
        # outputs = inputs * self.scale(z) + self.shift(z)
        # outputs = inputs + self.shift(z)
        return output
    
    def __repr__(self):
        return f'{self.__class__.__name__}(scale={self.scale is not None}, shift={self.shift is not None})'
    

class NullScaleShift(nn.Module):
    '''
    Null scale and shift layer for untrained properties. Identity function.
    '''
    def __init__(self):
        super(NullScaleShift, self).__init__()
        self.z_max = 0

    def forward(self, inputs, z):
        return inputs