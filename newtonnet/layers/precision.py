import torch

def get_precision_by_string(key):
    '''
    Get the precision for data and model.
    '''
    if key in ['float32', 'float', 'single']:
        return torch.float32
    elif key in ['float64', 'double']:
        return torch.float64
    elif key in ['float16', 'half']:
        return torch.float16
    else:
        raise ValueError(f'precision {key} is not supported')