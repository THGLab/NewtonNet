import torch

def set_precison_by_string(key):
    '''
    Set the default precision of torch tensors.
    
    Args:
        precision (str): The precision to be used. Can be 'single' or 'double'.
    '''
    if key == 'single':
        torch.set_default_dtype(torch.float32)
        return torch.float32
    elif key == 'double':
        torch.set_default_dtype(torch.float64)
        return torch.float64
    else:
        raise ValueError(f'precision {key} is not supported')