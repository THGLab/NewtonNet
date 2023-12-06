import torch

def set_precison_by_string(precision):
    '''
    Set the default precision of torch tensors.
    
    Args:
        precision (str): The precision to be used. Can be 'single' or 'double'.
    '''
    if precision == 'single':
        torch.set_default_dtype(torch.float32)
        return torch.float32
    elif precision == 'double':
        torch.set_default_dtype(torch.float64)
        return torch.float64
    else:
        raise ValueError(f'precision {precision} is not supported')