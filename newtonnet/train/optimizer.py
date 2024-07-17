from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, OneCycleLR


def get_optimizer_by_string(optimizer_name, parameters, **kwargs):
    """Get optimizer by string.

    Parameters
    ----------
    optimizer_name: str
        Name of the optimizer.

    parameters: iterable
        Parameters to be optimized.

    kwargs: dict
        Keyword arguments.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer.

    """
    if optimizer_name == 'adam':
        optimizer = Adam(parameters, **kwargs)
    elif optimizer_name == 'sgd':
        optimizer = SGD(parameters, **kwargs)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(parameters, **kwargs)
    elif optimizer_name == 'adamw':
        optimizer = AdamW(parameters, **kwargs)
    else:
        raise ValueError(f'optimizer {optimizer_name} is not supported')
    return optimizer


def get_scheduler_by_string(scheduler_name, optimizer, **kwargs):
    """Get scheduler by string.

    Parameters
    ----------
    scheduler_name: str
        Name of the scheduler.

    optimizer: torch.optim.Optimizer
        Optimizer.

    kwargs: dict
        Keyword arguments.

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler
        Scheduler.

    """
    if scheduler_name == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_name == 'lambda':
        scheduler = LambdaLR(optimizer, **kwargs)
    elif scheduler_name == 'onecycle':
        scheduler = OneCycleLR(optimizer, **kwargs)
    else:
        raise ValueError(f'scheduler {scheduler_name} is not supported')
    return scheduler