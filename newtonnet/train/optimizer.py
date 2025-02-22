from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, OneCycleLR, LinearLR, CosineAnnealingLR, ChainedScheduler


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


def get_scheduler_by_string(scheduler_list, optimizer):
    """Get scheduler by string.

    Parameters
    ----------
    scheduler_list: list[(str, dict)]
        List of scheduler names and keyword arguments.

    optimizer: torch.optim.Optimizer
        Optimizer.

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler
        Scheduler.

    """
    if scheduler_list is None:
        return None
    scheduler = []
    for scheduler_name, scheduler_kwargs in scheduler_list:
        if scheduler_name == 'plateau':
            scheduler.append(ReduceLROnPlateau(optimizer, **scheduler_kwargs))
        elif scheduler_name == 'lambda':
            scheduler.append(LambdaLR(optimizer, **scheduler_kwargs))
        elif scheduler_name == 'onecycle':
            scheduler.append(OneCycleLR(optimizer, **scheduler_kwargs))
        elif scheduler_name == 'linear':
            scheduler.append(LinearLR(optimizer, **scheduler_kwargs))
        elif scheduler_name == 'cosine':
            scheduler.append(CosineAnnealingLR(optimizer, **scheduler_kwargs))
        else:
            raise ValueError(f'scheduler {scheduler_name} is not supported')
    if len(scheduler) == 1:
        return scheduler[0]
    else:
        return ChainedScheduler(scheduler)