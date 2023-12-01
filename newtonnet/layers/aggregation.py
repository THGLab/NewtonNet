import torch


def get_aggregation_by_string(key):
    if key == "sum":
        aggregation = torch.sum
    elif key == 'mean':
        aggregation = torch.mean
    elif key == 'max':
        aggregation = torch.max
    else:
        raise NotImplementedError(f'The aggregation function {key} is unknown.')
    return aggregation