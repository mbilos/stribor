import torch

def safe_softmax(x, dim=-1):
    """ Same as `torch.softmax` but returns 0
        instead of nan when whole row is -inf.
    """
    x = torch.softmax(x, dim)
    x = torch.nan_to_num(x)
    return x
