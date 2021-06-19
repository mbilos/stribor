import torch
import torch.nn as nn
import torch.nn.functional as F


def diff(x, dim=-1):
    """ Inverse of x.cumsum(dim=dim).
    Compute differences between subsequent elements of the tensor.
    Only works on dims -1 and -2.

    Args:
        x: Input tensor of arbitrary shape.
    Returns:
        diff: Tensor of the the same shape as x.
    """
    if dim == 1:
        if x.dim() == 2:
            dim = -1
        elif x.dim() == 3:
            dim = -2
        else:
            raise ValueError('If dim=1, tensor must have 2 or 3 dimensions')

    if dim == 2:
        if x.dim() == 3:
            dim = -1
        elif x.dim() == 4:
            dim = -2
        else:
            raise ValueError('If dim=2, tensor should have 3 or 4 dimensions')

    if dim == -1:
        return x - F.pad(x, (1, 0))[..., :-1]
    elif dim == -2:
        return x - F.pad(x, (0, 0, 1, 0))[..., :-1, :]
    else:
        raise ValueError("dim must be equal to -1 or -2")


class Cumsum(nn.Module):
    def __init__(self, dim):
        """Compute cumulative sum along the specified dimension of the tensor.
        Args:
            dim: Tensor dimension over which to perform the summation, -1 or -2.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x, **kwargs):
        y = x.cumsum(self.dim)
        return y, torch.zeros_like(y)

    def inverse(self, y, **kwargs):
        x = diff(y, self.dim)
        return x, torch.zeros_like(x)

class Diff(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.base_flow = Cumsum(*args, **kwargs)

    def forward(self, x, **kwargs):
        return self.base_flow.inverse(x, **kwargs)

    def inverse(self, x, **kwargs):
        return self.base_flow.forward(x, **kwargs)


class CumsumAxis(nn.Module):
    def __init__(self, axis):
        """Compute cumulative sum along the specified axis in the last dimension.
        Args:
            axis: Axis in the last dimension over which to perform the summation.
        """
        super().__init__()
        self.axis = axis

    def forward(self, x, **kwargs):
        y = x.clone()
        y[..., self.axis] = y[..., self.axis].cumsum(-1)
        return y, torch.zeros_like(y)

    def inverse(self, y, **kwargs):
        x = y.clone()
        x[..., self.axis] = diff(x[..., self.axis], -1)
        return x, torch.zeros_like(x)

class DiffAxis(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.base_flow = CumsumAxis(*args, **kwargs)

    def forward(self, x, **kwargs):
        return self.base_flow.inverse(x, **kwargs)

    def inverse(self, x, **kwargs):
        return self.base_flow.forward(x, **kwargs)
