import torch
import torch.nn as nn
import torch.nn.functional as F


class Reverse(nn.Module):
    def __init__(self, dims=[-1]):
        super().__init__()
        self.dims = dims

    def forward(self, x, **kwargs):
        y = torch.flip(x, self.dims)
        return y, torch.zeros_like(y)

    def inverse(self, y, **kwargs):
        x = torch.flip(y, self.dims)
        return x, torch.zeros_like(x)
