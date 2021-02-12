import nf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def exclusive_mean_pooling(x, mask):
    emb = x.sum(-2, keepdims=True)
    N = mask.sum(-2, keepdim=True)
    y = (emb - x) / torch.max(N, torch.ones_like(N))[0]
    return y

def exclusive_max_pooling(x, mask):
    if x.shape[-2] == 1: # If only one element in set
        return torch.zeros_like(x)

    first, second = torch.topk(x, 2, dim=-2).values.chunk(2, dim=-2)
    indicator = (x == first).float()
    y = (1 - indicator) * first + indicator * second
    return y

class ExclusivePooling(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, pooling, **kwargs):
        super().__init__()
        self.pooling = pooling
        self.in_dim = in_dim
        self.set_emb = nf.net.DiffeqMLP(in_dim + 1, hidden_dims, out_dim)

    def forward(self, t, x, mask=None, **kwargs):
        if mask is None:
            mask = torch.ones(*x.shape[:-1], 1)
        else:
            mask = mask[...,0,None]

        x = self.set_emb(t, x) * mask
        if self.pooling == 'mean':
            y = exclusive_mean_pooling(x, mask)
        elif self.pooling == 'max':
            y = exclusive_max_pooling(x, mask)
        y = y.unsqueeze(-2).repeat_interleave(self.in_dim, dim=-2)
        return y
