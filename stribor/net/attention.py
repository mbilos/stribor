import stribor as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value, n_heads=1, mask_diagonal=False, mask=None):
    *rest, D = query.shape
    query = query.view(*rest, n_heads, D // n_heads).transpose(-2, -3)
    value = value.view(*rest, n_heads, D // n_heads).transpose(-2, -3)
    key = key.view(*rest, n_heads, D // n_heads).transpose(-2, -3)

    att = query @ key.transpose(-1, -2) * (1 / key.shape[-1])**0.5

    if mask_diagonal:
        att.masked_fill_(torch.eye(att.shape[-1]).bool(), -np.inf)
    if mask is not None:
        att_mask = 1 - mask.transpose(-1, -2).unsqueeze(-2).repeat_interleave(att.shape[-2], dim=-2)
        att.masked_fill_(att_mask.bool(), -np.inf)

    att = st.util.safe_softmax(att, -1)

    y = att @ value

    y = y.transpose(-2, -3).reshape(*rest, -1)

    if mask is not None:
        y = y * mask
    return y

class Attention(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, n_heads=1, mask_diagonal=False, **kwargs):
        super().__init__()

        self.mask_diagonal = mask_diagonal
        self.n_heads = n_heads
        self._shape = (n_heads, hidden_dims[-1] // n_heads)

        self.key = st.net.MLP(in_dim, hidden_dims[:-1], hidden_dims[-1])
        self.query = st.net.MLP(in_dim, hidden_dims[:-1], hidden_dims[-1])
        self.value = st.net.MLP(in_dim, hidden_dims[:-1], hidden_dims[-1])
        self.proj = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, query, key, value, mask=None, **kwargs):
        y = attention(self.query(query), self.key(key), self.value(value), self.n_heads, self.mask_diagonal, mask)
        return self.proj(y)

class SelfAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_heads=1, mask_diagonal=False, **kwargs):
        super().__init__()
        self.attention = Attention(in_dim, hidden_dim, out_dim, n_heads, mask_diagonal)

    def forward(self, x, mask=None, **kwargs):
        y = self.attention(x, x, x, mask=mask, **kwargs)
        return y

class InducedSelfAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_heads=1, n_points=32, **kwargs):
        super().__init__()

        self.att1 = Attention(in_dim, hidden_dim, in_dim, n_heads)
        self.att2 = Attention(in_dim, hidden_dim, out_dim, n_heads)
        self.points = nn.Parameter(torch.Tensor(1, n_points, in_dim).uniform_(-1., 1.))

    def forward(self, x, mask=None, **kwargs):
        h = self.points.repeat(x.shape[0], 1, 1)
        h = self.att1(h, x, x, mask=mask, **kwargs)
        y = self.att2(x, h, h, **kwargs)
        return y
