# Code adapted from Pyro
# https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/transforms/basic.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ELU(nn.Module):
    def forward(self, x, **kwargs):
        y = F.elu(x)
        ljd = -F.relu(-x)
        return y, ljd

    def inverse(self, y, **kwargs):
        zero = torch.zeros_like(y)
        log_term = torch.log1p(y + 1e-8)
        x = torch.max(y, zero) + torch.min(log_term, zero)
        ljd = F.relu(-log_term)
        return x, ljd

class LeakyReLU(nn.Module):
    def leaky_relu(self, x, t):
        zeros = torch.zeros_like(x)
        y = torch.max(zeros, x) + t * torch.min(zeros, x)
        ljd = torch.where(x >= 0., torch.zeros_like(x), torch.ones_like(x) * torch.log(t))
        return y, ljd

    def forward(self, x, t=0.01, reverse=False, **kwargs):
        if isinstance(t, int) or isinstance(t, float):
            t = torch.Tensor([t])
        else: # If t is tensor, treat it like time, identitiy at t=0
            t = 1 - torch.tanh(t)

        y, ljd = self.leaky_relu(x, 1 / t if reverse else t)
        return y, ljd

    def inverse(self, y, **kwargs):
        return self.forward(y, reverse=True, **kwargs)
