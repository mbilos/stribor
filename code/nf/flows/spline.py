# Implementation of "Neural Spline Flows" (https://arxiv.org/abs/1906.04032)
# Code adapted from https://github.com/bayesiains/nsf

import nf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class Spline(nn.Module):
    """ Spline flow.
    Elementwise transformation of input vector using spline functions
    defined inside the rectangle between (lower,lower) and (upper,upper) points.

    Args:
        n_bins: Number of bins/spline knots
        lower: Lower bound of the spline transformation domain.
        upper: Upper bound
        spline_type: Which spline function, quadratic is recommended.
        latent_dim: Dimension of the input latent vector
        init: Lower and upper bound for parameter uniform initialization
    """
    def __init__(self, dim, n_bins=5, lower=0, upper=1, spline_type='quadratic',
                 latent_dim=None, init=0.001, **kwargs):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.dim = dim
        self.n_bins = n_bins

        if spline_type == 'quadratic':
            self.spline = nf.util.unconstrained_rational_quadratic_spline
            self.derivative_dim = n_bins + 1
        elif spline_type == 'cubic':
            self.spline = nf.util.unconstrained_cubic_spline
            self.derivative_dim = 2
        else:
            raise ValueError('spline_type must be either quadratic or cubic')

        if latent_dim is None:
            self.width = nn.Parameter(torch.Tensor(dim, n_bins).uniform_(-init, init))
            self.height = nn.Parameter(torch.randn(dim, n_bins).uniform_(-init, init))
            self.derivative = nn.Parameter(torch.randn(dim, self.derivative_dim).uniform_(-init, init))
        else:
            self.proj = nn.Linear(latent_dim, dim * (n_bins * 2 + self.derivative_dim))
            self.proj.weight.data.uniform_(-init, init)
            self.proj.bias.data.fill_(0)

    def get_params(self, latent):
        if latent is None:
            return self.width, self.height, self.derivative
        else:
            params = self.proj(latent)
            params = params.view(*params.shape[:-1], self.dim, self.n_bins * 2 + self.derivative_dim)
            width = params[...,:self.n_bins]
            height = params[...,self.n_bins:2*self.n_bins]
            derivative = params[...,2*self.n_bins:]
            return width, height, derivative

    def forward(self, x, latent=None, **kwargs):
        w, h, d = self.get_params(latent)
        return self.spline(x, w, h, d, inverse=False, lower=self.lower, upper=self.upper)

    def inverse(self, y, latent=None, **kwargs):
        w, h, d = self.get_params(latent)
        return self.spline(y, w, h, d, inverse=True, lower=self.lower, upper=self.upper)
