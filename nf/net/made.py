# Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
# Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MADE']

class MaskedLinear(nn.Linear):
    """ Same as Linear except has a configurable mask on the weights. """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation='Tanh', final_activation=None,
                 num_masks=1, natural_ordering=False, reverse_ordering=False, return_per_dim=False, **kwargs):
        """
        out_dim: integer; Number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if out_dim is e.g. 2x larger than in_dim (perhaps the mean and std), then the first in_dim
              will be all the means and the second in_dim will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of runin_dimg the tests for this file makes this a bit more clear with examples.
        num_masks: Can be used to train ensemble over orderings/connections
        natural_ordering: Force natural ordering of dimensions, don't use random permutations
        return_per_dim: Whether to return in (..., in_dim, out_dim / in_dim) format, with correct assignment
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.return_per_dim = return_per_dim
        assert self.out_dim % self.in_dim == 0, "out_dim must be integer multiple of in_dim"

        # define a simple MLP neural net
        self.net = []
        hs = [in_dim] + self.hidden_dims + [out_dim]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.append(MaskedLinear(h0, h1))
            self.net.append(getattr(nn, activation)())
        self.net.pop() # pop the last activation for the output layer
        if final_activation is not None:
            self.net.append(getattr(nn, final_activation)())

        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.reverse_ordering = reverse_ordering
        self.num_masks = num_masks
        self.seed = 0 # for cycling through num_masks orderings

        self.m = {}
        self.update_masks() # builds the initial self.m connectivity

    def update_masks(self):
        if self.m and self.num_masks == 1:
            return # only a single seed, skip for efficiency

        L = len(self.hidden_dims)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        if self.natural_ordering:
            self.m[-1] = np.arange(self.in_dim)
            if self.reverse_ordering:
                self.m[-1] = self.m[-1][::-1]
        else:
            self.m[-1] = rng.permutation(self.in_dim)

        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), max(self.in_dim-1, 1), size=self.hidden_dims[l])

        # construct the mask matrices
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < self.m[-1][None,:])

        # handle the case where out_dim = in_dim * k, for integer k > 1
        if self.out_dim > self.in_dim:
            k = int(self.out_dim / self.in_dim)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x, **kwargs):
        original_shape = x.shape

        x = x.view(-1, original_shape[-1])
        y = self.net(x)

        y = y.view(*original_shape[:-1], -1, self.in_dim).transpose(-1, -2)

        return y if self.return_per_dim else y.reshape(*y.shape[:-2], -1)
