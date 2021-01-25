import nf
import torch
import torch.nn as nn
import torch.nn.functional as F


class Coupling(nn.Module):
    """
    Coupling layer.
    Splits data into 2 parts based on mask. One part generates
    parameters of the flow that will transform the rest.
    Efficient (identical) computation in both directions.

    Args:
        flow: Elementwise flow, should have `latent_dim` the same size as `net` output.
        net: Instance of `nf.net`.
        mask: Mask name, e.g. 'ordered_right_half', see `nf.util.mask` for other options.
        set_data: If data is in (..., N, dim) form.
    """
    def __init__(self, flow, net, mask, set_data=False, **kwargs):
        super().__init__()

        self.flow = flow
        self.net = net
        self.mask_func = nf.util.mask.get_mask(mask) # Initializes mask generator
        self.set_data = set_data

    def get_mask(self, x):
        if self.set_data:
            *rest, N, D = x.shape
            return self.mask_func(N).unsqueeze(-1).expand(*rest, N, D)
        else:
            return self.mask_func(x.shape[-1]).expand_as(x)

    def forward(self, x, latent=None, **kwargs):
        mask = self.get_mask(x)

        z = self.net(x * mask)
        if latent is not None:
            z = torch.cat([z, latent], -1)
        y, log_jac = self.flow.forward(x, latent=z)

        y = y * (1 - mask) + x * mask
        log_jac = log_jac * (1 - mask)
        return y, log_jac

    def inverse(self, y, latent=None, **kwargs):
        mask = self.get_mask(y)

        z = self.net(y * mask)
        if latent is not None:
            z = torch.cat([z, latent], -1)
        x, log_jac = self.flow.inverse(y, latent=z)

        x = x * (1 - mask) + y * mask # [0 | f(y)] + [y | 0]
        log_jac = log_jac * (1 - mask)
        return x, log_jac
