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

    Example:
    >> x = torch.rand(64, 2)
    >> f = nf.Coupling(nf.Affine(2, latent_dim=32), nf.net.MLP(2, [64], 32), mask='ordered_right_half')
    >> y, ljd = f(x) # returns output and log Jacobian diagonal, both with shapes (64, 2)
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
            return self.mask_func(N).unsqueeze(-1).expand(*rest, N, D).to(x)
        else:
            return self.mask_func(x.shape[-1]).expand_as(x).to(x)

    def forward(self, x, latent=None, **kwargs):
        mask = self.get_mask(x)

        z = self.net(x * mask)
        if latent is not None:
            z = torch.cat([z, latent], -1)
        y, ljd = self.flow.forward(x, latent=z)

        y = y * (1 - mask) + x * mask
        ljd = ljd * (1 - mask)
        return y, ljd

    def inverse(self, y, latent=None, **kwargs):
        mask = self.get_mask(y)

        z = self.net(y * mask)
        if latent is not None:
            z = torch.cat([z, latent], -1)
        x, ljd = self.flow.inverse(y, latent=z)

        x = x * (1 - mask) + y * mask # [0 | f(y)] + [y | 0]
        ljd = ljd * (1 - mask)
        return x, ljd


class ContinuousAffineCoupling(nn.Module):
    """
    Continuous affine coupling layer.
    Similar to `Coupling` but only applies affine transformation
    which now depends on time `t`.

    Args:
        net: Instance of `nf.net`. Output dim must be `2 * dim`.
        time_net: Instance of `nf.net.time_net`. Same output dim as `net`.
        mask: Mask name, e.g. 'ordered_right_half', see `nf.util.mask` for other options.
    """
    def __init__(self, dim, net, time_net, mask, **kwargs):
        super().__init__()

        self.dim = dim
        if dim == 1 and mask != 'none':
            raise ValueError('When dim=1, mask has to be `none`')
        self.net = net
        self.mask_func = nf.util.mask.get_mask(mask) # Initializes mask generator
        self.time_net = time_net

    def get_mask(self, x):
        return self.mask_func(x.shape[-1]).expand_as(x).to(x)

    def forward(self, x, t, latent=None, **kwargs):
        """ Input: x (..., dim), t (..., 1) """
        mask = self.get_mask(x)
        z = torch.cat([x * mask, t], -1)
        if self.dim == 1:
            z = z * 0
        if latent is not None:
            z = torch.cat([z, latent], -1)

        scale, shift = self.net(z).chunk(2, dim=-1)
        t_scale, t_shift = self.time_net(t).chunk(2, dim=-1)

        y = x * torch.exp(scale * t_scale) + shift * t_shift

        y = y * (1 - mask) + x * mask
        ljd = scale * t_scale * (1 - mask)

        return y, ljd

    def inverse(self, y, t, latent=None, **kwargs):
        mask = self.get_mask(y)
        z = torch.cat([y * mask, t], -1)
        if self.dim == 1:
            z = z * 0
        if latent is not None:
            z = torch.cat([z, latent], -1)

        scale, shift = self.net(z).chunk(2, dim=-1)
        t_scale, t_shift = self.time_net(t).chunk(2, dim=-1)

        x = (y - shift * t_shift) * torch.exp(-scale * t_scale)
        x = x * (1 - mask) + y * mask
        ljd = -scale * t_scale * (1 - mask)

        return x, ljd
