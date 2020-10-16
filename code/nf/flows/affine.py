import torch
import torch.nn as nn


class Identity(nn.Module):
    """
    Identity flow.
    Doesn't change input, determinant is 1.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        return x, torch.zeros_like(x)

    def inverse(self, y, **kwargs):
        return y, torch.zeros_like(y)


class Affine(nn.Module):
    """
    Affine flow.
    `y = a * x + b`

    Args:
        dim: Distribution domain dimension.
        latent_dim: Dimension of latent input (Default: None)
    """
    def __init__(self, dim, latent_dim=None, **kwargs):
        super().__init__()

        if latent_dim is None:
            bound = 1 / (dim)**0.05
            self.log_scale = nn.Parameter(torch.Tensor(1, dim).uniform_(-bound, bound))
            self.shift = nn.Parameter(torch.Tensor(1, dim).uniform_(-bound, bound))
        else:
            self.proj = nn.Linear(latent_dim, dim * 2)
            self.proj.bias.data.fill_(0.0)

    def get_params(self, latent):
        if latent is None:
            return self.log_scale, self.shift
        else:
            log_scale, shift = self.proj(latent).chunk(2, dim=-1)
            return log_scale, shift

    def forward(self, x, latent=None, **kwargs):
        log_scale, shift = self.get_params(latent)
        y = x * torch.exp(log_scale) + shift
        return y, log_scale.expand_as(y)

    def inverse(self, y, latent=None, **kwargs):
        log_scale, shift = self.get_params(latent)
        x = (y - shift) * torch.exp(-log_scale)
        return x, -log_scale.expand_as(x)


class AffineFixed(nn.Module):
    """
    Fixed affine flow with predifined unlearnable parameters.

    Args:
        shift: List of floats, length equal to data dim.
        scale: Same as shift, but all values have to be >0.
    """
    def __init__(self, scale, shift, **kwargs):
        super().__init__()

        self.scale = torch.Tensor(scale)
        self.shift = torch.Tensor(shift)

        assert (self.scale >= 0).all()

    def forward(self, x, **kwargs):
        y = self.scale * x + self.shift
        return y, self.scale.log().expand_as(x)

    def inverse(self, y, **kwargs):
        x = (y - self.shift) / self.scale
        return x, -self.scale.log().expand_as(x)
