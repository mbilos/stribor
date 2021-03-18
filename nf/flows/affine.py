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


class AffinePLU(nn.Module):
    """
    Affine layer Wx+b where W=PLU correspoding to PLU factorization.

    Args:
        dim: Dimension of input data.
    """
    def __init__(self, dim, **kwargs):
        super().__init__()

        self.P = torch.eye(dim)[torch.randperm(dim)]
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        self.log_diag = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1, dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.log_diag)
        nn.init.xavier_uniform_(self.bias)

    def get_LU(self):
        eye = torch.eye(self.weight.shape[1])
        L = torch.tril(self.weight, -1) + eye
        U = torch.triu(self.weight, 1) + eye * self.log_diag.exp()
        return L, U

    def forward(self, x, **kwargs):
        """ Input: x (..., dim) """
        L, U = self.get_LU()
        y = (self.P @ (L @ (U @ x.unsqueeze(-1)))).squeeze(-1) + self.bias
        ljd = self.log_diag.expand_as(x)
        return y, ljd

    def inverse(self, y, **kwargs):
        L, U = self.get_LU()
        y = torch.triangular_solve(self.P.T @ (y - self.bias).unsqueeze(-1), L, upper=False)[0]
        x = torch.triangular_solve(y, U, upper=True)[0].squeeze(-1)
        ljd = -self.log_diag.expand_as(x)
        return x, ljd


class ContinuousAffinePLU(nn.Module):
    """
    Continuous version of `AffinePLU` without P.
    Weights and biases depend on time.

    Args:
        dim: Dimension of input data.
        time_net: Instance of `nf.net.time_net`.
            Must have output of size `dim^2 + dim`.
    """
    def __init__(self, dim, time_net, **kwargs):
        super().__init__()

        self.dim = dim
        self.time_net = time_net

    def get_params(self, t):
        params = self.time_net(t)
        b, W = params[...,:self.dim], params[...,self.dim:]
        W = W.view(*W.shape[:-1], self.dim, self.dim)

        eye = torch.eye(self.dim)
        log_D = W.diagonal(dim1=-2, dim2=-1)
        L = torch.tril(W, -1) + eye
        U = torch.triu(W, 1) + eye * log_D.unsqueeze(-1).exp()

        return L, log_D, U, b

    def forward(self, x, t, **kwargs):
        """ Input: x (..., dim); t (..., 1) """
        L, log_D, U, b = self.get_params(t)
        y = (L @ (U @ x.unsqueeze(-1))).squeeze(-1) + b
        ljd = log_D.expand_as(x)
        return y, ljd

    def inverse(self, y, t, **kwargs):
        L, log_D, U, b = self.get_params(t)
        y = torch.triangular_solve((y - b).unsqueeze(-1), L, upper=False)[0]
        x = torch.triangular_solve(y, U, upper=True)[0].squeeze(-1)
        ljd = -log_D.expand_as(x)
        return x, ljd


class AffineExponential(nn.Module):
    def __init__(self, dim, bias=True, **kwargs):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, dim))
            nn.init.xavier_uniform_(self.bias)

    def get_time(self, t, x):
        if isinstance(t, float) or isinstance(t, int):
            t = torch.ones(*x.shape[:-1], 1) * t
        return t.unsqueeze(-1)

    def forward(self, x, t=1, **kwargs):
        """ Input: x (..., dim); t (..., 1) """
        t = self.get_time(t, x)

        y = (torch.matrix_exp(self.weight * t) @ x.unsqueeze(-1)).squeeze(-1)
        if hasattr(self, 'bias'):
            y = y + self.bias * t.squeeze(-1)

        ljd = (self.weight * t).diagonal(dim1=-2, dim2=-1)
        return y, ljd

    def inverse(self, y, t=1, **kwargs):
        t = -self.get_time(t, y)

        if hasattr(self, 'bias'):
            y = y - self.bias * t.squeeze(-1).abs()
        x = (torch.matrix_exp(self.weight * t) @ y.unsqueeze(-1)).squeeze(-1)

        ljd = (self.weight * t).diagonal(dim1=-2, dim2=-1)
        return x, ljd
