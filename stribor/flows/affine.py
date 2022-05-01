from typing import Union, Optional, Tuple
from torchtyping import TensorType
from numbers import Number

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from stribor import Transform, ElementwiseTransform


class Affine(ElementwiseTransform):
    """
    Affine flow `y = a * x + b` where `a` and `b` are vectors and operations
    are applied elementwise.

    Example:
    >>> torch.manual_seed(123)
    >>> dim, latent_dim = 2, 50
    >>> f = st.Affine(dim, st.net.MLP(latent_dim, [64, 64], 2 * dim))
    >>> f(torch.ones(1, dim), latent=torch.ones(1, latent_dim))
    (tensor([[0.7575, 0.9410]], tensor([[-0.1745, -0.1350]])

    Args:
        dim (int): Dimension of data
        latent_net (nn.Module): Neural network that maps `[..., latent]` to `[..., 2 * dim]`
        scale (tensor): Scaling coefficient `a`
        shift (tensor): Shift coefficient `b`
    """
    def __init__(
        self,
        dim: int,
        *,
        latent_net: Optional[nn.Module] = None,
        scale: Union[Number, TensorType['dim']] = None,
        shift: Union[Number, TensorType['dim']] = None,
        **kwargs,
    ):
        super().__init__()

        self.latent_net = latent_net

        if latent_net is None:
            if scale is None:
                self.log_scale = nn.Parameter(torch.empty(1, dim))
                self.shift = nn.Parameter(torch.empty(1, dim))
                nn.init.xavier_uniform_(self.log_scale)
                nn.init.xavier_uniform_(self.shift)
            else:
                if isinstance(scale, Number):
                    scale = torch.Tensor([scale])
                    shift = torch.Tensor([shift])

                assert torch.all(scale > 0), '`scale` mush have positive values'
                self.log_scale = scale.log()
                self.shift = shift

    def _get_params(
        self,
        latent: Optional[TensorType[..., 'latent']],
    ) -> Tuple[TensorType[..., 'dim'], TensorType[..., 'dim']]:
        if self.latent_net is None:
            return self.log_scale, self.shift
        else:
            log_scale, shift = self.latent_net(latent).chunk(2, dim=-1)
            return log_scale, shift

    def forward(
        self,
        x: TensorType[..., 'dim'],
        latent: Optional[TensorType[..., 'latent']] = None,
        **kwargs,
    ) -> TensorType[..., 'dim']:
        y, _ = self.forward_and_log_det_jacobian(x, latent)
        return y

    def inverse(
        self,
        y: TensorType[..., 'dim'],
        latent: Optional[TensorType[..., 'latent']] = None,
        **kwargs,
    ) -> TensorType[..., 'dim']:
        x, _ = self.inverse_and_log_det_jacobian(y, latent)
        return x

    def log_det_jacobian(
        self,
        x: TensorType[..., 'dim'],
        y: Optional[TensorType[..., 'dim']] = None,
        latent: Optional[TensorType[..., 'latent']] = None,
        **kwargs,
    ) -> TensorType[..., 1]:
        _, log_det_jacobian = self.forward_and_log_det_jacobian(x, latent)
        return log_det_jacobian

    def forward_and_log_det_jacobian(self,
        x: TensorType[..., 'dim'],
        latent: Optional[TensorType[..., 'latent']] = None,
        *,
        reverse: bool = False,
        **kwargs,
    ) -> Tuple[TensorType[..., 'dim'], TensorType[..., 1]]:
        log_scale, shift = self._get_params(latent)
        if reverse:
            y = (x - shift) * torch.exp(-log_scale)
        else:
            y = x * torch.exp(log_scale) + shift
        return y, log_scale.expand_as(x).sum(-1, keepdim=True)

    def inverse_and_log_det_jacobian(self, y, latent=None, **kwargs):
        x, log_det_jacobian = self.forward_and_log_det_jacobian(y, latent, reverse=True)
        return x, -log_det_jacobian

    def log_diag_jacobian(
        self,
        x: TensorType[..., 'dim'],
        y: Optional[TensorType[..., 'dim']] = None,
        latent: Optional[TensorType[..., 'latent']] = None,
        **kwargs,
    ) -> TensorType[..., 'dim']:
        log_scale, _ = self._get_params(latent)
        return log_scale.expand_as(x)


class AffineLU(Transform):
    """
    Invertible linear layer `Wx+b` where `W=LU` is LU factorized.

    Args:
        dim: Dimension of input data
    """
    def __init__(self, dim: int, **kwargs):
        super().__init__()

        self.diag_ones = torch.eye(dim)

        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.log_diag = nn.Parameter(torch.empty(1, dim))
        self.bias = nn.Parameter(torch.empty(1, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.log_diag)
        nn.init.xavier_uniform_(self.bias)

    @property
    def L(self) -> TensorType['dim', 'dim']:
        return torch.tril(self.weight, -1) + self.diag_ones

    @property
    def U(self) -> TensorType['dim', 'dim']:
        return torch.triu(self.weight, 1) + self.diag_ones * self.log_diag.exp()

    def forward(self, x: TensorType[..., 'dim'], **kwargs) -> TensorType[..., 'dim']:
        return x @ (self.L @ self.U) + self.bias

    def inverse(self, y: TensorType[..., 'dim'], **kwargs) -> TensorType[..., 'dim']:
        x = y - self.bias
        x = torch.linalg.solve_triangular(self.U, x, upper=True, left=False)
        x = torch.linalg.solve_triangular(self.L, x, upper=False, left=False)
        return x

    def log_det_jacobian(
        self,
        x: TensorType[..., 'dim'],
        y: Optional[TensorType[..., 'dim']] = None,
        **kwargs,
    ) -> TensorType[..., 1]:
        return self.log_diag.expand_as(x).sum(-1, keepdim=True)

    def jacobian(
        self,
        x: TensorType[..., 'dim'],
        y: TensorType[..., 'dim'],
        **kwargs,
    ) -> TensorType[..., 'dim', 'dim']:
        return ((self.L @ self.U).T).expand(*x.shape[:-1], -1, -1)


class MatrixExponential(Transform):
    """
    Matrix exponential transformation `y = exp(W * t) @ x + b`.
    Corresponds to a solution of the linear ODE `dx/dt = W @ x`
    when `bias=False`.

    Example:
    >>> torch.manual_seed(123)
    >>> f = stribor.MatrixExponential(2)
    >>> x = torch.rand(1, 2)
    >>> f(x, t=1.)
    (tensor([[0.0798, 1.3169]], tensor([[-0.4994,  0.4619]])
    >>> f(x, t=torch.ones(1, 1))
    (tensor([[0.0798, 1.3169]], tensor([[-0.4994,  0.4619]])

    Args:
        dim (int): Dimension of data
    """
    def __init__(self, dim: int, bias: bool = False, **kwargs):
        super().__init__()
        self.dim = dim

        self._weight = nn.Parameter(torch.empty(dim, dim))
        self.diag = nn.Parameter(torch.empty(dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization from nn.Linear
        nn.init.kaiming_uniform_(self._weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.diag, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def lu(self):
        eye = torch.eye(self.dim).to(self._weight)
        L = torch.tril(self._weight, diagonal=-1) + eye
        U = torch.triu(self._weight) + eye
        return L, U

    @property
    def weight(self):
        # NOTE: Computation might be slow due to inverse
        L, U = self.lu()
        W = L @ U
        W_inv = torch.linalg.inv(W)
        return (W * self.diag) @ W_inv

    def get_time(self, t, shape):
        if isinstance(t, Number):
            t = torch.ones(*shape[:-1], 1) * t
        return t

    def forward(
        self,
        x: TensorType[..., 'dim'],
        t: Union[Number, TensorType[..., 1]] = 1.0,
        *,
        reverse: bool = False,
        **kwargs
    ) -> TensorType[..., 'dim']:
        if reverse:
            t = -t

        t = self.get_time(t, x.shape)

        L, U = self.lu()

        x = torch.linalg.solve_triangular(L, x.unsqueeze(-1), upper=False, unitriangular=True).squeeze(-1)
        x = torch.linalg.solve_triangular(U, x.unsqueeze(-1), upper=True, unitriangular=False).squeeze(-1)

        x = x * (self.diag * t).exp()

        x = F.linear(x, U)
        x = F.linear(x, L)

        if self.bias is not None:
            x += self.bias
        return x

    def inverse(
        self,
        y: TensorType[..., 'dim'],
        t: Union[Number, TensorType[..., 1]] = 1.0,
        **kwargs,
    ) -> TensorType[..., 'dim']:
        return self.forward(y, t=t, reverse=True)

    def log_det_jacobian(
        self,
        x: TensorType[..., 'dim'],
        y: Optional[TensorType[..., 'dim']] = None,
        t: TensorType[..., 'dim'] = 1.0,
        **kwargs,
    ) -> TensorType[..., 1]:
        t = self.get_time(t, x.shape)
        ldj = (self.weight * t.unsqueeze(-1)).diagonal(dim1=-2, dim2=-1)
        return ldj.expand_as(x).sum(-1, keepdim=True)

    def jacobian(
        self,
        x: TensorType[..., 'dim'],
        y: TensorType[..., 'dim'],
        t: TensorType[..., 'dim'] = 1.0,
        **kwargs,
    ) -> TensorType[..., 'dim', 'dim']:
        t = self.get_time(t, x.shape)
        W = torch.matrix_exp(self.weight * t.unsqueeze(-1))
        return W.expand(*x.shape[:-1], -1, -1)
