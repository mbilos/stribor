from typing import Optional
from torchtyping import TensorType

import math
import torch
import torch.nn.functional as F
from torch.distributions import constraints

from stribor import ElementwiseTransform


class ELU(ElementwiseTransform):
    """
    Exponential linear unit.
    Adapted from Pyro (https://pyro.ai).
    """

    bijective = True
    domain: constraints.real
    codomain: constraints.positive

    def forward(
        self,
        x: TensorType[..., 'dim'],
        **kwargs,
    ) -> TensorType[..., 'dim']:
        return F.elu(x)

    def inverse(
        self,
        y: TensorType[..., 'dim'],
        **kwargs,
    ) -> TensorType[..., 'dim']:
        zero = torch.zeros_like(y)
        log_term = torch.log1p(y)
        x = torch.max(y, zero) + torch.min(log_term, zero)
        return x

    def log_det_jacobian(
        self,
        x: TensorType[..., 'dim'],
        y: Optional[TensorType[..., 'dim']] = None,
    ) -> TensorType[..., 1]:
        log_diag_jacobian = self.log_diag_jacobian(x, y)
        return log_diag_jacobian.sum(-1, keepdim=True)

    def jacobian(
        self,
        x: TensorType[..., 'dim'],
        y: TensorType[..., 'dim'],
        **kwargs,
    ) -> TensorType[..., 'dim', 'dim']:
        return torch.diag_embed(self.log_diag_jacobian(x, y).exp())

    def log_diag_jacobian(
        self,
        x: TensorType[..., 'dim'],
        y: TensorType[..., 'dim'],
        **kwargs,
    ) -> TensorType[..., 'dim']:
        return -F.relu(-x)


class LeakyReLU(ElementwiseTransform):
    """
    Leaky ReLU.
    For `x >= 0` returns `x`, else returns `negative_slope * x`.
    Adapted from Pyro (https://pyro.ai).

    Args:
        negative_slope (float): Controls the angle of the negative slope. Default: 0.01
    """

    bijective = True
    domain: constraints.real
    codomain: constraints.real

    def __init__(self, negative_slope: float = 0.01, **kwargs):
        super().__init__()
        assert negative_slope > 0, '`negative_slope` must be positive'
        self.negative_slope = negative_slope

    def _leaky_relu(
        self,
        x: TensorType[..., 'dim'],
        negative_slope: float,
    ) -> TensorType[..., 'dim']:
        zeros = torch.zeros_like(x)
        y = torch.max(zeros, x) + negative_slope * torch.min(zeros, x)
        return y

    def forward(self, x: TensorType[..., 'dim'], **kwargs) -> TensorType[..., 'dim']:
        return self._leaky_relu(x, self.negative_slope)

    def inverse(self, y: TensorType[..., 'dim'], **kwargs) -> TensorType[..., 'dim']:
        return self._leaky_relu(y, 1 / self.negative_slope)

    def log_det_jacobian(
        self,
        x: TensorType[..., 'dim'],
        y: Optional[TensorType[..., 'dim']] = None,
    ) -> TensorType[..., 1]:
        return self.log_diag_jacobian(x, y).sum(-1, keepdim=True)

    def jacobian(
        self,
        x: TensorType[..., 'dim'],
        y: TensorType[..., 'dim'],
        **kwargs,
    ) -> TensorType[..., 'dim', 'dim']:
        return torch.diag_embed(self.log_diag_jacobian(x, y).exp())

    def log_diag_jacobian(
        self,
        x: TensorType[..., 'dim'],
        y: TensorType[..., 'dim'],
        **kwargs,
    ) -> TensorType[..., 'dim']:
        left = torch.zeros_like(x)
        right = torch.ones_like(x) * math.log(self.negative_slope)
        return torch.where(x >= 0., left, right)
