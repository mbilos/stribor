from torchtyping import TensorType

import torch
import torch.nn.functional as F

from stribor import ElementwiseTransform


class Sigmoid(ElementwiseTransform):
    """
    Sigmoid transformation.

    Code adapted from torch.distributions.transforms
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, x: TensorType[..., 'dim'], **kwargs,
    ) -> TensorType[..., 'dim']:
        finfo = torch.finfo(x.dtype)
        y = torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1. - finfo.eps)
        return y

    def inverse(
        self, y: TensorType[..., 'dim'], **kwargs,
    ) -> TensorType[..., 'dim']:
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        x = y.log() - (-y).log1p()
        return x

    def log_det_jacobian(
        self, x: TensorType[..., 'dim'], y: TensorType[..., 'dim'], **kwargs,
    ) -> TensorType[..., 1]:
        return self.log_diag_jacobian(x, y, **kwargs).sum(-1, keepdim=True)

    def log_diag_jacobian(
        self, x: TensorType[..., 'dim'], y: TensorType[..., 'dim'], **kwargs,
    ) -> TensorType[..., 'dim']:
        return -F.softplus(-x) - F.softplus(x)

class Logit(Sigmoid):
    """
    Logit transformation. Inverse of sigmoid.
    """
    def forward(self, x, **kwargs):
        return super().inverse(x, **kwargs)

    def inverse(self, y, **kwargs):
        return super().forward(y, **kwargs)

    def log_diag_jacobian(
        self, x: TensorType[..., 'dim'], y: TensorType[..., 'dim'], **kwargs,
    ) -> TensorType[..., 'dim']:
        return -super().log_diag_jacobian(y, x) # Flip sign and order
