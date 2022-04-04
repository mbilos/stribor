from torchtyping import TensorType

import torch
from stribor import ElementwiseTransform


class Identity(ElementwiseTransform):
    """
    Identity transformation.
    Doesn't change the input, log-Jacobian is 0.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: TensorType[..., 'dim'], **kwargs) -> TensorType[..., 'dim']:
        return x

    def inverse(self, y: TensorType[..., 'dim'], **kwargs) -> TensorType[..., 'dim']:
        return y

    def log_det_jacobian(self, x: TensorType[..., 'dim'], y: TensorType[..., 'dim'], **kwargs) -> TensorType[..., 1]:
        return torch.zeros_like(x[...,:1])

    def log_diag_jacobian(self, x: TensorType[..., 'dim'], y: TensorType[..., 'dim'], **kwargs) -> TensorType[..., 'dim']:
        return torch.zeros_like(x)
