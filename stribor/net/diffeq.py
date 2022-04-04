from typing import List, Union
from torchtyping import TensorType
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import stribor as st

#####################################################
# Use `divergence=approximate` in ContinuousTransform
#####################################################

class DiffeqNet(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(
        self,
        t: TensorType[1],
        x: TensorType[..., 'dim'],
        latent: TensorType[..., 'latent'] = None,
        **kwargs,
    ) -> TensorType[..., 'out']:
        pass


class DiffeqConcat(DiffeqNet):
    """
    Differential equation that concatenates the input with time.

    Args:
        net (nn.Module): Neural network with `dim + 1 (+ latent)`
        input size and `dim` output size.
    """
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(
        self,
        t: TensorType[1],
        x: TensorType[..., 'dim'],
        latent: TensorType[..., 'latent'] = None,
        **kwargs,
    ) -> TensorType[..., 'out']:
        t = torch.ones_like(x[..., :1]) * t
        input = torch.cat([t, x], -1)
        if latent is not None:
            input = torch.cat([input, latent], -1)
        return self.net(input, **kwargs)


class DiffeqMLP(DiffeqConcat):
    """
    Differential equation defined with MLP.

    Example:
    >>> batch, dim = 32, 3
    >>> net = stribor.net.DiffeqMLP(dim + 1, [64, 64], dim)
    >>> x = torch.randn(batch, dim)
    >>> t = torch.rand(batch, 1)
    >>> net(t, x).shape
    torch.Size([32, 3])

    Args:
        Same as in `st.net.MLP`
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        activation: str = 'Tanh',
        final_activation: str = None,
        **kwargs,
    ):
        super().__init__(st.net.MLP(in_dim, hidden_dims, out_dim, activation, final_activation))


class DiffeqDeepset(DiffeqConcat):
    """
    Differential equation defined with permutation equivariant network.

    Args:
        Same as in `st.net.EquivariantNet`
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        activation: str = 'Tanh',
        final_activation: str = None,
        **kwargs,
    ):
        super().__init__(st.net.EquivariantNet(in_dim, hidden_dims, out_dim, activation, final_activation))


class DiffeqSelfAttention(DiffeqConcat):
    """
    Differential equation defined with self attention network.

    Args:
        Same as in `st.net.SelfAttention`
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: List[int],
        out_dim: int,
        n_heads: int = 1,
        mask_diagonal: bool = False,
        **kwargs,
    ):
        super().__init__(st.net.SelfAttention(in_dim, hidden_dim, out_dim, n_heads, mask_diagonal))
