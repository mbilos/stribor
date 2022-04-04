from typing import Union, List, Optional, Tuple
from torchtyping import TensorType

import torch
import torch.nn as nn

import stribor as st
from stribor.net import DiffeqNet

###################################################################
# Volume preserving, zero trace nets. Use `divergence=exact` in CNF
###################################################################

class DiffeqZeroTraceMLP(DiffeqNet):
    """
    Zero trace MLP transformation based on MADE network.

    Example:
    >>> dim = 3
    >>> net = stribor.net.DiffeqZeroTraceMLP(dim, [64], dim, return_log_det_jac=False)
    >>> x = torch.randn(32, dim)
    >>> t = torch.rand(32, 1)
    >>> net(t, x).shape
    torch.Size([32, 3])

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
        return_log_det_jac (bool, optional): Whether to return the log-Jacobian
            diagonal values (always zero). Default: True
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        return_log_det_jac: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__()
        self.return_log_det_jac = return_log_det_jac
        self.net1 = st.net.MADE(in_dim, hidden_dims, out_dim, natural_ordering=True,
                                reverse_ordering=False, return_per_dim=True)
        self.net2 = st.net.MADE(in_dim, hidden_dims, out_dim, natural_ordering=True,
                                reverse_ordering=True, return_per_dim=True)

    def forward(
        self, t: TensorType[..., 1], x: TensorType[..., 'dim'], **kwargs,
    ) -> Union[
        TensorType[..., 'out'],
        Tuple[TensorType[..., 'out'], TensorType[..., 'dim']]
    ]:
        y = self.net1(x, **kwargs) + self.net2(x, **kwargs)
        y = y.reshape(*y.shape[:-2], -1)
        return (y, torch.zeros_like(x)) if self.return_log_det_jac else y


def exclusive_sum_pooling(
    x: TensorType[..., 'N', 'D'], mask: Union[TensorType[..., 'N', 1], TensorType[..., 'N', 'D']],
) -> TensorType[..., 'N', 'D']:
    emb = x.sum(-2, keepdims=True)
    return emb - x

def exclusive_mean_pooling(
    x: TensorType[..., 'N', 'D'], mask: Union[TensorType[..., 'N', 1], TensorType[..., 'N', 'D']],
) -> TensorType[..., 'N', 'D']:
    emb = exclusive_sum_pooling(x, mask)
    N = mask.sum(-2, keepdim=True)
    y = emb / torch.max(N - 1, torch.ones_like(N))[0]
    return y

def exclusive_max_pooling(
    x: TensorType[..., 'N', 'D'], mask: Union[TensorType[..., 'N', 1], TensorType[..., 'N', 'D']],
) -> TensorType[..., 'N', 'D']:
    if x.shape[-2] == 1: # If only one element in set
        return torch.zeros_like(x)

    first, second = torch.topk(x, 2, dim=-2).values.chunk(2, dim=-2)
    indicator = (x == first).float()
    y = (1 - indicator) * first + indicator * second
    return y


class ZeroTraceEquivariantEncoder(nn.Module):
    """
    Pooling layer with zero trace Jacobian.

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
        pooling (str): Which type of pooling to use. Options: 'mean', 'max', 'sum'
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        pooling: str,
        **kwargs,
    ):
        super().__init__()
        self.pooling = pooling
        self.in_dim = in_dim
        self.set_emb = st.net.DiffeqMLP(in_dim + 1, hidden_dims, out_dim)

    def forward(
        self,
        t: TensorType[..., 1],
        x: TensorType[..., 'N', 'D'],
        mask: Optional[Union[TensorType[..., 'N', 1], TensorType[..., 'N', 'D']]] = None,
        **kwargs,
    ) -> TensorType[..., 'N', 'D', 'O']:
        if mask is None:
            mask = torch.ones(*x.shape[:-1], 1)
        else:
            mask = mask[...,0,None]

        x = self.set_emb(t, x) * mask
        if self.pooling == 'mean':
            y = exclusive_mean_pooling(x, mask)
        elif self.pooling == 'max':
            y = exclusive_max_pooling(x, mask)
        elif self.pooling == 'sum':
            y = exclusive_sum_pooling(x, mask)
        y = y.unsqueeze(-2).repeat_interleave(self.in_dim, dim=-2)
        return y


class DiffeqZeroTraceDeepSet(DiffeqNet):
    """
    Zero trace deepset transformation.

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
        pooling (str, optional): Which pooling to use. Default: 'max'
        return_log_det_jac (bool, optional): Whether to return the log-Jacobian
            diagonal values (always zero). Default: True
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        pooling: str = 'max',
        return_log_det_jac: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__()
        self.elementwise = st.net.MADE(in_dim, hidden_dims, out_dim, return_per_dim=True)
        self.interaction = ZeroTraceEquivariantEncoder(in_dim, hidden_dims, out_dim // in_dim, pooling)
        self.return_log_det_jac = return_log_det_jac

    def forward(
        self,
        t: TensorType[..., 1],
        x: TensorType[..., 'N', 'D'],
        mask: Optional[Union[TensorType[..., 'N', 1], TensorType[..., 'N', 'D']]] = None,
        latent: Optional[TensorType[...]] = None,
        **kwargs,
    ) -> Union[
        TensorType[..., 'N', 'O'],
        Tuple[TensorType[..., 'N', 'O'], TensorType[..., 'N', 'D']]
    ]:
        div = torch.zeros_like(x)
        if latent is not None:
            x = torch.cat([x, latent], -1)
        if mask is None:
            mask = torch.ones(*x.shape[:-1], 1).to(x)
        y = self.elementwise(x) + self.interaction(t, x, mask=mask)
        y = y * mask.unsqueeze(-1)
        y = y.reshape(*y.shape[:-2], -1)
        return (y, div) if self.return_log_det_jac else y


class DiffeqZeroTraceAttention(DiffeqNet):
    """
    Zero trace attention transformation.

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
        n_heads (int, optional): Number of heads in multihead attention. Default: 1
        return_log_det_jac (bool, optional): Whether to return the log-Jacobian
            diagonal values (always zero). Default: True
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        n_heads: int = 1,
        return_log_det_jac: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.return_log_det_jac = return_log_det_jac

        self.q = st.net.MADE(in_dim, hidden_dims[:-1], hidden_dims[-1] * in_dim, return_per_dim=True)
        self.k = st.net.MLP(in_dim, hidden_dims[:-1], hidden_dims[-1])
        self.v = st.net.MLP(in_dim, hidden_dims[:-1], hidden_dims[-1])
        self.proj = st.net.MLP(hidden_dims[-1], [], out_dim // in_dim)

    def forward(self,
        t: TensorType[..., 1],
        x: TensorType[..., 'N', 'D'],
        mask: Optional[TensorType[..., 'N', 'D']] = None,
        latent: Optional[TensorType[...]] = None,
        **kwargs,
    ) -> Union[
        TensorType[..., 'N', 'O'],
        Tuple[TensorType[..., 'N', 'O'], TensorType[..., 'N', 'D']]
    ]:
        query = self.q(x).transpose(-2, -3) # (B, N, D) -> (B, D, N, H)
        key = self.k(x).unsqueeze(-2).repeat_interleave(x.shape[-1], dim=-2).transpose(-2, -3) # (B, D, N, H)
        value = self.v(x).unsqueeze(-2).repeat_interleave(x.shape[-1], dim=-2).transpose(-2, -3) # (B, D, N, H)

        y = st.net.attention(query, key, value, self.n_heads, True, mask) # (B, D, N, H)
        y = y.transpose(-2, -3) # (B, N, D, H)
        y = self.proj(y).view(*y.shape[:-2], -1) # (B, N, D, O) -> (B, N, D * O)

        return (y, torch.zeros_like(x)) if self.return_log_det_jac else y
