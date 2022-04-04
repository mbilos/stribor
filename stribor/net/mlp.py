from typing import Callable, List
from torchtyping import TensorType

import torch.nn as nn

class MLP(nn.Module):
    """
    Simple multi-layer neural network.

    Example:
    >>> torch.manual_seed(123)
    >>> net = stribor.net.MLP(2, [64, 64], 1)
    >>> net(torch.randn(1, 2))
    tensor([[-0.0132]], grad_fn=<AddmmBackward>)

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
        activation (str, optional): Activation function from `torch.nn`.
            Default: 'Tanh'
        final_activation (str, optional): Last activation. Default: None
        nn_linear_wrapper_func (callable, optional): Wrapper function for `nn.Linear`,
            e.g. st.util.spectral_norm. Default: None
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        activation: str = 'Tanh',
        final_activation: str = None,
        nn_linear_wrapper_func: Callable = None,
        **kwargs,
    ):
        super().__init__()

        if not nn_linear_wrapper_func:
            nn_linear_wrapper_func = lambda x: x

        hidden_dims = hidden_dims[:]
        hidden_dims.append(out_dim)
        layers = [nn.Linear(in_dim, hidden_dims[0])]

        for i in range(len(hidden_dims) - 1):
            layers.append(getattr(nn, activation)())
            layers.append(nn_linear_wrapper_func(nn.Linear(hidden_dims[i], hidden_dims[i+1])))
        layers[-1].bias.data.fill_(0.0)

        if final_activation is not None:
            layers.append(getattr(nn, final_activation)())

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: TensorType[..., 'dim'],
        **kwargs,
    ) -> TensorType[..., 'out']:
        return self.net(x)
