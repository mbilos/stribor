import stribor as st
import torch
import torch.nn as nn

class EquivariantLayer(nn.Module):
    """
    Similar to "On Learning Sets of Symmetric Elements" (https://arxiv.org/abs/2002.08599)
    and "Deep Sets" (https://arxiv.org/abs/1703.06114) etc.

    Args:
        in_dim: Dimension of input
        out_dim: Dimension of output
    """
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.l1 = nn.Linear(in_dim, out_dim)
        self.l2 = nn.Linear(in_dim, out_dim)

    def forward(self, x, mask=None, **kwargs):
        y1 = self.l1(x)
        y2 = self.l2(x.sum(-2, keepdim=True))

        if mask is not None:
            mask = mask[...,0,None]
            y1 = y1 * mask
            y2 = y2 * mask / mask.sum(-2, keepdim=True)
        else:
            y2 = y2 / x.shape[-2]

        return y1 + y2

class EquivariantNet(nn.Module):
    """
    Neural network with equivariant layer. Given set input of shape (..., N, dim)
    it will produce output of same shape with permutation invariant layers.
    Same as MLP from `st.net` but with `EquivariantLayer` instead of `nn.Linear`.

    Args:
        in_dim: (int) Last dimension of the input tensor
        hidden_dims: (list) Hidden dimensions corresponding to layers
        out_dim: (int) Last dimenstion of the output tensor
        activation: (str) Name of the activation function from `torch.nn`
        final_activation: (str) Name of the activation function from `torch.nn`
    """
    def __init__(self, in_dim, hidden_dims, out_dim, activation='Tanh', final_activation=None, **kwargs):
        super().__init__()

        self.activation = getattr(nn, activation)()
        self.final_activation = getattr(nn, final_activation)() if final_activation else nn.Identity()

        hidden_dims = [in_dim] + hidden_dims + [out_dim]
        self.layers = []
        for in_, out_ in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.append(EquivariantLayer(in_, out_))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, mask=None, **kwargs):
        """ For input (..., N, in_dim) returns (..., N, out_dim) """
        for layer in self.layers[:-1]:
            x = layer(x, mask=mask)
            x = self.activation(x)
        x = self.layers[-1](x, mask=mask)
        x = self.final_activation(x)
        return x
