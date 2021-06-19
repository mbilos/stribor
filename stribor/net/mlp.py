import torch
import torch.nn as nn

class MLP(nn.Module):
    """ Simple multi-layer perceptron.
    Args:
        in_dim: (int) Last dimension of the input tensor
        hidden_dims: (list) Hidden dimensions of the layers
        out_dim: (int) Last dimenstion of the output tensor
        activation: (str) Name of the activation function from `torch.nn`
        final_activation: (str) Name of the activation function from `torch.nn`
        wrapper_func: (callable) Wrapper function for `nn.Linear`, e.g. spectral_norm
    """
    def __init__(self, in_dim, hidden_dims, out_dim, activation='Tanh',
                 final_activation=None, wrapper_func=None, **kwargs):
        super().__init__()

        if not wrapper_func:
            wrapper_func = lambda x: x

        hidden_dims = hidden_dims[:]
        hidden_dims.append(out_dim)
        layers = [nn.Linear(in_dim, hidden_dims[0])]

        for i in range(len(hidden_dims) - 1):
            layers.append(getattr(nn, activation)())
            layers.append(wrapper_func(nn.Linear(hidden_dims[i], hidden_dims[i+1])))
        layers[-1].bias.data.fill_(0.0)

        if final_activation is not None:
            layers.append(getattr(nn, final_activation)())

        self.net = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        """ For input (..., in_dim) returns (..., out_dim) """
        return self.net(x)
