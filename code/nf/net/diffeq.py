import nf
import torch
import torch.nn as nn

SMOOTH_ACTIVATIONS = ['Tanh', 'Softplus', 'Elu', 'Identity', 'Sigmoid', None]

class DiffeqMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation='Tanh', final_activation=None, **kwargs):
        super().__init__()
        assert activation in SMOOTH_ACTIVATIONS and final_activation in SMOOTH_ACTIVATIONS
        self.net = nf.net.MLP(in_dim + 1, hidden_dims, out_dim, activation, final_activation, **kwargs)

    def forward(self, t, x, latent=None):
        t = torch.ones_like(x[:, :1]) * t
        input = torch.cat([t, x], 1)
        if latent is not None:
            input = torch.cat([input, latent], 1)
        return self.net(input)
