import nf
import torch
import torch.nn as nn


# REGULAR RESNET
class ResNetBlock(nn.Module):
    def __init__(self, dim, hidden_dims, activation, final_activation, **kwargs):
        super().__init__()
        self.net = nf.net.MLP(dim, hidden_dims, dim, activation, final_activation)

    def forward(self, x):
        return x + self.net(x)

class ResNet(nn.Module):
    def __init__(self, dim, hidden_dims, num_layers, activation, final_activation, **kwargs):
        super().__init__()
        blocks = []
        for _ in range(num_layers):
            blocks.append(ResNetBlock(dim, hidden_dims, activation, final_activation))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# INVERTIBLE RESNET
class IResNetBlock(nn.Module):
    def __init__(self, dim, hidden_dims, activation, final_activation, coeff, n_power_iterations, **kwargs):
        super().__init__()
        wrapper = lambda layer: nf.util.spectral_norm(layer, coeff, n_power_iterations=n_power_iterations)
        self.net = nf.net.MLP(dim, hidden_dims, dim, activation, final_activation, wrapper_func=wrapper)

    def forward(self, x):
        return x + self.net(x)

    def inverse(self, y, iterations=100):
        # fixed-point iteration
        x = y
        for _ in range(iterations):
            residual = self.net(x)
            x = y - residual
        return x

class IResNet(nn.Module):
    def __init__(self, dim, hidden_dims, num_layers, activation, final_activation,
                 coeff=0.97, n_power_iterations=5, **kwargs):
        super().__init__()
        blocks = []
        for _ in range(num_layers):
            blocks.append(IResNetBlock(dim, hidden_dims, activation, final_activation, coeff, n_power_iterations))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def inverse(self, y):
        for block in reversed(self.blocks):
            y = block.inverse(y)
        return y


# CONTINUOUS INVERTIBLE RESNET
class ContinuousIResNetBlock(nn.Module):
    def __init__(self, dim, hidden_dims, activation, final_activation, time_net, time_hidden_dim,
                 coeff, n_power_iterations, invertible=True, **kwargs):
        super().__init__()
        wrapper = None
        if invertible:
            wrapper = lambda layer: nf.util.spectral_norm(layer, coeff, n_power_iterations=n_power_iterations)
        self.net = nf.net.MLP(dim + 1, hidden_dims, dim, activation, final_activation, wrapper_func=wrapper)
        self.time_net = getattr(nf.net, time_net)(dim, hidden_dim=time_hidden_dim)

    def forward(self, x, t):
        return x + self.time_net(t) * self.net(torch.cat([x, t], -1))

    def inverse(self, y, t, iterations=100):
        if not self.invertible:
            raise NotImplementedError
        # fixed-point iteration
        x = y
        for _ in range(iterations):
            residual = self.time_net(t) * self.net(torch.cat([y, t], -1))
            x = y - residual
        return x

class ContinuousIResNet(nn.Module):
    def __init__(self, dim, hidden_dims, num_layers, activation, final_activation, time_net,
                 time_hidden_dim=None, coeff=0.97, n_power_iterations=5, invertible=True, **kwargs):
        super().__init__()
        blocks = []
        for _ in range(num_layers):
            blocks.append(ContinuousIResNetBlock(dim, hidden_dims, activation, final_activation, time_net,
                                                 time_hidden_dim, coeff, n_power_iterations, invertible))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, t):
        for block in self.blocks:
            x = block(x, t)
        return x

    def inverse(self, y, t):
        for block in reversed(self.blocks):
            y = block.inverse(y, t)
        return y
