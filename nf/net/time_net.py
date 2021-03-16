import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeIdentity(nn.Module):
    def __init__(self, out_dim, **kwargs):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, t):
        assert t.shape[-1] == 1
        return t.repeat_interleave(self.out_dim, dim=-1)


class TimeLinear(nn.Module):
    def __init__(self, out_dim, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(1, out_dim))
        nn.init.xavier_uniform_(self.scale)

    def forward(self, t):
        return self.scale * t


class TimeTanh(TimeLinear):
    def forward(self, t):
        return torch.tanh(self.scale * t)


class TimeLog(TimeLinear):
    def forward(self, t):
        return torch.log(self.scale.exp() * t + 1)


class TimeFourier(nn.Module):
    def __init__(self, out_dim, hidden_dim, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_dim, hidden_dim))
        self.scale = nn.Parameter(torch.Tensor(out_dim, hidden_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.scale)

    def forward(self, t):
        t = t.unsqueeze(-1)
        t = F.softmax(self.weight, -1) * torch.sin(self.scale * t)
        t = t.sum(-1)
        return t
