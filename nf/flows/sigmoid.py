# https://github.com/pytorch/pytorch/blob/master/torch/distributions/transforms.py#L349

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sigmoid(nn.Module):
    """ Sigmoid flow. """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        finfo = torch.finfo(x.dtype)
        y = torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1. - finfo.eps)
        log_jac = -F.softplus(-x) - F.softplus(x)
        return y, log_jac

    def inverse(self, y, **kwargs):
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)

        x = y.log() - (-y).log1p()
        log_jac = -y.log() - (-y).log1p()
        return x, log_jac

class Logit(nn.Module):
    """ Logit flow. Inverse of sigmoid function. """
    def __init__(self, **kwargs):
        super().__init__()
        self.base_flow = Sigmoid(**kwargs)

    def forward(self, x, **kwargs):
        return self.base_flow.inverse(x)

    def inverse(self, x, **kwargs):
        return self.base_flow.forward(x)
