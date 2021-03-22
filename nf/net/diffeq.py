import nf
import torch
import torch.nn as nn

############################################
# Regular nets, use `divergence=approximate`
############################################

class DiffeqConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = None

    def forward(self, t, x, latent=None, **kwargs):
        t = torch.ones_like(x[..., :1]) * t
        input = torch.cat([t, x], -1)
        if latent is not None:
            input = torch.cat([input, latent], -1)
        return self.net(input, **kwargs)

class DiffeqMLP(DiffeqConcat):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nf.net.MLP(*args, **kwargs)

class DiffeqEquivariantNet(DiffeqConcat):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nf.net.EquivariantNet(*args, **kwargs)

class DiffeqSelfAttention(DiffeqConcat):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nf.net.SelfAttention(*args, **kwargs)

class DiffeqDeepSet(DiffeqConcat):
    def __init__(self, in_dim, hidden_dims, out_dim, **kwargs):
        super().__init__()
        self.net1 = nf.net.MLP(in_dim + 1, hidden_dims[:-1], hidden_dims[-1])
        self.net2 = nf.net.MLP(hidden_dims[-1] + in_dim + 1, [], out_dim)
        self.net = self._forward

    def _forward(self, x, **kwargs):
        h = self.net1(x)
        h = torch.max(h, -2, keepdims=True)[0].repeat_interleave(h.shape[-2], -2)
        x = torch.cat([x, h], -1)
        return self.net2(x)


##########################################################
# Volume preserving - 0 trace nets, use `divergence=exact`
##########################################################

class DiffeqZeroTraceMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, return_log_det_jac=True, **kwargs):
        super().__init__()
        self.return_log_det_jac = return_log_det_jac
        self.net1 = nf.net.MADE(in_dim, hidden_dims, out_dim, natural_ordering=True,
                                reverse_ordering=False, return_per_dim=True)
        self.net2 = nf.net.MADE(in_dim, hidden_dims, out_dim, natural_ordering=True,
                                reverse_ordering=True, return_per_dim=True)

    def forward(self, t, x, **kwargs):
        y = self.net1(x, **kwargs) + self.net2(x, **kwargs)
        y = y.reshape(*y.shape[:-2], -1)
        return (y, torch.zeros_like(x)) if self.return_log_det_jac else y

class DiffeqZeroTraceDeepSet(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, pooling='max', return_log_det_jac=True, **kwargs):
        super().__init__()
        self.elementwise = nf.net.MADE(in_dim, hidden_dims, out_dim, return_per_dim=True)
        self.interaction = nf.net.ExclusivePooling(in_dim, hidden_dims, out_dim // in_dim, pooling)
        self.return_log_det_jac = return_log_det_jac

    def forward(self, t, x, mask=None, latent=None, **kwargs):
        if latent is not None:
            x = torch.cat([x, latent], -1)
        if mask is None:
            mask = torch.ones(*x.shape[:-1], 1).to(x)
        y = self.elementwise(x) + self.interaction(t, x, mask=mask)
        y = y * mask.unsqueeze(-1)
        y = y.reshape(*y.shape[:-2], -1)
        return (y, torch.zeros_like(x)) if self.return_log_det_jac else y

class DiffeqZeroTraceAttention(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, n_heads=1, return_log_det_jac=True, **kwargs):
        super().__init__()
        self.n_heads = n_heads
        self.return_log_det_jac = return_log_det_jac

        self.q = nf.net.MADE(in_dim, hidden_dims[:-1], hidden_dims[-1] * in_dim, return_per_dim=True)
        self.k = nf.net.MLP(in_dim, hidden_dims[:-1], hidden_dims[-1])
        self.v = nf.net.MLP(in_dim, hidden_dims[:-1], hidden_dims[-1], return_per_dim=True)
        self.proj = nf.net.MLP(hidden_dims[-1], [], out_dim // in_dim)

    def forward(self, t, x, mask=None, **kwargs):
        query = self.q(x).transpose(-2, -3) # (B, N, D) -> (B, D, N, H)
        # value = self.v(x).transpose(-2, -3)
        key = self.k(x).unsqueeze(-2).repeat_interleave(x.shape[-1], dim=-2).transpose(-2, -3) # (B, D, N, H)
        value = self.v(x).unsqueeze(-2).repeat_interleave(x.shape[-1], dim=-2).transpose(-2, -3) # (B, D, N, H)

        y = nf.net.attention(query, key, value, self.n_heads, True, mask) # (B, D, N, H)
        y = y.transpose(-2, -3) # (B, N, D, H)
        y = self.proj(y).view(*y.shape[:-2], -1) # (B, N, D, O) -> (B, N, D * O)

        return (y, torch.zeros_like(x)) if self.return_log_det_jac else y


################################################################
# Nets with closed form calculated trace, use `divergence=exact`
################################################################

class DiffeqExactTrace(nn.Module):
    """ Network with exact trace.
    Recommended to directly use tested implementations, e.g. `DiffeqExactTraceMLP`.
    Args:
        exclusive_net: Hollow Jacobian net, e.g. DiffeqMADE(dim, hidden_dims, dim * hidden_dim)
        dimwise_net: Per dimension net, e.g. DiffeqMLP(hidden_dim + 2, hidden_dims, 1)
    """
    def __init__(self, exclusive_net, dimwise_net, return_log_det_jac=True, **kwargs):
        super().__init__()
        self.exclusive_net = exclusive_net
        self.dimwise_net = dimwise_net
        self.return_log_det_jac = return_log_det_jac

    def forward(self, t, x, latent=None, **kwargs):
        params = nf.util.flatten_params(self.exclusive_net, self.dimwise_net)
        y, jac = nf.net.FuncAndDiagJac.apply(self.exclusive_net, self.dimwise_net, t, x, latent, params)
        return (y, jac) if self.return_log_det_jac else y

class DiffeqExactTraceMLP(DiffeqExactTrace):
    """ Exact trace networks that mimics MLP architecture.
    Args:
        in_dim: (int) Input dimension
        hidden_dims: (list) Hidden dimensions, e.g. [64, 64]
        out_dim: (int) Output dimension
        d_h: (int) Size of conditioning vector per each dimension
        latent_dim: (int) Size of latent vector
    """
    def __init__(self, in_dim, hidden_dims, out_dim, d_h, latent_dim=0, return_log_det_jac=True, **kwargs):
        exclusive_net = DiffeqZeroTraceMLP(in_dim, hidden_dims, d_h * out_dim, return_log_det_jac=False, return_per_dim=True)
        dimwise_net = DiffeqMLP(d_h + latent_dim + 2, hidden_dims, 1)
        super().__init__(exclusive_net, dimwise_net, return_log_det_jac)

class DiffeqExactTraceDeepSet(DiffeqExactTrace):
    def __init__(self, in_dim, hidden_dims, out_dim, d_h, latent_dim=0, pooling='max', return_log_det_jac=True, **kwargs):
        exclusive_net = DiffeqZeroTraceDeepSet(in_dim, hidden_dims, d_h * out_dim, return_log_det_jac=False)
        dimwise_net = DiffeqMLP(d_h + latent_dim + 2, hidden_dims, 1)
        super().__init__(exclusive_net, dimwise_net, return_log_det_jac)

class DiffeqExactTraceAttention(DiffeqExactTrace):
    def __init__(self, in_dim, hidden_dims, out_dim, d_h, latent_dim=0, n_heads=1, return_log_det_jac=True, **kwargs):
        exclusive_net = DiffeqZeroTraceAttention(in_dim, hidden_dims, d_h * out_dim, n_heads, return_log_det_jac=False)
        dimwise_net = DiffeqMLP(d_h + latent_dim + 2, hidden_dims, 1)
        super().__init__(exclusive_net, dimwise_net, return_log_det_jac)
