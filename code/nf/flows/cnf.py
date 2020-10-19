# Continous normalizing flow (https://arxiv.org/abs/1810.01367) [https://github.com/rtqichen/ffjord]

import nf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint, odeint

__all__ = ['ContinuousFlow']


def divergence_bf(dx, y, *args):
    diag = torch.zeros_like(y)
    for i in range(y.shape[1]):
        diag[:,i] += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return diag

def divergence_approx(f, y, e):
    return torch.autograd.grad(f, y, e, create_graph=True)[0] * e


class ODEfunc(nn.Module):
    def __init__(self, diffeq, divergence_fn=None, rademacher=False, returns_divergence=False):
        super().__init__()
        self.diffeq = diffeq
        self.rademacher = rademacher
        self.returns_divergence = returns_divergence

        if divergence_fn == 'brute_force':
            self.divergence_fn = divergence_bf
        elif divergence_fn == 'approximate':
            self.divergence_fn = divergence_approx
        elif not returns_divergence:
            raise ValueError('Divergence function should be "brute_force" or "approximate".')

        self.register_buffer('_num_evals', torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        self._num_evals += 1

        y = states[0]
        t = torch.Tensor([t]).to(y)
        latent = None if len(states) == 2 else states[2]

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1
            else:
                self._e = torch.randn_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)

            if self.returns_divergence:
                dy, divergence = self.diffeq(t, y, latent=latent)
            else:
                dy = self.diffeq(t, y, latent=latent)
                if not self.training:
                    divergence = divergence_bf(dy, y)
                else:
                    divergence = self.divergence_fn(dy, y, self._e)

        dlatent = () if latent is None else (torch.zeros_like(latent),)
        return (dy, -divergence) + dlatent


class ContinuousFlow(nn.Module):
    def __init__(self, dim, net=None, T=1.0, divergence_fn='approximate', use_adjoint=True,
                 solver='dopri5', solver_options={}, test_solver=None, test_solver_options=None,
                 returns_divergence=False, rademacher=False, atol=1e-5, rtol=1e-3, **kwargs):
        """
        Continuous normalizing flow.

        Args:
            dim: Input data dimension
            net: Neural net that defines a differential equation, instance of `net`
            T: Integrate from 0 until T (Default: 1.0)
            divergence_fn: How to calculate divergence, 'approximate' or 'brute_force'
            use_adjoint: Whether to use adjoint method for backpropagation
            solver: ODE black-box solver, adaptive: dopri5, dopri8, bosh3, adaptive_heun;
                fixed-step: euler, midpoint, rk4, explicit_adams, implicit_adams
            solver_options: Additional options, e.g. {'step_size': 10}
            test_solver: Same as solver, used during evaluation
            test_solver_options: Same as solver_options, used during evaluation
            returns_divergence: If 'net' calculates divergence directly (Default: False)
            rademacher: Whether to use rademacher sampling (Default: False)
            atol: Tolerance (Default: 1e-5)
            rtol: Tolerance (Default: 1e-5)
        """
        super().__init__()

        self.T = T
        self.dim = dim

        self.odefunc = ODEfunc(net, divergence_fn, rademacher, returns_divergence)

        self.integrate = odeint_adjoint if use_adjoint else odeint

        self.solver = solver
        self.solver_options = solver_options
        self.test_solver = test_solver or solver
        self.test_solver_options = solver_options if test_solver_options is None else test_solver_options

        self.atol = atol
        self.rtol = rtol

    def forward(self, x, latent=None, reverse=False, **kwargs):
        # Set inputs
        *shape, dim = x.shape
        x = x.view(-1, dim)
        logp = torch.zeros_like(x)

        # Set integration times
        integration_times = torch.tensor([0.0, self.T]).to(x)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics
        self.odefunc.before_odeint()

        initial = (x, logp)
        if latent is not None:
            initial += (latent.view(-1, latent.shape[-1]),)

        # Solve ODE
        state_t = self.integrate(
            self.odefunc,
            initial,
            integration_times,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver if self.training else self.test_solver,
            options=self.solver_options if self.training else self.test_solver_options,
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        # Collect outputs with correct shape
        x, logp = [s.view(*shape, dim) for s in state_t[:2]]
        return x, -logp

    def inverse(self, x, logp=None, latent=None, **kwargs):
        return self.forward(x, logp=logp, latent=latent, reverse=True)

    def num_evals(self):
        return self.odefunc._num_evals.item()

def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
