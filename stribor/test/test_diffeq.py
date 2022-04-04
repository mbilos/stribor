import torch
import pytest
import numpy as np
import stribor as st
from stribor.test.base import *


@pytest.mark.parametrize('input_shape', [(2, 10), (7, 4, 5)])
@pytest.mark.parametrize('latent_dim', [0, 1, 13])
def test_cnf_computation(input_shape, latent_dim):
    np.random.seed(123)
    torch.manual_seed(123)

    x = torch.rand(*input_shape)
    latent = torch.randn(*x.shape[:-1], latent_dim) if latent_dim > 0 else None

    dim = x.shape[-1]
    f = st.ContinuousTransform(
        dim,
        net=st.net.DiffeqMLP(dim + latent_dim + 1, [32], dim),
        atol=1e-8,
        rtol=1e-8,
        divergence='compute',
        solver='dopri5',
        has_latent=latent is not None,
    )

    check_inverse_transform(f, x, latent=latent)
    check_log_jacobian_determinant(f, x, latent=latent)
    check_gradients_not_nan(f, x, latent=latent)


@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('latent_dim', [0, 1, 32])
@pytest.mark.parametrize('diffeq', ['DiffeqMLP'])
@pytest.mark.parametrize('hidden_dims', [[], [32], [32, 64]])
@pytest.mark.parametrize('solver,solver_options', [('dopri5', None), ('rk4', {'solver_step': 0.1})])
@pytest.mark.parametrize('rademacher', [True, False])
@pytest.mark.parametrize('use_adjoint', [True, False])
def test_cnf_definition(input_shape, latent_dim, diffeq, hidden_dims, solver, solver_options, rademacher, use_adjoint):
    np.random.seed(123)
    torch.manual_seed(123)

    x = torch.rand(*input_shape)
    latent = torch.randn(*x.shape[:-1], latent_dim) if latent_dim > 0 else None

    dim = x.shape[-1]
    f = st.ContinuousTransform(
        dim,
        net=getattr(st.net, diffeq)(dim + latent_dim + 1, hidden_dims, dim),
        solver=solver,
        solver_options=solver_options,
        rademacher=rademacher,
        use_adjoint=use_adjoint,
        has_latent=latent is not None,
    )

    check_gradients_not_nan(f, x, latent=latent)


@pytest.mark.parametrize('input_shape', [(1, 1, 1), (3, 4, 5), (2, 3, 4, 5)])
@pytest.mark.parametrize('odenet', ['DiffeqDeepset', 'DiffeqSelfAttention'])
def test_diffeq_equivariant(input_shape, odenet):
    torch.manual_seed(123)

    x = torch.rand(*input_shape)

    dim = input_shape[-1]
    f = st.ContinuousTransform(
        dim,
        net=getattr(st.net, odenet)(dim + 1, [13], dim),
        atol=1e-8,
        rtol=1e-8,
        divergence='compute',
        solver='dopri5',
    )

    check_inverse_transform(f, x)
    check_gradients_not_nan(f, x)
