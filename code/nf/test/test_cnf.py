import torch
import pytest
import numpy as np
import nf
from nf.test.base import check_inverse, check_jacobian, check_one_training_step


@pytest.mark.parametrize('input_shape', [(2, 10), (7, 4, 5)])
@pytest.mark.parametrize('latent_dim', [None, 32])
def test_cnf_computation(input_shape, latent_dim):
    np.random.seed(123)
    torch.manual_seed(123)

    x = torch.rand(*input_shape)
    latent = torch.randn(*x.shape[:-1], latent_dim) if latent_dim else None

    dim = x.shape[-1]
    in_dim = dim if latent_dim is None else dim + latent_dim
    transforms = [nf.ContinuousFlow(dim, net=nf.net.DiffeqMLP(in_dim, [32], dim), atol=1e-8, rtol=1e-8,
                  divergence_fn='brute_force', solver='dopri5')]
    model = nf.Flow(torch.distributions.Normal(torch.zeros(dim), torch.ones(dim)), transforms)

    y, log_jac_y = model.forward(x, latent=latent)
    x_, log_jac_x = model.inverse(y, latent=latent)

    check_inverse(x, x_)
    check_jacobian(log_jac_x, log_jac_y)
    check_one_training_step(input_shape[-1], model, x, latent)


@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('latent_dim', [None, 1, 32])
@pytest.mark.parametrize('diffeq', ['DiffeqMLP'])
@pytest.mark.parametrize('hidden_dims', [[], [32], [32, 64]])
@pytest.mark.parametrize('solver,solver_options', [('dopri5', None), ('rk4', {'solver_step': 0.1})])
@pytest.mark.parametrize('rademacher', [True, False])
@pytest.mark.parametrize('use_adjoint', [True, False])
def test_cnf_definition(input_shape, latent_dim, diffeq, hidden_dims, solver, solver_options, rademacher, use_adjoint):
    np.random.seed(123)
    torch.manual_seed(123)

    x = torch.rand(*input_shape)
    latent = torch.randn(*x.shape[:-1], latent_dim) if latent_dim else None

    dim = x.shape[-1]
    in_dim = dim if latent_dim is None else dim + latent_dim
    cnf = nf.ContinuousFlow(dim,
                            net=getattr(nf.net, diffeq)(in_dim, hidden_dims, dim),
                            solver=solver,
                            solver_options=solver_options,
                            rademacher=rademacher,
                            use_adjoint=use_adjoint)
    model = nf.Flow(torch.distributions.Normal(torch.zeros(dim), torch.ones(dim)), [cnf])
