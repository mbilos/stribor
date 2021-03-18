import torch
import pytest
import numpy as np
import nf
from nf.test.base import *

def build_affine(model, dim, num_layers, bias, latent_dim):
    base_dist = nf.Normal(torch.zeros(dim), torch.ones(dim))
    transforms = []
    for _ in range(num_layers):
        transforms.append(getattr(nf, model)(dim, latent_dim=latent_dim, bias=bias))
    return nf.Flow(base_dist, transforms)

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('num_layers', [1, 4])
@pytest.mark.parametrize('latent_dim', [None, 1, 32])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('model', ['Affine', 'AffinePLU', 'AffineExponential'])
def test_affine(input_shape, num_layers, latent_dim, bias, model):
    np.random.seed(123)
    torch.manual_seed(123)

    model = build_affine(model, input_shape[-1], num_layers, bias, latent_dim)

    x = torch.randn(*input_shape)
    latent = torch.randn(*x.shape[:-1], latent_dim) if latent_dim else None

    y, log_jac_y = model.forward(x, latent=latent)
    x_, log_jac_x = model.inverse(y, latent=latent)

    check_inverse(x, x_)
    check_jacobian(log_jac_x, log_jac_y)
    check_one_training_step(input_shape[-1], model, x, latent)
    if latent is None:
        check_log_jacobian_determinant(model, x)


def build_continuous_affine(dim, time_net, num_layers):
    base_dist = nf.Normal(torch.zeros(dim), torch.ones(dim))
    transforms = []
    for _ in range(num_layers):
        transforms.append(nf.ContinuousAffinePLU(dim, getattr(nf.net, time_net)(dim**2 + dim, hidden_dim=8)))
    return nf.Flow(base_dist, transforms)

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('time_net', ['TimeLinear', 'TimeTanh', 'TimeLog', 'TimeFourier'])
@pytest.mark.parametrize('num_layers', [1, 3])
def test_contiouous_coupling_flow(input_shape, time_net, num_layers):
    np.random.seed(123)
    torch.manual_seed(123)

    model = build_continuous_affine(input_shape[-1], time_net, num_layers)

    x = torch.rand(*input_shape)
    t = torch.rand(*input_shape[:-1], 1)

    y, log_jac_y = model.forward(x, t=t)
    x_, log_jac_x = model.inverse(y, t=t)

    check_inverse(x, x_)
    check_jacobian(log_jac_x, log_jac_y)
    check_one_training_step(input_shape[-1], model, x, t=t, latent=None)

    # Check initial condition
    x = torch.rand(*input_shape)
    t = torch.zeros_like(t)

    y, log_jac_y = model.forward(x, t=t)
    assert (x == y).all() and log_jac_y.sum() == 0, 'Initial condition not satisfied in forward'
    y, log_jac_y = model.inverse(x, t=t)
    assert (x == y).all() and log_jac_y.sum() == 0, 'Initial condition not satisfied in inverse'
