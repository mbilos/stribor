import torch
import pytest
import numpy as np
import stribor as st
from stribor.test.base import check_inverse, check_jacobian, check_one_training_step

def build_coupling_flow(dim, hidden_dims, latent_dim, mask, flow_type, num_layers):
    base_dist = st.Normal(torch.zeros(dim), torch.ones(dim))
    transforms = []
    for _ in range(num_layers):
        transforms.append(st.Coupling(
            getattr(st, flow_type)(dim, latent_dim=hidden_dims[-1], n_bins=5),
            st.net.MLP(dim + (latent_dim or 0), hidden_dims, hidden_dims[-1]),
            mask
        ))
    return st.Flow(base_dist, transforms)

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('hidden_dims', [[32], [32, 64]])
@pytest.mark.parametrize('latent_dim', [None, 1, 32])
@pytest.mark.parametrize('mask', ['ordered_left_half'])
@pytest.mark.parametrize('flow_type', ['Spline', 'Affine'])
@pytest.mark.parametrize('num_layers', [1, 3])
def test_coupling_flow(input_shape, hidden_dims, latent_dim, mask, flow_type, num_layers):
    np.random.seed(123)
    torch.manual_seed(123)

    model = build_coupling_flow(input_shape[-1], hidden_dims, latent_dim, mask, flow_type, num_layers)

    x = torch.rand(*input_shape)
    latent = torch.randn(*x.shape[:-1], latent_dim) if latent_dim else None

    y, log_jac_y = model.forward(x, latent=latent)
    x_, log_jac_x = model.inverse(y, latent=latent)

    check_inverse(x, x_)
    check_jacobian(log_jac_x, log_jac_y)
    check_one_training_step(input_shape[-1], model, x, latent)



def build_continuous_coupling_flow(dim, hidden_dims, latent_dim, time_net, num_layers):
    base_dist = st.Normal(torch.zeros(dim), torch.ones(dim))
    transforms = []
    for i in range(num_layers):
        in_dim = dim + 1
        if latent_dim is not None:
            in_dim += latent_dim
        transforms.append(st.ContinuousAffineCoupling(
            dim,
            st.net.MLP(in_dim, hidden_dims, 2 * dim),
            time_net=getattr(st.net, time_net)(2 * dim, hidden_dim=hidden_dims[-1]),
            mask='none' if dim == 1 else f'ordered_{i%2}'
        ))
    return st.Flow(base_dist, transforms)

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('hidden_dims', [[32], [32, 64]])
@pytest.mark.parametrize('latent_dim', [None, 1, 32])
@pytest.mark.parametrize('time_net', ['TimeIdentity', 'TimeLinear', 'TimeTanh', 'TimeLog', 'TimeFourier'])
@pytest.mark.parametrize('num_layers', [1, 3])
def test_contiouous_coupling_flow(input_shape, hidden_dims, latent_dim, time_net, num_layers):
    np.random.seed(123)
    torch.manual_seed(123)

    model = build_continuous_coupling_flow(input_shape[-1], hidden_dims, latent_dim, time_net, num_layers)

    x = torch.rand(*input_shape)
    t = torch.rand(*input_shape[:-1], 1)
    latent = torch.randn(*x.shape[:-1], latent_dim) if latent_dim else None

    y, log_jac_y = model.forward(x, t=t, latent=latent)
    x_, log_jac_x = model.inverse(y, t=t, latent=latent)

    assert (x != y).any(), 'No transformation performed'
    check_inverse(x, x_)
    check_jacobian(log_jac_x, log_jac_y)
    check_one_training_step(input_shape[-1], model, x, t=t, latent=latent)

    # Check initial condition
    x = torch.rand(*input_shape)
    t = torch.zeros_like(t)
    latent = torch.randn(*x.shape[:-1], latent_dim) if latent_dim else None

    y, log_jac_y = model.forward(x, t=t, latent=latent)
    assert (x == y).all() and log_jac_y.sum() == 0
    y, log_jac_y = model.inverse(x, t=t, latent=latent)
    assert (x == y).all() and log_jac_y.sum() == 0
