import torch
import pytest
import numpy as np
import stribor as st
from stribor.test.base import check_inverse, check_jacobian, check_one_training_step

def build_spline(dim, n_bins, lower, upper, spline_type, latent_dim, num_layers):
    base_dist = st.Normal(torch.zeros(dim), torch.ones(dim))
    transforms = []
    for _ in range(num_layers):
        params_size = dim * (3 * n_bins - 1) if spline_type == 'quadratic' else dim * (2 * n_bins + 2)
        net = None if latent_dim is None else st.net.MLP(latent_dim, [32], params_size)
        transforms.append(st.Spline(dim, n_bins=n_bins, lower=lower, upper=upper, spline_type=spline_type, latent_net=net))
    return st.Flow(base_dist, transforms)

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('n_bins', [1, 4])
@pytest.mark.parametrize('lower,upper', [(-1, 3), (0, 1)])
@pytest.mark.parametrize('spline_type', ['quadratic'])
@pytest.mark.parametrize('latent_dim', [None, 1, 32])
@pytest.mark.parametrize('num_layers', [0, 1, 4])
def test_spline(input_shape, n_bins, lower, upper, spline_type, latent_dim, num_layers):
    np.random.seed(123)
    torch.manual_seed(123)

    model = build_spline(input_shape[-1], n_bins, lower, upper, spline_type, latent_dim, num_layers)

    x = torch.rand(*input_shape)
    latent = torch.randn(1, latent_dim) if latent_dim is not None else None

    y, log_jac_y = model.forward(x, latent=latent)
    x_, log_jac_x = model.inverse(y, latent=latent)

    check_inverse(x, x_)
    check_jacobian(log_jac_x, log_jac_y)
    if num_layers > 0:
        check_one_training_step(input_shape[-1], model, x, latent)


@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
def test_spline_free_bounding_box_param(input_shape):
    np.random.seed(123)
    torch.manual_seed(123)

    x = torch.rand(*input_shape)

    n_bins = 5
    w = torch.randn(1, n_bins) / 100
    h = torch.randn(1, n_bins) / 100
    d = torch.randn(1, n_bins - 1) / 100

    func = st.util.unconstrained_rational_quadratic_spline
    y, log_jac_y = func(x, w, h, d, left=-1, bottom=-3, top=2, right=1)
    x_, log_jac_x = func(y, w, h, d, left=-1, bottom=-3, top=2, right=1, inverse=True)

    check_inverse(x, x_)
    check_jacobian(log_jac_x, log_jac_y)
