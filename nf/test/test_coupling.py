import torch
import pytest
import numpy as np
import nf
from nf.test.base import check_inverse, check_jacobian, check_one_training_step

def build_coupling_flow(dim, hidden_dims, latent_dim, mask, flow_type, num_layers):
    base_dist = torch.distributions.Normal(torch.zeros(dim), torch.ones(dim))
    transforms = []
    for _ in range(num_layers):
        lat_dim = hidden_dims[-1]
        if latent_dim is not None:
            lat_dim += latent_dim
        transforms.append(nf.Coupling(
            getattr(nf, flow_type)(dim, latent_dim=lat_dim, n_bins=5),
            nf.net.MLP(dim, hidden_dims, hidden_dims[-1]),
            mask
        ))
    return nf.Flow(base_dist, transforms)

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('hidden_dims', [[32], [32, 64]])
@pytest.mark.parametrize('latent_dim', [None, 1, 32])
@pytest.mark.parametrize('mask', ['ordered_left_half'])
@pytest.mark.parametrize('flow_type', ['Spline', 'Affine'])
@pytest.mark.parametrize('num_layers', [0, 1])
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
    if num_layers > 0:
        check_one_training_step(input_shape[-1], model, x, latent)
