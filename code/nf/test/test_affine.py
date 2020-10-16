import torch
import pytest
import numpy as np
import nf
from nf.test.base import check_inverse, check_jacobian, check_one_training_step

def build_affine(dim, num_layers, latent_dim):
    base_dist = torch.distributions.Normal(torch.zeros(dim), torch.ones(dim))
    transforms = []
    for _ in range(num_layers):
        transforms.append(nf.Affine(dim, latent_dim=latent_dim))
    return nf.Flow(base_dist, transforms)

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('num_layers', [0, 1, 4])
@pytest.mark.parametrize('latent_dim', [None, 1, 32])
def test_affine(input_shape, num_layers, latent_dim):
    np.random.seed(123)
    torch.manual_seed(123)

    model = build_affine(input_shape[-1], num_layers, latent_dim)

    x = torch.randn(*input_shape)
    latent = torch.randn(*x.shape[:-1], latent_dim) if latent_dim else None

    y, log_jac_y = model.forward(x, latent=latent)
    x_, log_jac_x = model.inverse(y, latent=latent)

    check_inverse(x, x_)
    check_jacobian(log_jac_x, log_jac_y)

    if num_layers > 0:
        check_one_training_step(input_shape[-1], model, x, latent)
