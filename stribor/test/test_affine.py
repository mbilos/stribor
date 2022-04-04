import torch
import pytest
import numpy as np
import stribor as st
from stribor.test.base import *


@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('scalar_param', [True, False])
def test_fixed_affine(input_shape, scalar_param):
    torch.manual_seed(123)
    np.random.seed(123)

    dim = input_shape[-1]

    x = torch.randn(*input_shape)

    scale = np.random.rand() if scalar_param else torch.rand(dim)
    shift = np.random.normal() if scalar_param else torch.randn(dim)

    f = st.Affine(dim, scale=scale, shift=shift)

    check_inverse_transform(f, x)
    check_log_jacobian_determinant(f, x)


@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('latent_dim', [1, 13])
def test_latent_affine(input_shape, latent_dim):
    torch.manual_seed(123)
    dim = input_shape[-1]

    x = torch.randn(*input_shape)
    latent = torch.randn(*input_shape[:-1], latent_dim)

    f = st.Affine(dim, latent_net=st.net.MLP(latent_dim, [32], 2 * dim))

    check_inverse_transform(f, x, latent=latent)
    check_log_jacobian_determinant(f, x, latent=latent)
    check_gradients_not_nan(f, x, latent=latent)



@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
def test_lu_affine(input_shape):
    torch.manual_seed(123)
    dim = input_shape[-1]

    x = torch.randn(*input_shape)

    f = st.AffineLU(dim)

    check_inverse_transform(f, x)
    check_log_jacobian_determinant(f, x)
    check_gradients_not_nan(f, x)


@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
def test_matrix_exponential(input_shape):
    torch.manual_seed(123)
    dim = input_shape[-1]

    x = torch.randn(*input_shape)

    f = st.MatrixExponential(dim)

    check_inverse_transform(f, x)
    check_log_jacobian_determinant(f, x)
    check_gradients_not_nan(f, x)
