import torch
import pytest
import stribor as st
from stribor.test.base import *


@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('latent_dim', [0, 1, 13])
def test_affine_coupling(input_shape, latent_dim):
    torch.manual_seed(123)
    dim = input_shape[-1]

    x = torch.randn(*input_shape)
    if latent_dim == 0:
        latent = None
    else:
        latent = torch.randn(*input_shape[:-1], latent_dim)

    f = st.Coupling(
        transform=st.Affine(dim, latent_net=st.net.MLP(dim + latent_dim, [13], 2 * dim)),
        mask='ordered_left_half',
    )

    check_inverse_transform(f, x, latent=latent)
    check_log_jacobian_determinant(f, x, latent=latent)
    check_gradients_not_nan(f, x, latent=latent)


@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('latent_dim', [0, 1, 13])
def test_continuous_affine_coupling(input_shape, latent_dim):
    torch.manual_seed(123)
    dim = input_shape[-1]

    x = torch.randn(*input_shape)
    if latent_dim == 0:
        latent = None
    else:
        latent = torch.randn(*input_shape[:-1], latent_dim)

    t = torch.randn_like(x[...,:1])

    f = st.ContinuousAffineCoupling(
        latent_net=st.net.MLP(dim + 1 + latent_dim, [13], 2 * dim),
        time_net=st.net.TimeLinear(2 * dim),
        mask='ordered_left_half',
    )

    check_inverse_transform(f, x, t=t, latent=latent)
    check_log_jacobian_determinant(f, x, t=t, latent=latent)
    check_gradients_not_nan(f, x, t=t, latent=latent)
