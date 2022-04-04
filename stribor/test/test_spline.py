import torch
import pytest
import numpy as np
import stribor as st
from stribor.test.base import *


@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('n_bins', [1, 3, 10])
@pytest.mark.parametrize('latent_dim', [0, 1, 13])
@pytest.mark.parametrize('spline', ['cubic', 'quadratic'])
def test_spline(input_shape, n_bins, latent_dim, spline):
    np.random.seed(123)
    torch.manual_seed(123)

    x = torch.rand(*input_shape) * 2
    latent = torch.randn(*input_shape[:-1], latent_dim) if latent_dim != 0 else None

    dim = input_shape[-1]
    out_spline_dim = dim * (2 * n_bins + 2) if spline == 'cubic' else dim * (3 * n_bins - 1)

    f = st.Spline(
        dim=dim,
        n_bins=n_bins,
        latent_net=st.net.MLP(latent_dim, [12], out_spline_dim) if latent is not None else None,
        lower=0,
        upper=2,
        spline_type=spline,
    )

    check_inverse_transform(f, x, latent=latent)
    check_log_jacobian_determinant(f, x, latent=latent)
    check_gradients_not_nan(f, x, latent=latent)


@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('spline', ['quadratic'])
def test_quadratic_spline_free_bounding_box_param(input_shape, spline):
    torch.manual_seed(123)

    x = torch.rand(*input_shape)

    class UnboundedSpline(st.Spline):
        def forward_and_log_diag_jacobian(self, x, latent=None, *, reverse=False, **kwargs):
            w, h, d = self._get_params(latent)
            return self.spline(x, w, h, d, inverse=reverse, left=-1, bottom=-3, top=2, right=1)

    f = UnboundedSpline(dim=input_shape[-1], n_bins=5, spline_type=spline)

    check_inverse_transform(f, x)
    check_log_jacobian_determinant(f, x)
    check_gradients_not_nan(f, x)
