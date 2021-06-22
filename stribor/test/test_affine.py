import torch
import pytest
import numpy as np
import stribor as st
from stribor.test.base import *

#########################
# Affine transformation #
#########################
def build_affine(model, dim, num_layers, latent_dim):
    base_dist = st.Normal(torch.zeros(dim), torch.ones(dim))
    transforms = []
    for _ in range(num_layers):
        # Only Affine has latent_net implemented
        net = None if latent_dim is None else st.net.MLP(latent_dim, [64], 2*dim)
        transforms.append(getattr(st, model)(dim, latent_net=net))
    return st.Flow(base_dist, transforms)

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('num_layers', [1, 4])
@pytest.mark.parametrize('latent_dim', [None, 1, 32])
@pytest.mark.parametrize('model', ['Affine', 'AffinePLU'])
def test_affine(input_shape, num_layers, latent_dim, model):
    torch.manual_seed(123)

    model = build_affine(model, input_shape[-1], num_layers, latent_dim)

    x = torch.randn(*input_shape)
    latent = torch.randn(*x.shape[:-1], latent_dim) if latent_dim else None

    y, log_jac_y = model.forward(x, latent=latent)
    x_, log_jac_x = model.inverse(y, latent=latent)

    check_inverse(x, x_)
    check_jacobian(log_jac_x, log_jac_y)
    check_one_training_step(input_shape[-1], model, x, latent)
    if latent is None:
        check_log_jacobian_determinant(model, x)

@pytest.mark.parametrize('model_name', ['Affine', 'AffinePLU'])
def test_affine_auc(model_name):
    torch.manual_seed(123)

    for dim, check in zip([1, 2], [check_area_under_pdf_1D, check_area_under_pdf_2D]):
        model = build_affine(model_name, dim, num_layers=1, latent_dim=None)
        check(model)


######################
# Matrix exponential #
######################
@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('num_layers', [1, 4])
@pytest.mark.parametrize('latent_dim', [None, 1, 32])
@pytest.mark.parametrize('scalar_time', [True, False])
def test_matrix_exponential(input_shape, num_layers, latent_dim, scalar_time):
    torch.manual_seed(123)

    model = build_affine('MatrixExponential', input_shape[-1], num_layers, latent_dim)

    x = torch.randn(*input_shape)
    latent = torch.randn(*x.shape[:-1], latent_dim) if latent_dim else None
    t = 1.2 if scalar_time else torch.rand(*x.shape[:-1], 1)

    y, log_jac_y = model.forward(x, latent=latent, t=t)
    x_, log_jac_x = model.inverse(y, latent=latent, t=t)

    check_inverse(x, x_)
    check_jacobian(log_jac_x, log_jac_y)
    check_one_training_step(input_shape[-1], model, x, latent, t=t)

def test_matrix_exponential_auc():
    torch.manual_seed(123)

    for dim, check in zip([1, 2], [check_area_under_pdf_1D, check_area_under_pdf_2D]):
        model = build_affine('MatrixExponential', dim, num_layers=1, latent_dim=None)
        check(model, input_time=True)
