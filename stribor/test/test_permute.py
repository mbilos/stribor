import torch
import pytest
import numpy as np
import stribor as st
from stribor.test.base import check_inverse, check_jacobian

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('dims', [[-1], [0], [0, 1]])
def test_flip(input_shape, dims):
    torch.manual_seed(123)

    model = st.Flip(dims)
    x = torch.randn(*input_shape)

    y, log_jac_y = model.forward(x)
    x_, log_jac_x = model.inverse(y)

    check_inverse(x, x_)

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
def test_permute(input_shape):
    torch.manual_seed(123)

    model = st.Permute(input_shape[-1])
    x = torch.randn(*input_shape)

    y, log_jac_y = model.forward(x)
    x_, log_jac_x = model.inverse(y)

    check_inverse(x, x_)
