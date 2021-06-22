import torch
import pytest
import numpy as np
import stribor as st
from stribor.test.base import check_inverse, check_jacobian

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (7, 4, 5), (2, 3, 4, 5)])
@pytest.mark.parametrize('dim', [-2, -1])
def test_cumsum(input_shape, dim):
    np.random.seed(123)
    torch.manual_seed(123)

    model = st.Cumsum(dim)
    x = torch.randn(*input_shape)

    y, log_jac_y = model.forward(x)
    x_, log_jac_x = model.inverse(y)

    check_inverse(x, x_)
    check_jacobian(log_jac_x, log_jac_y)

@pytest.mark.parametrize('input_shape', [(2, 10), (7, 4, 5), (2, 3, 4, 5)])
@pytest.mark.parametrize('column', [1, -1, 3])
def test_cumsum_column(input_shape, column):
    np.random.seed(123)
    torch.manual_seed(123)

    model = st.CumsumColumn(column)
    x = torch.randn(*input_shape)

    y, log_jac_y = model.forward(x)
    x_, log_jac_x = model.inverse(y)

    check_inverse(x, x_)
    check_jacobian(log_jac_x, log_jac_y)
