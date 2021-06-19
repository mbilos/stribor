import torch
import pytest
import numpy as np
import stribor as st
from stribor.test.base import check_inverse, check_jacobian

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
def test_sigmoid(input_shape):
    np.random.seed(123)
    torch.manual_seed(123)

    model = st.Sigmoid()
    x = torch.randn(*input_shape)

    y, log_jac_y = model.forward(x)
    x_, log_jac_x = model.inverse(y)

    check_inverse(x, x_)
    check_jacobian(log_jac_x, log_jac_y)
