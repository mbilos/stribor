import torch
import pytest
import numpy as np
import stribor as st
from stribor.test.base import *

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
def test_sigmoid(input_shape):
    np.random.seed(123)
    torch.manual_seed(123)

    f = st.Sigmoid()
    x = torch.randn(*input_shape)

    check_inverse_transform(f, x)
    check_log_jacobian_determinant(f, x)

    g = st.Logit()
    y = torch.rand(*input_shape) * 0.5 + 0.25

    check_inverse_transform(g, y)
    check_log_jacobian_determinant(g, y)
