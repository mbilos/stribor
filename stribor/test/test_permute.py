import torch
import pytest
import numpy as np
import stribor as st
from stribor.test.base import *

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
@pytest.mark.parametrize('dims', [[-1]])
def test_flip(input_shape, dims):
    torch.manual_seed(123)

    f = st.Flip(dims)
    x = torch.randn(*input_shape)

    check_inverse_transform(f, x)
    check_log_jacobian_determinant(f, x)

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (10, 2), (7, 4, 5)])
def test_permute(input_shape):
    torch.manual_seed(123)

    f = st.Permute(input_shape[-1])
    x = torch.randn(*input_shape)

    check_inverse_transform(f, x)
    check_log_jacobian_determinant(f, x)
