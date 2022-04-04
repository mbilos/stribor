import torch
import pytest
import stribor as st
from stribor.test.base import *

@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (7, 4, 5), (2, 3, 4, 5)])
@pytest.mark.parametrize('dim', [-1])
def test_cumsum(input_shape, dim):
    torch.manual_seed(123)

    f = st.Cumsum(dim)
    x = torch.randn(*input_shape)

    check_inverse_transform(f, x)
    check_log_jacobian_determinant(f, x)

    y = f(x)
    assert torch.all(y == x.cumsum(dim))


@pytest.mark.parametrize('input_shape', [(1, 1), (2, 10), (7, 4, 5), (2, 3, 4, 5)])
@pytest.mark.parametrize('dim', [-1])
def test_diff(input_shape, dim):
    torch.manual_seed(123)

    f = st.Diff(dim)
    x = torch.randn(*input_shape)
    y = x.cumsum(dim)

    check_inverse_transform(f, y)
    check_log_jacobian_determinant(f, y)

    x_ = f(y)
    assert torch.allclose(x, x_)


@pytest.mark.parametrize('input_shape', [(2, 10), (7, 4, 5), (2, 3, 4, 5)])
@pytest.mark.parametrize('column', [1, -1, 3])
def test_cumsum_column(input_shape, column):
    torch.manual_seed(123)

    f = st.CumsumColumn(column)
    x = torch.randn(*input_shape)

    check_inverse_transform(f, x)
