import torch
import pytest
import numpy as np
import stribor as st
from stribor.test.base import *

@pytest.mark.parametrize('name', ['ELU', 'LeakyReLU'])
def test_activations(name):
    x = torch.randn(3, 4, 5)
    f = getattr(st, name)()

    check_inverse_transform(f, x)
    check_log_jacobian_determinant(f, x)

@pytest.mark.parametrize('func', [torch.tanh, torch.sigmoid, torch.relu])
def test_continuous_activation(func):
    torch.manual_seed(123)

    f = st.ContinuousActivation(func)
    x = torch.randn(3, 4, 5)

    # Initial condition, identity at 0
    t = torch.zeros(3, 4, 1)
    assert torch.allclose(f(x, t), x)

    # For high values of t, same behavior as before
    t = torch.ones(3, 4, 1) * 100
    assert torch.allclose(f(x, t), func(x))

    f = st.ContinuousActivation(func, learnable=True)
    check_gradients_not_nan(f, x, t=t)

@pytest.mark.parametrize('log_time', [True, False])
def test_continuous_tanh(log_time):
    torch.manual_seed(123)

    f = st.ContinuousTanh(log_time)
    x = torch.randn(100, 5, 10)
    t = torch.rand_like(x[...,:1]) * 5

    check_inverse_transform(f, x, t=t)
    check_log_jacobian_determinant(f, x, t=t)
