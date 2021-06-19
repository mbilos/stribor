import torch
import pytest
import numpy as np
import stribor as st
from stribor.test.base import *

@pytest.mark.parametrize('name', ['ELU', 'LeakyReLU'])
def test_activations(name):
    x = torch.randn(3, 4, 5)
    f = getattr(st, name)()

    y, y_ldj = f(x)
    x_inv, x_ldj = f.inverse(y)

    check_inverse(x, x_inv)
    check_jacobian(y_ldj, x_ldj)
    check_log_jacobian_determinant(f, x)

