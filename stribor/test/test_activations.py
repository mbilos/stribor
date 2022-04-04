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
