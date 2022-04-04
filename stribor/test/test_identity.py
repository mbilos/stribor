import torch
import stribor as st
from stribor.test.base import *

def test_identity():
    torch.manual_seed(123)

    f = st.Identity()
    x = torch.randn(2, 3, 5, 4)

    check_inverse_transform(f, x)
    check_log_jacobian_determinant(f, x)
