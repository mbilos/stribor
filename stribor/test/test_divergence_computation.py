import pytest
import stribor as st
import torch
import torch.nn as nn


@pytest.mark.parametrize('input_dim', [(10, 2), (2, 10), (5, 10, 2), (23, 17, 7), (3, 7, 11, 13)])
def test_exact_divergence_from_jacobian(input_dim):
    torch.manual_seed(123)

    x = torch.rand(*input_dim).requires_grad_(True)

    # Linear function
    f1 = lambda x: x
    div1 = st.util.divergence_from_jacobian(f1, x)
    assert torch.isclose(div1, torch.ones_like(x)).all(), 'Incorrect divergence from Jacobian of linear function'
    div1 = st.util.divergence_exact(f1(x), x)
    assert torch.isclose(div1, torch.ones_like(x)).all(), 'Incorrect divergence from Jacobian of linear function'

    # Polynomial
    f2 = lambda x: x**3 - 4 * x**2 + 7 * x - 14
    div2 = st.util.divergence_from_jacobian(f2, x)
    assert torch.isclose(div2, 3 * x**2 - 8 * x + 7).all(), 'Incorrect divergence from Jacobian of polynomial function'
    div2 = st.util.divergence_exact(f2(x), x)
    assert torch.isclose(div2, 3 * x**2 - 8 * x + 7).all(), 'Incorrect exact divergence of polynomial function'

    # Linear layer
    f3 = nn.Linear(input_dim[-1], input_dim[-1])
    div3 = st.util.divergence_from_jacobian(f3, x)
    assert torch.isclose(div3, torch.diagonal(f3.weight).expand(input_dim)).all(), 'Incorrect divergence from Jacobian of linear layer'
    div3 = st.util.divergence_exact(f3(x), x)
    assert torch.isclose(div3, torch.diagonal(f3.weight).expand(input_dim)).all(), 'Incorrect exact divergence of linear layer'

    # Deep set
    f4 = lambda x: x**2 + (x**3).sum(-2, keepdim=True).repeat_interleave(x.shape[-2], dim=-2)
    div4 = st.util.divergence_from_jacobian(f4, x)
    assert torch.isclose(div4, 2 * x + 3 * x**2).all(), 'Incorrect divergence from Jacobian of deepset operation'
    div4 = st.util.divergence_exact_for_sets(f4(x), x)
    assert torch.isclose(div4, 2 * x + 3 * x**2).all(), 'Incorrect exact divergence of deepset operation'
