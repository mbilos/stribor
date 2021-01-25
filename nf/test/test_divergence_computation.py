import pytest
import nf
import torch
import torch.nn as nn


@pytest.mark.parametrize('input_dim', [(10, 2), (2, 10), (5, 10, 2), (23, 17, 7), (3, 7, 11, 13)])
def test_exact_divergence_from_jacobian(input_dim):
    torch.manual_seed(123)

    x = torch.rand(*input_dim)

    # Linear function
    f1 = lambda x: x
    div1 = nf.util.divergence_from_jacobian(f1, x)
    assert torch.isclose(div1, torch.ones_like(x)).all(), 'Incorrect divergence of linear function'

    # Polynomial
    f2 = lambda x: x**3 - 4 * x**2 + 7 * x - 14
    div2 = nf.util.divergence_from_jacobian(f2, x)
    assert torch.isclose(div2, 3 * x**2 - 8 * x + 7).all(), 'Incorrect divergence of polynomial function'

    # Linear layer
    f3 = nn.Linear(input_dim[-1], input_dim[-1])
    div3 = nf.util.divergence_from_jacobian(f3, x)
    assert torch.isclose(div3, torch.diagonal(f3.weight).expand(input_dim)).all(), 'Incorrect divergence of linear layer'

    # Sum
    f4 = lambda x: x.sum().expand_as(x)
    div4 = nf.util.divergence_from_jacobian(f4, x)
    assert torch.isclose(div4, torch.ones_like(x)).all(), 'Incorrect divergence of sum operation'
