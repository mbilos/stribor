import pytest
import stribor as st
import torch

@pytest.mark.parametrize('input_dim', [(1, 1, 1), (10, 1, 1), (4, 4, 1), (3, 5, 2), (5, 3, 2, 3)])
@pytest.mark.parametrize('hidden_dims', [[], [13], [47, 11]])
@pytest.mark.parametrize('out_dim', [1, 2, 7])
@pytest.mark.parametrize('pooling', ['mean', 'max'])
def test_exclusive_set_net_zero_trace(input_dim, hidden_dims, out_dim, pooling):
    torch.manual_seed(123)

    dim = input_dim[-1]
    model = st.net.DiffeqZeroTraceDeepSet(dim, hidden_dims, dim * out_dim, pooling, return_log_det_jac=False)
    x = torch.randn(*input_dim).requires_grad_(True)

    t = torch.Tensor([1])
    f = lambda x: model(t, x)
    trace_exact = st.util.divergence_from_jacobian(f, x)

    assert trace_exact.sum() == 0
    assert torch.allclose(trace_exact, torch.zeros_like(trace_exact), atol=1e-6)


@pytest.mark.parametrize('input_dim', [(1, 1, 1), (10, 1, 1), (4, 4, 1), (3, 5, 2), (5, 3, 2, 3)])
@pytest.mark.parametrize('pooling', ['mean', 'max'])
def test_exclusive_set_net_masking(input_dim, pooling):
    torch.manual_seed(123)

    model = st.net.DiffeqZeroTraceDeepSet(input_dim[-1], [32], input_dim[-1] * 4, pooling, return_log_det_jac=False)

    # Regular masked output
    x = torch.randn(*input_dim)
    mask = torch.rand(*input_dim[:-1], 1).round()
    t = torch.Tensor([1])
    y1 = model(t, x, mask)

    # Output after changing masked elements
    x_perm = x + x * (1 - mask) * torch.rand(*input_dim)
    y2 = model(t, x_perm, mask)

    assert torch.isclose(y1, y2, atol=1e-5).all()
