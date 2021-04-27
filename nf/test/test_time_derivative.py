import pytest
import torch
import numpy as np
import nf


@pytest.mark.parametrize('input_dim', [(1, 1), (2, 3, 1)])
@pytest.mark.parametrize('output_dim', [1, 5])
@pytest.mark.parametrize('function', ['TimeIdentity', 'TimeLinear', 'TimeTanh', 'TimeLog', 'TimeFourier'])
def test_time_derivative(input_dim, output_dim, function):
    f = getattr(nf.net.time_net, function)(output_dim, hidden_dim=8)

    t = torch.rand(*input_dim) * 5
    output = f(t)
    dt = f.derivative(t)

    assert dt.shape == output.shape and output.shape[-1] == output_dim
    assert len(t.shape) == len(output.shape) and t.shape[:-1] == output.shape[:-1]

    dt_true = torch.autograd.functional.jacobian(f, t, vectorize=True)
    for _ in range(len(dt_true.shape) // 2):
        dt_true = dt_true.sum(-1)

    assert torch.isclose(dt, dt_true).all()
