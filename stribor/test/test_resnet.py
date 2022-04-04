import pytest
import torch
import stribor as st
from stribor.test.base import *

@pytest.mark.parametrize('input_dim', [(10, 2), (2, 10), (1, 10, 2), (5, 10, 2)])
@pytest.mark.parametrize('hidden_dims', [[], [32, 64]])
@pytest.mark.parametrize('num_layers', [1, 3])
def test_resnet(input_dim, hidden_dims, num_layers):
    torch.manual_seed(123)

    dim = input_dim[-1]
    model = st.net.ResNet(dim, hidden_dims, num_layers)

    x = torch.randn(*input_dim)
    y = model(x)
    assert x.shape == y.shape


@pytest.mark.parametrize('input_dim', [(10, 2), (2, 10), (1, 10, 2), (5, 10, 2)])
@pytest.mark.parametrize('hidden_dims', [[], [32, 64]])
@pytest.mark.parametrize('num_layers', [1])
def test_invertible_resnet(input_dim, hidden_dims, num_layers):
    torch.manual_seed(123)

    dim = input_dim[-1]
    f = st.IResNet(dim, hidden_dims)

    for _ in range(5): # Run a few times so spectral_norm kicks in
        x = torch.randn(*input_dim)
        y = f(x)

    x = torch.randn(*input_dim)
    check_inverse_transform(f, x)


@pytest.mark.parametrize('input_dim', [(10, 2), (2, 10), (1, 10, 2), (5, 10, 2)])
@pytest.mark.parametrize('hidden_dims', [[], [32, 64]])
@pytest.mark.parametrize('num_layers', [1])
@pytest.mark.parametrize('time_net', ['TimeTanh', 'TimeFourierBounded'])
def test_continuous_invertible_resnet(input_dim, hidden_dims, num_layers, time_net):
    torch.manual_seed(123)

    dim = input_dim[-1]
    f = st.ContinuousIResNet(
        dim,
        hidden_dims,
        time_net=getattr(st.net, time_net)(dim, hidden_dim=8),
        n_power_iterations=10
    )

    x = torch.randn(*input_dim)
    t = torch.rand(*input_dim[:-1], 1) * 10
    for _ in range(10):
        y = f(x, t=t)

    x = torch.randn(*input_dim)
    check_inverse_transform(f, x, t=t)

    t = torch.zeros(*input_dim[:-1], 1)
    y = f(x, t=t)
    assert torch.allclose(x, y)
