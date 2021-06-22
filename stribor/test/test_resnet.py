import pytest
import torch
import stribor as st
from stribor.test.base import check_inverse, check_one_training_step

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
    model = st.net.InvertibleResNet(dim, hidden_dims, num_layers, n_power_iterations=10)

    for _ in range(5): # Run a few times so spectral_norm kicks in
        x = torch.randn(*input_dim)
        y = model(x)
        y_inv = model.inverse(y)

    assert x.shape == y.shape
    check_inverse(x, y_inv)


@pytest.mark.parametrize('input_dim', [(10, 2), (2, 10), (1, 10, 2), (5, 10, 2)])
@pytest.mark.parametrize('hidden_dims', [[], [32, 64]])
@pytest.mark.parametrize('num_layers', [1])
@pytest.mark.parametrize('time_net', ['TimeTanh', 'TimeFourierBounded'])
def test_resnet_flow(input_dim, hidden_dims, num_layers, time_net):
    torch.manual_seed(123)

    dim = input_dim[-1]
    model = st.net.ResNetFlow(dim, hidden_dims, num_layers, time_net=time_net,
                              time_hidden_dim=8, n_power_iterations=10)

    x = torch.randn(*input_dim)
    t = torch.rand(*input_dim[:-1], 1) * 10
    for _ in range(5):
        y = model(x, t)
        y_inv = model.inverse(y, t)

    assert x.shape == y.shape
    check_inverse(x, y_inv)

    t = torch.zeros(*input_dim[:-1], 1)
    y = model(x, t)
    assert torch.isclose(x, y).all()
