import pytest
import nf
import torch
import torch.nn as nn

class ModelExact(nn.Module):
    """ The same forward function as defined in FuncAndDiagJac
        but without detaching and without custom gradients.
    """
    def __init__(self, exclusive_net, dimwise_net):
        super().__init__()
        self.exclusive_net = exclusive_net
        self.dimwise_net = dimwise_net

    def forward(self, t, x):
        shape = x.shape
        h = self.exclusive_net(t, x)
        x = x.view(-1, 1)
        h = h.reshape(x.shape[0], -1)
        output = self.dimwise_net(t, x, h)
        return output.view(*shape)

def check_if_model_is_implemented_correctly(input_shape, model):
    # Define the exact model to which we are comparing with
    model_exact = ModelExact(model.exclusive_net, model.dimwise_net)

    # Input data
    t = torch.Tensor([1])
    x = torch.randn(input_shape)

    # Computed output and Jacobian diagonal
    y, jac = model(t, x)

    # Exact output and Jacobian diagonal
    t_ = t.clone().requires_grad_(True)
    x_ = x.clone().requires_grad_(True)
    f = lambda t, x: model_exact(t, x)
    y_exact = f(t_, x_)
    jac_exact = nf.util.divergence_from_jacobian(f, (t_, x_))[1]

    # Check if forward pass and jacobian are correct
    assert torch.isclose(y, y_exact).all(), 'Output is incorrect'
    assert torch.isclose(jac, jac_exact).all(), 'Jacobian is incorrect'

    # Simulate loss and test if backward step works
    loss1 = y.mean()
    loss1.backward()

    loss2 = y_exact.mean()
    loss2.backward()

    # Check if losses for both models are exactly the same
    assert all([torch.isclose(x1.grad, x2.grad).all() for x1, x2 in zip(model.parameters(), model_exact.parameters())]), \
        'Gradients for two networks are not the same'


@pytest.mark.parametrize('input_shape', [(10, 2), (1, 1, 3), (3, 7, 2), (2, 3, 5, 2)])
@pytest.mark.parametrize('hidden_dims', [[], [64, 32]])
@pytest.mark.parametrize('latent_dim', [1, 2, 5])
def test_diffeq_exact_trace_mlp(input_shape, hidden_dims, latent_dim):
    torch.manual_seed(123)
    model = nf.net.DiffeqExactTraceMLP(input_shape[-1], hidden_dims, input_shape[-1], latent_dim)
    check_if_model_is_implemented_correctly(input_shape, model)


@pytest.mark.parametrize('input_shape', [(3, 7, 2), (2, 3, 5, 2)])
@pytest.mark.parametrize('hidden_dims', [[], [64, 32]])
@pytest.mark.parametrize('latent_dim', [1, 2, 5])
@pytest.mark.parametrize('pooling', ['max', 'mean'])
def test_diffeq_exact_trace_deepset(input_shape, hidden_dims, latent_dim, pooling):
    torch.manual_seed(123)
    model = nf.net.DiffeqExactTraceDeepSet(input_shape[-1], hidden_dims, input_shape[-1], latent_dim, pooling)
    check_if_model_is_implemented_correctly(input_shape, model)


@pytest.mark.parametrize('input_shape', [(1, 1, 8), (3, 7, 2), (2, 3, 5, 2)])
@pytest.mark.parametrize('hidden_dims', [[64, 32]])
@pytest.mark.parametrize('latent_dim', [1, 3, 7])
@pytest.mark.parametrize('n_heads', [1, 2, 8])
def test_diffeq_exact_trace_attention(input_shape, hidden_dims, latent_dim, n_heads):
    torch.manual_seed(123)
    model = nf.net.DiffeqExactTraceAttention(input_shape[-1], hidden_dims, input_shape[-1], latent_dim, n_heads)
    check_if_model_is_implemented_correctly(input_shape, model)
