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

    def forward(self, t, x, latent=None):
        shape = x.shape
        h = self.exclusive_net(t, x)
        x = x.view(-1, 1)
        h = h.reshape(x.shape[0], -1)
        if latent is not None:
            latent = latent.unsqueeze(-2).repeat_interleave(shape[-1], -2)
            h = torch.cat([h, latent.reshape(x.shape[0], -1)], -1)
        output = self.dimwise_net(t, x, h)
        return output.view(*shape)

def check_if_model_is_implemented_correctly(input_shape, model, latent_dim):
    # Define the exact model to which we are comparing with
    model_exact = ModelExact(model.exclusive_net, model.dimwise_net)

    # Input data
    t = torch.Tensor([1])
    x = torch.randn(input_shape)
    if latent_dim == 0:
        latent = None
    else:
        latent = torch.randn(*x.shape[:-1], latent_dim)

    # Computed output and Jacobian diagonal
    y, jac = model(t, x, latent=latent)

    # Exact output and Jacobian diagonal
    t_ = t.clone().requires_grad_(True)
    x_ = x.clone().requires_grad_(True)
    if latent is None:
        f = lambda t, x: model_exact(t, x, None)
        y_exact = f(t_, x_)
        jac_exact = nf.util.divergence_from_jacobian(f, (t_, x_))[1]
    else:
        latent_ = latent.clone().requires_grad_(True)
        f = lambda t, x, latent: model_exact(t, x, latent)
        y_exact = f(t_, x_, latent_)
        jac_exact = nf.util.divergence_from_jacobian(f, (t_, x_, latent_))[1]

    # Check if forward pass and jacobian are correct
    assert torch.isclose(y, y_exact).all(), 'Output is incorrect'
    assert torch.isclose(jac, jac_exact).all(), 'Jacobian is incorrect'

    # Simulate loss and test if backward step works
    loss1 = y.mean()
    loss1.backward()

    loss2 = y_exact.mean()
    loss2.backward()

    # Check if losses for both models are exactly the same
    assert all([torch.isclose(g1.grad, g2.grad).all() for g1, g2 in zip(model.parameters(), model_exact.parameters())]), \
        'Gradients for two networks are not the same'

@pytest.mark.parametrize('input_shape', [(10, 2), (1, 1, 3), (3, 7, 2), (2, 3, 5, 2)])
@pytest.mark.parametrize('hidden_dims', [[], [64, 32]])
@pytest.mark.parametrize('d_h', [1, 2, 5])
@pytest.mark.parametrize('latent_dim', [0, 1, 32])
def test_diffeq_exact_trace_mlp(input_shape, hidden_dims, d_h, latent_dim):
    torch.manual_seed(123)
    model = nf.net.DiffeqExactTraceMLP(input_shape[-1], hidden_dims, input_shape[-1], d_h, latent_dim=latent_dim)
    check_if_model_is_implemented_correctly(input_shape, model, latent_dim)


@pytest.mark.parametrize('input_shape', [(3, 7, 2), (2, 3, 5, 2)])
@pytest.mark.parametrize('hidden_dims', [[], [64, 32]])
@pytest.mark.parametrize('d_h', [1, 2, 5])
@pytest.mark.parametrize('pooling', ['max', 'mean'])
@pytest.mark.parametrize('latent_dim', [0, 1, 32])
def test_diffeq_exact_trace_deepset(input_shape, hidden_dims, d_h, latent_dim, pooling):
    torch.manual_seed(123)
    model = nf.net.DiffeqExactTraceDeepSet(input_shape[-1], hidden_dims, input_shape[-1], d_h,
                                           latent_dim=latent_dim, pooling=pooling)
    check_if_model_is_implemented_correctly(input_shape, model, latent_dim)


@pytest.mark.parametrize('input_shape', [(1, 2, 8), (3, 7, 2), (2, 3, 5, 2)])
@pytest.mark.parametrize('hidden_dims', [[64, 32]])
@pytest.mark.parametrize('d_h', [1, 3, 7])
@pytest.mark.parametrize('n_heads', [1, 2, 8])
@pytest.mark.parametrize('latent_dim', [0, 1, 32])
def test_diffeq_exact_trace_attention(input_shape, hidden_dims, d_h, latent_dim, n_heads):
    torch.manual_seed(123)
    model = nf.net.DiffeqExactTraceAttention(input_shape[-1], hidden_dims, input_shape[-1], d_h,
                                             latent_dim=latent_dim, n_heads=n_heads)
    check_if_model_is_implemented_correctly(input_shape, model, latent_dim)


@pytest.mark.parametrize('input_shape', [(1, 2, 8), (3, 7, 2), (2, 3, 5, 2)])
@pytest.mark.parametrize('hidden_dims', [[32, 64]])
@pytest.mark.parametrize('d_h', [1, 3, 7])
@pytest.mark.parametrize('latent_dim', [1])
def test_backpropagation_through_encoder(input_shape, hidden_dims, d_h, latent_dim):
    class Model(nn.Module):
        def __init__(self, flow):
            super().__init__()
            self.flow = flow
            self.encoder = nf.net.MLP(input_shape[-1], [17], latent_dim)

        def forward(self, t, x):
            latent = self.encoder(x)
            y = self.flow(t, x, latent=latent)
            return y

    torch.manual_seed(123)
    t = torch.Tensor([1])
    x = torch.randn(input_shape)
    flow = nf.net.DiffeqExactTraceMLP(input_shape[-1], hidden_dims, input_shape[-1], d_h,
                                        latent_dim=latent_dim, return_log_det_jac=False)
    model = Model(flow)
    y = model(t, x)
    loss = y.mean()
    loss.backward()

    torch.manual_seed(123)
    t = torch.Tensor([1])
    x_ = torch.randn(input_shape)
    flow_exact = nf.net.DiffeqExactTraceMLP(input_shape[-1], hidden_dims, input_shape[-1], d_h,
                                            latent_dim=latent_dim, return_log_det_jac=False)
    model_exact = Model(ModelExact(flow_exact.exclusive_net, flow_exact.dimwise_net))
    y_exact = model_exact(t, x_)
    loss_exact = y_exact.mean()
    loss_exact.backward()

    assert (y == y_exact).all(), 'Outputs are not the same'

    # Check if losses for both models are exactly the same
    for (n1, g1), (n2, g2) in zip(model.named_parameters(), model_exact.named_parameters()):
        assert n1 == n2, 'Names are not the same'
        assert (g1.data == g2.data).all(), 'Parameter values are not the same'
        assert torch.isclose(g1.grad, g2.grad).all(), f'Gradients are not the same'
