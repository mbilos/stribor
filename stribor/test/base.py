from typing import Union
from torchtyping import TensorType

import torch
from stribor import Transform, ElementwiseTransform


def check_inverse_transform(f: Transform, x: TensorType[..., 'dim'], **kwargs):
    y = f(x, **kwargs)
    x_ = f.inverse(y, **kwargs)
    assert torch.allclose(x, x_, atol=1e-4), 'Inverse function for a given forward function is not correct.'


def _check_log_det_operations(f, x, **kwargs):
    y = f(x, **kwargs)
    ljd = f.log_det_jacobian(x, y, **kwargs)

    _, ljd1 = f.forward_and_log_det_jacobian(x, **kwargs)
    assert torch.allclose(ljd, ljd1), 'Error in implementing `forward_and_log_diag_jacobian`'

    _, ljd2 = f.inverse_and_log_det_jacobian(y, **kwargs)
    assert torch.allclose(ljd, -ljd2, atol=1e-4), 'Error in implementing `inverse_and_log_diag_jacobian`'

def _get_full_jacobian(f, x, **kwargs):
    x = x.view(-1, x.shape[-1])
    kwargs = { k: v.view(-1, v.shape[-1]) if isinstance(v, torch.Tensor) else v for k,v in kwargs.items() }
    y = f(x, **kwargs)

    jacobian = torch.autograd.functional.jacobian(lambda v: f(v, **kwargs), (x,), strict=True)[0]
    jacobian = jacobian.permute(0, 2, 1, 3).sum(0)

    jacobian_inverse = torch.autograd.functional.jacobian(lambda v: f.inverse(v, **kwargs), (y,), strict=True)[0]
    jacobian_inverse = jacobian_inverse.permute(0, 2, 1, 3).sum(0)

    return x, kwargs, jacobian, jacobian_inverse

def _check_log_det_jacobian(f, x, jacobian, reverse=False, **kwargs):
    y = f(x, **kwargs)
    log_det_jacobian = torch.det(jacobian).abs().log()
    if reverse:
        _, log_det_jacobian_model = f.inverse_and_log_det_jacobian(y, **kwargs)
    else:
        _, log_det_jacobian_model = f.forward_and_log_det_jacobian(x, **kwargs)
    assert torch.allclose(log_det_jacobian, log_det_jacobian_model.squeeze(-1), atol=1e-4), 'Jacobian determinant is incorrect'

def _check_whole_jacobian(f, x, jacobian, **kwargs):
    y = f(x, **kwargs)
    try:
        jacobian_model = f.jacobian(x, y, **kwargs)
    except NotImplementedError:
        pass
    else:
        assert torch.allclose(jacobian_model, jacobian, atol=1e-4), 'Jacobian is incorrect'

def _check_log_diag_jacobian(f, x, jacobian, **kwargs):
    y = f(x, **kwargs)
    try:
        log_diag_jacobian_model = f.log_diag_jacobian(x, y, **kwargs)
    except AttributeError:
        pass
    else:
        log_diag_jacobian = torch.diagonal(jacobian, dim1=-2, dim2=-1).log()
        assert torch.allclose(log_diag_jacobian, log_diag_jacobian_model, atol=1e-4), 'Jacobian diagonal is incorrect'

def check_log_jacobian_determinant(f: Union[Transform, ElementwiseTransform], x: TensorType[..., 'dim'], **kwargs):
    _check_log_det_operations(f, x, **kwargs)

    x, kwargs, jacobian, jacobian_inverse = _get_full_jacobian(f, x, **kwargs)

    _check_log_det_jacobian(f, x, jacobian, **kwargs)
    _check_log_det_jacobian(f, x, jacobian_inverse, reverse=True, **kwargs)
    _check_whole_jacobian(f, x, jacobian, **kwargs)
    _check_log_diag_jacobian(f, x, jacobian, **kwargs)


def check_gradients_not_nan(f, x, **kwargs):
    y = f(x, **kwargs)
    loss = y.mean()
    loss.backward()

    assert not any([torch.isnan(x.grad).any().item() for x in f.parameters()])


def check_area_under_pdf_1D(model, input_time=False):
    a, N = 10, 1000
    x = torch.linspace(-a, a, N).unsqueeze(-1)
    t = torch.ones_like(x) if input_time else None

    prob = model.log_prob(x, t=t).exp()
    integral = prob.sum() * 2 * a / N
    assert integral > 0.98 and integral < 1., f'Model doesn\'t define a proper 1D density (integral={integral:.5f})'

def check_area_under_pdf_2D(f, input_time=False):
    a, N = 10, 200
    x = torch.stack(torch.meshgrid(torch.linspace(-a, a, N), torch.linspace(-a, a, N)), -1).view(-1, 2)
    t = torch.ones(x.shape[0], 1) if input_time else None

    prob = f.log_prob(x, t=t).exp()
    integral = prob.sum() * (2 * a / N)**2
    assert integral > 0.98 and integral < 1., f'Model doesn\'t define a proper 2D density (integral={integral:.5f})'
