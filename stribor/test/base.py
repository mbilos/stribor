import torch

def check_inverse(x, y_inverse):
    assert torch.isclose(x, y_inverse, atol=1e-4).all(), \
        "Inverse function for a given forward function is not correct."

def check_jacobian(jac, jac_inverse):
    assert torch.isclose(jac, -jac_inverse, atol=1e-4).all(), \
        "Inverse jacobian for a given forward jacobian is not correct."

def check_log_jacobian_determinant(model, x):
    x = x.view(-1, x.shape[-1])
    _, ljd = model(x)
    jacobian = torch.autograd.functional.jacobian(lambda x: model(x)[0], (x,), strict=True)[0]
    jacobian = jacobian.permute(0, 2, 1, 3).sum(0)
    assert torch.isclose(ljd.sum(1), torch.det(jacobian).abs().log(), atol=1e-4).all(), 'Jacobian is incorrect'

def check_one_training_step(dim, model, x, latent, **kwargs):
    log_prob = model.log_prob(x, latent=latent, **kwargs)
    loss = -log_prob.sum(-1).mean()
    loss.backward()

    assert not any([torch.isnan(x.grad).any().item() for x in model.parameters()])

def check_area_under_pdf_1D(model, input_time=False):
    a, N = 10, 1000
    x = torch.linspace(-a, a, N).unsqueeze(-1)
    t = torch.ones_like(x) if input_time else None

    prob = model.log_prob(x, t=t).exp()
    integral = prob.sum() * 2 * a / N
    assert integral > 0.98 and integral < 1., f'Model doesn\'t define a proper 1D density (integral={integral:.5f})'

def check_area_under_pdf_2D(model, input_time=False):
    a, N = 10, 200
    x = torch.stack(torch.meshgrid(torch.linspace(-a, a, N), torch.linspace(-a, a, N)), -1).view(-1, 2)
    t = torch.ones(x.shape[0], 1) if input_time else None

    prob = model.log_prob(x, t=t).exp()
    integral = prob.sum() * (2 * a / N)**2
    assert integral > 0.98 and integral < 1., f'Model doesn\'t define a proper 2D density (integral={integral:.5f})'
