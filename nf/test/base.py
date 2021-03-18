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
