import torch

def check_inverse(x, y_inverse):
    assert torch.isclose(x, y_inverse, rtol=1e-4, atol=1e-4).all(), \
        "Inverse function for a given forward function is not correct."

def check_jacobian(jac, jac_inverse):
    assert torch.isclose(jac, -jac_inverse, rtol=1e-4, atol=1e-4).all(), \
        "Inverse jacobian for a given forward jacobian is not correct."

def check_one_training_step(dim, model, x, latent):
    log_prob = model.log_prob(x, latent=latent)
    loss = -log_prob.sum(-1).mean()
    loss.backward()

    assert not any([torch.isnan(x.grad).any().item() for x in model.parameters()])
