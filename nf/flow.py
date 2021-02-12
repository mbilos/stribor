import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

class Flow(nn.Module):
    """ Normalizing flow.

    Args:
        base_dist: Instance of torch.distributions
        transforms: List of transformations from `nf.flows`

    Example:
    >> flow = nf.Flow(nf.Normal(0, 1), [nf.Identity()])
    >> flow.forward(torch.Tensor([[1]])) # Returns y and log_jac_diag
    (tensor([1.]), tensor([0.]))
    >> flow.sample(5) # Output will differ every time
    tensor([0.1695, 1.9026, 0.4640, 0.7100, 0.2773])
    """
    def __init__(self, base_dist, transforms):
        super().__init__()
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, latent=None, mask=None, reverse=False, **kwargs):
        """
        Args:
            x: Input from base density (..., dim)
            latent: Conditional vector, same shape as y
            mask: Tenstor of 0 and 1, same shape as y
            reverse: Whether to use the inverse function

        Returns:
            y: Output in target density (..., dim)
            log_jac_diag: Diagonal logarithm of Jacobian (..., dim)
        """
        transforms = self.transforms[::-1] if reverse else self.transforms
        _mask = 1 if mask is None else mask

        log_jac_diag = torch.zeros_like(x).to(x)
        for t in transforms:
            if reverse:
                x, ld = t.inverse(x * _mask, latent=latent, mask=mask, **kwargs)
            else:
                x, ld = t.forward(x * _mask, latent=latent, mask=mask, **kwargs)
            log_jac_diag += ld * _mask
        return x, log_jac_diag

    def inverse(self, y, latent=None, mask=None, **kwargs):
        """ Inverse of forward function with the same arguments. """
        return self.forward(y, latent=latent, mask=mask, reverse=True, **kwargs)

    def log_prob(self, x, **kwargs):
        """ Calculates log-probability of a sample.
        Args:
            x: Input with shape (..., dim)
            latent: latent with shape (..., latent_dim). All transforms need to know about latent dim.
        Returns:
            log_prob: Log-probability of the input (..., 1)
        """
        x, log_jac_diag = self.inverse(x, **kwargs)
        log_prob = self.base_dist.log_prob(x) + log_jac_diag.sum(-1)
        return log_prob.unsqueeze(-1)

    def sample(self, num_samples, latent=None, mask=None, **kwargs):
        """ Transforms samples from a base to target distribution.
        Uses reparametrization trick.
        Args:
            num_samples: (tuple or int) Shape of samples
            latent: Contex for conditional sampling with shape (latent_dim)
        Returns:
            x: Samples from target distribution (*num_samples, dim)
        """
        if isinstance(num_samples, int):
            num_samples = (num_samples,)

        x = self.base_dist.rsample(num_samples)
        x, log_jac_diag = self.forward(x, **kwargs)
        return x
