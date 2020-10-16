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
    >> flow = nf.Flow(torch.distributions.Normal(0, 1), [nf.Identity()])
    >> flow.forward(torch.Tensor([[1]])) # Returns y and log_jac_diag
    (tensor([1.]), tensor([0.]))
    >> flow.sample(5) # Output will differ every time
    tensor([0.1695, 1.9026, 0.4640, 0.7100, 0.2773])
    """
    def __init__(self, base_dist, transforms):
        super().__init__()
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, latent=None, mask=None, *args, **kwargs):
        if mask is None:
            mask = 1

        log_jac_diag = torch.zeros_like(x).to(x)
        for transform in self.transforms:
            x, ld = transform(x * mask, latent=latent, mask=mask)
            log_jac_diag += ld * mask
        return x, log_jac_diag

    def inverse(self, y, latent=None, mask=None, *args, **kwargs):
        """
        Returns:
            y: Input from target density (..., dim)
            log_jac_diag: Diagonal logarithm of Jacobian (..., dim)
            mask: Tenstor of 0 and 1 (..., dim)
        """
        if mask is None:
            mask = 1

        log_jac_diag = torch.zeros_like(y).to(y)
        for transform in self.transforms[::-1]:
            y, ld = transform.inverse(y * mask, latent=latent, mask=mask)
            log_jac_diag += ld * mask
        return y, log_jac_diag

    def log_prob(self, x, latent=None, mask=None, **kwargs):
        """ Calculates log-probability of a sample with a series
        of invertible transformations.

        Args:
            x: Input with shape (..., dim)
            latent: latent with shape (..., latent_dim). All transforms need to know about latent dim.
        Returns:
            log_prob: Log-probability of the input (..., 1)
        """
        x, log_jac_diag = self.inverse(x, latent=latent, mask=mask, **kwargs)
        log_prob = (self.base_dist.log_prob(x) + log_jac_diag).sum(-1, keepdim=True)
        return log_prob

    def sample(self, num_samples, latent=None, mask=None, **kwargs):
        """ Transforms samples from a base distribution to get
        a sample from the distribution defined by normalizing flow

        Args:
            num_samples: (tuple or int) Shape of samples
            latent: Contex for conditional sampling with shape (latent_dim)
        Returns:
            x: Samples from target distribution (*num_samples, dim)
        """
        if isinstance(num_samples, int):
            num_samples = (num_samples,)

        x = self.base_dist.rsample(num_samples)
        x, log_jac_diag = self.forward(x, latent=latent, mask=mask, **kwargs)
        return x
