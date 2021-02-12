import torch
import torch.distributions as td

class Normal(td.Independent):
    def __init__(self, loc, scale, reinterpreted_batch_ndims=None, **kwargs):
        if reinterpreted_batch_ndims is None: # Handle 1D input, e.g. Normal(0, 1)
            reinterpreted_batch_ndims = 1 if isinstance(loc, torch.Tensor) else 0
        super().__init__(td.Normal(loc, scale, **kwargs), reinterpreted_batch_ndims)

class MultivariateNormal(td.MultivariateNormal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
