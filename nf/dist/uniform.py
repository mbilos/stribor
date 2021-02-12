import torch.distributions as td

class Uniform(td.Independent):
    def __init__(self, low, high, reinterpreted_batch_ndims=1, **kwargs):
        super().__init__(td.Normal(low, high, **kwargs), reinterpreted_batch_ndims)
