import stribor as st
import torch

def test_normal():
    x = torch.randn(4, 3, 5, 2)
    p0 = torch.distributions.Normal(torch.zeros(2), torch.ones(2)).log_prob(x).sum(-1)
    p1 = st.Normal(torch.zeros(2), torch.ones(2)).log_prob(x)
    p2 = st.MultivariateNormal(torch.zeros(2), torch.eye(2)).log_prob(x)
    assert (p0 == p1).all() and torch.isclose(p1, p2).all()
