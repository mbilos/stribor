import torch
import stribor as st
from stribor.test.base import *

def test_normalizing_flow():
    torch.manual_seed(123)

    dim = 2
    n_bins = 3
    hidden_dims = [13]

    f = st.NormalizingFlow(
        st.UnitNormal(dim),
        [
            st.Coupling(st.Affine(dim, latent_net=st.net.MLP(dim, hidden_dims, 2 * dim)), mask='ordered_1'),
            st.ContinuousTransform(
                dim,
                net=st.net.DiffeqMLP(dim + 1, hidden_dims, dim),
                divergence='compute',
                atol=1e-8,
                rtol=1e-8,
            ),
            st.Flip(dims=[-1]),
            st.Sigmoid(),
            st.Coupling(
                st.Spline(
                    dim,
                    n_bins=n_bins,
                    latent_net=st.net.MLP(dim, hidden_dims, st.util.cubic_spline_latent_dim(dim, n_bins)),
                    spline_type='cubic',
                ),
                mask='ordered_0',
            ),
            st.Logit(),
        ]
    )

    x = torch.randn(3, 4, dim)
    check_inverse_transform(f, x)
    check_log_jacobian_determinant(f, x)
    check_gradients_not_nan(f, x)
    check_area_under_pdf_2D(f)


def test_doc_example():
    torch.manual_seed(123)
    dim = 2

    f = st.NormalizingFlow(st.UnitNormal(dim), [st.Affine(dim)])

    lp = f.log_prob(torch.randn(3, 2))
    assert torch.allclose(lp, torch.Tensor([[-1.7560], [-1.7434], [-2.1792]]), atol=1e-4)

    s = f.sample(1)
    assert torch.allclose(s, torch.Tensor([[-0.5204,  0.4196]]), atol=1e-4)
