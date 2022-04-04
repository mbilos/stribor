import torch
import stribor as st

def test_neural_flow():
    torch.manual_seed(123)

    dim = 2

    f = st.NeuralFlow([
        st.ContinuousAffineCoupling(
            latent_net=st.net.MLP(dim, [32], 2 * dim),
            time_net=st.net.TimeLinear(dim),
            mask='ordered_0',
            concatenate_time=False,
        ),
        st.ContinuousIResNet(
            dim,
            [32, 32],
            time_net=st.net.TimeTanh(dim),
        )
    ])

    x = torch.randn(10, 4, 2)
    t = torch.zeros_like(x[...,:1])

    y = f(x, t=t)
    assert (x == y).all()

    t0 = torch.randn_like(x[...,:1])

    y = f(x, t=t0, t0=t0)
    assert torch.allclose(x, y)
