import stribor as st
import torch

def test_permuation_equivariance():
    torch.manual_seed(123)

    net = st.net.EquivariantLayer(2, 4)
    x = torch.randn(3, 5, 2)

    y1 = net(x)
    y2 = net(torch.flip(x, [1]))

    assert torch.isclose(y1, torch.flip(y2, [1])).all()
