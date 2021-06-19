import stribor as st
import torch

def test_ordered_right_half_mask():
    gen = st.util.get_mask('ordered_right_half')

    assert torch.eq(gen(1), torch.Tensor([1])).all()
    assert torch.eq(gen(2), torch.Tensor([0, 1])).all()
    assert torch.eq(gen(5), torch.Tensor([0, 0, 1, 1, 1])).all()

def test_ordered_left_half_mask():
    gen = st.util.get_mask('ordered_left_half')

    assert torch.eq(gen(1), torch.Tensor([1])).all()
    assert torch.eq(gen(2), torch.Tensor([1, 0])).all()
    assert torch.eq(gen(5), torch.Tensor([1, 1, 0, 0, 0])).all()

def test_parity_even_mask():
    gen = st.util.get_mask('parity_even')

    assert torch.eq(gen(1), torch.Tensor([1])).all()
    assert torch.eq(gen(2), torch.Tensor([0, 1])).all()
    assert torch.eq(gen(5), torch.Tensor([0, 1, 0, 1, 0])).all()

def test_parity_odd_mask():
    gen = st.util.get_mask('parity_odd')

    assert torch.eq(gen(1), torch.Tensor([1])).all()
    assert torch.eq(gen(2), torch.Tensor([1, 0])).all()
    assert torch.eq(gen(5), torch.Tensor([1, 0, 1, 0, 1])).all()
