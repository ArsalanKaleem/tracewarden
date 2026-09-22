import torch

from tracewarden.models import DriftNet, StreamGuard
from tracewarden.models.common import n_params


def test_streamguard_is_causal():
    torch.manual_seed(0)
    m = StreamGuard(d_in=20, d=32, layers=2, heads=4, ff=64).eval()
    x = torch.randn(1, 8, 20)
    et = torch.tensor([[0, 1] * 4])
    mask = torch.ones(1, 8, dtype=torch.bool)
    h1, p1 = m(x, et, mask)
    x2 = x.clone()
    x2[0, 5:] = torch.randn(3, 20)          # change the future
    h2, p2 = m(x2, et, mask)
    assert torch.allclose(h1[0, :5], h2[0, :5], atol=1e-5)
    assert torch.allclose(p1[0, :5], p2[0, :5], atol=1e-5)


def test_padding_does_not_change_scores():
    torch.manual_seed(0)
    m = StreamGuard(d_in=20, d=32, layers=2, heads=4, ff=64).eval()
    x = torch.randn(1, 6, 20)
    et = torch.tensor([[0, 1, 0, 1, 0, 1]])
    h, _ = m(x, et, torch.ones(1, 6, dtype=torch.bool))
    xp = torch.cat([x, torch.zeros(1, 4, 20)], 1)
    etp = torch.cat([et, torch.zeros(1, 4, dtype=torch.long)], 1)
    mp = torch.tensor([[True] * 6 + [False] * 4])
    hp, _ = m(xp, etp, mp)
    assert torch.allclose(h, hp[:, :6], atol=1e-5)


def test_driftnet_shapes_and_size():
    m = DriftNet(d_in=772)
    lt, ls = m(torch.randn(3, 11, 772), torch.ones(3, 11, dtype=torch.bool))
    assert lt.shape == (3,) and ls.shape == (3, 11, 4)
    assert n_params(m) < 2_000_000
