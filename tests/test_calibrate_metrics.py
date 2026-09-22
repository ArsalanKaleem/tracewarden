import numpy as np

from tracewarden.calibrate import calibrate, conformal_threshold, decide
from tracewarden.metrics import localization, prevention, trajectory_prf
from tracewarden.training import decode_events


def test_conformal_budget_holds():
    rng = np.random.default_rng(0)
    cal, test = rng.beta(2, 8, 2000), rng.beta(2, 8, 20000)
    t = conformal_threshold(cal, 0.01)
    assert (test >= t).mean() <= 0.015
    th = calibrate(cal, 0.05)
    assert th["block"] >= th["hold"]
    assert decide(th["block"] + 0.01, th) == "block" and decide(0.0, th) == "allow"


def test_metrics_basic():
    assert trajectory_prf([1, 0, 1], [1, 0, 0])["precision"] == 1.0
    gold = [np.array([0, 1, 2, 2]), np.array([0, 0, 0])]
    pred = [np.array([0, 1, 2, 0]), np.array([0, 0, 0])]
    loc = localization(gold, pred)
    assert loc["injection_em"] == 1.0 and abs(loc["hijack_iou"] - 0.5) < 1e-9
    pv = prevention([np.array([0.1, 0.2, 0.9, 0.1]), np.array([0.1, 0.1, 0.1])], gold, 0.5)
    assert pv["first_hijack_caught"] == 1.0 and pv["benign_interruption"] == 0.0


def test_decode_events():
    lab = decode_events(np.array([0.0, 0.0, 0.9]), np.array([0.0, 0.8, 0.0]))
    assert lab.tolist() == [0, 1, 2]                       # B I H
    lab = decode_events(np.array([0.0, 0.0, 0.0]), np.array([0.8, 0.0, 0.0]))
    assert lab.tolist() == [3, 0, 0]                       # F B B (poison, never obeyed)
