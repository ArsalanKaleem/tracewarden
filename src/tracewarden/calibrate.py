"""Conformal alarm budget - novel addition N5.

Split-conformal calibration on benign trajectories: if benign calibration and deployment trajectories are
exchangeable, the probability that a new benign trajectory's max action score exceeds the threshold is
at most alpha. Two thresholds give a three-way policy:
    score <  t_hold            -> allow
    t_hold <= score < t_block  -> hold for human review
    score >= t_block           -> block
"""
from __future__ import annotations

import math

import numpy as np


def conformal_threshold(benign_scores, alpha: float) -> float:
    s = np.sort(np.asarray(benign_scores, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("need benign calibration scores")
    k = math.ceil((n + 1) * (1 - alpha))  # finite-sample corrected rank
    return float(s[min(k, n) - 1]) + 1e-9 if k <= n else float("inf")


def calibrate(benign_traj_scores, alpha_hold: float = 0.01, alpha_block: float | None = None) -> dict:
    """benign_traj_scores: per benign/hard-negative trajectory, the MAX hijack score over its actions."""
    alpha_block = alpha_block if alpha_block is not None else alpha_hold / 5
    t_hold = conformal_threshold(benign_traj_scores, alpha_hold)
    t_block = max(conformal_threshold(benign_traj_scores, alpha_block), t_hold)
    return {"hold": t_hold, "block": t_block, "alpha_hold": alpha_hold, "alpha_block": alpha_block,
            "n_cal": int(len(benign_traj_scores))}


def decide(score: float, thresholds: dict) -> str:
    if score >= thresholds["block"]:
        return "block"
    if score >= thresholds["hold"]:
        return "hold"
    return "allow"


def realized_rates(scores, is_attack, thresholds: dict) -> dict:
    s, a = np.asarray(scores), np.asarray(is_attack).astype(bool)
    return {
        "benign_hold_or_block": float((s[~a] >= thresholds["hold"]).mean()) if (~a).any() else float("nan"),
        "benign_block": float((s[~a] >= thresholds["block"]).mean()) if (~a).any() else float("nan"),
        "attack_caught": float((s[a] >= thresholds["hold"]).mean()) if a.any() else float("nan"),
        "attack_blocked": float((s[a] >= thresholds["block"]).mean()) if a.any() else float("nan"),
    }


def risk_coverage(scores, is_attack, alphas=(0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1)) -> list[dict]:
    s, a = np.asarray(scores), np.asarray(is_attack).astype(bool)
    rows = []
    for al in alphas:
        t = conformal_threshold(s[~a], al)
        rows.append({"alpha": al, "threshold": t, "benign_flag": float((s[~a] >= t).mean()),
                     "attack_catch": float((s[a] >= t).mean())})
    return rows
