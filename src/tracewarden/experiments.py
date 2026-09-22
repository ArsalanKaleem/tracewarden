"""Reusable experiment building blocks used by the CLI and scripts/."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .calibrate import calibrate, realized_rates, risk_coverage
from .encoders import EmbeddingCache
from .featurize import ALL_GROUPS, build_split
from .metrics import bootstrap, full_report, trajectory_prf
from .models import DriftNet, StreamGuard
from .models.common import n_params
from .schema import Trajectory, load_jsonl
from .training import TrainConfig, decode_events, fit, predict_events, predict_steps


def load_processed(folder: str | Path, parts=("train", "val", "test")) -> dict[str, list[Trajectory]]:
    folder = Path(folder)
    return {p: load_jsonl(str(folder / f"{p}.jsonl")) for p in parts if (folder / f"{p}.jsonl").exists()}


def parse_groups(s: str | None) -> frozenset:
    if s is None or s == "all":
        return ALL_GROUPS
    if s in ("none", ""):
        return frozenset()
    g = frozenset(x.strip() for x in s.split(","))
    bad = g - ALL_GROUPS
    if bad:
        raise ValueError(f"unknown feature groups {bad}; choose from {sorted(ALL_GROUPS)}")
    return g


def make_model(kind: str, d_in: int, **kw):
    return StreamGuard(d_in=d_in, **kw) if kind == "streamguard" else DriftNet(d_in=d_in, **kw)


def train(kind: str, train_t, val_t, cache: EmbeddingCache, groups=ALL_GROUPS, cfg: TrainConfig | None = None,
          model_kw: dict | None = None):
    cfg = cfg or TrainConfig()
    item_kind = "events" if kind == "streamguard" else "steps"
    use_world = "world" in groups
    tr = build_split(train_t, cache, item_kind, groups, use_world)
    va = build_split(val_t, cache, item_kind, groups, use_world)
    model = make_model(kind, tr[0].x.shape[1], **(model_kw or {}))
    if cfg.verbose:
        print(f"{kind}: {n_params(model):,} trainable params | train {len(tr)} | val {len(va)} | d_in {tr[0].x.shape[1]}")
    model, hist = fit(model, tr, va, cfg, item_kind)
    return model, hist


def score(kind: str, model, trajs, cache: EmbeddingCache, groups=ALL_GROUPS, t_hijack=0.5, t_poison=0.5):
    """Returns dict with per-trajectory predictions ready for metrics."""
    item_kind = "events" if kind == "streamguard" else "steps"
    items = build_split(trajs, cache, item_kind, groups, "world" in groups)
    gold = [it.step_labels for it in items]
    if kind == "streamguard":
        preds = predict_events(model, items)
        hij = [h for h, _ in preds]
        steps = [decode_events(h, p, t_hijack, t_poison) for h, p in preds]
        traj_score = [float(h.max()) if len(h) else 0.0 for h in hij]
    else:
        preds = predict_steps(model, items)
        hij = None
        steps = [s for _, s, _ in preds]
        traj_score = [p for p, _, _ in preds]
    return {"gold": gold, "pred": steps, "hijack": hij, "traj_score": np.asarray(traj_score),
            "y": np.asarray([it.y_traj for it in items]), "cat": [it.category for it in items],
            "comp": [it.meta.get("compliance") for it in items]}


def evaluate(s: dict, threshold: float = 0.5, ci: bool = True, reps: int = 500) -> tuple[dict, dict]:
    yp = (s["traj_score"] >= threshold).astype(int)
    rep = full_report(s["gold"], s["pred"], s["y"], yp, s["cat"], s["comp"], s["hijack"], threshold)
    cis = {}
    if ci:
        n = len(s["y"])
        cis["traj/f1"] = bootstrap(lambda idx: trajectory_prf(s["y"][idx], yp[idx])["f1"], n, reps)
        from .metrics import localization, step_f1

        cis["step_f1/macro"] = bootstrap(
            lambda idx: step_f1([s["gold"][i] for i in idx], [s["pred"][i] for i in idx])["macro"], n, reps)
        cis["hijack_iou"] = bootstrap(
            lambda idx: localization([s["gold"][i] for i in idx], [s["pred"][i] for i in idx])["hijack_iou"], n, reps)
    return rep, cis


def calibrate_on(s: dict, alpha: float) -> tuple[dict, list[dict]]:
    benign = s["traj_score"][s["y"] == 0]
    th = calibrate(benign, alpha_hold=alpha)
    return th, risk_coverage(s["traj_score"], s["y"])


def rates(s: dict, th: dict) -> dict:
    return realized_rates(s["traj_score"], s["y"], th)


# ------------------------------------------------------------------ shortcut audit (N4)
def shortcut_audit(train_t: list[Trajectory], test_t: list[Trajectory]) -> dict:
    """How well do lookups on identity or tool template predict compromise, with no model at all?"""
    from .features.anonymize import world_fingerprint

    def lookup(key):
        tab = defaultdict(Counter)
        for t in train_t:
            tab[key(t)][t.compromised] += 1
        prior = Counter(t.compromised for t in train_t).most_common(1)[0][0]
        hits, preds = 0, []
        for t in test_t:
            k = key(t)
            if k in tab:
                hits += 1
                preds.append(tab[k].most_common(1)[0][0])
            else:
                preds.append(prior)
        y = [t.compromised for t in test_t]
        acc = float(np.mean([p == g for p, g in zip(preds, y)]))
        return {"accuracy": acc, "coverage": hits / len(test_t), **trajectory_prf(y, preds)}

    out = {
        "majority_class": lookup(lambda t: 0),
        "world_identity": lookup(world_fingerprint),
        "company_only": lookup(lambda t: (t.world or {}).get("company")),
        "tool_sequence": lookup(lambda t: tuple(s.tool for s in t.steps)),
        "n_steps": lookup(lambda t: len(t.steps)),
    }
    per_domain = {}
    for d in sorted({t.domain for t in test_t}):
        tr = [t for t in train_t if t.domain == d]
        te = [t for t in test_t if t.domain == d]
        if tr and te:
            per_domain[d] = shortcut_audit_single(tr, te, world_fingerprint)
    out["world_identity_by_domain"] = per_domain
    return out


def shortcut_audit_single(train_t, test_t, key) -> float:
    tab = defaultdict(Counter)
    for t in train_t:
        tab[key(t)][t.compromised] += 1
    prior = Counter(t.compromised for t in train_t).most_common(1)[0][0]
    preds = [tab[key(t)].most_common(1)[0][0] if key(t) in tab else prior for t in test_t]
    return float(np.mean([p == t.compromised for p, t in zip(preds, test_t)]))


def dump(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
