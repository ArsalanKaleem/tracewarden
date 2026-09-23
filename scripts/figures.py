"""Paper figures in the style of the DriftNet paper: small multi-panel figures, muted palette,
panel titles as "(a) ...", light grids, vector PDF output (plus PNG for the README).

    # training dynamics of one run (needs runs/<name>.history.json written by `tracewarden train`)
    python scripts/figures.py dynamics --history runs/sg.history.json --out paper/figures

    # post-hoc vs pre-dispatch, from two evaluate --out .json files
    python scripts/figures.py causality --posthoc paper/results/driftnet.json \
        --pre paper/results/sg_all.json --out paper/figures

    # risk-coverage + realized conformal budget (needs a checkpoint and its cache)
    python scripts/figures.py coverage --checkpoint runs/sg.pt --data data/processed \
        --cache cache/hashing --out paper/figures

    # per-class scores and the row-normalised step confusion matrix
    python scripts/figures.py steps --checkpoint runs/sg.pt --data data/processed \
        --cache cache/hashing --out paper/figures

    # encoder / seed comparison, from run_ablation.py .json files
    python scripts/figures.py encoders --runs hashing=paper/results/seeds_hashing.json \
        mpnet=paper/results/seeds_mpnet.json --out paper/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ------------------------------------------------------------------ house style
BLUE, ORANGE, GREEN, PINK, GREY = "#1f77b4", "#e8802a", "#2ca58d", "#c77dad", "#9aa3ad"
SERIES = [BLUE, ORANGE, GREEN, PINK]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 8.5,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#d8dce1",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.6,
})


def panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", pad=7)


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def _load(p: str) -> dict:
    return json.loads(Path(p).read_text())


def _report(p: str) -> dict:
    d = _load(p)
    return d.get("report", d)


# ------------------------------------------------------------------ 1. training dynamics
def fig_dynamics(a) -> None:
    hist = _load(a.history)
    ep = [h["epoch"] for h in hist]
    best = max(hist, key=lambda h: (round(h["val_f1"], 4), -h.get("val_loss", 0)))["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))

    ax1.plot(ep, [h["loss"] for h in hist], color=BLUE, label="train loss")
    ax1.plot(ep, [h.get("val_loss", np.nan) for h in hist], color=ORANGE, label="val loss")
    ax1.axvline(best, color=GREY, ls="--", lw=0.9)
    ax1.text(best, ax1.get_ylim()[1] * 0.97, " selected epoch", color=GREY, fontsize=7,
             rotation=90, va="top", ha="left")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend()
    panel(ax1, "a", "Joint loss, selected configuration")

    ax2.plot(ep, [h["val_f1"] for h in hist], color=BLUE, label="trajectory F1")
    ax2.axvline(best, color=GREY, ls="--", lw=0.9)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("validation F1"); ax2.legend(loc="lower right")
    panel(ax2, "b", "Validation F1")
    save(fig, Path(a.out), a.name or "fig_dynamics")


# ------------------------------------------------------------------ 2. price of causality
CAUSAL_METRICS = [("traj/f1", "trajectory\nF1"), ("step_f1/macro", "macro step\nF1"),
                  ("injection_em", "injection\nexact match"), ("hijack_iou", "hijacked\nIoU")]


def fig_causality(a) -> None:
    post, pre = _report(a.posthoc), _report(a.pre)
    labels = [lab for _, lab in CAUSAL_METRICS]
    x = np.arange(len(labels))
    w = 0.36
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})

    for i, (src, name, col) in enumerate([(post, "post-hoc (DriftNet)", BLUE),
                                          (pre, "pre-dispatch (StreamGuard)", ORANGE)]):
        vals = [src.get(k, np.nan) for k, _ in CAUSAL_METRICS]
        bars = ax1.bar(x + (i - 0.5) * w, vals, w, label=name, color=col, edgecolor="white", linewidth=0.6)
        ax1.bar_label(bars, fmt="%.3f", fontsize=6.6, padding=1.5)
    ax1.set_xticks(x, labels); ax1.set_ylim(0.75, 1.02); ax1.set_ylabel("score")
    ax1.legend(loc="lower left", ncols=1)
    ax1.grid(axis="x", visible=False)
    panel(ax1, "a", "Detection holds, localization pays")

    keys = [("prevent/first_hijack_caught", "first hijack\ncaught"),
            ("prevent/blocked_before_execution", "hijacked calls\nblocked"),
            ("prevent/benign_interruption", "benign runs\ninterrupted")]
    vals = [pre.get(k, np.nan) for k, _ in keys]
    cols = [GREEN, GREEN, PINK]
    bars = ax2.bar([lab for _, lab in keys], vals, 0.55, color=cols, edgecolor="white", linewidth=0.6)
    ax2.bar_label(bars, fmt="%.3f", fontsize=6.8, padding=2)
    ax2.set_ylim(0, 1.08); ax2.set_ylabel("rate")
    ax2.grid(axis="x", visible=False)
    panel(ax2, "b", "What only a pre-dispatch guard can report")
    save(fig, Path(a.out), a.name or "fig_causality")


# ------------------------------------------------------------------ shared: score a checkpoint
def _score(a):
    from tracewarden.encoders import EmbeddingCache
    from tracewarden.experiments import load_processed, score
    from tracewarden.featurize import ALL_GROUPS
    from tracewarden.training import load_checkpoint

    model, ck = load_checkpoint(a.checkpoint)
    groups = frozenset(ck.get("groups") or ALL_GROUPS)
    th = ck.get("thresholds") or {}
    trajs = load_processed(a.data, (a.split,))[a.split]
    s = score(ck["kind"], model, trajs, EmbeddingCache(a.cache), groups,
              t_hijack=th.get("hold", 0.5), t_poison=th.get("poison", 0.5))
    return s, th


# ------------------------------------------------------------------ 3. risk-coverage
def fig_coverage(a) -> None:
    from tracewarden.calibrate import risk_coverage

    s, th = _score(a)
    rc = risk_coverage(s["traj_score"], s["y"], alphas=[0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    ax1.plot([r["benign_flag"] for r in rc], [r["attack_catch"] for r in rc], marker="o",
             ms=4, color=BLUE)
    for r in rc:
        if r["alpha"] in (0.0025, 0.01, 0.05):
            ax1.annotate(f"  α={r['alpha']:g}", (r["benign_flag"], r["attack_catch"]),
                         fontsize=7, color=GREY, va="center")
    ax1.set_xscale("log")
    ax1.set_xlabel("benign trajectories flagged"); ax1.set_ylabel("attacks caught")
    panel(ax1, "a", "Risk–coverage")

    cats = ["benign", "hard_negative", "failed_attack", "attacked"]
    flagged = [float(np.mean((s["traj_score"] >= th.get("hold", 0.5))[[i for i, c in enumerate(s["cat"])
               if c == cat]])) if cat in s["cat"] else np.nan for cat in cats]
    cols = [GREEN, PINK, ORANGE, BLUE]
    bars = ax2.bar([c.replace("_", "\n") for c in cats], flagged, 0.55, color=cols,
                   edgecolor="white", linewidth=0.6)
    ax2.bar_label(bars, fmt="%.3f", fontsize=6.8, padding=2)
    ax2.axhline(th.get("alpha_hold", 0.01), color=GREY, ls="--", lw=0.9)
    ax2.text(len(cats) - 0.4, th.get("alpha_hold", 0.01), " budget", fontsize=7, color=GREY, va="bottom")
    ax2.set_ylabel("flag rate"); ax2.set_ylim(0, 1.08)
    ax2.grid(axis="x", visible=False)
    panel(ax2, "b", "Flag rate per true category")
    save(fig, Path(a.out), a.name or "fig_coverage")


# ------------------------------------------------------------------ 4. step head + confusion
def fig_steps(a) -> None:
    from tracewarden.schema import LABELS

    s, _ = _score(a)
    g = np.concatenate(s["gold"]); p = np.concatenate(s["pred"])
    keep = g >= 0; g, p = g[keep], p[keep]
    short = ["benign", "injection\npoint", "hijacked", "failed\ninjection"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(LABELS)); w = 0.26
    for j, (metric, shade) in enumerate([("precision", 0.45), ("recall", 0.72), ("F1", 1.0)]):
        vals = []
        for k in range(len(LABELS)):
            tp = int(((p == k) & (g == k)).sum()); fp = int(((p == k) & (g != k)).sum())
            fn = int(((p != k) & (g == k)).sum())
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            vals.append({"precision": pr, "recall": rc, "F1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0})
        v = [100 * vals[k][metric] for k in range(len(LABELS))]
        cols = [_shade(SERIES[k], shade) for k in range(len(LABELS))]
        ax1.bar(x + (j - 1) * w, v, w, color=cols, edgecolor="white", linewidth=0.5,
                label=metric if j >= 0 else None)
    ax1.set_xticks(x, short); ax1.set_ylim(78, 101.5); ax1.set_ylabel("score (%)")
    ax1.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_shade(GREY, s_)) for s_ in (0.45, 0.72, 1.0)]
    ax1.legend(handles, ["precision", "recall", "F1"], loc="lower center", ncols=3,
               frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    panel(ax1, "a", "Step head: precision / recall / F1 per class")

    cm = np.zeros((len(LABELS), len(LABELS)))
    for t, q in zip(g, p):
        cm[t, q] += 1
    cm = 100 * cm / cm.sum(1, keepdims=True).clip(1)
    ax2.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if cm[i, j] >= 0.05:
                ax2.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=7,
                         color="white" if cm[i, j] > 55 else "#1d2433")
    ax2.set_xticks(range(len(LABELS)), short, fontsize=7)
    ax2.set_yticks(range(len(LABELS)), short, fontsize=7)
    ax2.set_xlabel("predicted"); ax2.set_ylabel("gold")
    ax2.grid(visible=False)
    panel(ax2, "b", "Step confusion (row-normalised, %)")
    save(fig, Path(a.out), a.name or "fig_steps")


def _shade(hexcol: str, f: float) -> tuple:
    c = np.array(matplotlib.colors.to_rgb(hexcol))
    return tuple(1 - f * (1 - c))


# ------------------------------------------------------------------ 5. encoders / seeds
ENC_METRICS = [("traj/f1", "trajectory F1"), ("injection_em", "injection exact match"),
               ("prevent/benign_interruption", "benign runs interrupted")]


def fig_encoders(a) -> None:
    runs = {}
    for spec in a.runs:
        name, path = spec.split("=", 1)
        runs[name] = _load(path)
    fig, axes = plt.subplots(1, len(ENC_METRICS), figsize=(7.4, 2.7))
    for ax, (key, title), letter in zip(axes, ENC_METRICS, "abcdef"):
        for i, (name, rows) in enumerate(runs.items()):
            vals = [r[key] for r in rows if key in r]
            ax.scatter([i] * len(vals), vals, s=26, color=SERIES[i % 4], zorder=3)
            if vals:
                ax.hlines(np.mean(vals), i - 0.22, i + 0.22, color=SERIES[i % 4], lw=1.4)
        ax.set_xticks(range(len(runs)), list(runs), fontsize=7.5)
        ax.set_xlim(-0.6, len(runs) - 0.4)
        ax.grid(axis="x", visible=False)
        panel(ax, letter, title)
    save(fig, Path(a.out), a.name or "fig_encoders")


# ------------------------------------------------------------------ cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, needs_model=False):
        p.add_argument("--out", default="paper/figures")
        p.add_argument("--name", default=None)
        if needs_model:
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--data", required=True)
            p.add_argument("--cache", required=True)
            p.add_argument("--split", default="test")

    p = sub.add_parser("dynamics"); p.add_argument("--history", required=True); common(p)
    p.set_defaults(fn=fig_dynamics)
    p = sub.add_parser("causality"); p.add_argument("--posthoc", required=True)
    p.add_argument("--pre", required=True); common(p); p.set_defaults(fn=fig_causality)
    p = sub.add_parser("coverage"); common(p, True); p.set_defaults(fn=fig_coverage)
    p = sub.add_parser("steps"); common(p, True); p.set_defaults(fn=fig_steps)
    p = sub.add_parser("encoders"); p.add_argument("--runs", nargs="+", required=True,
                                                   help="name=path/to/ablation.json ...")
    common(p); p.set_defaults(fn=fig_encoders)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
