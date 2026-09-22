"""Day 10 - ablation grid (N1-N4). Trains each variant, evaluates on a split, writes a markdown table.

python scripts/run_ablation.py --data data/processed --cache cache/mpnet --out paper/results/ablation.md
(add --shuffled-test data/processed_shuf to also score on identity-shuffled test data)
"""
import argparse
import json
from pathlib import Path

from tracewarden.encoders import EmbeddingCache
from tracewarden.experiments import evaluate, load_processed, parse_groups, score, train
from tracewarden.schema import load_jsonl
from tracewarden.training import TrainConfig, save_checkpoint

VARIANTS = [
    # name, model, groups, use shuffled augmentation
    ("DriftNet (post-hoc, world feats)", "driftnet", "world", False),
    ("StreamGuard text only", "streamguard", "none", False),
    ("+ world", "streamguard", "world", False),
    ("+ world + provenance", "streamguard", "world,prov", False),
    ("+ world + provenance + anchor", "streamguard", "world,prov,anchor", False),
    ("all features", "streamguard", "all", False),
    ("all features + identity-shuffle aug", "streamguard", "all", True),
]
COLS = ["traj/f1", "step_f1/macro", "injection_em", "hijack_iou", "prevent/first_hijack_caught",
        "prevent/benign_interruption", "recall/delayed_execution", "recall/partial_hijack"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma list of variant indices")
    ap.add_argument("--out", default="paper/results/ablation.md")
    a = ap.parse_args()

    D = load_processed(a.data)
    shuf = Path(a.data) / "train_shuf.jsonl"
    aug = load_jsonl(str(shuf)) if shuf.exists() else []
    cache = EmbeddingCache(a.cache)
    runs = Path("runs/ablation")
    runs.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = [int(i) for i in a.only.split(",")] if a.only else range(len(VARIANTS))
    for i in idx:
        name, kind, g, use_aug = VARIANTS[i]
        if use_aug and not aug:
            print("skip", name, "(no train_shuf.jsonl; run `tracewarden data ... --shuffle-copies 1`)")
            continue
        groups = parse_groups(g)
        for seed in range(a.seeds):
            cfg = TrainConfig(epochs=a.epochs, seed=42 + seed)
            model, _ = train(kind, D["train"] + (aug if use_aug else []), D["val"], cache, groups, cfg)
            save_checkpoint(str(runs / f"v{i}_s{seed}.pt"), model, kind, cache.encoder, groups, train_cfg=cfg)
            rep, ci = evaluate(score(kind, model, D[a.split], cache, groups), reps=200)
            rows.append({"variant": name, "seed": seed, **{c: rep.get(c, float("nan")) for c in COLS},
                         "traj_f1_ci": ci.get("traj/f1")})
            print(json.dumps(rows[-1], default=str))
    lines = ["| variant | seed | " + " | ".join(COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['seed']} | " + " | ".join(f"{r[c]:.3f}" for c in COLS) + " |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(rows, indent=1, default=str))
    print("\n".join(lines))
