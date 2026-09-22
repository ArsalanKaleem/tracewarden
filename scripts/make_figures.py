"""Paper figures from results JSON: adaptation curve and risk-coverage curve.

python scripts/make_figures.py --adaptation paper/results/adaptation.json --out paper/figures
python scripts/make_figures.py --checkpoint runs/sg.pt --data data/processed --cache cache/mpnet --out paper/figures
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptation", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.adaptation:
        rows = json.loads(Path(a.adaptation).read_text())
        by = defaultdict(list)
        for r in rows:
            by[r["k"]].append(r["traj_f1"])
        ks = sorted(by)
        mean = [sum(by[k]) / len(by[k]) for k in ks]
        plt.figure(figsize=(4.2, 3))
        plt.plot(ks, mean, marker="o")
        plt.xlabel("real labeled trajectories")
        plt.ylabel("trajectory F1 (real test)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "adaptation.pdf")
    if a.checkpoint:
        from tracewarden.calibrate import risk_coverage
        from tracewarden.encoders import EmbeddingCache
        from tracewarden.experiments import load_processed, score
        from tracewarden.featurize import ALL_GROUPS
        from tracewarden.training import load_checkpoint

        model, ck = load_checkpoint(a.checkpoint)
        s = score(ck["kind"], model, load_processed(a.data, (a.split,))[a.split], EmbeddingCache(a.cache),
                  frozenset(ck.get("groups") or ALL_GROUPS))
        rc = risk_coverage(s["traj_score"], s["y"], alphas=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
        plt.figure(figsize=(4.2, 3))
        plt.plot([r["benign_flag"] for r in rc], [r["attack_catch"] for r in rc], marker="o")
        plt.xscale("log")
        plt.xlabel("benign trajectories flagged")
        plt.ylabel("attacks caught")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "risk_coverage.pdf")
    print("figures in", out)
