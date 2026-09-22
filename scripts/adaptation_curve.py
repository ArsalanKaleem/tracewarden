"""Day 17 - how many real trajectories close the transfer gap?

Fine-tunes a synthetic-trained StreamGuard on k real Dev-real trajectories (k in --ks), evaluates on a
held-out real test set, and re-calibrates thresholds on the real benign dev data ("calibration-only" = k=0
with recalibration, the cheapest deployment fix).

python scripts/adaptation_curve.py --checkpoint runs/sg.pt --real data/agentdojo_qwen --cache cache/mpnet_dojo \
   --ks 0,25,50,100,200,500 --out paper/results/adaptation.json
"""
import argparse
import copy
import json
import random
from pathlib import Path

from tracewarden.calibrate import calibrate
from tracewarden.encoders import EmbeddingCache
from tracewarden.experiments import evaluate, load_processed, rates, score
from tracewarden.featurize import ALL_GROUPS, build_split
from tracewarden.training import TrainConfig, fit, load_checkpoint

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--real", required=True, help="processed folder with train (=Dev-real) / val / test")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ks", default="0,25,50,100,200,500")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="paper/results/adaptation.json")
    a = ap.parse_args()
    base, ck = load_checkpoint(a.checkpoint)
    groups = frozenset(ck.get("groups") or ALL_GROUPS)
    R = load_processed(a.real)
    cache = EmbeddingCache(a.cache)
    val = R.get("val") or R["train"][:100]
    rows = []
    for k in [int(x) for x in a.ks.split(",")]:
        for r in range(a.repeats if k else 1):
            model = copy.deepcopy(base)
            if k:
                pool = list(R["train"])
                random.Random(r).shuffle(pool)
                items = build_split(pool[:k], cache, "events", groups)
                model, _ = fit(model, items, build_split(val, cache, "events", groups),
                               TrainConfig(epochs=a.epochs, lr=a.lr, seed=r, patience=5, verbose=False), "events")
            sv = score("streamguard", model, val, cache, groups)
            th = calibrate(sv["traj_score"][sv["y"] == 0], 0.01)  # recalibrate on real benign dev data
            st = score("streamguard", model, R["test"], cache, groups, th["hold"])
            rep, _ = evaluate(st, th["hold"], ci=False)
            rows.append({"k": k, "repeat": r, "traj_f1": rep["traj/f1"], "macro_step_f1": rep["step_f1/macro"],
                         "first_hijack_caught": rep["prevent/first_hijack_caught"], **rates(st, th)})
            print(json.dumps(rows[-1]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
