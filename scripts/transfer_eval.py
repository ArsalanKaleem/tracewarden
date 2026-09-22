"""Day 16 - the synthetic-to-real transfer table (RQ1/RQ2).

python scripts/transfer_eval.py --checkpoint runs/sg.pt \
   --eval agentdrift:data/processed:test:cache/mpnet \
   --eval real-A:data/agentdojo_qwen:test:cache/mpnet_dojo \
   --eval real-B:data/agentdojo_llama:test:cache/mpnet_dojo \
   --out paper/results/transfer.md
Each --eval is name:processed_folder:split:cache_folder.
"""
import argparse
import json
from pathlib import Path

from tracewarden.encoders import EmbeddingCache
from tracewarden.experiments import evaluate, load_processed, rates, score
from tracewarden.featurize import ALL_GROUPS
from tracewarden.training import load_checkpoint

COLS = ["traj/precision", "traj/recall", "traj/f1", "step_f1/macro", "injection_em", "hijack_iou",
        "prevent/blocked_before_execution", "prevent/first_hijack_caught", "prevent/benign_interruption"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--out", default="paper/results/transfer.md")
    a = ap.parse_args()
    model, ck = load_checkpoint(a.checkpoint)
    groups = frozenset(ck.get("groups") or ALL_GROUPS)
    th = ck.get("thresholds") or {}
    t = th.get("hold", 0.5)
    rows = []
    for spec in a.eval:
        name, folder, split, cache = spec.split(":")
        c = EmbeddingCache(cache)
        if c.encoder != ck["encoder"]:
            raise SystemExit(f"{name}: cache encoded with {c.encoder}, model trained with {ck['encoder']}")
        s = score(ck["kind"], model, load_processed(folder, (split,))[split], c, groups, t, th.get("poison", 0.5))
        rep, ci = evaluate(s, t, reps=500)
        if th:
            rep.update({f"conformal/{k}": v for k, v in rates(s, th).items()})
        rows.append({"eval": name, "n": len(s["y"]), **rep, "ci": ci})
        print(name, json.dumps({k: round(rep[k], 4) for k in COLS if k in rep}))
    cols = COLS + (["conformal/benign_hold_or_block"] if th else [])
    lines = ["| test set | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in rows:
        lines.append(f"| {r['eval']} | {r['n']} | " + " | ".join(f"{r.get(c, float('nan')):.3f}" for c in cols) + " |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(rows, indent=1, default=float))
    print("\n".join(lines))
