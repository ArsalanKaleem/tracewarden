"""Per-suite transfer and per-trajectory scores (JAIR revision).

python scripts/per_suite_eval.py runs/sg.pt paper/results/per_suite_hash.json \
    real7B:data/real_qwen:cache/real_qwen real14B:data/real_qwen14b:cache/real_qwen14b
Set TW_T=0.5 to override the threshold stored in the checkpoint (needed for runs/sg_mpnet.pt,
whose stored threshold was overwritten by a calibration on real data).
"""
import json
import os
import sys

import numpy as np

from tracewarden.encoders import EmbeddingCache
from tracewarden.experiments import evaluate, load_processed, score
from tracewarden.featurize import ALL_GROUPS
from tracewarden.training import load_checkpoint

ckpt, out = sys.argv[1], sys.argv[2]
sets = [s.split(":") for s in sys.argv[3:]]
model, ck = load_checkpoint(ckpt)
groups = frozenset(ck.get("groups") or ALL_GROUPS)
th = ck.get("thresholds") or {}
t = float(os.environ["TW_T"]) if os.environ.get("TW_T") else th.get("hold", 0.5)
print("encoder", ck["encoder"], "hold", t, file=sys.stderr)
res = {"threshold": t, "encoder": ck["encoder"], "rows": [], "traj": []}
KEYS = ["traj/precision", "traj/recall", "traj/f1", "step_f1/macro", "injection_em", "hijack_iou",
        "prevent/first_hijack_caught", "prevent/benign_interruption"]
for name, folder, cache in sets:
    trajs = load_processed(folder, ("test",))["test"]
    s = score(ck["kind"], model, trajs, EmbeddingCache(cache), groups, t, th.get("poison", 0.5))
    rep, _ = evaluate(s, t, ci=False)
    res["rows"].append({"set": name, "suite": "all", "n": len(trajs), **{k: rep.get(k) for k in KEYS}})
    for i, tr in enumerate(trajs):
        res["traj"].append({"set": name, "id": tr.id, "suite": tr.domain, "category": tr.category,
                            "y": int(s["y"][i]), "score": float(s["traj_score"][i]),
                            "gold": "".join("BIHF"[g] if isinstance(g, (int, np.integer)) else str(g)[0] for g in s["gold"][i]),
                            "pred": "".join("BIHF"[g] if isinstance(g, (int, np.integer)) else str(g)[0] for g in s["pred"][i]),
                            "nsteps": len(tr.steps), "world": bool(any(tr.world.values()) if tr.world else False)})
    for suite in sorted({tr.domain for tr in trajs}):
        idx = [i for i, tr in enumerate(trajs) if tr.domain == suite]
        sub = {k: ([v[i] for i in idx] if isinstance(v, list) else (v[idx] if isinstance(v, np.ndarray) else v))
               for k, v in s.items()}
        rep, _ = evaluate(sub, t, ci=False)
        res["rows"].append({"set": name, "suite": suite, "n": len(idx),
                            "attacked": int(sub["y"].sum()), **{k: rep.get(k) for k in KEYS}})
for r in res["rows"]:
    print(r["set"], r["suite"], r["n"], " ".join(f"{k.split('/')[-1]}={r[k]:.3f}" for k in KEYS if r.get(k) is not None))
json.dump(res, open(out, "w"), indent=1, default=float)
