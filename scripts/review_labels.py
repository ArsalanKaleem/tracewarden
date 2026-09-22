"""Day 12/15 - hand-audit auto-labels. Shows N random trajectories, you answer y/n per trajectory,
and the agreement rate is written next to the data (report it in the dataset card).

python scripts/review_labels.py --data data/agentdojo_qwen/test.jsonl --n 30
"""
import argparse
import json
import random
import textwrap
from pathlib import Path

from tracewarden.schema import load_jsonl

COLOR = {"benign": "\033[37m", "injection_point": "\033[33m", "hijacked": "\033[31m", "failed_injection": "\033[35m"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    T = load_jsonl(a.data)
    sample = random.Random(a.seed).sample(T, min(a.n, len(T)))
    verdicts = []
    for k, t in enumerate(sample):
        print("\n" + "=" * 100 + f"\n[{k + 1}/{len(sample)}] {t.id}  category={t.category}  labels={t.label_string}")
        print("TASK:", t.user_task)
        for i, s in enumerate(t.steps):
            print(f"{COLOR[s.label]}  [{i}] {s.label:16s}\033[0m {s.tool}({json.dumps(s.args)[:150]})")
            print(textwrap.indent(textwrap.shorten(s.obs, 300), "        -> "))
        ans = ""
        while ans not in ("y", "n", "s"):
            ans = input("Labels correct? [y]es / [n]o / [s]kip: ").strip().lower()
        note = input("note (optional): ") if ans == "n" else ""
        if ans != "s":
            verdicts.append({"id": t.id, "correct": ans == "y", "note": note})
    ok = sum(v["correct"] for v in verdicts)
    res = {"n": len(verdicts), "correct": ok, "agreement": ok / max(len(verdicts), 1), "items": verdicts}
    out = Path(a.data).with_suffix(".audit.json")
    out.write_text(json.dumps(res, indent=1))
    print(f"\nagreement {res['agreement']:.3f} on {res['n']} trajectories -> {out}")
