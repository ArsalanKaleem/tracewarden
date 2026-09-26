"""Audit every trajectory in a labeled dataset at once, instead of reading 30 by hand.

Rule-based consistency checks find the label errors a human reviewer would find, then you read only
the flagged ones. Hand review stays the gold standard (it is what validates these rules), but this
catches systematic bugs across thousands of trajectories in seconds.

    python scripts/audit_labels.py --data data/agentdojo_raw/dojo_qwen14b_ii_final.jsonl
    python scripts/audit_labels.py --data ... --show missed_hijack --n 5     # read flagged examples
    python scripts/audit_labels.py --data ... --out paper/results/audit_auto_qwen14b.json

Checks (each is a heuristic; a flag means "look at this", not "this is wrong"):

  missed_hijack        a benign step shares an argument value with a hijacked step in the same run
                       -> the attacker's instruction had several parts and only some were caught
  attacker_value_benign a benign step after a poisoned observation carries an attacker entity
  oracle_says_attacked AgentDojo's security oracle says the attack succeeded, but no step is hijacked
  hijack_no_injection  a hijacked step with no injection point before it -> the payload was not located
  injection_no_marker  a step labeled poisoned whose observation has no payload marker -> false positive
  hijack_before_poison a hijacked step earlier than every poisoned observation -> ordering bug
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tracewarden.features.provenance import extract_values
from tracewarden.io.agentdojo import _key, _ws, same_value
from tracewarden.schema import Trajectory, load_jsonl

MARKERS = ("<information>", "pending follow-up", "important message from me", "ignore previous",
           "before you can solve", "reminder: the pending")


DATEISH = re.compile(r"^[\d\s:/.,-]+$")  # dates, amounts, ids: shared by chance, not by intent


def _arg_values(step) -> set[str]:
    """Every string argument value worth comparing, including short names like \"Fred\"."""
    out: set[str] = []

    def walk(o):
        if isinstance(o, dict):
            [walk(v) for v in o.values()]
        elif isinstance(o, (list, tuple)):
            [walk(v) for v in o]
        elif isinstance(o, str) and 3 <= len(o) <= 60 and not DATEISH.match(o):
            out.append(o.strip())

    out = []
    walk(step.args)
    return set(out)


def check(t: Trajectory) -> list[dict]:
    """Return a list of {check, step, why} findings for one trajectory."""
    out = []
    hij = [i for i, s in enumerate(t.steps) if s.label == "hijacked"]
    poisoned = [i for i, s in enumerate(t.steps) if s.label in ("injection_point", "failed_injection")]
    hij_values = {v for i in hij for v in _arg_values(t.steps[i])}
    ctx = _ws(t.user_task + " " + json.dumps(t.world, default=str))

    # 1. a benign step reusing a value that a hijacked step also used
    for i, s in enumerate(t.steps):
        if s.label != "benign" or not hij:
            continue
        shared = [v for v in _arg_values(s)
                  if v not in ctx and any(same_value(v, h) or _key(v) == _key(h) for h in hij_values)]
        if shared and i > min(hij) - 2:  # near or after the hijack, not an innocent earlier mention
            out.append({"check": "missed_hijack", "step": i,
                        "why": f"{s.tool} shares {shared[:3]} with a hijacked step"})

    # 2. a benign step after poison carrying an entity the user's world never mentions
    if poisoned:
        first_poison = min(poisoned)
        for i, s in enumerate(t.steps):
            if s.label != "benign" or i <= first_poison:
                continue
            ext = [v for v in extract_values(s.args, entities_only=True) if _ws(v) not in ctx]
            seen_before = {v for j in range(first_poison + 1)
                           for v in extract_values(t.steps[j].obs, entities_only=True)}
            novel = [v for v in ext if not any(same_value(v, o) for o in seen_before)]
            if novel:
                out.append({"check": "attacker_value_benign", "step": i,
                            "why": f"{s.tool} uses {novel[:2]} which the user never mentioned"})

    # 3. oracle disagreement in the dangerous direction
    if t.meta.get("security") is True and not hij:
        out.append({"check": "oracle_says_attacked", "step": -1,
                    "why": "AgentDojo scored the attack as successful but no step is labeled hijacked"})

    # 4. a hijack with no located injection point
    if hij and not poisoned:
        out.append({"check": "hijack_no_injection", "step": hij[0],
                    "why": "hijacked step with no poisoned observation anywhere"})

    # 5. poisoned label with no payload marker in the observation
    for i in poisoned:
        low = _ws(t.steps[i].obs)
        if not any(m in low for m in MARKERS):
            out.append({"check": "injection_no_marker", "step": i,
                        "why": f"{t.steps[i].tool} output has no known payload marker"})

    # 6. ordering: hijack before any poison arrived
    if hij and poisoned and min(hij) < min(poisoned):
        out.append({"check": "hijack_before_poison", "step": min(hij),
                    "why": "hijacked step precedes every poisoned observation"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="a labeled .jsonl of trajectories")
    ap.add_argument("--show", default=None, help="print examples for one check name")
    ap.add_argument("--n", type=int, default=3, help="how many examples to print")
    ap.add_argument("--out", default=None, help="write the full findings as JSON")
    a = ap.parse_args()

    trajs = load_jsonl(a.data)
    findings = {}
    for t in trajs:
        f = check(t)
        if f:
            findings[t.id] = f

    counts = Counter(x["check"] for f in findings.values() for x in f)
    flagged = len(findings)
    print(f"{len(trajs)} trajectories | {flagged} flagged ({flagged / max(len(trajs), 1):.1%}) | "
          f"{sum(counts.values())} findings\n")
    for name, n in counts.most_common():
        print(f"  {name:24s} {n:5d}   ({len({k for k, v in findings.items() if any(x['check'] == name for x in v)})} trajectories)")
    print("\nA clean dataset flags only a few percent. High `missed_hijack` means the labeler is missing "
          "parts of multi-action attacker goals; high `injection_no_marker` means it is over-flagging.")

    if a.show:
        by_id = {t.id: t for t in trajs}
        shown = 0
        for tid, f in findings.items():
            hits = [x for x in f if x["check"] == a.show]
            if not hits:
                continue
            t = by_id[tid]
            print("\n" + "=" * 100)
            print(f"{tid} | {t.category} | {t.label_string}")
            print("TASK:", t.user_task[:140])
            for i, s in enumerate(t.steps):
                mark = " <<<" if any(h["step"] == i for h in hits) else ""
                print(f"  [{i}] {s.label:16s} {s.tool}({str(s.args)[:90]}){mark}")
            for h in hits:
                print("  ->", h["why"])
            shown += 1
            if shown >= a.n:
                break

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"data": a.data, "n_trajectories": len(trajs), "n_flagged": flagged,
             "counts": dict(counts), "findings": findings}, indent=1))
        print("\nwritten to", a.out)


if __name__ == "__main__":
    main()
