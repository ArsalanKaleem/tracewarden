"""Argument provenance (taint) features - novel addition N2.

For every salient value in a proposed tool call's arguments, find where it first appeared:
the user's task, the user's known world, an earlier (untrusted) tool observation, or nowhere.
A value that only an untrusted observation introduced is "tainted". Only information available
before the call executes is used, so the features are valid for pre-dispatch blocking.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .normalize import normalize

PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    "url": r"https?://[^\s\"'<>]+",
    "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b",
    "phone": r"\+?\d[\d\s().-]{7,}\d",
    "number": r"\b\d{4,}(?:[.,]\d+)?\b",
    "path": r"(?:/[\w.-]+){2,}|[A-Za-z]:\\[^\s]+",
    "domain": r"\b[\w-]+\.(?:com|net|org|io|co|ai|xyz|info|biz|dev|app|me)\b",
}
RX = {k: re.compile(v) for k, v in PATTERNS.items()}
MIN_FREE_TEXT = 6


def extract_values(obj: Any, entities_only: bool = False) -> list[str]:
    """All salient atomic values: entities (emails, urls, ibans...) plus longer free-text strings."""
    out: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif o is not None and not isinstance(o, bool):
            s = normalize(str(o)).strip()
            found = [m.group(0) for r in RX.values() for m in r.finditer(s)]
            if found:
                out.extend(found)
            elif not entities_only and len(s) >= MIN_FREE_TEXT and len(s) <= 200:
                out.append(s)

    walk(obj)
    return list(dict.fromkeys(v.lower().strip(".,;") for v in out if v))


def provenance_feats(args: dict, user_task: str, world: dict, prior_obs: list[str]) -> list[float]:
    """Features for ONE proposed call. prior_obs = observations of steps strictly before this call."""
    vals = extract_values(args)
    if not vals:
        return [0.0] * N_PROV
    task = normalize(user_task).lower()
    wtxt = normalize(json.dumps(world, default=str)).lower()
    obs_l = [normalize(o).lower() for o in prior_obs]
    t = len(obs_l)
    in_task = [v in task for v in vals]
    in_world = [v in wtxt for v in vals]
    first = [next((j for j, o in enumerate(obs_l) if v in o), None) for v in vals]
    tainted = [f is not None and not a and not b for f, a, b in zip(first, in_task, in_world)]
    novel = [f is None and not a and not b for f, a, b in zip(first, in_task, in_world)]
    dists = [t - f for f, tn in zip(first, tainted) if tn]
    n = len(vals)
    return [1.0, sum(in_task) / n, sum(in_world) / n, sum(tainted) / n, float(any(tainted)),
            sum(novel) / n, (min(dists) if dists else 0) / 10.0]


N_PROV = 7
PROV_NAMES = ["has_values", "frac_task", "frac_world", "frac_tainted", "any_tainted", "frac_novel", "taint_dist"]
