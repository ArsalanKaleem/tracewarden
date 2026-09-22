"""Loader for the AgentDrift benchmark (arXiv 2609.06972, CC BY 4.0).

Record schema (v2.0): id, agent, category, source_category, task, world{user,email,company,date,contacts[]},
steps[{thought, tool, args, obs, label}], attack_type, compliance, split, version.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import Step, Trajectory

PARTS = ("train", "val", "test")


def record_to_traj(r: dict) -> Trajectory:
    steps = [
        Step(tool=s["tool"], args=s.get("args") or {}, thought=s.get("thought", ""),
             obs=s.get("obs", ""), label=s.get("label"))
        for s in r["steps"]
    ]
    return Trajectory(
        id=str(r["id"]), steps=steps, user_task=r.get("task", ""), world=r.get("world", {}),
        domain=r.get("agent", ""), category=r.get("category", ""), split=r.get("split", ""),
        source="agentdrift",
        meta={k: r.get(k) for k in ("source_category", "attack_type", "compliance", "version") if k in r},
    )


def load_part(root: str | Path, part: str) -> list[Trajectory]:
    """root = path to `data_taskdisjoint` (recommended) or `data`. part in train|val|test."""
    root = Path(root)
    jl = root / f"{part}.jsonl"
    if jl.exists():
        with open(jl, encoding="utf-8") as f:
            trajs = [record_to_traj(json.loads(line)) for line in f if line.strip()]
    else:
        trajs = [record_to_traj(json.loads(p.read_text(encoding="utf-8")))
                 for p in sorted((root / part).glob("*.json"))]
    for t in trajs:  # the folder defines the split (records carry the stratified split)
        t.split = part
    return trajs


def load(root: str | Path) -> dict[str, list[Trajectory]]:
    return {p: load_part(root, p) for p in PARTS}
