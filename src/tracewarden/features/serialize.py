"""Text views of a step. Each view is encoded separately (see encoders.encode_corpus)."""
from __future__ import annotations

import json

from ..schema import Step, Trajectory
from .normalize import normalize

MAX_CHARS = 2000  # all-mpnet-base-v2 truncates at 384 tokens anyway


def _args(args: dict) -> str:
    return json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)


def full(s: Step) -> str:  # DriftNet's serialization (post-hoc model)
    return normalize(f"TOOL: {s.tool} | THOUGHT: {s.thought} | ARGS: {_args(s.args)} | OBS: {s.obs}")[:MAX_CHARS]


def action(s: Step) -> str:  # what the agent proposes, known BEFORE the call executes
    return normalize(f"TOOL: {s.tool} | THOUGHT: {s.thought} | ARGS: {_args(s.args)}")[:MAX_CHARS]


def obs(s: Step) -> str:  # what came back from the tool
    return normalize(f"OBS from {s.tool}: {s.obs}")[:MAX_CHARS]


def task(t: Trajectory) -> str:
    return normalize(f"TASK: {t.user_task}")[:MAX_CHARS]


VIEWS = {"full": full, "action": action, "obs": obs}
