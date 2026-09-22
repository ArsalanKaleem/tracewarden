"""The one internal trajectory format used by every part of the project.

AgentDrift, AgentDojo logs, and live runtime sessions are all converted into these classes,
which is what makes synthetic-to-real transfer experiments possible.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

LABELS = ["benign", "injection_point", "hijacked", "failed_injection"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
SHORT = {"benign": "B", "injection_point": "I", "hijacked": "H", "failed_injection": "F"}


@dataclass
class Step:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    obs: str = ""
    label: str | None = None  # one of LABELS; None for unlabeled runtime steps

    def __post_init__(self):
        if self.label is not None and self.label not in LABEL2ID:
            raise ValueError(f"unknown label {self.label!r}")
        if not isinstance(self.args, dict):
            self.args = {"_raw": self.args}
        self.obs = "" if self.obs is None else str(self.obs)
        self.thought = "" if self.thought is None else str(self.thought)


@dataclass
class Trajectory:
    id: str
    steps: list[Step]
    user_task: str = ""
    world: dict[str, Any] = field(default_factory=dict)
    domain: str = ""
    category: str = ""  # benign | attacked | failed_attack | hard_negative
    split: str = ""
    source: str = ""  # agentdrift | agentdojo-<model> | runtime | ...
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- label helpers -------------------------------------------------
    @property
    def labeled(self) -> bool:
        return all(s.label is not None for s in self.steps)

    @property
    def compromised(self) -> int:
        return int(any(s.label == "hijacked" for s in self.steps))

    @property
    def label_string(self) -> str:
        return "".join(SHORT.get(s.label or "", "?") for s in self.steps)

    @property
    def injection_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.steps) if s.label == "injection_point"]

    @property
    def poisoned_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.steps) if s.label in ("injection_point", "failed_injection")]

    @property
    def hijacked_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.steps) if s.label == "hijacked"]

    # ---- (de)serialization --------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trajectory:
        d = dict(d)
        d["steps"] = [s if isinstance(s, Step) else Step(**s) for s in d["steps"]]
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def save_jsonl(trajs: list[Trajectory], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(t.to_json() + "\n")


def load_jsonl(path: str) -> list[Trajectory]:
    with open(path, encoding="utf-8") as f:
        return [Trajectory.from_dict(json.loads(line)) for line in f if line.strip()]
