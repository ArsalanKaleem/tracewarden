"""Rollback / incident report - novel addition N9: turn step labels into recovery instructions."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .schema import LABELS, Trajectory


@dataclass
class Report:
    trajectory_id: str
    verdict: str                              # compromised | attempted | clean
    step_labels: list[str]
    hijack_scores: list[float]
    poison_scores: list[float]
    injection_points: list[int] = field(default_factory=list)
    hijacked_steps: list[int] = field(default_factory=list)
    rollback: list[dict] = field(default_factory=list)     # executed actions to undo, latest first
    quarantine: list[dict] = field(default_factory=list)   # content sources to distrust
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        icon = {"compromised": "COMPROMISED", "attempted": "ATTACK ATTEMPTED (resisted)", "clean": "CLEAN"}[self.verdict]
        md = [f"## TraceWarden report: {icon}", f"Trajectory `{self.trajectory_id}`", ""]
        md += ["| step | label | hijack score | poison score |", "|---|---|---|---|"]
        for i, (lab, h, p) in enumerate(zip(self.step_labels, self.hijack_scores, self.poison_scores)):
            md.append(f"| {i} | {lab} | {h:.3f} | {p:.3f} |")
        if self.quarantine:
            md += ["", "### Quarantine these sources"]
            md += [f"- step {q['step']}: output of `{q['tool']}` - \"{q['excerpt']}\"" for q in self.quarantine]
        if self.rollback:
            md += ["", "### Roll back these actions (latest first)"]
            md += [f"- step {r['step']}: `{r['tool']}` with args `{json.dumps(r['args'], ensure_ascii=False)}`"
                   for r in self.rollback]
        if self.notes:
            md += ["", "### Notes"] + [f"- {n}" for n in self.notes]
        return "\n".join(md)


def build_report(t: Trajectory, labels: list[int], hijack: list[float], poison: list[float],
                 executed: list[bool] | None = None) -> Report:
    names = [LABELS[k] for k in labels]
    executed = executed if executed is not None else [True] * len(t.steps)
    inj = [i for i, n in enumerate(names) if n == "injection_point"]
    hij = [i for i, n in enumerate(names) if n == "hijacked"]
    verdict = "compromised" if hij else ("attempted" if "failed_injection" in names else "clean")
    rb = [{"step": i, "tool": t.steps[i].tool, "args": t.steps[i].args} for i in reversed(hij) if executed[i]]
    q = [{"step": i, "tool": t.steps[i].tool, "excerpt": t.steps[i].obs[:160].replace("\n", " ")}
         for i, n in enumerate(names) if n in ("injection_point", "failed_injection")]
    notes = []
    blocked = [i for i in hij if not executed[i]]
    if blocked:
        notes.append(f"{len(blocked)} hijacked action(s) were stopped before execution: steps {blocked}.")
    if hij and not inj:
        notes.append("Hijack detected without a localized injection point; review all tool outputs before step "
                     f"{hij[0]}.")
    return Report(t.id, verdict, names, [float(x) for x in hijack], [float(x) for x in poison], inj, hij, rb, q, notes)
