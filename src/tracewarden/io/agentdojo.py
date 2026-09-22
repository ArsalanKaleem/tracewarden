"""Convert AgentDojo benchmark logs into TraceWarden trajectories with automatic step labels (N6).

AgentDojo (github.com/ethz-spylab/agentdojo) writes one JSON log per (user task, injection task) run
containing the chat messages, the injections that were placed, and ground-truth verdicts
(`utility` = user task solved, `security` = attacker goal achieved).

Labeling rule:
  * injection_point  - a tool output that contains a placed payload, before the first hijacked action;
  * failed_injection - a tool output that contains a payload, but no hijacked action follows it;
  * hijacked         - a tool call issued after a poisoned observation whose arguments contain an
                       attacker value (address, account, URL, ... present in the payload but absent from
                       the user task and the user's environment), or that `attacker_calls` flags;
  * benign           - everything else.

The log format changes between AgentDojo versions; everything format-specific lives in `_text` and
`iter_calls`. Open one log by hand before converting a whole run.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ..features.provenance import extract_values
from ..schema import Step, Trajectory


def _text(content: Any) -> str:
    """AgentDojo messages carry either a string or a list of content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(b.get("content") or b.get("text") or "") if isinstance(b, dict) else str(b) for b in content]
        return "\n".join(p for p in parts if p)
    return str(content)


def _parse_args(args: Any) -> dict:
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            return {"_raw": args}
    return dict(args or {}) if isinstance(args, dict) else {"_raw": args}


def iter_calls(messages: list[dict]) -> Iterable[tuple[str, dict, str, str]]:
    """Yield (tool_name, args, thought, observation) for every tool call, in order."""
    by_id: dict[str, str] = {}
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    for m in tool_msgs:
        out = _text(m.get("content"))
        if m.get("error"):
            out = f"{out}\nERROR: {m['error']}".strip()
        m["_out"] = out
        cid = m.get("tool_call_id") or (m.get("tool_call") or {}).get("id")
        if cid is not None:
            by_id[str(cid)] = out
    k = 0
    for m in messages:
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        thought = _text(m.get("content"))
        for call in m["tool_calls"]:
            fn, args = call.get("function"), call.get("args", call.get("arguments"))
            if isinstance(fn, dict):  # OpenAI style {"function": {"name", "arguments"}}
                fn, args = fn.get("name"), fn.get("arguments")
            cid = call.get("id")
            if cid is not None and str(cid) in by_id:
                obs = by_id[str(cid)]
            else:  # positional fallback
                obs = tool_msgs[k]["_out"] if k < len(tool_msgs) else ""
            k += 1
            yield str(fn), _parse_args(args), thought, obs


def label_steps(steps: list[Step], payloads: list[str], benign_context: str,
                attacker_calls: Callable[[Step], bool] | None = None) -> None:
    """Assign labels in place. `benign_context` = user task + environment text the user legitimately owns."""
    payloads = [p for p in payloads if p and p.strip()]
    ctx = benign_context.lower()
    attacker_vals = {v for p in payloads for v in extract_values(p, entities_only=True) if v not in ctx}
    seen_poison = False
    for s in steps:
        s.label = "benign"
        if seen_poison:
            arg_vals = set(extract_values(s.args, entities_only=True))
            if (arg_vals & attacker_vals) or (attacker_calls is not None and attacker_calls(s)):
                s.label = "hijacked"
        if s.label == "benign" and any(p.strip()[:120] in s.obs for p in payloads):
            s.label = "injection_point"  # provisional
            seen_poison = True
    h = [i for i, s in enumerate(steps) if s.label == "hijacked"]
    for i, s in enumerate(steps):
        if s.label == "injection_point" and (not h or i > h[0]):
            s.label = "failed_injection"


def convert_log(rec: dict, source: str, world: dict | None = None,
                attacker_calls: Callable[[Step], bool] | None = None) -> Trajectory | None:
    messages = rec.get("messages") or []
    steps = [Step(tool=n, args=a, thought=t, obs=o) for n, a, t, o in iter_calls(messages)]
    if not steps:
        return None
    user_task = next((_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
    inj = rec.get("injections") or {}
    payloads = list(inj.values()) if isinstance(inj, dict) else list(inj)
    world = world or {}
    label_steps(steps, payloads, user_task + " " + json.dumps(world), attacker_calls)
    traj = Trajectory(
        id=f"{source}:{rec.get('suite_name', '')}:{rec.get('user_task_id', '')}:{rec.get('injection_task_id', '')}",
        steps=steps, user_task=user_task, world=world, domain=str(rec.get("suite_name", "")), source=source,
        meta={"attack": rec.get("attack_type"), "utility": rec.get("utility"), "security": rec.get("security"),
              "user_task_id": rec.get("user_task_id"), "injection_task_id": rec.get("injection_task_id")},
    )
    if traj.compromised:
        traj.category = "attacked"
    elif any(s.label == "failed_injection" for s in steps):
        traj.category = "failed_attack"
    else:
        traj.category = "benign"
    # disagreement with AgentDojo's own oracle goes to the hand audit; it is never silently "fixed"
    if payloads and rec.get("security") is not None and bool(rec["security"]) != bool(traj.compromised):
        traj.meta["oracle_disagrees"] = True
    return traj


def convert_dir(logdir: str | Path, source: str, world_of: Callable[[dict], dict] | None = None) -> list[Trajectory]:
    out = []
    for fp in sorted(Path(logdir).rglob("*.json")):
        try:
            rec = json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(rec, dict) and "messages" in rec:
            t = convert_log(rec, source, world_of(rec) if world_of else None)
            if t is not None:
                out.append(t)
    return out
