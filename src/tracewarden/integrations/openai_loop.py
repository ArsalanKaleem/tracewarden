"""Adapter for any OpenAI-compatible tool-calling loop (OpenAI, vLLM, Ollama, Groq, ...)."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..guard import Decision, Session


class ToolCallBlocked(Exception):
    def __init__(self, decision: Decision):
        super().__init__(decision.reason)
        self.decision = decision


def guarded_dispatch(session: Session, name: str, arguments: dict | str, execute: Callable[[str, dict], Any],
                     thought: str = "", on_hold: Callable[[Decision], bool] | None = None) -> str:
    """Check a proposed call, run it only if allowed, record the output. Returns the text the agent sees.

    on_hold(decision) -> True to approve a held call (e.g. ask a human); default: do not run it.
    """
    args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
    d = session.check(name, args, thought)
    if d.action == "block" or (d.action == "hold" and not (on_hold and on_hold(d))):
        session.skip()
        return json.dumps({"error": f"tool call {'blocked' if d.action == 'block' else 'held for review'} by TraceWarden",
                           "reason": d.reason})
    out = execute(name, args)
    text = out if isinstance(out, str) else json.dumps(out, default=str)
    session.observe(text)
    return text


def run_tool_calls(session: Session, message: Any, tools: dict[str, Callable[..., Any]],
                   on_hold: Callable[[Decision], bool] | None = None) -> list[dict]:
    """Execute every tool call in an OpenAI chat-completion `message`, guarded.

    Returns the list of {"role": "tool", ...} messages to append to the conversation.
    """
    thought = getattr(message, "content", None) or ""
    out = []
    for tc in getattr(message, "tool_calls", None) or []:
        fn = tc.function
        text = guarded_dispatch(session, fn.name, fn.arguments, lambda n, a: tools[n](**a), thought, on_hold)
        out.append({"role": "tool", "tool_call_id": tc.id, "content": text})
    return out
