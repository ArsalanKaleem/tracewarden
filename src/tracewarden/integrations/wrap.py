"""Framework-agnostic tool wrapper. Works for plain functions and, via `wrap_langchain_tool`,
for LangChain / LangGraph tools (which expose .name and .invoke)."""
from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Callable
from typing import Any

from ..guard import Session


def guard_tool(session_getter: Callable[[], Session], name: str | None = None):
    """Decorator: @guard_tool(lambda: current_session) on any tool function."""

    def deco(fn: Callable[..., Any]):
        tool_name = name or fn.__name__
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            sess = session_getter()
            d = sess.check(tool_name, dict(bound.arguments))
            if not d.allowed:
                sess.skip()
                return f"[TraceWarden {d.action}] {d.reason}"
            out = fn(*args, **kwargs)
            sess.observe(out if isinstance(out, str) else json.dumps(out, default=str))
            return out

        return wrapper

    return deco


def wrap_langchain_tool(tool: Any, session_getter: Callable[[], Session]) -> Any:
    """Patch a LangChain BaseTool in place so every invoke() is checked first. Returns the same tool."""
    original = tool.invoke

    def invoke(input, config=None, **kwargs):  # noqa: A002 - LangChain's signature
        args = input if isinstance(input, dict) else {"input": input}
        if isinstance(args, dict) and "args" in args and "name" in args:  # ToolCall dict
            args = args["args"]
        sess = session_getter()
        d = sess.check(tool.name, args)
        if not d.allowed:
            sess.skip()
            return f"[TraceWarden {d.action}] {d.reason}"
        out = original(input, config, **kwargs)
        sess.observe(str(getattr(out, "content", out)))
        return out

    object.__setattr__(tool, "invoke", invoke)
    return tool
