"""EXPERIMENTAL MCP proxy: sits between an MCP client (Claude Desktop, an IDE, an agent) and one upstream
MCP server, scoring every tools/call before forwarding it.

    tracewarden-mcp --checkpoint tracewarden-core.pt --task "what the user asked" \
        --world world.json -- python -m some_upstream_server

Limitation: an MCP server never sees the user's prompt, so pass --task (or leave it empty and accept weaker
goal-anchoring features). Check the installed `mcp` SDK version; its server APIs have changed between releases.
Requires: pip install "tracewarden[mcp]"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def _serve(args) -> None:
    import mcp.types as types
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    from ..guard import Guard

    guard = Guard.load(args.checkpoint, encoder=args.encoder)
    world = json.load(open(args.world)) if args.world else {}
    session = guard.session(user_task=args.task or "", world=world, session_id="mcp")
    upstream = StdioServerParameters(command=args.upstream[0], args=args.upstream[1:])

    async with stdio_client(upstream) as (ur, uw), ClientSession(ur, uw) as client:
        await client.initialize()
        server = Server("tracewarden-proxy")

        @server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return (await client.list_tools()).tools

        @server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            d = session.check(name, arguments or {})
            if d.action == "block" or (d.action == "hold" and not args.allow_holds):
                session.skip()
                print(f"[tracewarden] {d.action} {name}: {d.reason}", file=sys.stderr)
                return [types.TextContent(type="text", text=f"Tool call {d.action}ed by TraceWarden: {d.reason}")]
            res = await client.call_tool(name, arguments or {})
            text = "\n".join(getattr(c, "text", "") for c in res.content)
            session.observe(text)
            return list(res.content)

        async with stdio_server() as (r, w):
            await server.run(r, w, server.create_initialization_options())


def main(argv=None) -> None:
    p = argparse.ArgumentParser("tracewarden-mcp")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--encoder", default=None)
    p.add_argument("--task", default="")
    p.add_argument("--world", default=None, help="JSON file with the user's known world (email, contacts)")
    p.add_argument("--allow-holds", action="store_true", help="forward held calls (log only)")
    p.add_argument("upstream", nargs=argparse.REMAINDER, help="-- command to start the upstream MCP server")
    a = p.parse_args(argv)
    a.upstream = [x for x in a.upstream if x != "--"]
    if not a.upstream:
        p.error("give the upstream server command after --")
    asyncio.run(_serve(a))


if __name__ == "__main__":
    main()
