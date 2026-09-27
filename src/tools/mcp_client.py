"""Consume the custom MCP server through langchain-mcp-adapters (stdio).

`MCPToolbox` keeps ONE stdio session open for a whole CLI run (no process-per-call), exposes the
server's tools as LangChain tools and reads resources. Client-side protocol events are appended to
logs/mcp_transcript.jsonl alongside the server-side records.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import AsyncExitStack
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from src import config
from src.observability.run_context import append_jsonl, now_iso

SERVER_NAME = "discharge"
SERVER_PATH = config.ROOT / "mcp_server" / "server.py"


def _log(method: str, detail: dict[str, Any], started: float, status: str = "ok") -> None:
    append_jsonl(config.MCP_TRANSCRIPT, {"timestamp": now_iso(), "side": "client", "method": method,
                                        "status": status, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                                        **detail})


class MCPToolbox:
    def __init__(self) -> None:
        self.client = MultiServerMCPClient({
            SERVER_NAME: {"command": sys.executable, "args": [str(SERVER_PATH)], "transport": "stdio",
                          "cwd": str(config.ROOT),
                          "env": {**os.environ, "MCP_TRANSCRIPT_PATH": str(config.MCP_TRANSCRIPT)}},
        })
        self._stack = AsyncExitStack()
        self.session = None
        self.tools: dict[str, BaseTool] = {}

    async def __aenter__(self) -> "MCPToolbox":
        t0 = time.perf_counter()
        self.session = await self._stack.enter_async_context(self.client.session(SERVER_NAME))
        _log("initialize", {"server": SERVER_NAME, "transport": "stdio"}, t0)
        t0 = time.perf_counter()
        tools = await load_mcp_tools(self.session)
        self.tools = {t.name: t for t in tools}
        _log("tools/list", {"tools": sorted(self.tools)}, t0)
        t0 = time.perf_counter()
        res = await self.session.list_resources()
        _log("resources/list", {"resources": [str(r.uri) for r in res.resources]}, t0)
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    def tool(self, name: str) -> BaseTool:
        return self.tools[name]

    async def read_resource(self, uri: str) -> str:
        t0 = time.perf_counter()
        result = await self.session.read_resource(uri)
        text = "".join(getattr(c, "text", "") or "" for c in result.contents)
        _log("resources/read", {"uri": uri, "chars": len(text)}, t0)
        return text
