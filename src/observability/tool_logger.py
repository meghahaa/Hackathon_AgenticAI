"""Tool-invocation logging middleware (AC-07).

Every tool call made by an agent goes through `call_tool`, which enforces a timeout, the
patient-scope check, and appends one JSON line to logs/tool_calls.jsonl:
{timestamp, run_id, session_id, agent, tool_name, args, result, latency_ms, status, error, llm_providers,
 trace_id, span_id}  (llm_providers: model providers that answered in this run so far, e.g. ["gemini", "groq"])
Args/results are PHI-masked before they are written.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from langchain_core.tools import BaseTool

from src import config
from src.guardrails.guards import mask_obj
from src.observability.run_context import (RUN_ID, SESSION_ID, append_jsonl, current_otel_ids, now_iso,
                                          run_providers)


class ToolScopeError(PermissionError):
    """Raised when a tool call targets a patient other than the session's active patient."""


def _truncate(obj: Any, limit: int = 1500) -> Any:
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return obj if len(text) <= limit else text[:limit] + f"...[truncated {len(text) - limit} chars]"


def log_tool_call(*, agent: str, tool_name: str, args: dict, result: Any, latency_ms: float, status: str,
                  error: str | None = None) -> dict:
    record = {
        "timestamp": now_iso(),
        "run_id": RUN_ID.get(),
        "session_id": SESSION_ID.get(),
        "agent": agent,
        "tool_name": tool_name,
        "args": mask_obj(args),
        "result": _truncate(mask_obj(result)),
        "latency_ms": round(latency_ms, 2),
        "status": status,
        "error": error,
        "llm_providers": run_providers(),
        **current_otel_ids(),
    }
    append_jsonl(config.TOOL_LOG, record)
    return record


def _parse_result(raw: Any) -> Any:
    """MCP adapters return text content (optionally with artifacts); decode JSON when possible."""
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        raw = "".join(part.get("text", "") for part in raw)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


async def call_tool(tool: BaseTool, args: dict, *, agent: str, active_patient_id: str | None = None,
                    timeout_s: float | None = None) -> Any:
    """Invoke a LangChain/MCP tool with scope check, timeout and structured logging."""
    start = time.perf_counter()
    try:
        requested = args.get("patient_id")
        if requested and active_patient_id and requested.upper() != active_patient_id.upper():
            raise ToolScopeError("tool call targets a patient outside the active session scope")
        raw = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout_s or config.TOOL_TIMEOUT_S)
        result = _parse_result(raw)
        status = "error" if isinstance(result, dict) and result.get("error") else "ok"
        log_tool_call(agent=agent, tool_name=tool.name, args=args, result=result,
                      latency_ms=(time.perf_counter() - start) * 1000, status=status,
                      error=result.get("error") if status == "error" else None)
        return result
    except Exception as exc:
        status = "timeout" if isinstance(exc, asyncio.TimeoutError) else (
            "blocked" if isinstance(exc, ToolScopeError) else "error")
        log_tool_call(agent=agent, tool_name=tool.name, args=args, result=None,
                      latency_ms=(time.perf_counter() - start) * 1000, status=status,
                      error=f"{type(exc).__name__}: {str(exc)[:300]}")
        raise
