"""Per-run correlation context shared by the tool log, the audit trail and Phoenix spans."""
from __future__ import annotations

import contextvars
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="no-run")
SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="no-session")
# every model request of the current run (src.llm.ModelChain): {"provider", "model", "run_name", "ok", "error"}.
# A list set once per run and appended to (not re-set), so requests made in child tasks are visible to the run.
MODEL_CALLS: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar("model_calls", default=None)


def record_model_call(**call: Any) -> None:
    calls = MODEL_CALLS.get()
    if calls is not None:
        calls.append(call)


def run_providers() -> list[str]:
    """Providers that answered a model request in the current run so far (e.g. ["gemini"], ["gemini", "groq"])."""
    return sorted({c["provider"] for c in MODEL_CALLS.get() or [] if c.get("ok")})


def provider_summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-provider request counts ({provider: {"ok": n, "failed": m}}) and whether any Groq request answered."""
    out: dict[str, dict[str, int]] = {}
    for c in calls:
        slot = out.setdefault(c["provider"], {"ok": 0, "failed": 0})
        slot["ok" if c.get("ok") else "failed"] += 1
    return {"by_provider": out, "groq_calls": out.get("groq", {}).get("ok", 0) + out.get("groq", {}).get("failed", 0),
            "groq_involved": "groq" in out}

_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def current_otel_ids() -> dict[str, str | None]:
    """trace_id/span_id linking a log record to its Phoenix span: the span of the LangGraph node /
    LangChain run currently executing (openinference), else the active OpenTelemetry span."""
    try:
        from opentelemetry import trace

        span = None
        try:
            from openinference.instrumentation.langchain import get_current_span

            span = get_current_span()
        except Exception:
            span = None
        ctx = (span or trace.get_current_span()).get_span_context()
        if ctx and ctx.is_valid:
            return {"trace_id": format(ctx.trace_id, "032x"), "span_id": format(ctx.span_id, "016x")}
    except Exception:
        pass
    return {"trace_id": None, "span_id": None}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str, ensure_ascii=False)
    with _lock, path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
