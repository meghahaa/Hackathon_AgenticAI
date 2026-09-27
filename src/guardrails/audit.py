"""Audit middleware: appends every consequential agent action to logs/agent_actions.jsonl (AC-10).

Record schema: {timestamp, run_id, session_id, actor, action, tool, decision, reason, patient_ref,
                llm_providers, trace_id, span_id, details}. llm_providers: model providers that answered in this run
so far (a Groq request additionally writes its own actor=llm_router, action=quota_fallback record). Patient identifiers are pseudonymised; details are PHI-masked.
"""
from __future__ import annotations

from typing import Any

from src import config
from src.guardrails.guards import mask_obj, patient_ref
from src.observability.run_context import (RUN_ID, SESSION_ID, append_jsonl, current_otel_ids, now_iso,
                                          run_providers)


def audit(actor: str, action: str, decision: str, *, tool: str | None = None, reason: str = "",
          patient_id: str | None = None, details: dict[str, Any] | None = None) -> dict[str, Any]:
    record = {
        "timestamp": now_iso(),
        "run_id": RUN_ID.get(),
        "session_id": SESSION_ID.get(),
        "actor": actor,
        "action": action,
        "tool": tool,
        "decision": decision,
        "reason": reason,
        "patient_ref": patient_ref(patient_id) if patient_id else None,
        "llm_providers": run_providers(),
        **current_otel_ids(),
        "details": mask_obj(details or {}),
    }
    append_jsonl(config.AUDIT_LOG, record)
    return record
