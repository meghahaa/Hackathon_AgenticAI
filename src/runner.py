"""Shared async driver used by the CLI, the evidence regenerator, the eval harness and the API."""
from __future__ import annotations

import time
import uuid
from contextlib import AsyncExitStack
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.graph import SIGNOFF_DECISIONS, CopilotRuntime, build_graph, run_config
from src.guardrails.guards import mask_phi, patient_ref
from src.memory.memory_store import LongTermMemory, open_checkpointer
from src.observability.run_context import MODEL_CALLS, RUN_ID, SESSION_ID, provider_summary
from src.tools.mcp_client import MCPToolbox


class Copilot:
    """Async context manager owning the MCP session, the checkpointer and the compiled graph."""

    def __init__(self, checkpoint_db=None, memory_db=None) -> None:
        self._stack = AsyncExitStack()
        self.checkpoint_db, self.memory_db = checkpoint_db, memory_db
        self.graph = None

    async def __aenter__(self) -> "Copilot":
        import asyncio

        from src.tools.rag_tool import get_index

        # latency optimisation: build the embedding index once at startup instead of inside the first
        # tool call (cold start measured at ~17 s in the baseline run, reports/optimization/)
        await asyncio.to_thread(get_index)
        toolbox = await self._stack.enter_async_context(MCPToolbox())
        saver = await self._stack.enter_async_context(open_checkpointer(self.checkpoint_db))
        self.runtime = CopilotRuntime(toolbox, LongTermMemory(self.memory_db))
        self.graph = build_graph(self.runtime, checkpointer=saver)
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    async def turn(self, session_id: str, patient_id: str, message: str) -> dict[str, Any]:
        return await self._drive(session_id, patient_id, mask_phi(message), "copilot_turn",
                                 {"messages": [HumanMessage(message)], "patient_id": patient_id})

    async def pending_signoff(self, session_id: str) -> dict[str, Any] | None:
        """The PHI-masked interrupt payload if this session is paused at clinician_signoff, else None."""
        snap = await self.graph.aget_state({"configurable": {"thread_id": session_id}})
        if "clinician_signoff" not in (snap.next or ()) or not snap.interrupts:
            return None
        return {"patient_id": snap.values.get("patient_id"), "payload": snap.interrupts[0].value}

    async def resume(self, session_id: str, decision: str, clinician: str, note: str = "") -> dict[str, Any]:
        """Clinician sign-off: resumes the paused graph with approve / reject / request_changes."""
        if decision not in SIGNOFF_DECISIONS:
            raise ValueError(f"decision must be one of {sorted(SIGNOFF_DECISIONS)}")
        pending = await self.pending_signoff(session_id)
        if pending is None:
            raise ValueError(f"session {session_id!r} has no draft awaiting clinician sign-off")
        cmd = Command(resume={"decision": decision, "clinician": clinician, "note": note})
        return await self._drive(session_id, pending["patient_id"], mask_phi(f"signoff:{decision} {note}".strip()),
                                 "copilot_signoff", cmd)

    async def _drive(self, session_id: str, patient_id: str, input_masked: str, span_name: str,
                     graph_input: Any) -> dict[str, Any]:
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        t_run, t_ses = RUN_ID.set(run_id), SESSION_ID.set(session_id)
        calls: list[dict[str, Any]] = []  # every model request of this run, with its provider (src.llm.ModelChain)
        t_calls = MODEL_CALLS.set(calls)
        started = time.perf_counter()
        trace_id = None
        try:
            from openinference.instrumentation import using_attributes
            from opentelemetry import trace

            tracer = trace.get_tracer("discharge-copilot")
            with using_attributes(session_id=session_id, metadata={"run_id": run_id}),                     tracer.start_as_current_span(span_name, attributes={
                        "openinference.span.kind": "AGENT", "run_id": run_id, "session.id": session_id,
                        "patient_ref": patient_ref(patient_id), "input.value": input_masked}) as span:
                ctx = span.get_span_context()
                trace_id = format(ctx.trace_id, "032x") if ctx.is_valid else None
                state = await self.graph.ainvoke(graph_input, run_config(session_id, run_id))
                # interrupted at clinician_signoff: the draft exists (rendered by output_guard) but is not final
                interrupts = state.get("__interrupt__") or []
                signoff = interrupts[0].value if interrupts else None
                status = "AWAITING_CLINICIAN_SIGNOFF" if signoff is not None else state.get("status")
                span.set_attribute("output.value", state.get("final_response", "")[:4000])
                span.set_attribute("copilot.status", status or "")
                span.set_attribute("copilot.intent", str(state.get("intent")))
                providers = provider_summary(calls)
                span.set_attribute("copilot.providers", ",".join(sorted(providers["by_provider"])))
                span.set_attribute("copilot.groq_calls", providers["groq_calls"])
            return {
                "run_id": run_id, "trace_id": trace_id, "session_id": session_id,
                "patient_ref": patient_ref(patient_id), "input": input_masked,
                "intent": state.get("intent"), "status": status,
                # status the graph set before the pause (gate / escalate / output guard); what eval checks compare
                "draft_status": state.get("status"), "awaiting_clinician_signoff": signoff is not None,
                "signoff_request": signoff, "risk_level": state.get("risk_level"),
                "requires_clinician_approval": state.get("requires_clinician_approval"),
                "approval_reasons": state.get("approval_reasons", []),
                "degradations": state.get("degradations", []),
                # provider of every model request in this run; groq_involved marks the Groq quota fallback
                "model_calls": calls, "providers": providers,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "response": state.get("final_response", ""),
                "_state": state,
            }
        finally:
            RUN_ID.reset(t_run)
            SESSION_ID.reset(t_ses)
            MODEL_CALLS.reset(t_calls)
