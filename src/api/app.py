"""Optional bonus: async FastAPI streaming endpoint (Server-Sent Events).

    uvicorn src.api.app:app --port 8000
    curl -N -X POST localhost:8000/v1/discharge/stream -H "content-type: application/json" \
         -d '{"session_id":"api-1","patient_id":"PT-1001","message":"Draft the discharge plan"}'

Streams one event per graph node as it completes (PHI-masked), then a final event with the answer.
A drafted plan / schedule or an order-change request pauses the graph (status AWAITING_CLINICIAN_SIGNOFF, an
`interrupt` event); the clinician resumes it with:

    curl -X POST localhost:8000/v1/discharge/api-1/signoff -H "content-type: application/json" \
         -d '{"decision":"approve","clinician":"attending physician","note":"reviewed"}'
"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.graph import run_config
from src.guardrails.guards import mask_obj, mask_phi
from src.observability.run_context import RUN_ID, SESSION_ID
from src.observability.tracing import init_tracing
from src.runner import Copilot

STATE: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_tracing()
    async with Copilot() as copilot:
        STATE["copilot"] = copilot
        yield


app = FastAPI(title="Discharge-Planning & Follow-up Copilot", lifespan=lifespan)


class TurnRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: f"api-{uuid.uuid4().hex[:8]}")
    patient_id: str = Field(pattern=r"^PT-\d{4}$")
    message: str = Field(min_length=1, max_length=4000)


class SignoffRequest(BaseModel):
    decision: Literal["approve", "reject", "request_changes"]
    clinician: str = Field(min_length=1, max_length=200)
    note: str = Field(default="", max_length=2000)


@app.get("/healthz")
async def healthz():
    return {"ok": "copilot" in STATE}


@app.post("/v1/discharge/stream")
async def stream(req: TurnRequest):
    copilot = STATE["copilot"]
    run_id = f"run-{uuid.uuid4().hex[:12]}"

    async def events():
        t1, t2 = RUN_ID.set(run_id), SESSION_ID.set(req.session_id)
        try:
            final, status = {}, None
            async for chunk in copilot.graph.astream(
                    {"messages": [HumanMessage(req.message)], "patient_id": req.patient_id},
                    run_config(req.session_id, run_id), stream_mode="updates"):
                for node, upd in chunk.items():
                    if node == "__interrupt__":  # paused at clinician_signoff; payload is already PHI-masked
                        status = "AWAITING_CLINICIAN_SIGNOFF"
                        payload = upd[0].value if upd else {}
                        yield f"event: interrupt\ndata: {json.dumps(mask_obj(payload))}\n\n"
                        final = {**final, "status": status, "signoff_request": mask_obj(payload)}
                        continue
                    upd = upd or {}
                    summary = {k: upd[k] for k in ("intent", "next_node", "risk_level", "status",
                                                   "requires_clinician_approval") if k in upd}
                    status = upd.get("status", status)
                    yield f"event: node\ndata: {json.dumps({'node': node, **mask_obj(summary)})}\n\n"
                    if "final_response" in upd and node == "output_guard":
                        final = {"answer": upd["final_response"], "status": status}
            yield f"event: final\ndata: {json.dumps({'run_id': run_id, **final})}\n\n"
        finally:
            RUN_ID.reset(t1)
            SESSION_ID.reset(t2)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/v1/discharge/{session_id}/signoff")
async def signoff(session_id: str, req: SignoffRequest):
    """Clinician sign-off: resumes a session paused at clinician_signoff (approve / reject / request_changes)."""
    copilot = STATE["copilot"]
    try:
        res = await copilot.resume(session_id, req.decision, req.clinician, req.note)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run_id": res["run_id"], "trace_id": res["trace_id"], "session_id": session_id,
            "patient_ref": res["patient_ref"], "status": res["status"], "answer": mask_phi(res["response"])}
