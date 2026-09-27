"""Custom MCP server for the Discharge-Planning Copilot (stdio transport).

Tools
  get_patient_record(patient_id)            de-identified synthetic discharge record
  get_prior_admissions(patient_id)          prior-admission history (readmission-risk input)
  find_followup_slots(specialty, ...)       open follow-up clinic slots in a date window
Resources
  resource://hospital/discharge_checklist   discharge checklist used by the plan-assembly agent
  resource://hospital/specialties           follow-up specialties the scheduler understands

Demographics never leave this server (direct identifiers are stripped at the boundary).
Every tools/call and resources/read is appended to logs/mcp_transcript.jsonl (server side).
Run standalone:  python mcp_server/server.py   (speaks MCP over stdin/stdout)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from src import config  # noqa: E402
from src.guardrails.guards import deidentify_record, mask_obj  # noqa: E402
from src.observability.run_context import append_jsonl, now_iso  # noqa: E402

mcp = FastMCP("discharge-copilot-mcp")

SPECIALTIES = ["primary_care", "cardiology", "heart_failure_clinic", "pulmonology", "endocrinology",
               "diabetes_educator", "orthopedics", "wound_check", "cardiac_rehab", "pulmonary_rehab"]

CHECKLIST = """# Discharge Checklist (synthetic, POL-GEN-01 companion)
1. Discharge diagnoses listed (primary first).
2. Medication list reproduced exactly as ordered (POL-MED-08) - no changes, additions or omissions.
3. High-risk medications highlighted with education points (anticoagulant, insulin, opioid, antiplatelet).
4. Readmission-risk tier computed (LACE, POL-RISK-02) with contributing factors stated.
5. Follow-up appointments proposed inside the policy windows, each citing its policy section.
6. Home-care / equipment / transportation needs addressed (POL-HOME-07).
7. Red-flag return precautions included.
8. Transitional-care phone call scheduled when required by risk tier or partial teach-back.
9. Plan marked DRAFT until a licensed clinician approves it; HIGH risk requires explicit approval.
"""


def _transcript(method: str, params: dict, result, started: float, status: str = "ok") -> None:
    append_jsonl(config.MCP_TRANSCRIPT, {
        "timestamp": now_iso(), "side": "server", "method": method, "params": mask_obj(params),
        "status": status, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "result_preview": json.dumps(mask_obj(result), default=str)[:400],
    })


def _load(patient_id: str) -> dict | None:
    path = config.PATIENTS_DIR / f"patient_{patient_id.upper()}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@mcp.tool()
def get_patient_record(patient_id: str) -> dict:
    """Return the de-identified synthetic discharge record (diagnoses, discharge medication orders,
    vitals, labs, utilization, social factors, untrusted nursing notes) for one patient ID like 'PT-1001'."""
    t0 = time.perf_counter()
    rec = _load(patient_id)
    if rec is None:
        out = {"error": "patient_not_found", "patient_id": patient_id}
        _transcript("tools/call:get_patient_record", {"patient_id": patient_id}, out, t0, "error")
        return out
    out = deidentify_record(rec)
    out.pop("prior_admissions", None)  # served by its own tool
    _transcript("tools/call:get_patient_record", {"patient_id": patient_id}, {"keys": sorted(out)}, t0)
    return out


@mcp.tool()
def get_prior_admissions(patient_id: str) -> dict:
    """Return prior admissions in the last 12 months for readmission-risk assessment."""
    t0 = time.perf_counter()
    rec = _load(patient_id)
    if rec is None:
        out = {"error": "patient_not_found", "patient_id": patient_id}
        _transcript("tools/call:get_prior_admissions", {"patient_id": patient_id}, out, t0, "error")
        return out
    out = {"patient_id": patient_id, "prior_admissions": rec.get("prior_admissions", []),
           "count_12mo": len(rec.get("prior_admissions", []))}
    _transcript("tools/call:get_prior_admissions", {"patient_id": patient_id}, out, t0)
    return out


@mcp.tool()
def find_followup_slots(specialty: str, earliest_date: str, latest_date: str,
                        preferred_weekday: str | None = None, telehealth_only: bool = False,
                        limit: int = 3) -> dict:
    """Find open follow-up clinic slots for `specialty` between earliest_date and latest_date
    (ISO dates, inclusive). Optionally prefer a weekday (e.g. 'Tuesday') or telehealth-capable slots."""
    t0 = time.perf_counter()
    params = {"specialty": specialty, "earliest_date": earliest_date, "latest_date": latest_date,
              "preferred_weekday": preferred_weekday, "telehealth_only": telehealth_only, "limit": limit}
    if specialty not in SPECIALTIES:
        out = {"error": "unknown_specialty", "valid_specialties": SPECIALTIES}
        _transcript("tools/call:find_followup_slots", params, out, t0, "error")
        return out
    try:
        lo, hi = date.fromisoformat(earliest_date), date.fromisoformat(latest_date)
    except ValueError:
        out = {"error": "invalid_date", "detail": "dates must be ISO format YYYY-MM-DD"}
        _transcript("tools/call:find_followup_slots", params, out, t0, "error")
        return out
    schedule = json.loads((config.DATA_DIR / "clinic_schedule.json").read_text(encoding="utf-8"))
    slots = [s for s in schedule.get(specialty, []) if lo <= date.fromisoformat(s["date"]) <= hi]
    if telehealth_only:
        slots = [s for s in slots if s["telehealth_available"]]
    if preferred_weekday:
        preferred = [s for s in slots if s["weekday"].lower() == preferred_weekday.lower()]
        slots = preferred + [s for s in slots if s not in preferred]
    out = {"specialty": specialty, "window": [earliest_date, latest_date], "slots": slots[: max(1, min(limit, 10))],
           "total_open": len(slots)}
    _transcript("tools/call:find_followup_slots", params, {"returned": len(out["slots"])}, t0)
    return out


@mcp.resource("resource://hospital/discharge_checklist")
def discharge_checklist() -> str:
    """Hospital discharge checklist (synthetic)."""
    t0 = time.perf_counter()
    _transcript("resources/read:discharge_checklist", {}, {"chars": len(CHECKLIST)}, t0)
    return CHECKLIST


@mcp.resource("resource://hospital/specialties")
def specialties() -> str:
    """Follow-up specialties understood by find_followup_slots."""
    return json.dumps(SPECIALTIES)


if __name__ == "__main__":
    mcp.run(transport="stdio")
