"""Deterministic rendering of structured outputs into the clinician-facing answer."""
from __future__ import annotations


def render_risk(risk: dict, status: str, reasons: list[str]) -> str:
    if not risk:
        return "Readmission risk could not be computed."
    lines = [f"READMISSION RISK: {risk['risk_level']} (LACE {risk['lace_score']}; "
             f"L={risk['lace_components']['L']} A={risk['lace_components']['A']} "
             f"C={risk['lace_components']['C']} E={risk['lace_components']['E']}) "
             f"[{risk.get('policy_citation')}]",
             "Factors behind the flag:"]
    lines += [f"  - {f['factor']}: {f['evidence']} (source: {f['source']})" for f in risk["factors"]]
    lines.append(f"Rationale: {risk['rationale']}")
    if status:
        lines.append(f"Status: {status}" + (f" - reasons: {', '.join(reasons)}" if reasons else ""))
    return "\n".join(lines)


def _schedule_lines(plan: dict) -> list[str]:
    out = []
    for a in plan["follow_up_appointments"]:
        slot = a["proposed_slot"] or "UNSCHEDULED"
        out.append(f"  - {a['specialty']}: {slot} ({a['target_window']}){' [telehealth]' if a['telehealth'] else ''}"
                   f" - {a['note']} [{a['policy_citation']}]")
    if plan.get("transitional_care_call"):
        out.append(f"  - {plan['transitional_care_call']} [readmission_risk_protocols.md#actions-by-tier]")
    return out


def render_schedule(plan: dict, risk: dict | None, status: str, reasons: list[str]) -> str:
    """Answer to a follow-up-schedule request: the schedule and what drives it, not the whole plan
    (relevance fix - see docs/failure-analysis.md F5)."""
    out = [f"DRAFT FOLLOW-UP SCHEDULE - {plan['patient_ref']}  |  STATUS: {plan.get('status') or status}"]
    if reasons:
        out.append("Clinician approval required: " + ", ".join(reasons))
    if risk:
        out.append(f"Readmission risk: {risk['risk_level']} (LACE {risk['lace_score']}) sets the follow-up windows "
                   f"[{risk.get('policy_citation')}]")
    out.append("")
    out.append("Proposed appointments (pending confirmation):")
    out += _schedule_lines(plan)
    if plan["home_care_and_logistics"]:
        out.append("")
        out.append("Logistics:")
        out += [f"  - {h}" for h in plan["home_care_and_logistics"]]
    out.append("")
    out.append("Policy citations: " + ", ".join(c for c in plan["policy_citations"] if "reconciliation" not in c))
    out.append("(Medications are unchanged from the orders; ask for the full discharge plan to see them.)")
    return "\n".join(out)


def render_plan(plan: dict, risk: dict | None, status: str, reasons: list[str]) -> str:
    out = [f"DRAFT DISCHARGE PLAN - {plan['patient_ref']}  |  STATUS: {plan.get('status') or status}"]
    if reasons:
        out.append("Clinician approval required: " + ", ".join(reasons))
    out.append(f"Primary diagnosis: {plan['primary_diagnosis']}")
    if plan["secondary_diagnoses"]:
        out.append("Secondary: " + "; ".join(plan["secondary_diagnoses"]))
    if risk:
        out.append("")
        out.append(render_risk(risk, "", []))
    out.append("")
    out.append("Medications (reproduced exactly as ordered - not altered "
               "[medication_reconciliation_policy.md#reconciliation-rule]):")
    for m in plan["medications"]:
        flag = "  <HIGH-RISK: education required>" if m["high_risk"] else ""
        out.append(f"  - {m['drug']} {m['dose']} {m['route']} {m['frequency']} ({m['status']}){flag}")
    out.append(f"Order integrity check: {'PASS' if plan['orders_unchanged'] else 'FAIL'}")
    out.append("")
    out.append("Follow-up schedule (proposed, pending confirmation):")
    out += _schedule_lines(plan)
    if plan["home_care_and_logistics"]:
        out.append("")
        out.append("Home care & logistics:")
        out += [f"  - {h}" for h in plan["home_care_and_logistics"]]
    if plan["patient_instructions"]:
        out.append("")
        out.append("Patient instructions:")
        out += [f"  - {i}" for i in plan["patient_instructions"]]
    if plan["red_flags"]:
        out.append("")
        out.append("Return precautions (red flags):")
        out += [f"  - {r}" for r in plan["red_flags"]]
    out.append("")
    out.append("Policy citations: " + ", ".join(plan["policy_citations"]))
    return "\n".join(out)
