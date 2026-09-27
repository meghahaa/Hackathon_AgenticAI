"""Deterministic, auditable clinical-logistics rules taken from the synthetic policy corpus.

The LLM explains and drafts; these rules compute the numbers (LACE tier, follow-up windows) so the
risk tier and follow-up windows are reproducible and testable. No clinical orders are ever changed.
"""
from __future__ import annotations

import re
from typing import Any

from src.schemas import FollowUpRequirement, RiskFactor

HIGH_RISK_DRUG_RE = re.compile(
    r"apixaban|rivaroxaban|warfarin|heparin|enoxaparin|insulin|oxycodone|morphine|hydromorphone|"
    r"ticagrelor|clopidogrel|prasugrel", re.I)

TIER_ORDER = ["LOW", "MEDIUM", "HIGH"]


def lace_points_los(days: float) -> int:
    if days < 1:
        return 0
    if days < 4:
        return int(days)
    if days <= 6:
        return 4
    if days <= 13:
        return 5
    return 7


def compute_lace(record: dict, prior_admissions: list[dict] | None = None,
                 session_facts: dict | None = None) -> dict[str, Any]:
    """LACE index per readmission_risk_protocols.md#lace-index-scoring plus modifiers/factors."""
    session_facts = session_facts or {}
    adm = record.get("admission", {})
    L = lace_points_los(float(adm.get("length_of_stay_days", 0)))
    A = 3 if adm.get("admission_type") == "emergent" else 0
    charlson = sum(int(c.get("charlson_points", 0)) for c in record.get("comorbidities", []))
    C = charlson if charlson <= 3 else 5
    E = min(int(record.get("utilization", {}).get("ed_visits_6mo", 0)), 4)
    score = L + A + C + E
    base = "LOW" if score <= 4 else ("MEDIUM" if score <= 9 else "HIGH")

    factors: list[RiskFactor] = [
        RiskFactor(factor=f"LACE {score} ({base})", evidence=f"L={L} A={A} C={C} E={E}",
                   source="readmission_risk_protocols.md#lace-index-scoring")
    ]
    if L >= 4:
        factors.append(RiskFactor(factor="Prolonged length of stay",
                                  evidence=f"{adm.get('length_of_stay_days')} days", source="admission"))
    if A:
        factors.append(RiskFactor(factor="Emergent admission", evidence="admission_type=emergent", source="admission"))
    if C:
        factors.append(RiskFactor(factor="Comorbidity burden", evidence=f"Charlson points={charlson}",
                                  source="comorbidities"))
    if E:
        factors.append(RiskFactor(factor="Recent ED utilisation", evidence=f"{E} ED visits in 6 months",
                                  source="utilization"))

    social = dict(record.get("social", {}))
    if "has_transportation" in session_facts:
        social["has_transportation"] = session_facts["has_transportation"]
    if session_facts.get("transport_by"):
        social["has_transportation"] = True
    lives_alone_no_cg = bool(social.get("lives_alone")) and not social.get("caregiver")
    no_transport = social.get("has_transportation") is False
    modifier = lives_alone_no_cg or no_transport
    tier = base
    if modifier:
        tier = TIER_ORDER[min(TIER_ORDER.index(base) + 1, 2)]
        why = []
        if lives_alone_no_cg:
            why.append("lives alone with no caregiver")
        if no_transport:
            why.append("no transportation to follow-up")
        factors.append(RiskFactor(factor="Social risk modifier", evidence="; ".join(why),
                                  source="readmission_risk_protocols.md#social-risk-modifier"))

    prior = prior_admissions if prior_admissions is not None else record.get("prior_admissions", [])
    if len(prior) >= 2:
        factors.append(RiskFactor(factor="Frequent admissions", evidence=f"{len(prior)} admissions in 12 months",
                                  source="readmission_risk_protocols.md#additional-contributing-factors"))
    meds = record.get("discharge_medication_orders", [])
    if len(meds) >= 5:
        factors.append(RiskFactor(factor="Polypharmacy", evidence=f"{len(meds)} discharge medications",
                                  source="readmission_risk_protocols.md#additional-contributing-factors"))
    hr = [m["drug"] for m in meds if HIGH_RISK_DRUG_RE.search(m["drug"])]
    if hr:
        factors.append(RiskFactor(factor="High-risk medication", evidence=", ".join(hr),
                                  source="readmission_risk_protocols.md#additional-contributing-factors"))
    spo2 = record.get("vitals_at_discharge", {}).get("spo2")
    if spo2 is not None and spo2 <= 92:
        factors.append(RiskFactor(factor="Low discharge SpO2", evidence=f"SpO2 {spo2}%",
                                  source="readmission_risk_protocols.md#additional-contributing-factors"))
    return {"lace_score": score, "lace_components": {"L": L, "A": A, "C": C, "E": E}, "base_tier": base,
            "social_modifier_applied": modifier, "risk_level": tier, "factors": factors}


# section anchor -> (specialty for the scheduler, requirement text, min, max days)
SECTION_RULES: dict[str, list[tuple[str, str, int, int]]] = {
    "cardiology_followup_policy.md#heart-failure": [
        ("heart_failure_clinic", "Heart-failure clinic visit", 1, 7)],
    "cardiology_followup_policy.md#acute-myocardial-infarction-and-pci": [
        ("cardiology", "Cardiology visit after MI/PCI", 1, 14),
        ("cardiac_rehab", "Outpatient cardiac rehabilitation enrolment", 7, 30)],
    "pulmonary_followup_policy.md#copd-exacerbation": [
        ("pulmonology", "Pulmonology visit with inhaler technique re-check", 1, 7),
        ("pulmonary_rehab", "Pulmonary rehabilitation referral", 7, 28)],
    "pulmonary_followup_policy.md#community-acquired-pneumonia": [
        ("primary_care", "PCP visit after pneumonia", 1, 14)],
    "endocrine_followup_policy.md#diabetic-ketoacidosis": [
        ("endocrinology", "Endocrinology / diabetes clinic visit", 1, 7),
        ("diabetes_educator", "Diabetes-educator session", 1, 14)],
    "orthopedic_followup_policy.md#total-joint-arthroplasty": [
        ("wound_check", "Wound check", 10, 14), ("orthopedics", "Surgeon follow-up at 6 weeks", 40, 44)],
    "general_discharge_policy.md#primary-care-follow-up": [
        ("primary_care", "Primary care follow-up", 1, 14)],
}

DIAGNOSIS_QUERIES = [  # (diagnosis matcher, retrieval query, focus term, policy section that must be retrieved)
    (re.compile(r"heart failure|HFrEF", re.I), "heart failure follow-up after discharge", "heart failure",
     "cardiology_followup_policy.md#heart-failure"),
    (re.compile(r"myocardial infarction|NSTEMI|STEMI|PCI", re.I), "follow-up after myocardial infarction PCI",
     "myocardial infarction", "cardiology_followup_policy.md#acute-myocardial-infarction-and-pci"),
    (re.compile(r"COPD", re.I), "COPD exacerbation follow-up", "COPD",
     "pulmonary_followup_policy.md#copd-exacerbation"),
    (re.compile(r"pneumonia", re.I), "community-acquired pneumonia follow-up", "pneumonia",
     "pulmonary_followup_policy.md#community-acquired-pneumonia"),
    (re.compile(r"ketoacidosis|DKA", re.I), "diabetic ketoacidosis follow-up", "ketoacidosis",
     "endocrine_followup_policy.md#diabetic-ketoacidosis"),
    (re.compile(r"arthroplasty", re.I), "total joint arthroplasty post-operative follow-up", "arthroplasty",
     "orthopedic_followup_policy.md#total-joint-arthroplasty"),
]


def rule_requirements(citations: set[str], risk_level: str | None) -> list[FollowUpRequirement]:
    """Deterministic follow-up requirements for the retrieved (cited) policy sections."""
    reqs: list[FollowUpRequirement] = []
    for cit in sorted(citations):
        for specialty, text, lo, hi in SECTION_RULES.get(cit, []):
            if specialty == "primary_care" and risk_level == "HIGH":
                hi = min(hi, 7)
            reqs.append(FollowUpRequirement(requirement=text, specialty=specialty, within_days_min=lo,
                                            within_days_max=hi, citation=cit))
    return reqs
