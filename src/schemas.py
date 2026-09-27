"""Structured outputs validated at every node boundary."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Intent = Literal[
    "full_discharge_plan",
    "followup_schedule",
    "risk_assessment",
    "policy_question",
    "clinical_order_change",
    "recall_prior",
    "ambiguous",
    "out_of_scope",
]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class IntentDecision(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=400)


class FollowUpRequirement(BaseModel):
    requirement: str
    specialty: str
    within_days_min: int = Field(ge=0, le=90)
    within_days_max: int = Field(ge=1, le=90)
    citation: str

    @field_validator("within_days_max")
    @classmethod
    def _window(cls, v, info):
        lo = info.data.get("within_days_min", 0)
        return max(v, lo)


class PolicyFindings(BaseModel):
    requirements: list[FollowUpRequirement] = Field(default_factory=list)
    answer: str = ""
    citations: list[str] = Field(default_factory=list)


class RiskFactor(BaseModel):
    factor: str
    evidence: str
    source: str  # record field or policy citation


class RiskAssessment(BaseModel):
    lace_score: int = Field(ge=0, le=19)
    lace_components: dict[str, int]
    base_tier: RiskLevel
    social_modifier_applied: bool
    risk_level: RiskLevel
    factors: list[RiskFactor]
    rationale: str
    policy_citation: str = "readmission_risk_protocols.md#risk-tiers"


class RiskNarrative(BaseModel):
    rationale: str = Field(max_length=900)


class MedicationLine(BaseModel):
    drug: str
    dose: str
    route: str
    frequency: str
    status: str
    high_risk: bool = False


class Appointment(BaseModel):
    specialty: str
    target_window: str
    proposed_slot: Optional[str] = None
    telehealth: bool = False
    policy_citation: str
    note: str = ""


class PlanNarrative(BaseModel):
    patient_instructions: list[str] = Field(default_factory=list, max_length=10)
    red_flags: list[str] = Field(default_factory=list, max_length=10)


class DischargePlan(BaseModel):
    patient_ref: str
    primary_diagnosis: str
    secondary_diagnoses: list[str]
    medications: list[MedicationLine]
    orders_unchanged: bool
    follow_up_appointments: list[Appointment]
    home_care_and_logistics: list[str]
    transitional_care_call: Optional[str]
    patient_instructions: list[str]
    red_flags: list[str]
    risk_level: Optional[RiskLevel]
    policy_citations: list[str]
    checklist_source: str = "resource://hospital/discharge_checklist"
    status: str = "DRAFT"
