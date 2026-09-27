"""Input/output guardrails wired into the graph's I/O nodes (AC-06, AC-10, NFR-03, NFR-05).

Layers:
  * PHI masking      - registry of every synthetic identifier + regex patterns + Presidio NER (spaCy
                       en_core_web_sm; degrades to registry + regex and reports "ner": "unavailable").
  * Input guard      - prompt-injection detection (regex, then the LLM Guard PromptInjection classifier;
                       regex-only with "injection_classifier": "unavailable" if it cannot load),
                       cross-patient access, raw-PHI requests.
  * Output guard     - PHI leakage (incl. NER), other-patient leakage, clinical-order integrity, draft labelling.
  * patient_ref()    - HMAC pseudonym used instead of patient IDs in logs / traces.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

from src import config

log = logging.getLogger(__name__)
PATIENT_ID_RE = re.compile(r"\bPT-\d{4}\b", re.I)
PHI_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("MRN", re.compile(r"\bMRN[-\s:]?\d{6,10}\b", re.I)),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?:\+?1[-\s.]?)?\(?\d{3}\)?[-\s.]\d{3,4}[-\s.]?\d{4}\b|\+1-555-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("DOB", re.compile(r"\b(?:DOB|date of birth)[:\s]*\d{4}-\d{2}-\d{2}\b", re.I)),
]

INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+instructions",
    r"disregard\s+(?:all\s+)?(?:the\s+|any\s+)?(?:previous|prior|earlier|above|your)\s+"
    r"(?:instructions|rules|guidelines|guidance|directions)",
    r"forget\s+(?:all\s+)?(?:about\s+)?(?:(?:your|the|any)\s+)?(?:(?:previous|prior|earlier)\s+)?"
    r"(?:instructions|rules|guidelines|guidance)",
    r"new\s+(?:policy|instructions?|rules?|directive)\s+from\s+(?:the\s+)?(?:admin|administrator|system|developer|it)\b",
    r"approvals?\s+(?:is|are)\s+no\s+longer\s+(?:needed|required|necessary)",
    r"pretend\s+(?:that\s+)?(?:the\s+)?(?:risk|patient|plan|approval|tier|lace)",
    r"(?:finali[sz]e|approve|sign(?:\s|-)?off|release|send|submit|discharge)\b[^.?!\n]{0,30}?\bwithout\s+"
    r"(?:\w+\s+){0,2}(?:approval|sign-?off|review)",
    r"you\s+are\s+now\s+(?:in\s+)?(?:admin|developer|god|dan)\b",
    r"\badmin\s+mode\b",
    r"\bjailbreak\b",
    r"(?:reveal|print|show|repeat)\s+(?:your|the)\s+(?:system\s+)?prompt",
    r"skip\s+(?:the\s+)?(?:clinician\s+)?approval",
    r"mark\s+(?:this|all|every)\s+(?:patient|patients|case)s?\s+(?:as\s+)?low",
    r"override\s+(?:the\s+)?(?:guardrails?|safety|policy)",
    r"act\s+as\s+(?:an?\s+)?(?:unrestricted|different)",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.I)

RAW_PHI_REQUEST_RE = re.compile(
    r"\b(?:show|give|print|list|tell|what(?:'s| is))\b[^.?!]{0,40}\b(?:name|mrn|medical record number|"
    r"date of birth|dob|address|phone|ssn|social security)\b",
    re.I,
)
# loose references to another patient: "PT 1002", "PT1002", "patient 1002", "patient #1002", "bed 4 patient"
# loose references to a patient ID: "PT 1002", "PT1002", "pt_1002", "PT#1002", "PT - 1002", "patient 1002",
# "patient #1003", "patient no. 1002", "patient ID: 1002" (not dates such as "patient 2026-09-22"). Group 1 = PT form,
# group 2 = "patient" form. Also matches canonical PT-####. Used for cross-patient detection, for masking
# (-> the same pref_ pseudonym as PT-####) and by scripts/scan_evidence.py.
LOOSE_PATIENT_ID_RE = re.compile(
    r"\bPT\s*[-–—_#:.]?\s*(\d{4})\b(?!-\d)"
    r"|\bpatient\s*(?:[-#:]\s*|(?:no\.?|num(?:ber)?|id)\s*[-#:]?\s*)?(\d{4})\b(?!-\d)", re.I)
BED_REFERENCE_RE = re.compile(r"\bbed\s*#?\s*\d+[a-z]?(?:'s)?\s+patient\b|\bpatient\s+(?:in|on|at)\s+bed\s*#?\s*\d+", re.I)
BULK_ACCESS_RE = re.compile(r"\b(?:all|every|other|another)\s+patients?\b|\blist\s+(?:the\s+)?patients\b", re.I)


# ----------------------------------------------------------------------------- identifiers
def patient_ref(patient_id: str | None) -> str:
    """Stable pseudonym (HMAC-SHA256) so logs can correlate a case without storing the ID."""
    if not patient_id:
        return "pref_none"
    digest = hmac.new(config.PHI_HASH_SALT.encode(), patient_id.upper().encode(), hashlib.sha256).hexdigest()
    return f"pref_{digest[:10]}"


@lru_cache(maxsize=1)
def _identifier_registry() -> list[tuple[str, str]]:
    """(literal, label) pairs for every direct identifier in the synthetic dataset."""
    pairs: list[tuple[str, str]] = []
    for path in sorted(config.PATIENTS_DIR.glob("patient_*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        demo = rec.get("demographics", {})
        name = demo.get("name", "")
        if name:
            pairs.append((name, "NAME"))
            pairs.extend((part, "NAME") for part in name.split() if len(part) > 3)
        for key, label in (("mrn", "MRN"), ("dob", "DOB"), ("phone", "PHONE"), ("address", "ADDRESS")):
            if demo.get(key):
                pairs.append((demo[key], label))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)  # longest first
    return pairs


def _presidio():
    """Optional Presidio NER layer (catches names/phones not in the registry). None if unavailable."""
    if not hasattr(_presidio, "_engine"):
        _presidio._engine = None
        try:
            import spacy  # noqa: F401
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            for model in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
                try:
                    spacy.load(model)
                except OSError:
                    continue
                provider = NlpEngineProvider(nlp_configuration={
                    "nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": model}]})
                _presidio._engine = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
                _presidio._model = model
                break
        except Exception as exc:
            log.warning("Presidio NER unavailable: %s", exc)
            _presidio._engine = None
    return _presidio._engine


def ner_status() -> str:
    """"presidio:<spaCy model>" when the NER layer is active, else "unavailable" (registry + regex only)."""
    return f"presidio:{_presidio._model}" if _presidio() is not None else "unavailable"


PRESIDIO_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "LOCATION"]
# Clinical vocabulary that NER models sometimes mislabel as PERSON/LOCATION.
_NER_ALLOW = re.compile(r"^(?:LACE|COPD|NSTEMI|PCI|PCP|DAPT|HFrEF|ARNI|SpO2|ED|SC|PO|IM|SL|BP|HR|"
                        r"Furosemide|Metoprolol|Apixaban|Ticagrelor|Tiotropium|Albuterol|Glucagon)$", re.I)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
_CLINICAL_RECORD_KEYS = ("admission", "discharge_diagnoses", "comorbidities", "discharge_medication_orders", "labs",
                         "functional_status")


@lru_cache(maxsize=1)
def _domain_vocab() -> frozenset[str]:
    """Words of the copilot's own clinical vocabulary: policy corpus, LACE rules, render templates and the clinical
    (never demographic / free-text) fields of the records. NER spans made only of these words are clinical terms
    ("Charlson", "Polypharmacy", "Orthopedic Surgery"), not people or places. Patient name parts are excluded."""
    texts = [p.read_text(encoding="utf-8") for p in sorted(config.POLICY_DIR.glob("*.md"))]
    texts += [(config.ROOT / "src" / f).read_text(encoding="utf-8") for f in ("clinical_rules.py", "render.py")]
    for path in sorted(config.PATIENTS_DIR.glob("patient_*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        texts.append(json.dumps([rec.get(k) for k in _CLINICAL_RECORD_KEYS]))
    names = {w.lower() for literal, label in _identifier_registry() if label == "NAME" for w in _WORD_RE.findall(literal)}
    return frozenset(w.lower() for t in texts for w in _WORD_RE.findall(t)) - names


def _is_domain_term(span: str) -> bool:
    words = [w.lower() for w in _WORD_RE.findall(span)]
    return bool(words) and all(w in _domain_vocab() for w in words)


def _loose_ref(m: re.Match) -> str:
    """"PT 1002" -> pref_x, "patient #1002" -> "patient pref_x": the same pseudonym as PT-1002."""
    ref = patient_ref(f"PT-{m.group(1) or m.group(2)}")
    return ref if m.group(1) else f"patient {ref}"


def find_patient_ids(text: str) -> list[str]:
    """Every patient-ID reference in text, canonical or loose (evidence scan + tests)."""
    return [m.group(0) for m in LOOSE_PATIENT_ID_RE.finditer(text)]


def mask_phi(text: Any, *, use_ner: bool = False, keep_patient_ids: bool = False,
             hits: list[dict] | None = None) -> str:
    """Mask synthetic PHI. Registry + regex always run; Presidio NER optionally (slower). If `hits` is given,
    one {"layer": registry|regex|presidio, "label"} entry is appended per masked identifier."""
    if text is None:
        return ""
    s = text if isinstance(text, str) else json.dumps(text, default=str)
    for literal, label in _identifier_registry():
        if literal and literal in s:
            if hits is not None:
                hits.extend({"layer": "registry", "label": label} for _ in range(s.count(literal)))
            s = s.replace(literal, f"[{label}]")
    for label, rx in PHI_PATTERNS:
        if hits is not None:
            hits.extend({"layer": "regex", "label": label} for _ in rx.finditer(s))
        s = rx.sub(f"[{label}]", s)
    if not keep_patient_ids:
        s = PATIENT_ID_RE.sub(lambda m: patient_ref(m.group(0)), s)
        s = LOOSE_PATIENT_ID_RE.sub(_loose_ref, s)
    if use_ner and (engine := _presidio()) is not None:
        try:
            results = engine.analyze(text=s, language="en", entities=PRESIDIO_ENTITIES, score_threshold=0.6)
            for r in sorted(results, key=lambda r: r.start, reverse=True):
                span = s[r.start : r.end]
                if span.startswith("[") or span.startswith("pref_") or _NER_ALLOW.match(span) or _is_domain_term(span):
                    continue
                if hits is not None:
                    hits.append({"layer": "presidio", "label": r.entity_type})
                s = s[: r.start] + f"[{r.entity_type}]" + s[r.end :]
        except Exception:
            pass
    return s


def mask_obj(obj: Any) -> Any:
    """Recursively mask PHI inside dict/list structures (used for log args/results)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("patient_id",) and isinstance(v, str):
                out["patient_ref"] = patient_ref(v)
            elif k == "demographics":
                out[k] = "[REDACTED_DEMOGRAPHICS]"
            else:
                out[k] = mask_obj(v)
        return out
    if isinstance(obj, list):
        return [mask_obj(v) for v in obj]
    if isinstance(obj, str):
        return mask_phi(obj)
    return obj


def find_phi(text: str) -> list[str]:
    """Return labels of any PHI still present (used by output guard + tests)."""
    hits = [label for literal, label in _identifier_registry() if literal and literal in text]
    hits += [label for label, rx in PHI_PATTERNS if rx.search(text)]
    return sorted(set(hits))


def deidentify_record(record: dict) -> dict:
    """Strip direct identifiers from a patient record before it reaches any model or tool output."""
    rec = {k: v for k, v in record.items() if k != "demographics"}
    rec["patient_ref"] = patient_ref(record.get("patient_id"))
    return rec


# ----------------------------------------------------------------------------- input guard
@dataclass
class GuardResult:
    allowed: bool
    action: str  # allow | sanitize | block
    reasons: list[str] = field(default_factory=list)
    sanitized_text: str = ""
    details: dict = field(default_factory=dict)  # which layers ran (ner, injection_classifier); goes to the audit

    def as_dict(self) -> dict:
        return {"allowed": self.allowed, "action": self.action, "reasons": self.reasons, "details": self.details}


def _injection_scanner():
    """LLM Guard PromptInjection scanner (DeBERTa-v3 classifier), loaded once. None if disabled or unavailable."""
    if not hasattr(_injection_scanner, "_scanner"):
        _injection_scanner._scanner = None
        if config.INJECTION_CLASSIFIER == "llm-guard":
            try:
                from llm_guard.input_scanners import PromptInjection
                from llm_guard.input_scanners.prompt_injection import MatchType

                _injection_scanner._scanner = PromptInjection(threshold=config.INJECTION_CLASSIFIER_THRESHOLD,
                                                              match_type=MatchType.FULL)
            except Exception as exc:  # package or model missing (e.g. offline without a cached model)
                log.warning("LLM Guard PromptInjection unavailable, regex-only injection check: %s", exc)
    return _injection_scanner._scanner


def classify_injection(text: str) -> dict:
    """Second injection layer. Returns {"injection_classifier": "llm-guard" | "disabled" | "unavailable",
    "flagged": bool, "risk_score": float | None}; never raises (falls back to regex-only)."""
    if config.INJECTION_CLASSIFIER != "llm-guard":
        return {"injection_classifier": "disabled", "flagged": False, "risk_score": None}
    scanner = _injection_scanner()
    if scanner is None:
        return {"injection_classifier": "unavailable", "flagged": False, "risk_score": None}
    try:
        _, is_valid, risk = scanner.scan(text)
    except Exception as exc:
        log.warning("LLM Guard PromptInjection scan failed: %s", exc)
        return {"injection_classifier": "unavailable", "flagged": False, "risk_score": None}
    return {"injection_classifier": "llm-guard", "flagged": not is_valid, "risk_score": round(float(risk), 4)}


def _mentioned_patient_ids(message: str) -> set[str]:
    ids = {m.upper() for m in PATIENT_ID_RE.findall(message)}
    ids |= {f"PT-{a or b}" for a, b in LOOSE_PATIENT_ID_RE.findall(message)}
    return ids


def input_guard(message: str, active_patient_id: str | None) -> GuardResult:
    reasons: list[str] = []
    details: dict[str, Any] = {}
    if INJECTION_RE.search(message):
        reasons.append("prompt_injection_detected")
    mentioned = _mentioned_patient_ids(message)
    if (active_patient_id and mentioned - {active_patient_id.upper()}) or BED_REFERENCE_RE.search(message):
        reasons.append("cross_patient_access_attempt")
    if BULK_ACCESS_RE.search(message):
        reasons.append("bulk_or_other_patient_access_attempt")
    if RAW_PHI_REQUEST_RE.search(message):
        reasons.append("raw_phi_request")
    if len(message) > 4000:
        reasons.append("input_too_long")
    if reasons:
        details["injection_classifier"] = "skipped_regex_blocked"
    else:  # layer 2 only runs once every regex check has passed
        c = classify_injection(message)
        details["injection_classifier"] = c["injection_classifier"]
        if c["risk_score"] is not None:
            details["injection_classifier_risk"] = c["risk_score"]
        if c["flagged"]:
            reasons.append("prompt_injection_classifier")

    sanitized = mask_phi(message, keep_patient_ids=True)
    if reasons:
        return GuardResult(False, "block", reasons, sanitized, details)
    if sanitized != message:
        return GuardResult(True, "sanitize", ["phi_masked_in_input"], sanitized, details)
    return GuardResult(True, "allow", [], sanitized, details)


# ----------------------------------------------------------------------------- output guard
def orders_fingerprint(meds: Iterable[dict]) -> str:
    norm = [
        "|".join(str(m.get(k, "")).strip().lower() for k in ("drug", "dose", "route", "frequency"))
        for m in meds
    ]
    return hashlib.sha256("\n".join(sorted(norm)).encode()).hexdigest()


def verify_orders_unchanged(plan_meds: list[dict], source_orders: list[dict]) -> bool:
    return orders_fingerprint(plan_meds) == orders_fingerprint(source_orders)


def output_guard(text: str, *, active_patient_id: str | None, plan: dict | None = None,
                 source_orders: list[dict] | None = None) -> GuardResult:
    reasons: list[str] = []
    other_ids = {m.upper() for m in PATIENT_ID_RE.findall(text)} - {(active_patient_id or "").upper()}
    if other_ids:
        reasons.append("other_patient_reference_in_output")
    if find_phi(text):
        reasons.append("phi_in_output_masked")
    if plan is not None and source_orders is not None:
        if not verify_orders_unchanged(plan.get("medications", []), source_orders):
            reasons.append("clinical_orders_altered")
    if re.search(INJECTION_RE, text):
        reasons.append("injected_instruction_echoed")

    hits: list[dict] = []
    masked = mask_phi(text, use_ner=True, hits=hits)
    ner_hits = [h["label"] for h in hits if h["layer"] == "presidio"]
    if ner_hits:
        reasons.append("phi_ner_masked")
    details = {"ner": ner_status(), "ner_entities": ner_hits}
    if "clinical_orders_altered" in reasons or "other_patient_reference_in_output" in reasons:
        return GuardResult(False, "block", reasons, masked, details)
    if reasons:
        return GuardResult(True, "sanitize", reasons, masked, details)
    return GuardResult(True, "allow", [], masked, details)
