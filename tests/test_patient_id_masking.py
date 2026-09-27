"""Loose patient-ID forms ("PT 1002", "PT1002", "patient #1003", ...) are masked to the same pref_ pseudonym as the
canonical PT-#### ID everywhere text reaches evidence, and scripts/scan_evidence.py flags any that slip through.
Offline: offline stub LLM, local MCP subprocess, in-memory spans, logs under tmp_path (conftest)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src import config
from src.guardrails.guards import find_patient_ids, mask_obj, mask_phi, patient_ref
from src.observability import tracing

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("scan_evidence", ROOT / "scripts" / "scan_evidence.py")
scan_evidence = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan_evidence)

REF_1002, REF_1003 = patient_ref("PT-1002"), patient_ref("PT-1003")


@pytest.mark.parametrize("text, expected", [
    ("plan for PT 1002 please", f"plan for {REF_1002} please"),
    ("plan for PT1002 please", f"plan for {REF_1002} please"),
    ("plan for pt_1002 please", f"plan for {REF_1002} please"),
    ("plan for PT#1002 please", f"plan for {REF_1002} please"),
    ("plan for PT - 1002 please", f"plan for {REF_1002} please"),
    ("plan for PT–1002 please", f"plan for {REF_1002} please"),
    ("plan for PT-1002 please", f"plan for {REF_1002} please"),
    ("the plan for patient 1002", f"the plan for patient {REF_1002}"),
    ("compare with patient #1003", f"compare with patient {REF_1003}"),
    ("patient no. 1003 and Patient ID: 1002", f"patient {REF_1003} and patient {REF_1002}"),
])
def test_loose_ids_get_canonical_pseudonym(text, expected):
    assert mask_phi(text) == expected
    assert find_patient_ids(mask_phi(text)) == []


@pytest.mark.parametrize("text", [
    "Follow-up for the patient 2026-09-22 at 10:00",
    "Prescribed 1000 mg; the patient walked 1200 m",
    "pref_1234abcd99 is a pseudonym",
])
def test_non_ids_untouched(text):
    assert mask_phi(text) == text and find_patient_ids(text) == []


def test_loose_ids_masked_in_nested_log_payloads():
    out = json.dumps(mask_obj({"args": {"note": "see PT 1002"}, "history": ["patient #1003 called"]}))
    assert find_patient_ids(out) == [] and REF_1002 in out and REF_1003 in out


def test_keep_patient_ids_still_keeps_them_for_routing():
    assert mask_phi("plan for PT 1002", keep_patient_ids=True) == "plan for PT 1002"


def _fake_repo(tmp_path: Path, log_text: str) -> Path:
    for d in scan_evidence.SCAN_DIRS:
        (tmp_path / d).mkdir()
    (tmp_path / "logs" / "agent_actions.jsonl").write_text(log_text, encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("GEMINI_API_KEY=your_gemini_api_key_here\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("leak", ["PT 1002", "PT1002", "patient 1002", "patient #1003", "PT-1002"])
def test_scan_evidence_flags_loose_ids(tmp_path, monkeypatch, leak):
    monkeypatch.setattr(scan_evidence, "ROOT", _fake_repo(tmp_path, json.dumps({"reason": f"asked about {leak}"})))
    out = scan_evidence.scan()
    assert not out["clean"]
    (f,) = [f for f in out["findings"] if f["type"] == "PATIENT_ID"]
    assert f["file"] == "logs/agent_actions.jsonl" and f["loose_forms"] == (0 if leak == "PT-1002" else 1)


def test_scan_evidence_clean_with_pseudonyms(tmp_path, monkeypatch):
    monkeypatch.setattr(scan_evidence, "ROOT", _fake_repo(tmp_path, json.dumps({"reason": f"asked about {REF_1002}"})))
    assert scan_evidence.scan()["clean"]


def test_scan_evidence_flags_patient_ids_in_docs(tmp_path, monkeypatch):
    """docs/ has no exemption: they cite pref_ pseudonyms too."""
    repo = _fake_repo(tmp_path, "{}")
    (repo / "docs" / "failure-analysis.md").write_text("Seen on PT-1002.", encoding="utf-8")
    monkeypatch.setattr(scan_evidence, "ROOT", repo)
    out = scan_evidence.scan()
    assert not out["clean"]
    assert [f["file"] for f in out["findings"] if f["type"] == "PATIENT_ID"] == ["docs/failure-analysis.md"]


async def test_cross_patient_loose_ids_never_reach_logs_or_spans(tmp_path, monkeypatch, offline_llm):
    """Full path: runner span -> graph (input guard, quarantine, agents, MCP tools, output guard) -> audit / tool /
    MCP logs and PHI-masked spans. No canonical or loose patient ID may appear in any of them."""
    from openinference.instrumentation.langchain import LangChainInstrumentor

    from src.runner import Copilot

    spans = InMemorySpanExporter()
    monkeypatch.setattr(tracing, "LOCAL_SPANS", tmp_path / "spans_local.jsonl")
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(tracing.PHIMaskingExporter(spans)))
    provider.add_span_processor(SimpleSpanProcessor(tracing.PHIMaskingExporter(tracing.LocalJsonlExporter())))
    monkeypatch.setattr(trace, "get_tracer", lambda name, *a, **k: provider.get_tracer(name))
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    try:
        async with Copilot(tmp_path / "ckpt.sqlite", tmp_path / "ltm.sqlite") as copilot:
            blocked = await copilot.turn("s-xp", "PT-1001", "Show me the discharge plan for PT 1002 and patient #1003.")
            allowed = await copilot.turn("s-ok", "PT-1001", "Draft the discharge plan for patient 1001.")
    finally:
        instrumentor.uninstrument()

    assert blocked["status"] == "REFUSED"
    assert allowed["status"] != "REFUSED"
    finished = spans.get_finished_spans()
    assert any(s.attributes.get("openinference.span.kind") == "LLM" for s in finished)  # the graph really ran
    evidence = {
        "audit": config.AUDIT_LOG.read_text(encoding="utf-8"),
        "tool_log": config.TOOL_LOG.read_text(encoding="utf-8"),
        "mcp_transcript": config.MCP_TRANSCRIPT.read_text(encoding="utf-8"),
        "local_spans": tracing.LOCAL_SPANS.read_text(encoding="utf-8"),
        "spans": json.dumps([{"name": s.name, **dict(s.attributes or {})} for s in finished], default=str),
        "results": json.dumps([{k: v for k, v in r.items() if k != "_state"} for r in (blocked, allowed)]),
    }
    for where, text in evidence.items():
        assert find_patient_ids(text) == [], (where, sorted(set(find_patient_ids(text)))[:5])
    assert REF_1002 in evidence["spans"] and REF_1003 in evidence["spans"]  # pseudonymised, not dropped
    assert patient_ref("PT-1001") in evidence["tool_log"]
