"""DeepEval judge spans (copilot.component=eval_judge) must not leak into the agent's golden signals: they are
excluded from agent latency, trace/turn counts and cost, and reported in golden_signals.json "eval_judge"."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from src import config
from src import llm as llm_mod

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("export_traces", ROOT / "scripts" / "export_traces.py")
export_traces = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_traces)

S = 1_000_000_000  # ns


@pytest.fixture(autouse=True)
def _no_real_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)  # golden_signals reads eval_report.json from here


def _row(trace, span, name, kind, start_s, end_s, parent=None, **attrs):
    return {"name": name, "context.trace_id": trace, "context.span_id": span, "parent_id": parent,
            "span_kind": kind, "start_time": pd.Timestamp(start_s, unit="s", tz="UTC"),
            "end_time": pd.Timestamp(end_s, unit="s", tz="UTC"), "status_code": "OK", **attrs}


def _phoenix_frame() -> pd.DataFrame:
    """Phoenix export shape: copilot.* attributes nested as a JSON string under attributes.copilot."""
    agent = [
        _row("t1", "a0", "copilot_turn", "CHAIN", 0, 10, **{"attributes.run_id": "run-1",
                                                               "attributes.session.id": "s1"}),
        _row("t1", "a1", "rate_limit_wait", "CHAIN", 1, 3, "a0",
             **{"attributes.copilot": json.dumps({"span_role": "rate_limit_wait", "component": "agent"})}),
        _row("t1", "a2", "ChatGoogleGenerativeAI", "LLM", 3, 4, "a0",
             **{"attributes.llm.model_name": config.GEMINI_MODEL, "attributes.llm.token_count.prompt": 1000,
                "attributes.llm.token_count.completion": 100}),
    ]
    judge = [_row(f"j{i}", f"j{i}", "rate_limit_wait", "CHAIN", 20 + i, 50 + i,
                  **{"attributes.copilot": json.dumps({"span_role": "rate_limit_wait", "component": "eval_judge"})})
             for i in range(3)]
    return pd.DataFrame(agent + judge)


def test_judge_spans_excluded_from_agent_signals():
    spans = export_traces.normalise(_phoenix_frame())
    assert spans.loc[spans["name"] == "copilot_turn", "component"].item() == "agent"
    assert (spans["component"] == "eval_judge").sum() == 3
    g = export_traces.golden_signals(spans, "test")

    # agent-only: the 3 judge traces / spans / 30 s waits do not appear
    assert g["traffic"]["traces"] == 1 and g["traffic"]["spans"] == 3 and g["traffic"]["turns"] == 1
    assert g["latency"]["rate_limit_wait"] == {"count": 1, "p50_ms": 2000.0, "p95_ms": 2000.0, "max_ms": 2000.0}
    assert g["latency"]["thinking"]["count"] == 1 and g["latency"]["thinking"]["p50_ms"] == 1000.0
    assert g["latency"]["thinking_including_rate_limit"]["p50_ms"] == 3000.0
    agent_cost = spans.loc[spans["component"] == "agent", "cost_usd"].sum()
    assert g["cost"]["total_usd"] == round(float(agent_cost), 6)
    assert g["cost"]["mean_usd_per_turn"] == round(float(agent_cost), 6)
    assert g["window"]["last_span"] == str(pd.Timestamp(10, unit="s", tz="UTC"))

    # reported separately instead
    j = g["eval_judge"]
    assert j["spans"] == 3 and j["traces"] == 3
    assert j["rate_limit_wait"]["count"] == 3 and j["rate_limit_wait"]["p50_ms"] == 30000.0
    assert j["cost_usd_from_spans"] == 0.0
    assert "eval_judge" in g["latency_definitions"]["scope"]


def test_real_judge_spans_are_tagged_and_excluded(tmp_path, monkeypatch):
    """End to end: spans the judge actually emits -> local masked span sink -> exporter."""
    pytest.importorskip("deepeval")
    from src.eval_judge import JUDGE_COMPONENT, GeminiFallbackJudge
    from src.observability import tracing

    assert JUDGE_COMPONENT == export_traces.JUDGE_COMPONENT
    monkeypatch.setattr(tracing, "LOCAL_SPANS", tmp_path / "spans_local.jsonl")
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(tracing.PHIMaskingExporter(tracing.LocalJsonlExporter())))
    tracer = provider.get_tracer("discharge-copilot")
    monkeypatch.setattr(llm_mod.trace, "get_tracer", lambda name, *a, **k: provider.get_tracer(name))

    class FastLimiter:
        def acquire(self, *, blocking=True):
            return True

        async def aacquire(self, *, blocking=True):
            return True

    monkeypatch.setattr(llm_mod, "_RATE_LIMITER", FastLimiter())

    # one agent turn (agent-tagged wait span + LLM span with tokens)
    with tracer.start_as_current_span("copilot_turn", attributes={"run_id": "run-1", "session.id": "s1"}):
        llm_mod.rate_limit_wait("unit")
        with tracer.start_as_current_span("ChatGoogleGenerativeAI", attributes={
                "openinference.span.kind": "LLM", "llm.model_name": config.GEMINI_MODEL,
                "llm.token_count.prompt": 500, "llm.token_count.completion": 50}):
            pass

    # two judge requests, outside any agent turn (as in scripts/run_eval.py)
    class FakeModels:
        def generate_content(self, model, contents, config=None):
            return type("R", (), {"text": "ok", "usage_metadata": None})()

    monkeypatch.setitem(llm_mod._MODEL_LIMITERS, config.GEMINI_JUDGE_MODEL, FastLimiter())  # judge's own budget
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key-offline-test")  # client is built, then faked; no call
    judge = GeminiFallbackJudge()
    judge.model = type("FakeClient", (), {"models": FakeModels()})()
    judge.generate("judge this")
    judge.generate("judge this too")

    raw = pd.DataFrame([json.loads(l) for l in tracing.LOCAL_SPANS.read_text(encoding="utf-8").splitlines()])
    spans = export_traces.normalise(raw)
    judge_spans = spans[spans["component"] == JUDGE_COMPONENT]
    assert len(judge_spans) == 2 and set(judge_spans["name"]) == {"rate_limit_wait"}

    g = export_traces.golden_signals(spans, "local-masked-otel-copy")
    assert g["traffic"]["traces"] == 1 and g["traffic"]["turns"] == 1
    assert g["latency"]["rate_limit_wait"]["count"] == 1
    assert g["eval_judge"]["spans"] == 2 and g["eval_judge"]["rate_limit_wait"]["count"] == 2
    assert g["cost"]["total_usd"] == g["cost"]["mean_usd_per_turn"] > 0
