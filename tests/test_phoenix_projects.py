"""Judge spans go to their own Phoenix project (PHOENIX_PROJECT_NAME-eval), the exporter reads agent and judge
projects separately, and Gemini prices are registered in Phoenix through its GraphQL model mutations. Offline."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src import config
from src import llm as llm_mod
from src.observability import tracing

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("export_traces", ROOT / "scripts" / "export_traces.py")
export_traces = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_traces)


class FastLimiter:
    def acquire(self, *, blocking=True):
        return True

    async def aacquire(self, *, blocking=True):
        return True


def _provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_eval_project_name_derived_from_agent_project():
    assert config.PHOENIX_EVAL_PROJECT != config.PHOENIX_PROJECT
    assert config.PHOENIX_EVAL_PROJECT.startswith(config.PHOENIX_PROJECT)


def test_judge_wait_spans_routed_to_eval_provider(monkeypatch):
    agent_provider, agent_spans = _provider()
    eval_provider, eval_spans = _provider()
    monkeypatch.setattr(llm_mod.trace, "get_tracer", lambda name, *a, **k: agent_provider.get_tracer(name))
    monkeypatch.setattr(llm_mod, "_RATE_LIMITER", FastLimiter())
    monkeypatch.setattr(tracing, "_HANDLE", tracing.TracingHandle(
        provider=agent_provider, phoenix_url=None, project=config.PHOENIX_PROJECT, eval_provider=eval_provider,
        eval_project=config.PHOENIX_EVAL_PROJECT))

    with agent_provider.get_tracer("t").start_as_current_span("copilot_turn"):
        llm_mod.rate_limit_wait("agent-call")
        llm_mod.rate_limit_wait("judge:x", component="eval_judge")  # even inside a turn it is its own root

    assert [s.name for s in agent_spans.get_finished_spans()] == ["rate_limit_wait", "copilot_turn"]
    (judge,) = eval_spans.get_finished_spans()
    assert judge.attributes["copilot.component"] == "eval_judge"  # tag kept as the exporter's safety net
    assert judge.parent is None


def test_judge_spans_fall_back_to_global_tracer_before_init(monkeypatch):
    provider, spans = _provider()
    monkeypatch.setattr(llm_mod.trace, "get_tracer", lambda name, *a, **k: provider.get_tracer(name))
    monkeypatch.setattr(llm_mod, "_RATE_LIMITER", FastLimiter())
    monkeypatch.setattr(tracing, "_HANDLE", None)
    llm_mod.rate_limit_wait("judge:x", component="eval_judge")
    assert [s.attributes["copilot.component"] for s in spans.get_finished_spans()] == ["eval_judge"]


def test_register_model_pricing_creates_and_updates(monkeypatch):
    calls = []

    def fake_graphql(query, variables=None):
        calls.append((query, variables))
        if "generativeModels" in query:
            return {"generativeModels": {"edges": [
                {"node": {"id": "R2VuZXJhdGl2ZU1vZGVsOjE=", "name": config.GEMINI_MODEL, "kind": "CUSTOM"}},
                {"node": {"id": "builtin", "name": config.GEMINI_FALLBACK_MODEL, "kind": "BUILT_IN"}}]}}
        return {}

    monkeypatch.setattr(tracing, "_graphql", fake_graphql)
    result = tracing.register_model_pricing()
    models = list(dict.fromkeys([config.GEMINI_MODEL, config.GEMINI_FALLBACK_MODEL, config.GEMINI_JUDGE_MODEL]))
    assert list(result) == models
    assert result[config.GEMINI_MODEL] == "updated"
    assert all(result[m] == "created" for m in models if m != config.GEMINI_MODEL)  # built-ins are not edited

    by_name = {v["input"]["name"]: (q, v["input"]) for q, v in calls if v}
    q, inp = by_name[config.GEMINI_MODEL]
    assert "updateModel" in q and inp["id"] == "R2VuZXJhdGl2ZU1vZGVsOjE="
    import re

    assert re.search(inp["namePattern"], config.GEMINI_MODEL) and re.search(inp["namePattern"],
                                                                            "models/" + config.GEMINI_MODEL)
    assert not re.search(inp["namePattern"], config.GEMINI_MODEL + "-preview")
    costs = {c["tokenType"]: c for c in inp["costs"]}
    assert costs["input"]["kind"] == "PROMPT" and costs["output"]["kind"] == "COMPLETION"
    assert costs["input"]["costPerMillionTokens"] == config.PRICING_PER_MTOK[config.GEMINI_MODEL]["input"]
    assert costs["output"]["costPerMillionTokens"] == config.PRICING_PER_MTOK[config.GEMINI_MODEL]["output"]
    assert "createModel" in by_name[config.GEMINI_JUDGE_MODEL][0]


def test_register_model_pricing_never_raises(monkeypatch):
    def down(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(tracing, "_graphql", down)
    assert all(v.startswith("error:") for v in tracing.register_model_pricing().values())


def _row(trace, span, name, kind, start_s, end_s, parent=None, **attrs):
    return {"name": name, "context.trace_id": trace, "context.span_id": span, "parent_id": parent,
            "span_kind": kind, "start_time": pd.Timestamp(start_s, unit="s", tz="UTC"),
            "end_time": pd.Timestamp(end_s, unit="s", tz="UTC"), "status_code": "OK", **attrs}


def test_export_reads_agent_and_eval_projects_separately(tmp_path, monkeypatch):
    agent = pd.DataFrame([
        _row("t1", "a0", "copilot_turn", "CHAIN", 0, 10, **{"attributes.run_id": "run-1",
                                                               "attributes.session.id": "s1"}),
        _row("t1", "a1", "ChatGoogleGenerativeAI", "LLM", 3, 4, "a0",
             **{"attributes.llm.model_name": config.GEMINI_MODEL, "attributes.llm.token_count.prompt": 1000,
                "attributes.llm.token_count.completion": 100}),
    ])
    # judge spans in the eval project; one untagged, which still counts as judge traffic there
    judge = pd.DataFrame([
        _row("j1", "j1", "rate_limit_wait", "CHAIN", 20, 50,
             **{"attributes.copilot": json.dumps({"span_role": "rate_limit_wait", "component": "eval_judge"})}),
        _row("j2", "j2", "rate_limit_wait", "CHAIN", 60, 70),
    ])
    asked = []

    def fake_fetch(project=None, local_fallback=True):
        asked.append((project, local_fallback))
        return (agent, "phoenix.client") if project == config.PHOENIX_PROJECT else (judge, "phoenix.client")

    monkeypatch.setattr(tracing, "fetch_spans_dataframe", fake_fetch)
    monkeypatch.setattr(export_traces, "screenshot_phoenix", lambda *a, **k: False)
    monkeypatch.setattr(export_traces, "render_chart", lambda spans, out: out.write_bytes(b"png"))
    monkeypatch.setattr(config, "TRACES_DIR", tmp_path / "traces")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)

    g = export_traces.export()
    assert asked == [(config.PHOENIX_PROJECT, True), (config.PHOENIX_EVAL_PROJECT, False)]
    assert g["phoenix_project"] == config.PHOENIX_PROJECT
    assert g["traffic"]["traces"] == 1 and g["traffic"]["spans"] == 2 and g["traffic"]["turns"] == 1
    assert g["eval_judge"]["phoenix_project"] == config.PHOENIX_EVAL_PROJECT
    assert g["eval_judge"]["spans"] == 2 and g["eval_judge"]["rate_limit_wait"]["count"] == 2
    assert g["eval_judge"]["trace_export"] == "traces/phoenix_eval_spans.parquet"
    assert len(pd.read_parquet(tmp_path / "traces" / "phoenix_spans.parquet")) == 2
    assert len(pd.read_parquet(tmp_path / "traces" / "phoenix_eval_spans.parquet")) == 2
    csv = pd.read_csv(tmp_path / "dashboard_data.csv")
    assert {"prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "component"} <= set(csv.columns)
    assert csv["total_tokens"].sum() == 1100 and csv["cost_usd"].sum() > 0
    assert not (tmp_path / "dashboard_spans.png").exists()
