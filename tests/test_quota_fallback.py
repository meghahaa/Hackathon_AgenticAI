"""Groq last-resort fallback (QUOTA_FALLBACK_PROVIDER=groq, off by default).

Groq is called ONLY when the flag is on (with GROQ_API_KEY), the primary has failed (after its one retry when it
stalled) AND the Gemini fallback then returned 429 (quota), 503 (overloaded) or 404 (unavailable) - never when the
Gemini fallback succeeds, never on a first stall alone, never after a fallback timeout / 504 / other error. Groq
requests use the same limiter, masking and
tracing as Gemini, keep tool calling, and are labelled provider=groq (span, audit record, run record). The judge uses
GROQ_JUDGE_MODEL only on a Gemini judge quota error with the flag on, and never judges with a model that produced the
answer. Offline: Gemini and Groq transports are faked; no request leaves the machine."""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src import config
from src import llm as llm_mod
from src.guardrails.guards import _identifier_registry, find_patient_ids, find_phi, patient_ref
from src.observability import run_context

QUOTA = "429 RESOURCE_EXHAUSTED. You exceeded your current quota (GenerateRequestsPerDayPerProjectPerModel-FreeTier)"
STALL = "504 DEADLINE_EXCEEDED. Deadline expired before operation could complete."
UNAVAILABLE = "503 UNAVAILABLE. This model is currently experiencing high demand."
NOT_FOUND = "404 NOT_FOUND. This model models/gemini-x is no longer available to new users."
INVALID = "400 INVALID_ARGUMENT. Request contains an invalid argument."


class SpyLimiter:
    def __init__(self) -> None:
        self.acquired = 0

    def acquire(self, *, blocking: bool = True) -> bool:
        self.acquired += 1
        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        self.acquired += 1
        return True


@pytest.fixture
def providers(monkeypatch):
    """Fake Gemini (per-model scripted errors) + fake Groq transport; returns a namespace of recorders."""
    exporter = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(llm_mod.trace, "get_tracer", lambda name, *a, **k: tp.get_tracer(name))
    limiter = SpyLimiter()
    monkeypatch.setattr(llm_mod, "_RATE_LIMITER", limiter)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key-offline-test")
    monkeypatch.setattr(config, "GEMINI_FALLBACK_MODEL", "gemini-fallback-test")
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-groq-key-offline-test")
    monkeypatch.setattr(config, "QUOTA_FALLBACK_PROVIDER", "none")
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 0)  # one attempt: the routing inside it is what is tested
    ns = type("NS", (), {})()
    ns.limiter, ns.exporter = limiter, exporter
    ns.requests: list[str] = []
    ns.errors: dict[str, list[str]] = {}  # model -> errors to raise, one per request (then success)
    ns.groq_seen: list[dict] = []

    async def gemini_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        ns.requests.append(self.model)
        queue = ns.errors.get(self.model) or []
        if queue:
            raise RuntimeError(queue.pop(0))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"ok from {self.model}"))])

    async def groq_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        ns.requests.append(f"groq:{self.model_name}")
        ns.groq_seen.append({"messages": messages, "kwargs": kwargs})
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok from groq"))])

    monkeypatch.setattr(ChatGoogleGenerativeAI, "_agenerate", gemini_agenerate)
    monkeypatch.setattr(ChatGroq, "_agenerate", groq_agenerate)
    llm_mod.set_llm(None)
    calls: list[dict] = []
    token = run_context.MODEL_CALLS.set(calls)
    ns.calls = calls
    yield ns
    run_context.MODEL_CALLS.reset(token)
    llm_mod.set_llm(None)


def _enable_groq(monkeypatch):
    monkeypatch.setattr(config, "QUOTA_FALLBACK_PROVIDER", "groq")
    llm_mod.set_llm(None)  # rebuild the chain with the Groq member


async def _ask(text: str = "hi"):
    return await llm_mod.ainvoke_with_retry([HumanMessage(text)], run_name="unit")


# ------------------------------------------------------------------ when Groq may (not) be called
async def test_flag_off_never_calls_groq_even_when_both_gemini_models_hit_quota(providers):
    providers.errors = {config.GEMINI_MODEL: [QUOTA], "gemini-fallback-test": [QUOTA]}
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        await _ask()
    assert providers.requests == [config.GEMINI_MODEL, "gemini-fallback-test"]
    assert providers.groq_seen == []


async def test_flag_on_without_groq_key_never_calls_groq(providers, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    _enable_groq(monkeypatch)
    providers.errors = {config.GEMINI_MODEL: [QUOTA], "gemini-fallback-test": [QUOTA]}
    with pytest.raises(RuntimeError):
        await _ask()
    assert providers.groq_seen == []


@pytest.mark.parametrize("primary, fallback", [
    ([QUOTA], [STALL]),                # fallback stalled (504): not a Groq trigger
    ([QUOTA], ["TimeoutError"]),       # fallback timed out
    ([UNAVAILABLE], [INVALID]),        # fallback failed with another error
    ([STALL, STALL], ["500 INTERNAL"]),
])
async def test_flag_on_no_groq_after_other_fallback_errors(providers, monkeypatch, primary, fallback):
    _enable_groq(monkeypatch)
    providers.errors = {config.GEMINI_MODEL: list(primary), "gemini-fallback-test": list(fallback)}
    with pytest.raises(RuntimeError):
        await _ask()
    assert providers.groq_seen == []
    assert not any(c["provider"] == "groq" for c in providers.calls)


@pytest.mark.parametrize("primary", [[QUOTA], [UNAVAILABLE], [STALL, STALL], [INVALID]])
async def test_flag_on_no_groq_when_the_gemini_fallback_succeeds(providers, monkeypatch, primary):
    _enable_groq(monkeypatch)
    providers.errors = {config.GEMINI_MODEL: list(primary)}
    msg = await _ask()
    assert msg.content == "ok from gemini-fallback-test" and providers.groq_seen == []


async def test_flag_on_no_groq_on_a_first_stall_alone(providers, monkeypatch):
    _enable_groq(monkeypatch)
    providers.errors = {config.GEMINI_MODEL: [STALL], "gemini-fallback-test": [QUOTA]}
    msg = await _ask()  # the primary's one retry answers: neither the fallback nor Groq is sent
    assert msg.content == f"ok from {config.GEMINI_MODEL}"
    assert providers.requests == [config.GEMINI_MODEL, config.GEMINI_MODEL] and providers.groq_seen == []


@pytest.mark.parametrize("primary", [[QUOTA], [UNAVAILABLE], [STALL, STALL], [INVALID]])
@pytest.mark.parametrize("fallback, trigger", [([QUOTA], "quota"), ([UNAVAILABLE], "overloaded"),
                                               ([NOT_FOUND], "unavailable")])
async def test_flag_on_groq_after_primary_failed_and_fallback_429_503_404(providers, monkeypatch, primary, fallback,
                                                                          trigger):
    _enable_groq(monkeypatch)
    providers.errors = {config.GEMINI_MODEL: list(primary), "gemini-fallback-test": list(fallback)}
    msg = await _ask()
    assert msg.content == "ok from groq"
    assert providers.requests == [config.GEMINI_MODEL] * len(primary) + ["gemini-fallback-test",
                                                                          f"groq:{config.GROQ_MODEL}"]
    audit = [json.loads(l) for l in config.AUDIT_LOG.read_text(encoding="utf-8").splitlines()]
    (rec,) = [r for r in audit if r["action"] == "quota_fallback"]
    assert rec["details"]["trigger"] == trigger and rec["reason"] == f"primary failed; Gemini fallback {trigger}"
    (fb,) = [sp for sp in providers.exporter.get_finished_spans() if sp.name == "quota_fallback"]
    assert fb.attributes["copilot.trigger"] == trigger


async def test_stalled_primary_is_retried_once_before_the_fallback(providers):
    providers.errors = {config.GEMINI_MODEL: [STALL]}
    msg = await _ask()
    assert msg.content == f"ok from {config.GEMINI_MODEL}"
    assert providers.requests == [config.GEMINI_MODEL, config.GEMINI_MODEL]  # no fallback needed
    providers.errors = {config.GEMINI_MODEL: [STALL, STALL]}
    providers.requests.clear()
    msg = await _ask()
    assert providers.requests == [config.GEMINI_MODEL, config.GEMINI_MODEL, "gemini-fallback-test"]
    # a non-stall error goes straight to the fallback (no second primary request)
    providers.errors = {config.GEMINI_MODEL: [UNAVAILABLE]}
    providers.requests.clear()
    await _ask()
    assert providers.requests == [config.GEMINI_MODEL, "gemini-fallback-test"]


async def test_groq_answers_after_both_quota_errors_and_is_labelled_everywhere(providers, monkeypatch):
    _enable_groq(monkeypatch)
    providers.errors = {config.GEMINI_MODEL: [QUOTA], "gemini-fallback-test": [QUOTA]}
    msg = await _ask()
    assert msg.content == "ok from groq"
    assert providers.requests == [config.GEMINI_MODEL, "gemini-fallback-test", f"groq:{config.GROQ_MODEL}"]
    # reply + run record
    assert msg.response_metadata["copilot_provider"] == "groq"
    assert llm_mod.answered_by(msg) == {"model": config.GROQ_MODEL, "provider": "groq"}
    assert [(c["route"], c["provider"], c["ok"]) for c in providers.calls] == [
        ("primary", "gemini", False), ("fallback", "gemini", False), ("quota_fallback", "groq", True)]
    assert run_context.provider_summary(providers.calls)["groq_involved"] is True
    assert run_context.run_providers() == ["groq"]
    # audit record
    audit = [json.loads(l) for l in config.AUDIT_LOG.read_text(encoding="utf-8").splitlines()]
    (rec,) = [r for r in audit if r["action"] == "quota_fallback"]
    assert rec["decision"] == "groq" and rec["details"]["provider"] == "groq"
    assert rec["details"]["model"] == config.GROQ_MODEL and rec["details"]["run_name"] == "unit"
    # spans: a quota_fallback span and a rate_limit_wait span, both provider=groq
    spans = providers.exporter.get_finished_spans()
    (fb,) = [s for s in spans if s.name == "quota_fallback"]
    assert fb.attributes["copilot.provider"] == "groq" and fb.attributes["copilot.model"] == config.GROQ_MODEL
    waits = [s.attributes.get("copilot.provider") for s in spans if s.name == "rate_limit_wait"]
    assert "groq" in waits


async def test_groq_request_is_rate_limited_like_gemini(providers, monkeypatch):
    _enable_groq(monkeypatch)
    providers.errors = {config.GEMINI_MODEL: [QUOTA], "gemini-fallback-test": [QUOTA]}
    await _ask()
    assert providers.limiter.acquired == len(providers.requests) == 3  # one token per request, Groq included


async def test_requests_to_groq_contain_no_patient_ids_or_phi(providers, monkeypatch):
    _enable_groq(monkeypatch)
    providers.errors = {config.GEMINI_MODEL: [QUOTA], "gemini-fallback-test": [QUOTA]}
    name = next(lit for lit, label in _identifier_registry() if len(lit) > 3)
    text = f"Plan for PT-1001 (also PT 1002, patient #1003); patient name {name}."
    await _ask(text)
    (seen,) = providers.groq_seen
    sent = " ".join(str(m.content) for m in seen["messages"])
    assert find_patient_ids(sent) == [] and find_phi(sent) == []
    assert patient_ref("PT-1001") in sent


async def test_groq_keeps_tool_calling(providers, monkeypatch):
    _enable_groq(monkeypatch)
    providers.errors = {config.GEMINI_MODEL: [QUOTA], "gemini-fallback-test": [QUOTA]}

    def search_followup_policy(query: str) -> str:
        """Fake policy search."""
        return query

    llm = llm_mod.get_tool_llm([search_followup_policy])
    await llm_mod.ainvoke_with_retry([HumanMessage("what is the policy for PT-1001?")], run_name="policy_tool_loop",
                                     llm=llm)
    (seen,) = providers.groq_seen
    tools = seen["kwargs"].get("tools") or []
    assert [t["function"]["name"] for t in tools] == ["search_followup_policy"]
    assert find_patient_ids(" ".join(str(m.content) for m in seen["messages"])) == []


# ------------------------------------------------------------------ judge
class FakeGemini:
    def __init__(self, error: str | None) -> None:
        self.error, self.requests = error, []

    @property
    def models(self):
        return self

    def generate_content(self, model, contents, config=None):
        self.requests.append(model)
        if self.error:
            raise RuntimeError(self.error)
        return type("R", (), {"text": '{"verdict": "yes"}', "usage_metadata": None})()


class FakeGroq:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.chat = type("C", (), {"completions": self})()

    def create(self, model, messages, temperature=0.0):
        self.prompts.append(messages[0]["content"])
        msg = type("M", (), {"content": '{"verdict": "yes"}'})()
        return type("R", (), {"choices": [type("Ch", (), {"message": msg})()], "usage": None})()


@pytest.fixture
def judge_env(monkeypatch):
    pytest.importorskip("deepeval")
    import src.eval_judge as eval_judge

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key-offline-test")
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-groq-key-offline-test")
    monkeypatch.setattr(config, "GEMINI_JUDGE_FALLBACK_MODEL", "")
    monkeypatch.setattr(eval_judge, "model_limiter", lambda m: SpyLimiter())
    monkeypatch.setattr(eval_judge.time, "sleep", lambda s: None)
    return eval_judge


def _judge(eval_judge, gemini_error: str | None):
    judge = eval_judge.GeminiFallbackJudge()
    judge.model = FakeGemini(gemini_error)
    judge._groq_client = FakeGroq()
    return judge


def test_judge_flag_off_never_uses_groq_on_quota(judge_env, monkeypatch):
    monkeypatch.setattr(config, "QUOTA_FALLBACK_PROVIDER", "none")
    judge = _judge(judge_env, QUOTA)
    assert judge.groq_model is None
    with pytest.raises(RuntimeError, match="judge failed"):  # -> run_eval marks the metric "unjudged"
        judge.generate("judge this")
    assert judge._groq_client.prompts == []


def test_judge_flag_on_uses_groq_judge_model_only_on_quota(judge_env, monkeypatch):
    monkeypatch.setattr(config, "QUOTA_FALLBACK_PROVIDER", "groq")
    judge = _judge(judge_env, UNAVAILABLE)  # not a quota error: no Groq
    with pytest.raises(RuntimeError):
        judge.generate("judge this")
    assert judge._groq_client.prompts == []

    judge = _judge(judge_env, QUOTA)
    name = next(lit for lit, label in _identifier_registry() if len(lit) > 3)
    assert judge.generate(f"Actual output: plan for PT-1001, {name}") == '{"verdict": "yes"}'
    (prompt,) = judge._groq_client.prompts
    assert find_patient_ids(prompt) == [] and find_phi(prompt) == []  # masked like the Gemini judge prompt
    usage = judge.usage_summary()
    assert usage["groq_calls"] == 1 and usage["by_provider"]["groq"] == 1
    assert [c["provider"] for c in judge.calls] == ["gemini", "groq"]


def test_judge_never_uses_the_model_that_produced_the_answer(judge_env, monkeypatch):
    monkeypatch.setattr(config, "QUOTA_FALLBACK_PROVIDER", "groq")
    judge = _judge(judge_env, QUOTA)
    judge.set_producers({config.GEMINI_MODEL, config.GROQ_JUDGE_MODEL})  # the Groq judge model wrote this answer
    with pytest.raises(RuntimeError):
        judge.generate("judge this")
    assert judge._groq_client.prompts == []
    judge.set_producers({config.GEMINI_JUDGE_MODEL})  # the Gemini judge model wrote it: no judge left
    with pytest.raises(RuntimeError, match="distinct from the models that produced"):
        judge.generate("judge this")


def test_groq_judge_must_differ_from_the_groq_agent_model(judge_env, monkeypatch):
    monkeypatch.setattr(config, "QUOTA_FALLBACK_PROVIDER", "groq")
    monkeypatch.setattr(config, "GROQ_JUDGE_MODEL", config.GROQ_MODEL)
    assert judge_env.GeminiFallbackJudge().groq_model is None


# ------------------------------------------------------------------ reporting
def test_eval_quality_split_by_provider():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).resolve().parent.parent / "scripts" / "run_eval.py")
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)

    def row(cid, ok, groq, status="DRAFT_READY_FOR_CLINICIAN_SIGNOFF"):
        m = {"score": 1.0 if ok else 0.0, "success": ok, "status": "judged"}
        return {"id": cid, "passed": ok, "status": status, "groq_involved": groq,
                "metrics": {"hallucination": m, "faithfulness": m}}
    rows = [row("A", True, False), row("B", True, False), row("C", False, True)]
    gem = run_eval.quality_block([r for r in rows if not r["groq_involved"]])
    groq = run_eval.quality_block([r for r in rows if r["groq_involved"]])
    assert gem["case_ids"] == ["A", "B"] and gem["accuracy"] == 1.0 and gem["hallucination_rate_answers"] == 0.0
    assert groq["case_ids"] == ["C"] and groq["hallucination_rate_answers"] == 1.0


def test_export_labels_provider_and_splits_groq_turns(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    import pandas as pd

    spec = importlib.util.spec_from_file_location("export_traces",
                                                  Path(__file__).resolve().parent.parent / "scripts" / "export_traces.py")
    export_traces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_traces)
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)

    def span(trace, sid, name, kind, t0, t1, **attrs):
        return {"name": name, "context.trace_id": trace, "context.span_id": sid, "parent_id": None,
                "span_kind": kind, "start_time": pd.Timestamp(t0, unit="s", tz="UTC"),
                "end_time": pd.Timestamp(t1, unit="s", tz="UTC"), "status_code": "OK", **attrs}
    frame = pd.DataFrame([
        span("g", "g0", "copilot_turn", "CHAIN", 0, 4),
        span("g", "g1", "ChatGoogleGenerativeAI", "LLM", 1, 2, **{"attributes.llm.model_name": config.GEMINI_MODEL,
                                                                   "attributes.llm.provider": "google"}),
        span("q", "q0", "copilot_turn", "CHAIN", 10, 20),
        span("q", "q1", "quota_fallback", "CHAIN", 11, 13, **{"attributes.copilot": json.dumps(
            {"span_role": "quota_fallback", "provider": "groq", "run_name": "plan_narrative"})}),
        span("q", "q2", "ChatGroq", "LLM", 11, 13, **{"attributes.llm.model_name": config.GROQ_MODEL,
                                                      "attributes.llm.provider": "groq",
                                                      "attributes.llm.token_count.prompt": 1000}),
    ])
    spans = export_traces.normalise(frame)
    assert dict(zip(spans["span_id"], spans["provider"])) == {"g0": "", "g1": "gemini", "q0": "", "q1": "groq",
                                                               "q2": "groq"}
    p = export_traces.golden_signals(spans, "test")["providers"]
    assert p["headline_scope"] == "gemini_only"
    assert p["gemini_only"]["turns"] == 1 and p["groq_involved"]["turns"] == 1
    assert p["llm_calls_by_provider"] == {"gemini": 1, "groq": 1}
    assert p["groq_calls_by_agent_node"] == {"plan_narrative": 1}
    assert p["groq_involved"]["mean_cost_usd_per_turn"] == round(1000 * config.PRICING_PER_MTOK[config.GROQ_MODEL]["input"] / 1e6, 6)
    spans.to_csv(tmp_path / "dashboard_data.csv", index=False)
    assert "provider" in pd.read_csv(tmp_path / "dashboard_data.csv").columns
