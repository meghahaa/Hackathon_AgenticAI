"""Golden-signal hygiene: the client-side rate-limiter wait is traced in its own span, not inside the LLM call."""
from __future__ import annotations

import time
from typing import ClassVar

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src import llm as llm_mod


class RecordingFakeLLM(FakeListChatModel):
    """Records the active span and time when the model is actually called."""

    seen: ClassVar[list] = []

    async def ainvoke(self, *args, **kwargs):
        span = trace.get_current_span()
        RecordingFakeLLM.seen.append((span.get_span_context().span_id, time.time_ns()))
        return await super().ainvoke(*args, **kwargs)


def _tracer_provider(monkeypatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(llm_mod.trace, "get_tracer", lambda name, *a, **k: provider.get_tracer(name))
    return exporter


async def test_rate_limit_wait_span_is_separate_from_llm_call(monkeypatch):
    exporter = _tracer_provider(monkeypatch)
    acquired = []
    limiter = InMemoryRateLimiter(requests_per_second=1000, check_every_n_seconds=0.001, max_bucket_size=1)
    real_aacquire = limiter.aacquire

    async def spy_aacquire(*a, **k):
        acquired.append(trace.get_current_span().get_span_context().span_id)
        return await real_aacquire(*a, **k)

    monkeypatch.setattr(limiter, "aacquire", spy_aacquire)
    monkeypatch.setattr(llm_mod, "_RATE_LIMITER", limiter)
    RecordingFakeLLM.seen = []
    fake = RecordingFakeLLM(responses=["ok"])

    msg = await llm_mod.ainvoke_with_retry([HumanMessage("hi")], run_name="unit", llm=fake)

    assert msg.content == "ok"
    spans = [s for s in exporter.get_finished_spans() if s.name == "rate_limit_wait"]
    assert len(spans) == 1
    wait = spans[0]
    assert wait.attributes["openinference.span.kind"] == "CHAIN"
    assert wait.attributes["copilot.span_role"] == "rate_limit_wait"
    assert wait.attributes["copilot.rate_limited"] is True
    # the limiter was acquired inside the rate_limit_wait span ...
    assert acquired == [wait.context.span_id]
    # ... and the model call happened after that span closed, outside it
    (llm_span_id, llm_started_ns), = RecordingFakeLLM.seen
    assert llm_span_id != wait.context.span_id
    assert wait.end_time <= llm_started_ns


def test_gemini_client_has_no_embedded_rate_limiter(monkeypatch):
    monkeypatch.setattr(llm_mod.config, "GEMINI_API_KEY", "test-key-not-used")
    model = llm_mod._gemini(llm_mod.config.GEMINI_MODEL)
    assert getattr(model, "rate_limiter", None) is None


async def test_offline_stub_emits_wait_span_without_throttling(monkeypatch):
    exporter = _tracer_provider(monkeypatch)

    async def must_not_acquire(*a, **k):
        raise AssertionError("offline stub makes no API requests and must not be throttled")

    monkeypatch.setattr(llm_mod._RATE_LIMITER, "aacquire", must_not_acquire)
    await llm_mod.ainvoke_with_retry([HumanMessage("hi")], run_name="unit", llm=llm_mod.OfflineStubChatModel())
    spans = [s for s in exporter.get_finished_spans() if s.name == "rate_limit_wait"]
    assert len(spans) == 1 and spans[0].attributes["copilot.rate_limited"] is False
