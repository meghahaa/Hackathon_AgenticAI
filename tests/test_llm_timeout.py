"""Gemini deadlines: every request (primary or fallback) gets its own GEMINI_REQUEST_TIMEOUT_S, and time spent waiting
for a rate-limit token never counts against it (nor against the LLM_TIMEOUT_S backstop). Offline: the Gemini
transport is faked and the shared limiter is replaced by one whose every token takes longer than the deadline."""
from __future__ import annotations

import asyncio
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI

from src import config
from src import llm as llm_mod

TIMEOUT_S = 0.3
TOKEN_WAIT_S = 0.6  # every token wait alone is longer than the deadline


class SlowLimiter:
    def __init__(self) -> None:
        self.waits = 0

    async def aacquire(self, *, blocking: bool = True) -> bool:
        self.waits += 1
        await asyncio.sleep(TOKEN_WAIT_S)
        return True

    def acquire(self, *, blocking: bool = True) -> bool:
        raise AssertionError("async path only")


@pytest.fixture
def gemini(monkeypatch):
    """Fake-key Gemini primary + fallback; returns (limiter, requests, fail_models, model_delay)."""
    limiter = SlowLimiter()
    monkeypatch.setattr(llm_mod, "_RATE_LIMITER", limiter)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key-offline-test")
    monkeypatch.setattr(config, "GEMINI_FALLBACK_MODEL", "gemini-fallback-test")
    monkeypatch.setattr(config, "LLM_TIMEOUT_S", TIMEOUT_S * 3)  # backstop above the deadline, as in production
    monkeypatch.setattr(config, "GEMINI_REQUEST_TIMEOUT_S", TIMEOUT_S)
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 0)
    requests: list[str] = []
    fail_models: set[str] = set()
    model_delay = {"s": 0.0}

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        requests.append(self.model)
        if self.model in fail_models:
            raise RuntimeError("503 UNAVAILABLE (fake)")
        await asyncio.sleep(model_delay["s"])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    monkeypatch.setattr(ChatGoogleGenerativeAI, "_agenerate", _agenerate)
    llm_mod.set_llm(None)
    yield limiter, requests, fail_models, model_delay
    llm_mod.set_llm(None)


def test_request_deadline_is_configured_on_the_client(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_REQUEST_TIMEOUT_S", 12.0)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key-offline-test")  # builds the client only; no call
    assert llm_mod._gemini("gemini-any").timeout == 12.0  # also sent to the API as the request deadline


async def test_fallback_token_wait_does_not_count_against_the_timeout(gemini):
    """Primary fails fast -> the fallback model waits for its own token inside the same attempt. That wait (0.6 s)
    is longer than the deadline (0.3 s) but must not time the request out."""
    limiter, requests, fail_models, _ = gemini
    fail_models.add(config.GEMINI_MODEL)
    msg = await llm_mod.ainvoke_with_retry([HumanMessage("hi")], run_name="unit")
    assert msg.content == "ok"
    assert requests == [config.GEMINI_MODEL, "gemini-fallback-test"]
    assert limiter.waits == 2  # the attempt's token + the fallback's own token, both slower than the deadline


async def test_stalled_primary_is_cut_at_deadline_and_fallback_gets_a_full_deadline(gemini, monkeypatch):
    """The primary stalls (would run 10 s): it is cut off at GEMINI_REQUEST_TIMEOUT_S and, being a stall, sent once
    more (stalls again, cut again). The fallback is then sent and needs 0.25 s, more than any shared budget would have
    left (backstop 0.4 s), so it only succeeds because every request's deadline is fresh."""
    limiter, requests, _, _ = gemini
    monkeypatch.setattr(config, "LLM_TIMEOUT_S", 0.4)
    timing: dict[str, list[float]] = {}

    async def _stall_then_ok(self, messages, stop=None, run_manager=None, **kwargs):
        requests.append(self.model)
        t0 = time.monotonic()
        try:
            await asyncio.sleep(10.0 if self.model == config.GEMINI_MODEL else TIMEOUT_S * 0.83)
        finally:
            timing.setdefault(self.model, []).append(time.monotonic() - t0)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"ok from {self.model}"))])

    monkeypatch.setattr(ChatGoogleGenerativeAI, "_agenerate", _stall_then_ok)
    t0 = time.monotonic()
    msg = await llm_mod.ainvoke_with_retry([HumanMessage("hi")], run_name="unit")
    assert msg.content == "ok from gemini-fallback-test"
    # stall -> the primary once more -> the fallback, which was actually sent
    assert requests == [config.GEMINI_MODEL, config.GEMINI_MODEL, "gemini-fallback-test"]
    for t in timing[config.GEMINI_MODEL]:  # each primary request cut at its own deadline (Windows clock ~15 ms)
        assert TIMEOUT_S - 0.05 <= t < TIMEOUT_S + 0.2
    assert timing["gemini-fallback-test"][0] >= TIMEOUT_S * 0.8  # fallback ran to completion on its own deadline
    assert limiter.waits == 3  # every request's token wait (0.6 s each) was not charged to it either
    assert msg.response_metadata["copilot_provider"] == "gemini"
    assert time.monotonic() - t0 < 6  # nowhere near the 10 s stall


async def test_stalled_primary_and_fallback_both_time_out(gemini):
    _, requests, _, model_delay = gemini
    model_delay["s"] = TIMEOUT_S * 3
    with pytest.raises(RuntimeError, match="TimeoutError"):
        await llm_mod.ainvoke_with_retry([HumanMessage("hi")], run_name="unit")
    # stall -> primary once more -> fallback; each cut at its own deadline
    assert requests == [config.GEMINI_MODEL, config.GEMINI_MODEL, "gemini-fallback-test"]
