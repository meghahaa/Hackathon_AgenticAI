"""Every Gemini request, on every code path, takes a token from the ONE shared limiter inside a rate_limit_wait
span before the model is called. Offline: the Gemini transport (_generate/_agenerate/_stream/_astream) is faked,
so nothing leaves the process. A path that reaches the model without the limiter makes requests != acquisitions."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from src import config
from src import llm as llm_mod

ROOT = Path(__file__).resolve().parent.parent


class SpyLimiter:
    """Records, for each acquisition, the name of the span it happened in."""

    def __init__(self) -> None:
        self.acquired: list[str] = []

    def _record(self) -> bool:
        self.acquired.append(getattr(llm_mod.trace.get_current_span(), "name", None))
        return True

    def acquire(self, *, blocking: bool = True) -> bool:
        return self._record()

    async def aacquire(self, *, blocking: bool = True) -> bool:
        return self._record()


@pytest.fixture
def gemini(monkeypatch):
    """Gemini 'configured' with a fake key and a fake transport; returns (limiter, requests, fail_models)."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(llm_mod.trace, "get_tracer", lambda name, *a, **k: provider.get_tracer(name))
    limiter = SpyLimiter()
    monkeypatch.setattr(llm_mod, "_RATE_LIMITER", limiter)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key-offline-test")
    monkeypatch.setattr(config, "GEMINI_FALLBACK_MODEL", "gemini-fallback-test")
    requests: list[str] = []
    fail_models: set[str] = set()

    def reply(model) -> AIMessage:
        requests.append(model.model)
        if model.model in fail_models:
            raise RuntimeError("503 UNAVAILABLE (fake)")
        return AIMessage(content='{"answer": "ok"}')

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=reply(self))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return _generate(self, messages)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content=reply(self).content))

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content=reply(self).content))

    for name, fn in (("_generate", _generate), ("_agenerate", _agenerate), ("_stream", _stream),
                     ("_astream", _astream)):
        monkeypatch.setattr(ChatGoogleGenerativeAI, name, fn)
    llm_mod.set_llm(None)
    yield limiter, requests, fail_models
    llm_mod.set_llm(None)


def _assert_every_request_throttled(limiter: SpyLimiter, requests: list[str]) -> None:
    assert requests, "the path under test never reached the model"
    assert len(limiter.acquired) == len(requests), f"{len(requests)} Gemini requests, {len(limiter.acquired)} tokens"
    assert set(limiter.acquired) == {"rate_limit_wait"}


async def test_agent_path_takes_exactly_one_token_per_request(gemini):
    limiter, requests, _ = gemini
    await llm_mod.ainvoke_with_retry([HumanMessage("hi")], run_name="unit")
    _assert_every_request_throttled(limiter, requests)
    assert requests == [config.GEMINI_MODEL]  # prepaid token is not taken twice


async def test_fallback_model_waits_for_its_own_token(gemini):
    limiter, requests, fail_models = gemini
    fail_models.add(config.GEMINI_MODEL)
    await llm_mod.ainvoke_with_retry([HumanMessage("hi")], run_name="unit")
    assert requests == [config.GEMINI_MODEL, "gemini-fallback-test"]
    _assert_every_request_throttled(limiter, requests)


class Answer(BaseModel):
    answer: str


def lookup(q: str) -> str:
    """Fake tool."""
    return q


@pytest.mark.parametrize("call", ["invoke", "ainvoke", "batch", "bind_tools", "structured", "stream", "astream"])
async def test_direct_model_use_is_throttled(gemini, call):
    """get_primary_llm() is what LangMem receives; libraries may call it in any of these ways."""
    limiter, requests, _ = gemini
    model = llm_mod.get_primary_llm()
    msg = [HumanMessage("hi")]
    if call == "invoke":
        model.invoke(msg)
    elif call == "ainvoke":
        await model.ainvoke(msg)
    elif call == "batch":
        await model.abatch([msg, msg])
    elif call == "bind_tools":
        await model.bind_tools([lookup]).ainvoke(msg)
    elif call == "structured":
        await model.with_structured_output(Answer, method="json_schema").ainvoke(msg)
    elif call == "stream":
        list(model.stream(msg))
    else:
        [c async for c in model.astream(msg)]
    _assert_every_request_throttled(limiter, requests)


async def test_langmem_memory_extraction_is_throttled(gemini):
    pytest.importorskip("langmem")  # without LangMem this path cannot call Gemini at all
    from src.memory.memory_store import extract_memories

    limiter, requests, _ = gemini
    await extract_memories([HumanMessage("Plan discharge.")], {"intent": "full_discharge_plan", "status": "DRAFT"})
    _assert_every_request_throttled(limiter, requests)


def test_deepeval_judge_is_throttled(gemini, monkeypatch):
    pytest.importorskip("deepeval")
    from src.eval_judge import GeminiFallbackJudge

    limiter, requests, _ = gemini

    class FakeResponse:
        text = '{"answer": "ok"}'
        usage_metadata = None

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            requests.append(model)
            return FakeResponse()

    # the judge has its own per-model budget (never the agent's): spy on that limiter
    monkeypatch.setitem(llm_mod._MODEL_LIMITERS, config.GEMINI_JUDGE_MODEL, limiter)
    judge = GeminiFallbackJudge()
    judge.model = type("FakeClient", (), {"models": FakeModels()})()
    judge.generate("judge this", Answer)
    judge.generate("judge this")
    _assert_every_request_throttled(limiter, requests)


def test_no_unthrottled_gemini_client_is_built_elsewhere():
    """Only src/llm.py (rate-limited chat model) and src/eval_judge.py (judge, uses rate_limit_wait) may talk
    to Gemini. A new call site must route through them, or be added here together with a throttling test."""
    pattern = re.compile(r"ChatGoogleGenerativeAI|GoogleGenerativeAI|langchain_google_genai|from google import genai"
                         r"|google\.genai|generate_content\(")
    allowed = {"src/llm.py", "src/eval_judge.py"}
    offenders = [str(p.relative_to(ROOT).as_posix()) for d in ("src", "scripts", "mcp_server")
                 for p in (ROOT / d).rglob("*.py")
                 if pattern.search(p.read_text(encoding="utf-8")) and p.relative_to(ROOT).as_posix() not in allowed]
    assert offenders == []


def test_no_unthrottled_groq_client_is_built_elsewhere():
    """The Groq quota fallback has the same two call sites: src/llm.py (RateLimitedChatGroq, masked + throttled) and
    src/eval_judge.py (Groq judge, masked prompt + rate_limit_wait)."""
    pattern = re.compile(r"ChatGroq|langchain_groq|from groq import|import groq\b|chat\.completions\.create\(")
    allowed = {"src/llm.py", "src/eval_judge.py"}
    offenders = [str(p.relative_to(ROOT).as_posix()) for d in ("src", "scripts", "mcp_server")
                 for p in (ROOT / d).rglob("*.py")
                 if pattern.search(p.read_text(encoding="utf-8")) and p.relative_to(ROOT).as_posix() not in allowed]
    assert offenders == []
