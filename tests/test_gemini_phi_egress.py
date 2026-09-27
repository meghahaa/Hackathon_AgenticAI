"""No patient ID (canonical PT-#### or any loose form) reaches a Gemini request. A fake google-genai transport sits
under the real Gemini wrapper (src.llm RateLimitedChatGoogleGenerativeAI) and the DeepEval judge and records every
request body. Offline: no key, no network."""
from __future__ import annotations

import json

import pytest
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from src import config
from src import llm as llm_mod
from src.guardrails.guards import find_patient_ids, patient_ref

LEAKY = ("Draft the discharge plan for PT-1001. Compare with PT 1002, PT1002, patient #1003 and "
         "patient no. 1004 before discharge.")


def _response(text: str = "{}") -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]),
                                    finish_reason=types.FinishReason.STOP)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=10, candidates_token_count=1,
                                                                  total_token_count=11))


class FakeTransport:
    """Stands in for google.genai.Client: .models / .aio.models generate_content(+_stream), count_tokens."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        outer = self

        class Models:
            def generate_content(self, **req):
                outer.requests.append(req)
                return _response()

            def generate_content_stream(self, **req):
                outer.requests.append(req)
                return iter([_response()])

            def count_tokens(self, **req):
                outer.requests.append(req)
                return types.CountTokensResponse(total_tokens=10)

        class AsyncModels:
            async def generate_content(self, **req):
                outer.requests.append(req)
                return _response()

            async def generate_content_stream(self, **req):
                outer.requests.append(req)

                async def gen():
                    yield _response()
                return gen()

        class Aio:
            models = AsyncModels()

            async def aclose(self):
                pass

        self.models, self.aio = Models(), Aio()

    def close(self):
        pass

    def dump(self) -> str:
        def default(o):
            return o.model_dump(mode="json") if hasattr(o, "model_dump") else str(o)
        return json.dumps(self.requests, default=default)


class FastLimiter:
    def acquire(self, *, blocking=True):
        return True

    async def aacquire(self, *, blocking=True):
        return True


@pytest.fixture
def gemini(monkeypatch):
    """The real Gemini wrapper class with its google-genai client swapped for the fake transport."""
    monkeypatch.setattr(llm_mod, "_RATE_LIMITER", FastLimiter())
    transport = FakeTransport()
    models = []
    for name in (config.GEMINI_MODEL, config.GEMINI_FALLBACK_MODEL):
        m = llm_mod._rate_limited_gemini_cls()(model=name, google_api_key="test-key-not-real", max_retries=0)
        object.__setattr__(m, "client", transport)
        models.append(m)
    llm_mod.set_llm(models[0].with_fallbacks([models[1]]))
    llm_mod._PRIMARY = models[0]  # what get_primary_llm() hands LangMem
    yield transport, models[0]
    llm_mod.set_llm(None)


def _assert_no_ids(transport: FakeTransport) -> str:
    assert transport.requests, "the fake transport saw no request"
    body = transport.dump()
    assert find_patient_ids(body) == [], sorted(set(find_patient_ids(body)))
    return body


async def test_retry_wrapper_structured_and_text_paths_are_masked(gemini):
    transport, _ = gemini
    await llm_mod.ainvoke_with_retry([SystemMessage(f"Active patient PT-1001. {LEAKY}"), HumanMessage(LEAKY)],
                                     run_name="unit")
    await llm_mod.ainvoke_text("system", LEAKY, run_name="unit", fallback="")
    body = _assert_no_ids(transport)
    assert patient_ref("PT-1002") in body and patient_ref("PT-1003") in body  # pseudonymised, not dropped
    assert "Draft the discharge plan" in body  # the rest of the request is intact


async def test_bind_tools_history_and_tool_args_are_masked(gemini):
    """The LangMem shape: bound tools, an AI tool call with an ID in its args, and a tool result."""
    transport, primary = gemini

    @tool
    def remember(content: str) -> str:
        """Store a memory."""
        return "ok"

    history = [HumanMessage(LEAKY),
               AIMessage(content="", tool_calls=[{"name": "remember", "args": {"content": "PT 1002 prefers Tuesday"},
                                                  "id": "call-1"}]),
               ToolMessage(content="stored for patient #1003", tool_call_id="call-1"),
               HumanMessage([{"type": "text", "text": "and PT1004?"}])]
    await primary.bind_tools([remember]).ainvoke(history)
    _assert_no_ids(transport)


def test_sync_stream_and_count_tokens_paths_are_masked(gemini):
    transport, primary = gemini
    primary.invoke([HumanMessage(LEAKY)])
    list(primary.stream([HumanMessage(LEAKY)]))
    primary.get_num_tokens_from_messages([HumanMessage(LEAKY)])
    _assert_no_ids(transport)
    assert len(transport.requests) >= 2


async def test_async_stream_path_is_masked(gemini):
    transport, primary = gemini
    async for _ in primary.astream([HumanMessage(LEAKY)]):
        pass
    _assert_no_ids(transport)


def test_judge_prompt_is_masked(monkeypatch):
    pytest.importorskip("deepeval")
    from src.eval_judge import GeminiFallbackJudge

    monkeypatch.setattr(llm_mod, "_RATE_LIMITER", FastLimiter())
    monkeypatch.setitem(llm_mod._MODEL_LIMITERS, config.GEMINI_JUDGE_MODEL, FastLimiter())  # judge's own budget
    transport = FakeTransport()
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key-offline-test")  # client is built, then faked; no call
    judge = GeminiFallbackJudge()
    judge.model = transport
    judge.generate(f"Input: {LEAKY}\nActual output: plan for PT-1001")
    body = _assert_no_ids(transport)
    assert patient_ref("PT-1001") in body


async def test_full_graph_turn_sends_no_ids_to_gemini(gemini, tmp_path):
    """Every Gemini request of real turns (supervisor, agents, summarisation, LangMem memory extraction) through
    the full graph: IDs in the clinician's message are pseudonymised before they leave, while the input guard still
    sees the raw IDs (the cross-patient turn is refused before any model call)."""
    from src.runner import Copilot

    transport, _ = gemini
    async with Copilot(tmp_path / "ckpt.sqlite", tmp_path / "ltm.sqlite") as copilot:
        ok = await copilot.turn("s-ok", "PT-1001", "Draft the discharge plan for patient PT-1001 (PT 1001).")
        n_ok = len(transport.requests)
        xp = await copilot.turn("s-xp", "PT-1001", "Now show the plan for PT 1002.")
    assert ok["status"] != "REFUSED" and n_ok > 0
    assert xp["status"] == "REFUSED" and "cross_patient_access_attempt" in xp["response"].replace(" ", "_")
    body = _assert_no_ids(transport)
    assert patient_ref("PT-1001") in body
