"""Gemini model access with timeouts, retries and graceful degradation (NFR-04).

Google Gemini is the model provider. If no key is configured and ALLOW_OFFLINE_LLM=1,
a deterministic offline stub is used so the pipeline degrades gracefully instead of crashing;
every node then falls back to its rule-based path. Stub output is labelled "offline-stub".

Request routing (ModelChain), per ainvoke_with_retry attempt:
  1. GEMINI_MODEL; on a stall (request deadline) or 504 DEADLINE_EXCEEDED it is sent once more;
  2. GEMINI_FALLBACK_MODEL;
  3. only if QUOTA_FALLBACK_PROVIDER=groq, the primary has failed (after its one retry when it stalled) AND the
     Gemini fallback then returned 429 (quota), 503 (overloaded) or 404 (model unavailable): GROQ_MODEL. Never when
     the Gemini fallback succeeds, never on a first stall alone, never after a fallback timeout / other error. This
     deviates from the brief's Gemini-only rule, is off by default, and every Groq request is labelled provider=groq
     (span metadata + quota_fallback span, audit record, run results). Groq requests go through the same limiter, the
     same mask_phi and the same tracing as Gemini.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import json
import logging
import re
import time
from typing import Any, Optional, Type, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable, RunnableWithFallbacks
from opentelemetry import trace
from pydantic import BaseModel, ValidationError

from src import config

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class OfflineStubChatModel(BaseChatModel):
    """Deterministic stand-in used only when Gemini is unavailable. Returns an empty JSON
    object so callers take their deterministic fallback path. Never used when a key is set."""

    model_name: str = "offline-stub"

    @property
    def _llm_type(self) -> str:
        return "offline-stub"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt_chars = sum(len(str(m.content)) for m in messages)
        msg = AIMessage(
            content="{}",
            usage_metadata={"input_tokens": prompt_chars // 4, "output_tokens": 1, "total_tokens": prompt_chars // 4 + 1},
            response_metadata={"model_name": self.model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


_LLM: Optional[BaseChatModel] = None
_PRIMARY: Optional[BaseChatModel] = None
# cost/latency governance: ONE client-side request budget (GEMINI_RPM) shared by every agent Gemini request (agent
# nodes, fallback model, LangMem extraction) so quota errors are avoided. It is always acquired inside its own
# "rate_limit_wait" span, before the model call starts, so throttling is not counted as model latency.
_RATE_LIMITER = InMemoryRateLimiter(requests_per_second=config.GEMINI_RPM / 60.0,
                                    check_every_n_seconds=0.1, max_bucket_size=1)
# the DeepEval judge never uses an agent model, so each judge model has its own budget (GEMINI_JUDGE_RPM, Gemini quota
# is per model) and judge traffic does not slow the agent
_MODEL_LIMITERS: dict[str, InMemoryRateLimiter] = {}


def model_limiter(model: str) -> InMemoryRateLimiter:
    """The per-model limiter for a judge model (created on first use)."""
    if model not in _MODEL_LIMITERS:
        _MODEL_LIMITERS[model] = InMemoryRateLimiter(requests_per_second=config.GEMINI_JUDGE_RPM / 60.0,
                                                     check_every_n_seconds=0.1, max_bucket_size=1)
    return _MODEL_LIMITERS[model]
# set by ainvoke_with_retry: one token already taken for this attempt, consumed by the first Gemini request so the
# primary model is not throttled twice; a fallback model in the same attempt finds it spent and waits for its own.
_PREPAID: contextvars.ContextVar[Optional[list[bool]]] = contextvars.ContextVar("rate_limit_prepaid", default=None)


class _ModelClock:
    """Model time of the current request in an ainvoke_with_retry attempt: rate-limit token waits are excluded, and
    every Gemini request restarts it, so a fallback model gets a fresh budget instead of what is left of the
    primary's. It backs up the per-request GEMINI_REQUEST_TIMEOUT_S deadline (and times non-Gemini models)."""

    def __init__(self) -> None:
        self.start = time.monotonic()
        self.waited = 0.0
        self.wait_start: Optional[float] = None

    def restart(self) -> None:
        self.start, self.waited, self.wait_start = time.monotonic(), 0.0, None

    def pause(self) -> None:
        self.wait_start = time.monotonic()

    def resume(self) -> None:
        if self.wait_start is not None:
            self.waited += time.monotonic() - self.wait_start
            self.wait_start = None

    def model_elapsed(self) -> float:
        now = time.monotonic()
        return now - self.start - self.waited - (now - self.wait_start if self.wait_start is not None else 0.0)


_CLOCK: contextvars.ContextVar[Optional[_ModelClock]] = contextvars.ContextVar("model_clock", default=None)


async def _await_model(aw: Any, budget_s: float) -> Any:
    """Like asyncio.wait_for, but rate-limit token waits inside `aw` do not count against `budget_s`."""
    clock = _ModelClock()
    token = _CLOCK.set(clock)
    try:
        task = asyncio.ensure_future(aw)  # the task copies the context, so its limiter waits see this clock
    finally:
        _CLOCK.reset(token)
    try:
        while True:
            remaining = budget_s - clock.model_elapsed()
            if remaining <= 0:
                raise asyncio.TimeoutError(f"model request exceeded {budget_s}s (rate-limit waits excluded)")
            done, _ = await asyncio.wait({task}, timeout=remaining)
            if done:
                return task.result()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.wait({task})
        if not task.cancelled():
            task.exception()  # retrieved, so a cancelled/failed request never logs "never retrieved"


def _wait_span(run_name: str, attempt: int, throttled: bool, component: str, provider: str = "gemini"):
    # judge spans go to the separate Phoenix eval project once tracing is up; copilot.component stays as the
    # safety net that lets scripts/export_traces.py keep evaluation traffic out of the agent's golden signals
    tracer, parent = None, None
    if component == "eval_judge":  # src.eval_judge.JUDGE_COMPONENT
        from opentelemetry.context import Context

        from src.observability.tracing import eval_tracer

        tracer, parent = eval_tracer(), Context()  # root span: never parented into an agent-project trace
    tracer = tracer or trace.get_tracer("discharge-copilot")
    return tracer.start_as_current_span("rate_limit_wait", context=parent, attributes={
        "openinference.span.kind": "CHAIN", "copilot.span_role": "rate_limit_wait", "copilot.component": component,
        "copilot.run_name": run_name, "copilot.attempt": attempt, "copilot.rate_limited": throttled,
        "copilot.provider": provider,
    })


async def arate_limit_wait(run_name: str, attempt: int = 0, throttled: bool = True, component: str = "agent",
                           provider: str = "gemini") -> None:
    """Take a token from the shared request limiter inside a rate_limit_wait span."""
    clock = _CLOCK.get()
    with _wait_span(run_name, attempt, throttled, component, provider):
        if throttled:
            if clock:
                clock.pause()
            try:
                await _RATE_LIMITER.aacquire()
            finally:
                if clock:
                    clock.resume()


def rate_limit_wait(run_name: str, attempt: int = 0, component: str = "agent",
                    limiter: Optional[InMemoryRateLimiter] = None, provider: str = "gemini") -> None:
    """Blocking variant for synchronous callers (e.g. the DeepEval judge running in a worker thread, which passes
    its own per-model limiter)."""
    clock = _CLOCK.get()
    with _wait_span(run_name, attempt, True, component, provider):
        if clock:
            clock.pause()
        try:
            (limiter or _RATE_LIMITER).acquire()
        finally:
            if clock:
                clock.resume()


def _take_prepaid() -> bool:
    prepaid = _PREPAID.get()
    if prepaid and prepaid[0]:
        prepaid[0] = False
        return True
    return False


def model_name() -> str:
    return config.GEMINI_MODEL if config.GEMINI_API_KEY else "offline-stub"


def _mask_values(obj: Any) -> Any:
    """mask_phi on every string value (keys untouched, so tool-call args keep their schema)."""
    from src.guardrails.guards import mask_phi

    if isinstance(obj, str):
        return mask_phi(obj)
    if isinstance(obj, dict):
        return {k: _mask_values(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_mask_values(v) for v in obj)
    return obj


def _mask_content(content: Any) -> Any:
    # text only: other part fields (e.g. base64 thought signatures) must reach Gemini byte-for-byte
    if isinstance(content, str):
        return _mask_values(content)
    if isinstance(content, list):
        return [_mask_values(p) if isinstance(p, str)
                else {**p, "text": _mask_values(p["text"])} if isinstance(p, dict) and isinstance(p.get("text"), str)
                else p for p in content]
    return content


def mask_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Copies of `messages` with PHI masked in text content and in tool-call arguments."""
    out = []
    for m in messages:
        upd: dict[str, Any] = {}
        content = _mask_content(m.content)
        if content != m.content:
            upd["content"] = content
        if getattr(m, "tool_calls", None):
            upd["tool_calls"] = [{**tc, "args": _mask_values(tc.get("args", {}))} for tc in m.tool_calls]
        fc = m.additional_kwargs.get("function_call")
        if isinstance(fc, dict) and isinstance(fc.get("arguments"), str):
            upd["additional_kwargs"] = {**m.additional_kwargs, "function_call": {**fc, "arguments": _mask_values(fc["arguments"])}}
        out.append(m.model_copy(update=upd) if upd else m)
    return out


class _ThrottledRequests:
    """Mixin for every chat model that sends requests to a provider (Gemini, and Groq when the quota fallback is on).

    Rate limiting: a token from the shared limiter before EVERY request, however the model is called
    (ainvoke_with_retry, ModelChain, bind_tools/with_structured_output as used by LangMem, batch, stream).
    invoke/ainvoke/batch all funnel through generate/agenerate, which run before the LLM callback span opens, so the
    wait stays in its own rate_limit_wait span (labelled with the provider).

    Deadline: every request gets its own GEMINI_REQUEST_TIMEOUT_S, counted from after its token wait. It is enforced
    inside _agenerate, so a stalled request fails like any request error (the LLM span closes with ERROR) and the
    ModelChain then decides what is sent next, with a full deadline of its own."""

    def _request_model(self) -> str:
        return getattr(self, "model", None) or getattr(self, "model_name", "")

    def _request_provider(self) -> str:
        return config.provider_of(self._request_model())

    def _wait_label(self) -> str:
        return f"{self._request_provider()}:{self._request_model()}"

    def generate(self, *args: Any, **kwargs: Any):
        if not _take_prepaid():
            rate_limit_wait(self._wait_label(), provider=self._request_provider())
        return super().generate(*args, **kwargs)

    async def agenerate(self, *args: Any, **kwargs: Any):
        if not _take_prepaid():
            await arate_limit_wait(self._wait_label(), provider=self._request_provider())
        clock = _CLOCK.get()
        if clock:
            clock.restart()  # fresh budget for this request; an earlier request's model time is not carried over
        return await super().agenerate(*args, **kwargs)

    async def _agenerate(self, *args: Any, **kwargs: Any):
        return await asyncio.wait_for(super()._agenerate(*args, **kwargs), timeout=config.GEMINI_REQUEST_TIMEOUT_S)

    def stream(self, input: Any, config: Any = None, *, stop: Any = None, **kwargs: Any):
        # a non-streaming model goes through invoke -> generate, which already waits
        if self._should_stream(async_api=False, **{**kwargs, "stream": True}) and not _take_prepaid():
            rate_limit_wait(self._wait_label(), provider=self._request_provider())
        yield from super().stream(input, config, stop=stop, **kwargs)

    async def astream(self, input: Any, config: Any = None, *, stop: Any = None, **kwargs: Any):
        if self._should_stream(async_api=True, **{**kwargs, "stream": True}) and not _take_prepaid():
            await arate_limit_wait(self._wait_label(), provider=self._request_provider())
        async for chunk in super().astream(input, config, stop=stop, **kwargs):
            yield chunk


@functools.cache
def _rate_limited_gemini_cls() -> type[BaseChatModel]:
    from langchain_google_genai import ChatGoogleGenerativeAI

    class RateLimitedChatGoogleGenerativeAI(_ThrottledRequests, ChatGoogleGenerativeAI):
        """Gemini chat model: throttled and deadline-bound (_ThrottledRequests).

        It is also the single point where PHI leaves for Gemini: every generate request (_generate, _agenerate,
        _stream, _astream) is built by _prepare_request, and the count-tokens request by get_num_tokens; both mask
        all text first (mask_phi: canonical + loose patient IDs -> pref_ pseudonyms, registry identifiers, PHI
        patterns). The input guard has already run on the raw text, so cross-patient detection still sees the real
        IDs."""

        def _prepare_request(self, messages: list[BaseMessage], *args: Any, **kwargs: Any):
            return super()._prepare_request(mask_messages(messages), *args, **kwargs)

        def get_num_tokens(self, text: str) -> int:
            return super().get_num_tokens(_mask_values(text))

    return RateLimitedChatGoogleGenerativeAI


@functools.cache
def _rate_limited_groq_cls() -> type[BaseChatModel]:
    from langchain_groq import ChatGroq

    class RateLimitedChatGroq(_ThrottledRequests, ChatGroq):
        """Groq chat model for the quota fallback only (QUOTA_FALLBACK_PROVIDER=groq). Same limiter, same deadline and
        same masking as Gemini: every request path (_generate, _agenerate, _stream, _astream) and the token count
        mask all text and tool-call arguments with mask_phi before anything is sent to Groq."""

        def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any):
            return super()._generate(mask_messages(messages), *args, **kwargs)

        async def _agenerate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any):
            return await super()._agenerate(mask_messages(messages), *args, **kwargs)

        def _stream(self, messages: list[BaseMessage], *args: Any, **kwargs: Any):
            yield from super()._stream(mask_messages(messages), *args, **kwargs)

        async def _astream(self, messages: list[BaseMessage], *args: Any, **kwargs: Any):
            async for chunk in super()._astream(mask_messages(messages), *args, **kwargs):
                yield chunk

        def get_num_tokens(self, text: str) -> int:
            return super().get_num_tokens(_mask_values(text))

    return RateLimitedChatGroq


def _gemini(model: str) -> BaseChatModel:
    return _rate_limited_gemini_cls()(
        model=model,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.0,
        max_retries=0,  # retries are handled by ainvoke_with_retry so they are visible in traces
        timeout=config.GEMINI_REQUEST_TIMEOUT_S,  # also sent as the API deadline
    )


def _groq(model: str) -> BaseChatModel:
    return _rate_limited_groq_cls()(
        model=model,
        api_key=config.GROQ_API_KEY,
        temperature=0.0,
        max_retries=0,
        timeout=config.GEMINI_REQUEST_TIMEOUT_S,  # same per-request deadline as Gemini
    )


_STALL_RE = re.compile(r"\b504\b|DEADLINE_EXCEEDED")
_QUOTA_RE = re.compile(r"\b429\b|RESOURCE_EXHAUSTED")


def is_stall(exc: BaseException) -> bool:
    """The request hit its deadline (GEMINI_REQUEST_TIMEOUT_S) or Gemini answered 504 DEADLINE_EXCEEDED."""
    return isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or bool(_STALL_RE.search(str(exc)))


def is_quota(exc: BaseException) -> bool:
    """Quota error: 429 RESOURCE_EXHAUSTED (e.g. the Gemini free tier's requests-per-day limit)."""
    return bool(_QUOTA_RE.search(str(exc)))


_GROQ_TRIGGERS = (("quota", _QUOTA_RE), ("overloaded", re.compile(r"\b503\b|UNAVAILABLE")),
                  ("unavailable", re.compile(r"\b404\b|NOT_FOUND")))


def groq_trigger(exc: BaseException) -> Optional[str]:
    """Why a Gemini fallback error hands the request to Groq: "quota" (429), "overloaded" (503) or "unavailable"
    (404); None for anything else (timeouts / 504 stalls, 5xx other than 503, client errors), which never go to Groq."""
    return next((kind for kind, rx in _GROQ_TRIGGERS if rx.search(str(exc))), None)


class ModelChain(Runnable):
    """The agent's model routing: GEMINI_MODEL (sent once more after a stall / 504), then GEMINI_FALLBACK_MODEL, then -
    only when QUOTA_FALLBACK_PROVIDER=groq and the Gemini fallback returned 429 / 503 / 404 (groq_trigger) after the
    primary had failed - GROQ_MODEL. Every request is
    recorded in the run (src.observability.run_context.MODEL_CALLS) with its provider, its LLM span carries
    metadata provider/model/route, and a Groq request also opens a quota_fallback span and writes an audit record.
    The answering provider/model are added to the reply's response_metadata (copilot_provider / copilot_model)."""

    def __init__(self, members: list[tuple[str, str, str, Optional[Runnable]]]) -> None:
        # (route, provider, model, runnable); route in {"primary", "fallback", "quota_fallback"}
        self.members = [m for m in members if m[3] is not None]

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ModelChain":
        return ModelChain([(r, p, m, llm.bind_tools(tools, **kwargs)) for r, p, m, llm in self.members])

    def _member(self, route: str):
        return next((m for m in self.members if m[0] == route), None)

    def _next(self, history: list[tuple[str, BaseException]]):
        """What to send after `history` (routes tried so far, with their errors), or None to give up."""
        if not history:
            return self._member("primary")
        route, exc = history[-1]
        if route == "primary":
            if is_stall(exc) and sum(r == "primary" for r, _ in history) == 1:
                return self._member("primary")  # one more try of the primary after a stall / 504
            return self._member("fallback")
        if route == "fallback" and config.groq_enabled() and groq_trigger(exc):
            # the primary has already failed (after its one retry if it stalled): the fallback is the last Gemini try
            return self._member("quota_fallback")
        return None

    @staticmethod
    def _config(config_: Any, route: str, provider: str, model: str) -> dict:
        cfg = dict(config_ or {})
        cfg["metadata"] = {**(cfg.get("metadata") or {}), "provider": provider, "model": model, "route": route}
        return cfg

    @staticmethod
    def _tag(msg: Any, provider: str, model: str) -> Any:
        if isinstance(msg, BaseMessage):
            msg.response_metadata = {**(msg.response_metadata or {}), "copilot_provider": provider,
                                     "copilot_model": model}
        return msg

    @staticmethod
    def _record(route: str, provider: str, model: str, run_name: str, exc: Optional[BaseException]) -> None:
        from src.guardrails.guards import mask_phi
        from src.observability.run_context import record_model_call

        record_model_call(route=route, provider=provider, model=model, run_name=run_name, ok=exc is None,
                          error=None if exc is None else mask_phi(f"{type(exc).__name__}: {str(exc)[:160]}"))

    @staticmethod
    def _groq_span(model: str, run_name: str, history: list[tuple[str, BaseException]]):
        """quota_fallback span + audit record for a Groq request (the disclosed deviation from Gemini-only)."""
        from src.guardrails.audit import audit

        reason = f"primary failed; Gemini fallback {groq_trigger(history[-1][1])}"
        audit("llm_router", "quota_fallback", "groq", reason=reason, details={
            "trigger": groq_trigger(history[-1][1]),
            "provider": "groq", "model": model, "run_name": run_name,
            "gemini_errors": [f"{r}: {type(e).__name__}" for r, e in history]})
        return trace.get_tracer("discharge-copilot").start_as_current_span("quota_fallback", attributes={
            "openinference.span.kind": "CHAIN", "copilot.span_role": "quota_fallback", "copilot.provider": "groq",
            "copilot.model": model, "copilot.run_name": run_name, "copilot.reason": reason,
            "copilot.trigger": groq_trigger(history[-1][1]) or ""})

    def _step(self, history: list[tuple[str, BaseException]], run_name: str):
        step = self._next(history)
        if step is None:
            return None, None
        _, provider, model, _ = step
        cm = self._groq_span(model, run_name, history) if provider == "groq" else contextlib.nullcontext()
        return step, cm

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        history: list[tuple[str, BaseException]] = []
        run_name = (config or {}).get("run_name", "")
        while True:
            step, cm = self._step(history, run_name)
            if step is None:
                raise history[-1][1]
            route, provider, model, llm = step
            try:
                with cm:
                    msg = llm.invoke(input, self._config(config, route, provider, model), **kwargs)
            except Exception as exc:
                self._record(route, provider, model, run_name, exc)
                history.append((route, exc))
                continue
            self._record(route, provider, model, run_name, None)
            return self._tag(msg, provider, model)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        history: list[tuple[str, BaseException]] = []
        run_name = (config or {}).get("run_name", "")
        while True:
            step, cm = self._step(history, run_name)
            if step is None:
                raise history[-1][1]
            route, provider, model, llm = step
            try:
                with cm:
                    msg = await llm.ainvoke(input, self._config(config, route, provider, model), **kwargs)
            except Exception as exc:
                self._record(route, provider, model, run_name, exc)
                history.append((route, exc))
                continue
            self._record(route, provider, model, run_name, None)
            return self._tag(msg, provider, model)


def get_llm() -> Runnable:
    """The agent's ModelChain: Gemini primary (one more try after a stall / 504), Gemini fallback, and the Groq quota
    fallback only when QUOTA_FALLBACK_PROVIDER=groq (with GROQ_API_KEY). Offline stub only when no key is configured."""
    global _LLM, _PRIMARY
    if _LLM is not None:
        return _LLM
    if config.GEMINI_API_KEY:
        _PRIMARY = _gemini(config.GEMINI_MODEL)
        fallback = config.GEMINI_FALLBACK_MODEL if config.GEMINI_FALLBACK_MODEL != config.GEMINI_MODEL else ""
        _LLM = ModelChain([
            ("primary", "gemini", config.GEMINI_MODEL, _PRIMARY),
            ("fallback", "gemini", fallback, _gemini(fallback) if fallback else None),
            ("quota_fallback", "groq", config.GROQ_MODEL, _groq(config.GROQ_MODEL) if config.groq_enabled() else None),
        ])
    elif config.ALLOW_OFFLINE_LLM:
        log.warning("GEMINI_API_KEY not set - using deterministic offline stub (degraded mode)")
        _LLM = OfflineStubChatModel()
    else:
        raise RuntimeError("GEMINI_API_KEY is required (set ALLOW_OFFLINE_LLM=1 to run degraded).")
    return _LLM


def get_primary_llm() -> BaseChatModel:
    """The primary chat model object (needed by libraries that call bind_tools, e.g. LangMem). It throttles
    itself on the shared limiter, so direct use outside ainvoke_with_retry is still rate-limited. LangMem never gets
    the Groq quota fallback: on any failure it degrades to the deterministic extractor."""
    get_llm()
    return _PRIMARY or _LLM


def get_tool_llm(tools: list[Any]) -> Optional[Runnable]:
    """The model chain with `tools` bound (every member, Groq included when enabled), for bounded tool-calling loops.
    None when the offline stub (or a model without tool calling) is active, so callers take their deterministic path."""
    llm = get_llm()
    if isinstance(llm, OfflineStubChatModel):
        return None
    try:
        if isinstance(llm, RunnableWithFallbacks):
            return llm.runnable.bind_tools(tools).with_fallbacks([f.bind_tools(tools) for f in llm.fallbacks])
        return llm.bind_tools(tools)
    except (NotImplementedError, AttributeError):
        return None


def set_llm(llm: Optional[BaseChatModel]) -> None:
    """Test hook: inject a model (or reset with None)."""
    global _LLM, _PRIMARY
    _LLM, _PRIMARY = llm, None


async def ainvoke_with_retry(messages: list[BaseMessage], *, run_name: str, llm: Runnable | None = None) -> AIMessage:
    """Invoke the LLM with a timeout and exponential-backoff retries. Raises on final failure."""
    llm = llm or get_llm()
    throttled = not isinstance(llm, OfflineStubChatModel)  # the offline stub makes no API requests
    last: Exception | None = None
    for attempt in range(config.LLM_MAX_RETRIES + 1):
        await arate_limit_wait(run_name, attempt, throttled)
        prepaid = _PREPAID.set([throttled])
        try:
            # backstop per model request (restarted for a fallback); rate-limit token waits are excluded
            return await _await_model(
                llm.ainvoke(messages, config={"run_name": run_name, "metadata": {"attempt": attempt}}),
                config.LLM_TIMEOUT_S,
            )
        except Exception as exc:  # timeout, quota, transport
            last = exc
            log.warning("LLM call %s failed (attempt %d): %s", run_name, attempt + 1, type(exc).__name__)
            if attempt < config.LLM_MAX_RETRIES:
                quota = any(t in str(exc) for t in ("429", "RESOURCE_EXHAUSTED", "quota"))
                await asyncio.sleep((20.0 if quota else 1.5) * (2**attempt))
        finally:
            _PREPAID.reset(prepaid)
    raise RuntimeError(f"LLM call {run_name} failed after retries: {type(last).__name__}: {last}") from last


def answered_by(msg: Any) -> dict[str, str]:
    """{"model", "provider"} of the model that produced `msg` (tagged by ModelChain), else the configured model."""
    meta = getattr(msg, "response_metadata", None) or {}
    model = meta.get("copilot_model") or model_name()
    return {"model": model, "provider": meta.get("copilot_provider") or config.provider_of(model)}


def _content_text(msg: AIMessage) -> str:
    c = msg.content
    if isinstance(c, list):  # Gemini may return content parts
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


async def ainvoke_structured(
    schema: Type[T],
    system: str,
    user: str,
    *,
    run_name: str,
    fallback: T,
) -> tuple[T, dict[str, Any]]:
    """Ask the model for JSON matching `schema`; validate with pydantic at the node boundary.

    Returns (validated_object, meta). On any failure returns `fallback` with meta.degraded=True
    so the graph keeps going (graceful degradation) and the failure is observable.
    """
    sys_prompt = (
        f"{system}\n\nRespond with ONLY a JSON object matching this JSON schema:\n"
        f"{json.dumps(schema.model_json_schema())}"
    )
    try:
        msg = await ainvoke_with_retry([SystemMessage(sys_prompt), HumanMessage(user)], run_name=run_name)
        data = extract_json(_content_text(msg))
        if not data:
            raise ValueError("empty JSON from model")
        return schema.model_validate(data), {"degraded": False, **answered_by(msg)}
    except (ValidationError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        return fallback, {"degraded": True, "model": model_name(), "error": f"{type(exc).__name__}: {str(exc)[:300]}"}


async def ainvoke_text(system: str, user: str, *, run_name: str, fallback: str) -> tuple[str, dict[str, Any]]:
    try:
        msg = await ainvoke_with_retry([SystemMessage(system), HumanMessage(user)], run_name=run_name)
        text = _content_text(msg).strip()
        if not text or text == "{}":
            raise ValueError("empty model output")
        return text, {"degraded": False, **answered_by(msg)}
    except (ValueError, RuntimeError) as exc:
        return fallback, {"degraded": True, "model": model_name(), "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
