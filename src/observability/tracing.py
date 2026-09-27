"""Arize Phoenix instrumentation (mandated). Called from the run path by scripts/run_cli.py,
scripts/regenerate_evidence.py and src/api/app.py via `init_tracing()`.

 * Launches Phoenix in-process (UI at http://localhost:$PHOENIX_PORT) with a local SQLite working dir.
 * Builds an OpenTelemetry TracerProvider tagged with the Phoenix project name.
 * Every span passes through `PHIMaskingExporter` before leaving the process, so identifiers in
   prompts/tool payloads are masked in Phoenix and in the committed trace export (NFR-05).
 * A second masked copy of every span is written to state/spans_local.jsonl as a fallback source
   for the exporter if the Phoenix query API is unavailable.
 * openinference LangChainInstrumentor emits LLM / CHAIN / TOOL spans for every node, model and tool call.
 * DeepEval judge spans (copilot.component=eval_judge) go through a second provider tagged with the
   PHOENIX_EVAL_PROJECT project, so the agent project (and its dashboard) holds agent traffic only.
 * `register_model_pricing()` registers per-token prices for the Gemini models (config.PRICING_PER_MTOK)
   through Phoenix's GraphQL createModel/updateModel mutations, so Phoenix computes span cost itself.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from src import config
from src.guardrails.guards import mask_phi

log = logging.getLogger(__name__)
LOCAL_SPANS = config.STATE_DIR / "spans_local.jsonl"


def _mask_value(v: Any) -> Any:
    if isinstance(v, str):
        return mask_phi(v)
    if isinstance(v, (list, tuple)):
        return type(v)(mask_phi(x) if isinstance(x, str) else x for x in v)
    return v


def mask_span(span: ReadableSpan) -> ReadableSpan:
    attrs = {k: _mask_value(v) for k, v in (span.attributes or {}).items()}
    return ReadableSpan(
        name=span.name, context=span.context, parent=span.parent, resource=span.resource, attributes=attrs,
        events=span.events, links=span.links, kind=span.kind, status=span.status, start_time=span.start_time,
        end_time=span.end_time, instrumentation_scope=span.instrumentation_scope,
    )


class PHIMaskingExporter(SpanExporter):
    def __init__(self, inner: SpanExporter) -> None:
        self.inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self.inner.export([mask_span(s) for s in spans])

    def shutdown(self) -> None:
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return getattr(self.inner, "force_flush", lambda *_: True)(timeout_millis)


class LocalJsonlExporter(SpanExporter):
    """Fallback span sink (already-masked spans arrive here via PHIMaskingExporter)."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        LOCAL_SPANS.parent.mkdir(parents=True, exist_ok=True)
        with LOCAL_SPANS.open("a", encoding="utf-8") as fh:
            for s in spans:
                fh.write(json.dumps({
                    "name": s.name,
                    "context.trace_id": format(s.context.trace_id, "032x"),
                    "context.span_id": format(s.context.span_id, "016x"),
                    "parent_id": format(s.parent.span_id, "016x") if s.parent else None,
                    "span_kind": (s.attributes or {}).get("openinference.span.kind", "UNKNOWN"),
                    "start_time": s.start_time, "end_time": s.end_time,
                    "status_code": s.status.status_code.name,
                    "attributes": dict(s.attributes or {}),
                }, default=str) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover
        pass


def _phoenix_up() -> bool:
    import urllib.request

    try:
        return urllib.request.urlopen(f"http://localhost:{config.PHOENIX_PORT}/healthz", timeout=2).status == 200
    except Exception:
        return False


def use_utf8_output() -> None:
    """Windows: Phoenix prints non-ASCII characters at launch, which a cp1252 console/redirected stdout cannot
    encode ('charmap' codec can't encode...), so the in-process launch failed. PYTHONUTF8=1 fixes child interpreters
    (`phoenix serve`, the regenerate script's subprocesses); this process is already running, so its stdout/stderr
    are switched to UTF-8 in place."""
    import sys

    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        if (getattr(stream, "encoding", "") or "").lower().replace("-", "") != "utf8" and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _start_phoenix() -> None:
    """Launch Phoenix in-process; if that fails (slow first-run migrations on Windows), run
    `python -m phoenix.server.main serve` as a local child process and wait for /healthz."""
    import subprocess
    import sys
    import time

    use_utf8_output()
    try:
        import phoenix as px

        px.launch_app(port=config.PHOENIX_PORT, use_temp_dir=False)
        if _phoenix_up():
            return
    except Exception as exc:
        log.warning("in-process Phoenix launch failed (%s); starting `phoenix serve`", exc)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    logf = open(config.STATE_DIR / "phoenix_server.log", "a", encoding="utf-8")
    subprocess.Popen([sys.executable, "-m", "phoenix.server.main", "serve"], env=env, stdout=logf,
                     stderr=subprocess.STDOUT, cwd=str(config.ROOT))
    for _ in range(90):
        if _phoenix_up():
            return
        time.sleep(1)
    log.error("Phoenix did not come up; spans will still be kept in %s", LOCAL_SPANS)


def ensure_phoenix(launch: bool = True) -> bool:
    """Phoenix reachable on PHOENIX_PORT (/healthz)? Launches it first (same way as init_tracing) if asked."""
    os.environ.setdefault("PHOENIX_WORKING_DIR", config.PHOENIX_WORKING_DIR)
    os.environ.setdefault("PHOENIX_PORT", str(config.PHOENIX_PORT))
    if launch and not _phoenix_up():
        _start_phoenix()
    return _phoenix_up()


def otlp_accepts_spans() -> tuple[bool, str]:
    """POST an empty OTLP/HTTP export (a valid protobuf with no spans, so nothing is stored) to /v1/traces."""
    import urllib.request

    url = f"http://localhost:{config.PHOENIX_PORT}/v1/traces"
    req = urllib.request.Request(url, data=b"", method="POST", headers={"Content-Type": "application/x-protobuf"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 202), f"{url} -> HTTP {resp.status}"
    except Exception as exc:
        return False, f"{url} -> {type(exc).__name__}: {exc}"


_GQL_MODELS ="query { generativeModels(first: 500) { edges { node { id name kind } } } }"
_GQL_UPSERT = """mutation($input: %sModelMutationInput!) { %sModel(input: $input) { model { id name } } }"""


def _graphql(query: str, variables: Optional[dict] = None) -> dict:
    import urllib.request

    req = urllib.request.Request(f"http://localhost:{config.PHOENIX_PORT}/graphql", method="POST",
                                 data=json.dumps({"query": query, "variables": variables or {}}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read())
    if body.get("errors"):
        raise RuntimeError(body["errors"][0].get("message"))
    return body["data"]


def register_model_pricing() -> dict[str, str]:
    """Create/update a custom Phoenix model (Settings -> Models) with input/output USD per 1M tokens for
    GEMINI_MODEL, GEMINI_FALLBACK_MODEL and GEMINI_JUDGE_MODEL (plus GROQ_MODEL / GROQ_JUDGE_MODEL when the Groq quota
    fallback is enabled). Phoenix prices spans at ingest, so this must run before traffic.
    Returns {model: "created" | "updated" | "error: ..."}; never raises."""
    groq = (config.GROQ_MODEL, config.GROQ_JUDGE_MODEL) if config.groq_enabled() else ()
    names = dict.fromkeys(m for m in (config.GEMINI_MODEL, config.GEMINI_FALLBACK_MODEL, config.GEMINI_JUDGE_MODEL,
                                      *groq) if m)
    try:
        existing = {e["node"]["name"]: e["node"]["id"] for e in _graphql(_GQL_MODELS)["generativeModels"]["edges"]
                    if e["node"]["kind"] == "CUSTOM"}
    except Exception as exc:
        log.warning("Phoenix model pricing not registered: %s", exc)
        return {m: f"error: {exc}" for m in names}
    result = {}
    for model in names:
        price = config.PRICING_PER_MTOK[model]
        payload = {"name": model, "provider": "groq" if config.provider_of(model) == "groq" else "google",
                   "namePattern": rf"(^|/){re.escape(model)}$",  # "gemini-x" or "models/gemini-x"
                   "costs": [{"tokenType": "input", "kind": "PROMPT", "costPerMillionTokens": price["input"]},
                             {"tokenType": "output", "kind": "COMPLETION", "costPerMillionTokens": price["output"]}]}
        op = "update" if model in existing else "create"
        if op == "update":
            payload["id"] = existing[model]
        try:
            _graphql(_GQL_UPSERT % (op.capitalize(), op), {"input": payload})
            result[model] = op + "d"
        except Exception as exc:
            log.warning("Phoenix pricing for %s failed: %s", model, exc)
            result[model] = f"error: {exc}"
    return result


@dataclass
class TracingHandle:
    provider: TracerProvider
    phoenix_url: Optional[str]
    project: str
    eval_provider: Optional[TracerProvider] = None
    eval_project: Optional[str] = None
    pricing: Optional[dict] = None

    def flush(self) -> None:
        self.provider.force_flush()
        if self.eval_provider is not None:
            self.eval_provider.force_flush()


_HANDLE: Optional[TracingHandle] = None


def init_tracing(launch_phoenix: bool = True) -> TracingHandle:
    """Idempotent. Launch Phoenix (in-process), wire the masked OTLP exporter and instrument LangChain."""
    global _HANDLE
    if _HANDLE is not None:
        return _HANDLE
    import sys

    for stream in (sys.stdout, sys.stderr):  # Phoenix prints emoji; Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    os.environ.setdefault("PHOENIX_WORKING_DIR", config.PHOENIX_WORKING_DIR)
    os.environ.setdefault("PHOENIX_PORT", str(config.PHOENIX_PORT))
    url = f"http://localhost:{config.PHOENIX_PORT}/"
    if launch_phoenix and not _phoenix_up():
        _start_phoenix()
    pricing = register_model_pricing() if _phoenix_up() else None

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    def make_provider(project: str) -> TracerProvider:
        p = TracerProvider(resource=Resource.create({
            "openinference.project.name": project, "service.name": "discharge-copilot"}))
        p.add_span_processor(SimpleSpanProcessor(PHIMaskingExporter(
            OTLPSpanExporter(endpoint=f"http://localhost:{config.PHOENIX_PORT}/v1/traces"))))
        p.add_span_processor(SimpleSpanProcessor(PHIMaskingExporter(LocalJsonlExporter())))
        return p

    provider = make_provider(config.PHOENIX_PROJECT)
    trace_api.set_tracer_provider(provider)

    from openinference.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument(tracer_provider=provider)
    _HANDLE = TracingHandle(provider=provider, phoenix_url=url, project=config.PHOENIX_PROJECT,
                            eval_provider=make_provider(config.PHOENIX_EVAL_PROJECT),
                            eval_project=config.PHOENIX_EVAL_PROJECT, pricing=pricing)
    log.info("Phoenix tracing active: %s (project=%s, judge project=%s, pricing=%s)", url, config.PHOENIX_PROJECT,
             config.PHOENIX_EVAL_PROJECT, pricing)
    return _HANDLE


def get_tracer():
    return trace_api.get_tracer("discharge-copilot")


def eval_tracer():
    """Tracer for DeepEval judge spans (PHOENIX_EVAL_PROJECT), or None before init_tracing()."""
    if _HANDLE is None or _HANDLE.eval_provider is None:
        return None
    return _HANDLE.eval_provider.get_tracer("discharge-copilot")


def fetch_spans_dataframe(project: Optional[str] = None, local_fallback: bool = True):
    """Pull one project's spans from the Phoenix server (spans.get_spans_dataframe); fall back to the local masked
    copy (which holds agent AND judge spans; the exporter separates them by copilot.component)."""
    import pandas as pd

    project = project or config.PHOENIX_PROJECT
    endpoint = f"http://localhost:{config.PHOENIX_PORT}"
    try:  # Phoenix >= 12: phoenix.client
        from phoenix.client import Client

        df = Client(base_url=endpoint).spans.get_spans_dataframe(project_identifier=project,
                                                                 limit=100000, timeout=120)
        if df is not None and len(df):
            return df.reset_index(drop=True), "phoenix.client.Client().spans.get_spans_dataframe"
    except Exception as exc:
        log.warning("phoenix.client spans query failed: %s", exc)
    try:  # older Phoenix: px.Client().get_spans_dataframe()
        import phoenix as px

        df = px.Client(endpoint=endpoint).get_spans_dataframe(project_name=project)
        if df is not None and len(df):
            return df.reset_index(drop=True), "phoenix.Client().get_spans_dataframe"
    except Exception as exc:
        log.warning("px.Client().get_spans_dataframe unavailable: %s", exc)
    if local_fallback and LOCAL_SPANS.exists():
        rows = [json.loads(l) for l in LOCAL_SPANS.read_text(encoding="utf-8").splitlines() if l.strip()]
        return pd.DataFrame(rows), "local-masked-otel-copy"
    return pd.DataFrame(), "none"
