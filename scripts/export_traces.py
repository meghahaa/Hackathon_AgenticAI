"""Export Phoenix spans and derive the golden-signals report + cost/latency dashboard.

Phoenix projects
  PHOENIX_PROJECT_NAME              agent traffic: the screenshots and the agent golden signals come from here
  PHOENIX_PROJECT_NAME + "-eval"    DeepEval judge spans (config.PHOENIX_EVAL_PROJECT): the "eval_judge" block

Outputs
  traces/phoenix_spans.parquet      px.Client().get_spans_dataframe() export of the agent project (PHI-masked at source)
  traces/phoenix_spans.jsonl        same spans, line-delimited (easy to grep span ids)
  traces/phoenix_eval_spans.parquet judge spans from the eval project (when there are any)
  reports/dashboard_data.csv        per-span latency / token / cost rows (agent + judge, "component" column;
                                    "provider" column: gemini / groq for model-request spans)
  reports/golden_signals.json       p50/p95 latency by span type (thinking = model only / rate_limit_wait /
                                    thinking_including_rate_limit / acting / tool / end-to-end),
                                    agent spans only; DeepEval judge spans (copilot.component=eval_judge)
                                    are reported separately in the "eval_judge" block,
                                    tokens in/out, cost estimate, error rates, budgets, accuracy +
                                    hallucination rate imported from reports/eval_report.json
  reports/dashboard_chart.png       chart rendered from dashboard_data.csv
  reports/dashboard.png             screenshot of the live Phoenix project Metrics view (latency, tokens, cost;
                                    headless Edge/Chrome) when available, else a copy of dashboard_chart.png
                                    (flagged in JSON)
  reports/dashboard_spans.png       screenshot of the same project's Spans view (when a browser is available)

Run standalone (Phoenix persisted in .phoenix/):  python scripts/export_traces.py [--since ISO]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

log = logging.getLogger("export")
JUDGE_COMPONENT = "eval_judge"  # src/eval_judge.JUDGE_COMPONENT (not imported: pulls in deepeval)
NODE_NAMES = {"input_guard", "supervisor", "readmission_risk_agent", "policy_retrieval_agent", "plan_assembly_agent",
              "human_approval_gate", "clarify", "escalate", "output_guard"}
LATENCY_DEFINITIONS = {
    "thinking": "model-only time: duration of LLM spans (openinference span kind LLM). Since the rate-limiter "
                "split, the client-side GEMINI_RPM wait is no longer inside these spans.",
    "rate_limit_wait": "time spent waiting for the client-side InMemoryRateLimiter (GEMINI_RPM) before a model "
                       "request: duration of the 'rate_limit_wait' spans emitted by src/llm.ainvoke_with_retry.",
    "thinking_including_rate_limit": "for comparison with runs before the split: each LLM span's duration plus "
                                     "the rate_limit_wait span that immediately preceded it in the same trace.",
    "acting": "duration of LangGraph node spans (agents, guards, approval gate).",
    "tool": "duration of TOOL spans (MCP / RAG tool calls).",
    "end_to_end": "duration of the copilot_turn root span (one user turn).",
    "scope": "all categories, trace/turn counts, tokens, cost and errors cover agent spans only (Phoenix project "
             "PHOENIX_PROJECT_NAME). DeepEval judge spans are sent to the separate PHOENIX_PROJECT_NAME-eval project "
             "and reported in the eval_judge block; any span tagged copilot.component=eval_judge that still reaches "
             "the agent project is excluded the same way.",
}
BUDGETS = {"p95_end_to_end_latency_ms": 90000, "mean_cost_usd_per_turn": 0.01, "max_llm_calls_per_turn": 8,
           "tool_error_rate": 0.25}


def _col(df: pd.DataFrame, *names: str):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df), index=df.index)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Uniform columns regardless of whether spans came from Phoenix or the local fallback."""
    out = pd.DataFrame(index=df.index)
    out["trace_id"] = _col(df, "context.trace_id")
    out["span_id"] = _col(df, "context.span_id")
    out["parent_id"] = _col(df, "parent_id")
    out["name"] = _col(df, "name")
    out["span_kind"] = _col(df, "span_kind", "attributes.openinference.span.kind")
    start = pd.to_datetime(_col(df, "start_time"), utc=True, errors="coerce")
    end = pd.to_datetime(_col(df, "end_time"), utc=True, errors="coerce")
    if start.isna().all():  # local fallback stores ns epoch ints
        start = pd.to_datetime(_col(df, "start_time"), unit="ns", utc=True)
        end = pd.to_datetime(_col(df, "end_time"), unit="ns", utc=True)
    out["start_time"], out["end_time"] = start, end
    out["latency_ms"] = (end - start).dt.total_seconds() * 1000
    out["status_code"] = _col(df, "status_code")
    attrs = df["attributes"] if "attributes" in df.columns else None

    def attr(key: str):
        s = _col(df, f"attributes.{key}")
        if attrs is not None and s.isna().all():
            s = attrs.map(lambda a: (a or {}).get(key) if isinstance(a, dict) else None)
        return s

    out["model"] = attr("llm.model_name")
    out["prompt_tokens"] = pd.to_numeric(attr("llm.token_count.prompt"), errors="coerce").fillna(0).astype(int)
    out["completion_tokens"] = pd.to_numeric(attr("llm.token_count.completion"), errors="coerce").fillna(0).astype(int)
    out["session_id"] = attr("session.id")
    out["run_id"] = attr("run_id")

    def copilot_attr(key: str):
        # Phoenix nests "copilot.*" attributes under attributes.copilot (dict live, JSON string once exported);
        # the local fallback keeps the flat dotted key
        nested = _col(df, "attributes.copilot").map(
            lambda v: json.loads(v) if isinstance(v, str) and v.startswith("{") else v)
        s = nested.map(lambda v: v.get(key) if isinstance(v, dict) else None)
        return s.where(s.notna(), attr(f"copilot.{key}"))

    out["component"] = copilot_attr("component").fillna("agent")
    out["run_name"] = copilot_attr("run_name")
    # model provider of LLM, rate_limit_wait and quota_fallback spans ("gemini" / "groq"): the copilot.provider tag,
    # else openinference's llm.provider, else the model id; empty for spans that are not a model request
    by_llm_attr = attr("llm.provider").map(
        lambda v: {"groq": "groq", "google": "gemini", "google_genai": "gemini"}.get(str(v).lower()) if v else None)
    by_model = out["model"].map(lambda m: config.provider_of(m) if isinstance(m, str) and m else None)
    tagged = copilot_attr("provider")
    out["provider"] = tagged.where(tagged.notna(), by_llm_attr.where(by_llm_attr.notna(), by_model)).fillna("")

    def price(row):
        model = str(row["model"] or "")
        p = next((v for k, v in config.PRICING_PER_MTOK.items() if model.endswith(k) or k == model),
                 config.PRICING_PER_MTOK.get(config.GEMINI_MODEL))
        return (row["prompt_tokens"] * p["input"] + row["completion_tokens"] * p["output"]) / 1e6

    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    out["cost_usd"] = out.apply(price, axis=1).round(8)
    out["category"] = np.select(
        [out["name"] == "rate_limit_wait", out["span_kind"] == "LLM", out["span_kind"] == "TOOL",
         out["name"].isin(NODE_NAMES), out["name"] == "copilot_turn"],
        ["rate_limit_wait", "thinking", "tool", "acting", "end_to_end"], default="other")
    # propagate run/session from the copilot_turn root span to every span in the trace
    roots = out[out["name"] == "copilot_turn"].set_index("trace_id")
    if len(roots):
        rmap = attr("run_id").where(out["name"] == "copilot_turn")
        out["run_id"] = out["trace_id"].map(dict(zip(out.loc[rmap.notna(), "trace_id"], rmap.dropna())))
        smap = out.loc[out["name"] == "copilot_turn", ["trace_id", "session_id"]].dropna()
        out["session_id"] = out["trace_id"].map(dict(zip(smap["trace_id"], smap["session_id"])))
    return out.sort_values("start_time").reset_index(drop=True)


def pct(s: pd.Series, q: float):
    s = s.dropna()
    return round(float(np.percentile(s, q)), 1) if len(s) else None


def thinking_including_rate_limit(spans: pd.DataFrame) -> pd.Series:
    """Each LLM span's latency plus the rate_limit_wait span that ended just before it in the same trace."""
    waits = spans[spans["category"] == "rate_limit_wait"]
    out = []
    for _, row in spans[spans["category"] == "thinking"].iterrows():
        prev = waits[(waits["trace_id"] == row["trace_id"]) & (waits["end_time"] <= row["start_time"])]
        wait = prev.sort_values("end_time")["latency_ms"].iloc[-1] if len(prev) else 0.0
        out.append(row["latency_ms"] + wait)
    return pd.Series(out, dtype=float)


def judge_signals(judge: pd.DataFrame, source: str | None = None) -> dict:
    s = judge.loc[judge["category"] == "rate_limit_wait", "latency_ms"]
    return {
        "component": JUDGE_COMPONENT,
        "phoenix_project": config.PHOENIX_EVAL_PROJECT,
        "span_source": source,
        "note": "DeepEval judge traffic (scripts/run_eval.py), excluded from the agent's latency, trace/turn counts, "
                "tokens, cost and errors above. Judge requests use google-genai directly and are not traced as LLM "
                "spans; judge token usage is in reports/eval_report.json judge.usage.",
        "spans": int(len(judge)), "traces": int(judge["trace_id"].nunique()),
        "rate_limit_wait": {"count": int(s.count()), "p50_ms": pct(s, 50), "p95_ms": pct(s, 95),
                            "max_ms": round(float(s.max()), 1) if len(s) else None,
                            "total_s": round(float(s.sum()) / 1000, 1)},
        "span_errors": int((judge["status_code"] == "ERROR").sum()),
        "cost_usd_from_spans": round(float(judge["cost_usd"].sum()), 6),
    }


def _groq_calls_by_step() -> dict:
    """Groq requests per regeneration step, from the run records of each step's report."""
    def turns(name: str, key: str | None = None) -> list[dict]:
        path = config.REPORTS_DIR / name
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get(key or "cases", [])

    def groq(rows: list[dict], field: str) -> int:
        return sum(int((r.get(field) or {}).get("groq_calls") or 0) for r in rows)
    ev = turns("eval_report.json")
    return {"demo_sessions": groq(turns("cli_run_results.json"), "providers"),
            "red_team": groq(turns("redteam_results.json", "attacks"), "providers"),
            "eval_agent_turns": groq(ev, "agent_providers"), "eval_judge": groq(ev, "judge_providers")}


def provider_split(spans: pd.DataFrame) -> dict:
    """Latency / cost / volume for turns that used Gemini only, and separately for turns with any Groq request."""
    groq_traces = set(spans.loc[spans["provider"] == "groq", "trace_id"])
    llm = spans[spans["span_kind"] == "LLM"]
    out = {}
    for label, mask in (("gemini_only", ~spans["trace_id"].isin(groq_traces)),
                        ("groq_involved", spans["trace_id"].isin(groq_traces))):
        part = spans[mask]
        turns = part[part["name"] == "copilot_turn"]
        think = part.loc[part["category"] == "thinking", "latency_ms"]
        e2e = turns["latency_ms"]
        cost = part.groupby("trace_id")["cost_usd"].sum()
        out[label] = {"turns": int(len(turns)), "llm_calls": int((part["span_kind"] == "LLM").sum()),
                      "thinking_p50_ms": pct(think, 50), "thinking_p95_ms": pct(think, 95),
                      "end_to_end_p50_ms": pct(e2e, 50), "end_to_end_p95_ms": pct(e2e, 95),
                      "mean_cost_usd_per_turn": round(float(cost[cost.index.isin(turns["trace_id"])].mean()), 6)
                      if len(turns) else None}
    fallback = spans[spans["name"] == "quota_fallback"]
    return {"headline_scope": "gemini_only",
            "note": "Groq is a last-resort quota fallback (QUOTA_FALLBACK_PROVIDER=groq, off by default; a deviation "
                    "from the brief's Gemini-only rule). Headline metrics are Gemini-only; turns with any Groq request "
                    "are reported apart.",
            "llm_calls_by_provider": {k: int(v) for k, v in llm["provider"].replace("", "unknown").value_counts().items()},
            "groq_calls_by_agent_node": {str(k): int(v) for k, v in fallback["run_name"].value_counts().items()},
            "groq_calls_by_step": _groq_calls_by_step(),
            **out}


def golden_signals(all_spans: pd.DataFrame, source: str, judge_source: str | None = None) -> dict:
    is_judge = all_spans["component"] == JUDGE_COMPONENT
    spans = all_spans[~is_judge]
    lat = {}
    for cat in ("thinking", "rate_limit_wait", "thinking_including_rate_limit", "acting", "tool", "end_to_end"):
        s = thinking_including_rate_limit(spans) if cat == "thinking_including_rate_limit"             else spans.loc[spans["category"] == cat, "latency_ms"]
        lat[cat] = {"count": int(s.count()), "p50_ms": pct(s, 50), "p95_ms": pct(s, 95),
                    "max_ms": round(float(s.max()), 1) if len(s) else None}
    by_node = (spans[spans["category"].isin(["acting", "tool"])].groupby("name")["latency_ms"]
               .agg(["count", lambda s: pct(s, 50), lambda s: pct(s, 95)]))
    by_node.columns = ["count", "p50_ms", "p95_ms"]
    llm = spans[spans["span_kind"] == "LLM"]
    turns = spans[spans["name"] == "copilot_turn"]
    per_trace_cost = spans.groupby("trace_id")["cost_usd"].sum()
    per_trace_llm = llm.groupby("trace_id").size()
    tool_log = [json.loads(l) for l in config.TOOL_LOG.read_text(encoding="utf-8").splitlines()] \
        if config.TOOL_LOG.exists() else []
    tool_err = [r for r in tool_log if r["status"] != "ok"]
    eval_path = config.REPORTS_DIR / "eval_report.json"
    ev = json.loads(eval_path.read_text(encoding="utf-8"))["summary"] if eval_path.exists() else {}
    e2e_p95 = lat["end_to_end"]["p95_ms"]
    mean_cost = round(float(per_trace_cost[per_trace_cost.index.isin(turns["trace_id"])].mean()), 6) if len(turns) else None
    signals = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/export_traces.py",
        "span_source": source,
        "phoenix_project": config.PHOENIX_PROJECT,
        "trace_export": "traces/phoenix_spans.parquet",
        "window": {"first_span": str(spans["start_time"].min()), "last_span": str(spans["end_time"].max())},
        "traffic": {"spans": int(len(spans)), "traces": int(spans["trace_id"].nunique()), "turns": int(len(turns)),
                    "llm_calls": int(len(llm)), "tool_calls": int((spans["span_kind"] == "TOOL").sum()),
                    "sessions": int(spans["session_id"].nunique())},
        "latency": {**lat, "by_node": by_node.reset_index().to_dict(orient="records")},
        "latency_definitions": LATENCY_DEFINITIONS,
        "tokens": {"input": int(llm["prompt_tokens"].sum()), "output": int(llm["completion_tokens"].sum()),
                   "total": int(llm["prompt_tokens"].sum() + llm["completion_tokens"].sum()),
                   "mean_input_per_llm_call": round(float(llm["prompt_tokens"].mean()), 1) if len(llm) else None,
                   "models": sorted({str(m) for m in llm["model"].dropna()})},
        "cost": {"pricing_usd_per_mtok": {k: v for k, v in config.PRICING_PER_MTOK.items()},
                 "pricing_source": config.PRICE_SOURCE,
                 "total_usd": round(float(spans["cost_usd"].sum()), 6), "mean_usd_per_turn": mean_cost,
                 "max_llm_calls_per_turn": int(per_trace_llm.max()) if len(per_trace_llm) else 0,
                 "formula": "sum(prompt_tokens*input_price + completion_tokens*output_price)/1e6 over LLM spans"},
        "errors": {"span_errors": int((spans["status_code"] == "ERROR").sum()),
                   "span_error_rate": round(float((spans["status_code"] == "ERROR").mean()), 4) if len(spans) else 0,
                   "tool_calls_logged": len(tool_log), "tool_non_ok": len(tool_err),
                   "tool_error_rate": round(len(tool_err) / len(tool_log), 4) if tool_log else 0,
                   "tool_status_counts": pd.Series([r["status"] for r in tool_log]).value_counts().to_dict()},
        "eval_judge": judge_signals(all_spans[is_judge], judge_source or source),
        "providers": provider_split(spans),
        # headline = answers only: refusals / escalations / clarifications mostly don't engage with the record
        # context, so the HallucinationMetric (DeepEval >= 4: share of contexts agreed with, higher is better) can
        # score them 0.0 without anything being invented, and the overall rate then counts them as hallucinated
        "quality": {"source": "reports/eval_report.json", "accuracy": ev.get("accuracy"),
                    "hallucination_rate_answers": ev.get("hallucination_rate_answers"),
                    "hallucination_rate_overall": ev.get("hallucination_rate"),
                    "hallucination_rate_non_answers": ev.get("hallucination_rate_non_answers"),
                    "hallucination_rate_note": "headline is hallucination_rate_answers; non-answers (REFUSED, "
                                               "ESCALATED_TO_HUMAN, NEEDS_CLARIFICATION) can score 0.0 on the "
                                               "context-agreement metric without inventing anything (the judge "
                                               "counts not engaging with the record context as disagreement), so "
                                               "they are included only in hallucination_rate_overall",
                    "judged_coverage": ev.get("judged_coverage"),
                    "mean_faithfulness": ev.get("mean_faithfulness"),
                    "mean_answer_relevancy": ev.get("mean_answer_relevancy"),
                    # headline metrics for Gemini-only cases, and apart for cases with any Groq request
                    "headline_scope": ev.get("headline_scope", "gemini_only"),
                    "by_provider": ev.get("by_provider"),
                    "groq_calls_by_step": ev.get("groq_calls_by_step")},
    }
    checks = {
        "p95_end_to_end_latency_ms": (e2e_p95, e2e_p95 is not None and e2e_p95 <= BUDGETS["p95_end_to_end_latency_ms"]),
        "mean_cost_usd_per_turn": (mean_cost, mean_cost is not None and mean_cost <= BUDGETS["mean_cost_usd_per_turn"]),
        "max_llm_calls_per_turn": (signals["cost"]["max_llm_calls_per_turn"],
                                   signals["cost"]["max_llm_calls_per_turn"] <= BUDGETS["max_llm_calls_per_turn"]),
        "tool_error_rate": (signals["errors"]["tool_error_rate"],
                            signals["errors"]["tool_error_rate"] <= BUDGETS["tool_error_rate"]),
    }
    signals["budgets"] = {k: {"budget": BUDGETS[k], "actual": v, "within_budget": ok} for k, (v, ok) in checks.items()}
    return signals


def render_chart(spans: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Discharge Copilot - latency / tokens / cost (from reports/dashboard_data.csv, Phoenix spans)")
    spans = spans[spans["component"] != JUDGE_COMPONENT]  # agent traffic only
    cats = ["thinking", "rate_limit_wait", "acting", "tool", "end_to_end"]
    data = [spans.loc[spans["category"] == c, "latency_ms"].dropna() / 1000 for c in cats]
    axes[0, 0].boxplot(data, tick_labels=cats, showfliers=True)
    axes[0, 0].set_ylabel("seconds")
    axes[0, 0].set_title("Latency by span type")
    nodes = spans[spans["category"].isin(["acting", "tool"])].groupby("name")["latency_ms"].quantile(0.95) / 1000
    nodes.sort_values().plot.barh(ax=axes[0, 1], color="#3b6ea5")
    axes[0, 1].set_title("p95 latency by node / tool (s)")
    turns = spans[spans["name"] == "copilot_turn"][["trace_id", "session_id", "latency_ms"]]
    tok = spans.groupby("trace_id")[["prompt_tokens", "completion_tokens"]].sum()
    t = turns.merge(tok, left_on="trace_id", right_index=True)
    x = np.arange(len(t))
    axes[1, 0].bar(x, t["prompt_tokens"], label="input tokens", color="#5b8def")
    axes[1, 0].bar(x, t["completion_tokens"], bottom=t["prompt_tokens"], label="output tokens", color="#f0a35e")
    axes[1, 0].set_title("Tokens per turn")
    axes[1, 0].set_xlabel("turn #")
    axes[1, 0].legend()
    cost = spans.groupby("trace_id")["cost_usd"].sum()
    c = turns.merge(cost.rename("cost"), left_on="trace_id", right_index=True)
    axes[1, 1].bar(np.arange(len(c)), c["cost"] * 1000, color="#4caf7d")
    axes[1, 1].axhline(BUDGETS["mean_cost_usd_per_turn"] * 1000, color="red", ls="--", label="budget / turn")
    axes[1, 1].set_title("Estimated cost per turn (USD x 1000)")
    axes[1, 1].set_xlabel("turn #")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _browser_shot(browser: str, url: str, out: Path) -> bool:
    # --virtual-time-budget lets the page run (GraphQL fetches + chart rendering) for 30 s of virtual time
    # before the capture; it only advances while nothing is loading, so charts are drawn when it fires
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1600,1900",
                    "--virtual-time-budget=30000", "--run-all-compositor-stages-before-draw", f"--screenshot={out}",
                    url], timeout=120, check=False, capture_output=True)
    return out.exists() and out.stat().st_size > 20000


def screenshot_phoenix(out: Path, spans_out: Path | None = None, window: tuple | None = None) -> bool:
    """Headless-browser screenshots of the agent project in the running Phoenix UI: the Metrics view (latency,
    token counts, cost) to `out`, and the Spans list to `spans_out`. `window` = (first, last) span timestamps pins the
    UI time range to the exported run (default: Phoenix's last 7 days). Returns True if the Metrics capture worked."""
    candidates = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  shutil.which("msedge") or "", shutil.which("google-chrome") or "", shutil.which("chromium") or ""]
    browser = next((c for c in candidates if c and Path(c).exists()), None)
    if not browser:
        return False
    try:
        import urllib.parse
        import urllib.request

        from phoenix.client import Client

        pid = Client(base_url=f"http://localhost:{config.PHOENIX_PORT}").projects.get(
            project_name=config.PHOENIX_PROJECT)["id"]
        url = f"http://localhost:{config.PHOENIX_PORT}/projects/{pid}"
        urllib.request.urlopen(url, timeout=5)
        query = ""
        if window is not None:  # Phoenix UI custom time range (timeRangeStart / timeRangeEnd search params)
            pad = pd.Timedelta(minutes=2)
            start, end = (pd.Timestamp(window[0]) - pad).floor("min"), (pd.Timestamp(window[1]) + pad).ceil("min")
            query = "?" + urllib.parse.urlencode({"timeRangeStart": start.isoformat(), "timeRangeEnd": end.isoformat()})
        if spans_out is not None and not _browser_shot(browser, f"{url}/spans{query}", spans_out):
            log.warning("Phoenix Spans screenshot failed")
        return _browser_shot(browser, f"{url}/metrics{query}", out)
    except Exception as exc:
        log.warning("Phoenix screenshot failed: %s", exc)
        return False


def export(since: str | None = None, launch: bool = False) -> dict:
    from src.observability.tracing import fetch_spans_dataframe, init_tracing

    if launch:
        init_tracing()
    raw, source = fetch_spans_dataframe(config.PHOENIX_PROJECT)
    if raw.empty:
        raise SystemExit("no spans found - run scripts/run_cli.py --demo first (with tracing)")
    # judge spans live in their own project; the local fallback copy already holds them next to the agent spans
    judge_raw, judge_source = (pd.DataFrame(), source) if source == "local-masked-otel-copy" else \
        fetch_spans_dataframe(config.PHOENIX_EVAL_PROJECT, local_fallback=False)

    def load(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        n = normalise(df)
        if since:
            n = n[n["start_time"] >= pd.Timestamp(since)].reset_index(drop=True)
            df = df[df["context.span_id"].isin(n["span_id"])] if "context.span_id" in df.columns else df
        return df, n

    raw, spans = load(raw)
    if not judge_raw.empty:
        judge_raw, judge_spans = load(judge_raw)
        judge_spans["component"] = JUDGE_COMPONENT  # everything in the eval project is judge traffic
        spans = pd.concat([spans, judge_spans]).sort_values("start_time").reset_index(drop=True)
    config.TRACES_DIR.mkdir(exist_ok=True)
    # parquet/jsonl of the raw span export (nested objects serialised to JSON strings)
    for frame, name in ((raw, "phoenix_spans"), (judge_raw, "phoenix_eval_spans")):
        if frame.empty:
            if name == "phoenix_eval_spans":  # no stale judge export from an earlier run
                (config.TRACES_DIR / f"{name}.parquet").unlink(missing_ok=True)
            continue
        flat = frame.copy()
        for c in flat.columns:
            if flat[c].dtype == object:
                flat[c] = flat[c].map(lambda v: v if v is None or isinstance(v, str)
                                      else json.dumps(v, default=str) if isinstance(v, (dict, list, tuple, np.ndarray))
                                      else str(v))
        flat.to_parquet(config.TRACES_DIR / f"{name}.parquet", index=False)
        if name == "phoenix_spans":
            flat.to_json(config.TRACES_DIR / f"{name}.jsonl", orient="records", lines=True, date_format="iso")
    spans.to_csv(config.REPORTS_DIR / "dashboard_data.csv", index=False)
    signals = golden_signals(spans, source, judge_source)
    if not judge_raw.empty:
        signals["eval_judge"]["trace_export"] = "traces/phoenix_eval_spans.parquet"
    render_chart(spans, config.REPORTS_DIR / "dashboard_chart.png")
    shot, spans_shot = config.REPORTS_DIR / "dashboard.png", config.REPORTS_DIR / "dashboard_spans.png"
    spans_shot.unlink(missing_ok=True)
    agent = spans[spans["component"] != JUDGE_COMPONENT]
    if screenshot_phoenix(shot, spans_shot, (agent["start_time"].min(), agent["end_time"].max())):
        signals["dashboard"] = {"png": "reports/dashboard.png",
                                "kind": "phoenix-ui-screenshot (headless browser): project Metrics view",
                                "phoenix_project": config.PHOENIX_PROJECT,
                                "time_range": signals["window"],
                                "spans_png": "reports/dashboard_spans.png" if spans_shot.exists() else None,
                                "chart": "reports/dashboard_chart.png", "data": "reports/dashboard_data.csv"}
    else:
        shutil.copy(config.REPORTS_DIR / "dashboard_chart.png", shot)
        signals["dashboard"] = {"png": "reports/dashboard.png", "kind": "chart-from-dashboard_data.csv (no browser)",
                                "data": "reports/dashboard_data.csv"}
    (config.REPORTS_DIR / "golden_signals.json").write_text(json.dumps(signals, indent=2, default=str), encoding="utf-8")
    print(f"spans={len(spans)} traces={spans['trace_id'].nunique()} source={source}")
    print(json.dumps({k: signals[k] for k in ("traffic", "tokens", "cost", "budgets")}, indent=2, default=str)[:2500])
    return signals


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser()
    p.add_argument("--since", default=None, help="only export spans that started at/after this ISO timestamp")
    p.add_argument("--no-launch", action="store_true", help="do not start Phoenix (it is already running)")
    args = p.parse_args()
    export(args.since, launch=not args.no_launch)


if __name__ == "__main__":
    main()
