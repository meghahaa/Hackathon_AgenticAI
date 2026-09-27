"""Recompute the cost of frozen runs at the current prices, without touching the frozen files.

Final run v1 and the optimisation baseline were costed with placeholder prices (flash-lite 0.10 / 0.40 USD per 1M
tokens, see their golden_signals.json cost.pricing_source). This script re-reads their frozen span exports, prices
every LLM span's recorded token counts with config.PRICING_PER_MTOK (the same normalise() and formula as
scripts/export_traces.py) and writes a new file. It also reports the like-for-like turns compared in
docs/optimization-note.md, from the frozen spans and from the current traces/phoenix_spans.parquet.

    python scripts/reprice_run.py      ->  reports/optimization/final_run_v1_repriced.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from export_traces import JUDGE_COMPONENT, normalise, pct  # noqa: E402
from src import config  # noqa: E402

OUT = ROOT / "reports" / "optimization" / "final_run_v1_repriced.json"
RUNS = {  # frozen span export, frozen golden signals (for the original placeholder cost)
    "final_run_v1": ("reports/failure_evidence/final_run_v1/phoenix_spans.parquet",
                     "reports/failure_evidence/final_run_v1/golden_signals.json"),
    "baseline": ("reports/optimization/phoenix_spans_before.parquet",
                 "reports/optimization/golden_signals_before.json"),
}
CURRENT = "traces/phoenix_spans.parquet"
# like-for-like turn "Please assemble the discharge plan for this patient." (demo session S1, HIGH risk)
TURNS = {"baseline": ("reports/optimization/phoenix_spans_before.parquet", "run-c2a2a26912f4"),
         "final_run_v1": ("reports/failure_evidence/final_run_v1/phoenix_spans.parquet", "run-b97b81524fd6"),
         "current": (CURRENT, "run-fb5b4e0b7ea1")}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spans(rel: str) -> pd.DataFrame:
    spans = normalise(pd.read_parquet(ROOT / rel))
    return spans[spans["component"] != JUDGE_COMPONENT]


def reprice(rel: str, signals_rel: str) -> dict:
    spans = _spans(rel)
    llm = spans[spans["span_kind"] == "LLM"]
    turns = spans[spans["name"] == "copilot_turn"]
    per_trace = spans.groupby("trace_id")["cost_usd"].sum()
    by_model = llm.groupby(llm["model"].fillna("unknown")).agg(
        llm_calls=("span_id", "size"), input_tokens=("prompt_tokens", "sum"), output_tokens=("completion_tokens", "sum"),
        cost_usd=("cost_usd", "sum"))
    original = json.loads((ROOT / signals_rel).read_text(encoding="utf-8"))["cost"]
    return {
        "span_export": rel, "span_export_sha256": _sha(ROOT / rel),
        "turns": int(len(turns)), "llm_calls": int(len(llm)),
        "tokens": {"input": int(llm["prompt_tokens"].sum()), "output": int(llm["completion_tokens"].sum())},
        "by_model": {m: {k: (round(float(v), 6) if k == "cost_usd" else int(v)) for k, v in row.items()}
                     for m, row in by_model.iterrows()},
        "current_prices": {"total_usd": round(float(spans["cost_usd"].sum()), 6),
                           "mean_usd_per_turn": round(float(per_trace[per_trace.index.isin(turns["trace_id"])].mean()), 6)
                           if len(turns) else None},
        "original": {"source": signals_rel, "total_usd": original.get("total_usd"),
                     "mean_usd_per_turn": original.get("mean_usd_per_turn"),
                     "pricing_source": original.get("pricing_source"),
                     "pricing_usd_per_mtok": {m: original.get("pricing_usd_per_mtok", {}).get(m)
                                              for m in by_model.index}},
    }


def turn(rel: str, run_id: str) -> dict:
    spans = _spans(rel)
    root = spans[(spans["name"] == "copilot_turn") & (spans["run_id"] == run_id)]
    if root.empty:
        return {"span_export": rel, "run_id": run_id, "found": False}
    t = spans[spans["trace_id"] == root["trace_id"].iloc[0]]
    llm = t[t["span_kind"] == "LLM"]
    wait = t[t["name"] == "rate_limit_wait"]["latency_ms"]
    return {"span_export": rel, "run_id": run_id, "trace_id": root["trace_id"].iloc[0],
            "root_span_id": root["span_id"].iloc[0],
            "end_to_end_s": round(float(root["latency_ms"].iloc[0]) / 1000, 1),
            "llm_calls": int(len(llm)), "llm_error_spans": int((llm["status_code"] == "ERROR").sum()),
            "llm_models": sorted({str(m) for m in llm["model"].dropna()}),
            # before the rate-limiter split (no rate_limit_wait spans) the GEMINI_RPM wait sits inside the LLM spans
            "rate_limit_wait_inside_llm_spans": wait.empty,
            "llm_span_total_s": round(float(llm["latency_ms"].sum()) / 1000, 1),
            "llm_span_p50_ms": pct(llm["latency_ms"], 50),
            "rate_limit_wait_s": round(float(wait.sum()) / 1000, 1) if len(wait) else None,
            "tokens": {"input": int(llm["prompt_tokens"].sum()), "output": int(llm["completion_tokens"].sum())},
            "cost_usd_current_prices": round(float(t["cost_usd"].sum()), 6)}


def main() -> None:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/reprice_run.py",
        "note": "Frozen snapshots are read only, never modified. 'original' is the cost as first reported, with the "
                "placeholder prices of that time; 'current_prices' re-prices the same recorded token counts.",
        "pricing_usd_per_mtok": config.PRICING_PER_MTOK, "pricing_source": config.PRICE_SOURCE,
        "formula": "sum(prompt_tokens*input_price + completion_tokens*output_price)/1e6 over LLM spans "
                   "(scripts/export_traces.normalise)",
        "runs": {name: reprice(*paths) for name, paths in RUNS.items()},
        "like_for_like_turn": {"message": "Please assemble the discharge plan for this patient.",
                               **{name: turn(*args) for name, args in TURNS.items()}},
    }
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["runs"], indent=2, default=str))
    print(json.dumps(report["like_for_like_turn"], indent=2, default=str))


if __name__ == "__main__":
    main()
