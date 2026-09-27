"""Agent-level evaluation harness (AC-12) -> reports/eval_report.json

1. Runs every golden-set case through the real copilot (MCP server, RAG, Gemini) in a fresh session.
   A case is one message (`input`) or a list of `turns` in one session (the last turn is scored, earlier turns
   state facts it must use); `prior_session` turns run first in a separate session (return-visit recall).
2. Agent accuracy (deterministic): intent, risk tier, approval routing, status, required citations.
3. DeepEval with Gemini as LLM-as-judge: HallucinationMetric (vs record + tool + policy context) and
   FaithfulnessMetric (vs retrieval context) on EVERY case that produced content, refusals included;
   AnswerRelevancyMetric on answers only (skipped for refusals / escalations / clarification questions).
   The judge never uses the agent model (src/eval_judge.py). A metric it cannot score is "unjudged" with the
   reason; unjudged metrics are retried once at the end of the run, and the summary reports coverage.
   Without GEMINI_API_KEY the judge falls back to an offline lexical-grounding heuristic and the
   report says so (`judge.mode = offline-heuristic`).
4. Provider labels: every case records the provider of each agent request (all its turns) and of each judge request.
   A case is "groq_involved" if any of them went to Groq (QUOTA_FALLBACK_PROVIDER=groq, off by default). The summary
   reports the headline metrics for Gemini-only cases and, separately, for Groq-involved cases, plus the number of
   Groq calls per step (agent turns / judge) and per agent node.

Run:  python scripts/run_eval.py            (tracing on, so eval runs also land in Phoenix)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from src import config  # noqa: E402
from src.guardrails.guards import mask_obj, mask_phi  # noqa: E402
from src.observability.run_context import provider_summary  # noqa: E402

GOLDEN = config.SAMPLE_INPUTS / "golden_set.json"
REPORT = config.REPORTS_DIR / "eval_report.json"
THRESHOLD = 0.5
# turns that answer nothing: answer relevancy is not meaningful there, hallucination/faithfulness still are
NON_ANSWER_STATUSES = {"REFUSED", "ESCALATED_TO_HUMAN", "NEEDS_CLARIFICATION"}
METRICS = ("hallucination", "faithfulness", "answer_relevancy")
log = logging.getLogger("eval")


def _as_list(v):
    return v if isinstance(v, list) else [v]


def accuracy_checks(case: dict, res: dict) -> dict:
    st = res["_state"]
    checks = {}
    if "expected_intent" in case:
        checks["intent"] = res["intent"] in _as_list(case["expected_intent"])
    if "expected_risk_level" in case:
        checks["risk_level"] = res["risk_level"] == case["expected_risk_level"]
    if "expected_requires_approval" in case:
        checks["approval_routing"] = bool(res["requires_clinician_approval"]) == case["expected_requires_approval"]
    if "expected_status" in case:
        # an interrupted run reports AWAITING_CLINICIAN_SIGNOFF; the routed status is the one before the pause
        checks["status"] = (res.get("draft_status") or res["status"]) == case["expected_status"]
    if "expected_citations" in case:
        checks["citations"] = all(c in res["response"] for c in case["expected_citations"])
    if "expected_answer_contains" in case:
        checks["answer_content"] = all(s.lower() in res["response"].lower() for s in case["expected_answer_contains"])
    plan = st.get("discharge_plan")
    if plan:
        checks["orders_unchanged"] = bool(plan.get("orders_unchanged"))
    checks["no_phi_in_output"] = not re.search(r"MRN-\d|\+1-555-\d{4}|PT-\d{4}", res["response"])
    return checks


def build_contexts(res: dict) -> tuple[list[str], list[str]]:
    """retrieval_context = retrieved policy chunks; context = retrieval + record view + tool results."""
    st = res["_state"]
    hits = st.get("scratchpad", {}).get("policy_retrieval_agent.hits", [])
    retrieval = [f"[{h['citation']}] {h['text']}" for h in hits]
    rec = mask_obj({k: v for k, v in (st.get("record") or {}).items() if k != "nursing_notes"})
    ctx = list(retrieval)
    ctx.append("PATIENT RECORD (de-identified): " + json.dumps(rec)[:5000])
    if st.get("risk"):
        ctx.append("COMPUTED RISK: " + json.dumps(st["risk"])[:3000])
    if st.get("discharge_plan"):
        p = st["discharge_plan"]
        ctx.append("SCHEDULER RESULTS: " + json.dumps(p.get("follow_up_appointments", [])))
        ctx.append("HOME CARE RULES APPLIED: " + json.dumps(p.get("home_care_and_logistics", [])))
    if st.get("memory_recall"):
        ctx.append("LONG-TERM MEMORY: " + json.dumps(st["memory_recall"]))
    if st.get("session_facts"):  # facts the clinician stated in earlier turns of this session
        ctx.append("SESSION FACTS (stated by the clinician earlier in this session): " + json.dumps(st["session_facts"]))
    return retrieval or ctx, ctx


def applicable_metrics(res: dict) -> list[str]:
    """Hallucination + faithfulness for every case that produced content (refusals included); answer relevancy
    only for answers."""
    if not (res.get("response") or "").strip():
        return []
    names = ["hallucination", "faithfulness"]
    if (res.get("draft_status") or res["status"]) not in NON_ANSWER_STATUSES:
        names.append("answer_relevancy")
    return names


def unjudged(reason: str) -> dict:
    return {"score": None, "success": None, "status": "unjudged", "reason": reason[:400]}


# ------------------------------------------------------------------ judges
def gemini_judge():
    """DeepEval judge = Gemini (JSON mode, retries, never the agent model) - see src/eval_judge.py."""
    from src.eval_judge import GeminiFallbackJudge

    return GeminiFallbackJudge()


async def deepeval_scores(message, res, judge, names) -> dict:
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
    from deepeval.test_case import LLMTestCase

    retrieval, ctx = build_contexts(res)
    tc = LLMTestCase(input=message, actual_output=res["response"], retrieval_context=retrieval, context=ctx)
    make = {
        "hallucination": lambda: HallucinationMetric(threshold=THRESHOLD, model=judge, include_reason=True,
                                                     async_mode=False),
        "faithfulness": lambda: FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True,
                                                   async_mode=False),
        "answer_relevancy": lambda: AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=False,
                                                          async_mode=False),
    }
    out = {}
    for name in names:
        metric = make[name]()
        t0 = time.perf_counter()
        try:
            await asyncio.to_thread(metric.measure, tc)
            if metric.score is None:
                raise ValueError("metric returned no score")
            out[name] = {"score": round(float(metric.score), 4), "success": bool(metric.is_successful()),
                         "status": "judged", "reason": (metric.reason or "")[:600],
                         "latency_s": round(time.perf_counter() - t0, 1)}
        except Exception as exc:
            out[name] = unjudged(f"{type(exc).__name__}: {str(exc)[:300]}")
    return out


def offline_scores(message, res, names) -> dict:
    """Heuristic judge used only without a Gemini key: sentence-level lexical grounding."""
    _retrieval, ctx = build_contexts(res)
    blob = " ".join(ctx).lower()
    sents = [s for s in re.split(r"[\n.]+", res["response"]) if len(s.split()) >= 4]
    def grounded(s):
        toks = [t for t in re.findall(r"[a-z0-9]{4,}", s.lower())]
        return not toks or sum(t in blob for t in toks) / len(toks) >= 0.6
    g = [grounded(s) for s in sents] or [True]
    faith = sum(g) / len(g)
    q = set(re.findall(r"[a-z]{4,}", message.lower()))
    rel = len([t for t in q if t in res["response"].lower()]) / max(1, len(q))
    # same polarity as DeepEval 4 HallucinationMetric: score = share of grounded content (higher = better)
    out = {"hallucination": {"score": round(faith, 4), "success": faith >= THRESHOLD, "status": "judged"},
           "faithfulness": {"score": round(faith, 4), "success": faith >= THRESHOLD, "status": "judged"},
           "answer_relevancy": {"score": round(rel, 4), "success": rel >= THRESHOLD, "status": "judged"}}
    return {k: v for k, v in out.items() if k in names}


async def run_case(copilot, case: dict, tag: str) -> tuple[dict, list[dict], list[dict]]:
    """Runs a golden case; returns the scored (last) turn, the run references of every earlier turn, and every
    agent model request of the case (all turns, with provider)."""
    pid, earlier, calls = case["patient_id"], [], []
    for n, msg in enumerate(case.get("prior_session", [])):  # return visit: an earlier, separate session
        r = await copilot.turn(f"eval-{case['id']}-prior-{tag}", pid, msg)
        calls += r.get("model_calls", [])
        earlier.append({"session": "prior", "turn": n + 1, "run_id": r["run_id"], "trace_id": r["trace_id"],
                        "status": r["status"]})
    turns = case.get("turns") or [case["input"]]
    sid = f"eval-{case['id']}-{tag}"
    for n, msg in enumerate(turns[:-1]):  # multi-turn: facts stated here must be used in the last turn
        r = await copilot.turn(sid, pid, msg)
        calls += r.get("model_calls", [])
        earlier.append({"session": "same", "turn": n + 1, "run_id": r["run_id"], "trace_id": r["trace_id"],
                        "status": r["status"]})
    res = await copilot.turn(sid, pid, turns[-1])
    return res, earlier, calls + res.get("model_calls", [])


def _judge_calls(judge, since: int) -> list[dict]:
    return [{"provider": c.get("provider", "gemini"), "model": c["model"], "ok": c["ok"]}
            for c in (judge.calls[since:] if judge else [])]


def quality_block(rows: list[dict]) -> dict:
    """Headline metrics over `rows` (used for all cases, Gemini-only cases and Groq-involved cases)."""
    content = [r for r in rows if r["metrics"]]
    judged = [r for r in content if all(m["status"] == "judged" for m in r["metrics"].values())]
    hall = [r for r in content if r["metrics"].get("hallucination", {}).get("score") is not None]

    def rate(scored):
        return round(sum(not r["metrics"]["hallucination"]["success"] for r in scored) / len(scored), 4) \
            if scored else None

    def mean(name):
        vals = [r["metrics"][name]["score"] for r in content if r["metrics"].get(name, {}).get("score") is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {"cases": len(rows), "case_ids": [r["id"] for r in rows],
            "accuracy": round(sum(r["passed"] for r in rows) / len(rows), 4) if rows else None,
            "judged_coverage": round(len(judged) / len(content), 4) if content else None,
            "hallucination_rate_answers": rate([r for r in hall if r["status"] not in NON_ANSWER_STATUSES]),
            "hallucination_rate": rate(hall),
            "mean_faithfulness": mean("faithfulness"), "mean_answer_relevancy": mean("answer_relevancy")}


# ------------------------------------------------------------------ main
async def main_async(args) -> dict:
    from src.runner import Copilot

    if not args.no_tracing:
        from src.observability.tracing import init_tracing

        init_tracing()
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]
    if args.limit:
        cases = cases[: args.limit]
    judge_mode = "gemini" if config.GEMINI_API_KEY else "offline-heuristic"
    judge, judge_error = None, None
    if judge_mode == "gemini":
        try:
            judge = gemini_judge()
        except ValueError as exc:  # no judge model distinct from the agent model: nothing is judged
            judge_error = str(exc)
    tag = uuid.uuid4().hex[:6]
    rows, results = [], {}
    async with Copilot() as copilot:
        for case in cases:
            res, earlier, agent_calls = await run_case(copilot, case, tag)
            message = (case.get("turns") or [case["input"]])[-1]
            producers = {c["model"] for c in agent_calls if c.get("ok")}  # never judge with the answer's own model
            results[case["id"]] = (message, res, producers)
            checks = accuracy_checks(case, res)
            names = applicable_metrics(res)
            n_judge = len(judge.calls) if judge else 0
            if judge_mode != "gemini":
                scores = offline_scores(message, res, names)
            elif judge:
                judge.set_producers(producers)
                scores = await deepeval_scores(message, res, judge, names)
            else:
                scores = {n: unjudged(judge_error) for n in names}
            judge_calls = _judge_calls(judge, n_judge)
            rows.append({
                "id": case["id"], "input": mask_phi(message), "run_id": res["run_id"], "trace_id": res["trace_id"],
                **({"turns": [mask_phi(t) for t in case["turns"]]} if case.get("turns") else {}),
                **({"prior_session": [mask_phi(t) for t in case["prior_session"]]} if case.get("prior_session") else {}),
                **({"earlier_turns": earlier} if earlier else {}),
                "intent": res["intent"], "status": res.get("draft_status") or res["status"],
                "awaiting_clinician_signoff": res.get("awaiting_clinician_signoff", False),
                "risk_level": res["risk_level"],
                "requires_clinician_approval": res["requires_clinician_approval"],
                "checks": checks, "passed": all(checks.values()), "metrics": scores,
                "degradations": res["degradations"], "latency_ms": res["latency_ms"],
                "agent_providers": provider_summary(agent_calls), "judge_providers": provider_summary(judge_calls),
                "groq_involved": any(c["provider"] == "groq" for c in agent_calls + judge_calls),
                "_agent_calls": agent_calls, "_judge_calls": judge_calls,
                "actual_output": res["response"][:3000],
            })
            print(f"{case['id']}: passed={rows[-1]['passed']} checks={checks} "
                  + " ".join(f"{k}={v.get('score')}" for k, v in scores.items()))
    if judge:  # one retry, at the end of the run, for every metric the judge could not score
        for row in rows:
            todo = [n for n, m in row["metrics"].items() if m["status"] == "unjudged"]
            if todo:
                message, res, producers = results[row["id"]]
                n_judge = len(judge.calls)
                judge.set_producers(producers)
                for name, m in (await deepeval_scores(message, res, judge, todo)).items():
                    row["metrics"][name] = {**m, "retried": True}
                row["_judge_calls"] += _judge_calls(judge, n_judge)
                row["judge_providers"] = provider_summary(row["_judge_calls"])
                row["groq_involved"] = any(c["provider"] == "groq" for c in row["_agent_calls"] + row["_judge_calls"])
                print(f"{row['id']}: retried {todo} -> " + " ".join(f"{n}={row['metrics'][n]['status']}" for n in todo))
    if not args.no_tracing:
        from src.observability.tracing import init_tracing

        init_tracing().flush()

    content = [r for r in rows if r["metrics"]]  # cases that produced content (every metric that applies was run)
    judged = [r for r in content if all(m["status"] == "judged" for m in r["metrics"].values())]
    not_judged = [r for r in content if r not in judged]

    def _mean(name):
        vals = [r["metrics"][name]["score"] for r in content if r["metrics"].get(name, {}).get("score") is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    coverage = {name: {"applicable": sum(name in r["metrics"] for r in content),
                       "judged": sum(r["metrics"].get(name, {}).get("status") == "judged" for r in content)}
                for name in METRICS}
    for c in coverage.values():
        c["coverage"] = round(c["judged"] / c["applicable"], 4) if c["applicable"] else None
    hall_scored = [r for r in content if r["metrics"].get("hallucination", {}).get("score") is not None]
    # DeepEval 4.x HallucinationMetric: score = fraction of context items the output AGREES with
    # (higher = better); a case is hallucinated when the metric fails (score < threshold).
    hallucinated = [r for r in hall_scored if not r["metrics"]["hallucination"]["success"]]

    def _rate(scored):
        return round(sum(not r["metrics"]["hallucination"]["success"] for r in scored) / len(scored), 4) if scored else None
    total_checks = sum(len(r["checks"]) for r in rows)
    agent_calls = [c for r in rows for c in r.pop("_agent_calls")]
    judge_calls = [c for r in rows for c in r.pop("_judge_calls")]
    groq_by_node: dict[str, int] = {}
    for c in agent_calls:
        if c["provider"] == "groq":
            groq_by_node[c.get("run_name") or "?"] = groq_by_node.get(c.get("run_name") or "?", 0) + 1
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/run_eval.py",
        "golden_set": str(GOLDEN.relative_to(ROOT)).replace("\\", "/"),
        "agent_model": config.GEMINI_MODEL if config.GEMINI_API_KEY else "offline-stub",
        "judge": {"framework": "deepeval" if judge_mode == "gemini" else "offline-heuristic",
                  "mode": judge_mode, "model": judge.get_model_name() if judge else None,
                  "unavailable_reason": judge_error,
                  "threshold": THRESHOLD, "usage": judge.usage_summary() if judge else None,
                  "metrics": ["HallucinationMetric", "FaithfulnessMetric", "AnswerRelevancyMetric"],
                  "metric_scope": "hallucination + faithfulness on every case that produced content (refusals "
                                  "included); answer relevancy skipped for " + ", ".join(sorted(NON_ANSWER_STATUSES))},
        "summary": {
            "cases": len(rows),
            "accuracy": round(sum(r["passed"] for r in rows) / len(rows), 4),
            "check_level_accuracy": round(sum(sum(r["checks"].values()) for r in rows) / max(1, total_checks), 4),
            "content_cases": len(content),
            "judged_cases": len(judged),
            "unjudged_cases": len(not_judged),
            "unjudged_case_ids": [r["id"] for r in not_judged],
            "judged_coverage": round(len(judged) / len(content), 4) if content else None,
            "judged_coverage_definition": "share of content-producing cases on which every applicable metric was "
                                          "scored by the judge (after one end-of-run retry)",
            "metric_coverage": coverage,
            "same_model_as_agent": bool(judge and judge.usage_summary()["same_model_as_agent"]),
            "hallucination_rate": round(len(hallucinated) / len(hall_scored), 4) if hall_scored else None,
            "hallucination_rate_definition": "share of scored cases failing DeepEval HallucinationMetric "
                                             "(context-agreement score < threshold)",
            # a refusal / escalation agrees with no record context, so it can score low without inventing anything
            "hallucination_rate_answers": _rate([r for r in hall_scored if r["status"] not in NON_ANSWER_STATUSES]),
            "hallucination_rate_non_answers": _rate([r for r in hall_scored if r["status"] in NON_ANSWER_STATUSES]),
            "mean_hallucination_metric_score": _mean("hallucination"),
            "hallucination_metric_polarity": "deepeval>=4: fraction of contexts agreed with; higher is better",
            "mean_faithfulness": _mean("faithfulness"),
            "mean_answer_relevancy": _mean("answer_relevancy"),
            "failed_case_ids": [r["id"] for r in rows if not r["passed"]],
            "hallucinated_case_ids": [r["id"] for r in hallucinated],
            # the headline is Gemini-only; cases where any agent or judge request went to Groq are reported apart
            "quota_fallback_provider": config.QUOTA_FALLBACK_PROVIDER if config.groq_enabled() else "none",
            "headline_scope": "gemini_only",
            "by_provider": {"gemini_only": quality_block([r for r in rows if not r["groq_involved"]]),
                            "groq_involved": quality_block([r for r in rows if r["groq_involved"]])},
            "groq_calls_by_step": {"agent_turns": sum(c["provider"] == "groq" for c in agent_calls),
                                   "judge": sum(c["provider"] == "groq" for c in judge_calls)},
            "groq_calls_by_agent_node": groq_by_node,
        },
        "cases": rows,
    }
    return report


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser()
    p.add_argument("--no-tracing", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=str(REPORT))
    args = p.parse_args()
    report = asyncio.run(main_async(args))
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
