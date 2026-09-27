"""ONE command that regenerates every evidence artifact from committed code + committed sample inputs
(NFR-02 / Reproducibility Rule):

    python scripts/regenerate_evidence.py

Steps (single process; Phoenix in-process, or an already-running local `phoenix serve`):
  0. preflight: Phoenix reachable + OTLP accepts spans, else stop before anything runs
  1. regenerate synthetic data                      data/
  2. reset runtime state + evidence logs             state/, logs/*.jsonl
  3. Phoenix tracing on                              http://localhost:6006
  4. run the committed demo sessions                 reports/cli_run_results.json, logs/*.jsonl
  5. run the red-team attack set                     reports/redteam_results.json
  6. run the golden-set evaluation (DeepEval, Gemini judge that is never the agent model) reports/eval_report.json
  7. export spans + golden signals + dashboard       traces/, reports/golden_signals.json, dashboard.*
  8. mine failure candidates for failure-analysis    reports/failure_candidates.json
  9. PII redaction demo + PHI/secrets scan           reports/pii_redaction_sample.json, reports/phi_secret_scan.json
 10. run the agent test-suite (verbose, -rA)         reports/pytest_report.txt, reports/pytest_junit.xml,
                                                     logs/memory_test.log
Flags: --quick, --write-evidence, --skip-eval, --skip-tests, --keep-state, --keep-phoenix-history
  --quick: reviewer fast path - the first QUICK_DEMO_SESSIONS demo sessions, one red-team attack per category
  (QUICK_REDTEAM_IDS) and the first QUICK_GOLDEN_CASES golden cases. Same steps and file names as the full run (which
  stays the default and is what the committed evidence comes from), but written under state/quick_run/ (logs/,
  traces/, reports/, runtime SQLite) and to the Phoenix projects <PHOENIX_PROJECT_NAME>-quick(-eval), so the
  committed evidence is untouched. Only --quick --write-evidence writes the subset into logs/, traces/ and reports/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config  # noqa: E402

# --quick subset: S1-S3 (multi-turn, return-visit recall, heart failure); the first attack of every red-team category;
# G01-G12 (the original golden set: every intent, including refusal, escalation and clarification)
QUICK_DEMO_SESSIONS = 3
QUICK_REDTEAM_IDS = "RT01,RT05,RT08,RT11,RT14,RT17,RT19,RT20"
QUICK_GOLDEN_CASES = 12
QUICK_RUN_DIR = ROOT / "state" / "quick_run"


def redirect_outputs(out: Path) -> dict[str, str]:
    """--quick without --write-evidence: every output of the run goes under `out` instead of logs/, traces/ and
    reports/. Sets config for this process and the environment for the child processes (synthetic data, PII demo,
    scan, pytest; the MCP server gets config.MCP_TRANSCRIPT from its client). Returns the environment it set."""
    logs, traces, reports = out / "logs", out / "traces", out / "reports"
    for d in (logs, traces, reports):
        d.mkdir(parents=True, exist_ok=True)
    project = f"{config.PHOENIX_PROJECT}-quick"
    paths = {"LOGS_DIR": logs, "TRACES_DIR": traces, "REPORTS_DIR": reports,
             "TOOL_LOG": logs / "tool_calls.jsonl", "AUDIT_LOG": logs / "agent_actions.jsonl",
             "MCP_TRANSCRIPT": logs / "mcp_transcript.jsonl", "MEMORY_TEST_LOG": logs / "memory_test.log",
             "CHECKPOINT_DB": out / "checkpoints.sqlite", "LONG_TERM_DB": out / "long_term_memory.sqlite"}
    for name, value in paths.items():
        setattr(config, name, value)
    config.PHOENIX_PROJECT, config.PHOENIX_EVAL_PROJECT = project, f"{project}-eval"
    env = {"COPILOT_LOG_DIR": str(logs), "COPILOT_TRACES_DIR": str(traces), "COPILOT_REPORTS_DIR": str(reports),
           "MCP_TRANSCRIPT_PATH": str(paths["MCP_TRANSCRIPT"]), "CHECKPOINT_DB": str(paths["CHECKPOINT_DB"]),
           "LONG_TERM_DB": str(paths["LONG_TERM_DB"]), "PHOENIX_PROJECT_NAME": project,
           "PHOENIX_EVAL_PROJECT_NAME": f"{project}-eval"}
    os.environ.update(env)
    return env


def step(n: int, msg: str) -> None:
    print(f"\n########## [{n}/10] {msg}", flush=True)


def preflight_phoenix() -> None:
    """Before ANY step: Phoenix must be up on PHOENIX_PORT (launched here if needed, as step 3 would) and its OTLP
    endpoint must accept spans. Otherwise every span export blocks the run on ~10 s of retries and the traces are
    lost (seen in development: 45 failed exports stretched one turn to 512 s), so stop before anything is reset."""
    from src.observability.tracing import ensure_phoenix, otlp_accepts_spans

    base = f"http://localhost:{config.PHOENIX_PORT}"
    if not ensure_phoenix(launch=True):
        problem = f"Phoenix is not reachable at {base}/healthz"
    else:
        ok, detail = otlp_accepts_spans()
        if ok:
            print(f"preflight: Phoenix up at {base}; OTLP endpoint accepts spans ({detail})", flush=True)
            return
        problem = f"Phoenix is up but its OTLP endpoint does not accept spans: {detail}"
    raise SystemExit(
        f"PREFLIGHT FAILED: {problem}.\nNothing was run: no data regenerated, no state or logs reset.\n"
        f"Start Phoenix (python -m phoenix.server.main serve) or free the ports / set PHOENIX_PORT and "
        f"PHOENIX_GRPC_PORT to unused ports (another Phoenix may hold 4317), then re-run.")


def reset_state() -> None:
    for p in (config.TOOL_LOG, config.AUDIT_LOG, config.MCP_TRANSCRIPT, config.STATE_DIR / "spans_local.jsonl",
              config.CHECKPOINT_DB, config.LONG_TERM_DB):
        if Path(p).exists():
            Path(p).unlink()


def run_tests(out_dir: Path) -> subprocess.CompletedProcess:
    """Full test suite, verbose with -rA (a summary line per test, passed ones too) -> pytest_report.txt, plus the
    JUnit XML of the same run -> pytest_junit.xml. Parametrised test ids carry synthetic patient IDs (e.g. the
    cross-patient guard cases), so both files are PHI-masked (mask_phi) before they are kept as evidence."""
    from src.guardrails.guards import mask_phi

    junit = out_dir / "pytest_junit.xml"
    r = subprocess.run([sys.executable, "-m", "pytest", "-v", "-rA", "-p", "no:cacheprovider", f"--junitxml={junit}"],
                       cwd=ROOT, capture_output=True, text=True)
    (out_dir / "pytest_report.txt").write_text(mask_phi(r.stdout + r.stderr[-5000:]), encoding="utf-8")
    if junit.exists():
        junit.write_text(mask_phi(junit.read_text(encoding="utf-8")), encoding="utf-8")
    return r


def mine_failures() -> dict:
    """Collect real failures from this run: non-ok tool calls, error spans, degradations, failed eval checks."""
    import pandas as pd

    out: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "generator": "scripts/regenerate_evidence.py::mine_failures"}
    tool = [json.loads(l) for l in config.TOOL_LOG.read_text(encoding="utf-8").splitlines()] if config.TOOL_LOG.exists() else []
    out["tool_failures"] = [{k: r[k] for k in ("timestamp", "run_id", "session_id", "agent", "tool_name", "status",
                                                "error", "trace_id", "span_id")} | {"result": r["result"]}
                            for r in tool if r["status"] != "ok"]
    spans = pd.read_parquet(config.TRACES_DIR / "phoenix_spans.parquet")
    err = spans[spans["status_code"] == "ERROR"] if "status_code" in spans.columns else spans.iloc[0:0]
    cols = [c for c in ("context.trace_id", "context.span_id", "name", "span_kind", "status_message") if c in err.columns]
    out["error_spans"] = err[cols].to_dict(orient="records")
    runs = []
    for src in ("cli_run_results.json", "eval_report.json"):
        p = config.REPORTS_DIR / src
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("cases", [])
        for r in rows:
            if r.get("degradations") or r.get("passed") is False:
                runs.append({"source": f"reports/{src}", "id": r.get("id"), "run_id": r["run_id"],
                             "trace_id": r["trace_id"], "session_id": r.get("session_id"), "intent": r["intent"],
                             "status": r["status"], "degradations": r.get("degradations"),
                             "failed_checks": [k for k, v in r.get("checks", {}).items() if not v],
                             "metrics": {k: v.get("score") for k, v in r.get("metrics", {}).items()}})
    out["degraded_or_failed_runs"] = runs
    (config.REPORTS_DIR / "failure_candidates.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"tool failures={len(out['tool_failures'])} error spans={len(out['error_spans'])} degraded/failed runs={len(runs)}")
    return out


async def run_all(args) -> None:
    import export_traces
    import run_cli
    import run_eval
    import run_redteam
    from src.observability.tracing import init_tracing

    started = datetime.now(timezone.utc).isoformat()
    step(3, "Phoenix tracing")
    h = init_tracing()
    if not args.keep_phoenix_history:  # UI + screenshot show only this regeneration
        try:
            from phoenix.client import Client

            client = Client(base_url=f"http://localhost:{config.PHOENIX_PORT}")
            for project in (h.project, h.eval_project):  # agent project + DeepEval judge project
                try:
                    client.projects.delete(project_name=project)
                except Exception as exc:
                    print(f"(phoenix project {project} reset skipped: {exc})")
        except Exception as exc:
            print(f"(phoenix project reset skipped: {exc})")
    print(f"Phoenix UI {h.phoenix_url} project={h.project} judge_project={h.eval_project} pricing={h.pricing}")

    step(4, "demo sessions through the graph")
    cli_args = run_cli.build_parser().parse_args(
        ["--demo"] + (["--demo-limit", str(QUICK_DEMO_SESSIONS)] if args.quick else []))
    results = await run_cli.main_async(cli_args)
    Path(cli_args.results_out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    step(5, "red-team attack set through the graph")
    rt_args = run_redteam.build_parser().parse_args(["--ids", QUICK_REDTEAM_IDS] if args.quick else [])
    rt_report = await run_redteam.main_async(rt_args)
    Path(rt_args.out).write_text(json.dumps(rt_report, indent=2, default=str), encoding="utf-8")

    if not args.skip_eval:
        step(6, "golden-set evaluation (DeepEval, Gemini judge)")
        report = await run_eval.main_async(
            argparse.Namespace(no_tracing=False, limit=QUICK_GOLDEN_CASES if args.quick else 0))
        run_eval.REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps(report["summary"], indent=2))
    h.flush()

    step(7, "export Phoenix spans, golden signals, dashboard")
    export_traces.export(since=started, launch=False)
    step(8, "mine failure candidates")
    mine_failures()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    from src.observability.tracing import use_utf8_output

    use_utf8_output()  # before any child process: they inherit PYTHONUTF8=1 (Windows cp1252 consoles)
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="reviewer fast path: a subset of demo sessions, red-team attacks and golden cases, written "
                        "under state/quick_run/")
    p.add_argument("--write-evidence", action="store_true",
                   help="with --quick: write the subset into logs/, traces/ and reports/ instead of state/quick_run/")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--keep-state", action="store_true")
    p.add_argument("--keep-phoenix-history", action="store_true")
    args = p.parse_args()
    if args.write_evidence and not args.quick:
        p.error("--write-evidence only applies to --quick (the full run always writes the evidence)")
    if args.quick and not args.write_evidence:
        redirect_outputs(QUICK_RUN_DIR)
        print(f"--quick: outputs go to {QUICK_RUN_DIR} (Phoenix project {config.PHOENIX_PROJECT}); "
              f"logs/, traces/ and reports/ are not touched", flush=True)

    preflight_phoenix()
    step(1, "synthetic data")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_synthetic_data.py")], check=True)
    step(2, "reset runtime state and evidence logs")
    if not args.keep_state:
        reset_state()
    asyncio.run(run_all(args))
    step(9, "PII redaction demo (registry / regex / Presidio) + PHI and secrets scan of committed evidence")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "pii_redaction_demo.py")], cwd=ROOT)
    scan_dirs = []  # default: the committed evidence (logs/ traces/ reports/ docs/); --quick: its own outputs + docs/
    if args.quick and not args.write_evidence:
        scan_dirs = ["--dirs", *(Path(d).relative_to(ROOT).as_posix()
                                 for d in (config.LOGS_DIR, config.TRACES_DIR, config.REPORTS_DIR)), "docs"]
    subprocess.run([sys.executable, str(ROOT / "scripts" / "scan_evidence.py"), *scan_dirs], cwd=ROOT)
    if not args.skip_tests:
        step(10, "agent tests (routing, loops, tool contracts, memory persistence, citations)")
        r = run_tests(config.REPORTS_DIR)
        print(r.stdout[-1500:])
    print("\nDone. Evidence regenerated.")


if __name__ == "__main__":
    main()
