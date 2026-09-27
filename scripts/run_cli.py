"""CLI for the Discharge-Planning & Follow-up Copilot.

  python scripts/run_cli.py --demo                          # run every committed sample session
  python scripts/run_cli.py --patient PT-1001 --message "Draft the discharge plan"
  python scripts/run_cli.py --patient PT-1003 --session my-session --interactive
  python scripts/run_cli.py --approve --session my-session --clinician "attending physician" --note "reviewed"
  python scripts/run_cli.py --reject | --request-changes --session my-session --note "..."

Drafted plans / schedules (any risk tier) and order-change requests pause at clinician_signoff (status
AWAITING_CLINICIAN_SIGNOFF); the checkpoint keeps the paused session, so sign-off can run in a later process.

Phoenix tracing is on by default (UI http://localhost:6006 while the process runs; --no-tracing to skip).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

DEMO = config.SAMPLE_INPUTS / "demo_cases.json"


def _print_turn(res: dict) -> None:
    print("=" * 100)
    print(f"[{res['session_id']}] {res['patient_ref']}  run={res['run_id']}  trace={res['trace_id']}")
    print(f"CLINICIAN> {res['input']}")
    print(f"intent={res['intent']} status={res['status']} risk={res['risk_level']} "
          f"approval={res['requires_clinician_approval']} latency={res['latency_ms']}ms")
    print("-" * 100)
    print(res["response"])
    if res.get("awaiting_clinician_signoff"):
        print("-" * 100)
        print(f"PAUSED for clinician sign-off (draft status {res['draft_status']}). Resume with: python scripts/"
              f"run_cli.py --approve|--reject|--request-changes --session {res['session_id']} --note \"...\"")


async def run_demo(copilot, sessions: list[dict]) -> list[dict]:
    results = []
    for s in sessions:
        for msg in s["turns"]:
            res = await copilot.turn(s["session_id"], s["patient_id"], msg)
            _print_turn(res)
            res.pop("_state", None)
            res["covers"] = s.get("covers", [])
            results.append(res)
    return results


async def main_async(args) -> list[dict]:
    from src.runner import Copilot

    if not args.no_tracing:
        from src.observability.tracing import init_tracing

        handle = init_tracing()
        print(f"Phoenix UI: {handle.phoenix_url}  project={handle.project}")
    results: list[dict] = []
    async with Copilot() as copilot:
        if args.decision:
            if not args.session:
                raise SystemExit("--approve/--reject/--request-changes need --session <paused session id>")
            try:
                res = await copilot.resume(args.session, args.decision, args.clinician, args.note)
            except ValueError as exc:
                raise SystemExit(f"sign-off refused: {exc}") from None
            _print_turn(res)
            res.pop("_state", None)
            results = [res]
        elif args.demo:
            sessions = json.loads(DEMO.read_text(encoding="utf-8"))["sessions"]
            if args.demo_limit:
                sessions = sessions[: args.demo_limit]
            if args.fresh_sessions:
                suffix = uuid.uuid4().hex[:6]
                for s in sessions:
                    s["session_id"] = f"{s['session_id']}-{suffix}"
            results = await run_demo(copilot, sessions)
        elif args.interactive:
            session = args.session or f"cli-{uuid.uuid4().hex[:8]}"
            print(f"Interactive session {session} for {args.patient}. Ctrl-C or 'exit' to quit.")
            while True:
                try:
                    msg = input("CLINICIAN> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if msg.lower() in ("exit", "quit"):
                    break
                if msg:
                    res = await copilot.turn(session, args.patient, msg)
                    _print_turn(res)
        else:
            res = await copilot.turn(args.session or f"cli-{uuid.uuid4().hex[:8]}", args.patient, args.message)
            _print_turn(res)
            res.pop("_state", None)
            results = [res]
    if not args.no_tracing:
        from src.observability.tracing import init_tracing

        init_tracing().flush()
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Discharge-Planning & Follow-up Copilot CLI")
    p.add_argument("--demo", action="store_true", help="run the committed sample sessions")
    p.add_argument("--patient", default="PT-1001")
    p.add_argument("--session", default=None, help="session/thread id (reuse to continue a conversation)")
    p.add_argument("--message", default="Please assemble the discharge plan for this patient.")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--no-tracing", action="store_true")
    signoff = p.add_mutually_exclusive_group()
    signoff.add_argument("--approve", dest="decision", action="store_const", const="approve",
                         help="clinician sign-off: approve the paused draft of --session")
    signoff.add_argument("--reject", dest="decision", action="store_const", const="reject")
    signoff.add_argument("--request-changes", dest="decision", action="store_const", const="request_changes")
    p.add_argument("--clinician", default="attending physician", help="clinician role recorded in the audit log")
    p.add_argument("--note", default="", help="sign-off note (PHI-masked before it is logged)")
    p.add_argument("--fresh-sessions", action="store_true", help="suffix demo session ids so threads start empty")
    p.add_argument("--demo-limit", type=int, default=0, help="run only the first N demo sessions (0 = all)")
    p.add_argument("--results-out", default=str(config.REPORTS_DIR / "cli_run_results.json"))
    return p


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    results = asyncio.run(main_async(args))
    if args.demo and results:
        Path(args.results_out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.results_out} ({len(results)} turns)")


if __name__ == "__main__":
    main()
