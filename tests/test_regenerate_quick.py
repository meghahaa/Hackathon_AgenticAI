"""scripts/regenerate_evidence.py --quick writes under state/quick_run/ and leaves logs/, traces/ and reports/ alone,
unless --write-evidence is also given. Offline: every step is recorded instead of run (no Phoenix, no Gemini)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from src import config

ROOT = Path(__file__).resolve().parent.parent
REDIRECTED = ("LOGS_DIR", "TRACES_DIR", "REPORTS_DIR", "TOOL_LOG", "AUDIT_LOG", "MCP_TRANSCRIPT", "MEMORY_TEST_LOG",
              "CHECKPOINT_DB", "LONG_TERM_DB", "PHOENIX_PROJECT", "PHOENIX_EVAL_PROJECT")
ENV = ("COPILOT_LOG_DIR", "COPILOT_TRACES_DIR", "COPILOT_REPORTS_DIR", "MCP_TRANSCRIPT_PATH", "CHECKPOINT_DB",
       "LONG_TERM_DB", "PHOENIX_PROJECT_NAME", "PHOENIX_EVAL_PROJECT_NAME")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def regen(tmp_path, monkeypatch):
    """The script with a fake repo root, every step recorded, and config + environment restored afterwards."""
    mod = _load("regenerate_evidence")
    for name in REDIRECTED:  # registered so monkeypatch puts the real values back after the redirect
        monkeypatch.setattr(config, name, getattr(config, name))
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in ENV:
        os.environ.pop(name, None)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "QUICK_RUN_DIR", tmp_path / "state" / "quick_run")
    rec: dict = {"subprocess": []}

    def run_all(args):
        rec["run_all"] = {"quick": args.quick, **{n: getattr(config, n) for n in REDIRECTED}}

    def run_tests(out_dir):
        rec["tests_out"] = out_dir
        return type("R", (), {"stdout": ""})()

    monkeypatch.setattr(mod, "preflight_phoenix", lambda: None)
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, **k: rec["subprocess"].append([str(c) for c in cmd]))
    monkeypatch.setattr(mod, "reset_state", lambda: rec.setdefault("reset", True))
    monkeypatch.setattr(mod, "run_all", run_all)
    monkeypatch.setattr(mod.asyncio, "run", lambda coro: None)
    monkeypatch.setattr(mod, "run_tests", run_tests)
    return mod, rec, tmp_path


def _scan_cmd(rec) -> list[str]:
    (cmd,) = [c for c in rec["subprocess"] if c[1].endswith("scan_evidence.py")]
    return cmd[2:]


def test_quick_writes_under_state_quick_run(regen, monkeypatch):
    mod, rec, root = regen
    real = {n: getattr(config, n) for n in REDIRECTED}
    monkeypatch.setattr(sys, "argv", ["regenerate_evidence.py", "--quick"])
    mod.main()
    quick = root / "state" / "quick_run"
    seen = rec["run_all"]
    assert seen["quick"] is True
    for name in ("LOGS_DIR", "TRACES_DIR", "REPORTS_DIR", "TOOL_LOG", "AUDIT_LOG", "MCP_TRANSCRIPT",
                 "MEMORY_TEST_LOG", "CHECKPOINT_DB", "LONG_TERM_DB"):
        assert quick in Path(seen[name]).parents, name  # every output of the run is under state/quick_run/
        assert seen[name] != real[name]
    assert seen["REPORTS_DIR"] == quick / "reports" and rec["tests_out"] == quick / "reports"
    assert seen["PHOENIX_PROJECT"] == f"{real['PHOENIX_PROJECT']}-quick"  # the evidence project is not reset
    assert seen["PHOENIX_EVAL_PROJECT"] == f"{real['PHOENIX_PROJECT']}-quick-eval"
    # child processes (synthetic data, PII demo, scan, pytest) inherit the redirect
    assert os.environ["COPILOT_LOG_DIR"] == str(quick / "logs")
    assert os.environ["COPILOT_TRACES_DIR"] == str(quick / "traces")
    assert os.environ["COPILOT_REPORTS_DIR"] == str(quick / "reports")
    assert all(os.environ[n] for n in ENV)
    assert _scan_cmd(rec) == ["--dirs", "state/quick_run/logs", "state/quick_run/traces", "state/quick_run/reports",
                              "docs"]
    assert not any((root / d).exists() for d in ("logs", "traces", "reports"))


def test_quick_with_write_evidence_writes_the_evidence_dirs(regen, monkeypatch):
    mod, rec, root = regen
    real = {n: getattr(config, n) for n in REDIRECTED}
    monkeypatch.setattr(sys, "argv", ["regenerate_evidence.py", "--quick", "--write-evidence"])
    mod.main()
    assert rec["run_all"] == {"quick": True, **real}
    assert rec["tests_out"] == real["REPORTS_DIR"]
    assert not any(n in os.environ for n in ENV)
    assert _scan_cmd(rec) == []  # default scan: the committed evidence
    assert not (root / "state" / "quick_run").exists()


def test_write_evidence_without_quick_is_rejected(regen, monkeypatch):
    mod, rec, _ = regen
    monkeypatch.setattr(sys, "argv", ["regenerate_evidence.py", "--write-evidence"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2 and "run_all" not in rec and rec["subprocess"] == []


def test_scan_evidence_scans_only_the_given_dirs(tmp_path, monkeypatch):
    scan_evidence = _load("scan_evidence")
    for d in ("logs", "state/quick_run/logs", "docs"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "logs" / "agent_actions.jsonl").write_text('{"reason": "PT-1002"}', encoding="utf-8")
    (tmp_path / "state" / "quick_run" / "logs" / "agent_actions.jsonl").write_text('{"reason": "PT 1003"}',
                                                                                      encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("GEMINI_API_KEY=your_gemini_api_key_here\n", encoding="utf-8")
    monkeypatch.setattr(scan_evidence, "ROOT", tmp_path)
    out = scan_evidence.scan(["state/quick_run/logs", "docs"])
    assert out["dirs"] == ["state/quick_run/logs", "docs"]
    assert [f["file"] for f in out["findings"]] == ["state/quick_run/logs/agent_actions.jsonl"]
