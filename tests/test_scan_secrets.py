"""scripts/scan_evidence.py flags a Groq-style API key (gsk_...) in evidence even when it is not the configured
GROQ_API_KEY, so the committed scan catches leaked Groq keys by pattern. Offline: fake repo under tmp_path."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("scan_evidence", ROOT / "scripts" / "scan_evidence.py")
scan_evidence = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan_evidence)

# built at runtime so no key-shaped literal is committed (would trip push-protection secret scanners)
FAKE_GROQ_KEY = "gsk" + "_" + "Fake0TestKey1NotReal2" * 2 + "abcdEFGH"


def _fake_repo(tmp_path: Path, log_text: str) -> Path:
    for d in scan_evidence.SCAN_DIRS:
        (tmp_path / d).mkdir()
    (tmp_path / "logs" / "tool_calls.jsonl").write_text(log_text, encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("GEMINI_API_KEY=your_gemini_api_key_here\n", encoding="utf-8")
    return tmp_path


def test_scan_evidence_flags_groq_key_pattern(tmp_path, monkeypatch):
    monkeypatch.setattr(scan_evidence.config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(scan_evidence.config, "GROQ_API_KEY", "")  # pattern alone must catch it
    monkeypatch.setattr(scan_evidence, "ROOT", _fake_repo(tmp_path, json.dumps({"headers": f"Bearer {FAKE_GROQ_KEY}"})))
    out = scan_evidence.scan()
    assert not out["clean"]
    assert {"file": "logs/tool_calls.jsonl", "type": "SECRET_PATTERN"} in out["findings"]


def test_scan_evidence_clean_without_groq_key(tmp_path, monkeypatch):
    monkeypatch.setattr(scan_evidence.config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(scan_evidence.config, "GROQ_API_KEY", "")
    monkeypatch.setattr(scan_evidence, "ROOT", _fake_repo(tmp_path, json.dumps({"provider": "groq", "model": "gsk_short"})))
    out = scan_evidence.scan()
    assert out["clean"], out["findings"]
