import os
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")


@pytest.fixture(autouse=True)
def _isolate_logs(tmp_path, monkeypatch):
    """Tests never write into the committed evidence logs; each test gets private log files."""
    from src import config

    monkeypatch.setattr(config, "TOOL_LOG", tmp_path / "tool_calls.jsonl")
    monkeypatch.setattr(config, "AUDIT_LOG", tmp_path / "agent_actions.jsonl")
    monkeypatch.setattr(config, "MCP_TRANSCRIPT", tmp_path / "mcp_transcript.jsonl")
    monkeypatch.setenv("MCP_TRANSCRIPT_PATH", str(tmp_path / "mcp_transcript.jsonl"))
    # the LLM Guard classifier (~750 MB model, may need a download) stays out of tests; its tests inject a fake
    monkeypatch.setattr(config, "INJECTION_CLASSIFIER", "off")
    # tests stay offline even when .env enables the Groq fallback: off unless a test turns it on (with fakes)
    monkeypatch.setattr(config, "QUOTA_FALLBACK_PROVIDER", "none")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    yield


@pytest.fixture
def offline_llm():
    """Deterministic model so agent tests are reproducible and cost nothing."""
    from src import llm

    llm.set_llm(llm.OfflineStubChatModel())
    yield
    llm.set_llm(None)
