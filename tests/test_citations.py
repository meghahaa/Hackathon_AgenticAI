"""Citation-Resolves rule: every evidence id or path cited in docs/ must resolve to a committed artifact."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted((ROOT / "docs").glob("*.md"))
EVIDENCE = [ROOT / "traces" / "phoenix_spans.jsonl", ROOT / "logs" / "tool_calls.jsonl",
            ROOT / "logs" / "agent_actions.jsonl", ROOT / "logs" / "mcp_transcript.jsonl",
            ROOT / "reports" / "eval_report.json", ROOT / "reports" / "cli_run_results.json",
            ROOT / "reports" / "failure_candidates.json", ROOT / "reports" / "redteam_results.json", *sorted((ROOT / "reports" / "optimization").glob("*")),
            *sorted((ROOT / "reports" / "failure_evidence").rglob("*.*"))]

SPAN_RE = re.compile(r"span_id[`*:\s=]*`?([0-9a-f]{16})\b")
TRACE_RE = re.compile(r"\b([0-9a-f]{32})\b")
RUN_RE = re.compile(r"\b(run-[0-9a-f]{12})\b")
PATH_RE = re.compile(r"\]\(\.\./([^)#\s]+)")  # markdown links like [x](../src/graph.py#L10)


def _corpus() -> str:
    parts = []
    for p in EVIDENCE:
        if p.exists() and p.suffix in (".jsonl", ".json", ".csv", ".log"):
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
        elif p.exists() and p.suffix == ".parquet":
            import pandas as pd

            parts.append(pd.read_parquet(p).to_json())
    return "\n".join(parts)


@pytest.fixture(scope="module")
def corpus() -> str:
    return _corpus()


@pytest.mark.parametrize("doc", DOCS, ids=[d.name for d in DOCS])
def test_cited_ids_resolve(doc, corpus):
    text = doc.read_text(encoding="utf-8")
    ids = set(SPAN_RE.findall(text)) | set(TRACE_RE.findall(text)) | set(RUN_RE.findall(text))
    missing = sorted(i for i in ids if i not in corpus)
    assert not missing, f"{doc.name} cites ids not present in committed evidence: {missing}"


@pytest.mark.parametrize("doc", DOCS, ids=[d.name for d in DOCS])
def test_cited_paths_exist(doc):
    text = doc.read_text(encoding="utf-8")
    missing = sorted({p for p in PATH_RE.findall(text) if not (ROOT / p).exists()})
    assert not missing, f"{doc.name} links to missing files: {missing}"
