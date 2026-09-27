"""Freeze the evidence of one run into reports/failure_evidence/<name>/ so failure-analysis citations
keep resolving after later regenerations overwrite logs/ and traces/.

The copy is pseudonymised on the way: any patient ID (PT-#### or PT####, e.g. inside an old session
label) is replaced by its HMAC pseudonym. Span/trace/run ids are untouched, so citations still resolve.
A manifest records the source files, their SHA-256 before/after and the transform applied.

    python scripts/snapshot_run.py validation_run
    python scripts/snapshot_run.py redteam_v1 state/redteam_live \
        --files redteam_results.json tool_calls.jsonl agent_actions.jsonl mcp_transcript.jsonl \
        --spans state/spans_local.jsonl

--files copies an ad-hoc run kept outside the evidence dirs (paths relative to the source root). --spans copies
only the spans whose trace id appears in the snapshot's copied files (the local span sink is shared by all runs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.guardrails.guards import patient_ref  # noqa: E402

PID_RE = re.compile(r"\bPT-?(\d{4})\b")
TRACE_ID_RE = re.compile(r'"trace_id": "([0-9a-f]{32})"')
SOURCES = ["traces/phoenix_spans.parquet", "reports/cli_run_results.json", "reports/eval_report.json",
           "reports/golden_signals.json", "logs/tool_calls.jsonl", "logs/agent_actions.jsonl",
           "reports/dashboard_data.csv", "reports/phi_secret_scan.json", "reports/pytest_report.txt",
           "reports/dashboard.png", "reports/dashboard_chart.png"]
BINARY = {".png"}  # copied byte-for-byte (screenshots/charts contain no text identifiers to rewrite)


def pseudonymise(text: str) -> tuple[str, int]:
    return PID_RE.subn(lambda m: patient_ref(f"PT-{m.group(1)}"), text)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


def _referenced_trace_ids(files: list[Path]) -> set[str]:
    ids: set[str] = set()
    for f in files:
        if f.suffix in (".json", ".jsonl"):
            ids |= set(TRACE_ID_RE.findall(f.read_text(encoding="utf-8")))
    return ids


def snapshot(name: str, src_root: Path = ROOT, sources: list[str] | None = None,
             spans_jsonl: Path | None = None) -> dict:
    out_dir = ROOT / "reports" / "failure_evidence" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "generator": "scripts/snapshot_run.py", "transform": "PT-#### / PT#### -> HMAC patient_ref",
                "source_root": _rel(src_root) or ".", "files": []}
    for rel in sources or SOURCES:
        src = src_root / rel
        if not src.exists():
            continue
        dst = out_dir / Path(rel).name
        replaced = 0
        if src.suffix in BINARY:
            dst.write_bytes(src.read_bytes())
        elif src.suffix == ".parquet":
            df = pd.read_parquet(src)
            for c in df.columns:
                if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                    def _fix(v):
                        nonlocal replaced
                        if isinstance(v, str):
                            v, n = pseudonymise(v)
                            replaced += n
                        return v
                    df[c] = df[c].map(_fix)
            df.to_parquet(dst, index=False)
        else:
            text, replaced = pseudonymise(src.read_text(encoding="utf-8"))
            dst.write_text(text, encoding="utf-8")
        manifest["files"].append({"source": rel, "snapshot": str(dst.relative_to(ROOT)).replace("\\", "/"),
                                  "sha256_source": sha(src), "sha256_snapshot": sha(dst),
                                  "patient_ids_pseudonymised": replaced})
    if spans_jsonl is not None and Path(spans_jsonl).exists():
        ids = _referenced_trace_ids([ROOT / f["snapshot"] for f in manifest["files"]])
        kept = [line for line in Path(spans_jsonl).read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("context.trace_id") in ids]
        dst = out_dir / "spans.jsonl"
        text, replaced = pseudonymise("".join(line + "\n" for line in kept))
        dst.write_text(text, encoding="utf-8")
        manifest["files"].append({"source": _rel(spans_jsonl), "snapshot": _rel(dst),
                                  "sha256_source": sha(Path(spans_jsonl)), "sha256_snapshot": sha(dst),
                                  "filter": "spans whose trace id is referenced by the snapshot's other files",
                                  "trace_ids_referenced": len(ids), "spans_kept": len(kept),
                                  "patient_ids_pseudonymised": replaced})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Freeze one run's evidence into reports/failure_evidence/<name>/")
    ap.add_argument("name", nargs="?", default="run")
    ap.add_argument("src_root", nargs="?", default=str(ROOT))
    ap.add_argument("--files", nargs="+", default=None, help="paths relative to src_root (default: the evidence set)")
    ap.add_argument("--spans", default=None, help="span JSONL to filter down to the snapshot's trace ids")
    a = ap.parse_args()
    print(json.dumps(snapshot(a.name, Path(a.src_root), a.files, Path(a.spans) if a.spans else None), indent=2))
