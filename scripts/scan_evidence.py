"""PHI + secrets scan over every committed artifact (NFR-01, NFR-05) -> reports/phi_secret_scan.json

Checks logs/, traces/, reports/ and docs/ for:
  * direct identifiers of the synthetic patients (names, MRNs, DOBs, phones, addresses)
  * raw patient IDs, canonical (PT-####) or loose ("PT 1002", "PT1002", "patient 1002", "patient #1003", ...;
    src.guardrails.guards.LOOSE_PATIENT_ID_RE) - evidence and docs must use HMAC pseudonyms (pref_...)
  * secrets: Google and Groq (gsk_...) API key patterns and the literal values of GEMINI_API_KEY / GROQ_API_KEY
Also verifies .gitignore covers .env and that .env.example holds no real key.
Run:  python scripts/scan_evidence.py      (exit code 1 if anything is found)
      --dirs <dir> ...  scans other directories instead (regenerate_evidence.py --quick: its state/quick_run/ outputs)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.guardrails.guards import _identifier_registry, find_patient_ids  # noqa: E402

SCAN_DIRS = ["logs", "traces", "reports", "docs"]
SECRET_RES = [re.compile(r"AIza[0-9A-Za-z_\-]{35}"), re.compile(r"\bAQ\.[0-9A-Za-z_\-]{20,}"),
              re.compile(r"\bgsk_[0-9A-Za-z]{20,}")]


def _text(p: Path) -> str:
    if p.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(p).to_json(date_format="iso")
    if p.suffix in (".png", ".jpg"):
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")


def scan(dirs: list[str] | None = None) -> dict:
    dirs = dirs or SCAN_DIRS
    registry = _identifier_registry()
    key = config.GEMINI_API_KEY
    groq_key = config.GROQ_API_KEY
    findings, scanned = [], 0
    for d in dirs:
        for p in sorted((ROOT / d).rglob("*")):
            if not p.is_file():
                continue
            scanned += 1
            text = _text(p)
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            for literal, label in registry:
                if len(literal) > 3 and literal in text:
                    findings.append({"file": rel, "type": f"PHI:{label}", "sample": literal[:3] + "***"})
            ids = set(find_patient_ids(text))
            if ids:
                loose = [i for i in ids if not re.fullmatch(r"PT-\d{4}", i, re.I)]
                findings.append({"file": rel, "type": "PATIENT_ID", "count": len(ids), "loose_forms": len(loose)})
            for rx in SECRET_RES:
                if rx.search(text):
                    findings.append({"file": rel, "type": "SECRET_PATTERN"})
            if key and key in text:
                findings.append({"file": rel, "type": "SECRET_LITERAL_GEMINI_API_KEY"})
            if groq_key and groq_key in text:
                findings.append({"file": rel, "type": "SECRET_LITERAL_GROQ_API_KEY"})
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    hygiene = {
        "gitignore_covers_env": bool(re.search(r"^\.env$", gitignore, re.M)),
        "env_example_has_placeholder_key": "GEMINI_API_KEY=your_gemini_api_key_here" in example,
        "env_example_contains_real_key": bool((key and key in example) or (groq_key and groq_key in example)),
    }
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "generator": "scripts/scan_evidence.py", "files_scanned": scanned, "dirs": dirs,
            "identifier_registry_size": len(registry), "findings": findings, "secrets_hygiene": hygiene,
            "clean": not findings and hygiene["gitignore_covers_env"] and not hygiene["env_example_contains_real_key"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", default=None,
                   help=f"directories to scan, relative to the repo root (default: {' '.join(SCAN_DIRS)})")
    out = scan(p.parse_args().dirs)
    (config.REPORTS_DIR / "phi_secret_scan.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("files_scanned", "clean", "secrets_hygiene")}, indent=2))
    for f in out["findings"][:20]:
        print("FINDING", f)
    sys.exit(0 if out["clean"] else 1)


if __name__ == "__main__":
    main()
