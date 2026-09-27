"""PII redaction demo: which masking layer (registry / regex / presidio) catches which identifier.

Three synthetic strings go through src.guardrails.guards.mask_phi:
  1. direct identifiers of a synthetic patient (name, MRN, DOB, phone, address)  -> identifier registry
  2. identifiers that are in no record (phone, e-mail, SSN)                       -> regex patterns
  3. a person and a place that are in no record and match no pattern             -> Presidio NER (spaCy)
The report never contains a raw value: "before" is a summary with each identifier replaced by <TYPE:n chars>,
and anything a layer missed is replaced by <MISSED:TYPE> in "after" (and counted as missed).

Run:  python scripts/pii_redaction_demo.py [--out reports/pii_redaction_sample.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.guardrails.guards import _identifier_registry, mask_phi, ner_status  # noqa: E402


def samples() -> list[dict]:
    demo = json.loads((config.PATIENTS_DIR / "patient_PT-1001.json").read_text(encoding="utf-8"))["demographics"]
    return [
        {"id": "S1", "expected_layer": "registry",
         "parts": ["Discharge summary for ", ("NAME", demo["name"]), " (", ("MRN", demo["mrn"]), "), DOB ",
                   ("DOB", demo["dob"]), ". Reach the patient on ", ("PHONE", demo["phone"]), " at ",
                   ("ADDRESS", demo["address"]), "."]},
        {"id": "S2", "expected_layer": "regex",
         "parts": ["Caregiver line changed to ", ("PHONE", "(555) 014-2297"), "; send the schedule to ",
                   ("EMAIL", "family.contact@example.org"), ". SSN on file ", ("SSN", "900-12-3456"), "."]},
        {"id": "S3", "expected_layer": "presidio",
         "parts": ["Her daughter ", ("PERSON", "Zebulon Farquhar"), " will drive her to clinic; they live in ",
                   ("LOCATION", "Springfield"), "."]},
    ]


def run(sample: dict) -> dict:
    text = "".join(p if isinstance(p, str) else p[1] for p in sample["parts"])
    idents = [p for p in sample["parts"] if not isinstance(p, str)]
    registry = {literal for literal, _ in _identifier_registry()}
    no_ner = mask_phi(text, keep_patient_ids=True)
    hits: list[dict] = []
    after = mask_phi(text, use_ner=True, keep_patient_ids=True, hits=hits)
    caught = []
    for kind, value in idents:
        if value not in no_ner:
            layer = "registry" if value in registry else "regex"
        elif value not in after:
            layer = "presidio"
        else:
            layer = "missed"
            after = after.replace(value, f"<MISSED:{kind}>")  # never write a raw value to the report
        caught.append({"type": kind, "chars": len(value), "layer": layer})
    before = "".join(p if isinstance(p, str) else f"<{p[0]}:{len(p[1])} chars>" for p in sample["parts"])
    return {"id": sample["id"], "expected_layer": sample["expected_layer"], "before_summary": before,
            "after": after, "identifiers": caught,
            "mask_phi_hits": [{"layer": h["layer"], "label": h["label"]} for h in hits],
            "all_masked": all(c["layer"] != "missed" for c in caught)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(config.REPORTS_DIR / "pii_redaction_sample.json"))
    args = p.parse_args()
    rows = [run(s) for s in samples()]
    by_layer: dict[str, int] = {}
    for r in rows:
        for c in r["identifiers"]:
            by_layer[c["layer"]] = by_layer.get(c["layer"], 0) + 1
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "generator": "scripts/pii_redaction_demo.py", "function": "src.guardrails.guards.mask_phi(use_ner=True)",
              "ner": ner_status(),
              "note": "before_summary replaces every identifier with <TYPE:n chars>; raw values are never written.",
              "summary": {"samples": len(rows), "identifiers": sum(len(r["identifiers"]) for r in rows),
                          "caught_by_layer": by_layer, "missed": by_layer.get("missed", 0)},
              "samples": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
