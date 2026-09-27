"""Two operational fixes: (1) on Windows the in-process Phoenix launch failed on a cp1252 stdout ('charmap' codec
can't encode ...), so tracing switches the process streams to UTF-8 and sets PYTHONUTF8=1 for child interpreters;
(2) a LangMem extraction failure logs its (PHI-masked) error message, not only the exception class."""
from __future__ import annotations

import io
import logging
import os
import sys

import pytest

from src import config
from src.guardrails.guards import patient_ref
from src.observability import tracing


def test_use_utf8_output_fixes_a_cp1252_stream(monkeypatch):
    raw = io.BytesIO()
    cp1252 = io.TextIOWrapper(raw, encoding="cp1252")
    with pytest.raises(UnicodeEncodeError):
        cp1252.write("✨ Phoenix")  # what Phoenix prints at launch
    cp1252 = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", cp1252)
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    tracing.use_utf8_output()
    assert sys.stdout.encoding.lower() == "utf-8"
    sys.stdout.write("✨ Phoenix")  # no longer raises
    assert os.environ["PYTHONUTF8"] == "1"  # inherited by `phoenix serve` and other child interpreters


async def test_langmem_failure_logs_the_masked_error_message(monkeypatch, caplog):
    pytest.importorskip("langmem")
    import langmem

    from src import llm as llm_mod
    from src.memory.memory_store import extract_memories

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key-offline-test")
    llm_mod.set_llm(None)

    def boom(*a, **k):
        raise RuntimeError("503 UNAVAILABLE while extracting memories for PT-1001")

    monkeypatch.setattr(langmem, "create_memory_manager", boom)
    with caplog.at_level(logging.WARNING, logger="src.memory.memory_store"):
        mems, extractor = await extract_memories([], {"intent": "full_discharge_plan", "status": "DRAFT"})
    llm_mod.set_llm(None)
    assert extractor == "deterministic" and mems
    (msg,) = [r.getMessage() for r in caplog.records if "LangMem extraction failed" in r.getMessage()]
    assert "503 UNAVAILABLE" in msg and patient_ref("PT-1001") in msg and "PT-1001" not in msg
