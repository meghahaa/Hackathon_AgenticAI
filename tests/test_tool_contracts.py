"""Tool-contract tests: input/output schema of every tool + at least one error path each.
MCP tools are exercised through the real stdio server via langchain-mcp-adapters."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src import config
from src.observability.tool_logger import ToolScopeError, call_tool
from src.tools.mcp_client import MCPToolbox
from src.tools.rag_tool import (PolicySearchOutput, _HashEmbedder, agentic_policy_search, get_index,
                                search_followup_policy)

# ------------------------------------------------------------------ agentic-RAG tool


def test_rag_tool_input_schema():
    schema = search_followup_policy.args_schema.model_json_schema()
    assert set(schema["properties"]) == {"query", "focus_terms", "k"}
    assert schema["required"] == ["query"]
    assert schema["properties"]["k"]["maximum"] == 8


async def test_rag_tool_output_schema_and_citations():
    out = await search_followup_policy.ainvoke({"query": "heart failure follow-up", "focus_terms": ["heart failure"]})
    parsed = PolicySearchOutput.model_validate(out)
    assert parsed.sufficient is True
    hf = "cardiology_followup_policy.md#heart-failure"
    if isinstance(get_index().embedder, _HashEmbedder):  # model not loadable here: degraded, coarser ranking
        assert parsed.embedder == "hashed-bow-fallback"  # the degradation is reported in the result
        assert hf in [r.citation for r in parsed.results]  # within the top-k
    else:  # sentence-transformers
        assert parsed.embedder == config.EMBED_MODEL
        assert parsed.results[0].citation == hf
    assert all("#" in r.citation and r.citation.split("#")[0].endswith(".md") for r in parsed.results)


def test_rag_tool_rewrites_query_when_grade_fails():
    out = agentic_policy_search("zzz qqq", focus_terms=["ketoacidosis"])
    assert out["rounds"] >= 2  # retrieval-in-the-loop: rewrite happened
    assert out["query_trace"][0]["grade"] != "sufficient"


async def test_rag_tool_error_path_rejects_invalid_input():
    with pytest.raises(ValidationError):
        await search_followup_policy.ainvoke({"query": "", "k": 3})
    with pytest.raises(ValidationError):
        await search_followup_policy.ainvoke({"query": "copd follow-up", "k": 99})
    assert agentic_policy_search("   ") == {"error": "empty_query"}


# ------------------------------------------------------------------ MCP tools (real stdio server)
# NB: the MCP stdio session is opened inside each test (not an async-generator fixture) because
# anyio cancel scopes must be exited in the task that entered them.


def _parse(raw):
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, list):
        raw = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
    return json.loads(raw) if isinstance(raw, str) else raw


async def test_mcp_server_exposes_expected_tools_and_resource():
    async with MCPToolbox() as toolbox:
        assert set(toolbox.tools) == {"get_patient_record", "get_prior_admissions", "find_followup_slots"}
        checklist = await toolbox.read_resource("resource://hospital/discharge_checklist")
        assert "Discharge Checklist" in checklist and "POL-MED-08" in checklist


async def test_mcp_tool_input_schemas():
    async with MCPToolbox() as toolbox:
        def props(name):
            s = toolbox.tool(name).args_schema
            return s if isinstance(s, dict) else s.model_json_schema()

        rec = props("get_patient_record")
        assert rec["required"] == ["patient_id"] and rec["properties"]["patient_id"]["type"] == "string"
        slots = props("find_followup_slots")
        assert set(slots["required"]) == {"specialty", "earliest_date", "latest_date"}
        assert {"preferred_weekday", "telehealth_only", "limit"} <= set(slots["properties"])


async def test_get_patient_record_output_is_deidentified():
    async with MCPToolbox() as toolbox:
        out = _parse(await toolbox.tool("get_patient_record").ainvoke({"patient_id": "PT-1001"}))
        assert "demographics" not in out
        assert out["patient_ref"].startswith("pref_")
        for key in ("discharge_diagnoses", "discharge_medication_orders", "social", "admission"):
            assert key in out
        assert "Sundaram" not in json.dumps(out) and "MRN-40017721" not in json.dumps(out)


async def test_find_followup_slots_output_contract():
    async with MCPToolbox() as toolbox:
        out = _parse(await toolbox.tool("find_followup_slots").ainvoke(
            {"specialty": "cardiology", "earliest_date": "2026-09-21", "latest_date": "2026-10-04",
             "preferred_weekday": "Tuesday"}))
        assert set(out) >= {"specialty", "window", "slots", "total_open"}
        assert len(out["slots"]) <= 3
        for s in out["slots"]:
            assert set(s) == {"date", "weekday", "time", "telehealth_available"}
            assert "2026-09-21" <= s["date"] <= "2026-10-04"


@pytest.mark.parametrize("tool, args, error", [
    ("get_patient_record", {"patient_id": "PT-9999"}, "patient_not_found"),
    ("get_prior_admissions", {"patient_id": "PT-0000"}, "patient_not_found"),
    ("find_followup_slots", {"specialty": "astrology", "earliest_date": "2026-09-21", "latest_date": "2026-09-30"},
     "unknown_specialty"),
    ("find_followup_slots", {"specialty": "cardiology", "earliest_date": "21/09/2026", "latest_date": "2026-09-30"},
     "invalid_date"),
])
async def test_mcp_tool_error_paths(tool, args, error):
    async with MCPToolbox() as toolbox:
        out = _parse(await toolbox.tool(tool).ainvoke(args))
        assert out["error"] == error


async def test_logging_middleware_blocks_out_of_scope_patient_and_logs_it(tmp_path):
    async with MCPToolbox() as toolbox:
        from src import config

        with pytest.raises(ToolScopeError):
            await call_tool(toolbox.tool("get_patient_record"), {"patient_id": "PT-1003"}, agent="test",
                            active_patient_id="PT-1001")
        ok = await call_tool(toolbox.tool("get_patient_record"), {"patient_id": "PT-1001"}, agent="test",
                             active_patient_id="PT-1001")
        assert ok["patient_ref"].startswith("pref_")
        rows = [json.loads(l) for l in config.TOOL_LOG.read_text(encoding="utf-8").splitlines()]
        assert [r["status"] for r in rows] == ["blocked", "ok"]
        required = {"timestamp", "agent", "tool_name", "args", "result", "latency_ms", "status"}
        assert all(required <= set(r) for r in rows)
        assert "PT-1003" not in config.TOOL_LOG.read_text(encoding="utf-8")  # identifiers pseudonymised in logs
