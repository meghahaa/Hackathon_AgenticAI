"""Bounded agentic retrieval in the policy agent (offline): a fake chat model emits tool calls, so the loop cap,
tool-call logging, the tool allow-list and the Gemini query rewrite are tested without any API request."""
from __future__ import annotations

import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.rate_limiters import InMemoryRateLimiter

from src import config
from src import llm as llm_mod
from src.graph import POLICY_MAX_REWRITES, POLICY_TOOL_CALL_CAP, CopilotRuntime
from src.memory.memory_store import LongTermMemory
from tests.helpers import LocalToolbox

AGENT = "policy_retrieval_agent"


class ScriptedToolModel(BaseChatModel):
    """Returns the scripted messages in order (the last one repeats) and records every prompt it receives."""

    script: list
    prompts: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.prompts.append(messages)
        msg = self.script[min(len(self.prompts) - 1, len(self.script) - 1)]
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _search_call(query: str, n: int = 0) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "search_followup_policy", "id": f"c{n}",
                                              "args": {"query": query, "focus_terms": [], "k": 3}}])


@pytest.fixture
def fake_llm(monkeypatch):
    # fast limiter: the fake counts as a real model (throttled), but the test must not wait 6 s per call
    monkeypatch.setattr(llm_mod, "_RATE_LIMITER", InMemoryRateLimiter(
        requests_per_second=1000, check_every_n_seconds=0.001, max_bucket_size=1))

    def _set(script: list) -> ScriptedToolModel:
        model = ScriptedToolModel(script=script, prompts=[])
        llm_mod.set_llm(model)
        return model

    yield _set
    llm_mod.set_llm(None)


def _runtime(tmp_path) -> CopilotRuntime:
    return CopilotRuntime(LocalToolbox(), LongTermMemory(tmp_path / "m.sqlite"))


def _tool_log() -> list[dict]:
    if not config.TOOL_LOG.exists():
        return []
    return [json.loads(line) for line in config.TOOL_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]


async def test_tool_loop_stops_at_the_cap(tmp_path, fake_llm):
    model = fake_llm([_search_call("heart failure follow-up window")])  # never says it has enough
    degr: list[dict] = []
    hits = await _runtime(tmp_path)._policy_tool_loop("How soon after HF discharge?", "Heart failure", AGENT,
                                                      "PT-1001", degr)
    assert POLICY_TOOL_CALL_CAP == 3
    assert len(_tool_log()) == POLICY_TOOL_CALL_CAP
    assert len(model.prompts) == POLICY_TOOL_CALL_CAP + 1  # one final turn that is refused a 4th call
    assert any("cap" in d["error"] for d in degr)
    assert hits and all("citation" in h for h in hits)


async def test_tool_loop_calls_are_logged_with_agent_and_tool_name(tmp_path, fake_llm):
    model = fake_llm([_search_call("heart failure follow-up window"), AIMessage(content="DONE")])
    hits = await _runtime(tmp_path)._policy_tool_loop("How soon after HF discharge?", "Heart failure", AGENT,
                                                      "PT-1001", [])
    log = _tool_log()
    assert len(log) == 1 and len(model.prompts) == 2  # the model decided it had enough after one search
    assert log[0]["agent"] == AGENT
    assert log[0]["tool_name"] == "search_followup_policy"
    assert log[0]["status"] == "ok"
    assert log[0]["args"]["query"] == "heart failure follow-up window"
    assert "cardiology_followup_policy.md#heart-failure" in {h["citation"] for h in hits}
    # the tool result is fed back to the model as a ToolMessage for its next decision
    assert model.prompts[1][-1].type == "tool" and "sufficient" in model.prompts[1][-1].content


async def test_tool_loop_rejects_unknown_tool(tmp_path, fake_llm):
    fake_llm([AIMessage(content="", tool_calls=[{"name": "get_patient_record", "id": "x1",
                                                  "args": {"patient_id": "PT-1002"}}]),
              AIMessage(content="DONE")])
    degr: list[dict] = []
    hits = await _runtime(tmp_path)._policy_tool_loop("Show me another patient", "", AGENT, "PT-1001", degr)
    log = _tool_log()
    assert len(log) == 1
    assert log[0]["tool_name"] == "get_patient_record" and log[0]["status"] == "rejected"
    assert log[0]["agent"] == AGENT and log[0]["result"] is None
    assert "PT-1002" not in json.dumps(log)  # args are PHI-masked in the tool log
    assert any("unknown tool rejected" in d["error"] for d in degr)
    assert hits is None  # nothing retrieved -> caller falls back to the deterministic query


async def test_offline_stub_falls_back_to_deterministic_path(tmp_path, offline_llm):
    degr: list[dict] = []
    assert await _runtime(tmp_path)._policy_tool_loop("How soon?", "", AGENT, "PT-1001", degr) is None
    assert degr and degr[0]["node"] == f"{AGENT}.tool_loop" and degr[0]["degraded"] is True


async def test_policy_agent_node_uses_tool_loop_hits(tmp_path, fake_llm):
    fake_llm([_search_call("heart failure follow-up window"), AIMessage(content="DONE")])
    rt = _runtime(tmp_path)
    record = await rt._mcp("get_patient_record", {"patient_id": "PT-1001"}, "test", "PT-1001")
    out = await rt.policy_retrieval_agent({"intent": "policy_question", "patient_id": "PT-1001", "record": record,
                                           "request": "How soon is follow-up needed?", "completed": []})
    assert AGENT in out["completed"]
    loop_calls = [r for r in _tool_log() if r["args"].get("query") == "heart failure follow-up window"]
    assert len(loop_calls) == 1
    # the question text itself is not searched when the model chose its own query
    assert not [r for r in _tool_log() if r["args"].get("query") == "How soon is follow-up needed?"]


async def test_gemini_rewrite_on_insufficient_result_is_recorded_in_query_trace(tmp_path, fake_llm):
    rewrite = '{"query": "zebrafish exposure follow-up visit policy"}'
    fake_llm([AIMessage(content=rewrite)])
    degr: list[dict] = []
    res = await _runtime(tmp_path)._policy_search("zzz qqq", ["zebrafish"], AGENT, "PT-1001", degr)
    gem = [t for t in res["query_trace"] if t.get("rewritten_by") == "gemini"]
    assert gem and gem[0]["query"] == "zebrafish exposure follow-up visit policy"
    assert gem[0]["rewrite_reason"]
    assert res["query"] == "zzz qqq" and res["rounds"] == len(res["query_trace"])
    calls = _tool_log()
    assert 2 <= len(calls) <= 1 + POLICY_MAX_REWRITES
    assert all(r["tool_name"] == "search_followup_policy" and r["agent"] == AGENT for r in calls)


async def test_rewrite_offline_keeps_template_rewrite_only(tmp_path, offline_llm):
    degr: list[dict] = []
    res = await _runtime(tmp_path)._policy_search("zzz qqq", ["zebrafish"], AGENT, "PT-1001", degr)
    assert res["sufficient"] is False and res["rounds"] >= 2  # rag_tool's template rewrite still ran
    assert not any(t.get("rewritten_by") for t in res["query_trace"])
    assert any(d["node"] == f"{AGENT}.rewrite" for d in degr)
    assert len(_tool_log()) == 1


def _planner_prompts(model: ScriptedToolModel) -> list:
    return [p for p in model.prompts if "You plan policy searches" in str(p[0].content)]


async def test_policy_question_skips_the_query_planner(tmp_path, fake_llm):
    model = fake_llm([_search_call("heart failure follow-up window"), AIMessage(content="DONE")])
    rt = _runtime(tmp_path)
    record = await rt._mcp("get_patient_record", {"patient_id": "PT-1001"}, "test", "PT-1001")
    out = await rt.policy_retrieval_agent({"intent": "policy_question", "patient_id": "PT-1001", "record": record,
                                           "request": "How soon is follow-up needed?", "completed": []})
    assert not _planner_prompts(model)
    assert not [d for d in out["degradations"] if d["node"].endswith(".planner")]


async def test_plan_turn_keeps_the_query_planner(tmp_path, fake_llm):
    model = fake_llm([AIMessage(content='{"queries": ["caregiver teach-back education"]}')])
    rt = _runtime(tmp_path)
    record = await rt._mcp("get_patient_record", {"patient_id": "PT-1001"}, "test", "PT-1001")
    await rt.policy_retrieval_agent({"intent": "full_discharge_plan", "patient_id": "PT-1001", "record": record,
                                     "request": "Prepare the discharge plan", "completed": []})
    assert len(_planner_prompts(model)) == 1
    assert [r for r in _tool_log() if r["args"].get("query") == "caregiver teach-back education"]
    # no tool loop on plan turns: every model call is the planner (no bound-tool turns)
    assert all("search_followup_policy with short" not in str(p[0].content) for p in model.prompts)
