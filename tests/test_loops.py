"""Loop / cascade guard tests: a runaway worker cannot spin the graph forever."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph

from src import config
from src.graph import CopilotRuntime, build_graph, decide_next, route_from_supervisor, run_config
from src.memory.memory_store import LongTermMemory
from tests.helpers import LocalToolbox


def test_decide_next_escalates_when_step_budget_exhausted():
    state = {"intent": "full_discharge_plan", "completed": [], "step_count": config.MAX_AGENT_STEPS}
    assert decide_next(state) == "escalate"


def test_conditional_edge_escalates_past_budget_even_with_stale_next_node():
    state = {"next_node": "readmission_risk_agent", "step_count": config.MAX_AGENT_STEPS + 1}
    assert route_from_supervisor(state) == "escalate"


class StuckWorkerRuntime(CopilotRuntime):
    """A faulty worker that never reports completion -> the supervisor would re-route to it forever."""

    calls = 0

    async def readmission_risk_agent(self, state):
        StuckWorkerRuntime.calls += 1
        return {"audit_trail": [{"node": "stuck_worker"}]}


async def test_step_guard_stops_runaway_worker_loop(tmp_path, offline_llm):
    StuckWorkerRuntime.calls = 0
    graph = build_graph(StuckWorkerRuntime(LocalToolbox(), LongTermMemory(tmp_path / "m.sqlite")))
    state = await graph.ainvoke({"messages": [HumanMessage("What is the readmission risk?")], "patient_id": "PT-1004"},
                                run_config("loop", "t"))
    assert state["status"] == "ESCALATED_TO_HUMAN"
    assert "loop guard" in state["final_response"]
    # the supervisor routed to the stuck worker at most MAX_AGENT_STEPS times, then cut it off
    assert 1 <= StuckWorkerRuntime.calls <= config.MAX_AGENT_STEPS
    assert state["step_count"] == config.MAX_AGENT_STEPS + 1


def test_recursion_limit_backstop_raises_on_unbounded_cycle():
    """The hard LangGraph recursion_limit (used in run_config) stops a graph with no exit condition."""

    class S(TypedDict):
        n: Annotated[int, operator.add]

    g = StateGraph(S)
    g.add_node("a", lambda s: {"n": 1})
    g.add_node("b", lambda s: {"n": 1})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    app = g.compile()
    with pytest.raises(GraphRecursionError):
        app.invoke({"n": 0}, {"recursion_limit": 6})


def test_run_config_sets_recursion_limit_above_step_budget():
    cfg = run_config("s", "r")
    assert cfg["recursion_limit"] == config.RECURSION_LIMIT
    assert config.RECURSION_LIMIT > 2 * config.MAX_AGENT_STEPS + 2  # soft guard trips first
