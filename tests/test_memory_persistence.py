"""Tiered-memory test: short-term (checkpointed thread state) and long-term (semantic, cross-session)
memory both survive a full teardown and re-open of the stores. Writes logs/memory_test.log."""
from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

from src import config
from src.graph import CopilotRuntime, build_graph, run_config
from src.memory.memory_store import LongTermMemory, open_checkpointer
from tests.helpers import LocalToolbox

PID = "PT-1003"


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, msg: str) -> None:
        self.lines.append(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}")

    def write(self) -> None:
        config.MEMORY_TEST_LOG.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


async def _turn(graph, thread, msg):
    return await graph.ainvoke({"messages": [HumanMessage(msg)], "patient_id": PID}, run_config(thread, "mem-test"))


async def test_cross_session_memory_persistence(tmp_path, offline_llm):
    log = _Log()
    ckpt_db, ltm_db = tmp_path / "checkpoints.sqlite", tmp_path / "ltm.sqlite"
    log("TEST test_cross_session_memory_persistence (model=offline-stub, deterministic)")
    log(f"stores: checkpointer={ckpt_db.name} (AsyncSqliteSaver)  long_term={ltm_db.name} (SQLite+embeddings)")
    try:
        # ---------------- session A: three turns in one thread (short-term memory)
        async with open_checkpointer(ckpt_db) as saver:
            graph = build_graph(CopilotRuntime(LocalToolbox(), LongTermMemory(ltm_db)), checkpointer=saver)
            s1 = await _turn(graph, "session-A", "Please assemble the discharge plan for this patient.")
            log(f"A.t1 intent={s1['intent']} risk={s1['risk_level']} status={s1['status']}")
            assert s1["risk_level"] == "HIGH"
            s2 = await _turn(graph, "session-A", "Her daughter will drive her to appointments, she is available on Tuesdays.")
            log(f"A.t2 session_facts={s2['session_facts']}")
            s3 = await _turn(graph, "session-A", "Now redraft the follow-up schedule using that.")
            appts = s3["discharge_plan"]["follow_up_appointments"]
            tuesday = [a for a in appts if a["proposed_slot"] and a["proposed_slot"].startswith("Tuesday")]
            log(f"A.t3 uses earlier-turn facts: {len(tuesday)}/{len(appts)} appointments on Tuesday; "
                f"logistics={s3['discharge_plan']['home_care_and_logistics']}")
            assert s3["session_facts"]["preferred_weekday"] == "Tuesday"
            assert s3["session_facts"]["transport_by"] == "daughter"
            assert tuesday, "preferred weekday from an earlier turn was not applied"
            assert any("daughter" in h for h in s3["discharge_plan"]["home_care_and_logistics"])
            n_msgs = len(s3["messages"])
            log(f"A thread holds {n_msgs} messages after 3 turns")
        log("teardown: checkpointer connection closed, graph and runtime discarded")

        # ---------------- re-open short-term store: thread state survived
        async with open_checkpointer(ckpt_db) as saver2:
            graph2 = build_graph(CopilotRuntime(LocalToolbox(), LongTermMemory(ltm_db)), checkpointer=saver2)
            snap = await graph2.aget_state({"configurable": {"thread_id": "session-A"}})
            log(f"re-opened checkpointer: thread session-A messages={len(snap.values['messages'])} "
                f"facts={snap.values['session_facts']}")
            assert len(snap.values["messages"]) == n_msgs
            assert snap.values["session_facts"]["preferred_weekday"] == "Tuesday"

            # ---------------- session B: new thread, return visit -> long-term recall
            fresh = LongTermMemory(ltm_db)
            stored = fresh.all_for(PID)
            log(f"long-term store holds {len(stored)} memories for {fresh.namespace(PID)}")
            assert stored and all("PT-1003" not in m["content"] for m in stored)
            b = await _turn(graph2, "session-B", "Returning to this patient - what did we decide last time?")
            log(f"B.t1 intent={b['intent']} recalled={len(b['memory_recall'])} items")
            log("B.t1 response: " + b["final_response"].replace("\n", " | ")[:600])
            assert b["intent"] == "recall_prior"
            assert "HIGH" in b["final_response"]
            assert "Tuesday" in b["final_response"] or "daughter" in b["final_response"]
            empty = await graph2.aget_state({"configurable": {"thread_id": "session-B"}})
            assert len(empty.values["messages"]) == 2  # new thread did not inherit session-A messages
        log("RESULT PASS: short-term facts reused within session; thread state survived store re-open; "
            "prior-session context recalled from long-term memory in a new thread")
    except AssertionError as exc:
        log(f"RESULT FAIL: {exc}")
        raise
    finally:
        log.write()
