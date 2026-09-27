"""Tiered memory.

Tier 1 - short-term / working memory: LangGraph `AsyncSqliteSaver` checkpointer. Every node
         transition of a thread (session) is persisted in a SQLite file, so multi-turn context
         ("facts stated earlier in the interaction") survives across turns and process restarts.
Tier 2 - long-term / semantic memory: `LongTermMemory`, a SQLite table of per-patient memories
         keyed by an HMAC patient pseudonym, with sentence-transformer embeddings for semantic
         recall on a return visit (new session / new thread). Memories are extracted with LangMem's
         memory manager (Gemini) when available, else with a deterministic summariser. Only
         PHI-masked text is stored.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

from src import config
from src.guardrails.guards import mask_phi, patient_ref
from src.observability.run_context import now_iso

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- tier 1
@asynccontextmanager
async def open_checkpointer(db_path: Path | str | None = None) -> AsyncIterator[Any]:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = Path(db_path or config.CHECKPOINT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


# ----------------------------------------------------------------------------- tier 2
class LongTermMemory:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or config.LONG_TERM_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, namespace TEXT, kind TEXT, "
                "content TEXT, embedding TEXT, session_id TEXT, created_at TEXT)"
            )
            c.execute("CREATE INDEX IF NOT EXISTS ix_ns ON memories(namespace)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _embed(texts: list[str]) -> np.ndarray:
        from src.tools.rag_tool import get_index

        return np.asarray(get_index().embedder.encode(texts, normalize_embeddings=True), dtype="float32")

    @staticmethod
    def namespace(patient_id: str) -> str:
        return f"patient/{patient_ref(patient_id)}"

    CONSOLIDATE_AT = 0.9  # cosine similarity above which a new memory updates an existing one

    def put(self, patient_id: str, content: str, *, kind: str = "episodic", session_id: str = "") -> str:
        """Insert a memory, or consolidate it into a near-duplicate (the newer statement wins)."""
        content = mask_phi(content)
        emb = self._embed([content])[0]
        ns = self.namespace(patient_id)
        with self._conn() as c:
            rows = c.execute("SELECT id, embedding FROM memories WHERE namespace=?", (ns,)).fetchall()
            if rows:
                sims = np.asarray([json.loads(r[1]) for r in rows], dtype="float32") @ emb
                best = int(np.argmax(sims))
                if float(sims[best]) >= self.CONSOLIDATE_AT:
                    mid = rows[best][0]
                    c.execute("UPDATE memories SET content=?, embedding=?, session_id=?, created_at=? WHERE id=?",
                              (content, json.dumps(emb.tolist()), session_id, now_iso(), mid))
                    return mid
            mid = uuid.uuid4().hex
            c.execute("INSERT INTO memories VALUES (?,?,?,?,?,?,?)",
                      (mid, ns, kind, content, json.dumps(emb.tolist()), session_id, now_iso()))
        return mid

    def search(self, patient_id: str, query: str, k: int = 5) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, kind, content, embedding, session_id, created_at FROM memories "
                             "WHERE namespace=?", (self.namespace(patient_id),)).fetchall()
        if not rows:
            return []
        q = self._embed([query])[0]
        mats = np.asarray([json.loads(r[3]) for r in rows], dtype="float32")
        scores = mats @ q
        order = np.argsort(-scores)[:k]
        return [{"id": rows[i][0], "kind": rows[i][1], "content": rows[i][2], "session_id": rows[i][4],
                 "created_at": rows[i][5], "score": round(float(scores[i]), 4)} for i in order]

    def all_for(self, patient_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT kind, content, session_id, created_at FROM memories WHERE namespace=? "
                             "ORDER BY created_at", (self.namespace(patient_id),)).fetchall()
        return [{"kind": r[0], "content": r[1], "session_id": r[2], "created_at": r[3]} for r in rows]


def deterministic_memories(summary: dict[str, Any]) -> list[str]:
    """Fallback extractor: turn the end-of-turn outcome into durable, PHI-free memory statements."""
    out = []
    if summary.get("risk_level"):
        factors = ", ".join(summary.get("risk_factors", [])[:4])
        out.append(f"Readmission risk assessed as {summary['risk_level']} (LACE {summary.get('lace')}); "
                   f"factors: {factors}.")
    if summary.get("followups"):
        out.append("Proposed follow-ups: " + "; ".join(summary["followups"][:5]) + ".")
    if summary.get("session_facts"):
        out.append("Care logistics stated by clinician: " + json.dumps(summary["session_facts"]) + ".")
    if summary.get("status"):
        out.append(f"Plan status: {summary['status']}. Approval reasons: {', '.join(summary.get('approval_reasons', [])) or 'none'}.")
    if summary.get("intent") and not out:
        out.append(f"Clinician request handled as intent '{summary['intent']}'.")
    return out


async def extract_memories(messages: list[Any], summary: dict[str, Any]) -> tuple[list[str], str]:
    """Use LangMem's memory manager (Gemini) to extract semantic memories from the turn; fall back to
    deterministic extraction. Returns (memories, extractor_name)."""
    fallback = deterministic_memories(summary)
    if not config.GEMINI_API_KEY:
        return fallback, "deterministic"
    try:
        from langmem import create_memory_manager

        from src.llm import get_primary_llm

        manager = create_memory_manager(
            get_primary_llm(),
            instructions=("Extract durable discharge-planning facts about THIS patient's care: risk tier and "
                          "factors, follow-up decisions, caregiver/transport logistics, pending approvals. "
                          "Never include names, MRNs, dates of birth, phone numbers or addresses."),
            enable_inserts=True,
        )
        convo = [{"role": "user" if getattr(m, "type", "") == "human" else "assistant",
                  "content": mask_phi(str(m.content))[:3000]} for m in messages[-4:]]
        convo.append({"role": "user", "content": "Outcome JSON: " + json.dumps(summary, default=str)})
        extracted = await manager.ainvoke({"messages": convo})
        mems = []
        for item in extracted:
            content = getattr(item, "content", item)
            content = getattr(content, "content", content)
            mems.append(mask_phi(content if isinstance(content, str) else json.dumps(content, default=str)))
        mems = [m for m in mems if m.strip()]
        return (mems or fallback), ("langmem" if mems else "deterministic")
    except Exception as exc:
        log.warning("LangMem extraction failed (%s: %s); using deterministic memories", type(exc).__name__,
                    mask_phi(str(exc))[:300])
        return fallback, "deterministic"
