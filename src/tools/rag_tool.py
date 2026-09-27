"""Agentic-RAG tool over the synthetic discharge & follow-up policy corpus.

Retrieval-in-the-loop: retrieve -> grade (score + term coverage) -> if insufficient, rewrite the
query with the focus terms / section vocabulary and retrieve again (max 3 rounds). Every returned
chunk carries a citation id `<file>#<section-anchor>` that the agents must cite verbatim.

Index: Sentence-Transformers (all-MiniLM-L6-v2, local) + FAISS inner-product over normalised vectors.
If the embedding model cannot load (offline machine) a hashed bag-of-words embedder is used instead
and the degradation is reported in every result (`embedder` field).
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src import config

log = logging.getLogger(__name__)
GRADE_MIN_SCORE = 0.35
MAX_ROUNDS = 3


@dataclass
class Chunk:
    citation: str
    doc: str
    policy_id: str
    section: str
    text: str


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load_corpus() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(config.POLICY_DIR.glob("*.md")):
        body = path.read_text(encoding="utf-8")
        pid = re.search(r"Policy ID:\s*([A-Z0-9-]+)", body)
        policy_id = pid.group(1) if pid else path.stem
        for sec in re.split(r"\n(?=## )", body):
            if not sec.startswith("## "):
                continue
            title, _, text = sec[3:].partition("\n")
            chunks.append(Chunk(f"{path.name}#{_slug(title)}", path.name, policy_id, title.strip(),
                                " ".join(text.split())))
    return chunks


class _HashEmbedder:
    """Deterministic fallback embedder (degraded mode only)."""

    dim = 512

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out


class PolicyIndex:
    def __init__(self) -> None:
        import faiss

        self.chunks = load_corpus()
        try:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(config.EMBED_MODEL, device="cpu")
            self.embedder_name = config.EMBED_MODEL
        except Exception as exc:  # no network / model cache
            log.warning("sentence-transformers unavailable (%s); using hashed embedder", type(exc).__name__)
            self.embedder, self.embedder_name = _HashEmbedder(), "hashed-bow-fallback"
        docs = [f"{c.section}. {c.text}" for c in self.chunks]
        vecs = np.asarray(self.embedder.encode(docs, normalize_embeddings=True), dtype="float32")
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = np.asarray(self.embedder.encode([query], normalize_embeddings=True), dtype="float32")
        scores, idx = self.index.search(q, min(k, len(self.chunks)))
        return [
            {"citation": self.chunks[i].citation, "policy_id": self.chunks[i].policy_id,
             "section": self.chunks[i].section, "score": round(float(s), 4), "text": self.chunks[i].text}
            for s, i in zip(scores[0], idx[0]) if i >= 0
        ]


@lru_cache(maxsize=1)
def get_index() -> PolicyIndex:
    return PolicyIndex()


def _grade(results: list[dict], focus_terms: list[str]) -> tuple[bool, str]:
    if not results:
        return False, "no results"
    top = results[0]["score"]
    if top < GRADE_MIN_SCORE:
        return False, f"top score {top:.2f} < {GRADE_MIN_SCORE}"
    if focus_terms:
        blob = " ".join(r["text"].lower() + " " + r["section"].lower() for r in results)
        missing = [t for t in focus_terms if t.lower() not in blob]
        if len(missing) == len(focus_terms):
            return False, f"no focus term covered ({', '.join(missing)})"
    return True, "sufficient"


def agentic_policy_search(query: str, focus_terms: list[str] | None = None, k: int = 4) -> dict[str, Any]:
    """Retrieve -> grade -> rewrite loop. Returns results plus the query trace for observability."""
    if not query or not query.strip():
        return {"error": "empty_query"}
    focus_terms = [t for t in (focus_terms or []) if t]
    idx = get_index()
    trace, q = [], query
    results: list[dict] = []
    ok = False
    for round_no in range(1, MAX_ROUNDS + 1):
        results = idx.search(q, k)
        ok, why = _grade(results, focus_terms)
        trace.append({"round": round_no, "query": q, "top_score": results[0]["score"] if results else None,
                      "grade": why})
        if ok:
            break
        # rewrite: anchor on the focus terms and policy vocabulary
        q = f"{' '.join(focus_terms)} follow-up policy discharge {query}" if round_no == 1 else \
            f"{' '.join(focus_terms)} within days visit referral red-flag"
    return {"query": query, "sufficient": ok, "rounds": len(trace), "query_trace": trace,
            "embedder": idx.embedder_name, "results": results}


class PolicySearchInput(BaseModel):
    query: str = Field(..., min_length=3, description="Natural-language policy question")
    focus_terms: list[str] = Field(default_factory=list, description="Condition keywords that must be covered")
    k: int = Field(4, ge=1, le=8, description="Chunks to return")


class PolicyHit(BaseModel):
    citation: str
    policy_id: str
    section: str
    score: float
    text: str


class PolicySearchOutput(BaseModel):
    query: str
    sufficient: bool
    rounds: int
    query_trace: list[dict]
    embedder: str
    results: list[PolicyHit]


def _run(query: str, focus_terms: list[str] | None = None, k: int = 4) -> dict:
    out = agentic_policy_search(query, focus_terms, k)
    if "error" in out:
        return out
    return PolicySearchOutput.model_validate(out).model_dump()


search_followup_policy = StructuredTool.from_function(
    func=_run,
    name="search_followup_policy",
    description="Agentic retrieval over the synthetic discharge/follow-up policy corpus. Returns cited chunks.",
    args_schema=PolicySearchInput,
)
