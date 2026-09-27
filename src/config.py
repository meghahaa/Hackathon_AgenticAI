"""Central configuration. All secrets come from environment variables (.env), never from code."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
PATIENTS_DIR = DATA_DIR / "synthetic_patients"
POLICY_DIR = DATA_DIR / "policy_corpus"
SAMPLE_INPUTS = DATA_DIR / "sample_inputs"
LOGS_DIR = Path(os.getenv("COPILOT_LOG_DIR", ROOT / "logs"))  # override to keep ad-hoc runs out of evidence
TRACES_DIR = Path(os.getenv("COPILOT_TRACES_DIR", ROOT / "traces"))
REPORTS_DIR = Path(os.getenv("COPILOT_REPORTS_DIR", ROOT / "reports"))
STATE_DIR = ROOT / "state"  # runtime SQLite (git-ignored)

for _d in (LOGS_DIR, TRACES_DIR, REPORTS_DIR, STATE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TOOL_LOG = LOGS_DIR / "tool_calls.jsonl"
AUDIT_LOG = LOGS_DIR / "agent_actions.jsonl"
MCP_TRANSCRIPT = Path(os.getenv("MCP_TRANSCRIPT_PATH", LOGS_DIR / "mcp_transcript.jsonl"))
MEMORY_TEST_LOG = LOGS_DIR / "memory_test.log"

CHECKPOINT_DB = Path(os.getenv("CHECKPOINT_DB", STATE_DIR / "checkpoints.sqlite"))
LONG_TERM_DB = Path(os.getenv("LONG_TERM_DB", STATE_DIR / "long_term_memory.sqlite"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
if GEMINI_API_KEY.startswith("your_"):
    GEMINI_API_KEY = ""
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.1-flash-lite")
# optional second judge model; like GEMINI_JUDGE_MODEL it is never used if it equals GEMINI_MODEL (judge independence)
GEMINI_JUDGE_FALLBACK_MODEL = os.getenv("GEMINI_JUDGE_FALLBACK_MODEL", "")
ALLOW_OFFLINE_LLM = os.getenv("ALLOW_OFFLINE_LLM", "1") == "1"

# Last-resort fallback (off by default). "groq" DEVIATES from the brief's Gemini-only rule: Groq is called only after
# GEMINI_MODEL has failed (after its one retry when it stalled) AND GEMINI_FALLBACK_MODEL then returned 429 (quota),
# 503 (overloaded) or 404 (unavailable); never when the Gemini fallback succeeds, never on a first stall alone. The
# judge uses GROQ_JUDGE_MODEL on Gemini judge quota errors only. Every Groq call is labelled provider=groq.
QUOTA_FALLBACK_PROVIDER = os.getenv("QUOTA_FALLBACK_PROVIDER", "none").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY.startswith("your_"):
    GROQ_API_KEY = ""
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# must differ from GROQ_MODEL (never judge an answer with the model that produced it); dropped if equal
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b")


def groq_enabled() -> bool:
    return QUOTA_FALLBACK_PROVIDER == "groq" and bool(GROQ_API_KEY)

PHOENIX_PORT = int(os.getenv("PHOENIX_PORT", "6006"))
PHOENIX_PROJECT = os.getenv("PHOENIX_PROJECT_NAME", "discharge-copilot")
# DeepEval judge spans go to their own Phoenix project so the agent project/dashboard shows agent traffic only
PHOENIX_EVAL_PROJECT = os.getenv("PHOENIX_EVAL_PROJECT_NAME", PHOENIX_PROJECT + "-eval")
PHOENIX_WORKING_DIR = os.getenv("PHOENIX_WORKING_DIR", str(ROOT / ".phoenix"))

PHI_HASH_SALT = os.getenv("PHI_HASH_SALT", "local-dev-salt")
# second prompt-injection layer after the regex check: "llm-guard" (LLM Guard PromptInjection) or "off"
INJECTION_CLASSIFIER = os.getenv("INJECTION_CLASSIFIER", "llm-guard")
INJECTION_CLASSIFIER_THRESHOLD = float(os.getenv("INJECTION_CLASSIFIER_THRESHOLD", "0.92"))

MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "8"))
RECURSION_LIMIT = MAX_AGENT_STEPS * 3 + 10  # hard LangGraph backstop above the soft step guard
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))
# per-request Gemini deadline (primary and fallback each get their own; token waits excluded). Successful requests
# were measured at 0.9-2.3 s; stalled ones ran into the old 30 s deadline (504 DEADLINE_EXCEEDED)
GEMINI_REQUEST_TIMEOUT_S = float(os.getenv("GEMINI_REQUEST_TIMEOUT_S", "12"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
GEMINI_RPM = float(os.getenv("GEMINI_RPM", "10"))  # client-side request budget per minute
GEMINI_JUDGE_RPM = float(os.getenv("GEMINI_JUDGE_RPM", str(GEMINI_RPM)))  # per judge model, separate from the agent
TOOL_TIMEOUT_S = float(os.getenv("TOOL_TIMEOUT_S", "30"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# USD per 1M text tokens, used for cost = tokens x price (golden_signals.json and the Phoenix model prices).
# Defaults are the official Gemini API Standard paid-tier prices; GEMINI_PRICE_* (agent), GEMINI_FALLBACK_PRICE_*
# and GEMINI_JUDGE_PRICE_* env vars override them per role.
PRICE_SOURCE = ("Official Gemini API pricing page (ai.google.dev/gemini-api/docs/pricing), Standard paid tier, "
                "checked 2026-09-26; free-tier usage is billed $0, so this is the production-cost estimate.")
STANDARD_PRICES = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75},  # valid through 2026-12-31; 1.50 / 7.50 from 2027-01-01
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},  # no longer available to new API users (404)
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},  # valid through 2026-12-31; 3/5 probe requests 503 on 2026-09-27
}
# Groq on-demand prices, USD per 1M tokens, from the official model pages (console.groq.com/docs/model/<id>),
# checked 2026-09-27. GROQ_PRICE_* / GROQ_JUDGE_PRICE_* env vars override them.
GROQ_PRICE_SOURCE = "console.groq.com/docs/model/openai/gpt-oss-120b and .../gpt-oss-20b, checked 2026-09-27"
GROQ_PRICES = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
}


def provider_of(model: str | None) -> str:
    """'groq' for a Groq model id, 'offline' for the stub, else 'gemini' (the only other provider)."""
    m = str(model or "")
    if m in GROQ_PRICES or m in (GROQ_MODEL, GROQ_JUDGE_MODEL) or m.startswith("groq:"):
        return "groq"
    return "offline" if m == "offline-stub" else "gemini"


def _price(model: str, prefix: str) -> dict:
    base = STANDARD_PRICES.get(model) or GROQ_PRICES.get(model) or (
        STANDARD_PRICES["gemini-3.5-flash-lite"] if "lite" in model else STANDARD_PRICES["gemini-3.6-flash"])
    return {"input": float(os.getenv(f"{prefix}_INPUT_PER_MTOK", base["input"])),
            "output": float(os.getenv(f"{prefix}_OUTPUT_PER_MTOK", base["output"]))}


PRICING_PER_MTOK = {
    # list prices first; the role entries below (with their env overrides) win for an equal model id
    **STANDARD_PRICES, **GROQ_PRICES,
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "offline-stub": {"input": 0.0, "output": 0.0},
    GEMINI_JUDGE_MODEL: _price(GEMINI_JUDGE_MODEL, "GEMINI_JUDGE_PRICE"),  # before the agent, so an equal agent key wins
    GROQ_JUDGE_MODEL: _price(GROQ_JUDGE_MODEL, "GROQ_JUDGE_PRICE"),
    GEMINI_MODEL: _price(GEMINI_MODEL, "GEMINI_PRICE"),
    GEMINI_FALLBACK_MODEL: _price(GEMINI_FALLBACK_MODEL, "GEMINI_FALLBACK_PRICE"),
    GROQ_MODEL: _price(GROQ_MODEL, "GROQ_PRICE"),
}
