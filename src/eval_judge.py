"""DeepEval LLM-as-judge backed by Google Gemini (the only approved provider).

DeepEval's built-in GeminiModel has no retry/fallback, and the Gemini preview models return
503 "high demand" intermittently (see docs/failure-analysis.md F1). This judge keeps DeepEval's
metric logic unchanged and only supplies `generate`: JSON-mode Gemini with retries, a client-side request budget per
judge model (src.llm.model_limiter, traced as a rate_limit_wait span), and an optional second Gemini judge model.

Judge independence: the judge NEVER uses the agent model (GEMINI_MODEL). Allowed judges are GEMINI_JUDGE_MODEL and
the optional GEMINI_JUDGE_FALLBACK_MODEL; either is dropped if it equals GEMINI_MODEL. Per case, set_producers() also
excludes every model that produced the answer being judged (e.g. the Gemini fallback, or GROQ_MODEL). When no judge
model can score a metric, scripts/run_eval.py reports it as "unjudged" with the reason instead of falling back to the
agent model.

Groq quota fallback (off by default; QUOTA_FALLBACK_PROVIDER=groq deviates from the brief's Gemini-only rule): only
when every usable Gemini judge model returned a quota error (429 RESOURCE_EXHAUSTED), GROQ_JUDGE_MODEL scores the
metric. It must differ from GROQ_MODEL, and it goes through the same mask_phi, per-model limiter and rate_limit_wait
span (provider=groq) as the Gemini judge. Every judge call is recorded with its provider.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from src import config
from src.guardrails.guards import mask_phi
from src.llm import extract_json, is_quota, model_limiter, rate_limit_wait

# tag on every span the judge creates; scripts/export_traces.py reports these apart from the agent's signals
JUDGE_COMPONENT = "eval_judge"


class GeminiFallbackJudge(DeepEvalBaseLLM):
    def __init__(self, model: str | None = None, fallback: str | None = None) -> None:
        wanted = [m for m in (model or config.GEMINI_JUDGE_MODEL, fallback or config.GEMINI_JUDGE_FALLBACK_MODEL) if m]
        self.excluded = sorted({m for m in wanted if m == config.GEMINI_MODEL})  # never judge with the agent model
        self.models = list(dict.fromkeys(m for m in wanted if m != config.GEMINI_MODEL))
        if not self.models:
            raise ValueError(f"no judge model distinct from the agent model {config.GEMINI_MODEL!r} "
                             "(set GEMINI_JUDGE_MODEL / GEMINI_JUDGE_FALLBACK_MODEL)")
        # Groq judge: only with the flag on, and never the Groq model that produces agent answers
        self.groq_model = (config.GROQ_JUDGE_MODEL if config.groq_enabled() and config.GROQ_JUDGE_MODEL
                           and config.GROQ_JUDGE_MODEL != config.GROQ_MODEL else None)
        self.producers: set[str] = set()  # models that produced the answer currently being judged
        self.calls: list[dict[str, Any]] = []
        self.disabled: set[str] = set()
        self.quota_exhausted: set[str] = set()
        self._groq_client = None
        super().__init__(self.models[0])

    def set_producers(self, models) -> None:
        """Models that produced the answer about to be judged: none of them may judge it."""
        self.producers = set(models)

    def load_model(self):
        from google import genai

        return genai.Client(api_key=config.GEMINI_API_KEY)

    def get_model_name(self) -> str:
        groq = f"; groq quota fallback: {self.groq_model}" if self.groq_model else ""
        return f"gemini-judge({' -> '.join(self.models)}{groq})"

    def _groq_generate(self, prompt: str, attempt: int) -> str:
        """One GROQ_JUDGE_MODEL request (prompt already PHI-masked), through the judge's rate-limit wait."""
        if self._groq_client is None:
            from groq import Groq

            self._groq_client = Groq(api_key=config.GROQ_API_KEY, timeout=config.GEMINI_REQUEST_TIMEOUT_S * 5,
                                     max_retries=0)
        model = self.groq_model
        rate_limit_wait(f"judge:groq:{model}", attempt, component=JUDGE_COMPONENT, limiter=model_limiter(model),
                        provider="groq")
        t0 = time.perf_counter()
        try:
            resp = self._groq_client.chat.completions.create(model=model, temperature=0.0,
                                                              messages=[{"role": "user", "content": prompt}])
        except Exception as exc:
            self.calls.append({"provider": "groq", "model": model, "ok": False, "error": mask_phi(str(exc))[:160],
                               "latency_s": round(time.perf_counter() - t0, 2)})
            raise
        usage = getattr(resp, "usage", None)
        self.calls.append({"provider": "groq", "model": model, "ok": True,
                           "latency_s": round(time.perf_counter() - t0, 2),
                           "input_tokens": getattr(usage, "prompt_tokens", None),
                           "output_tokens": getattr(usage, "completion_tokens", None)})
        return resp.choices[0].message.content or ""

    @staticmethod
    def _parse(text: str, schema: Optional[type[BaseModel]]) -> Any:
        if schema is None:
            return text
        try:
            return schema.model_validate_json(text)
        except Exception:
            return schema.model_validate(extract_json(text))

    def generate(self, prompt: str, schema: Optional[type[BaseModel]] = None) -> Any:
        # the judge calls google-genai directly (not the LangChain wrapper in src.llm), so it masks PHI itself:
        # patient IDs (canonical + loose), registry identifiers and PHI patterns never reach Gemini
        prompt = mask_phi(prompt)
        last: Exception | None = None
        usable = [m for m in self.models if m not in self.producers]  # never judge with the answer's own model
        if not usable:
            raise RuntimeError("no judge model distinct from the models that produced this answer "
                               f"({sorted(self.producers)})")
        for attempt in range(3):
            for model in usable:
                if model in self.disabled:
                    continue
                rate_limit_wait(f"judge:{model}", attempt, component=JUDGE_COMPONENT, limiter=model_limiter(model))
                t0 = time.perf_counter()
                try:
                    cfg = {"response_mime_type": "application/json"} if schema else None
                    resp = self.model.models.generate_content(model=model, contents=prompt, config=cfg)
                    text = resp.text or ""
                    usage = getattr(resp, "usage_metadata", None)
                    self.calls.append({"provider": "gemini", "model": model, "ok": True,
                                       "latency_s": round(time.perf_counter() - t0, 2),
                                       "input_tokens": getattr(usage, "prompt_token_count", None),
                                       "output_tokens": getattr(usage, "candidates_token_count", None)})
                    return self._parse(text, schema)
                except Exception as exc:
                    last = exc
                    self.calls.append({"provider": "gemini", "model": model, "ok": False, "error": str(exc)[:160],
                                       "latency_s": round(time.perf_counter() - t0, 2)})
                    if is_quota(exc):
                        self.quota_exhausted.add(model)
                    # circuit breaker: quota exhausted / model retired will not recover within a run
                    if any(t in str(exc) for t in ("RESOURCE_EXHAUSTED", "exceeded your current quota", "NOT_FOUND")) \
                            and len(self.disabled) < len(self.models) - 1:
                        self.disabled.add(model)
            # Groq quota fallback: flag on, and every usable Gemini judge model returned a quota error
            if self.groq_model and self.groq_model not in self.producers \
                    and all(m in self.quota_exhausted for m in usable):
                try:
                    return self._parse(self._groq_generate(prompt, attempt), schema)
                except Exception as exc:
                    last = exc
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"judge failed on all judge models: {last}")

    async def a_generate(self, prompt: str, schema: Optional[type[BaseModel]] = None) -> Any:
        return await asyncio.to_thread(self.generate, prompt, schema)

    def usage_summary(self) -> dict[str, Any]:
        ok = [c for c in self.calls if c["ok"]]
        return {"calls": len(self.calls), "failed_calls": len(self.calls) - len(ok),
                "models": self.models, "excluded_agent_model": self.excluded,
                "groq_quota_fallback_model": self.groq_model,
                "by_model": {m: sum(1 for c in ok if c["model"] == m)
                             for m in (*self.models, *([self.groq_model] if self.groq_model else []))},
                "by_provider": {p: sum(1 for c in self.calls if c.get("provider") == p) for p in ("gemini", "groq")},
                "groq_calls": sum(1 for c in self.calls if c.get("provider") == "groq"),
                "circuit_broken_models": sorted(self.disabled),
                "same_model_as_agent": any(c["model"] == config.GEMINI_MODEL for c in ok),
                "input_tokens": sum(c.get("input_tokens") or 0 for c in ok),
                "output_tokens": sum(c.get("output_tokens") or 0 for c in ok),
                "errors_sample": [c["error"] for c in self.calls if not c["ok"]][:3]}
