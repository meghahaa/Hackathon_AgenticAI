# Optimization Note: Measured Before/After (Phoenix-derived)

Every report below is produced by committed code from Phoenix span exports; none is hand-written.

| | Baseline | Final run v1 (pre-improvement) | Current run |
|---|---|---|---|
| Golden-signals report ([`scripts/export_traces.py`](../scripts/export_traces.py)) | [`golden_signals_before.json`](../reports/optimization/golden_signals_before.json) | [`final_run_v1/golden_signals.json`](../reports/failure_evidence/final_run_v1/golden_signals.json) | [`reports/golden_signals.json`](../reports/golden_signals.json) |
| Span data | [`phoenix_spans_before.parquet`](../reports/optimization/phoenix_spans_before.parquet) | [`final_run_v1/phoenix_spans.parquet`](../reports/failure_evidence/final_run_v1/phoenix_spans.parquet) | [`traces/phoenix_spans.parquet`](../traces/phoenix_spans.parquet) |
| Cost at current prices ([`scripts/reprice_run.py`](../scripts/reprice_run.py)) | [`final_run_v1_repriced.json`](../reports/optimization/final_run_v1_repriced.json) → `runs.baseline` | same file → `runs.final_run_v1` | `golden_signals.json` → `cost` |

**Prices.** All costs are tokens × the official Gemini API Standard paid-tier prices in `config.PRICING_PER_MTOK`
(`gemini-3.5-flash-lite` $0.30 in / $2.50 out per 1M tokens; `cost.pricing_source` in `golden_signals.json`).
**Final run v1 and the baseline were originally costed with placeholder prices** (flash-lite $0.10 / $0.40,
flash $0.50 / $3.00; their `cost.pricing_source` says "assumed list prices"). Their original figures ($0.000337 per
turn for v1, $0.001312 for the baseline turn) are therefore not comparable. `scripts/reprice_run.py` re-prices their
frozen token counts at the current prices into a **new** file,
[`reports/optimization/final_run_v1_repriced.json`](../reports/optimization/final_run_v1_repriced.json). The frozen
snapshots are only read (the file records each span export's SHA-256).

## Final run v1 → current run (whole run)

The two runs have different workloads. v1 is 26 turns (demo + 12-case eval). The current run is 72 turns (demo +
21 red-team attacks + 32-case eval, many of them refusals that make no LLM call). Treat the whole-run rows as the
state of each run, not a controlled experiment. The like-for-like turn below is the controlled comparison.

| Metric | Final run v1 (pre-improvement) | Current run | Source field |
|---|---|---|---|
| Turns / LLM calls | 26 / 78 | 72 / 205 | `traffic` |
| End-to-end p50 / p95 | 18.8 s / 31.0 s | **15.8 s / 30.5 s** | `latency.end_to_end` |
| LLM span p50 / p95, *including* the rate-limiter wait | 5.7 s / 6.2 s (`thinking`, before the split) | 5.5 s / 6.3 s | v1 `latency.thinking`; current `latency.thinking_including_rate_limit` |
| Model-only thinking p50 / p95 | not measurable (wait was inside the LLM span) | **1.4 s / 2.0 s** | `latency.thinking` |
| Client-side rate-limit wait per call p50 / p95 | inside the LLM span | 4.1 s / 4.8 s | `latency.rate_limit_wait` |
| Tool p95 | 21.6 ms | 27.8 ms | `latency.tool` |
| `input_guard` p50 | 0.12 s | 0.65 s (Presidio NER + LLM Guard classifier now run) | `latency.by_node` |
| Span errors | 0 | 0 of 2,168 | `errors.span_errors` |
| Mean input tokens per LLM call | 749.9 | 740.2 | `tokens.mean_input_per_llm_call` |
| Mean cost per turn, **current prices** | **$0.001377** (re-priced; originally reported $0.000337 at placeholder prices) | **$0.001253** | `final_run_v1_repriced.json` → `runs.final_run_v1.current_prices`; `cost.mean_usd_per_turn` |
| Max LLM calls per turn | 5 | 5 | `cost.max_llm_calls_per_turn` |
| Budgets met | 4 of 4 | 4 of 4 | `budgets` |

The current run's cost and latency are Gemini-only: `providers.llm_calls_by_provider` = `{"gemini": 205}`, and every
`groq_calls_by_step` count is 0.

### Thinking latency before/after the rate-limit split

Before the split, the `GEMINI_RPM=10` limiter wait sat inside each LLM span, so v1's "thinking" p50 of 5.7 s was
mostly throttling. The current run separates the two. Model-only time is **p50 1.4 s / p95 2.0 s**, and the limiter
wait is **p50 4.1 s / p95 4.8 s** per call. Adding them back gives p50 5.5 s / p95 6.3 s, the same as v1's
figure. So the split did not make the model faster. It shows that about three quarters of each call's time is the
client-side request budget, not Gemini.

## Like-for-like turn: "Please assemble the discharge plan for this patient." (pref_1172896b29, HIGH risk)

From [`final_run_v1_repriced.json`](../reports/optimization/final_run_v1_repriced.json) → `like_for_like_turn`
(all three columns re-priced at current prices).

| Metric | Baseline `run-c2a2a26912f4` | Final run v1 `run-b97b81524fd6` | Current `run-fb5b4e0b7ea1` |
|---|---|---|---|
| Trace | `754acea392f3e682da0860874626ed5c` | `a75d2ca2470bd9f93a737e0c9b1155cc` | `04c97d8b944de50f4ef10d8db14bd50a` |
| End-to-end latency | 117.5 s | **34.1 s** | 41.2 s |
| LLM calls / of which ERROR | 8 / 4 (503) | 5 / 0 | 5 / 0 |
| Time in LLM spans | 50.0 s (incl. limiter wait) | 30.8 s (incl. limiter wait) | 7.6 s model-only + 22.1 s limiter wait |
| Tokens in / out | 4,069 / 582 | 4,581 / 603 | 4,746 / 597 |
| Cost at current prices | $0.003077 | $0.002882 | $0.002916 |

* **Baseline → v1:** −71 % latency, 4 errored calls removed. At current prices the cost fell by about 6 %
  ($0.003077 → $0.002882). The −47 % reported earlier ($0.00131 → $0.00070) came from the placeholder prices.
* **v1 → current: this turn is 7.1 s slower.** It is the first turn of the current run, and it paid two one-off
  costs: `input_guard` took 7.0 s (span_id `5c4b0788b6d8d3a7`; run-wide p50 0.65 s), and the first limiter wait took
  6.0 s (span_id `b24afaaae8b67e89`, the run's maximum). Model time for the same five calls fell from a
  wait-inflated 30.8 s to 7.6 s. Input tokens rose by about 4 %, and cost by about 1 %. The per-turn improvement shows in the
  whole-run end-to-end p50 (18.8 s → 15.8 s), not in this cold-start turn.

## What changed and why (each change is tied to a failure in [failure-analysis.md](failure-analysis.md))

1. **Model routing** (F1): the primary became the model that was actually healthy (`gemini-3.5-flash-lite`), with
   `gemini-3.6-flash` as the Gemini-only fallback. In the current run the fallback was never needed (205 of 205
   calls on the primary).
2. **Tighter, per-request deadline** (F2, F13): `LLM_TIMEOUT_S` 45 → 30, then a per-request
   `GEMINI_REQUEST_TIMEOUT_S` (12 s) so a stalled primary leaves the fallback a full deadline.
3. **Warm start** (F2): the FAISS index and the sentence-transformer model load once in `Copilot.__aenter__`.
4. **Fewer calls**: stable primary (8 → 5 LLM calls per full-plan turn), and `policy_question` turns no longer
   call the query planner (F13: 8 policy turns, 0 planner spans).
5. **Rate-limit split** (measurement, not speed): the limiter wait has its own `rate_limit_wait` span, so
   "thinking" is model-only time.

## What still dominates latency

Client-side pacing. Each full-plan turn makes 5 sequential LLM calls, and each waits for the `GEMINI_RPM=10`
budget (current run: wait p50 4.1 s vs model-only p50 1.4 s per call). On a paid tier, raising `GEMINI_RPM` is
the largest lever. Running independent calls concurrently (e.g. the query planner alongside the risk rationale)
would be the next code change. The guardrail models added in this round (Presidio NER + LLM Guard) cost about
0.5 s per turn at p50 on `input_guard` (0.12 s → 0.65 s), plus a one-off load on the first turn.

## Phoenix UI cost figure

`register_model_pricing()` registers the configured prices in Phoenix when tracing starts, so the current
[`reports/dashboard.png`](../reports/dashboard.png) (Metrics view) shows a real cost. The Final run v1 screenshot
([`final_run_v1/dashboard.png`](../reports/failure_evidence/final_run_v1/dashboard.png)) shows *Total Cost $0*,
because Phoenix had no price for `gemini-3.5-flash-lite` then. The repo's cost figures are always computed as
tokens × price by `export_traces.py` (and `reprice_run.py`), not read from the UI.
