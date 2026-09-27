# Compliance Mapping: EU AI Act · NIST AI RMF 1.0 · India DPDP Act 2023

**Scope and posture.** This is a hackathon prototype on synthetic data; nothing here is legal
advice. The mapping records which obligations would apply if the copilot were deployed, how the
design addresses each one today, and the committed artifact that evidences it.

**Classification.** A copilot that drafts discharge plans and flags readmission risk for clinicians
is plausibly a **high-risk AI system** under the EU AI Act: as a safety component of, or itself, a
medical-device-software function (Art. 6(1) + Annex I, MDR 2017/745), and/or through Annex III
access-to-essential-services use. We therefore map against the high-risk obligations (Arts. 9–15)
while noting the system is advisory, human-gated and never finalises or alters orders.

## 1. EU AI Act (Regulation (EU) 2024/1689)

| Obligation | How it is addressed | Evidence artifact |
|---|---|---|
| **Art. 9 Risk management system** | Living risk register with OWASP/NIST categories, controls and residual risk; failure analysis feeds fixes | [risk-register.md](risk-register.md), [failure-analysis.md](failure-analysis.md) |
| **Art. 10 Data & data governance** | Only synthetic, generated data with fixed seed; demographics stripped at MCP boundary; policy corpus versioned in repo. **Model egress:** every Gemini request is PHI-masked at one point, after the input guard has seen the raw text: `RateLimitedChatGoogleGenerativeAI._prepare_request` + `get_num_tokens` in [`src/llm.py`](../src/llm.py) (all agent, tool-loop and LangMem calls). The DeepEval judge masks its own prompt in [`src/eval_judge.py`](../src/eval_judge.py). Canonical and loose patient ids become `pref_` pseudonyms; registry identifiers and PHI patterns are masked ([`tests/test_gemini_phi_egress.py`](../tests/test_gemini_phi_egress.py), incl. `test_full_graph_turn_sends_no_ids_to_gemini`). Limit: egress masking is registry + regex (no NER), so a free-text name that is in no record and matches no pattern could still reach the model | [`scripts/generate_synthetic_data.py`](../scripts/generate_synthetic_data.py), [`mcp_server/server.py`](../mcp_server/server.py) (`deidentify_record`), [`src/llm.py`](../src/llm.py), [`tests/test_gemini_phi_egress.py`](../tests/test_gemini_phi_egress.py), [model-card.md §3](model-card.md#3-data) |
| **Art. 11 + Annex IV Technical documentation** | Architecture, models, intended use, limitations, evaluation method and results | [README.md](../README.md), [model-card.md](model-card.md), [`reports/eval_report.json`](../reports/eval_report.json) |
| **Art. 12 Record-keeping (automatic logs)** | Machine-generated audit trail for every consequential action; per-call tool log; full OTel traces with run/session correlation | [`logs/agent_actions.jsonl`](../logs/agent_actions.jsonl) ([`src/guardrails/audit.py`](../src/guardrails/audit.py)), [`logs/tool_calls.jsonl`](../logs/tool_calls.jsonl) ([`src/observability/tool_logger.py`](../src/observability/tool_logger.py)), [`traces/phoenix_spans.parquet`](../traces/phoenix_spans.parquet) |
| **Art. 13 Transparency to deployers** | Outputs state DRAFT/PENDING status, approval reasons, risk factors with sources, policy citations; model card lists limitations | sample outputs in [output-risk.md](output-risk.md#sample), [model-card.md §7](model-card.md#7-known-limitations-and-failure-modes) |
| **Art. 14 Human oversight** | `human_approval_gate`: HIGH risk, order-change requests, quarantined injections and degraded outputs are held. Every drafted discharge plan / follow-up schedule (any risk tier) and every order-change request then **pauses** at `clinician_signoff` (LangGraph `interrupt`, PHI-masked payload): nothing is finalised until a clinician resumes with approve / reject / request changes (CLI `--approve/--reject/--request-changes`, API `POST /v1/discharge/{session_id}/signoff`), and that decision is audited (`actor=clinician, action=signoff`). No code path sets an `APPROVED*` status without the resume (Art. 14(4)(a)-(e): the overseer can review, decide not to use, or reject the output). In the current run, 33 turns paused at `clinician_signoff` (e.g. the HIGH-risk demo plan run `run-fb5b4e0b7ea1`, span_id `8556084c49cd634e`, status `AWAITING_CLINICIAN_SIGNOFF`), and none of the 21 red-team attacks reached an `APPROVED*` status | [`src/graph.py`](../src/graph.py) `human_approval_gate`, `route_after_output`, `clinician_signoff`; [`src/runner.py`](../src/runner.py) `Copilot.resume`; [`tests/test_hitl.py`](../tests/test_hitl.py) (`test_status_never_approved_without_resume`, `test_resume_approve_sets_approved_by_clinician_and_audits`); [`reports/cli_run_results.json`](../reports/cli_run_results.json); [`reports/redteam_results.json`](../reports/redteam_results.json) `summary.reached_approved_status`; [output-risk.md](output-risk.md#clinician-sign-off-a-real-pause-langgraph-interrupt) |
| **Art. 15 Accuracy, robustness, cybersecurity** | Golden-set accuracy + hallucination/faithfulness (DeepEval); graceful degradation (retries, per-request deadline, Gemini fallback); layered prompt-injection defence (regex + LLM Guard classifier) and cross-patient defences, tested by a 21-attack red-team set (block rate 0.81; residuals in [failure-analysis.md](failure-analysis.md) F8–F11) | [`reports/eval_report.json`](../reports/eval_report.json), [`reports/redteam_results.json`](../reports/redteam_results.json), [`reports/golden_signals.json`](../reports/golden_signals.json), [`tests/`](../tests/) |
| **Art. 50 Transparency (AI interaction)** | Users are clinicians operating a named copilot; every output labels itself a draft by an AI copilot | [`src/render.py`](../src/render.py) |
| **Art. 72 Post-market monitoring** (if deployed) | Phoenix tracing + golden signals + budgets give the monitoring feed; regenerable from one command | [`scripts/regenerate_evidence.py`](../scripts/regenerate_evidence.py), [`scripts/export_traces.py`](../scripts/export_traces.py) |

## 2. NIST AI RMF 1.0 (+ AI 600-1 Generative AI Profile)

| Function / subcategory | How it is addressed | Evidence artifact |
|---|---|---|
| **GOVERN 1.1/1.2** legal & policy requirements understood | This mapping; intended-use and out-of-scope statements | [compliance.md](compliance.md), [model-card.md §4–5](model-card.md#4-intended-use) |
| **GOVERN 1.6 / 6.1** inventory, third-party & supply chain | Model inventory (Gemini only), dependency list, transitive SDKs noted | [model-card.md §1](model-card.md#1-models), [`requirements.txt`](../requirements.txt) |
| **GOVERN 3.2** human-AI roles | Clinician owns decisions; copilot drafts | [output-risk.md](output-risk.md) |
| **MAP 1.1 / 1.5** context & risk tolerance | Intended use, users, out-of-scope; risk tiers of outputs | [model-card.md](model-card.md), [output-risk.md](output-risk.md) |
| **MAP 2.3 / 3.4** limitations, human oversight proficiency | Limitations list; outputs show factors and citations to support review | [model-card.md §7](model-card.md#7-known-limitations-and-failure-modes) |
| **MAP 5.1** likelihood & impact of harms | Risk register likelihood/impact/residual | [risk-register.md](risk-register.md) |
| **MEASURE 2.3** performance in deployment-like conditions | End-to-end golden-set evaluation through the real graph, MCP server and Gemini | [`scripts/run_eval.py`](../scripts/run_eval.py) → [`reports/eval_report.json`](../reports/eval_report.json) |
| **MEASURE 2.5** validity / confabulation (600-1) | DeepEval HallucinationMetric + FaithfulnessMetric with Gemini judge; citation validation | [`reports/eval_report.json`](../reports/eval_report.json) `summary` |
| **MEASURE 2.6** safety / 2.7 security & resilience | Guardrail tests, loop guard tests, injection & cross-patient cases in golden set | [`tests/test_routing.py`](../tests/test_routing.py), [`tests/test_loops.py`](../tests/test_loops.py), [`tests/test_tool_contracts.py`](../tests/test_tool_contracts.py) |
| **MEASURE 2.9** explainability | Risk factors with evidence + source; policy citations on every requirement | [`src/clinical_rules.py`](../src/clinical_rules.py), [`src/render.py`](../src/render.py) |
| **MEASURE 2.10** privacy (600-1 Data Privacy) | PHI masking in answers/logs/traces (registry + regex + Presidio NER in `output_guard` and `quarantine`); every Gemini request PHI-masked on egress (registry + regex, no NER; see Art. 10 above); PHI scan of all evidence incl. `docs/` | [`src/guardrails/guards.py`](../src/guardrails/guards.py), [`src/llm.py`](../src/llm.py), [`tests/test_gemini_phi_egress.py`](../tests/test_gemini_phi_egress.py), [`reports/pii_redaction_sample.json`](../reports/pii_redaction_sample.json), [`reports/phi_secret_scan.json`](../reports/phi_secret_scan.json) |
| **MEASURE 3.1** tracking identified risks over time | Golden signals with budgets; before/after optimisation reports | [`reports/golden_signals.json`](../reports/golden_signals.json), [optimization-note.md](optimization-note.md) |
| **MANAGE 1.3 / 2.2** risk responses & mechanisms to supersede | Gate holds; output guard blocks; order-change refusal | [`src/graph.py`](../src/graph.py), [`src/guardrails/guards.py`](../src/guardrails/guards.py) |
| **MANAGE 2.3 / 2.4** respond to unknown risks, deactivate | Degraded-mode flags + escalation; step budget; fallback model | [`src/llm.py`](../src/llm.py), [`tests/test_loops.py`](../tests/test_loops.py) |
| **MANAGE 4.1** post-deployment monitoring | Phoenix + regenerable evidence pipeline | [`scripts/regenerate_evidence.py`](../scripts/regenerate_evidence.py) |

## 3. India Digital Personal Data Protection Act 2023 (DPDP)

| Obligation | How it is addressed | Evidence artifact |
|---|---|---|
| **s.4 / s.6** lawful processing with consent for a specified purpose | Only synthetic data; purpose (discharge planning) stated. Real deployment would need consent capture in the source EHR; noted as a deployment prerequisite | [model-card.md §3–4](model-card.md#3-data) |
| **s.8(3)** accuracy & completeness where data is used for decisions | Orders reproduced verbatim and verified; risk computed deterministically from record fields | [`src/guardrails/guards.py`](../src/guardrails/guards.py) `verify_orders_unchanged`, [`src/clinical_rules.py`](../src/clinical_rules.py) |
| **s.8(4)/(5)** reasonable security safeguards | Pseudonymised IDs (HMAC), demographics never leave the MCP server, masked logs/traces, secrets in env only. The processor (Google Gemini API) receives only PHI-masked requests: pseudonyms instead of patient ids, with registry identifiers and PHI patterns masked ([`tests/test_gemini_phi_egress.py`](../tests/test_gemini_phi_egress.py)). Limit: no NER on egress | [`mcp_server/server.py`](../mcp_server/server.py), [`src/llm.py`](../src/llm.py), [`src/observability/tracing.py`](../src/observability/tracing.py), [`reports/phi_secret_scan.json`](../reports/phi_secret_scan.json) |
| **s.8(6)** breach intimation readiness | Audit trail and traces allow reconstructing what was accessed per run/session | [`logs/agent_actions.jsonl`](../logs/agent_actions.jsonl), [`logs/tool_calls.jsonl`](../logs/tool_calls.jsonl) |
| **s.8(7)** erasure when purpose is served | Runtime state (`state/`) is disposable and git-ignored; long-term memory is namespaced per patient pseudonym so it can be deleted per principal | [`src/memory/memory_store.py`](../src/memory/memory_store.py), [`.gitignore`](../.gitignore) |
| **Data minimisation** (s.4/s.5 purpose limitation) | Per-agent field selection: each worker sees only the fields it needs | [`src/context/context_manager.py`](../src/context/context_manager.py) `AGENT_FIELDS`, `select_record_view` |
| **s.11–13** data principal rights (access, correction, grievance) | Out of scope for the prototype; per-patient memory namespace and audit trail are the hooks | [risk-register.md](risk-register.md) |

## Secrets management

* All secrets are environment variables loaded by python-dotenv ([`src/config.py`](../src/config.py));
  [`.env`](../.gitignore) is git-ignored (`.env`, `.env.*`, `!.env.example`); [`.env.example`](../.env.example)
  contains placeholders only.
* [`scripts/scan_evidence.py`](../scripts/scan_evidence.py) scans every committed log, trace, report and doc
  for Google API-key patterns and the literal configured key → [`reports/phi_secret_scan.json`](../reports/phi_secret_scan.json).
* **Rotation approach (documented, not automated in this cut):** keys are per-environment and never
  shared; rotate by issuing a new Gemini key, updating the secret store / `.env`, restarting, then
  revoking the old key. `PHI_HASH_SALT` rotation changes all pseudonyms, so it is rotated only with
  a re-keying of long-term memory. Production would use a secret manager with short-lived
  credentials rather than `.env` files.

## PHI scan

Result of the automated scan over `logs/`, `traces/`, `reports/`, `docs/` is in
[`reports/phi_secret_scan.json`](../reports/phi_secret_scan.json) (`clean`, `findings`, `secrets_hygiene`).

## Measured results

From the current run (all values machine-generated; field paths given). Every model call in it was Gemini
(`golden_signals.json` → `providers.llm_calls_by_provider`), so these are Gemini-only figures.

| Measure | Value | Source |
|---|---|---|
| Golden-set agent accuracy (32 cases: routing, tier, approval, status, citations, order integrity, no PHI) | **31/32 = 0.969** (G24 failed, [failure-analysis.md](failure-analysis.md) F11) | [`reports/eval_report.json`](../reports/eval_report.json) → `summary.accuracy` |
| Hallucination rate on answers (DeepEval HallucinationMetric, headline) | **0.0** | `summary.hallucination_rate_answers` |
| Hallucination rate, all 32 scored cases | 0.094 (3 non-answers scored 0.0 without inventing anything; see the README Results section) | `summary.hallucination_rate`, `summary.hallucination_rate_non_answers` |
| Faithfulness / answer relevancy (DeepEval) | **0.965** / **0.879** mean | `summary.mean_faithfulness`, `summary.mean_answer_relevancy` |
| Judged coverage / judge independence | **1.0**; judge `gemini-3.1-flash-lite`, `same_model_as_agent: false` (246 judge calls, 3 failed and retried) | `summary.judged_coverage`, `judge.usage` |
| Red-team block rate | **17/21 = 0.81**; 0 PHI leaks; 0 reached `APPROVED*` | [`reports/redteam_results.json`](../reports/redteam_results.json) → `summary` |
| End-to-end latency p50 / p95 | 15.8 s / 30.5 s (budget 90 s) | [`reports/golden_signals.json`](../reports/golden_signals.json) → `latency.end_to_end`, `budgets` |
| Model-only thinking p50 / p95 | 1.4 s / 2.0 s | `latency.thinking` |
| Mean estimated cost per turn | $0.001253 (budget $0.01) | `cost.mean_usd_per_turn` |
| Span error rate / tool error rate | 0 % / 0.27 % (1 of 366, an expected `patient_not_found`) | `errors` |
| PHI + secrets scan of evidence and docs | `clean: true` | [`reports/phi_secret_scan.json`](../reports/phi_secret_scan.json) |
| Agent tests | 274 passed | [`reports/pytest_report.txt`](../reports/pytest_report.txt) |

**Cost basis.** Costs are tokens × the official Gemini API **Standard paid-tier** prices
(ai.google.dev/gemini-api/docs/pricing, checked 2026-09-26; `gemini-3.5-flash-lite` $0.30 input / $2.50 output per
1M tokens), recorded in `golden_signals.json` → `cost.pricing_source` and `cost.pricing_usd_per_mtok`. The
evidence runs used the free tier, which bills $0, so the figure is a production-cost estimate. Final run v1 was
costed at placeholder prices. Its re-priced figure is in
[`reports/optimization/final_run_v1_repriced.json`](../reports/optimization/final_run_v1_repriced.json)
(see [optimization-note.md](optimization-note.md)). The pre-improvement results stay frozen in
[`reports/failure_evidence/final_run_v1/`](../reports/failure_evidence/final_run_v1/manifest.json).
