# EverMemBench Temporal budget control

Protocol version: `evermembench_temporal_budget_control_v1`
Recorded: 2026-09-15, before any answer for the new conditions was generated.

## Status and motivation

Exploratory control. In the sealed run
`.research_runs/evermembench_temporal_episodes_official_v1`, RAW+EPISODES scored
0.200 on the 300 Temporal Duration (`TP`) questions against 0.123 for RAW
(+0.077, 95% CI [+0.027, +0.123]). That result is already known and motivates this
control. On those questions RAW+EPISODES also exposed more raw messages to the answer
model (mean 17.4 unique source messages versus 10), so the gain may come from context
size rather than from episode structure. This control separates the two.

## Primary endpoint (fixed before new outcomes)

Paired accuracy difference **RAW+EPISODES − RAW-token-matched** on the 300 `TP`
questions, with a paired question bootstrap (20,000 samples, seed 20260912) and an exact
paired McNemar test. RAW+EPISODES is the sealed prediction and judgment.

Interpretation, fixed in advance:

- 95% CI lower bound > 0: episodes add accuracy beyond an equal-token raw context.
- 95% CI contains 0: the episode effect is not distinguishable from context size.
- 95% CI upper bound < 0: an equal-token raw context is at least as accurate.

## Conditions

All conditions use the sealed memory, the sealed `text-embedding-3-large` raw index and
question embeddings of the source run (read only, never recomputed), the official
open-ended answer prompt, `openai/gpt-4.1-mini` pinned to OpenAI (temperature 0,
max_tokens 1000), official answer post-processing, and the official judge
`google/gemini-3-flash-preview` pinned to Google AI Studio with the amended official
scoring (`evermembench_official_scoring_20260914`).

- `RAW` (sealed): raw top-10.
- `RAW+EPISODES` (sealed): unified top-10 with provenance expansion.
- `RAW-count-matched` (new): raw documents in the source run's raw ranking order, taking
  the first N distinct documents, where N is the number of **unique** raw source messages
  exposed by the sealed RAW+EPISODES documents for that question (duplicates removed).
- `RAW-token-matched` (new): raw documents in the same ranking order, added while the
  rendered context stays at or below the RAW+EPISODES context token count; the first
  document that would exceed it is included only if that brings the count strictly closer
  to the target. Tokens are counted with `tiktoken` `o200k_base` on the rendered context
  string exactly as inserted into the prompt.
- `RAW+EPISODES-rerun` (new, calibration only): the sealed RAW+EPISODES documents and
  prompt, answered and judged again in this run. It estimates run-to-run noise and is not
  part of the primary endpoint.

## Secondary analyses (all reported)

- RAW+EPISODES − RAW-count-matched.
- RAW-token-matched − RAW and RAW-count-matched − RAW.
- Calibration: RAW+EPISODES-rerun − RAW+EPISODES and per-question agreement. If the
  absolute difference is ≥ 0.03 or agreement is < 90%, cross-run comparisons are reported
  as noise-limited and the same-run difference RAW+EPISODES-rerun − RAW-token-matched is
  reported alongside the primary endpoint.
- Per condition: accuracy, mean unique raw source messages, mean context tokens, gold
  evidence recall and precision over unique exposed messages, API cost and latency.

## Guardrails

- The source run directory is never written. Missing cached vectors abort the run.
- Before any API call the runner verifies that the reconstructed raw top-10 equals the
  sealed RAW retrieval for every question and that reconstructed RAW and RAW+EPISODES
  prompts hash to keys present in the sealed answer cache.
- New work goes to `.research_runs/evermembench_temporal_budget_control_v1` with a
  resumable content-addressed cache. A run manifest records protocol and runner hashes;
  a later mismatch refuses to resume.
- A 5-question smoke run in a separate directory checks the pipeline first. Smoke
  outcomes are not used for any decision. Only defects that change no definition above
  may be fixed after smoke.
- No selection rule, rank depth, condition, scope or statistic changes after new outcomes.
  Extending the control beyond `TP` requires a separate protocol.
