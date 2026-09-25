# EverMemBench Temporal control: chronological presentation of the same context

Protocol version: `evermembench_chronological_order_control_v1`
Recorded: 2026-09-15, before any answer for the new conditions was generated. The protocol
file and runner are hashed into the run manifest before the first answer call.

## Status and motivation

Exploratory mechanistic control on the same 300 Temporal Duration (`TP`) questions as the
frozen budget control (`evermembench_temporal_budget_control_v1_20260915_021134_MSK`) and
the frozen unlinked-events ablation (`evermembench_unlinked_events_control_v1_20260915_022532_MSK`).

On these questions RAW+EPISODES = 0.200, RAW-token-matched = 0.123 and
RAW+EVENTS-token-matched = 0.170. An episode presents its events and their source messages
in chronological order, while the raw and unlinked-event contexts are presented in retrieval
rank order. Duration questions may be answerable from two dates once the relevant messages
are shown in time order. This control asks whether chronological presentation of the same
retrieved context, with explicit day offsets, reproduces the episode gain. It was proposed
after all earlier outcomes on these questions were known, so it is not an independent
replication.

## Conditions

Everything not listed here is identical to the two frozen controls: sealed memory, official
open-ended answer prompt, `openai/gpt-4.1-mini` (temperature 0, max_tokens 1000), official
answer post-processing, official judge `google/gemini-3-flash-preview` with the amended
official scoring, `o200k_base` token counting on the rendered context string.

The document set of each new condition is copied, not re-selected, from a frozen condition:

- `RAW-token-matched-chrono` (new): the exact documents of frozen `RAW-token-matched` for
  the question.
- `RAW+EVENTS-token-matched-chrono` (new): the exact documents of frozen
  `RAW+EVENTS-token-matched` for the question.

Presentation. Documents are sorted by timestamp ascending; ties are broken by document id.
A raw message's timestamp is the `[YYYY-MM-DD HH:MM:SS]` prefix of its rendered text (a
date-only prefix counts as `00:00:00`). An event document's timestamp is the earliest
timestamp among its source messages. Each context line is rendered as
`- [Day +N] {rendered document}`, where N is the number of calendar days between the
document's date and the date of the earliest document in the context. Document text is
unchanged. Because of the day prefixes the context is slightly longer than the frozen
token-matched context; token counts of both renderings are recorded and reported.

References (not re-generated): sealed `RAW`, `RAW+EPISODES`; frozen `RAW-token-matched`,
`RAW+EVENTS-token-matched`.

## Primary endpoint (fixed before new outcomes)

Paired accuracy difference **RAW+EPISODES − RAW-token-matched-chrono** on the 300 `TP`
questions; paired question bootstrap (20,000 samples, seed 20260912) and exact paired
McNemar test.

Interpretation, fixed in advance:

- 95% CI lower bound > 0: the episode gain is not reproduced by chronological presentation
  of the equal-token raw context.
- 95% CI contains 0: chronological presentation of raw messages and episodes are not
  distinguishable on these questions.
- 95% CI upper bound < 0: chronological raw presentation is more accurate than episodes.

## Secondary analyses (all reported)

- RAW-token-matched-chrono − RAW-token-matched (effect of ordering on the same raw context).
- RAW+EVENTS-token-matched-chrono − RAW+EVENTS-token-matched (effect of ordering on the same
  unlinked-event context).
- RAW+EPISODES − RAW+EVENTS-token-matched-chrono (linking beyond chronological presentation
  of the same events).
- For every comparison above: the paired difference with the five projects as clusters
  (cluster bootstrap, 20,000 samples, seed 20260912), per-project differences and the exact
  two-sided sign-flip test over the five project differences. With five clusters the smallest
  attainable sign-flip p is 0.0625; the cluster bootstrap is reported as descriptive.
- Per condition: accuracy, mean context tokens under both renderings, mean unique raw source
  messages, share of questions whose document order changed, gold evidence recall and
  precision over unique exposed messages, API cost and latency.

## Guardrails

- The source run and both frozen controls are never written. Before any answer call the
  runner verifies, for every question and new condition, that the frozen document ids render
  to a prompt whose SHA-256 equals the frozen `prompt_sha256` and reproduce the frozen exposed
  source ids.
- New work goes to `.research_runs/evermembench_chronological_order_control_v1` with a
  resumable cache. The run manifest records protocol, runner, source and frozen-control hashes
  and refuses to resume after a change.
- A 5-question smoke run in a separate directory comes first; smoke outcomes are not used for
  any decision. Only defects that change no definition above may be fixed after smoke.
- No condition, ordering rule, rendering, scope or statistic changes after new outcomes.
