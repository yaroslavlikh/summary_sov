# EverMemBench Temporal ablation: linked episodes versus unlinked events

Protocol version: `evermembench_unlinked_events_control_v1`
Recorded: 2026-09-15, before any answer for the new conditions was generated. The protocol
file and runner are hashed into the run manifest before the first answer call.

## Status and motivation

Exploratory mechanistic ablation on the same 300 Temporal Duration (`TP`) questions as
`evermembench_temporal_budget_control_v1` (frozen at
`.research_runs/frozen/evermembench_temporal_budget_control_v1_20260915_021134_MSK`).
That control showed RAW+EPISODES = 0.200 versus 0.123 for an equal-token raw context
(+0.077, 95% CI [+0.033, +0.120]). An episode combines two things: derived event
summaries with provenance, and their chronological linking into one storyline. This
ablation keeps the same derived events and provenance and removes only the linking.
The Temporal effect was discovered on these data, so this is not an independent
replication.

## Primary endpoint (fixed before new outcomes)

Paired accuracy difference **RAW+EPISODES − RAW+EVENTS-token-matched** on the 300 `TP`
questions; paired question bootstrap (20,000 samples, seed 20260912) and exact paired
McNemar test. RAW+EPISODES is the sealed prediction and judgment.

Interpretation, fixed in advance:

- 95% CI lower bound > 0: chronological linking adds accuracy beyond the same unlinked
  events at an equal token budget.
- 95% CI contains 0: the linking effect is not distinguishable from event summaries alone.
- 95% CI upper bound < 0: unlinked events at an equal token budget are at least as accurate.

## Conditions

Everything not listed here is identical to the budget control: sealed memory, sealed
`text-embedding-3-large` raw index and question embeddings (read only), official
open-ended answer prompt, `openai/gpt-4.1-mini` (temperature 0, max_tokens 1000),
official answer post-processing, official judge `google/gemini-3-flash-preview` with the
amended official scoring, `o200k_base` token counting on the rendered context string.

Event documents. Every event stored in the sealed `episodes.jsonl` (all events, including
events outside an episode's five-event hot window) becomes one document:

- index text: `Viewpoint owner: {owner}. Subject: {subject}. {event_text}` using the event's
  own owner and subject (`?` when missing), matching the episode index-text format;
- rendered text: `[DERIVED EVENT / owner={owner} / subject={subject}]`, the event text, and
  `[SOURCE id]` lines for the event's own validated source messages, matching the episode
  rendering without the multi-event chronology.

Event vectors are computed once with `text-embedding-3-large` into
`.research_runs/evermembench_unlinked_events_index_v1`, sealed with per-topic document and
matrix hashes, and reused read only. The unified index per topic is the sealed raw matrix
followed by the event matrix; ranking uses the same cosine scoring as the sealed run.

- `RAW` and `RAW+EPISODES`: sealed rows of the source run.
- `RAW-token-matched`: frozen rows of the budget control (reference only).
- `RAW+EVENTS` (new): unified raw+events top-10.
- `RAW+EVENTS-token-matched` (new): documents from the unified raw+events ranking, added
  while the rendered context stays at or below the RAW+EPISODES context token count for
  that question; the first overshooting document is included only if strictly closer.

No rerun calibration is repeated: the budget control measured run-to-run noise on these
questions (RAW+EPISODES rerun delta +0.000, agreement 96%).

## Secondary analyses (all reported)

- RAW+EPISODES − RAW+EVENTS.
- RAW+EVENTS-token-matched − RAW-token-matched (value of event summaries over raw text at
  an equal token budget).
- RAW+EVENTS − RAW.
- Per condition: accuracy, mean unique raw source messages, mean context tokens, mean number
  of derived documents in context, gold evidence recall and precision over unique exposed
  messages, API cost and latency.

## Guardrails

- The source run, the frozen budget control and the event index after sealing are never
  written. Missing sealed vectors abort; the event index refuses to change once sealed.
- Before any answer call the runner verifies, for every question, that the sealed raw top-10
  and the sealed RAW+EPISODES prompt are reproduced, and that the number of event documents
  equals the number of events in the sealed memory.
- New work goes to `.research_runs/evermembench_unlinked_events_control_v1` with a resumable
  cache. The run manifest records protocol, runner, source, control and event-index hashes
  and refuses to resume after a change.
- A 5-question smoke run in a separate directory comes first; smoke outcomes are not used
  for any decision. Only defects that change no definition above may be fixed after smoke.
- No condition, selection rule, rank depth, scope or statistic changes after new outcomes.
