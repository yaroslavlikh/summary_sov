# EverMemBench Temporal control: chronological order of the same retrieved context (v2)

Protocol version: `evermembench_chronological_order_control_v2`
Recorded: 2026-09-15, before any answer for any chronological condition was generated (no run
directory of v1 or v2 existed). The protocol file, runner, tests, input checksums, per-question
document ids and rendered-context hashes are written into the run manifest before the first
answer call.

## Supersedes v1 (never executed)

`EVERMEMBENCH_CHRONOLOGICAL_ORDER_CONTROL_PROTOCOL.md` (v1, sha256
`d4b83ff2881fce89b3a836cff877e4af48cd5bdfc8aa18cc827ce07106daefdf`) and its runner (sha256
`47939a96ad21d8581ac0dbfd6afeee4c9a45d9f35b4d4cd0853b4431ef32ce6a`) are kept unchanged; copies
are in `.research_runs/amendments/evermembench_chronological_control_v2_20260915/v1/`.

Reason for the change. Raw messages already carry absolute timestamps
(`[YYYY-MM-DD HH:MM:SS]`). v1 both sorted the documents and prefixed every context line with a
computed `[Day +N]` offset, so its primary condition changed two things at once: order and
content. v2 makes the primary control a pure permutation and keeps the day-offset rendering only
as a separate exploratory condition. v1 also lacked a pre-call freeze of document ids and
context hashes, explicit permutation checks, and wrote progress logs into the source run
directory through shared helpers; v2 fixes these implementation defects. The hypothesis, scope,
models, prompts, judge and statistics are unchanged.

## Motivation

On the frozen 300 Temporal Duration (`TP`) questions: RAW+EPISODES = 0.200,
RAW-token-matched = 0.123, RAW+EVENTS-token-matched = 0.170. Episodes present events and their
source messages in time order, while the token-matched raw and unlinked-event contexts are
presented in retrieval-rank order. This control tests whether ordering the same retrieved
documents by time explains the episode or event advantage. It was proposed after the earlier
outcomes on these questions were known, so it is a mechanism control, not a replication.

## Inputs (read only)

- Source run `evermembench_temporal_episodes_official_v1`, frozen at
  `.research_runs/frozen/evermembench_temporal_episodes_official_v1_20260915_021550_MSK`.
- Budget control, frozen at `.research_runs/frozen/evermembench_temporal_budget_control_v1_20260915_021134_MSK`.
- Unlinked-events control, frozen at
  `.research_runs/frozen/evermembench_unlinked_events_control_v1_20260915_022532_MSK`.

No memory extraction, episode or event rebuilding, embedding, retrieval, top-k or document
selection is performed. Document sets are copied from the frozen predictions.

## Conditions

Identical to the frozen controls in everything not listed: official open-ended answer prompt,
`openai/gpt-4.1-mini` (temperature 0, max_tokens 1000, provider pinned to OpenAI), official
answer post-processing, official judge `google/gemini-3-flash-preview` with the amended official
scoring (`official_evaluator_no_max_tokens_v1`), `o200k_base` token counting on the rendered
context string, the same document rendering (`- {rendered document}` lines joined by newlines).

Ordering rule. Documents are sorted ascending by timestamp, ties broken by document id. A raw
message's timestamp is the `[YYYY-MM-DD HH:MM:SS]` prefix of its rendered text (a date-only
prefix counts as `00:00:00`). A derived event's timestamp is the earliest timestamp among its
source messages.

1. `RAW-token-matched-chrono` — **primary control.** Exactly the documents of frozen
   `RAW-token-matched` for the question, each rendered byte-identically, in chronological order.
   Nothing is added or removed.
2. `RAW+EVENTS-token-matched-chrono` — secondary mechanism control. Exactly the documents of
   frozen `RAW+EVENTS-token-matched`, byte-identical, in chronological order.
3. `RAW-token-matched-chrono-relative` — exploratory ablation, never pooled with the ordering
   control. The documents and order of condition 1, each line rendered as
   `- [Day +N] {rendered document}`, where N is the number of calendar days from the date of the
   earliest document in the context. Its context is slightly longer than condition 1.

References (not re-generated): sealed `RAW`, `RAW+EPISODES`; frozen `RAW-token-matched`,
`RAW+EVENTS-token-matched`.

## Pre-call verification (every question, every new condition)

- exactly 300 questions, 60 per project, matching the frozen control scope;
- the frozen documents render to a prompt whose SHA-256 equals the frozen `prompt_sha256`;
- the multiset of document ids equals the frozen condition's; no duplicate ids;
- the multiset of exposed raw source ids equals the frozen condition's;
- every document block is byte-identical to its block in the frozen context, and the new context
  is exactly a permutation of those blocks (conditions 1 and 2);
- the prompt differs from the frozen prompt only by that context;
- every timestamp parses, and the sort keys are unique (deterministic order);
- the three frozen directories pass `SHA256SUMS`, and the live source files read by the runner
  equal their frozen copies.

The per-question inputs (source and ordered document ids, timestamps, context and prompt
hashes, token counts) are written to `inputs.jsonl` and hashed into `run_manifest.json` before
the first answer call; a resumed run must reproduce them exactly. Gold answers and gold evidence
are not loaded until all predictions are sealed.

## Primary endpoint (fixed before new outcomes)

Paired accuracy difference **RAW+EPISODES − RAW-token-matched-chrono** on the 300 `TP`
questions: paired question bootstrap (20,000 samples, seed 20260912) and exact paired McNemar.

Interpretation rule for the primary interval:

- lower bound > 0: the episode gain is not reproduced by chronological ordering of the
  equal-token raw context;
- interval contains 0: episodes and chronologically ordered raw context are not distinguishable;
- upper bound < 0: chronologically ordered raw context is more accurate than episodes.

Interpretation of the accuracy of `RAW-token-matched-chrono` (fixed bands, midpoints between the
frozen reference accuracies 0.123, 0.170 and 0.200):

- below 0.147: supports a contribution of event representation beyond context volume and order;
- from 0.147 to below 0.185: ordering explains part of the effect; event abstraction may keep an
  independent contribution;
- 0.185 or higher: an advantage of event representation over chronological presentation cannot be
  claimed.

Episode linking is described as contributing beyond chronologically ordered unlinked events only
if the question-bootstrap lower bound of RAW+EPISODES − RAW+EVENTS-token-matched-chrono is above 0;
otherwise its independent contribution is reported as not established.

## Secondary comparisons (all reported)

- RAW-token-matched-chrono − RAW-token-matched (pure ordering effect, raw);
- RAW+EVENTS-token-matched-chrono − RAW+EVENTS-token-matched (pure ordering effect, events);
- RAW+EPISODES − RAW+EVENTS-token-matched-chrono (linking beyond event extraction and ordering);
- RAW+EVENTS-token-matched-chrono − RAW-token-matched-chrono (events beyond raw, both ordered).

Exploratory: RAW-token-matched-chrono-relative − RAW-token-matched-chrono (day offsets);
RAW+EPISODES − RAW-token-matched-chrono-relative.

For every comparison: accuracy of both conditions, paired difference, question bootstrap 95% CI,
exact McNemar p, wins / losses / both correct / both wrong, per-project accuracies and
differences, a bootstrap over the five projects (20,000 samples, seed 20260912), and the exact
two-sided sign-flip test over the five project differences. With five projects the smallest
attainable two-sided sign-flip p is 0.0625 and the project bootstrap understates uncertainty;
both are descriptive.

Per condition: accuracy, mean context tokens, mean unique raw source messages, share of questions
whose order changed (new conditions), gold evidence recall and precision over unique exposed raw
messages, mean answer and judge latency where recorded, answer failure markers, judge parse
branches, re-requests and finish reasons. API usage and list-price cost are reported for new work,
together with the OpenRouter key usage before and after the run.

## Guardrails

- Source and frozen artifacts are never written; the runner logs only to its own directories.
- New work goes to `.research_runs/evermembench_chronological_order_control_v2` (full) and
  `.research_runs/evermembench_chronological_order_control_smoke_v2` (5 questions, first per
  project). Calls are cached and append-only; a resumed run refuses changed manifests or inputs.
- Smoke is only a technical check (failure markers, parsing, row counts); its accuracy is not
  inspected or used.
- After the full run: predictions and results are sealed, a Markdown report and `summary.json` are
  written, and the run and smoke directories are frozen with `MANIFEST.json` and `SHA256SUMS`.
- No condition, ordering rule, rendering, scope or statistic changes after new outcomes, and no
  rerun with a changed definition.
