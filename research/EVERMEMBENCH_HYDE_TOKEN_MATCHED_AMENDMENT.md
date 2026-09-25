# EverMemBench amendment: token-matched HyDE arm (experiment C2)

Amendment version: `evermembench_hyde_token_matched_v1`
Recorded: 2026-09-16, before any answer of experiment C or C2 was generated. Neither
`.research_runs/final_sprint_C_hyde_raw_v1` nor `.research_runs/final_sprint_C2_hyde_token_matched_v1`
existed at the time of recording. The amendment file, runner and its own preflight are hashed into
the run manifest before the first answer call.

## Status

Pre-outcome amendment that **adds** an arm to experiment C of
`research/FINAL_EXPERIMENTS_PROTOCOL.md`. It does not modify, re-scope or re-price experiment C:
C's conditions, prompt, retrieval, comparisons and ceiling stay byte-identical, and C's recorded
preflight artifact (`.research_runs/final_sprint_preflight_v1/preflight_evermembench.json`,
sha256 `d5bcb9e1…`) is not rewritten.

Temporal Duration (`TP`) was selected after the full EverMemBench results were seen. C2, like every
other TP result in this project, is a post-hoc discovery slice: its p-values are nominal and are not
corrected for multiple comparisons.

## Defect in the registered comparison being addressed

Experiment C registers three comparisons: `HyDE-RAW − RAW`,
`RAW+EVENTS-token-matched-chrono − HyDE-RAW` and `RAW+EVENTS-token-matched-chrono − RAW`.

The first is clean: `HyDE-RAW` and `RAW` are both a raw top-10, about 1,052 context tokens, and
differ only in the query used for retrieval. That comparison alone answers the reviewer question
this experiment exists for — whether cheap query-side expansion reproduces the temporal effect.

The second is not clean. `HyDE-RAW` uses about 1,052 context tokens against 2,456 for
`RAW+EVENTS-token-matched-chrono`, and the two are presented in different orders (retrieval rank
versus chronological). Both differences are known to move this slice on their own: the budget
control was built precisely because context volume confounds the episode effect, and the
chronological control moved `RAW-token-matched` from 0.123 to 0.140 by ordering alone. An
events-over-HyDE difference measured under C would therefore re-introduce exactly the two confounds
that the budget and chronological controls were built to remove. The protocol notes the asymmetry in
its report; this amendment measures it instead.

## What is added

One new condition, run as experiment **C2**: `HyDE-RAW-token-matched-chrono`, the same 300 `TP`
questions.

- **Generations are not regenerated.** C2 reads the sealed `hyde_queries.jsonl` of experiment C, so
  both HyDE arms use byte-identical hypothetical passages. C must be sealed before C2 starts; a
  missing or unsealed `hyde_queries.jsonl` aborts the run. C2 makes no generation call.
- **Retrieval.** The hypothetical passage is embedded with `text-embedding-3-large` and ranks the
  sealed raw index of the frozen source run to depth 400 (`control.RANK_DEPTH`), the same depth the
  frozen budget and unlinked controls used. The embedding is content-addressed, so it reproduces the
  vector C used for the same passage.
- **Selection.** Documents are added in rank order while the rendered context stays at or below the
  per-question `context_tokens` of the frozen `RAW+EVENTS-token-matched` for that question; the first
  document that would exceed it is included only if that brings the count strictly closer to the
  target. This is `control.select_token_matched`, unchanged, the same rule the frozen controls used.
- **Ordering.** Ascending by timestamp, ties broken by document id (`chrono.chronological`,
  unchanged), so C2 is presented exactly like `RAW+EVENTS-token-matched-chrono`.
- **Everything else identical to C:** document rendering (`- {rendered}` lines), official open-ended
  answer prompt, `openai/gpt-4.1-mini` pinned to OpenAI (temperature 0, `max_tokens` 1000), official
  answer post-processing, official judge `google/gemini-3-flash-preview` pinned to Google AI Studio
  (`official_evaluator_no_max_tokens_v1`), `o200k_base` token counting.

C2 therefore differs from `RAW+EVENTS-token-matched-chrono` in exactly one respect: what the
documents are (HyDE-retrieved raw messages versus derived events plus raw messages). Token budget,
ordering, rendering, prompt, models and judge are matched.

## Primary endpoint (fixed before any outcome)

Paired accuracy difference **`RAW+EVENTS-token-matched-chrono` − `HyDE-RAW-token-matched-chrono`**
on the 300 `TP` questions, with a paired question bootstrap (20,000 samples, seed 20260912) and an
exact paired McNemar test. `RAW+EVENTS-token-matched-chrono` is the frozen chronological control and
is not regenerated.

Interpretation, fixed in advance:

- 95% CI lower bound > 0: event representation keeps an advantage over query-side expansion at an
  equal token budget and an equal order.
- 95% CI contains 0: event representation and token-matched HyDE are not distinguishable; the
  reported event advantage over HyDE cannot be separated from context volume and order.
- 95% CI upper bound < 0: token-matched HyDE is more accurate than the event condition.

## Secondary comparisons (all reported)

- `HyDE-RAW-token-matched-chrono` − `RAW-token-matched-chrono`: query-side expansion beyond raw
  retrieval at an equal budget and order.
- `HyDE-RAW-token-matched-chrono` − `HyDE-RAW`: the budget and ordering effect inside HyDE itself.
- `HyDE-RAW-token-matched-chrono` − `RAW`.

For every comparison: accuracy of both conditions, paired difference, question bootstrap 95% CI,
exact McNemar p, wins / losses / both correct / both wrong, per-project accuracies and differences,
a bootstrap over the five projects (20,000 samples, seed 20260912), and the exact two-sided
sign-flip test over the five project differences. With five projects the smallest attainable
two-sided sign-flip p is 0.0625 and the project bootstrap understates uncertainty; both are
descriptive.

Per condition: accuracy, mean context tokens, mean unique raw source messages, gold evidence recall
and precision over unique exposed messages, answer failure markers, judge parse branches and
re-requests, API usage and list-price cost.

## Guardrails

- Experiment C's run directory is read only for C2; its sealed `hyde_queries.jsonl` is verified
  against `hyde_queries_meta.json` and hashed into the C2 manifest.
- The frozen source, budget, unlinked and chronological directories are read only and verified
  against their `SHA256SUMS` before any call, as in C.
- New work goes to `.research_runs/final_sprint_C2_hyde_token_matched_v1` with a resumable
  content-addressed cache. Per-question document ids, context and prompt hashes and token counts are
  written to `inputs.jsonl` and hashed into `run_manifest.json` before the first answer call; a
  resumed run must reproduce them exactly.
- Gold answers and gold evidence are loaded only after the new predictions are sealed.
- Rank-depth exhaustion is counted per question and reported; the frozen `RAW-token-matched`
  condition filled a comparable budget well inside depth 400, so exhaustion is expected to be zero.
- The ceiling comes from this amendment's own preflight,
  `.research_runs/amendments/evermembench_hyde_token_matched_20260916/preflight_c2.json`. The
  protocol's preflight artifact is not regenerated.
- Stop criteria of `research/FINAL_EXPERIMENTS_PROTOCOL.md` §5 apply unchanged.
- No condition, selection rule, ordering rule, rendering, scope or statistic changes after any
  outcome, and no rerun with a changed definition.

## Cost

No generation call and no new question scope; C2 adds answers over longer contexts plus their
judgments. Computed offline by `--preflight-c2` and recorded in
`.research_runs/amendments/evermembench_hyde_token_matched_20260916/preflight_c2.json` before any
call:

| item | new calls | expected USD | ceiling USD |
|---|---:|---:|---:|
| HyDE generations | 0 (reused from C) | 0.0000 | 0.0000 |
| embeddings (`text-embedding-3-large`) | 3 batches | 0.0035 | 0.0086 |
| answers (`openai/gpt-4.1-mini`) | 300 | 0.3456 | 0.9783 |
| judge (`google/gemini-3-flash-preview`) | ≤ 300 | 0.1122 | 1.0902 |
| **total** | | **0.4613** | **2.0771** |

Answer input is 852,856 tokens over 300 questions, matching a mean target of 2,456.1 context tokens
per question. The runner stops when its running list-price spend × 1.10 exceeds the ceiling.

Together with experiment C (expected $0.3811, ceiling $2.0843) the HyDE block costs an expected
**$0.84**, ceiling **$4.16**.

## Execution order

C2 reads C's sealed generations, so C runs first:

```bash
python3 -m research.final_sprint_evermembench --run C      # seals hyde_queries.jsonl
python3 -m research.final_sprint_evermembench --freeze C
python3 -m research.final_sprint_evermembench --run C2     # reuses those generations
python3 -m research.final_sprint_evermembench --freeze C2
```
