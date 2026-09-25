# GroupMemBench amendment: answers regenerated with a larger completion budget

Amendment version: `groupmembench_answer_cap_rerun_v1`
Recorded: 2026-09-15, after the original GroupMemBench run was scored and reported, before any
answer under this amendment was generated. The amendment file and runner are hashed into the
run manifest before the first answer call.

## Status

Post-outcome measurement amendment. It does **not** replace the protocol's primary result.
`research/PAPER_BENCHMARK_PROTOCOL.md` fixed the GroupMemBench primary endpoint, and the
original run remains the reported primary result: RAW 0.340, RAW+EPISODES 0.300, paired
delta −0.040, 95% CI [−0.093, +0.013], McNemar p = 0.238. Results under this amendment are
reported next to it as a secondary, measurement-corrected analysis.

## Defect being corrected

`paper_benchmark_common.CachedAPI.chat` sends `max_completion_tokens = max(4 * max_tokens,
2000)` for `gpt-5`. The answer call used `max_tokens=512`, i.e. a 2,048-token completion
budget that includes hidden reasoning. In the original run 102 of 1,490 answers were empty
(finished without visible text after exhausting the budget) and were scored incorrect, more
often for RAW+EPISODES, whose contexts are longer. The budget was not specified by the
protocol text and is not part of the official GroupMemBench setup; it is an implementation
limit of this adapter.

## What changes

Only the answer completion budget: the answer call passes `max_tokens=2048`, so `gpt-5`
receives `max_completion_tokens = 8192`. All 1,490 answers (745 questions × RAW,
RAW+EPISODES) are regenerated, not only the empty ones, so that both conditions and all
questions come from one sampling regime.

## What stays identical

- Sealed memory (`episodes.jsonl`, snapshot hash verified), sealed retrieval: the retrieved
  document ids of every question and condition are copied from the sealed predictions, and the
  runner verifies, before any call, that the reconstructed answer request (system prompt, user
  prompt, model, temperature, original `max_tokens=512`) hashes to a key present in the sealed
  answer cache. A mismatch aborts.
- Official GroupMemBench agent and judge prompts at commit `e2682e01ff49`, model `gpt-5` for
  answers and judging, answer and judge parsing (`_split_final`, `_parse_judgment`), judge call
  (`max_tokens=256`, i.e. budget 2,000; the original run had no empty judge output).
- Endpoint definitions and statistics of the frozen protocol.

## Endpoints under this amendment (fixed before new answers)

- Primary endpoint of the protocol recomputed on regenerated answers: paired accuracy delta
  RAW+EPISODES − RAW on filtered Finance and Technology, knowledge_update and temporal
  questions (150); paired question bootstrap (20,000 samples, seed 20260912) and exact McNemar.
- All 745 questions; every domain × question type cell; the five categories shared with the
  published evaluation (606 questions).
- Measurement health: empty answers per condition (target: none), answer output tokens.
- Stability: per condition, accuracy of regenerated versus original answers (paired).

Interpretation of the recomputed primary endpoint, fixed in advance:

- 95% CI lower bound > 0: with an adequate completion budget, episodes improve the primary
  endpoint.
- 95% CI contains 0: no difference is detected with an adequate completion budget.
- 95% CI upper bound < 0: with an adequate completion budget, episodes reduce accuracy on the
  primary endpoint.

In all cases the paper reports the original result as the protocol's primary endpoint and
this result as the amendment.

## Guardrails

- The original run directory is read only; its predictions, results and episode snapshot are
  hashed into the run manifest.
- New work goes to `.research_runs/groupmembench_answer_cap_rerun_v1` with a resumable cache.
  The manifest records amendment, runner, source hashes and scope, and refuses to resume after
  a change.
- A smoke run (first question of each domain, both conditions) in a separate directory comes
  first; its outcomes are not used for any decision. If smoke shows an empty answer at 8,192
  tokens, the run stops and a further amendment is recorded before continuing.
- GroupMemBench data, questions, prompts and per-question outputs are not redistributed;
  only aggregates and hashes are published.
- Estimated cost from the original run's usage: USD 25–40 (`gpt-5` answers and judgments).
