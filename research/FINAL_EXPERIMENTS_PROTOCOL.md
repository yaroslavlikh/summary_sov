# Final validation sprint: protocol and preflight

Version: `final_sprint_v1`. Recorded 2026-09-15 after a read-only audit and an offline preflight;
no API call has been made for any experiment below. Every run hashes this file, its runner, the
dependencies and its frozen inputs into `run_manifest.json` before the first call, uses its own
append-only run directory and `api_usage.jsonl`, and is frozen with `MANIFEST.json` and
`SHA256SUMS` afterwards. Existing sealed or frozen runs are read only. Production code is not
touched. Execution order is A → B → C → D; each block stops for an explicit `GO NEXT`.

Temporal Duration (TP) was selected after the full EverMemBench results were seen. Every TP
result, old or new, is a post-hoc discovery slice. The full 2,400-question run of experiment A is
an exploratory robustness check, not a confirmation.

## 1. Frozen inputs (read only)

| artifact | path | SHA-256 |
|---|---|---|
| EverMemBench source run, `SHA256SUMS` (573 files) | `.research_runs/frozen/evermembench_temporal_episodes_official_v1_20260915_021550_MSK/` | `9dae482ef6e984b119dc76ed5a4fa214d5f5f643a1a68e0cde4b70ce712cf37f` |
| ↳ `run/predictions.jsonl` (RAW, RAW+EPISODES, 4,800) | | `2d25c1fd0b884ee40c6605c8bafbef226ea72357b0c62e70b20cfc2fb2c9f8ba` |
| ↳ `run/results.jsonl` | | `fb9646a610a5eba5292205e4b419ad755171b6336327aceeff7eb20ca92ad583` |
| ↳ `run/episodes.jsonl` (query-independent memory) | | `2068fa9ca1a2e4507f855fbbccac5afd5c200ba22f0b871d328acd4622c2114d` |
| ↳ `run/attach_decisions.jsonl` | | `b4a8fbc850b606cbd218116aae1770f5a68c6d371efbe70b3bc8f0e97636e63e` |
| ↳ `run/openrouter_chat_cache.jsonl` | | `b918de49b6202a36684df302405e26b438ee87b545f61b5461b9bfdddbe1130d` |
| ↳ `run/api_usage.jsonl` | | `e7c7058aabdbcdedb3f4f6a2c25938f2a968067081b4893b65bdbff1a6ae8b0e` |
| Budget control, `SHA256SUMS` (31) | `.research_runs/frozen/evermembench_temporal_budget_control_v1_20260915_021134_MSK/` | `cd90231d87962d2ca152495cf7510fb9fb52c9a1164f66799b96825bbb4b7089` |
| ↳ `run/predictions.jsonl` / `run/results.jsonl` | | `4ac09713…e53712` / `02904a20…91b8d9` |
| Unlinked-events control, `SHA256SUMS` (170) | `.research_runs/frozen/evermembench_unlinked_events_control_v1_20260915_022532_MSK/` | `ca9a75a4b2b30ad731bc8492b7c0e537f42d549ddd5bb1d39f47a09ca8ceb7fe` |
| ↳ `run/predictions.jsonl` / `run/results.jsonl` | | `0770e335…2deb6f` / `1d383fbf…65fa4f4c` |
| ↳ event index `related/evermembench_unlinked_events_index_v1/index_meta.json` | | `7c560190ff465b49e0433bfe79836e7ffa6e3b4b6924ca0e88e25d2ebd1f48f1` |
| Chronological control v2, `SHA256SUMS` (36) | `.research_runs/frozen/evermembench_chronological_order_control_v2_20260915_190741_MSK/` | `c83d8c168b7fe06b79a7c4336f4feda3aae813f059a3821baebc30cc63cefb02` |
| ↳ `run/predictions.jsonl` / `run/results.jsonl` | | `3d98438a…2b8b87f7` / `2fe74659…0aab3` |
| GroupMemBench source run (not frozen; hashed, read only) | `.research_runs/groupmembench_temporal_episodes_official_v1/` | |
| ↳ `predictions.jsonl` | | `b72076901af52133aa74b74857784853a89fd7dbe7d6d9de1a66b46dbdec3517` |
| ↳ `results.jsonl` | | `3ef95245e6e946f5a0bb1e19cebc17aef32f8b1c6ed92433635f071b2171863b` |
| ↳ `episodes.jsonl` | | `100f13d9cd0055935f56250af2afdb6343f2b93324d8c71e7a3f03fdeaa9a064` |
| ↳ `openai_chat_cache.jsonl` / `api_usage.jsonl` | | `e1ee99d9…ab600526` / `3a3a7d0a…6ab0f1f` |

Offline audit results: all four frozen EverMemBench directories pass `SHA256SUMS` and their sealed
predictions/results pass row-count and hash checks; the live unlinked-events index equals its frozen
copy. EverMemBench has 2,400 QA: 1,638 multiple-choice (scored locally by the official exact
comparison) and 762 open-ended (official LLM judge), TP = 300 open-ended.

Runners: `research/final_sprint_evermembench.py` (sha256 `37f36b07…fefa54` at preflight),
`research/final_sprint_groupmembench_repair.py` (`62df0718…0fcfefa54`), `research/final_sprint_audit_sheets.py`
(`e55769e8…b0f43`), tests `tests/test_final_sprint.py` (`33bfc5a1…bb1d8a5`). Preflight reports:
`.research_runs/final_sprint_preflight_v1/preflight_evermembench.json` (`d5bcb9e1…4c5d2`) and
`preflight_groupmembench_repair.json` (`94154652…19f335`).

## 2. Fixed harness

- Answer: `openai/gpt-4.1-mini` via OpenRouter, provider pinned to `openai`, `allow_fallbacks: false`,
  temperature 0, `max_tokens` 1000, official EverMemBench answer prompts (open-ended and multiple-choice
  templates), official answer post-processing.
- Judge: `google/gemini-3-flash-preview` via OpenRouter, provider pinned to `google-ai-studio`, no
  fallback, temperature 0, no `max_tokens`, official judge prompt and parser
  (`official_evaluator_no_max_tokens_v1`). Multiple-choice answers are scored locally.
- Retrieval: `text-embedding-3-large` vectors from the sealed source run, cosine top-10, the same scoring
  and ordering as the sealed runs.
- Cache reuse: before any call, the new run's cache is seeded in memory with every entry of the four frozen
  OpenRouter caches (8,467 entries). A call whose request payload hashes to an existing key is answered from
  that entry, and the prediction records `answer_cache: reused:<run>`; only misses reach the API. Gold answers
  and gold evidence are loaded only after the new predictions are sealed.
- GroupMemBench: unchanged direct OpenAI `gpt-5` protocol (see D).

## 3. Cost model

Prices: standard uncached list prices from `paper_benchmark_common.MODEL_PRICES_USD_PER_M`
(gpt-4.1-mini $0.40 / $1.60, gemini-3-flash $0.50 / $3.00, gpt-5 $1.25 / $10.00, text-embedding-3-large
$0.13 per million tokens). Calibration from the actual sealed `api_usage.jsonl`:

- answer prompt tokens = `o200k_base` count of the exact prompt + 7.0 (mean API overhead measured on the 2,400
  reconstructed sealed RAW prompts);
- answer output tokens: 2.0 (multiple-choice), 9.31 (open-ended), measured on the same prompts;
- judge: 483 input / 41.0 output tokens (all questions, source run); 506 / 40.3 (TP, chronological control);
- GroupMemBench: the 102 empty requests had 2,534 input tokens on average; judge 134 / 212.

Ceilings: every new answer uses its full `max_tokens`; judge input +20% and 1,000 output tokens; HyDE
generation 200 output tokens; all multiplied by 1.10 for provider fees. The runner stops when its running
list-price spend × 1.10 exceeds the ceiling.

## 4. Experiments

### A. RAW+EVENTS on all 2,400 EverMemBench QA (main)

- Memory: the 16,859 sealed events as individual documents (`[DERIVED EVENT / owner / subject]`, event text,
  raw source lines), no linking; sealed event index.
- Retrieval: unified raw + event cosine top-10, exactly as the frozen unlinked-events control.
- Baselines not regenerated: sealed RAW and RAW+EPISODES.
- Offline checks passed: 2,400/2,400 raw top-10 lists and RAW prompts reproduce the sealed run; 300/300 TP
  RAW+EVENTS top-10 lists equal the frozen unlinked control.
- Reuse: 1,068 answers. 781 questions have no event document in their top-10, so the prompt is byte-identical
  to the sealed RAW prompt; 287 TP prompts equal the frozen unlinked control.
- New calls: **1,332 answers**; **at most 442 judge calls** (open-ended questions with a new answer; fewer if an
  answer string repeats a cached judgment); 1,638 multiple-choice scored locally.
- Cost: expected **$1.12** (answers $0.96, judge $0.16); ceiling **$4.99**.
- Mean context: 1,177 tokens (RAW 1,052 on TP).

### B1. Events without description (300 TP)

- Documents: exactly the frozen `RAW+EVENTS-token-matched` documents, in exactly the frozen chronological
  order of `RAW+EVENTS-token-matched-chrono` (verified for 300/300).
- Rendering: raw documents unchanged; every event document replaced by its raw source messages
  (`[timestamp][group][speaker] text`), without event text, owner, subject or the derived label. The context is
  shorter (1,908 vs 2,456 tokens): removing the description is the intervention, so B1 is not token-matched.
- Reuse: 10 prompts (no event document) equal the frozen chronological prompts.
- New calls: **290 answers**, **at most 290 judge calls**. Cost: expected **$0.38**, ceiling **$1.86**.
- Comparisons: B1 − RAW+EVENTS-token-matched-chrono (primary for B1: does the description matter);
  B1 − RAW-token-matched-chrono; B1 − RAW+EVENTS-token-matched.

### B2. Chronological episodes (300 TP)

- Documents: exactly the sealed RAW+EPISODES top-10 documents (sealed prompts reproduced for 300/300), sorted
  globally by earliest source timestamp (ties by document id); each episode rendered unchanged, so its events stay
  in chronological order. Pure permutation verified for 300/300; order changes in 300/300.
- Reuse: 14 prompts already present in frozen caches.
- New calls: **286 answers**, **at most 286 judge calls**. Cost: expected **$0.44**, ceiling **$1.91**.
- Comparisons: B2 − RAW+EVENTS-token-matched-chrono (primary for B2: events vs episodes with symmetric order);
  B2 − RAW+EPISODES; B2 − RAW-token-matched-chrono.

### C. HyDE-RAW (300 TP)

- Step 1: `openai/gpt-4.1-mini` (same route, temperature 0, `max_tokens` 200) receives only the question text in
  this fixed prompt: “Write a short passage, two to four sentences, of the kind that could appear in a workplace
  team chat and would contain the answer to the question below. Write plausible chat content and invent specific
  details if needed. Output only the passage. Question: {question}”. No gold, evidence, category label, id or
  metadata. Generations are sealed in `hyde_queries.jsonl`; an empty generation stops the run.
- Step 2: the passage is embedded with `text-embedding-3-large` (direct OpenAI, cached in the run) and retrieves the
  raw top-10 from the sealed raw index; the retrieved ids are sealed in `hyde_retrieval.jsonl` before answering.
- Step 3: official answer and judge on raw messages only.
- New calls: **300 generations**, **3 embedding batches**, **300 answers** (fewer if a retrieval reproduces a
  cached prompt), **at most 300 judge calls**. Cost: expected **$0.38**, ceiling **$2.08**.
- Comparisons: HyDE-RAW − RAW; RAW+EVENTS-token-matched-chrono − HyDE-RAW; RAW+EVENTS-token-matched-chrono − RAW.
  HyDE-RAW uses about 1,050 context tokens (top-10 raw) versus 2,456 for the events condition; the report states
  this asymmetry.

### D. GroupMemBench empty-answer repair (v2)

- Audit: 1,490/1,490 sealed answer requests reproduced from sealed retrieval. Exactly 102 empty answers (RAW 44,
  RAW+EPISODES 58). For all 102 the sealed usage row shows 2,048 output tokens (= `max_completion_tokens`) and the
  cached completion is empty: hidden reasoning exhausted the budget.
- Repair: same messages, model `gpt-5`, direct OpenAI route, parsing and judge; only for the 102 failed requests the
  completion budget is 8,192, with one retry at 16,384 if the completion is empty or cut by length. Each repair
  stores the original error, retry configuration, per-attempt finish reason and token usage (including reasoning
  tokens) and the final completion. The 1,388 valid answers and their judgments are copied, not regenerated.
- New calls: **102 answers**, **at most 102 retries**, **102 judge calls**. Cost: expected **$4.64** (assuming ~4,000
  output tokens per repaired answer), ceiling **$27.77** (both attempts at full budget).
- Results: full 1,490-answer set re-scored; the original run stays the protocol's primary GroupMemBench result.
- This supersedes the prepared full-rerun amendment `research/GROUPMEMBENCH_ANSWER_CAP_AMENDMENT.md`, which is not
  executed.

### Totals

| experiment | new answer calls | other new calls | expected USD | ceiling USD | route |
|---|---:|---|---:|---:|---|
| A | 1,332 | ≤ 442 judge | 1.12 | 4.99 | OpenRouter |
| B1 | 290 | ≤ 290 judge | 0.38 | 1.86 | OpenRouter |
| B2 | 286 | ≤ 286 judge | 0.44 | 1.91 | OpenRouter |
| C | ≤ 300 | 300 generations, 3 embedding batches, ≤ 300 judge | 0.38 | 2.08 | OpenRouter + OpenAI embeddings |
| D | 102 | ≤ 102 retries, 102 judge | 4.64 | 27.77 | OpenAI direct |
| **total** | | | **6.96** | **38.61** | |

OpenRouter reserves credit per in-flight request, and the judge is called without `max_tokens`; with a low balance
use the documented worker limits (`--workers 4 --judge-workers 2`) or top up first.

## 5. Stop criteria

- Any frozen checksum, sealed row count, reconstruction or permutation check fails → no call is made.
- Running spend estimate × 1.10 exceeds the experiment ceiling → stop.
- Answer failure markers (`[EMPTY]`, `[INVALID]`, `[FAILED]`) exceed max(3, 2% of answers) → stop before judging.
- A judge output empty or length-cut three times → the official judge wrapper raises and the run stops.
- An empty HyDE generation → stop.
- D: the number of empty answers found is not exactly 102 → stop; answers still failing after 16,384 are reported
  as `still_failed`, never re-prompted differently.
- Rate limits or credit errors stop the run; completed work is cached and the same command resumes it. No definition
  changes after seeing any result; no rerun with a changed definition.

## 6. Result tables that will be produced

- A `report.md` / `summary.json`: conditions (RAW, RAW+EPISODES, RAW+EVENTS: accuracy, evidence recall and precision,
  unique sources, context tokens, answer latency); paired RAW+EVENTS − RAW and RAW+EVENTS − RAW+EPISODES (delta,
  question bootstrap CI, McNemar, win/loss/tie, per-project deltas, project bootstrap, sign-flip); all nine official
  categories with TP marked as a post-hoc slice; all-except-TP comparison; API calls, tokens, list cost and billed
  delta.
- B1, B2, C: condition table and the three paired comparisons listed above, same statistics, cost.
- D: repair outcome by condition (original empty / repaired / still failed); primary endpoint and all 745 questions
  for original and repaired answers; cost.
- After all blocks: `FINAL_EVIDENCE_TABLE.md`, `FINAL_RESULTS_FOR_PAPER.md`, manifests and exact commands.

## 7. Commands (run by the author after GO)

```bash
python3 -m research.final_sprint_evermembench --run A
python3 -m research.final_sprint_evermembench --freeze A
python3 -m research.final_sprint_evermembench --run B1
python3 -m research.final_sprint_evermembench --freeze B1
python3 -m research.final_sprint_evermembench --run B2
python3 -m research.final_sprint_evermembench --freeze B2
python3 -m research.final_sprint_evermembench --run C
python3 -m research.final_sprint_evermembench --freeze C
python3 -m research.final_sprint_groupmembench_repair --run
```

## 8. Manual audit tooling (no model verdicts)

`python3 -m research.final_sprint_audit_sheets` wrote `.research_runs/final_sprint_audit_sheets_v1/`:
`events_review.csv` (200 events: 50 per create / revise / augment / reaffirm, 5 per project × early/late half of the
project timeline) and `attach_review.csv` (200 decisions: 50 per NEW_EPISODE with candidates / REVISE / AUGMENT /
REAFFIRM, same stratification, candidate histories reconstructed at decision time, arbiter rationales omitted). All
verdict columns are empty; this is not a human audit until a person fills them in.

## 9. Deliberately not done

No new memory extraction or linking calls, no new memory types, prompts or architectures, no BM25, reranker, graph
baseline or neighbor windows, no additional benchmarks, no change of GroupMemBench model or route, no rerun of valid
answers, no model-assigned audit verdicts, no tuning after results.
