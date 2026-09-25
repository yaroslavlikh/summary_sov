# Frozen external-evaluation protocol for TemporalEpisodes

Protocol version: `paper_external_temporal_episodes_v1`  
Frozen before any GroupMemBench or EverMemBench answer is generated.

Amendment: during GroupMemBench memory construction, before QA loading or any
answer/judge call, a deterministic turn-ID namespace compatibility defect was
found and corrected. The frozen pre-fix cache, audit, checksums, and exact
rationale are retained in
`.research_runs/amendments/groupmembench_namespace_fix_20260912/README.md`.
No prompt, model, threshold, retrieval rule, endpoint, or generated extraction
response changed.

## Fixed system

- Memory is built query-independently from chronological bounded chat sessions.
- Extractor/attacher: `gpt-4o-mini-2024-07-18`, temperature 0, frozen
  `temporal_episode_v2` contract.
- Episode attachment candidate generation: top 3 by the existing 0.6 active-history
  + 0.4 latest-event cosine score; only the network/topic is a hard filter.
- Resolver choices: `ATTACH_REVISE`, `ATTACH_REAFFIRM`, `ATTACH_AUGMENT`,
  `NEW_EPISODE`; confidence floor 0.5; false splits are preferred to uncertain merges.
- Each immutable event has mechanically validated exact-quote provenance. At most two
  events may cite one source turn. Up to five latest events form the hot retrieval text;
  older events and their provenance remain stored.
- Evaluation compares exactly two conditions: `RAW` and one unified
  `RAW+EPISODES` corpus. Both use the same answer model, judge, prompts, embedding
  model and top-k within a benchmark. Derived passages disclose their raw sources.

## SocialMemBench (completed discovery/external run)

- Existing completed run is preserved, not rerun or tuned.
- Official-compatible internal harness: `gpt-4o-mini-2024-07-18`,
  `text-embedding-3-small`, cosine top 10, 1,031 QA across 43 networks.
- Report both question-weighted and network-weighted accuracy, network-clustered
  bootstrap confidence intervals, evidence precision/recall and all question types.
- Freeze command: `python3 -m research.paper_runs --freeze-social`.

## GroupMemBench (confirmatory external run)

- Public source revision: `e2682e01ff490acfe4fac2940159dce60307dfc9`.
- Four 30,000-message domains; every channel/phase is split chronologically into
  non-overlapping blocks of at most 40 messages before query-independent extraction.
- Official dense retrieval model `text-embedding-3-large`, top 10.
- Official agent and judge prompts, both using `gpt-5`.
- **Primary endpoint fixed in advance:** paired accuracy delta on
  `knowledge_update + temporal` in solvability-filtered Finance+Technology.
- Other question types are secondary. Healthcare+Manufacturing are an explicitly
  separate unfiltered robustness analysis and are never pooled into the primary claim.
- Statistics: paired question bootstrap (20,000 samples, seed 20260912) and exact
  paired McNemar test. Report every domain/type cell, usage, latency and storage.
- One command: `caffeinate -i python3 -m research.paper_runs --group`.

## EverMemBench-Dynamic (independent robustness run)

- Public code revision: `e10b3d52f0e4cfc5c124ad406b5d95c59c73738b`.
- Public data revision: `a6b210a32248e841967b7b64a64281d2ff3f669d`.
- All five topics: 51,023 messages, 3,570 non-empty day/group sessions, 2,400 QA.
- Each day/group is one natural bounded query-independent extraction session.
- Unified dense retrieval uses `text-embedding-3-large`, top 10.
- Exact official answer prompt/model/provider: `openai/gpt-4.1-mini` through OpenRouter,
  pinned to OpenAI with fallbacks disabled.
- Exact official judge prompt/model/provider: `google/gemini-3-flash-preview` through
  OpenRouter, pinned to Google AI Studio with fallbacks disabled. Multiple choice is
  scored exactly, as in the official harness.
- Primary endpoint: paired accuracy delta across all 2,400 QA. Secondary slices:
  topic, open/multiple-choice, major and minor ID categories, plus evidence
  precision/recall against the released reference spans.
- Statistics: paired question bootstrap, topic-cluster bootstrap (20,000 samples,
  seed 20260912) and exact paired McNemar test. Record usage, latency and storage.
- Requires `LLM_API_KEY=<OpenRouter key>` in `.env` in addition to `OPENAI_API_KEY`.
- One command: `caffeinate -i python3 -m research.paper_runs --ever`.

## Immutability and reporting

- Every API response is content-addressed and every completed unit is append-cached.
  Exit code 75 means rate limit; the launcher waits 65 seconds and resumes.
- Dataset/code revisions, prompt and data hashes, raw predictions, judgments, token
  usage, per-call latency, memory snapshot, decisions and final report are retained.
- Completed prediction and result streams are sealed with exact row counts, unique
  `(qa_key, condition)` identities and SHA-256 checksums before the next phase may run.
- Reports include retrieval latency and a conservative API-cost estimate from actual
  token usage at uncached list prices frozen on 2026-09-12. Pricing sources:
  [OpenAI API pricing](https://developers.openai.com/api/docs/pricing),
  [GPT-5 launch pricing](https://openai.com/index/introducing-gpt-5-for-developers/),
  [GPT-4.1 mini on OpenRouter](https://openrouter.ai/openai/gpt-4.1-mini/pricing), and
  [Gemini 3 Flash on OpenRouter](https://openrouter.ai/google/gemini-3-flash-preview/pricing).
- When a report is complete the launcher creates a read-only timestamped freeze with
  `MANIFEST.json` and `SHA256SUMS`.
- These are adapter evaluations under official harnesses, not official leaderboard
  submissions. Negative and non-significant results are reported unchanged.
