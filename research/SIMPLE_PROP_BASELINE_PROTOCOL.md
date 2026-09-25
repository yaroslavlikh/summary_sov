# SIMPLE-PROP baseline: protocol

Recorded 2026-09-21, before any paid call. Addresses the gap named in the manuscript's own
§10 ("Границы переноса" / limits): *"the work contains no comparison with a plain paraphrase
of the same messages equipped with the same sources and speaker/time fields."* Three
independent reviews of the manuscript (two LLM reviewers, one prior Claude session) converged
on this as the single highest-leverage missing experiment before submission.

## Question

The paper's headline evidence-delivery result is EVENTS − RAW: **+3.21 pp precision**
(95% CI [+2.84, +3.59]), **+0.64 pp recall**, on all 2,400 EverMemBench questions. Does this
survive against a baseline that is self-contained and carries identical provenance, but has
none of the Elem event ontology (no `viewpoint_owner` / `subject` / `event_type` /
`temporal_mode`)? If SIMPLE-PROP recovers most of the +3.21 pp, the schema's contribution is
mainly "make the text self-contained," which prior work (Dense X Retrieval, RAPTOR) already
established. If SIMPLE-PROP recovers little of it, the schema is doing something beyond
self-containment.

## What SIMPLE-PROP is

Same query-independent session clustering as the sealed EVENTS extraction
(`query_independent_episode_pipeline.build_query_independent_clusters`: every session becomes
one cluster containing its complete turn set, scoped purely from conversation structure, QA
never read). Same full-session context rendering, same model
(`gpt-4o-mini-2024-07-18`, temperature 0, `max_tokens=8192`), same cap of at most 2 items per
turn, same evidence contract (`{"turn_id", "quote"}`, quote mechanically verified as a literal
substring of the cited message via the same validator functions
(`temporal_episode_prototype._quote_is_valid_substring`, `_resolve_turn_id`,
`_is_generic_fragment`), same speaker/time metadata (every SIMPLE-PROP document embeds the
exact same `[SOURCE turn_id] [time][Group][Speaker] text` lines as EVENTS documents, reusing
`raw_by_id` unchanged).

The only thing that changes: the prompt asks for one self-contained rewritten sentence
(`text`) with no owner/subject/type/temporal-mode fields, and the retrieval document's
`index_text` is that sentence alone — no `"Viewpoint owner: ... Subject: ..."` prefix. No
episode linking: SIMPLE-PROP is a flat document pool, structurally matching the EVENTS
condition of the headline result (not the linked-episode condition).

## What is read-only (sealed, zero cost)

- Sealed RAW message index matrices and the 2,400 sealed question-embedding batches, from
  `evermembench_temporal_episodes_official_v1_20260915_021550_MSK/run/embeddings/`
  (`evermembench_temporal_budget_control.raw_index_matrix` / `.question_vectors`, which refuse
  to recompute if the cache is missing).
- Sealed gold source IDs, from `base.load_questions(include_gold=True)`.
- Sealed EVENTS document set and its embedding matrix, from
  `evermembench_unlinked_events_control.events_by_topic` and
  `final_sprint_evermembench.event_matrix` (the same sealed index that produced the paper's
  +3.21 pp figure) — used only as the right-hand side of the EVENTS − PROP comparison, never
  modified.

## What is new (paid)

- Extraction: one call per session cluster across the 5 EverMemBench projects — the same
  ~3,570 sessions the sealed EVENTS extraction covered, but **no linking/attach phase** (flat
  documents), so call count is a fraction of the sealed run's 20,424 combined
  extraction+attach calls.
- Embeddings: `text-embedding-3-large` on the new SIMPLE-PROP documents only; raw messages and
  question vectors are read from the sealed cache, never re-embedded.
- Both routed through `OPENAI_API_KEY` directly (matches how the sealed EVENTS memory and its
  embeddings were built — no OpenRouter, no answer/judge calls of any kind).

## Retrieval and evidence-delivery computation (offline, zero cost)

For each of the 2,400 questions: unified matrix = sealed RAW matrix + new SIMPLE-PROP matrix,
ranked with the same `rank_raw` scoring/ordering as `DenseIndex.search_vector`, top-10.
Exposed source IDs = union of retrieved documents' `source_ids`
(`evermembench_temporal_budget_control.unique_source_ids`). Evidence recall/precision against
`gold_source_ids`, computed exactly as the paper's own evidence metric
(`len(exposed & gold) / len(gold)` and `/ len(exposed)`). No answer generation, no judge — the
disputed claim is about retrieval, and testing it does not require regenerating accuracy.

## Statistics

Paired question bootstrap, 10,000 resamples, seed `20260917` (same seed and method as
`ksweep_delivery.py`, which computed this exact metric at other values of k). Two
comparisons: **PROP − RAW** (does self-contained rewriting alone improve delivery) and
**EVENTS − PROP** (does the event schema add anything on top of self-contained rewriting with
identical provenance). Per-project sensitivity reported for both.

## Budget

Preflight (offline) reports an exact token count from the real constructed prompts before any
call is made. Ceiling: **$6.00** total (extraction + embeddings), well above the ~$4.50–5.30
estimate given to the user before this run; the run stops itself if projected spend would
exceed the ceiling before starting the next project.

## Run

```bash
python3 -m research.simple_prop_baseline --preflight   # offline
python3 -m research.simple_prop_baseline --run          # paid, extraction + embeddings, all 5 projects
python3 -m research.simple_prop_baseline --report
python3 -m research.simple_prop_baseline --freeze
```

## Amendment, 2026-09-21: prompt placeholder fix (v1 -> v2)

The v1 JSON schema example in the extraction prompt used the literal placeholder
`"turn_id":"turn_id"`. On EverMemBench the model copied that literal string in 2,618 of
19,498 proposed items (13.4%), all of which the validator then correctly rejected as
unresolvable ids; on SocialMemBench it did so in nearly every item, leaving 13 surviving
propositions out of 1,481. This handicapped SIMPLE-PROP relative to EVENTS for a reason that
has nothing to do with the representation under test: 46% of all SIMPLE-PROP rejections on
EverMemBench were this bug, and a bug-free index would have been about 19% larger
(13,756 -> ~16,400 units, against 16,859 events).

The placeholder is replaced by `"<copy a real id from [[...]] above>"`, which cannot be copied
verbatim as a valid id. Both benchmarks are re-run from scratch as v2 with the corrected
prompt; v1 stays on disk as the record of the bug and is not used for any claim. The v1
EverMemBench numbers (PROP - RAW +2.00 pp, EVENTS - PROP +1.21 pp) are therefore superseded,
and the direction of the bias is known: the v1 gap favoured the schema.

## Amendment P1, 2026-09-21: does the Temporal Duration advantage need the schema?

B1 showed that removing the derived description from EVENTS costs 2.7 pp accuracy on Temporal
Duration while leaving evidence delivery byte-identical, i.e. the description carries something
beyond the source address. It did not show *which* description: the event schema, or any
self-contained rewriting.

P1 answers that. The SIMPLE-PROP index (v2) replaces the event index under exactly the controls
that produced `RAW+EVENTS-token-matched-chrono`: unified cosine ranking over raw messages plus
propositions, cut to the **same per-question token budget as the frozen EVENTS condition**, then
presented in the same chronological order. Answer model, judge, prompts and scoring are the
official ones, unchanged.

Comparisons: P1 − `RAW+EVENTS-token-matched-chrono` (does the schema beat plain paraphrase on
this slice), P1 − `RAW-token-matched-chrono` (does paraphrase beat raw at all here), and
P1 − B1 (paraphrase versus sources with no description at all).

Reading rule, fixed before the run: if P1 lands at or above the EVENTS condition, the Temporal
Duration advantage is carried by self-containment rather than by the event ontology, and the
temporal result stops being evidence for the schema. If P1 lands near the RAW condition, the
ontology is doing the work on this slice. Temporal Duration remains a post-hoc discovery slice
either way, so neither outcome is confirmatory.

New calls: 300 answers, at most 300 judge calls, ceiling **$2.50** (B1, the same shape, cost
$0.32). Run: `python3 -m research.final_sprint_evermembench --run P1`.
