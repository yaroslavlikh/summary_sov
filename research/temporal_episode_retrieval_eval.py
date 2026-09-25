"""Temporal episode retrieval evaluation -- answer-level test of whether
multi-episode retrieval improves Q8 answers, INCLUDING when a question's
old/new evidence live in two SEPARATE episodes (which the prior review
correctly identified as a real, non-accidental property of the episode
layer -- forcibly merging "party organizing", "minibus", and "share
payment" into one episode would be worse, not better).

`research/temporal_episode_prototype.py` is FROZEN here: this module only
reads its extraction/materialization functions, replaying from the fully
cached event/attach caches built by the prior 20-case run (0 new calls for
memory construction -- verified at runtime, not just assumed). This file
adds a new, separate retrieval + generation + judge layer on top, reusing
the SAME hybrid-retrieval/pack_context/generator machinery already used
throughout this project (research/stage4_9_hybrid_experiment.py,
research/socialmembench_pilot.py) -- no new retrieval algorithm, no new
generator prompt, no threshold changes.

Pipeline per question:
    question
        -> raw-message retrieval (BM25+dense, HybridIndex, full network
           history -- same as every earlier RAW baseline in this project)
        -> live episode retrieval (dense only, same embedding-retrieval
           mechanism the episode layer itself already uses for linking)
        -> evidence expansion (an episode candidate enters context only
           with ALL of its events' validated raw provenance -- pack_context's
           existing atomic-package rule, unchanged)
        -> ONE unified rerank across raw + episode candidates together
        -> answer (cites raw turn_ids only)

Two variants compared: RAW (baseline, unchanged from every earlier stage)
vs RAW+EPISODES. Scope: the same 20 frozen Q8 dev cases -- no held-out, no
production, no full SocialMemBench benchmark, no answer generation beyond
these 20*2 = 40 calls plus judging.

Run:
    python3 -m research.temporal_episode_retrieval_eval
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings import embed
from research.socialmembench_pilot import (
    Candidate, _cited_ids, _json, _normalized, _select_questions, _set_metrics,
    build_raw_candidates, load_data,
)
from research.stage4_5_audit import DATA_DIR, PER_TYPE, SEED, _split_old_new
from research.stage4_8_versioned_experiment import build_session_clusters
from research import stage4_9_hybrid_experiment as s49
from research.temporal_episode_prototype import (
    TemporalEpisode, extract_events_for_network, materialize_episodes,
)

LLM_CACHE = Path("/tmp/temporal_episode_retrieval_llm_cache.jsonl")
RESULTS_JSONL = Path("/tmp/temporal_episode_retrieval_results.jsonl")
REPORT_MD = Path("/tmp/temporal_episode_retrieval_report.md")
EMBED_CACHE_DIR = Path("/tmp/temporal_episode_retrieval_embeddings")

VARIANTS = ("RAW", "RAW+EPISODES")

_llm_cache: dict[str, Any] | None = None


def _load_kv_cache(path: Path) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["key"]] = row["value"]
    return cache


def _append_kv_cache(path: Path, key: str, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def cached_answer(question: str, options: Any, context: str) -> tuple[str, bool]:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = _load_kv_cache(LLM_CACHE)
    key = "answer:" + hashlib.sha256((question + json.dumps(options, sort_keys=True) + context).encode()).hexdigest()
    if key in _llm_cache:
        return _llm_cache[key], False
    from research.socialmembench_pilot import _answer

    result = _answer(question, options, context)
    _llm_cache[key] = result
    _append_kv_cache(LLM_CACHE, key, result)
    return result, True


def cached_judge_correctness(question: str, gold: str, answers: dict[str, str]) -> tuple[dict[str, float], bool]:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = _load_kv_cache(LLM_CACHE)
    key = "judge:" + hashlib.sha256((question + gold + json.dumps(answers, sort_keys=True)).encode()).hexdigest()
    if key in _llm_cache:
        return _llm_cache[key], False
    from research.socialmembench_pilot import _judge_open_answers

    result = _judge_open_answers(question, gold, answers)
    _llm_cache[key] = result
    _append_kv_cache(LLM_CACHE, key, result)
    return result, True


def build_episode_candidates(episodes: list[TemporalEpisode]) -> list[Candidate]:
    """Each episode's FULL provenance (every event, not just the 5 hot
    ones) is attached -- pack_context's existing rule only lets a derived
    candidate into context with ALL of its listed sources valid, so nothing
    partial or cherry-picked ever gets exposed."""
    candidates = []
    for ep in episodes:
        current = ep.current_event
        candidates.append(Candidate(
            candidate_id=f"episode:{ep.episode_id}", kind="derived:episode",
            text=ep.active_representation_text(),
            source_ids=ep.all_source_turn_ids(),
            asserted_by=(ep.viewpoint_owner,) if ep.viewpoint_owner else (),
            entities=(ep.subject,) if ep.subject else (),
            observed_at=current.observed_at if current else "",
        ))
    return candidates


def rebuild_all_episodes(q8_rows: list[dict], conversations) -> tuple[dict[str, list[TemporalEpisode]], dict]:
    """Mirrors run_dev_set's exact grouping (temporal_episode_prototype.py):
    for the 5 networks with 2 Q8 questions each, clusters from BOTH
    questions are combined and extraction runs ONCE per network, then
    materialize_episodes runs ONCE across all networks together -- NOT
    once per question. Replaying per-question independently would hit
    different attach-candidate states and diverge from the cached run
    (and likely from cache entirely, tripping the 0-new-calls check).
    Returns (episodes_by_network, call_counts)."""
    events_by_network: dict[str, list] = defaultdict(list)
    extraction_calls = 0
    for row in q8_rows:
        network_id = row["network_id"]
        anchors = _json(row["evidence_anchors_json"], [])
        clusters = build_session_clusters(anchors)
        events, _rejected, calls, _stats = extract_events_for_network(network_id, clusters, conversations)
        events_by_network[network_id].extend(events)
        extraction_calls += calls
    episodes_by_network, _decisions, attach_calls = materialize_episodes(dict(events_by_network))
    return episodes_by_network, {"extraction": extraction_calls, "attach": attach_calls}


def run_question(
    row: dict, hybrid_index: s49.HybridIndex, raw_by_source: dict[str, Candidate],
    episodes: list[TemporalEpisode],
) -> dict:
    question = row["question"]
    query_vector = _normalized([embed(question)])[0]
    raw_retrieved, episode_retrieved = hybrid_index.retrieve(question, query_vector)

    ranked_raw = s49.HybridIndex.rerank(query_vector, raw_retrieved)
    context_raw, packing_raw = s49.pack_context(ranked_raw, raw_by_source)

    ranked_combined = s49.HybridIndex.rerank(query_vector, raw_retrieved + episode_retrieved)
    context_ep, packing_ep = s49.pack_context(ranked_combined, raw_by_source)

    contexts = {"RAW": context_raw, "RAW+EPISODES": context_ep}
    packing = {"RAW": packing_raw, "RAW+EPISODES": packing_ep}

    new_calls = 0
    answers = {}
    for variant in VARIANTS:
        answer, was_new = cached_answer(question, {}, contexts[variant])
        answers[variant] = answer
        new_calls += int(was_new)
    correctness, judge_new = cached_judge_correctness(question, row["answer"], answers)
    new_calls += int(judge_new)

    cited = {v: _cited_ids(answers[v]) for v in VARIANTS}

    anchors = _json(row["evidence_anchors_json"], [])
    gold_ids = {a["turn_id"] for a in anchors if a.get("turn_id")}
    old_ids, new_ids, _unassigned, degenerate = _split_old_new(anchors)

    citation_metrics = {v: _set_metrics(cited[v], gold_ids) for v in VARIANTS}

    exposed_ep = set(packing["RAW+EPISODES"]["source_ids"])
    both_sides_exposed_via_episodes = bool(old_ids & exposed_ep) and bool(new_ids & exposed_ep)
    exposed_raw = set(packing["RAW"]["source_ids"])
    both_sides_exposed_via_raw = bool(old_ids & exposed_raw) and bool(new_ids & exposed_raw)

    old_episode_ids = {ep.episode_id for ep in episodes if set(ep.all_source_turn_ids()) & old_ids} if old_ids else set()
    new_episode_ids = {ep.episode_id for ep in episodes if set(ep.all_source_turn_ids()) & new_ids} if new_ids else set()
    same_episode = bool(old_episode_ids & new_episode_ids) if (old_episode_ids and new_episode_ids) else False
    split_across_episodes = bool(old_episode_ids and new_episode_ids) and not same_episode

    return {
        "qa_id": row["qa_id"], "network_id": row["network_id"], "question": question,
        "gold_answer": row["answer"], "gold_ids": sorted(gold_ids),
        "degenerate_single_point": degenerate,
        "old_ids": sorted(old_ids), "new_ids": sorted(new_ids),
        "old_episode_ids": sorted(old_episode_ids), "new_episode_ids": sorted(new_episode_ids),
        "same_episode": same_episode, "split_across_episodes": split_across_episodes,
        "answers": answers, "correctness": correctness,
        "cited_ids": {v: sorted(cited[v]) for v in VARIANTS},
        "citation_metrics": citation_metrics,
        "context_words": {v: packing[v]["context_words"] for v in VARIANTS},
        "episode_candidates_packed": packing["RAW+EPISODES"]["memory_candidate_ids"],
        "both_sides_exposed_via_episodes": both_sides_exposed_via_episodes,
        "both_sides_exposed_via_raw": both_sides_exposed_via_raw,
        "new_llm_calls": new_calls,
    }


def render_report(results: list[dict]) -> str:
    n = len(results)
    lines = [
        "# Temporal episode retrieval evaluation -- RAW vs RAW+EPISODES", "",
        f"**n = {n}** (the 20 frozen Q8 dev cases). This is a small sample -- reported as counts, not treated as "
        "statistically established. episode memory layer is FROZEN (research/temporal_episode_prototype.py "
        "unmodified); only retrieval/rerank/generation/judging are new here, all reused verbatim from existing "
        "project machinery (HybridIndex, pack_context, _answer, _judge_open_answers).", "",
        "## Aggregate", "",
        "| variant | mean correctness | mean citation precision | mean citation recall |",
        "|---|---|---|---|",
    ]
    for v in VARIANTS:
        mean_correct = sum(r["correctness"][v] for r in results) / n
        mean_prec = sum(r["citation_metrics"][v]["precision"] for r in results) / n
        mean_rec = sum(r["citation_metrics"][v]["recall"] for r in results) / n
        lines.append(f"| {v} | {mean_correct:.3f} | {mean_prec:.3f} | {mean_rec:.3f} |")

    wins = sum(1 for r in results if r["correctness"]["RAW+EPISODES"] > r["correctness"]["RAW"])
    losses = sum(1 for r in results if r["correctness"]["RAW+EPISODES"] < r["correctness"]["RAW"])
    ties = n - wins - losses
    lines += ["", f"## Paired correctness (n={n})", "", f"- RAW+EPISODES beats RAW: {wins}", f"- RAW beats RAW+EPISODES: {losses}", f"- ties: {ties}"]

    split_cases = [r for r in results if r["split_across_episodes"]]
    both_exposed_when_split = sum(1 for r in split_cases if r["both_sides_exposed_via_episodes"])
    lines += [
        "", "## Split-across-episodes cases -- the key structural question", "",
        f"- questions where old/new evidence live in DIFFERENT episodes: {len(split_cases)}/{n}",
        f"- of those, RAW+EPISODES exposed BOTH sides in context anyway: {both_exposed_when_split}/{len(split_cases)}"
        if split_cases else "- none (no case had old/new split across different episodes)",
    ]
    if split_cases:
        for r in split_cases:
            lines.append(
                f"  - {r['qa_id']}: old_episodes={r['old_episode_ids']} new_episodes={r['new_episode_ids']} "
                f"both_exposed={r['both_sides_exposed_via_episodes']} "
                f"correctness RAW={r['correctness']['RAW']:.1f} RAW+EP={r['correctness']['RAW+EPISODES']:.1f}"
            )

    lines += ["", "## Per-question detail", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"### {i}. {r['qa_id']} :: {r['network_id']}")
        lines.append(f"> {r['question']}")
        lines.append(f"- gold_ids: {r['gold_ids']}  old_ids: {r['old_ids']}  new_ids: {r['new_ids']}  degenerate: {r['degenerate_single_point']}")
        lines.append(f"- old_episode_ids: {r['old_episode_ids']}  new_episode_ids: {r['new_episode_ids']}  "
                     f"same_episode: {r['same_episode']}  split_across_episodes: {r['split_across_episodes']}")
        for v in VARIANTS:
            lines.append(
                f"- {v}: correctness={r['correctness'][v]:.1f} precision={r['citation_metrics'][v]['precision']:.2f} "
                f"recall={r['citation_metrics'][v]['recall']:.2f} context_words={r['context_words'][v]} cited={r['cited_ids'][v]}"
            )
        lines.append(f"- episode candidates packed into RAW+EPISODES context: {r['episode_candidates_packed']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    conversations, qa, _personas = load_data(DATA_DIR)
    selected = _select_questions(qa, PER_TYPE, SEED)
    q8_rows = [row for row in selected if row["query_type"] == "Q8"]
    assert len(q8_rows) == 20, f"expected 20 Q8 dev cases, got {len(q8_rows)}"

    s49.EMBED_CACHE_DIR = EMBED_CACHE_DIR  # isolated -- never write into stage4_9's embedding cache

    print("Rebuilding episodes from fully-cached event/attach data (expect 0 new calls)...", file=sys.stderr)
    episodes_by_network, memory_new_calls = rebuild_all_episodes(q8_rows, conversations)
    if memory_new_calls["extraction"] or memory_new_calls["attach"]:
        raise RuntimeError(
            f"Expected the episode memory layer to be fully cached (frozen) -- got new calls {memory_new_calls}. "
            "Refusing to proceed silently; investigate before re-running."
        )
    print(f"  episodes rebuilt across {len(episodes_by_network)} networks, 0 new memory-layer calls confirmed", file=sys.stderr)

    raw_by_network, raw_by_source = build_raw_candidates(conversations)

    print(f"Running {len(q8_rows)} Q8 questions x {len(VARIANTS)} variants...", file=sys.stderr)
    results = []
    total_new_calls = 0
    for i, row in enumerate(q8_rows, 1):
        network_id = row["network_id"]
        raw = raw_by_network.get(network_id, [])
        episodes = episodes_by_network.get(network_id, [])
        episode_candidates = build_episode_candidates(episodes)
        hybrid_index = s49.HybridIndex(raw, episode_candidates, network_id, "episodes")
        result = run_question(row, hybrid_index, raw_by_source, episodes)
        total_new_calls += result["new_llm_calls"]
        results.append(result)
        print(f"  {i}/{len(q8_rows)} {row['qa_id']} (new_calls this q: {result['new_llm_calls']})", file=sys.stderr)

    with RESULTS_JSONL.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    REPORT_MD.write_text(render_report(results))

    print(f"Total new LLM calls (answer+judge): {total_new_calls}", file=sys.stderr)
    print(f"Results: {RESULTS_JSONL}", file=sys.stderr)
    print(f"Report: {REPORT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
