"""Stage 4.9: frozen hybrid retrieval ablation on the 20 Q8 dev cases.

Compares RAW, RAW+FLAT, and RAW+VERSIONED under one context-word budget.
The derived-memory corpora are rebuilt only from frozen Stage 4.7.1/4.8
artifacts.  No extraction or resolver calls are made.  Retrieved memory is
expanded to validated raw turns, and only raw turn IDs are citable.

This remains a dev/oracle-memory component experiment: the memory objects were
originally extracted from gold-anchor regions.  Raw retrieval itself runs over
the full network history.  Held-out data and production code are untouched.

Run:
    python3 -m research.stage4_9_hybrid_experiment
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings import embed, embed_batch
from research import stage4_8_versioned_experiment as s48
from research.socialmembench_pilot import (
    Candidate, NetworkIndex, _bm25_prepare, _bm25_rank, _cached_embeddings,
    _cited_ids, _dense_rank, _json, _normalized, _rrf_ids, _select_questions,
    _set_metrics, build_raw_candidates, load_data,
)
from research.stage4_5_audit import DATA_DIR, PER_TYPE, SEED
from research.stage4_7_1_context_extraction import CONTEXT_EXTRACTION_CACHE, ContextRecordV2

STAGE48_RESULTS = Path("/tmp/socialmembench_stage4_8_results.jsonl")
STAGE49_LLM_CACHE = Path("/tmp/socialmembench_stage4_9_llm_cache.jsonl")
RESULTS_JSONL = Path("/tmp/socialmembench_stage4_9_results.jsonl")
REPORT_MD = Path("/tmp/socialmembench_stage4_9_report.md")
EMBED_CACHE_DIR = Path("/tmp/socialmembench_stage4_9_embeddings")

VARIANTS = ("RAW", "RAW+FLAT", "RAW+VERSIONED")
RAW_RETRIEVAL_LIMIT = 12
MEMORY_RETRIEVAL_LIMIT = 12
# Fixed before answer generation.  At 400 words the cap is active for both
# hybrid arms without reducing the 12-result RAW baseline to a tiny window.
CONTEXT_WORD_BUDGET = 400


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"required frozen artifact is missing: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_extraction_cache() -> dict[str, dict]:
    return {row["cache_key"]: row for row in _load_jsonl(CONTEXT_EXTRACTION_CACHE)}


def rebuild_registry(
    records: list[ContextRecordV2], network_id: str, decisions: list[dict],
) -> dict[str, s48.Stage48Cell]:
    """Replay frozen Stage 4.8 decisions; never call the semantic resolver."""
    registry: dict[str, s48.Stage48Cell] = {}
    state_like = sorted(
        (record for record in records if record.record_type in ("STATE", "TRANSITION")),
        key=lambda record: record.observed_at,
    )
    if len(state_like) != len(decisions):
        raise ValueError(
            f"frozen decision mismatch for {network_id}: "
            f"{len(state_like)} records != {len(decisions)} decisions"
        )
    for record, decision in zip(state_like, decisions):
        if decision.get("claim") != record.claim:
            raise ValueError(f"frozen decision order mismatch for {network_id}: {record.claim!r}")
        s48.apply_decision(registry, record, decision, network_id)

    for record in records:
        if record.record_type in ("CAUSE", "REACTION"):
            for cell in registry.values():
                if cell.subject == s48._norm(record.subject):
                    cell.observations.append(
                        s48.Stage48Observation(kind=record.record_type.lower(), record=record)
                    )
    return registry


def _derived_candidates(
    case: dict, records: list[ContextRecordV2], registry: dict[str, s48.Stage48Cell],
    raw_by_source: dict[str, Candidate],
) -> tuple[list[Candidate], list[Candidate]]:
    def observed_at(source_ids: tuple[str, ...]) -> str:
        return max((str(raw_by_source[source_id].observed_at) for source_id in source_ids), default="")

    flat = [
        Candidate(
            candidate_id=f"flat:{case['qa_id']}:{index}",
            kind=f"derived:{record.record_type.lower()}",
            text=record.claim,
            source_ids=record.source_turn_ids,
            asserted_by=(record.viewpoint_owner,),
            entities=(record.subject,),
            observed_at=observed_at(record.source_turn_ids),
        )
        for index, record in enumerate(records)
    ]
    versioned = [
        Candidate(
            candidate_id=f"cell:{case['qa_id']}:{cell.cell_id}",
            kind="derived:versioned_cell",
            text=s48._cell_text(cell),
            source_ids=cell.source_turn_ids(),
            asserted_by=(cell.viewpoint_owner,),
            entities=(cell.subject,),
            observed_at=observed_at(cell.source_turn_ids()),
        )
        for cell in registry.values()
    ]
    return flat, versioned


def _dedupe_exact(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[tuple] = set()
    result = []
    for candidate in candidates:
        key = (candidate.kind, candidate.text, candidate.source_ids)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


class HybridIndex:
    """Hybrid first-stage retrieval plus one shared dense reranker."""

    def __init__(
        self, raw: list[Candidate], memory: list[Candidate], network_id: str, label: str,
    ) -> None:
        self.raw = raw
        self.memory = memory
        self.raw_bm25 = _bm25_prepare(raw)
        self.raw_vectors = _cached_embeddings(raw, EMBED_CACHE_DIR, f"{network_id}-raw")
        self.memory_vectors = (
            _cached_embeddings(memory, EMBED_CACHE_DIR, f"{network_id}-{label}")
            if memory else np.empty((0, self.raw_vectors.shape[1]), dtype=np.float32)
        )

    @staticmethod
    def _copy(candidate: Candidate, score: float, channels: set[str]) -> Candidate:
        return NetworkIndex._copy(candidate, score, channels)

    def retrieve(self, question: str, query_vector: np.ndarray) -> tuple[list[Candidate], list[Candidate]]:
        raw_bm25 = [self.raw[i].candidate_id for i in _bm25_rank(self.raw_bm25, question, 10)]
        raw_dense = [self.raw[i].candidate_id for i in _dense_rank(self.raw_vectors, query_vector, 10)]
        raw_by_id = {candidate.candidate_id: candidate for candidate in self.raw}
        raw = [
            self._copy(raw_by_id[candidate_id], score, channels)
            for candidate_id, score, channels in _rrf_ids([raw_bm25, raw_dense], RAW_RETRIEVAL_LIMIT)
        ]

        memory_ids = [
            self.memory[i].candidate_id
            for i in _dense_rank(self.memory_vectors, query_vector, MEMORY_RETRIEVAL_LIMIT)
        ]
        memory_by_id = {candidate.candidate_id: candidate for candidate in self.memory}
        memory = [self._copy(memory_by_id[candidate_id], 0.0, {"memory_dense"}) for candidate_id in memory_ids]
        return raw, memory

    @staticmethod
    def rerank(question_vector: np.ndarray, candidates: list[Candidate]) -> list[Candidate]:
        """One local semantic ranking criterion for raw and memory alike."""
        if not candidates:
            return []
        matrix = _normalized(embed_batch([candidate.text for candidate in candidates]))
        scores = np.sum(matrix * question_vector, axis=1)
        return [candidates[index] for index in np.argsort(-scores)]


def _word_count(text: str) -> int:
    return len(text.split())


def pack_context(
    ranked: list[Candidate], raw_by_source: dict[str, Candidate],
    budget: int = CONTEXT_WORD_BUDGET,
) -> tuple[str, dict[str, Any]]:
    """Pack atomic evidence units; a derived unit is kept only with all sources."""
    lines: list[str] = []
    words = 0
    rendered_sources: set[str] = set()
    rendered_candidates: list[str] = []
    rendered_memory: list[str] = []

    for candidate in ranked:
        if candidate.kind == "raw":
            source_id = candidate.source_ids[0]
            if source_id in rendered_sources:
                continue
            unit_lines = [f"[[{source_id}]] {candidate.observed_at} {candidate.text}"]
            new_sources = {source_id}
        else:
            valid_sources = [source_id for source_id in candidate.source_ids if source_id in raw_by_source]
            if len(valid_sources) != len(candidate.source_ids) or not valid_sources:
                continue
            unit_lines = [
                f"[DERIVED {candidate.candidate_id}] type={candidate.kind}; "
                f"asserted_by={list(candidate.asserted_by)}; entities={list(candidate.entities)}\n{candidate.text}"
            ]
            new_sources = set(valid_sources) - rendered_sources
            unit_lines.extend(
                f"  SOURCE [[{source_id}]] {raw_by_source[source_id].observed_at} "
                f"{raw_by_source[source_id].text}"
                for source_id in valid_sources if source_id in new_sources
            )

        unit = "\n".join(unit_lines)
        unit_words = _word_count(unit)
        if words + unit_words > budget:
            continue
        lines.append(unit)
        words += unit_words
        rendered_sources.update(new_sources)
        rendered_candidates.append(candidate.candidate_id)
        if candidate.kind != "raw":
            rendered_memory.append(candidate.candidate_id)

    return "\n".join(lines), {
        "context_words": words,
        "source_ids": sorted(rendered_sources),
        "candidate_ids": rendered_candidates,
        "memory_candidate_ids": rendered_memory,
    }


def _cases(qa) -> list[dict]:
    selected = _select_questions(qa, PER_TYPE, SEED)
    rows = [row for row in selected if row["query_type"] == "Q8"]
    if len(rows) != 20:
        raise ValueError(f"expected frozen 20 Q8 cases, got {len(rows)}")
    return [
        {
            "qa_id": row["qa_id"], "network_id": row["network_id"],
            "question": row["question"], "gold_answer": row["answer"],
            "gold_ids": {a["turn_id"] for a in _json(row["evidence_anchors_json"], []) if a.get("turn_id")},
            "clusters": s48.build_session_clusters(_json(row["evidence_anchors_json"], [])),
        }
        for row in rows
    ]


def build_frozen_memory_corpora(
    cases: list[dict], conversations, raw_by_source: dict[str, Candidate],
) -> tuple[dict[str, list[Candidate]], dict[str, list[Candidate]]]:
    stage48_results = {row["qa_id"]: row for row in _load_jsonl(STAGE48_RESULTS)}
    extraction_cache = _load_extraction_cache()
    flat_by_network: dict[str, list[Candidate]] = defaultdict(list)
    versioned_by_network: dict[str, list[Candidate]] = defaultdict(list)

    for case in cases:
        frozen = stage48_results.get(case["qa_id"])
        if frozen is None:
            raise ValueError(f"Stage 4.8 result missing for {case['qa_id']}")
        records, _window_turns, failures = s48.extract_records_for_case(case, conversations, extraction_cache)
        if len(failures) != len(frozen.get("extraction_failures", [])):
            raise ValueError(f"Stage 4.7.1 cache drift for {case['qa_id']}")
        registry = rebuild_registry(records, case["network_id"], frozen["resolver_decisions"])
        flat, versioned = _derived_candidates(case, records, registry, raw_by_source)
        flat_by_network[case["network_id"]].extend(flat)
        versioned_by_network[case["network_id"]].extend(versioned)

    return (
        {network: _dedupe_exact(items) for network, items in flat_by_network.items()},
        {network: _dedupe_exact(items) for network, items in versioned_by_network.items()},
    )


def run_question(
    case: dict, raw_index: HybridIndex, flat_index: HybridIndex,
    versioned_index: HybridIndex, raw_by_source: dict[str, Candidate],
) -> dict:
    question = case["question"]
    query_vector = _normalized([embed(question)])[0]
    raw_retrieved, _ = raw_index.retrieve(question, query_vector)
    _, flat_retrieved = flat_index.retrieve(question, query_vector)
    _, versioned_retrieved = versioned_index.retrieve(question, query_vector)

    retrieved = {
        "RAW": raw_retrieved,
        "RAW+FLAT": raw_retrieved + flat_retrieved,
        "RAW+VERSIONED": raw_retrieved + versioned_retrieved,
    }
    contexts: dict[str, str] = {}
    packing: dict[str, dict] = {}
    for variant in VARIANTS:
        ranked = HybridIndex.rerank(query_vector, retrieved[variant])
        contexts[variant], packing[variant] = pack_context(ranked, raw_by_source)

    answers = {variant: s48.cached_answer(question, {}, contexts[variant]) for variant in VARIANTS}
    correctness = s48.cached_judge_correctness(question, case["gold_answer"], answers)
    temporal_attribution = s48.cached_judge_temporal_attribution(question, case["gold_answer"], answers)
    cited = {variant: _cited_ids(answers[variant]) for variant in VARIANTS}
    citation_metrics = {variant: _set_metrics(cited[variant], case["gold_ids"]) for variant in VARIANTS}

    raw_sources = set(packing["RAW"]["source_ids"])
    diagnostics = {}
    for variant, memory_retrieved in (
        ("RAW+FLAT", flat_retrieved), ("RAW+VERSIONED", versioned_retrieved),
    ):
        valid_memory = [
            candidate for candidate in memory_retrieved
            if candidate.source_ids and all(source_id in raw_by_source for source_id in candidate.source_ids)
        ]
        exposed = set(packing[variant]["source_ids"])
        diagnostics[variant] = {
            "memory_retrieved": len(memory_retrieved),
            "memory_with_valid_provenance": len(valid_memory),
            "memory_packed": len(packing[variant]["memory_candidate_ids"]),
            "memory_source_ids_exposed": len({sid for c in valid_memory for sid in c.source_ids} & exposed),
            "gold_anchors_only_via_memory": sorted((exposed - raw_sources) & case["gold_ids"]),
            "raw_evidence_displaced": sorted(raw_sources - exposed),
        }

    return {
        "qa_id": case["qa_id"], "network_id": case["network_id"],
        "question": case["question"], "gold_answer": case["gold_answer"],
        "gold_ids": sorted(case["gold_ids"]),
        "n_clusters": len(case["clusters"]),
        "answers": answers,
        "correctness": correctness,
        "temporal_attribution": temporal_attribution,
        "cited_ids": {variant: sorted(ids) for variant, ids in cited.items()},
        "citation_metrics": citation_metrics,
        "packing": packing,
        "diagnostics": diagnostics,
    }


def _averages(results: list[dict], key) -> dict[str, float]:
    return {variant: sum(key(row, variant) for row in results) / len(results) for variant in VARIANTS}


def _paired(results: list[dict], left: str, right: str) -> tuple[int, int, int]:
    wins = sum(row["correctness"][left] > row["correctness"][right] for row in results)
    losses = sum(row["correctness"][left] < row["correctness"][right] for row in results)
    return wins, len(results) - wins - losses, losses


def render_report(results: list[dict]) -> str:
    correctness = _averages(results, lambda row, variant: row["correctness"][variant])
    temporal = _averages(results, lambda row, variant: row["temporal_attribution"][variant]["temporal_correctness"])
    attribution = _averages(results, lambda row, variant: row["temporal_attribution"][variant]["attribution_correctness"])
    precision = _averages(results, lambda row, variant: row["citation_metrics"][variant]["precision"])
    recall = _averages(results, lambda row, variant: row["citation_metrics"][variant]["recall"])
    lines = [
        "# Stage 4.9 -- hybrid retrieval with provenance expansion",
        "",
        "**Frozen dev/oracle-memory component experiment.** Memory objects came from gold-anchor regions in "
        "Stages 4.7.1/4.8, while RAW retrieval searches the full network history. This is not an official "
        "SocialMemBench score and not an end-to-end ingestion result. Extractor, resolver, thresholds, held-out, "
        "production, Stage 5, and Q6 were untouched.",
        "",
        f"- questions: {len(results)}",
        f"- context budget: {CONTEXT_WORD_BUDGET} words for every variant",
        "- first stage: raw BM25+dense; memory dense",
        "- shared reranker: local dense similarity over raw and derived candidate text",
        "- derived candidates are atomic packages and enter context only with every validated raw source",
        "",
        "## Aggregate metrics",
        "",
        "| metric | RAW | RAW+FLAT | RAW+VERSIONED |",
        "|---|---:|---:|---:|",
        f"| answer correctness | {correctness['RAW']:.3f} | {correctness['RAW+FLAT']:.3f} | {correctness['RAW+VERSIONED']:.3f} |",
        f"| temporal correctness | {temporal['RAW']:.3f} | {temporal['RAW+FLAT']:.3f} | {temporal['RAW+VERSIONED']:.3f} |",
        f"| attribution correctness | {attribution['RAW']:.3f} | {attribution['RAW+FLAT']:.3f} | {attribution['RAW+VERSIONED']:.3f} |",
        f"| citation precision | {precision['RAW']:.3f} | {precision['RAW+FLAT']:.3f} | {precision['RAW+VERSIONED']:.3f} |",
        f"| citation recall | {recall['RAW']:.3f} | {recall['RAW+FLAT']:.3f} | {recall['RAW+VERSIONED']:.3f} |",
        "",
        "## Paired correctness",
        "",
    ]
    for left, right in (("RAW+VERSIONED", "RAW"), ("RAW+VERSIONED", "RAW+FLAT")):
        wins, ties, losses = _paired(results, left, right)
        lines.append(f"- {left} vs {right}: {wins} wins / {ties} ties / {losses} losses")

    lines += ["", "## Retrieval and provenance diagnostics", ""]
    for variant in ("RAW+FLAT", "RAW+VERSIONED"):
        totals = Counter()
        for row in results:
            diag = row["diagnostics"][variant]
            for key in ("memory_retrieved", "memory_with_valid_provenance", "memory_packed", "memory_source_ids_exposed"):
                totals[key] += diag[key]
            totals["gold_anchors_only_via_memory"] += len(diag["gold_anchors_only_via_memory"])
            totals["raw_evidence_displaced"] += len(diag["raw_evidence_displaced"])
        lines.append(
            f"- {variant}: retrieved_memory={totals['memory_retrieved']}; "
            f"valid_provenance={totals['memory_with_valid_provenance']}; packed_memory={totals['memory_packed']}; "
            f"exposed_memory_sources={totals['memory_source_ids_exposed']}; "
            f"gold_anchors_only_via_memory={totals['gold_anchors_only_via_memory']}; "
            f"raw_sources_displaced={totals['raw_evidence_displaced']}"
        )

    delta = lambda row: row["correctness"]["RAW+VERSIONED"] - row["correctness"]["RAW"]
    improvements = sorted((row for row in results if delta(row) > 0), key=delta, reverse=True)[:5]
    regressions = sorted((row for row in results if delta(row) < 0), key=delta)[:5]
    for title, rows in (("Best improvements", improvements), ("Worst regressions", regressions)):
        lines += ["", f"## {title}: RAW+VERSIONED vs RAW", ""]
        if not rows:
            lines.append("- none")
        for row in rows:
            lines.append(f"### {row['qa_id']} ({delta(row):+.1f})")
            lines.append(f"> {row['question']}")
            lines.append(f"- RAW: {row['answers']['RAW']}")
            lines.append(f"- RAW+VERSIONED: {row['answers']['RAW+VERSIONED']}")
            lines.append(f"- diagnostics: {row['diagnostics']['RAW+VERSIONED']}")

    lines += ["", "## Per-question results", ""]
    for index, row in enumerate(results, 1):
        lines.append(f"### {index}. {row['qa_id']} :: {row['network_id']}")
        lines.append(f"> {row['question']}")
        lines.append(f"- gold: {row['gold_answer']}")
        for variant in VARIANTS:
            lines.append(
                f"- {variant}: correctness={row['correctness'][variant]:.1f}; "
                f"temporal={row['temporal_attribution'][variant]['temporal_correctness']:.0f}; "
                f"attribution={row['temporal_attribution'][variant]['attribution_correctness']:.0f}; "
                f"citation_precision={row['citation_metrics'][variant]['precision']:.2f}; "
                f"citation_recall={row['citation_metrics'][variant]['recall']:.2f}; "
                f"context_words={row['packing'][variant]['context_words']}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    conversations, qa, _personas = load_data(DATA_DIR)
    cases = _cases(qa)
    raw_by_network, raw_by_source = build_raw_candidates(conversations)
    flat_by_network, versioned_by_network = build_frozen_memory_corpora(cases, conversations, raw_by_source)

    # Reuse Stage 4.8's answer/judge wrappers with a separate immutable cache.
    s48.LLM_CALL_CACHE = STAGE49_LLM_CACHE
    s48._llm_cache = None

    indices = {}
    for network_id in {case["network_id"] for case in cases}:
        raw = raw_by_network[network_id]
        flat = flat_by_network.get(network_id, [])
        versioned = versioned_by_network.get(network_id, [])
        indices[network_id] = (
            HybridIndex(raw, [], network_id, "none"),
            HybridIndex(raw, flat, network_id, "flat"),
            HybridIndex(raw, versioned, network_id, "versioned"),
        )

    print(f"Running {len(cases)} frozen Q8 questions x 3 variants...", file=sys.stderr)
    results = []
    for index, case in enumerate(cases, 1):
        results.append(run_question(case, *indices[case["network_id"]], raw_by_source))
        print(f"  {index}/{len(cases)} {case['qa_id']}", file=sys.stderr)

    with RESULTS_JSONL.open("w") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    REPORT_MD.write_text(render_report(results))
    print(f"Results: {RESULTS_JSONL}", file=sys.stderr)
    print(f"Report: {REPORT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
