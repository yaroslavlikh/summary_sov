"""Small end-to-end smoke test for semantic-key memory linking.

Reuses the frozen 20-case Stage 4.9 Q8 component benchmark and adds one arm:
RAW+SEMANTIC_VERSIONED.  The new arm materializes the same extracted records
with a value-free natural-language key, embedding top-3 retrieval, a
precision-oriented LLM gate, and only then an update operation.  Embedding
similarity only proposes candidates; it can never override the gate.

This is deliberately not an official SocialMemBench score.  Its extraction
windows were selected from gold anchors, so it answers the narrower question:
does the new linker work when the relevant assertions have already survived
extraction?

Run:
    python3 -m research.semantic_key_end_to_end_smoke
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embeddings import embed_batch
from research import semantic_key_gate as skg
from research import semantic_key_gate_primary as primary_gate
from research import semantic_key_grounded_gate as grounded_gate
from research import semantic_slot_linking_v2 as v2
from research import stage4_8_versioned_experiment as s48
from research import stage4_9_hybrid_experiment as s49
from research.semantic_slot_pairwise_resolver import _operation_prompt, normalize_operation
from research.socialmembench_pilot import Candidate, _cited_ids, _normalized, _set_metrics, build_raw_candidates, load_data
from research.stage4_5_audit import DATA_DIR
from research.stage4_7_1_context_extraction import ContextRecordV2


OUT_DIR = Path(".research_runs/semantic_key_end_to_end_smoke_grounded_v1")
SEED_CACHE_DIR = Path(".research_runs/semantic_key_end_to_end_smoke_strict_v1")
NORMALIZATION_CACHE = OUT_DIR / "normalization_cache.jsonl"
GATE_CACHE = OUT_DIR / "gate_cache.jsonl"
OPERATION_CACHE = OUT_DIR / "operation_cache.jsonl"
LLM_CACHE = OUT_DIR / "answer_judge_cache.jsonl"
MATERIALIZATION = OUT_DIR / "materialization.jsonl"
RESULTS = OUT_DIR / "results.jsonl"
REPORT = OUT_DIR / "report.md"
CONFIG = OUT_DIR / "config.json"

SCHEMA_VERSION = "semantic_key_e2e_smoke_grounded_v1"
CACHE_SCHEMA_VERSION = "semantic_key_e2e_smoke_v1"
VARIANTS = ("RAW", "RAW+FLAT", "RAW+VERSIONED", "RAW+SEMANTIC_VERSIONED")
TOP_K = 3
GATE_THRESHOLD = 0.000001


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _load_cache(path: Path) -> dict[str, dict]:
    return {row["key"]: row["value"] for row in _read_jsonl(path)}


def _append_cache(path: Path, key: str, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def _call_cached(
    path: Path, cache: dict[str, dict], purpose: str, prompt: str, model: str = "fast",
) -> tuple[dict, bool]:
    key = hashlib.sha256(f"{CACHE_SCHEMA_VERSION}:{purpose}:{model}:{prompt}".encode()).hexdigest()
    if key in cache:
        return cache[key], False
    value = primary_gate._primary_only(prompt) if model == "primary" else v2._call_llm(prompt, model)
    cache[key] = value
    _append_cache(path, key, value)
    return value, True


def _normalization_prompt(records: list[ContextRecordV2]) -> str:
    rows = "\n".join(
        f"[{index}] claim={record.claim!r}; viewpoint_owner={record.viewpoint_owner!r}; "
        f"subject={record.subject!r}; state_description={record.state_description!r}; value={record.value!r}; "
        f"from_value={record.from_value!r}; to_value={record.to_value!r}"
        for index, record in enumerate(records)
    )
    return f"""Create a stable, value-free semantic memory key for every numbered record.
The key says WHOSE state/view it is, WHAT it concerns, and WHICH changeable
dimension it tracks.  It must not contain the current value, because later
opposite values must still retrieve the same key.

For an opinion such as 'Tigmen says Gordey is bad': state_holder=Tigmen,
viewpoint_owner=Tigmen, topic_object=Gordey, state_dimension=opinion.
For a self-state such as 'Seb's knee is locked': state_holder=Seb,
viewpoint_owner=null, topic_object=Seb's knee, state_dimension=condition.

Records:
{rows}

Return JSON only, exactly one item per index:
{{"items":[{{"record_index":0,"state_holder":"...","viewpoint_owner":null,
"topic_object":"...","state_dimension":"...","slot_question":"a value-free question"}}]}}"""


def _fallback_slot(record: ContextRecordV2, qa_id: str, index: int) -> v2.NormalizedSlot:
    holder = record.viewpoint_owner or record.subject or "unknown"
    topic = record.subject or holder
    dimension = record.state_description or "state"
    question = (
        f"What is {holder}'s {dimension}?" if v2._norm(holder) == v2._norm(topic)
        else f"What is {holder}'s {dimension} concerning {topic}?"
    )
    return _make_slot(record, qa_id, index, holder, record.viewpoint_owner or None, topic, dimension, question)


def _make_slot(
    record: ContextRecordV2, qa_id: str, index: int, holder: str,
    owner: str | None, topic: str, dimension: str, question: str,
) -> v2.NormalizedSlot:
    value = record.to_value or record.value or record.from_value or record.claim
    return v2.NormalizedSlot(
        case_qa_id=f"{qa_id}:{index}", side="stream", network_id="",
        state_holder=holder, viewpoint_owner=owner, topic_object=topic,
        state_dimension=dimension, slot_question=question, value=value,
        assertion_type=record.record_type, normalization_confidence=record.confidence,
        source_turn_ids=record.source_turn_ids, state_text=record.claim,
    )


def validate_key_items(output: dict, records: list[ContextRecordV2], qa_id: str) -> tuple[dict[int, v2.NormalizedSlot], Counter]:
    slots: dict[int, v2.NormalizedSlot] = {}
    rejected: Counter = Counter()
    for item in output.get("items", []):
        try:
            index = int(item.get("record_index"))
        except (TypeError, ValueError):
            rejected["invalid_index"] += 1
            continue
        if not 0 <= index < len(records) or index in slots:
            rejected["invalid_index"] += 1
            continue
        record = records[index]
        holder = str(item.get("state_holder") or "").strip()
        topic = str(item.get("topic_object") or "").strip()
        dimension = str(item.get("state_dimension") or "").strip()
        question = str(item.get("slot_question") or "").strip()
        raw_owner = item.get("viewpoint_owner")
        owner = None if raw_owner is None or str(raw_owner).strip().lower() in {"", "null", "none"} else str(raw_owner).strip()
        value = record.to_value or record.value or record.from_value or record.claim
        if not holder or not topic or not dimension or not question:
            rejected["missing_key_field"] += 1
            continue
        if v2._value_leaked(question, value):
            rejected["value_leak"] += 1
            continue
        slots[index] = _make_slot(record, qa_id, index, holder, owner, topic, dimension, question)
    return slots, rejected


@dataclass
class SemanticCell:
    cell: s48.Stage48Cell
    key_slot: v2.NormalizedSlot


def _current_slot(entry: SemanticCell) -> v2.NormalizedSlot:
    active = entry.cell.active_version
    record = active.record if active else entry.cell.versions[-1].record
    base = entry.key_slot
    return _make_slot(
        record, base.case_qa_id, 0, base.state_holder, base.viewpoint_owner,
        base.topic_object, base.state_dimension, base.slot_question,
    )


def rank_cells(query: v2.NormalizedSlot, cells: list[SemanticCell]) -> list[dict]:
    if not cells:
        return []
    query_text = skg.key_description(query)
    texts = [query_text] + [skg.key_description(cell.key_slot) for cell in cells]
    vectors = _normalized(embed_batch(texts))
    scored = sorted(
        ((float(vectors[0] @ vectors[index + 1]), entry) for index, entry in enumerate(cells)),
        key=lambda pair: pair[0], reverse=True,
    )[:TOP_K]
    return [
        {
            "candidate_id": entry.cell.cell_id, "retrieval_score": score,
            "key_description": skg.key_description(entry.key_slot), "entry": entry,
        }
        for score, entry in scored
    ]


def select_link(candidates: list[dict], gate_output: dict) -> tuple[str | None, str]:
    decisions = skg.normalize_gate_output(gate_output, len(candidates))
    enriched = [
        {**{k: v for k, v in candidate.items() if k != "entry"},
         **decisions.get(index, {"verdict": "MISSING", "confidence": 0.0, "match_score": 0.0})}
        for index, candidate in enumerate(candidates)
    ]
    strict = skg.select_match({"candidates": enriched}, GATE_THRESHOLD)
    if strict is not None:
        return strict, "llm_gate"
    return None, "new_cell"


def materialize_case(
    case: dict, records: list[ContextRecordV2], caches: dict[str, dict], turns: dict[str, dict],
) -> tuple[dict[str, s48.Stage48Cell], list[dict], Counter, int]:
    state_records = sorted(
        (record for record in records if record.record_type in ("STATE", "TRANSITION")),
        key=lambda record: record.observed_at,
    )
    output, called = _call_cached(
        NORMALIZATION_CACHE, caches["normalization"], "normalize",
        _normalization_prompt(state_records),
    ) if state_records else ({"items": []}, False)
    normalized, rejected = validate_key_items(output, state_records, case["qa_id"])
    stats = Counter(rejected)
    stats["normalization_calls"] += int(called)
    entries: dict[str, SemanticCell] = {}
    decisions = []
    new_calls = int(called)

    for index, record in enumerate(state_records):
        slot = normalized.get(index)
        if slot is None:
            slot = _fallback_slot(record, case["qa_id"], index)
            stats["normalization_fallback"] += 1
        slot = v2.NormalizedSlot(**{**slot.__dict__, "network_id": case["network_id"]})
        ranked = rank_cells(slot, list(entries.values()))
        selected = None
        link_source = "new_cell"
        gate_called = False
        if ranked:
            prompt = grounded_gate.grounded_prompt(
                slot,
                [
                    ({"key_description": row["key_description"]}, _current_slot(row["entry"]))
                    for row in ranked
                ],
                turns,
            )
            gate_output, gate_called = _call_cached(GATE_CACHE, caches["gate"], "gate", prompt, "primary")
            selected, link_source = select_link(ranked, gate_output)
            new_calls += int(gate_called)
            stats["gate_calls"] += int(gate_called)

        if selected is None:
            decision = {"decision": "NEW_CELL", "target_cell_id": None}
        else:
            candidate = _current_slot(entries[selected])
            op_output, op_called = _call_cached(
                OPERATION_CACHE, caches["operation"], "operation", _operation_prompt(slot, candidate),
            )
            new_calls += int(op_called)
            stats["operation_calls"] += int(op_called)
            operation, confidence = normalize_operation(op_output)
            if operation == "UNKNOWN":
                operation = "AUGMENT"
                stats["operation_defaulted"] += 1
            decision = {"decision": operation, "target_cell_id": selected, "confidence": confidence}

        before = set(entries)
        registry = {cell_id: entry.cell for cell_id, entry in entries.items()}
        s48.apply_decision(registry, record, decision, case["network_id"])
        if decision["decision"] == "NEW_CELL":
            cell_id = next(iter(set(registry) - before))
            entries[cell_id] = SemanticCell(registry[cell_id], slot)
        stats[decision["decision"]] += 1
        stats[link_source] += 1
        decisions.append({
            "claim": record.claim, "key": skg.key_description(slot),
            "top3": [{k: v for k, v in row.items() if k != "entry"} for row in ranked],
            "link_source": link_source, **decision,
        })

    for record in records:
        if record.record_type in ("CAUSE", "REACTION"):
            for entry in entries.values():
                if entry.cell.subject == s48._norm(record.subject):
                    entry.cell.observations.append(s48.Stage48Observation(record.record_type.lower(), record))
    return {cell_id: entry.cell for cell_id, entry in entries.items()}, decisions, stats, new_calls


def build_semantic_corpus(cases: list[dict], conversations, raw_by_source: dict[str, Candidate]) -> tuple[dict[str, list[Candidate]], list[dict], Counter, int]:
    extraction_cache = s49._load_extraction_cache()
    caches = {name: _load_cache(path) for name, path in (
        ("normalization", NORMALIZATION_CACHE), ("gate", GATE_CACHE), ("operation", OPERATION_CACHE),
    )}
    by_network: dict[str, list[Candidate]] = defaultdict(list)
    turns = {
        source_id: {
            "turn_id": source_id, "timestamp": candidate.observed_at,
            "speaker": ", ".join(candidate.asserted_by), "message": candidate.text,
        }
        for source_id, candidate in raw_by_source.items()
    }
    logs, totals, new_calls = [], Counter(), 0
    for case in cases:
        records, _turns, failures = s48.extract_records_for_case(case, conversations, extraction_cache)
        registry, decisions, stats, calls = materialize_case(case, records, caches, turns)
        _flat, candidates = s49._derived_candidates(case, records, registry, raw_by_source)
        for candidate in candidates:
            candidate.candidate_id = candidate.candidate_id.replace("cell:", "semantic-cell:", 1)
        by_network[case["network_id"]].extend(candidates)
        totals.update(stats)
        totals["records"] += len(records)
        totals["state_records"] += sum(r.record_type in ("STATE", "TRANSITION") for r in records)
        totals["cells"] += len(registry)
        totals["extraction_failures"] += len(failures)
        new_calls += calls
        logs.append({"qa_id": case["qa_id"], "network_id": case["network_id"], "stats": dict(stats), "decisions": decisions})
    return {network: s49._dedupe_exact(items) for network, items in by_network.items()}, logs, totals, new_calls


def run_question(case: dict, indices: tuple[s49.HybridIndex, ...], raw_by_source: dict[str, Candidate]) -> dict:
    raw_index, flat_index, versioned_index, semantic_index = indices
    question = case["question"]
    query_vector = _normalized([s49.embed(question)])[0]
    raw_retrieved, _ = raw_index.retrieve(question, query_vector)
    _, flat_retrieved = flat_index.retrieve(question, query_vector)
    _, versioned_retrieved = versioned_index.retrieve(question, query_vector)
    _, semantic_retrieved = semantic_index.retrieve(question, query_vector)
    retrieved = {
        "RAW": raw_retrieved,
        "RAW+FLAT": raw_retrieved + flat_retrieved,
        "RAW+VERSIONED": raw_retrieved + versioned_retrieved,
        "RAW+SEMANTIC_VERSIONED": raw_retrieved + semantic_retrieved,
    }
    contexts, packing = {}, {}
    for variant in VARIANTS:
        ranked = s49.HybridIndex.rerank(query_vector, retrieved[variant])
        contexts[variant], packing[variant] = s49.pack_context(ranked, raw_by_source)
    answers = {variant: s48.cached_answer(question, {}, contexts[variant]) for variant in VARIANTS}
    correctness = s48.cached_judge_correctness(question, case["gold_answer"], answers)
    temporal = s48.cached_judge_temporal_attribution(question, case["gold_answer"], answers)
    cited = {variant: _cited_ids(answers[variant]) for variant in VARIANTS}
    return {
        **{key: case[key] for key in ("qa_id", "network_id", "question", "gold_answer")},
        "gold_ids": sorted(case["gold_ids"]), "answers": answers, "correctness": correctness,
        "temporal_attribution": temporal,
        "citation_metrics": {variant: _set_metrics(cited[variant], case["gold_ids"]) for variant in VARIANTS},
        "packing": packing,
    }


def _averages(results: list[dict], getter) -> dict[str, float]:
    return {variant: sum(getter(row, variant) for row in results) / len(results) for variant in VARIANTS}


def _paired(results: list[dict], left: str, right: str) -> tuple[int, int, int]:
    wins = sum(row["correctness"][left] > row["correctness"][right] for row in results)
    losses = sum(row["correctness"][left] < row["correctness"][right] for row in results)
    return wins, len(results) - wins - losses, losses


def render_report(results: list[dict], totals: Counter, calls: dict[str, int]) -> str:
    correctness = _averages(results, lambda row, variant: row["correctness"][variant])
    temporal = _averages(results, lambda row, variant: row["temporal_attribution"][variant]["temporal_correctness"])
    attribution = _averages(results, lambda row, variant: row["temporal_attribution"][variant]["attribution_correctness"])
    precision = _averages(results, lambda row, variant: row["citation_metrics"][variant]["precision"])
    recall = _averages(results, lambda row, variant: row["citation_metrics"][variant]["recall"])
    header = "| metric | " + " | ".join(VARIANTS) + " |"
    separator = "|---|" + "---:|" * len(VARIANTS)
    metric_row = lambda label, values: "| " + label + " | " + " | ".join(f"{values[v]:.3f}" for v in VARIANTS) + " |"
    lines = [
        "# Semantic-key linker: 20-case end-to-end component smoke", "",
        "**Not an official SocialMemBench result.** The 20 Q8 memory inputs were extracted from gold-anchor regions. "
        "This isolates materialization + retrieval + answering after extraction, under the same 400-word budget.", "",
        f"- records/state records: {totals['records']}/{totals['state_records']}",
        f"- semantic cells: {totals['cells']}",
        f"- links accepted by LLM gate: {totals['llm_gate']}; new cells: {totals['new_cell']}",
        f"- operations: REVISE={totals['REVISE']}, OBSERVE={totals['OBSERVE']}, AUGMENT={totals['AUGMENT']}",
        f"- normalization fallbacks: {totals['normalization_fallback']}",
        f"- new calls: materialization={calls['materialization']}, answer/judge={calls['answer_judge']}", "",
        "## Aggregate metrics", "", header, separator,
        metric_row("answer correctness", correctness),
        metric_row("temporal correctness", temporal),
        metric_row("attribution correctness", attribution),
        metric_row("citation precision", precision),
        metric_row("citation recall", recall), "", "## Paired correctness", "",
    ]
    for right in ("RAW", "RAW+FLAT", "RAW+VERSIONED"):
        wins, ties, losses = _paired(results, "RAW+SEMANTIC_VERSIONED", right)
        lines.append(f"- semantic vs {right}: {wins} wins / {ties} ties / {losses} losses")
    lines += ["", "## Per question", ""]
    for row in results:
        lines.append(f"### {row['qa_id']}: {row['question']}")
        for variant in VARIANTS:
            lines.append(
                f"- {variant}: correctness={row['correctness'][variant]:.1f}; "
                f"temporal={row['temporal_attribution'][variant]['temporal_correctness']:.0f}; "
                f"attribution={row['temporal_attribution'][variant]['attribution_correctness']:.0f}; "
                f"citation_recall={row['citation_metrics'][variant]['recall']:.2f}"
            )
        lines.append("")
    return "\n".join(lines)


def _write_manifest() -> None:
    files = {}
    for path in sorted(OUT_DIR.iterdir()):
        if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS"}:
            files[path.name] = {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "files": files}, indent=2) + "\n")
    (OUT_DIR / "SHA256SUMS").write_text("".join(f"{meta['sha256']}  {name}\n" for name, meta in files.items()))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (NORMALIZATION_CACHE, GATE_CACHE, OPERATION_CACHE):
        seed = SEED_CACHE_DIR / path.name
        if not path.exists() and seed.exists():
            shutil.copyfile(seed, path)
    conversations, qa, _ = load_data(DATA_DIR)
    cases = s49._cases(qa)
    raw_by_network, raw_by_source = build_raw_candidates(conversations)
    flat_by_network, versioned_by_network = s49.build_frozen_memory_corpora(cases, conversations, raw_by_source)
    semantic_by_network, logs, totals, materialization_calls = build_semantic_corpus(cases, conversations, raw_by_source)
    with MATERIALIZATION.open("w") as handle:
        for row in logs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not LLM_CACHE.exists():
        seed = SEED_CACHE_DIR / LLM_CACHE.name
        shutil.copyfile(seed if seed.exists() else s49.STAGE49_LLM_CACHE, LLM_CACHE)
    before_answer_calls = len(_read_jsonl(LLM_CACHE))
    s48.LLM_CALL_CACHE = LLM_CACHE
    s48._llm_cache = None

    indices = {}
    for network_id in {case["network_id"] for case in cases}:
        raw = raw_by_network[network_id]
        indices[network_id] = (
            s49.HybridIndex(raw, [], network_id, "smoke-none"),
            s49.HybridIndex(raw, flat_by_network.get(network_id, []), network_id, "smoke-flat"),
            s49.HybridIndex(raw, versioned_by_network.get(network_id, []), network_id, "smoke-versioned"),
            s49.HybridIndex(raw, semantic_by_network.get(network_id, []), network_id, "smoke-semantic"),
        )
    results = []
    print(f"Running {len(cases)} Q8 smoke questions x 4 variants...", file=sys.stderr)
    for index, case in enumerate(cases, 1):
        results.append(run_question(case, indices[case["network_id"]], raw_by_source))
        print(f"  {index}/{len(cases)} {case['qa_id']}", file=sys.stderr)
    with RESULTS.open("w") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    answer_calls = len(_read_jsonl(LLM_CACHE)) - before_answer_calls
    calls = {"materialization": materialization_calls, "answer_judge": answer_calls}
    REPORT.write_text(render_report(results, totals, calls))
    CONFIG.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "cases": len(cases), "variants": VARIANTS,
        "top_k": TOP_K, "gate_threshold": GATE_THRESHOLD, "gate_model": "primary",
        "embedding_fallback": False,
        "new_calls": calls, "methodological_status": "dev/oracle-memory component smoke",
    }, indent=2) + "\n")
    _write_manifest()
    print(f"Report: {REPORT}", file=sys.stderr)


if __name__ == "__main__":
    main()
