"""Stage 5 -- blind held-out SocialMemBench run, three strictly separated
phases: --prepare-memory, --run-inference, --score.

This is NOT a new framework: every piece of retrieval/generation/judging
machinery is imported unmodified from research/socialmembench_pilot.py,
research/versioned_memory_cells.py, and research/stage4_9_hybrid_experiment.py
(HybridIndex, pack_context, CONTEXT_WORD_BUDGET=400). The only new code here
is the phase-separation discipline itself and the held-out-specific memory
build.

Held-out scope (frozen, from /tmp/socialmembench_heldout_manifest.json,
identifiers only): 51 QA across 6 networks --
  grp_0d1e2f3a, grp_4d5e6f7a, grp_6f7a8b9c, grp_7a8b9c0d, grp_8b9c0d1e, grp_a3b4c5d6

Why memory here is NOT the Stage 4.7.1/4.8/4.9 pipeline: that pipeline's
extraction was scoped to Q8 gold-anchor regions -- an oracle-evidence
component experiment by design, explicitly not meant to generalize to a
blind held-out run. Stage 5's memory instead reuses the ORIGINAL,
query-independent ingestion path (build_cell_assertion_cache ->
build_slot_resolution_cache -> materialize_cell_registries ->
validate_assertion/apply_assertion, all from versioned_memory_cells.py /
socialmembench_pilot.py, unmodified) over the full held-out conversation
history -- no QA content ever reaches it.

An earlier accidental unscoped extraction run (documented in Stage 4.5's
data protocol) already left all 33 held-out sessions' raw extraction (and
part of their slot resolution) sitting in the shared, on-disk caches. That
is a fact about processing, not about labels: it is reused here read-only,
verified complete BEFORE any use, and --prepare-memory refuses to trigger
new session extraction silently -- it stops and names exactly which
sessions are missing, if any are.

Phase separation is enforced structurally, not just by convention:
  - --prepare-memory takes no QA input at all (no function here that builds
    the memory snapshot accepts a QA DataFrame or reads qa.parquet).
  - --run-inference reads ONLY qa_id/network_id/query_type/question/
    answer_format/options_json from qa.parquet (column-restricted read),
    generates three answers per question, and never reads gold labels.
  - --score is the only phase that reads answer/correct_option/
    evidence_anchors_json, and it never re-generates an answer -- it only
    judges/scores the sealed predictions from --run-inference.

Run:
    python3 -m research.stage5_heldout --prepare-memory
    python3 -m research.stage5_heldout --run-inference   # requires separate authorization
    python3 -m research.stage5_heldout --score           # requires separate authorization
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from research.socialmembench_pilot import (
    Candidate, _cited_ids, _json, _mc_score, _norm, _set_metrics,
    build_cell_assertion_cache, build_raw_candidates, build_slot_resolution_cache,
    materialize_cell_registries,
)
from research.stage4_5_audit import DATA_DIR
from research import stage4_8_versioned_experiment as s48
from research import stage4_9_hybrid_experiment as s49
from research.versioned_memory_cells import MemoryCell

# Stage 5 gets its OWN embedding-cache directory -- never write into Stage
# 4.9's (/tmp/socialmembench_stage4_9_embeddings), even though HybridIndex's
# caching helper is reused verbatim. Held-out network_ids never collide with
# Stage 4.9's dev network_ids, but the isolation is made explicit anyway.
STAGE5_EMBED_CACHE_DIR = Path("/tmp/socialmembench_stage5_embeddings")

SCHEMA_VERSION = "stage5_v1"

HELDOUT_MANIFEST = Path("/tmp/socialmembench_heldout_manifest.json")
CELL_CACHE = Path("/tmp/socialmembench_cell_assertions.jsonl")
SLOT_CACHE = Path("/tmp/socialmembench_cell_slot_resolution.jsonl")

SNAPSHOT_JSONL = Path("/tmp/socialmembench_stage5_memory_snapshot.jsonl")
SNAPSHOT_META = Path("/tmp/socialmembench_stage5_memory_snapshot_meta.json")
PREDICTIONS_JSONL = Path("/tmp/socialmembench_stage5_predictions.jsonl")
PREDICTIONS_META = Path("/tmp/socialmembench_stage5_predictions_meta.json")
RESULTS_JSONL = Path("/tmp/socialmembench_stage5_results.jsonl")
REPORT_MD = Path("/tmp/socialmembench_stage5_report.md")
SCORE_META = Path("/tmp/socialmembench_stage5_score_meta.json")
STAGE5_LLM_CACHE = Path("/tmp/socialmembench_stage5_llm_cache.jsonl")

INFERENCE_COLUMNS = ["qa_id", "network_id", "query_type", "question", "answer_format", "options_json"]
SCORE_COLUMNS = ["qa_id", "answer", "answer_format", "correct_option", "evidence_anchors_json"]
EXPECTED_QA_COUNT = 51

VARIANTS = ("RAW", "RAW+FLAT", "RAW+VERSIONED")
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_ITERATIONS = 10000


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    return _sha256_bytes(path.read_bytes()) if path.exists() else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_heldout_manifest() -> dict:
    manifest = json.loads(HELDOUT_MANIFEST.read_text())
    if manifest.get("qa_count") != EXPECTED_QA_COUNT or len(manifest.get("qa", [])) != EXPECTED_QA_COUNT:
        raise RuntimeError(f"held-out manifest does not have exactly {EXPECTED_QA_COUNT} QA entries")
    return manifest


def _heldout_networks() -> list[str]:
    return sorted(_read_heldout_manifest()["networks"])


def _read_conversations_only(data_dir: Path) -> pd.DataFrame:
    """PHASE 1 boundary: reads conversations.parquet ONLY. Never touches
    qa.parquet -- there is no QA parameter anywhere in this function or in
    prepare_memory below."""
    return pd.read_parquet(data_dir / "conversations.parquet")


# --------------------------------------------------------------------------
# PHASE 1: --prepare-memory
# --------------------------------------------------------------------------

def _load_raw_cache_dict(path: Path) -> dict[tuple[str, str], dict]:
    cache: dict[tuple[str, str], dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[(row["network_id"], str(row["session_id"]))] = row
    return cache


def _count_pending_slot_llm_buckets(cache: dict[tuple[str, str], dict], slot_cache_path: Path, networks: set[str]) -> int:
    """Mirrors build_slot_resolution_cache's own bucketing exactly (network
    filter, then (owner, subject, facet, scope) buckets across sessions),
    WITHOUT calling the LLM -- singleton buckets are deterministic (no LLM),
    only an unresolved multi-item bucket would cost a real call."""
    resolved: set[tuple[str, str, int]] = set()
    if slot_cache_path.exists():
        for line in slot_cache_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                resolved.add((row["network_id"], row["session_id"], row["item_index"]))

    buckets: dict[tuple, list[tuple[str, str, int]]] = defaultdict(list)
    for (network_id, session_id), row in cache.items():
        if network_id not in networks:
            continue
        for item_index, item in enumerate(row.get("raw_items", [])):
            bucket_key = (
                network_id, _norm(item.get("viewpoint_owner", "")), _norm(item.get("subject", "")),
                _norm(item.get("facet", "")), _norm(item.get("scope_type") or "individual"),
            )
            buckets[bucket_key].append((network_id, session_id, item_index))

    pending = 0
    for entries in buckets.values():
        if all(entry in resolved for entry in entries):
            continue
        if len(entries) == 1:
            continue  # deterministic content-hash fallback, no LLM
        pending += 1
    return pending


def _render_versioned_cell_text(cell: MemoryCell) -> str:
    """No facet-graph, no relation-typed edges -- current active state
    version(s), prior (closed) versions, and reaffirming observations, using
    only fields versioned_memory_cells.py already defines. This schema has
    no CAUSE/REACTION record type (that concept belongs to the separate,
    explicitly-excluded Stage 4.7.1 extraction schema) -- omitted here, not
    faked."""
    lines: list[str] = []
    if cell.active_states:
        lines.append("CURRENT STATE(S):")
        for version in cell.active_states:
            lines.append(f"  [{version.operation}] {version.assertion_text}")
    closed_versions = [v for v in cell.state_versions if v.closed_at]
    if closed_versions:
        lines.append("PRIOR VERSIONS:")
        for version in closed_versions:
            lines.append(f"  [{version.operation}] {version.assertion_text}")
    if cell.observations:
        lines.append("OBSERVATIONS:")
        for observation in cell.observations:
            lines.append(f"  {observation.assertion_text}")
    return "\n".join(lines)


def _cell_all_source_ids(cell: MemoryCell) -> tuple[str, ...]:
    ids: list[str] = []
    for version in cell.state_versions:
        ids.extend(version.source_turn_ids)
    for observation in cell.observations:
        ids.extend(observation.source_turn_ids)
    return tuple(dict.fromkeys(ids))


def build_memory_candidates(
    conversations: pd.DataFrame, networks: list[str], cache: dict, slot_resolution: dict,
) -> tuple[list[Candidate], list[Candidate], dict[str, Any]]:
    """The ONE validated-assertion pass materialize_cell_registries already
    does is the single source for BOTH corpora below -- FLAT is one
    Candidate per validated assertion, VERSIONED is one Candidate per
    resulting MemoryCell. Neither corpus is built from the other; both read
    the same `validated_assertions` / `registries` return values."""
    registries, _stats, _merge_events, validated_assertions = materialize_cell_registries(
        conversations, cache, slot_resolution,
    )

    valid_turn_ids: dict[str, set[str]] = defaultdict(set)
    for row in conversations.itertuples(index=False):
        valid_turn_ids[row.network_id].add(row.turn_id)

    flat_candidates: list[Candidate] = []
    rejected_flat = 0
    for network_id in networks:
        for index, assertion in enumerate(validated_assertions.get(network_id, [])):
            sources = assertion.source_turn_ids
            if not sources or any(t not in valid_turn_ids[network_id] for t in sources):
                rejected_flat += 1
                continue
            flat_candidates.append(Candidate(
                candidate_id=f"flat:{network_id}:{index}", kind="derived:assertion", text=assertion.assertion_text,
                source_ids=sources, asserted_by=(assertion.viewpoint_owner,), entities=(assertion.subject,),
                observed_at=assertion.observed_at,
            ))

    versioned_candidates: list[Candidate] = []
    rejected_versioned = 0
    for network_id in networks:
        for cell in registries.get(network_id, {}).values():
            sources = _cell_all_source_ids(cell)
            if not sources or any(t not in valid_turn_ids[network_id] for t in sources):
                rejected_versioned += 1
                continue
            versioned_candidates.append(Candidate(
                candidate_id=f"cell:{network_id}:{cell.cell_id}", kind="derived:versioned_cell",
                text=_render_versioned_cell_text(cell), source_ids=sources,
                asserted_by=(cell.key.viewpoint_owner,), entities=(cell.key.subject,),
                observed_at=max((v.observed_at for v in cell.state_versions), default=""),
            ))

    validated_count = sum(len(validated_assertions.get(n, [])) for n in networks)
    return flat_candidates, versioned_candidates, {
        "validated_assertion_count": validated_count,
        "rejected_flat": rejected_flat, "rejected_versioned": rejected_versioned,
    }


def prepare_memory() -> dict[str, Any]:
    networks = _heldout_networks()
    expected_networks = sorted(networks)

    conversations_all = _read_conversations_only(DATA_DIR)
    conversations = conversations_all[conversations_all.network_id.isin(networks)].copy()

    found_networks = sorted(conversations.network_id.astype(str).unique().tolist())
    if found_networks != expected_networks:
        raise RuntimeError(f"expected exactly the 6 held-out networks {expected_networks}, found {found_networks}")

    session_pairs = sorted({(str(r.network_id), str(r.session_id)) for r in conversations.itertuples(index=False)})
    if len(session_pairs) != 33:
        raise RuntimeError(f"expected exactly 33 held-out sessions, found {len(session_pairs)}: {session_pairs}")

    cache_raw = _load_raw_cache_dict(CELL_CACHE)
    missing_sessions = [pair for pair in session_pairs if pair not in cache_raw]
    if missing_sessions:
        print("STOPPING: the following held-out sessions are missing from the raw extraction cache "
              f"({CELL_CACHE}) -- refusing to trigger new session extraction silently:", file=sys.stderr)
        for pair in missing_sessions:
            print(f"  {pair}", file=sys.stderr)
        raise SystemExit(1)

    # Safe now: every needed (network_id, session_id) is already cached, so
    # this call's internal "groups not yet cached" set is empty -- zero new
    # extraction LLM calls, verified by the check above, not just hoped for.
    cache = build_cell_assertion_cache(conversations, CELL_CACHE)
    raw_assertion_count = sum(len(cache[pair].get("raw_items", [])) for pair in session_pairs)

    network_set = set(networks)
    pending_slot_llm_buckets = _count_pending_slot_llm_buckets(cache, SLOT_CACHE, network_set)
    print(f"Pending slot-resolution LLM buckets (multi-item, not yet cached): {pending_slot_llm_buckets}", file=sys.stderr)
    if pending_slot_llm_buckets > 0:
        print("STOPPING: slot resolution would require new LLM calls -- not authorized during --prepare-memory. "
              "Re-run after obtaining explicit authorization for these calls.", file=sys.stderr)
        raise SystemExit(1)

    # Safe: only deterministic singleton fallbacks (or already-cached
    # entries) remain, confirmed by the zero count above.
    slot_resolution = build_slot_resolution_cache(cache, SLOT_CACHE, network_set)

    flat_candidates, versioned_candidates, build_stats = build_memory_candidates(
        conversations, networks, cache, slot_resolution,
    )

    snapshot_rows = (
        [{"corpus": "flat", **candidate.as_dict()} for candidate in flat_candidates]
        + [{"corpus": "versioned", **candidate.as_dict()} for candidate in versioned_candidates]
    )
    snapshot_bytes = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in snapshot_rows) + "\n").encode()
    new_checksum = _sha256_bytes(snapshot_bytes)

    reused = False
    if SNAPSHOT_JSONL.exists():
        existing_checksum = _sha256_file(SNAPSHOT_JSONL)
        if existing_checksum == new_checksum:
            reused = True
        else:
            raise RuntimeError(
                f"{SNAPSHOT_JSONL} already exists with a DIFFERENT checksum "
                f"(existing={existing_checksum}, newly computed={new_checksum}). "
                "The snapshot is immutable once created -- refusing to overwrite silently."
            )
    else:
        SNAPSHOT_JSONL.write_bytes(snapshot_bytes)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "networks": expected_networks,
        "session_count": len(session_pairs),
        "raw_assertion_count": raw_assertion_count,
        "validated_assertion_count": build_stats["validated_assertion_count"],
        "rejected_count": {"flat": build_stats["rejected_flat"], "versioned": build_stats["rejected_versioned"]},
        "flat_candidate_count": len(flat_candidates),
        "versioned_cell_count": len(versioned_candidates),
        "source_cache_paths": {"cell_assertion_cache": str(CELL_CACHE), "slot_resolution_cache": str(SLOT_CACHE)},
        "source_cache_sha256": {
            "cell_assertion_cache": _sha256_file(CELL_CACHE), "slot_resolution_cache": _sha256_file(SLOT_CACHE),
        },
        "snapshot_sha256": new_checksum,
        "snapshot_reused": reused,
        "created_at": _now_iso(),
        "qa_parquet_read": False,
        "qa_labels_read": False,
        "query_conditioned_extraction": False,
    }
    SNAPSHOT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


# --------------------------------------------------------------------------
# PHASE 2: --run-inference (implemented, NOT invoked without separate
# authorization; guarded below in main()).
# --------------------------------------------------------------------------

def _load_snapshot() -> tuple[list[dict], dict]:
    if not SNAPSHOT_META.exists() or not SNAPSHOT_JSONL.exists():
        raise RuntimeError("memory snapshot/metadata missing -- run --prepare-memory first")
    meta = json.loads(SNAPSHOT_META.read_text())
    actual_checksum = _sha256_file(SNAPSHOT_JSONL)
    if actual_checksum != meta.get("snapshot_sha256"):
        raise RuntimeError(
            f"snapshot checksum mismatch: file={actual_checksum} metadata={meta.get('snapshot_sha256')} -- "
            "the memory snapshot may have been modified since --prepare-memory ran. Refusing to run inference."
        )
    rows = [json.loads(line) for line in SNAPSHOT_JSONL.read_text().splitlines() if line.strip()]
    return rows, meta


def _read_qa_for_inference(manifest: dict) -> pd.DataFrame:
    qa_ids = {row["qa_id"] for row in manifest["qa"]}
    qa = pd.read_parquet(DATA_DIR / "qa.parquet", columns=INFERENCE_COLUMNS)
    qa = qa[qa.qa_id.isin(qa_ids)].copy()
    if len(qa) != EXPECTED_QA_COUNT:
        raise RuntimeError(f"expected exactly {EXPECTED_QA_COUNT} held-out QA rows, found {len(qa)}")
    if qa.qa_id.nunique() != EXPECTED_QA_COUNT:
        raise RuntimeError("held-out QA rows are not unique by qa_id")
    if set(qa.network_id.astype(str)) - set(manifest["networks"]):
        raise RuntimeError("a held-out QA row references a network outside the frozen manifest")
    return qa


def _candidate_from_row(row: dict) -> Candidate:
    return Candidate(
        candidate_id=row["candidate_id"], kind=row["kind"], text=row["text"],
        source_ids=tuple(row["source_ids"]), asserted_by=tuple(row.get("asserted_by", [])),
        entities=tuple(row.get("entities", [])), observed_at=row.get("observed_at", ""),
    )


def run_inference() -> None:
    snapshot_rows, snapshot_meta = _load_snapshot()
    manifest = _read_heldout_manifest()
    qa = _read_qa_for_inference(manifest)

    conversations_all = _read_conversations_only(DATA_DIR)
    raw_by_network, raw_by_source = build_raw_candidates(conversations_all)

    flat_by_network: dict[str, list[Candidate]] = defaultdict(list)
    versioned_by_network: dict[str, list[Candidate]] = defaultdict(list)
    for row in snapshot_rows:
        network_id = row["candidate_id"].split(":")[1]
        candidate = _candidate_from_row(row)
        if row["corpus"] == "flat":
            flat_by_network[network_id].append(candidate)
        else:
            versioned_by_network[network_id].append(candidate)

    s48.LLM_CALL_CACHE = STAGE5_LLM_CACHE
    s48._llm_cache = None
    s49.EMBED_CACHE_DIR = STAGE5_EMBED_CACHE_DIR

    already_done: dict[str, dict] = {}
    if PREDICTIONS_JSONL.exists():
        for line in PREDICTIONS_JSONL.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                already_done[row["qa_id"]] = row

    indices: dict[str, tuple[s49.HybridIndex, s49.HybridIndex, s49.HybridIndex]] = {}
    predictions = list(already_done.values())
    generation_calls = 0

    for row in qa.itertuples(index=False):
        if row.qa_id in already_done:
            continue  # resume: technical-failure resume only, never re-runs a seen answer
        network_id = row.network_id
        if network_id not in indices:
            raw = raw_by_network.get(network_id, [])
            flat = flat_by_network.get(network_id, [])
            versioned = versioned_by_network.get(network_id, [])
            indices[network_id] = (
                s49.HybridIndex(raw, [], network_id, "none"),
                s49.HybridIndex(raw, flat, network_id, "flat"),
                s49.HybridIndex(raw, versioned, network_id, "versioned"),
            )
        raw_index, flat_index, versioned_index = indices[network_id]

        from embeddings import embed
        from research.socialmembench_pilot import _normalized

        query_vector = _normalized([embed(row.question)])[0]
        raw_retrieved, _ = raw_index.retrieve(row.question, query_vector)
        _, flat_retrieved = flat_index.retrieve(row.question, query_vector)
        _, versioned_retrieved = versioned_index.retrieve(row.question, query_vector)

        retrieved = {
            "RAW": raw_retrieved, "RAW+FLAT": raw_retrieved + flat_retrieved,
            "RAW+VERSIONED": raw_retrieved + versioned_retrieved,
        }
        options = _json(row.options_json, {})
        contexts, packing, answers = {}, {}, {}
        for variant in VARIANTS:
            ranked = s49.HybridIndex.rerank(query_vector, retrieved[variant])
            contexts[variant], packing[variant] = s49.pack_context(ranked, raw_by_source)
            answers[variant] = s48.cached_answer(row.question, options, contexts[variant])
            generation_calls += 1
        cited = {variant: sorted(_cited_ids(answers[variant])) for variant in VARIANTS}

        raw_sources = set(packing["RAW"]["source_ids"])
        diagnostics = {}
        for variant, memory_retrieved in (("RAW+FLAT", flat_retrieved), ("RAW+VERSIONED", versioned_retrieved)):
            valid_memory = [
                c for c in memory_retrieved if c.source_ids and all(s in raw_by_source for s in c.source_ids)
            ]
            exposed = set(packing[variant]["source_ids"])
            diagnostics[variant] = {
                "memory_retrieved": len(memory_retrieved),
                "memory_with_valid_provenance": len(valid_memory),
                "memory_packed": len(packing[variant]["memory_candidate_ids"]),
                "raw_evidence_displaced": sorted(raw_sources - exposed),
            }

        predictions.append({
            "qa_id": row.qa_id, "network_id": network_id, "query_type": row.query_type,
            "answers": answers, "cited_ids": cited,
            "retrieval_candidates": {v: [c.candidate_id for c in retrieved[v]] for v in VARIANTS},
            "packed_candidates": {v: packing[v]["candidate_ids"] for v in VARIANTS},
            "exposed_source_ids": {v: packing[v]["source_ids"] for v in VARIANTS},
            "context_word_counts": {v: packing[v]["context_words"] for v in VARIANTS},
            "diagnostics": diagnostics,
        })
        with PREDICTIONS_JSONL.open("a") as f:
            f.write(json.dumps(predictions[-1], ensure_ascii=False) + "\n")

    if len(predictions) != EXPECTED_QA_COUNT:
        raise RuntimeError(f"expected {EXPECTED_QA_COUNT} sealed predictions, wrote {len(predictions)}")

    predictions_bytes = PREDICTIONS_JSONL.read_bytes()
    meta = {
        "schema_version": SCHEMA_VERSION,
        "predictions_sha256": _sha256_bytes(predictions_bytes),
        "snapshot_sha256": snapshot_meta["snapshot_sha256"],
        "context_word_budget": 400,
        "variants": list(VARIANTS),
        "generation_calls_this_run": generation_calls,
        "prediction_count": len(predictions),
        "qa_labels_read": False,
        "created_at": _now_iso(),
    }
    PREDICTIONS_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------
# PHASE 3: --score (implemented, NOT invoked without separate authorization)
# --------------------------------------------------------------------------

def _load_sealed_predictions() -> tuple[list[dict], dict]:
    if not PREDICTIONS_META.exists() or not PREDICTIONS_JSONL.exists():
        raise RuntimeError("sealed predictions/metadata missing -- run --run-inference first")
    meta = json.loads(PREDICTIONS_META.read_text())
    actual = _sha256_bytes(PREDICTIONS_JSONL.read_bytes())
    if actual != meta.get("predictions_sha256"):
        raise RuntimeError(f"predictions checksum mismatch: file={actual} metadata={meta.get('predictions_sha256')}")
    _snapshot_rows, snapshot_meta = _load_snapshot()
    if snapshot_meta["snapshot_sha256"] != meta.get("snapshot_sha256"):
        raise RuntimeError("predictions were generated against a different memory snapshot than the current one")
    predictions = [json.loads(line) for line in PREDICTIONS_JSONL.read_text().splitlines() if line.strip()]
    if len(predictions) != EXPECTED_QA_COUNT:
        raise RuntimeError(f"score requires exactly {EXPECTED_QA_COUNT} sealed predictions, found {len(predictions)}")
    return predictions, meta


def _bootstrap_ci(deltas: list[float], seed: int, iterations: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * iterations)]
    hi = means[int(0.975 * iterations) - 1]
    return lo, hi


def score() -> None:
    predictions, _pred_meta = _load_sealed_predictions()
    manifest = _read_heldout_manifest()
    qa_ids = {row["qa_id"] for row in manifest["qa"]}
    gold = pd.read_parquet(DATA_DIR / "qa.parquet", columns=SCORE_COLUMNS)
    gold = gold[gold.qa_id.isin(qa_ids)].set_index("qa_id")

    s48.LLM_CALL_CACHE = STAGE5_LLM_CACHE
    s48._llm_cache = None

    results = []
    for prediction in predictions:
        gold_row = gold.loc[prediction["qa_id"]]
        gold_ids = {a["turn_id"] for a in _json(gold_row["evidence_anchors_json"], []) if a.get("turn_id")}
        answers = prediction["answers"]

        if gold_row["answer_format"] == "multiple_choice":
            correctness = {v: _mc_score(answers[v], gold_row["correct_option"]) for v in VARIANTS}
            temporal_attribution = {v: {"temporal_correctness": None, "attribution_correctness": None} for v in VARIANTS}
        else:
            correctness = s48.cached_judge_correctness(prediction["qa_id"] + "|" + str(gold_row["answer"]), gold_row["answer"], answers)
            temporal_attribution = s48.cached_judge_temporal_attribution(
                prediction["qa_id"] + "|" + str(gold_row["answer"]), gold_row["answer"], answers,
            )

        citation_metrics = {}
        for variant in VARIANTS:
            cited = set(prediction["cited_ids"][variant])
            exposed = set(prediction["exposed_source_ids"][variant])
            metrics = _set_metrics(cited, gold_ids)
            metrics["validity"] = len(cited & exposed) / len(cited) if cited else 1.0
            citation_metrics[variant] = metrics

        results.append({
            "qa_id": prediction["qa_id"], "network_id": prediction["network_id"], "query_type": prediction["query_type"],
            "correctness": correctness, "temporal_attribution": temporal_attribution,
            "citation_metrics": citation_metrics, "answers": answers, "diagnostics": prediction["diagnostics"],
        })

    RESULTS_JSONL.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
    REPORT_MD.write_text(_render_score_report(results))
    SCORE_META.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "n": len(results), "created_at": _now_iso(),
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    }, indent=2, ensure_ascii=False))


def _render_score_report(results: list[dict]) -> str:
    lines = [
        "# Stage 5 -- blind held-out score report",
        "",
        "**This is a held-out split of the current LOCAL protocol, not an official SocialMemBench leaderboard "
        "submission.** Held-out conversations were previously processed by an earlier accidental unscoped "
        "extraction run, but QA question/answer/evidence/correct_option were never read before blind inference "
        "and scoring. Memory extraction is query-independent. FLAT and VERSIONED were built from the SAME "
        "validated assertion corpus. Absolute numbers here are NOT comparable to the oracle-evidence Stage "
        "4.8/4.9 component experiments -- different memory-construction regime entirely.",
        "",
        f"- n = {len(results)}",
        "",
    ]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["query_type"]].append(r)

    def _agg(rows: list[dict]) -> dict[str, float]:
        n = len(rows)
        out = {}
        for variant in VARIANTS:
            vals = [r["correctness"][variant] for r in rows if r["correctness"][variant] is not None]
            out[variant] = sum(vals) / len(vals) if vals else float("nan")
        return out

    lines += ["## Overall (n={})".format(len(results)), "", "| variant | correctness |", "|---|---|"]
    overall = _agg(results)
    for v in VARIANTS:
        lines.append(f"| {v} | {overall[v]:.3f} |")

    lines += ["", "## By query_type", ""]
    for qtype in sorted(by_type):
        rows = by_type[qtype]
        agg = _agg(rows)
        lines.append(f"- {qtype} (n={len(rows)}): " + ", ".join(f"{v}={agg[v]:.3f}" for v in VARIANTS))

    def _paired(left: str, right: str, rows: list[dict]) -> tuple[int, int, int, list[float]]:
        deltas = [r["correctness"][left] - r["correctness"][right] for r in rows if r["correctness"][left] is not None]
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        return wins, len(deltas) - wins - losses, losses, deltas

    lines += ["", "## Primary paired comparisons (overall)", ""]
    for left, right in (("RAW+VERSIONED", "RAW"), ("RAW+VERSIONED", "RAW+FLAT")):
        wins, ties, losses, deltas = _paired(left, right, results)
        ci = _bootstrap_ci(deltas, BOOTSTRAP_SEED, BOOTSTRAP_ITERATIONS) if deltas else (float("nan"), float("nan"))
        lines.append(f"- {left} vs {right}: {wins}W/{ties}T/{losses}L, mean delta 95% CI [{ci[0]:.3f}, {ci[1]:.3f}] (seed={BOOTSTRAP_SEED})")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-memory", action="store_true")
    group.add_argument("--run-inference", action="store_true")
    group.add_argument("--score", action="store_true")
    args = parser.parse_args()

    if args.prepare_memory:
        meta = prepare_memory()
        print(json.dumps(meta, indent=2, ensure_ascii=False))
    elif args.run_inference:
        run_inference()
        print(f"Predictions: {PREDICTIONS_JSONL}", file=sys.stderr)
    elif args.score:
        score()
        print(f"Results: {RESULTS_JSONL}", file=sys.stderr)
        print(f"Report: {REPORT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
