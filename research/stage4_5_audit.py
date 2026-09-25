"""Stage 4.5 -- structural evaluation of the v1 topic-scoped memory cells,
ZERO new ingestion LLM calls.

Fixed once already (Stage 4.5 first pass) and then corrected again after
review: the first Q8 audit picked the FIRST shared cell and conflated
"false split" with "extracted but never materialized" and with
single-session cases (marked n/a instead of classified); the first Q6
audit declared a group norm "found" from facet overlap alone, which is a
proxy dressed as a verdict. This version fixes both without touching
memory architecture, thresholds, the extractor prompt, or the slot
resolver -- only the EVALUATION code changed.

Three independent audits, run against data ALREADY cached on disk
(research/socialmembench_pilot.py's build_cell_assertion_cache /
build_slot_resolution_cache, resumed unchanged, never re-extracted):
  A. Q8 (temporal_shift) chain audit -- old/new evidence split by
     (session_index, turn order), extraction/materialization tracked
     SEPARATELY for each side, success checked across EVERY shared cell
     (not just the first), single-session cases classified, not skipped.
  B. Q6 (norm_vs_individual) scope audit -- a REVIEW SHEET, not a verdict
     table. Candidate group/subgroup cells are ranked by embedding
     similarity to the question for a human to read; nothing is declared
     "the norm" or "conflated" automatically from facet overlap.
  C. False-split diagnostic, no QA at all -- unchanged from the first
     pass (not in scope for this correction).

Data protocol: the 120 QA already used above are frozen as the
development/diagnostic set. The 6 SocialMemBench networks NOT among the
37 selected are recorded as a held-out test set -- this run never reads
their question/answer/evidence_anchors/correct_option, only lists
qa_id/network_id/query_type identifiers. Conversations from those networks
were partially processed by an EARLIER accidental unscoped extraction run
(documented, not repeated here) -- that is recorded honestly as "processed,
not leaked" (QA labels were never touched), not swept under "cache exists
so it's fine."

Run:
    python3 -m research.stage4_5_audit
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings import embed, embed_batch
from research.socialmembench_pilot import (
    _json, _norm, _select_questions, build_cell_assertion_cache,
    build_slot_resolution_cache, load_data, materialize_cell_registries,
)
from research.versioned_memory_cells import (
    CellKey, MemoryCell, MemoryStateVersion, _cosine, _temporal_windows_overlap,
)

DATA_DIR = Path("/tmp/socialmembench")
CELL_CACHE = Path("/tmp/socialmembench_cell_assertions.jsonl")
SLOT_CACHE = Path("/tmp/socialmembench_cell_slot_resolution.jsonl")
PER_TYPE = 20
SEED = 20260910

Q8_AUDIT_MD = Path("/tmp/socialmembench_q8_chain_audit.md")
Q8_AUDIT_JSONL = Path("/tmp/socialmembench_q8_chain_audit.jsonl")
Q6_AUDIT_MD = Path("/tmp/socialmembench_q6_scope_audit.md")
Q6_AUDIT_JSONL = Path("/tmp/socialmembench_q6_scope_audit.jsonl")
FALSE_SPLIT_MD = Path("/tmp/socialmembench_false_split_review.md")
DEV_DIAGNOSTIC_MANIFEST = Path("/tmp/socialmembench_dev_diagnostic_manifest.json")
HELDOUT_MANIFEST = Path("/tmp/socialmembench_heldout_manifest.json")
DATA_PROTOCOL_MD = Path("/tmp/socialmembench_stage4_5_data_protocol.md")

TRIGGER_KEYWORDS = (
    "reason", "because", "due to", "explain", "injury", "since", "caused",
    "trigger", "why", "explanation",
)


def _cell_source_ids(cell: MemoryCell) -> set[str]:
    ids: set[str] = set()
    for version in cell.state_versions:
        ids.update(version.source_turn_ids)
    for observation in cell.observations:
        ids.update(observation.source_turn_ids)
    return ids


def _rep_text(cell: MemoryCell) -> str:
    if cell.state_versions:
        return cell.state_versions[0].assertion_text
    if cell.observations:
        return cell.observations[0].assertion_text
    return cell.key.topic_key


def _is_trigger_anchor(anchor: dict) -> bool:
    text = (anchor.get("relevance") or "").casefold()
    return any(keyword in text for keyword in TRIGGER_KEYWORDS)


# --------------------------------------------------------------------------
# Part A: Q8 temporal-chain audit
# --------------------------------------------------------------------------

_TURN_NUM_RE = re.compile(r"t(\d+)$")


def _turn_order_key(anchor: dict) -> tuple[int, int]:
    """(session_index, turn number within session) -- turn_ids are of the
    deterministic form "<session_id>_t<NNN>" with NNN monotonically
    increasing within a session (see materialize_cell_registries's own
    suffix_to_full reconstruction, same assumption), so this is a real
    chronological order, not just a same-session tiebreak proxy."""
    turn_id = anchor.get("turn_id") or ""
    match = _TURN_NUM_RE.search(turn_id)
    turn_num = int(match.group(1)) if match else 0
    session_index = anchor.get("session_index")
    session_index = int(session_index) if session_index is not None else 0
    return (session_index, turn_num)


def _split_old_new(anchors: list[dict]) -> tuple[set[str], set[str], set[str], bool]:
    """Splits gold evidence anchors into an OLD group (earliest distinct
    (session_index, turn-order) point) and a NEW group (latest), by
    session_index primarily and turn order within a session as the
    tiebreak/split for single-session cases -- so a single-session
    temporal shift is split too, never routed to n/a. Anchors at neither
    extreme (a 3rd, middle session, say) are left unassigned rather than
    forced into either side. degenerate=True means every anchor shares the
    exact same time point -- there is no real before/after to order, and
    old==new by construction."""
    keyed = [a for a in anchors if a.get("turn_id")]
    if not keyed:
        return set(), set(), set(), True
    distinct_keys = sorted({_turn_order_key(a) for a in keyed})
    if len(distinct_keys) <= 1:
        ids = {a["turn_id"] for a in keyed}
        return ids, ids, set(), True
    old_key, new_key = distinct_keys[0], distinct_keys[-1]
    old_ids = {a["turn_id"] for a in keyed if _turn_order_key(a) == old_key}
    new_ids = {a["turn_id"] for a in keyed if _turn_order_key(a) == new_key}
    all_ids = {a["turn_id"] for a in keyed}
    unassigned = all_ids - old_ids - new_ids
    return old_ids, new_ids, unassigned, False


def _classify_shared_cells(
    shared_cell_ids: set[str], cell_by_id: dict[str, MemoryCell], old_ids: set[str], new_ids: set[str],
) -> tuple[str, Optional[MemoryStateVersion], Optional[MemoryStateVersion]]:
    """Scans EVERY shared cell (not just the first) -- success as soon as
    ANY of them shows a correctly-ordered, genuinely-transitioning old->new
    pair. Reports the best diagnostic across all of them when none
    qualify."""
    saw_temporal_issue = False
    for cell_id in shared_cell_ids:
        cell = cell_by_id[cell_id]
        old_versions = [v for v in cell.state_versions if set(v.source_turn_ids) & old_ids]
        new_versions = [v for v in cell.state_versions if set(v.source_turn_ids) & new_ids]
        if not old_versions or not new_versions:
            continue
        old_v = min(old_versions, key=lambda v: v.version_no)
        new_v = max(new_versions, key=lambda v: v.version_no)
        if new_v.version_no <= old_v.version_no:
            continue  # collapsed into the same (or an earlier) version -- not a real chain here
        if new_v.effective_from and old_v.effective_from and new_v.effective_from < old_v.effective_from:
            saw_temporal_issue = True
            continue
        if new_v.operation not in ("revise", "retract"):
            continue  # correctly ordered, but not represented as an actual transition
        return "chain_recovered", old_v, new_v
    return ("temporal_order" if saw_temporal_issue else "wrong_operation"), None, None


def audit_q8_case(row: dict[str, Any], cells: list[MemoryCell], extracted_ids: set[str]) -> dict[str, Any]:
    anchors = _json(row["evidence_anchors_json"], [])
    gold_ids = {a["turn_id"] for a in anchors if a.get("turn_id")}
    gold_source_count = len(gold_ids)

    result: dict[str, Any] = {
        "qa_id": row["qa_id"], "network_id": row["network_id"], "question": row["question"],
        "gold_source_count": gold_source_count, "degenerate_single_point": False,
        "old_evidence_ids": [], "new_evidence_ids": [], "unassigned_evidence_ids": [],
        "old_extracted_ids": [], "new_extracted_ids": [], "both_sides_extracted": False,
        "old_cell_ids": [], "new_cell_ids": [], "shared_cell_ids": [], "trigger_coverage": "n/a",
    }
    if gold_source_count == 0:
        result.update(failure_stage="extraction_missing_both", note="QA case has no evidence anchors at all")
        return result

    old_ids, new_ids, unassigned_ids, degenerate = _split_old_new(anchors)
    cell_by_id = {cell.cell_id: cell for cell in cells}

    old_extracted = old_ids & extracted_ids
    new_extracted = new_ids & extracted_ids
    extraction_missing_old = len(old_extracted) == 0
    extraction_missing_new = len(new_extracted) == 0
    both_sides_extracted = not extraction_missing_old and not extraction_missing_new

    old_cell_ids = {cell.cell_id for cell in cells if _cell_source_ids(cell) & old_ids}
    new_cell_ids = {cell.cell_id for cell in cells if _cell_source_ids(cell) & new_ids}
    shared_cell_ids = old_cell_ids & new_cell_ids

    trigger_ids = {a["turn_id"] for a in anchors if a.get("turn_id") and _is_trigger_anchor(a)}
    trigger_coverage = (
        "n/a" if not trigger_ids else round(len(trigger_ids & extracted_ids) / len(trigger_ids), 3)
    )

    result.update(
        degenerate_single_point=degenerate,
        old_evidence_ids=sorted(old_ids), new_evidence_ids=sorted(new_ids),
        unassigned_evidence_ids=sorted(unassigned_ids),
        old_extracted_ids=sorted(old_extracted), new_extracted_ids=sorted(new_extracted),
        both_sides_extracted=both_sides_extracted,
        old_cell_ids=sorted(old_cell_ids), new_cell_ids=sorted(new_cell_ids),
        shared_cell_ids=sorted(shared_cell_ids), trigger_coverage=trigger_coverage,
    )

    if extraction_missing_old and extraction_missing_new:
        result["failure_stage"] = "extraction_missing_both"
        return result
    if extraction_missing_old:
        result["failure_stage"] = "extraction_missing_old"
        return result
    if extraction_missing_new:
        result["failure_stage"] = "extraction_missing_new"
        return result

    if degenerate:
        # Only one real point of evidence (old == new by construction) --
        # there is nothing to order, so success is just "landed in a cell
        # at all", not a chain.
        if old_cell_ids:
            stage = "chain_recovered"
            if trigger_ids and trigger_coverage != 1.0:
                stage = "trigger_missing"
            result["failure_stage"] = stage
        else:
            result["failure_stage"] = "wrong_operation"
        return result

    if not old_cell_ids or not new_cell_ids:
        # Extracted+validated on (at least) one side, but never landed in
        # ANY cell (e.g. item D's content-free reject) -- a materialization
        # drop, not a split into two DIFFERENT cells.
        result["failure_stage"] = "wrong_operation"
        return result

    if not shared_cell_ids:
        result["failure_stage"] = "false_split"
        return result

    stage, old_v, new_v = _classify_shared_cells(shared_cell_ids, cell_by_id, old_ids, new_ids)
    if stage == "chain_recovered":
        provenance = len(gold_ids & (set(old_v.source_turn_ids) | set(new_v.source_turn_ids))) / gold_source_count
        result["provenance_completeness"] = round(provenance, 3)
        result["old_state_version_id"] = old_v.version_id
        result["new_state_version_id"] = new_v.version_id
        result["new_state_operation"] = new_v.operation
        if trigger_ids and trigger_coverage != 1.0:
            stage = "trigger_missing"
    result["failure_stage"] = stage
    return result


def render_q8_report(cases: list[dict[str, Any]]) -> str:
    stage_counts = Counter(c["failure_stage"] for c in cases)
    both_subset = [c for c in cases if c.get("both_sides_extracted")]
    both_stage_counts = Counter(c["failure_stage"] for c in both_subset)
    n, m = len(cases), len(both_subset)

    lines = [
        "# Q8 (temporal_shift) chain audit -- 20/20 cases, structural only, no new ingestion calls",
        "",
        "Evidence anchors split into OLD/NEW groups by (session_index, turn order within session) -- "
        "the earliest and latest distinct time point in the gold evidence, so single-session cases are "
        "split (and classified) too, never treated as n/a. QA evidence anchors used ONLY here, after "
        "materialization, for evaluation.",
        "",
        "## Unconditional metrics (over all 20 cases)",
        "",
        f"- extraction_missing_old: {sum(1 for c in cases if c['failure_stage']=='extraction_missing_old')}/{n}",
        f"- extraction_missing_new: {sum(1 for c in cases if c['failure_stage']=='extraction_missing_new')}/{n}",
        f"- extraction_missing_both: {sum(1 for c in cases if c['failure_stage']=='extraction_missing_both')}/{n}",
        f"- both_sides_extracted: {m}/{n}",
        f"- degenerate_single_point (no real before/after in the gold evidence): "
        f"{sum(1 for c in cases if c.get('degenerate_single_point'))}/{n}",
        "",
        "### failure_stage distribution (unconditional, all 20)",
        "",
    ]
    for stage in ("extraction_missing_old", "extraction_missing_new", "extraction_missing_both",
                  "false_split", "temporal_order", "wrong_operation", "trigger_missing", "chain_recovered"):
        lines.append(f"- {stage}: {stage_counts.get(stage, 0)}")
    lines += ["", f"## Conditional metrics (among the {m} cases where both_sides_extracted=True)", ""]
    if m:
        for stage in ("false_split", "temporal_order", "wrong_operation", "trigger_missing", "chain_recovered"):
            lines.append(f"- {stage}: {both_stage_counts.get(stage, 0)}/{m}")
    else:
        lines.append("- (no case had both sides extracted)")
    lines += ["", "## All 20 cases", ""]
    for i, case in enumerate(cases, 1):
        lines.append(f"### {i}. {case['qa_id']} :: {case['network_id']} :: failure_stage={case['failure_stage']}")
        lines.append(f"> {case['question']}")
        lines.append("")
        for key in (
            "gold_source_count", "degenerate_single_point",
            "old_evidence_ids", "new_evidence_ids", "unassigned_evidence_ids",
            "old_extracted_ids", "new_extracted_ids", "both_sides_extracted",
            "old_cell_ids", "new_cell_ids", "shared_cell_ids",
            "old_state_version_id", "new_state_version_id", "new_state_operation",
            "trigger_coverage", "provenance_completeness",
        ):
            if key in case:
                lines.append(f"- {key}: {case[key]}")
        if case.get("note"):
            lines.append(f"- note: {case['note']}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Part B: Q6 scope REVIEW SHEET (no auto verdicts)
# --------------------------------------------------------------------------

def build_q6_case(
    row: dict[str, Any], cells: list[MemoryCell], extracted_ids: set[str],
    group_cells: list[MemoryCell], group_vectors: list[Any], question_vector: Any,
) -> dict[str, Any]:
    anchors = _json(row["evidence_anchors_json"], [])
    gold_ids = {a["turn_id"] for a in anchors if a.get("turn_id")}
    gold_source_count = len(gold_ids)

    exception_cells = [cell for cell in cells if _cell_source_ids(cell) & gold_ids] if gold_source_count else []
    exception_coverage = (len(gold_ids & extracted_ids) / gold_source_count) if gold_source_count else 0.0

    candidate_exception_cells = [
        {
            "cell_id": cell.cell_id, "topic_key": cell.key.topic_key, "scope_type": cell.key.scope_type,
            "subject": cell.key.subject, "facet": cell.key.facet, "text": _rep_text(cell),
            "sources": sorted(_cell_source_ids(cell)), "matches_gold_ids": sorted(_cell_source_ids(cell) & gold_ids),
        }
        for cell in exception_cells
    ]

    ranked_group = sorted(
        ((cell, _cosine(question_vector, vec)) for cell, vec in zip(group_cells, group_vectors)),
        key=lambda pair: pair[1], reverse=True,
    )[:5]
    candidate_group_cells = [
        {
            "cell_id": cell.cell_id, "topic_key": cell.key.topic_key, "scope_type": cell.key.scope_type,
            "subject": cell.key.subject, "facet": cell.key.facet, "text": _rep_text(cell),
            "sources": sorted(_cell_source_ids(cell)), "similarity_to_question": round(sim, 3),
        }
        for cell, sim in ranked_group
    ]

    return {
        "qa_id": row["qa_id"], "network_id": row["network_id"], "question_norm": row["question"],
        "gold_source_count": gold_source_count,
        "computed_context": {
            "note": "informational only, NOT a verdict -- see manual fields below",
            "individual_exception_extracted": bool(exception_cells),
            "exception_coverage": round(exception_coverage, 3),
        },
        "candidate_exception_cells": candidate_exception_cells,
        "candidate_group_cells": candidate_group_cells,
        "manual": {
            "group_norm_semantic_match": "", "exception_semantic_match": "",
            "scopes_correct": "", "provenance_sufficient": "", "failure_stage": "",
        },
    }


def render_q6_report(cases: list[dict[str, Any]]) -> str:
    lines = [
        "# Q6 (norm_vs_individual) scope REVIEW SHEET -- 20/20 cases, manual review required, no auto verdicts",
        "",
        "Nothing here is auto-classified as correct. candidate_group_cells is a RANKING aid only -- cosine "
        "similarity between the question text and each group/subgroup-scope cell's assertion text, both via "
        "the same local embedding model already used elsewhere in this pipeline (no LLM, no new calls) -- it "
        "surfaces plausible candidates for a human to read and judge, it does NOT declare the norm found (the "
        "previous version of this audit did exactly that from facet overlap alone; removed). "
        "candidate_exception_cells is objectively computed (cells whose source_turn_ids intersect the QA gold "
        "evidence anchors) -- 'objective' means 'these cells cover the gold evidence', not 'correctly scoped' "
        "or 'not conflated', which still needs the manual fields below.",
        "",
        "## All 20 cases (fill in the 5 manual fields per case)",
        "",
    ]
    for i, case in enumerate(cases, 1):
        ctx = case["computed_context"]
        lines.append(f"### {i}. {case['qa_id']} :: {case['network_id']}")
        lines.append(f"> {case['question_norm']}")
        lines.append("")
        lines.append(f"- gold_source_count: {case['gold_source_count']}")
        lines.append(
            f"- computed (informational, not a verdict): "
            f"individual_exception_extracted={ctx['individual_exception_extracted']}, "
            f"exception_coverage={ctx['exception_coverage']}"
        )
        lines.append("")
        lines.append("**candidate_exception_cells:**")
        if case["candidate_exception_cells"]:
            for cand in case["candidate_exception_cells"]:
                lines.append(
                    f"- [{cand['topic_key']}] scope={cand['scope_type']} subject={cand['subject']} "
                    f"facet={cand['facet']} sources={cand['sources']} matches_gold={cand['matches_gold_ids']}"
                )
                lines.append(f"  \"{cand['text']}\"")
        else:
            lines.append("- (none found)")
        lines.append("")
        lines.append("**candidate_group_cells** (ranked by embedding similarity to the question, top 5, NOT a verdict):")
        if case["candidate_group_cells"]:
            for cand in case["candidate_group_cells"]:
                lines.append(
                    f"- [{cand['topic_key']}] scope={cand['scope_type']} subject={cand['subject']} "
                    f"facet={cand['facet']} sim={cand['similarity_to_question']} sources={cand['sources']}"
                )
                lines.append(f"  \"{cand['text']}\"")
        else:
            lines.append("- (no group/subgroup-scope cells exist in this network)")
        lines.append("")
        lines.append("Manual review (leave blank until reviewed):")
        lines.append("- group_norm_semantic_match: [ ] yes  [ ] no  [ ] unclear")
        lines.append("- exception_semantic_match: [ ] yes  [ ] no  [ ] unclear")
        lines.append("- scopes_correct: [ ] yes  [ ] no")
        lines.append("- provenance_sufficient: [ ] yes  [ ] no")
        lines.append("- failure_stage: ______________________")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Part C: false-split diagnostic (no QA) -- unchanged from the first pass
# --------------------------------------------------------------------------

def find_false_split_candidates(
    registries: dict[str, dict[CellKey, MemoryCell]], limit: int = 100,
) -> list[dict[str, Any]]:
    buckets: dict[tuple, list[MemoryCell]] = defaultdict(list)
    for network_id, registry in registries.items():
        for cell in registry.values():
            bucket_key = (network_id, cell.key.viewpoint_owner, cell.key.subject, cell.key.facet, cell.key.scope_type)
            buckets[bucket_key].append(cell)

    multi_buckets = {key: cells for key, cells in buckets.items() if len(cells) > 1}
    all_topic_keys = sorted({cell.key.topic_key for cells in multi_buckets.values() for cell in cells})
    if not all_topic_keys:
        return []
    vectors = embed_batch(all_topic_keys)
    embedding_by_topic = dict(zip(all_topic_keys, vectors))

    candidates = []
    for cells in multi_buckets.values():
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                cell_a, cell_b = cells[i], cells[j]
                if not cell_a.state_versions or not cell_b.state_versions:
                    continue
                sim = _cosine(
                    embedding_by_topic[cell_a.key.topic_key], embedding_by_topic[cell_b.key.topic_key],
                )
                rep_a, rep_b = cell_a.state_versions[0], cell_b.state_versions[0]
                shared_sources = set(rep_a.source_turn_ids) & set(rep_b.source_turn_ids)
                overlap = _temporal_windows_overlap(
                    rep_a.effective_from, rep_a.effective_to, rep_b.effective_from, rep_b.effective_to,
                )
                if shared_sources:
                    suggested = "duplicate"
                elif not overlap:
                    suggested = "separate"
                elif rep_a.normalized_value is None or rep_b.normalized_value is None:
                    suggested = "augment"
                elif _norm(rep_a.normalized_value) != _norm(rep_b.normalized_value):
                    suggested = "revise"
                else:
                    suggested = "reaffirm"
                candidates.append({
                    "network_id": cell_a.key.network_id, "viewpoint_owner": cell_a.key.viewpoint_owner,
                    "subject": cell_a.key.subject, "facet": cell_a.key.facet, "scope_type": cell_a.key.scope_type,
                    "topic_key_a": cell_a.key.topic_key, "topic_key_b": cell_b.key.topic_key,
                    "cell_id_a": cell_a.cell_id, "cell_id_b": cell_b.cell_id,
                    "text_a": rep_a.assertion_text, "text_b": rep_b.assertion_text,
                    "date_a": rep_a.effective_from, "date_b": rep_b.effective_from,
                    "sources_a": list(rep_a.source_turn_ids), "sources_b": list(rep_b.source_turn_ids),
                    "cosine_similarity": round(sim, 4), "suggested_relation": suggested,
                })
    candidates.sort(key=lambda c: c["cosine_similarity"], reverse=True)
    return candidates[:limit]


def render_false_split_report(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "# False-split diagnostic (no QA), top candidates -- deterministic scan only, NO auto-merge",
        "",
        f"Candidates: {len(candidates)} (pairs of DIFFERENT topic_key cells sharing network/viewpoint_owner/"
        "subject/facet/scope_type, ranked by topic_key embedding cosine similarity). Every pair listed here "
        "was already BELOW resolve_topic_slot's merge threshold at materialization time -- that is precisely "
        "why each is a candidate false split, not an already-applied merge.",
        "", "| # | network | who | facet/scope | topic_key A | topic_key B | cos | suggested | decision |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(candidates, 1):
        who = f"{c['viewpoint_owner']} -> {c['subject']}"
        facet_scope = f"{c['facet']} ({c['scope_type']})"
        lines.append(
            f"| {i} | {c['network_id']} | {who} | {facet_scope} | {c['topic_key_a']} | {c['topic_key_b']} | "
            f"{c['cosine_similarity']} | {c['suggested_relation']} | ☐ |"
        )
    lines += ["", "## Detail", ""]
    for i, c in enumerate(candidates, 1):
        lines.append(f"### {i}. {c['network_id']} :: {c['viewpoint_owner']} -> {c['subject']} :: {c['facet']} ({c['scope_type']})")
        lines.append(f"- A [{c['topic_key_a']}] {c['date_a']} sources={c['sources_a']}")
        lines.append(f"  \"{c['text_a']}\"")
        lines.append(f"- B [{c['topic_key_b']}] {c['date_b']} sources={c['sources_b']}")
        lines.append(f"  \"{c['text_b']}\"")
        lines.append(f"- cosine_similarity: {c['cosine_similarity']}, suggested_relation: {c['suggested_relation']}")
        lines.append("- Decision: [ ] separate  [ ] reaffirm  [ ] augment  [ ] revise  [ ] duplicate")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Data protocol: dev/diagnostic set vs held-out test set
# --------------------------------------------------------------------------

def write_data_protocol(
    qa, selected: list[dict[str, Any]], selected_networks: set[str],
    cache: dict[tuple[str, str], dict], slot_resolution: dict[tuple[str, str, int], dict],
) -> None:
    all_networks = sorted(qa.network_id.astype(str).unique().tolist())
    heldout_networks = sorted(n for n in all_networks if n not in selected_networks)

    dev_manifest = sorted(
        ({"qa_id": row["qa_id"], "network_id": row["network_id"], "query_type": row["query_type"]} for row in selected),
        key=lambda r: (r["network_id"], r["qa_id"]),
    )
    DEV_DIAGNOSTIC_MANIFEST.write_text(json.dumps({
        "role": "development/diagnostic set -- used for the Stage 4.5 A/B/C audits above, frozen",
        "seed": SEED, "per_type": PER_TYPE, "networks": sorted(selected_networks),
        "qa_count": len(dev_manifest), "qa": dev_manifest,
    }, indent=2, ensure_ascii=False))

    heldout_rows = qa[qa.network_id.astype(str).isin(heldout_networks)]
    heldout_manifest = sorted(
        ({"qa_id": str(r.qa_id), "network_id": str(r.network_id), "query_type": str(r.query_type)}
         for r in heldout_rows.itertuples(index=False)),
        key=lambda r: (r["network_id"], r["qa_id"]),
    )
    HELDOUT_MANIFEST.write_text(json.dumps({
        "role": "held-out test set -- identifiers ONLY; question/answer/evidence_anchors_json/correct_option "
                "were NOT read or stored anywhere in this pipeline",
        "networks": heldout_networks, "qa_count": len(heldout_manifest), "qa": heldout_manifest,
    }, indent=2, ensure_ascii=False))

    cached_sessions_touching_heldout = sorted({
        (str(net), str(sess)) for (net, sess) in cache if str(net) in heldout_networks
    })
    slot_entries_touching_heldout = sum(1 for key in slot_resolution if str(key[0]) in heldout_networks)

    lines = [
        "# Stage 4.5 data protocol",
        "",
        f"- Development/diagnostic set: {len(dev_manifest)} QA across {len(selected_networks)} networks "
        f"(seed={SEED}), frozen. Used for the A/B/C audits. Manifest: {DEV_DIAGNOSTIC_MANIFEST}",
        f"- Held-out test set: {len(heldout_manifest)} QA across {len(heldout_networks)} networks "
        f"({', '.join(heldout_networks)}). question/answer/evidence_anchors_json/correct_option for these rows "
        f"were NOT read or stored by this run -- only qa_id/network_id/query_type identifiers, in {HELDOUT_MANIFEST}. "
        "No eval ran against this set.",
        "",
        "## Ingestion-cache fact about the held-out networks (documented honestly, NOT called leakage)",
        "",
        f"An earlier session accidentally ran extraction against the full conversations table before scoping to "
        f"the 37 selected networks (the Stage 4 'accidental unscoped extraction run' incident). As a result, "
        f"{len(cached_sessions_touching_heldout)} sessions across the held-out networks are ALREADY present in "
        f"the raw extraction cache ({CELL_CACHE}), and {slot_entries_touching_heldout} slot-resolution cache "
        "entries also touch held-out networks -- conversations from those networks were processed by the "
        "extractor at some point. This is NOT QA leakage: no QA question, answer, evidence anchor, or "
        "correct_option from these networks has been read by any part of this pipeline, including this Stage "
        "4.5 run -- nothing was tuned against their labels, and this run triggered NO new extraction for them "
        "(they are simply outside the `conversations` slice passed to build_cell_assertion_cache here).",
        "",
        "## Sessions from held-out networks already in the cache",
        "",
    ]
    for net, sess in cached_sessions_touching_heldout:
        lines.append(f"- {net} / {sess}")
    DATA_PROTOCOL_MD.write_text("\n".join(lines))
    print(
        f"Data protocol: {DATA_PROTOCOL_MD} (dev manifest {DEV_DIAGNOSTIC_MANIFEST}, "
        f"held-out manifest {HELDOUT_MANIFEST})", file=sys.stderr,
    )


def main() -> None:
    conversations, qa, _personas = load_data(DATA_DIR)
    selected = _select_questions(qa, PER_TYPE, SEED)
    selected_networks = {row["network_id"] for row in selected}
    scoped_conversations = conversations[conversations.network_id.isin(selected_networks)]

    cache = build_cell_assertion_cache(scoped_conversations, CELL_CACHE)
    slot_resolution = build_slot_resolution_cache(cache, SLOT_CACHE, selected_networks)
    registries, _stats, _merge_events, validated_assertions = materialize_cell_registries(
        scoped_conversations, cache, slot_resolution,
    )

    extracted_ids_by_network = {
        network_id: {source_id for a in assertions for source_id in a.source_turn_ids}
        for network_id, assertions in validated_assertions.items()
    }

    q8_rows = [row for row in selected if row["query_type"] == "Q8"]
    q6_rows = [row for row in selected if row["query_type"] == "Q6"]
    assert len(q8_rows) == 20, f"expected 20 Q8 cases, got {len(q8_rows)}"
    assert len(q6_rows) == 20, f"expected 20 Q6 cases, got {len(q6_rows)}"

    q8_cases = [
        audit_q8_case(
            row, list(registries.get(row["network_id"], {}).values()),
            extracted_ids_by_network.get(row["network_id"], set()),
        )
        for row in q8_rows
    ]

    group_cells_by_network: dict[str, list[MemoryCell]] = defaultdict(list)
    for network_id, registry in registries.items():
        for cell in registry.values():
            if cell.key.scope_type in ("group", "subgroup"):
                group_cells_by_network[network_id].append(cell)
    all_group_cells = [cell for cells in group_cells_by_network.values() for cell in cells]
    group_texts = [_rep_text(cell) for cell in all_group_cells]
    group_vectors_flat = embed_batch(group_texts) if group_texts else []
    group_vectors_by_cell_id = dict(zip((c.cell_id for c in all_group_cells), group_vectors_flat))

    q6_cases = []
    for row in q6_rows:
        network_id = row["network_id"]
        cells = list(registries.get(network_id, {}).values())
        network_group_cells = group_cells_by_network.get(network_id, [])
        network_group_vectors = [group_vectors_by_cell_id[c.cell_id] for c in network_group_cells]
        question_vector = embed(row["question"])
        q6_cases.append(build_q6_case(
            row, cells, extracted_ids_by_network.get(network_id, set()),
            network_group_cells, network_group_vectors, question_vector,
        ))

    Q8_AUDIT_MD.write_text(render_q8_report(q8_cases))
    with Q8_AUDIT_JSONL.open("w") as f:
        for case in q8_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    Q6_AUDIT_MD.write_text(render_q6_report(q6_cases))
    with Q6_AUDIT_JSONL.open("w") as f:
        for case in q6_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    candidates = find_false_split_candidates(registries, limit=100)
    FALSE_SPLIT_MD.write_text(render_false_split_report(candidates))

    write_data_protocol(qa, selected, selected_networks, cache, slot_resolution)

    print(f"Q8 audit: {Q8_AUDIT_MD}, {Q8_AUDIT_JSONL}", file=sys.stderr)
    print(f"Q6 review sheet: {Q6_AUDIT_MD}, {Q6_AUDIT_JSONL}", file=sys.stderr)
    print(f"False-split review ({len(candidates)} candidates): {FALSE_SPLIT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
