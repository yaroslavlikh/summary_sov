"""Stage 4.6 -- oracle-ingestion component diagnostic: isolates STATE
LINKING from extraction, on the existing 20 Q8 dev cases ONLY.

Stage 4.5 measured the full pipeline (extraction -> slot resolution ->
linking) and found extraction itself failing on most Q8 cases (only 4/20
had both old and new sides extracted at all). That confound makes it
impossible to tell whether the LINKING layer (decide_state_operation /
resolve_topic_slot) is the problem or extraction is. This module removes
the confound: for each Q8 case, it force-extracts assertions from ONLY the
gold old-evidence turns and ONLY the gold new-evidence turns, in two
SEPARATE, independent LLM calls that never see each other or the QA
question/answer/relevance/expected relation -- then hands those (near-
oracle, guaranteed-present) assertions to two competing linkers:

  1. The EXISTING threshold-based linker (resolve_topic_slot +
     decide_state_operation from research/versioned_memory_cells.py),
     UNCHANGED -- not modified by this module at all.
  2. A NEW, separate experimental linker: deterministic candidate filter
     (same network/viewpoint_owner/subject/scope_type/facet) -> embedding
     top-3 retrieval -> a forced-JSON semantic resolver that chooses
     NEW_CELL/OBSERVE/AUGMENT/REVISE/RETRACT. This linker is implemented
     ENTIRELY in this module (apply_assertion_semantic below) -- it does
     NOT touch or reuse decide_state_operation/apply_assertion's mutation
     code, so the existing threshold linker in versioned_memory_cells.py
     is completely unmodified by this experiment.

Neither the oracle extractor nor the semantic resolver is ever shown QA
question/answer/correct_option/evidence-anchor relevance text or the
expected relation -- by construction, since prompts here are built only
from raw conversation turns and from other memory cells' own fields.

Also measures both linkers' FALSE-MERGE behavior on the corpus-wide,
non-QA false-split candidate pairs already produced by Stage 4.5's Part C
(research/stage4_5_audit.py's find_false_split_candidates) -- these pairs
were flagged as "maybe the same topic, needs human review", so a linker
that merges them without a human deciding first is a real risk, especially
for the new semantic resolver which is less conservative by design than
the existing 0.86-cosine threshold.

Every new LLM output (oracle extraction, semantic resolver) is cached to
disk, content-addressed where order-independence matters, so a rerun of
evaluation/reporting costs zero new calls.

Run:
    python3 -m research.stage4_6_oracle_linking
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings import embed
from research.socialmembench_pilot import (
    _json, _norm, _parse_json_object, _select_questions, build_cell_assertion_cache,
    build_slot_resolution_cache, load_data, materialize_cell_registries,
)
from research.stage4_5_audit import (
    CELL_CACHE, DATA_DIR, PER_TYPE, SEED, SLOT_CACHE,
    _cell_source_ids, _rep_text, _split_old_new, find_false_split_candidates,
)
from research.versioned_memory_cells import (
    MODALITIES, SCOPE_TYPES, Assertion, CellKey, MemoryCell, MemoryObservation,
    MemoryStateVersion, apply_assertion, resolve_topic_slot, validate_assertion, _cell_id, _cosine,
)

ORACLE_EXTRACTION_CACHE = Path("/tmp/socialmembench_stage4_6_oracle_extraction.jsonl")
SEMANTIC_RESOLVER_CACHE = Path("/tmp/socialmembench_stage4_6_semantic_resolver.jsonl")
RESULTS_JSONL = Path("/tmp/socialmembench_stage4_6_results.jsonl")
REPORT_MD = Path("/tmp/socialmembench_stage4_6_report.md")
DECISIONS_MD = Path("/tmp/socialmembench_stage4_6_decisions_review.md")

SEMANTIC_DECISIONS = ("NEW_CELL", "OBSERVE", "AUGMENT", "REVISE", "RETRACT")
# Self-contained safety net for THIS experimental linker only -- not a
# change to any production/existing threshold (TOPIC_MATCH_THRESHOLD,
# REVISE_CONFIDENCE_THRESHOLD in versioned_memory_cells.py are untouched).
SEMANTIC_CONFIDENCE_FLOOR = 0.5


# --------------------------------------------------------------------------
# 1-3. Independent oracle extraction on ONLY the old/new gold turns
# --------------------------------------------------------------------------

def _oracle_extract_prompt(turn_lines: str) -> str:
    """Deliberately shown NOTHING but the raw turns below -- no question, no
    answer, no evidence-anchor relevance text, no hint of what relation (if
    any) is being tested. Mirrors socialmembench_pilot._cell_extract_prompt's
    schema so the output is directly compatible with validate_assertion."""
    return f"""You will see a SMALL, isolated set of chat turns -- not a full
conversation, just the turns listed below. You do not know why these turns
were selected, what question (if any) they might answer, or what happened in
any other turn or session. Extract atomic state assertions from ONLY what is
written below.

Extract 1-4 atomic assertions. For each:
- viewpoint_owner: whose view/knowledge/state this is.
- subject: the person, group, or decision the assertion is actually about.
- facet: a short normalized aspect (e.g. location, occupation, availability,
  attitude, preference, relationship, group_decision, group_norm, commitment).
- scope_type: one of {sorted(SCOPE_TYPES)}.
- topic_key: a short, normalized slug for the SPECIFIC state/topic (not just
  the facet) -- if in doubt, more specific rather than broader.
- assertion_text: the durable claim, preserving attribution if this is an
  opinion or a report of someone else's words.
- normalized_value: a short stable label for the state/value if there is one,
  else null.
- modality: one of {sorted(MODALITIES)}.
- confidence: 0.0-1.0.
- source_turn_ids: real turn_id values copied EXACTLY from the turns below.
- effective_from / effective_to: ISO date if given, else "" / null.
- temporal_scope: one of ongoing, dated_event, recurring, unspecified.
- temporal_precision: one of exact, approximate, unknown.

Return JSON only:
{{"items":[{{"viewpoint_owner":"...","subject":"...","facet":"...","topic_key":"...",
"scope_type":"...","assertion_text":"...","normalized_value":"...","modality":"...",
"confidence":0.0,"source_turn_ids":["turn_id"],"effective_from":"","effective_to":"",
"temporal_scope":"unspecified","temporal_precision":"unknown"}}]}}

Turns:
{turn_lines}"""


def _extract_side(turns_frame) -> dict[str, Any]:
    from llm.groq_client import get_chat_model

    lines = [
        f"[[{row.turn_id}]] {row.timestamp} {row.speaker_display_name}: {row.message}"
        for row in turns_frame.itertuples(index=False)
    ]
    prompt = _oracle_extract_prompt(chr(10).join(lines))
    message = get_chat_model("fast", 0).invoke(prompt)
    parsed = _parse_json_object(message.content if isinstance(message.content, str) else "") or {}
    return {"raw_items": [item for item in parsed.get("items", []) if isinstance(item, dict)]}


def build_oracle_extraction_cache(cases: list[dict], conversations) -> dict[tuple[str, str], dict]:
    cached: dict[tuple[str, str], dict] = {}
    if ORACLE_EXTRACTION_CACHE.exists():
        for line in ORACLE_EXTRACTION_CACHE.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cached[(row["qa_id"], row["side"])] = row

    pending = [
        (case, side) for case in cases for side in ("old", "new")
        if case[f"{side}_ids"] and (case["qa_id"], side) not in cached
    ]
    if pending:
        print(f"Oracle-extracting {len(pending)} old/new sides for Q8 dev cases...", file=sys.stderr)
        ORACLE_EXTRACTION_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with ORACLE_EXTRACTION_CACHE.open("a") as out:
            for done, (case, side) in enumerate(pending, 1):
                ids = case[f"{side}_ids"]
                frame = conversations[
                    (conversations.network_id == case["network_id"]) & (conversations.turn_id.isin(ids))
                ].sort_values("timestamp")
                result = _extract_side(frame)
                row = {"qa_id": case["qa_id"], "side": side, "network_id": case["network_id"], **result}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                cached[(case["qa_id"], side)] = row
                if done % 10 == 0 or done == len(pending):
                    print(f"  extracted {done}/{len(pending)}", file=sys.stderr)
    return cached


def _validated_side_assertions(
    qa_id: str, side: str, network_id: str, ids: set[str], raw_items: list[dict], conversations,
) -> list[Assertion]:
    """valid_source_ids is restricted to EXACTLY the turns shown for this
    side -- the model could only have truthfully cited what it was shown,
    so a citation to any other (even real) turn_id is fabricated here and
    dropped by validate_assertion."""
    if not ids:
        return []
    frame = conversations[(conversations.network_id == network_id) & (conversations.turn_id.isin(ids))]
    valid_source_ids = {str(t) for t in frame.turn_id}
    timestamp_by_turn = {str(row.turn_id): str(row.timestamp) for row in frame.itertuples(index=False)}
    assertions = []
    for raw in raw_items:
        item = dict(raw)
        item["conversation_id"] = f"{qa_id}:{side}"
        cited = [str(s) for s in item.get("source_turn_ids", []) if str(s) in valid_source_ids]
        item["source_turn_ids"] = cited
        item["observed_at"] = max((timestamp_by_turn[s] for s in cited), default="")
        item["effective_from"] = item.get("effective_from") or item["observed_at"]
        assertion = validate_assertion(item, valid_source_ids, network_id)
        if assertion is not None:
            assertions.append(assertion)
    return assertions


# --------------------------------------------------------------------------
# 4. Existing threshold-based linker (UNCHANGED, reused as-is)
# --------------------------------------------------------------------------

def run_threshold_linker(old_assertions: list[Assertion], new_assertions: list[Assertion]) -> dict[CellKey, MemoryCell]:
    registry: dict[CellKey, MemoryCell] = {}
    topic_embeddings: dict[tuple, list[tuple[str, Any]]] = defaultdict(list)
    for assertion in sorted(old_assertions + new_assertions, key=lambda a: a.observed_at):
        bucket_key = (
            assertion.network_id, _norm(assertion.viewpoint_owner), _norm(assertion.subject),
            _norm(assertion.facet), assertion.scope_type,
        )
        candidates = topic_embeddings[bucket_key]
        new_embedding = embed(assertion.topic_key)
        resolved_topic = resolve_topic_slot(assertion.topic_key, new_embedding, candidates)
        if resolved_topic is None:
            resolved_topic = assertion.topic_key
            topic_embeddings[bucket_key].append((assertion.topic_key, new_embedding))
        if resolved_topic != assertion.topic_key:
            assertion = _dc_replace(assertion, topic_key=resolved_topic)
        apply_assertion(registry, assertion, recorded_at=assertion.observed_at)
    return registry


# --------------------------------------------------------------------------
# 5-6. NEW experimental semantic linker -- separate from the threshold one
# --------------------------------------------------------------------------

def _semantic_candidate_filter(registry: dict[CellKey, MemoryCell], assertion: Assertion) -> list[MemoryCell]:
    """Deterministic candidate filter: same network, viewpoint_owner,
    subject, scope_type, and facet (normalized exact match -- "compatible
    facet" is implemented as normalized-equal; no facet-synonym table
    exists in this codebase, so exact match after casefold/strip is the
    honest, defensible reading, documented here rather than silently
    assumed)."""
    return [
        cell for cell in registry.values()
        if cell.key.network_id == assertion.network_id
        and cell.key.viewpoint_owner == _norm(assertion.viewpoint_owner)
        and cell.key.subject == _norm(assertion.subject)
        and cell.key.scope_type == assertion.scope_type
        and cell.key.facet == _norm(assertion.facet)
    ]


def _top3_by_embedding(assertion: Assertion, candidates: list[MemoryCell]) -> list[MemoryCell]:
    if not candidates:
        return []
    query_vec = embed(assertion.assertion_text)
    scored = [(cell, _cosine(query_vec, embed(_rep_text(cell)))) for cell in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [cell for cell, _ in scored[:3]]


def _semantic_resolver_prompt(assertion_desc: dict, candidates_desc: list[dict]) -> str:
    lines = []
    for letter, cand in zip("ABC", candidates_desc):
        lines.append(
            f"[{letter}] topic_key={cand['topic_key']!r} current_value={cand['normalized_value']!r} "
            f"modality={cand['modality']} temporal_scope={cand['temporal_scope']} "
            f"effective_from={cand['effective_from']!r} effective_to={cand['effective_to']!r}\n"
            f"    \"{cand['text']}\""
        )
    candidates_block = "\n".join(lines) if lines else "(no existing candidate cells)"
    return f"""You are deciding how a NEW memory assertion about someone's state relates
to EXISTING memory cells about the same person/subject/facet/scope. You do not know
what question (if any) this relates to, and you must not guess one -- decide only
from the content below.

NEW assertion:
  text: "{assertion_desc['text']}"
  normalized_value: {assertion_desc['normalized_value']!r}
  modality: {assertion_desc['modality']}
  confidence: {assertion_desc['confidence']}
  temporal_scope: {assertion_desc['temporal_scope']}
  effective_from: {assertion_desc['effective_from']!r}
  effective_to: {assertion_desc['effective_to']!r}

Existing candidate cells (top 3 by similarity, or fewer):
{candidates_block}

Choose exactly one decision:
- NEW_CELL: this is genuinely a different topic/state from every candidate.
- OBSERVE: this repeats/reaffirms a candidate's CURRENT state (same real-world value).
- AUGMENT: this adds a compatible detail to a candidate's current state WITHOUT
  replacing it (the underlying state did not actually change).
- REVISE: the underlying real-world state actually changed from a candidate's
  current value to something incompatible, OR the text explicitly signals a
  change (e.g. "no longer", "instead", "used to X but now Y", "changed to").
  IMPORTANT: two differently-WORDED values are NOT by themselves evidence of
  REVISE -- only choose REVISE if the state itself is genuinely different, not
  merely described differently.
- RETRACT: a candidate's current state is explicitly withdrawn/no longer holds,
  with nothing new asserted in its place.

If genuinely uncertain, choose NEW_CELL -- a missed link is recoverable, a false
merge is not.

Return JSON only:
{{"decision":"NEW_CELL|OBSERVE|AUGMENT|REVISE|RETRACT","target":"A"|"B"|"C"|null,
"confidence":0.0,"reason":"one short sentence"}}"""


def _parse_semantic_decision(parsed: dict, num_candidates: int) -> tuple[str, Optional[int], float, str, bool]:
    """Pure parsing/validation logic, separated out so it's unit-testable
    without an LLM call. Returns (decision, target_index_or_None,
    confidence, reason, defaulted)."""
    decision = str(parsed.get("decision") or "").strip().upper()
    target_letter = parsed.get("target")
    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(parsed.get("reason") or "").strip()

    target_index = None
    if target_letter in ("A", "B", "C"):
        index = "ABC".index(target_letter)
        if index < num_candidates:
            target_index = index

    defaulted = False
    if decision not in SEMANTIC_DECISIONS:
        decision, target_index, defaulted = "NEW_CELL", None, True
    elif decision != "NEW_CELL" and target_index is None:
        decision, defaulted = "NEW_CELL", True
    elif confidence < SEMANTIC_CONFIDENCE_FLOOR:
        decision, target_index, defaulted = "NEW_CELL", None, True
    return decision, target_index, confidence, reason, defaulted


_SEMANTIC_CACHE: Optional[dict[str, dict]] = None


def _load_semantic_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if SEMANTIC_RESOLVER_CACHE.exists():
        for line in SEMANTIC_RESOLVER_CACHE.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["prompt_hash"]] = row
    return cache


def run_semantic_resolver(assertion: Assertion, candidates: list[MemoryCell]) -> dict[str, Any]:
    global _SEMANTIC_CACHE
    if not candidates:
        return {
            "decision": "NEW_CELL", "target_cell_id": None, "confidence": 1.0,
            "reason": "no candidate cells", "defaulted": False, "candidates_seen": [],
        }
    if _SEMANTIC_CACHE is None:
        _SEMANTIC_CACHE = _load_semantic_cache()

    assertion_desc = {
        "text": assertion.assertion_text, "normalized_value": assertion.normalized_value,
        "modality": assertion.modality, "confidence": assertion.confidence,
        "temporal_scope": assertion.temporal_scope, "effective_from": assertion.effective_from,
        "effective_to": assertion.effective_to,
    }
    candidates_desc = [
        {
            "topic_key": cell.key.topic_key,
            "normalized_value": (cell.active_states[0].normalized_value if cell.active_states else
                                  (cell.state_versions[-1].normalized_value if cell.state_versions else None)),
            "modality": (cell.active_states[0].modality if cell.active_states else
                         (cell.state_versions[-1].modality if cell.state_versions else "")),
            "temporal_scope": (cell.active_states[0].temporal_scope if cell.active_states else
                                (cell.state_versions[-1].temporal_scope if cell.state_versions else "")),
            "effective_from": (cell.active_states[0].effective_from if cell.active_states else
                                (cell.state_versions[-1].effective_from if cell.state_versions else "")),
            "effective_to": (cell.active_states[0].effective_to if cell.active_states else
                              (cell.state_versions[-1].effective_to if cell.state_versions else None)),
            "text": _rep_text(cell),
        }
        for cell in candidates
    ]
    prompt = _semantic_resolver_prompt(assertion_desc, candidates_desc)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

    if prompt_hash in _SEMANTIC_CACHE:
        parsed = _SEMANTIC_CACHE[prompt_hash]["parsed"]
    else:
        from llm.groq_client import get_chat_model

        message = get_chat_model("fast", 0).invoke(prompt)
        raw_response = message.content if isinstance(message.content, str) else ""
        parsed = _parse_json_object(raw_response) or {}
        row = {"prompt_hash": prompt_hash, "prompt": prompt, "raw_response": raw_response, "parsed": parsed}
        SEMANTIC_RESOLVER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with SEMANTIC_RESOLVER_CACHE.open("a") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        _SEMANTIC_CACHE[prompt_hash] = row

    decision, target_index, confidence, reason, defaulted = _parse_semantic_decision(parsed, len(candidates))
    target_cell = candidates[target_index] if target_index is not None else None
    return {
        "decision": decision, "target_cell_id": (target_cell.cell_id if target_cell else None),
        "confidence": confidence, "reason": reason, "defaulted": defaulted,
        "candidates_seen": [c.cell_id for c in candidates],
    }


def apply_assertion_semantic(
    registry: dict[CellKey, MemoryCell], assertion: Assertion, *, recorded_at: str,
) -> tuple[MemoryStateVersion | MemoryObservation, dict[str, Any]]:
    """The NEW experimental linker's own mutation logic -- deliberately
    NOT sharing code with versioned_memory_cells.apply_assertion, so the
    existing threshold linker stays completely unmodified by this
    experiment."""
    candidates = _top3_by_embedding(assertion, _semantic_candidate_filter(registry, assertion))
    decision_info = run_semantic_resolver(assertion, candidates)
    decision = decision_info["decision"]
    key = CellKey(
        assertion.network_id, _norm(assertion.viewpoint_owner), _norm(assertion.subject),
        _norm(assertion.facet), assertion.topic_key, assertion.scope_type,
    )

    if decision == "NEW_CELL":
        cell = registry.get(key)
        if cell is None:
            cell = MemoryCell(cell_id=_cell_id(key), key=key)
            registry[key] = cell
        version_no = len(cell.state_versions) + 1
        version = MemoryStateVersion(
            version_id=f"{cell.cell_id}:v{version_no}", cell_id=cell.cell_id, version_no=version_no,
            operation="create", assertion_text=assertion.assertion_text, normalized_value=assertion.normalized_value,
            modality=assertion.modality, confidence=assertion.confidence, observed_at=assertion.observed_at,
            effective_from=assertion.effective_from, effective_to=assertion.effective_to,
            temporal_scope=assertion.temporal_scope, temporal_precision=assertion.temporal_precision,
            recorded_at=recorded_at, conversation_id=assertion.conversation_id,
            source_turn_ids=assertion.source_turn_ids,
        )
        cell.state_versions.append(version)
        cell.active_state_version_ids.append(version.version_id)
        return version, decision_info

    target_cell = next(c for c in registry.values() if c.cell_id == decision_info["target_cell_id"])

    if decision in ("OBSERVE", "AUGMENT"):
        target_state_id = target_cell.active_state_version_ids[-1] if target_cell.active_state_version_ids else ""
        observation = MemoryObservation(
            observation_id=f"{target_cell.cell_id}:o{len(target_cell.observations) + 1}",
            cell_id=target_cell.cell_id, state_version_id=target_state_id,
            assertion_text=assertion.assertion_text, modality=assertion.modality, confidence=assertion.confidence,
            observed_at=assertion.observed_at, recorded_at=recorded_at, conversation_id=assertion.conversation_id,
            source_turn_ids=assertion.source_turn_ids, observation_kind=decision.lower(),
        )
        target_cell.observations.append(observation)
        for index, state in enumerate(target_cell.state_versions):
            if state.version_id == target_state_id:
                target_cell.state_versions[index] = _dc_replace(
                    state,
                    source_turn_ids=tuple(dict.fromkeys(state.source_turn_ids + assertion.source_turn_ids)),
                    confidence=max(state.confidence, assertion.confidence),
                )
                break
        return observation, decision_info

    # REVISE / RETRACT -- closes every currently active version of the
    # target cell (a per-case mini registry typically has at most one).
    version_no = len(target_cell.state_versions) + 1
    version = MemoryStateVersion(
        version_id=f"{target_cell.cell_id}:v{version_no}", cell_id=target_cell.cell_id, version_no=version_no,
        operation=decision.lower(), assertion_text=assertion.assertion_text, normalized_value=assertion.normalized_value,
        modality=assertion.modality, confidence=assertion.confidence, observed_at=assertion.observed_at,
        effective_from=assertion.effective_from, effective_to=assertion.effective_to,
        temporal_scope=assertion.temporal_scope, temporal_precision=assertion.temporal_precision,
        recorded_at=recorded_at, conversation_id=assertion.conversation_id, source_turn_ids=assertion.source_turn_ids,
    )
    target_cell.state_versions.append(version)
    for active_id in list(target_cell.active_state_version_ids):
        for index, state in enumerate(target_cell.state_versions):
            if state.version_id == active_id:
                target_cell.state_versions[index] = _dc_replace(
                    state, closed_at=assertion.observed_at, superseded_by=version.version_id,
                )
                break
        target_cell.active_state_version_ids.remove(active_id)
    if decision != "RETRACT":
        target_cell.active_state_version_ids.append(version.version_id)
    return version, decision_info


def run_semantic_linker(
    old_assertions: list[Assertion], new_assertions: list[Assertion],
) -> tuple[dict[CellKey, MemoryCell], list[dict[str, Any]]]:
    registry: dict[CellKey, MemoryCell] = {}
    decisions: list[dict[str, Any]] = []
    for assertion in sorted(old_assertions + new_assertions, key=lambda a: a.observed_at):
        _, decision_info = apply_assertion_semantic(registry, assertion, recorded_at=assertion.observed_at)
        decisions.append({
            "assertion_text": assertion.assertion_text, "source_turn_ids": list(assertion.source_turn_ids),
            **decision_info,
        })
    return registry, decisions


# --------------------------------------------------------------------------
# 7. Evaluation -- co-cell / ordered-chain / revision recall, both linkers
# --------------------------------------------------------------------------

def _evaluate_linked_cells(cells: list[MemoryCell], old_ids: set[str], new_ids: set[str]) -> dict[str, Any]:
    cell_by_id = {cell.cell_id: cell for cell in cells}
    old_cell_ids = {cell.cell_id for cell in cells if _cell_source_ids(cell) & old_ids}
    new_cell_ids = {cell.cell_id for cell in cells if _cell_source_ids(cell) & new_ids}
    shared = old_cell_ids & new_cell_ids
    co_cell_recall = bool(shared) and bool(old_cell_ids) and bool(new_cell_ids)

    ordered_chain_recall = False
    revision_recall = False
    best_new_operation = None
    for cell_id in shared:
        cell = cell_by_id[cell_id]
        old_versions = [v for v in cell.state_versions if set(v.source_turn_ids) & old_ids]
        new_versions = [v for v in cell.state_versions if set(v.source_turn_ids) & new_ids]
        if not old_versions or not new_versions:
            continue
        old_v = min(old_versions, key=lambda v: v.version_no)
        new_v = max(new_versions, key=lambda v: v.version_no)
        if new_v.version_no > old_v.version_no:
            ordered_chain_recall = True
            best_new_operation = new_v.operation
            if new_v.operation in ("revise", "retract"):
                revision_recall = True
                break
    return {
        "co_cell_recall": co_cell_recall, "ordered_chain_recall": ordered_chain_recall,
        "revision_recall": revision_recall, "shared_cell_ids": sorted(shared),
        "new_state_operation": best_new_operation,
    }


def run_case(row: dict[str, Any], conversations, oracle_cache: dict[tuple[str, str], dict]) -> dict[str, Any]:
    anchors = _json(row["evidence_anchors_json"], [])
    old_ids, new_ids, unassigned_ids, degenerate = _split_old_new(anchors)
    network_id = row["network_id"]

    old_raw = oracle_cache.get((row["qa_id"], "old"), {}).get("raw_items", [])
    new_raw = oracle_cache.get((row["qa_id"], "new"), {}).get("raw_items", [])
    old_assertions = _validated_side_assertions(row["qa_id"], "old", network_id, old_ids, old_raw, conversations)
    new_assertions = _validated_side_assertions(row["qa_id"], "new", network_id, new_ids, new_raw, conversations)

    old_extracted_ids = {s for a in old_assertions for s in a.source_turn_ids}
    new_extracted_ids = {s for a in new_assertions for s in a.source_turn_ids}

    result: dict[str, Any] = {
        "qa_id": row["qa_id"], "network_id": network_id, "degenerate_single_point": degenerate,
        "old_evidence_ids": sorted(old_ids), "new_evidence_ids": sorted(new_ids),
        "old_extraction_coverage": (round(len(old_extracted_ids & old_ids) / len(old_ids), 3) if old_ids else 0.0),
        "new_extraction_coverage": (round(len(new_extracted_ids & new_ids) / len(new_ids), 3) if new_ids else 0.0),
        "old_assertions": [a.assertion_text for a in old_assertions],
        "new_assertions": [a.assertion_text for a in new_assertions],
    }

    if degenerate or not old_assertions or not new_assertions:
        empty_eval = {
            "co_cell_recall": False, "ordered_chain_recall": False, "revision_recall": False,
            "shared_cell_ids": [], "new_state_operation": None,
        }
        result["threshold"] = dict(empty_eval)
        result["semantic"] = dict(empty_eval, decisions=[])
        result["note"] = "degenerate single-point case" if degenerate else "extraction missing on at least one side"
        return result

    threshold_registry = run_threshold_linker(old_assertions, new_assertions)
    result["threshold"] = _evaluate_linked_cells(list(threshold_registry.values()), old_ids, new_ids)

    semantic_registry, decisions = run_semantic_linker(old_assertions, new_assertions)
    semantic_eval = _evaluate_linked_cells(list(semantic_registry.values()), old_ids, new_ids)
    semantic_eval["decisions"] = decisions
    result["semantic"] = semantic_eval
    return result


# --------------------------------------------------------------------------
# False-merge check on Stage 4.5 Part C's non-QA candidate pairs
# --------------------------------------------------------------------------

def _cell_to_pseudo_assertion(cell: MemoryCell) -> Assertion:
    rep = cell.state_versions[0] if cell.state_versions else None
    return Assertion(
        network_id=cell.key.network_id, viewpoint_owner=cell.key.viewpoint_owner, subject=cell.key.subject,
        facet=cell.key.facet, topic_key=cell.key.topic_key, scope_type=cell.key.scope_type,
        assertion_text=(rep.assertion_text if rep else _rep_text(cell)),
        modality=(rep.modality if rep else "asserted"), confidence=(rep.confidence if rep else 0.5),
        observed_at=(rep.observed_at if rep else ""), source_turn_ids=(rep.source_turn_ids if rep else ()),
        normalized_value=(rep.normalized_value if rep else None),
        effective_from=(rep.effective_from if rep else ""), effective_to=(rep.effective_to if rep else None),
        temporal_scope=(rep.temporal_scope if rep else "unspecified"),
        temporal_precision=(rep.temporal_precision if rep else "unknown"),
    )


def evaluate_false_merge_pairs(
    registries: dict[str, dict[CellKey, MemoryCell]],
) -> list[dict[str, Any]]:
    cell_by_id = {cell.cell_id: cell for registry in registries.values() for cell in registry.values()}
    candidates = find_false_split_candidates(registries, limit=100)
    rows = []
    for cand in candidates:
        cell_a = cell_by_id.get(cand["cell_id_a"])
        cell_b = cell_by_id.get(cand["cell_id_b"])
        if cell_a is None or cell_b is None:
            continue

        vec_b = embed(cell_b.key.topic_key)
        threshold_merge = resolve_topic_slot(
            cell_b.key.topic_key, vec_b, [(cell_a.key.topic_key, embed(cell_a.key.topic_key))],
        ) is not None

        pseudo_new = _cell_to_pseudo_assertion(cell_b)
        decision_info = run_semantic_resolver(pseudo_new, [cell_a])

        rows.append({
            **cand,
            "threshold_would_merge": threshold_merge,
            "semantic_decision": decision_info["decision"],
            "semantic_confidence": decision_info["confidence"],
            "semantic_reason": decision_info["reason"],
            "semantic_defaulted": decision_info["defaulted"],
            "semantic_would_merge": decision_info["decision"] != "NEW_CELL",
        })
    return rows


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def render_report(cases: list[dict[str, Any]], false_merge_rows: list[dict[str, Any]]) -> str:
    usable = [c for c in cases if not c.get("degenerate_single_point") and c["old_assertions"] and c["new_assertions"]]
    n = len(cases)

    def _count(items, linker, metric):
        return sum(1 for c in items if c[linker].get(metric))

    lines = [
        "# Stage 4.6 -- oracle-ingestion component diagnostic (Q8 dev cases, extraction isolated from linking)",
        "",
        "Extraction (old side and new side, independently, no QA content shown) is the SAME for both linkers "
        "below -- only the LINKING decision differs. `usable` cases are ones where both sides had a non-empty, "
        "non-degenerate oracle extraction (extraction failure is reported separately, not conflated with a "
        "linking failure).",
        "",
        f"- Q8 dev cases: {n}",
        f"- usable (both sides extracted, non-degenerate): {len(usable)}/{n}",
        f"- mean old_extraction_coverage: {sum(c['old_extraction_coverage'] for c in cases) / n:.3f}",
        f"- mean new_extraction_coverage: {sum(c['new_extraction_coverage'] for c in cases) / n:.3f}",
        "",
        "## Paired threshold-vs-semantic table (over the usable cases)",
        "",
        "| metric | threshold (existing) | semantic (new, experimental) |",
        "|---|---|---|",
        f"| co_cell_recall | {_count(usable,'threshold','co_cell_recall')}/{len(usable)} | {_count(usable,'semantic','co_cell_recall')}/{len(usable)} |",
        f"| ordered_chain_recall | {_count(usable,'threshold','ordered_chain_recall')}/{len(usable)} | {_count(usable,'semantic','ordered_chain_recall')}/{len(usable)} |",
        f"| revision_recall | {_count(usable,'threshold','revision_recall')}/{len(usable)} | {_count(usable,'semantic','revision_recall')}/{len(usable)} |",
        "",
        "## False-merge check (Stage 4.5 Part C non-QA candidate pairs, no QA involved)",
        "",
        f"- candidate pairs checked: {len(false_merge_rows)}",
        f"- threshold_would_merge: {sum(1 for r in false_merge_rows if r['threshold_would_merge'])}/{len(false_merge_rows)}",
        f"- semantic_would_merge: {sum(1 for r in false_merge_rows if r['semantic_would_merge'])}/{len(false_merge_rows)}",
        f"- semantic defaulted-to-NEW_CELL (low confidence or malformed output): "
        f"{sum(1 for r in false_merge_rows if r['semantic_defaulted'])}/{len(false_merge_rows)}",
        "", "## All 20 Q8 cases", "",
    ]
    for i, case in enumerate(cases, 1):
        lines.append(f"### {i}. {case['qa_id']} :: {case['network_id']}")
        lines.append(f"- degenerate_single_point: {case['degenerate_single_point']}")
        lines.append(f"- old_extraction_coverage: {case['old_extraction_coverage']}  new_extraction_coverage: {case['new_extraction_coverage']}")
        lines.append(f"- old_assertions: {case['old_assertions']}")
        lines.append(f"- new_assertions: {case['new_assertions']}")
        t, s = case["threshold"], case["semantic"]
        lines.append(
            f"- threshold: co_cell_recall={t['co_cell_recall']} ordered_chain_recall={t['ordered_chain_recall']} "
            f"revision_recall={t['revision_recall']} new_state_operation={t['new_state_operation']}"
        )
        lines.append(
            f"- semantic:  co_cell_recall={s['co_cell_recall']} ordered_chain_recall={s['ordered_chain_recall']} "
            f"revision_recall={s['revision_recall']} new_state_operation={s['new_state_operation']}"
        )
        if case.get("note"):
            lines.append(f"- note: {case['note']}")
        lines.append("")
    return "\n".join(lines)


def render_decisions_review(cases: list[dict[str, Any]], false_merge_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage 4.6 -- decisions review sheet (all merges/revisions, for manual inspection)",
        "",
        "Every semantic-resolver decision that is NOT NEW_CELL, from both the Q8 dev-case linking runs and the "
        "Stage 4.5 Part C false-merge check, with full inputs/provenance. Nothing here has been accepted or "
        "rejected -- this is raw material for manual review.",
        "", "## Q8 dev-case semantic-linker decisions (non-NEW_CELL only)", "",
    ]
    any_q8_merge = False
    for case in cases:
        for decision in case.get("semantic", {}).get("decisions", []):
            if decision["decision"] == "NEW_CELL":
                continue
            any_q8_merge = True
            lines.append(f"### {case['qa_id']} :: {case['network_id']}")
            lines.append(f"- assertion: \"{decision['assertion_text']}\" sources={decision['source_turn_ids']}")
            lines.append(
                f"- decision: {decision['decision']}  target_cell_id={decision['target_cell_id']}  "
                f"confidence={decision['confidence']}  defaulted={decision['defaulted']}"
            )
            lines.append(f"- reason: {decision['reason']}")
            lines.append(f"- candidates_seen: {decision['candidates_seen']}")
            lines.append("- Decision: [ ] correct  [ ] false merge  [ ] unclear")
            lines.append("")
    if not any_q8_merge:
        lines.append("(none -- every Q8 semantic-linker decision was NEW_CELL)")
        lines.append("")

    lines += ["## False-split candidate pairs -- both linkers' verdicts", "",
              "| # | network | who | topic A | topic B | cos | threshold_merge | semantic_decision | conf | defaulted | decision |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, row in enumerate(false_merge_rows, 1):
        who = f"{row['viewpoint_owner']} -> {row['subject']}"
        lines.append(
            f"| {i} | {row['network_id']} | {who} | {row['topic_key_a']} | {row['topic_key_b']} | "
            f"{row['cosine_similarity']} | {row['threshold_would_merge']} | {row['semantic_decision']} | "
            f"{row['semantic_confidence']} | {row['semantic_defaulted']} | ☐ |"
        )
    lines += ["", "## False-split candidate pairs -- detail (only where semantic_would_merge=True)", ""]
    merging = [r for r in false_merge_rows if r["semantic_would_merge"]]
    if not merging:
        lines.append("(none -- the semantic resolver chose NEW_CELL on every false-split candidate pair)")
    for i, row in enumerate(merging, 1):
        lines.append(f"### {i}. {row['network_id']} :: {row['viewpoint_owner']} -> {row['subject']} :: {row['facet']} ({row['scope_type']})")
        lines.append(f"- A [{row['topic_key_a']}] \"{row['text_a']}\"")
        lines.append(f"- B [{row['topic_key_b']}] \"{row['text_b']}\"")
        lines.append(f"- cosine_similarity: {row['cosine_similarity']}  threshold_would_merge: {row['threshold_would_merge']}")
        lines.append(f"- semantic_decision: {row['semantic_decision']}  confidence: {row['semantic_confidence']}  reason: {row['semantic_reason']}")
        lines.append("- Decision: [ ] correct merge  [ ] false merge  [ ] unclear")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    conversations, qa, _personas = load_data(DATA_DIR)
    selected = _select_questions(qa, PER_TYPE, SEED)
    selected_networks = {row["network_id"] for row in selected}
    scoped_conversations = conversations[conversations.network_id.isin(selected_networks)]
    q8_rows = [row for row in selected if row["query_type"] == "Q8"]
    assert len(q8_rows) == 20, f"expected 20 Q8 cases, got {len(q8_rows)}"

    case_meta = []
    for row in q8_rows:
        anchors = _json(row["evidence_anchors_json"], [])
        old_ids, new_ids, _unassigned, _degenerate = _split_old_new(anchors)
        case_meta.append({"qa_id": row["qa_id"], "network_id": row["network_id"], "old_ids": old_ids, "new_ids": new_ids})

    oracle_cache = build_oracle_extraction_cache(case_meta, conversations)

    cases = [run_case(row, conversations, oracle_cache) for row in q8_rows]

    # Reuse the existing (already cached, zero-new-call) corpus materialization
    # for the false-merge check against Stage 4.5 Part C's candidate pairs.
    cell_cache = build_cell_assertion_cache(scoped_conversations, CELL_CACHE)
    slot_resolution = build_slot_resolution_cache(cell_cache, SLOT_CACHE, selected_networks)
    registries, _stats, _merge_events, _validated = materialize_cell_registries(
        scoped_conversations, cell_cache, slot_resolution,
    )
    false_merge_rows = evaluate_false_merge_pairs(registries)

    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL.open("w") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    REPORT_MD.write_text(render_report(cases, false_merge_rows))
    DECISIONS_MD.write_text(render_decisions_review(cases, false_merge_rows))

    print(f"Results: {RESULTS_JSONL}", file=sys.stderr)
    print(f"Report: {REPORT_MD}", file=sys.stderr)
    print(f"Decisions review: {DECISIONS_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
