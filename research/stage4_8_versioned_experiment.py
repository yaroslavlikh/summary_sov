"""Stage 4.8 -- closed oracle-evidence component experiment: RAW vs FLAT vs
VERSIONED, on the frozen 20 Q8 dev cases.

RESULT MODE, not diagnostic mode: this module is built ONCE from the Stage
4.7.1 exact-evidence provenance contract (evidence + related_anchor_ids,
validated quotes, server-derived source_turn_ids -- reused unmodified from
research/stage4_7_1_context_extraction.py) and run to completion. Whatever
the outcome, it is reported as-is; this file is not meant to be iterated on
after inspecting individual question failures.

Methodological label (repeated in the report, not just here): this is an
ORACLE-EVIDENCE COMPONENT experiment. Evidence-anchor regions were selected
from Q8 gold metadata, so every variant is scoped to a small, favorable,
per-question evidence window -- not full-session/network ingestion. It is
NOT honest end-to-end ingestion and its absolute numbers are not comparable
to official SocialMemBench results. What IS a fair, controlled comparison:
all three variants (A/B/C) see the SAME evidence scope per question and go
through the SAME answer/judge calls -- only how that evidence is organized
differs.

  A. RAW     -- the raw conversation turns from every chronological
               evidence-anchor cluster for the question (all clusters,
               including ones Stage 4.5's old/new split used to drop as
               "unassigned"), presented directly with citations. Context
               baseline only, not the primary comparison.
  B. FLAT    -- Stage 4.7.1 STATE/TRANSITION/CAUSE/REACTION records for the
               question, retrieved and exposed with their raw evidence.
               No persistent linking across records.
  C. VERSIONED -- the SAME records, linked by a semantic resolver
               (NEW_CELL/OBSERVE/AUGMENT/REVISE/RETRACT) into persistent
               cells scoped per question, with query-conditioned unfolding
               (current state, prior versions, attached causes/reactions,
               raw provenance).

Primary research question: does C (VERSIONED) beat B (FLAT), i.e. does
explicit versioning add value beyond simply retrieving all extracted
observations. A is context only.

Candidate-filter change from Stage 4.6 (per this stage's explicit
instructions): facet is REMOVED from the hard filter (Stage 4.7.1's schema
does not even extract facet/topic_key/scope_type any more -- re-adding them
here would be exactly the kind of redesign this stage was told not to do).
Hard compatibility is network + resolved viewpoint_owner + resolved
subject; "compatible scope" collapses to a no-op for this dataset (Q8 dev
cases are individual-state narratives; no scope_type exists in the Stage
4.7.1 schema to gate on -- documented, not silently assumed). state_description
and claim text are used only as soft embedding/rerank signals for candidate
ranking.

Held-out QA, production code, versioned_memory_cells.py, and Stage 5 are
untouched by this module.

Run:
    python3 -m research.stage4_8_versioned_experiment
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings import embed
from research.socialmembench_pilot import (
    Candidate, _cited_ids, _json, _norm, _parse_json_object, _select_questions,
    _set_metrics, load_data, render_context,
)
from research.stage4_5_audit import DATA_DIR, PER_TYPE, SEED
from research.stage4_7_1_context_extraction import (
    CONTEXT_EXTRACTION_CACHE, ContextRecordV2, _cache_key as s471_cache_key,
    _context_extract_prompt, build_group_context, dedupe_records, validate_context_record,
)
from research.versioned_memory_cells import _cosine

RESOLVER_CACHE = Path("/tmp/socialmembench_stage4_8_semantic_resolver.jsonl")
LLM_CALL_CACHE = Path("/tmp/socialmembench_stage4_8_llm_cache.jsonl")
RESULTS_JSONL = Path("/tmp/socialmembench_stage4_8_results.jsonl")
REPORT_MD = Path("/tmp/socialmembench_stage4_8_report.md")

SEMANTIC_DECISIONS = ("NEW_CELL", "OBSERVE", "AUGMENT", "REVISE", "RETRACT")
SEMANTIC_CONFIDENCE_FLOOR = 0.5
VARIANTS = ("RAW", "FLAT", "VERSIONED")


# --------------------------------------------------------------------------
# Generic content-hashed LLM-output cache (answer/judge calls) -- separate
# from the extraction cache (Stage 4.7.1's, reused read-only) and the
# resolver cache (this stage's own).
# --------------------------------------------------------------------------

_llm_cache: Optional[dict[str, Any]] = None


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


def cached_answer(question: str, options: Any, context: str) -> str:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = _load_kv_cache(LLM_CALL_CACHE)
    key = "answer:" + hashlib.sha256((question + json.dumps(options, sort_keys=True) + context).encode()).hexdigest()
    if key in _llm_cache:
        return _llm_cache[key]
    from research.socialmembench_pilot import _answer

    result = _answer(question, options, context)
    _llm_cache[key] = result
    _append_kv_cache(LLM_CALL_CACHE, key, result)
    return result


def cached_judge_correctness(question: str, gold: str, answers: dict[str, str]) -> dict[str, float]:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = _load_kv_cache(LLM_CALL_CACHE)
    key = "judge_correctness:" + hashlib.sha256((question + gold + json.dumps(answers, sort_keys=True)).encode()).hexdigest()
    if key in _llm_cache:
        return _llm_cache[key]
    from research.socialmembench_pilot import _judge_open_answers

    result = _judge_open_answers(question, gold, answers)
    _llm_cache[key] = result
    _append_kv_cache(LLM_CALL_CACHE, key, result)
    return result


def _judge_temporal_and_attribution_prompt(question: str, gold: str, answers: dict[str, str]) -> str:
    rendered = "\n\n".join(f"{name}:\n{text}" for name, text in answers.items())
    return f"""For each answer below, judge two things independently, each exactly 0.0 or 1.0.
Judge only against the gold answer and the question -- ignore citation formatting,
ignore verbosity, do not compare systems to each other.

- temporal_correctness: does the answer correctly identify that a CHANGE occurred
  and correctly describe both the earlier state/behavior and the resulting one
  (not just one side, not a vague "something changed" with no content)?
- attribution_correctness: does the answer correctly attribute each state/action/
  opinion to the correct person (never confusing who experienced or said what,
  never swapping the subject and the viewpoint holder)?

Question: {question}
Gold answer: {gold}

Answers:
{rendered}

Return JSON only, one entry per system name:
{{"System name": {{"temporal_correctness": 0.0, "attribution_correctness": 0.0}}}}"""


def cached_judge_temporal_attribution(question: str, gold: str, answers: dict[str, str]) -> dict[str, dict[str, float]]:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = _load_kv_cache(LLM_CALL_CACHE)
    key = "judge_temporal:" + hashlib.sha256((question + gold + json.dumps(answers, sort_keys=True)).encode()).hexdigest()
    if key in _llm_cache:
        return _llm_cache[key]
    from llm.groq_client import get_chat_model

    prompt = _judge_temporal_and_attribution_prompt(question, gold, answers)
    message = get_chat_model("primary", 0).invoke(prompt)
    parsed = _parse_json_object(message.content if isinstance(message.content, str) else "") or {}
    result = {
        name: {
            "temporal_correctness": float((parsed.get(name) or {}).get("temporal_correctness", 0.0)),
            "attribution_correctness": float((parsed.get(name) or {}).get("attribution_correctness", 0.0)),
        }
        for name in answers
    }
    _llm_cache[key] = result
    _append_kv_cache(LLM_CALL_CACHE, key, result)
    return result


# --------------------------------------------------------------------------
# Step 1-2: every chronological evidence-anchor cluster (session-grouped,
# not just old/new extremes), extracted via the Stage 4.7.1 contract.
# --------------------------------------------------------------------------

def build_session_clusters(anchors: list[dict]) -> list[tuple[int, set[str]]]:
    """Every DISTINCT session_index among the gold anchors becomes its own
    chronological cluster -- this recovers anchors Stage 4.5's old/new split
    dropped as "unassigned" (any session that wasn't the global min/max)."""
    by_session: dict[int, set[str]] = defaultdict(set)
    for anchor in anchors:
        if anchor.get("turn_id"):
            session_index = anchor.get("session_index", 0)
            by_session[session_index].add(anchor["turn_id"])
    return sorted(by_session.items())


def build_extraction_cache(cases: list[dict], conversations) -> dict[str, dict]:
    """Reuses Stage 4.7.1's exact prompt/cache-key/context-window logic
    (imported, not reimplemented) over an arbitrary number of clusters per
    case. Appends to the SAME immutable Stage 4.7.1 cache file -- the cache
    key is pure content hash (network+anchor_ids+schema+prompt_hash), so a
    cluster whose anchor set matches an already-cached old/new group is a
    free hit; only genuinely new clusters (e.g. previously "unassigned"
    middle sessions) cost a new call."""
    cached: dict[str, dict] = {}
    if CONTEXT_EXTRACTION_CACHE.exists():
        for line in CONTEXT_EXTRACTION_CACHE.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cached[row["cache_key"]] = row

    to_call: list[tuple[str, str]] = []  # (key, prompt)
    seen_keys: set[str] = set()
    for case in cases:
        for _session_index, anchor_ids in case["clusters"]:
            context_text, _window_ids, _messages = build_group_context(conversations, case["network_id"], anchor_ids)
            if not context_text:
                continue
            prompt = _context_extract_prompt(context_text, sorted(anchor_ids))
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            key = s471_cache_key(case["network_id"], anchor_ids, prompt_hash)
            if key in cached or key in seen_keys:
                continue
            seen_keys.add(key)
            to_call.append((key, prompt))

    if to_call:
        from llm.groq_client import get_chat_model

        print(f"Extracting {len(to_call)} new anchor clusters (Stage 4.7.1 contract, Stage 4.8 run)...", file=sys.stderr)
        CONTEXT_EXTRACTION_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CONTEXT_EXTRACTION_CACHE.open("a") as out:
            for done, (key, prompt) in enumerate(to_call, 1):
                message = get_chat_model("fast", 0).invoke(prompt)
                parsed = _parse_json_object(message.content if isinstance(message.content, str) else "") or {}
                row = {
                    "cache_key": key, "schema_version": "ctx_v1_1",
                    "raw_items": [item for item in parsed.get("items", []) if isinstance(item, dict)],
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                cached[key] = row
                if done % 10 == 0 or done == len(to_call):
                    print(f"  extracted {done}/{len(to_call)}", file=sys.stderr)
    return cached


def extract_records_for_case(case: dict, conversations, extraction_cache: dict) -> tuple[list[ContextRecordV2], dict[str, dict], list[dict]]:
    """Returns (deduped records across all clusters, window turn info by
    turn_id, extraction_failures for this case)."""
    network_id = case["network_id"]
    all_records: list[ContextRecordV2] = []
    window_turns: dict[str, dict] = {}
    failures: list[dict] = []

    for session_index, anchor_ids in case["clusters"]:
        context_text, window_ids, message_by_turn = build_group_context(conversations, network_id, anchor_ids)
        if not context_text:
            failures.append({"qa_id": case["qa_id"], "session_index": session_index, "reason": "empty_context"})
            continue
        prompt = _context_extract_prompt(context_text, sorted(anchor_ids))
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        key = s471_cache_key(network_id, anchor_ids, prompt_hash)
        cache_row = extraction_cache.get(key)
        if cache_row is None:
            failures.append({"qa_id": case["qa_id"], "session_index": session_index, "reason": "missing_from_cache"})
            continue

        frame = conversations[(conversations.network_id == network_id) & (conversations.turn_id.isin(window_ids))]
        timestamp_by_turn = {str(r.turn_id): str(r.timestamp) for r in frame.itertuples(index=False)}
        cluster_records = []
        for item in cache_row.get("raw_items", []):
            record, _reason = validate_context_record(item, window_ids, anchor_ids, message_by_turn, timestamp_by_turn)
            if record is not None:
                cluster_records.append(record)
        if not cluster_records:
            failures.append({"qa_id": case["qa_id"], "session_index": session_index, "reason": "zero_valid_records"})
        all_records.extend(cluster_records)
        for turn_row in frame.itertuples(index=False):
            window_turns[str(turn_row.turn_id)] = {
                "speaker": turn_row.speaker_display_name, "timestamp": str(turn_row.timestamp), "message": turn_row.message,
            }

    return dedupe_records(all_records), window_turns, failures


# --------------------------------------------------------------------------
# Step 5-6: semantic resolver, facet removed from the hard filter --
# scoped ENTIRELY to this experiment, does not touch Stage 4.6's resolver.
# --------------------------------------------------------------------------

@dataclass
class Stage48Version:
    version_id: str
    operation: str  # create | revise | retract
    record: ContextRecordV2
    closed: bool = False


@dataclass
class Stage48Observation:
    kind: str  # observe | augment | cause | reaction
    record: ContextRecordV2


@dataclass
class Stage48Cell:
    cell_id: str
    network_id: str
    viewpoint_owner: str
    subject: str
    versions: list[Stage48Version] = field(default_factory=list)
    observations: list[Stage48Observation] = field(default_factory=list)

    @property
    def active_version(self) -> Optional[Stage48Version]:
        for version in reversed(self.versions):
            if not version.closed:
                return version
        return None

    def source_turn_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for version in self.versions:
            ids.extend(version.record.source_turn_ids)
        for observation in self.observations:
            ids.extend(observation.record.source_turn_ids)
        return tuple(dict.fromkeys(ids))


def _record_embed_text(record: ContextRecordV2) -> str:
    parts = [record.state_description or "", record.claim]
    return " ".join(p for p in parts if p)


def _candidate_filter(registry: dict[str, Stage48Cell], record: ContextRecordV2, network_id: str) -> list[Stage48Cell]:
    """Hard filter per this stage's instructions: network + resolved
    viewpoint_owner + resolved subject ONLY -- facet is not part of the
    Stage 4.7.1 schema at all any more, so it cannot gate here (removed,
    not just loosened). "Compatible scope" is a documented no-op: this
    dataset's Q8 dev cases are individual-state narratives and the
    extraction schema carries no scope_type to compare."""
    return [
        cell for cell in registry.values()
        if cell.network_id == network_id
        and cell.viewpoint_owner == _norm(record.viewpoint_owner)
        and cell.subject == _norm(record.subject)
    ]


def _top3_by_embedding(record: ContextRecordV2, candidates: list[Stage48Cell]) -> list[Stage48Cell]:
    """facet/state_description/claim text are used ONLY here -- as a soft
    embedding-similarity ranking signal over the hard-filtered candidate
    set, never as a hard gate."""
    if not candidates:
        return []
    query_vec = embed(_record_embed_text(record))
    scored = []
    for cell in candidates:
        rep = cell.active_version.record if cell.active_version else (cell.versions[-1].record if cell.versions else None)
        if rep is None:
            continue
        scored.append((cell, _cosine(query_vec, embed(_record_embed_text(rep)))))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [cell for cell, _ in scored[:3]]


def _semantic_resolver_prompt(record: ContextRecordV2, candidates_desc: list[dict]) -> str:
    lines = []
    for letter, cand in zip("ABC", candidates_desc):
        lines.append(
            f"[{letter}] state_description={cand['state_description']!r} current_value={cand['value']!r} "
            f"record_type={cand['record_type']} temporal_mode={cand['temporal_mode']}\n"
            f"    \"{cand['claim']}\""
        )
    candidates_block = "\n".join(lines) if lines else "(no existing candidate cells)"
    return f"""You are deciding how a NEW memory record about someone's state relates to
EXISTING memory cells about the same person/subject. You do not know what question
(if any) this relates to, and you must not guess one -- decide only from the
content below.

NEW record:
  record_type: {record.record_type}
  claim: "{record.claim}"
  state_description: {record.state_description!r}
  value: {record.value!r}
  from_value: {record.from_value!r}
  to_value: {record.to_value!r}
  temporal_mode: {record.temporal_mode}

Existing candidate cells (top 3 by similarity, or fewer):
{candidates_block}

Choose exactly one decision:
- NEW_CELL: this is genuinely a different topic/state from every candidate.
- OBSERVE: this repeats/reaffirms a candidate's CURRENT state (same real-world value).
- AUGMENT: this adds a compatible detail to a candidate's current state WITHOUT
  replacing it (the underlying state did not actually change).
- REVISE: the underlying real-world state actually changed from a candidate's
  current value to something incompatible, OR the record explicitly signals a
  change. IMPORTANT: two differently-WORDED values are NOT by themselves
  evidence of REVISE -- only choose REVISE if the state itself is genuinely
  different, not merely described differently.
- RETRACT: a candidate's current state is explicitly withdrawn/no longer holds,
  with nothing new asserted in its place.

If genuinely uncertain, choose NEW_CELL -- a missed link is recoverable, a false
merge is not.

Return JSON only:
{{"decision":"NEW_CELL|OBSERVE|AUGMENT|REVISE|RETRACT","target":"A"|"B"|"C"|null,
"confidence":0.0,"reason":"one short sentence"}}"""


def _parse_semantic_decision(parsed: dict, num_candidates: int) -> tuple[str, Optional[int], float, str, bool]:
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


_resolver_cache: Optional[dict[str, dict]] = None


def run_semantic_resolver(record: ContextRecordV2, candidates: list[Stage48Cell]) -> dict[str, Any]:
    global _resolver_cache
    if not candidates:
        return {
            "decision": "NEW_CELL", "target_cell_id": None, "confidence": 1.0,
            "reason": "no candidate cells", "defaulted": False, "candidates_seen": [],
        }
    if _resolver_cache is None:
        _resolver_cache = _load_kv_cache(RESOLVER_CACHE)

    candidates_desc = [
        {
            "state_description": (cell.active_version.record.state_description if cell.active_version else None),
            "value": (cell.active_version.record.value if cell.active_version else None),
            "record_type": (cell.active_version.record.record_type if cell.active_version else ""),
            "temporal_mode": (cell.active_version.record.temporal_mode if cell.active_version else ""),
            "claim": (cell.active_version.record.claim if cell.active_version else ""),
        }
        for cell in candidates
    ]
    prompt = _semantic_resolver_prompt(record, candidates_desc)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    key = "resolver:" + prompt_hash

    if key in _resolver_cache:
        parsed = _resolver_cache[key]
    else:
        from llm.groq_client import get_chat_model

        message = get_chat_model("fast", 0).invoke(prompt)
        parsed = _parse_json_object(message.content if isinstance(message.content, str) else "") or {}
        _resolver_cache[key] = parsed
        _append_kv_cache(RESOLVER_CACHE, key, parsed)

    decision, target_index, confidence, reason, defaulted = _parse_semantic_decision(parsed, len(candidates))
    target_cell = candidates[target_index] if target_index is not None else None
    return {
        "decision": decision, "target_cell_id": (target_cell.cell_id if target_cell else None),
        "confidence": confidence, "reason": reason, "defaulted": defaulted,
        "candidates_seen": [c.cell_id for c in candidates],
    }


def apply_decision(registry: dict[str, Stage48Cell], record: ContextRecordV2, decision_info: dict, network_id: str) -> None:
    decision = decision_info["decision"]
    if decision == "NEW_CELL":
        cell_id = f"cell_{len(registry) + 1}"
        cell = Stage48Cell(cell_id=cell_id, network_id=network_id, viewpoint_owner=_norm(record.viewpoint_owner), subject=_norm(record.subject))
        cell.versions.append(Stage48Version(version_id=f"{cell_id}:v1", operation="create", record=record))
        registry[cell_id] = cell
        return

    target_cell = registry[decision_info["target_cell_id"]]
    if decision in ("OBSERVE", "AUGMENT"):
        target_cell.observations.append(Stage48Observation(kind=decision.lower(), record=record))
        return

    if target_cell.active_version is not None:
        target_cell.active_version.closed = True
    version_no = len(target_cell.versions) + 1
    # A retract itself is never "active" -- it records that nothing
    # currently holds, not a new ongoing state. A revise's new version IS
    # active (it's the new current state).
    new_version = Stage48Version(
        version_id=f"{target_cell.cell_id}:v{version_no}", operation=decision.lower(), record=record,
        closed=(decision == "RETRACT"),
    )
    target_cell.versions.append(new_version)


def build_versioned_cells(records: list[ContextRecordV2], network_id: str) -> tuple[dict[str, Stage48Cell], list[dict]]:
    registry: dict[str, Stage48Cell] = {}
    decisions_log: list[dict] = []

    state_like = sorted(
        (r for r in records if r.record_type in ("STATE", "TRANSITION")), key=lambda r: r.observed_at,
    )
    for record in state_like:
        candidates = _top3_by_embedding(record, _candidate_filter(registry, record, network_id))
        decision_info = run_semantic_resolver(record, candidates)
        decisions_log.append({
            "claim": record.claim, "source_turn_ids": list(record.source_turn_ids), **decision_info,
        })
        apply_decision(registry, record, decision_info, network_id)

    for record in records:
        if record.record_type in ("CAUSE", "REACTION"):
            for cell in registry.values():
                if cell.subject == _norm(record.subject):
                    cell.observations.append(Stage48Observation(kind=record.record_type.lower(), record=record))

    return registry, decisions_log


# --------------------------------------------------------------------------
# Rendering the three variants (same Candidate/render_context machinery
# already used by the production pilot, for a consistent citation format).
# --------------------------------------------------------------------------

def _cell_text(cell: Stage48Cell) -> str:
    lines = []
    active = cell.active_version
    if active is not None:
        record = active.record
        lines.append(f"CURRENT STATE ({active.operation}): {record.claim}")
        if record.state_description or record.value:
            lines.append(f"  state_description={record.state_description!r} value={record.value!r}")
    closed_versions = [v for v in cell.versions if v.closed]
    if closed_versions:
        lines.append("PRIOR VERSIONS:")
        for version in closed_versions:
            lines.append(f"  [{version.operation}] {version.record.claim}")
    causes = [o.record for o in cell.observations if o.kind == "cause"]
    reactions = [o.record for o in cell.observations if o.kind == "reaction"]
    reaffirms = [o.record for o in cell.observations if o.kind in ("observe", "augment")]
    if causes:
        lines.append("CAUSES:")
        lines.extend(f"  {c.claim}" for c in causes)
    if reactions:
        lines.append("REACTIONS:")
        lines.extend(f"  {r.claim}" for r in reactions)
    if reaffirms:
        lines.append("ADDITIONAL OBSERVATIONS:")
        lines.extend(f"  {r.claim}" for r in reaffirms)
    return "\n".join(lines)


def build_variant_contexts(
    window_turns: dict[str, dict], records: list[ContextRecordV2], registry: dict[str, Stage48Cell],
) -> dict[str, str]:
    ordered_turn_ids = sorted(window_turns, key=lambda t: (window_turns[t]["timestamp"], t))
    raw_by_source = {
        turn_id: Candidate(
            candidate_id=f"raw:{turn_id}", kind="raw", text=f"{info['speaker']}: {info['message']}",
            source_ids=(turn_id,), observed_at=info["timestamp"],
        )
        for turn_id, info in window_turns.items()
    }

    candidates_a = [raw_by_source[t] for t in ordered_turn_ids]
    context_a, _exposed_a = render_context(candidates_a, raw_by_source, expand_provenance=False)

    candidates_b = [
        Candidate(
            candidate_id=f"flat:{i}", kind=f"derived:{record.record_type.lower()}",
            text=f"{record.claim}" + (
                f" (state_description={record.state_description!r}, value={record.value!r})"
                if record.state_description or record.value else ""
            ) + (
                f" (from={record.from_value!r}, to={record.to_value!r})" if record.from_value or record.to_value else ""
            ),
            source_ids=record.source_turn_ids, asserted_by=(record.viewpoint_owner,), entities=(record.subject,),
        )
        for i, record in enumerate(records)
    ]
    context_b, _exposed_b = render_context(candidates_b, raw_by_source, expand_provenance=True)

    candidates_c = [
        Candidate(
            candidate_id=f"cell:{cell.cell_id}", kind="derived:versioned_cell", text=_cell_text(cell),
            source_ids=cell.source_turn_ids(), asserted_by=(cell.viewpoint_owner,), entities=(cell.subject,),
        )
        for cell in registry.values()
    ]
    context_c, _exposed_c = render_context(candidates_c, raw_by_source, expand_provenance=True)

    return {"RAW": context_a, "FLAT": context_b, "VERSIONED": context_c}


# --------------------------------------------------------------------------
# Per-question orchestration
# --------------------------------------------------------------------------

def run_question(case: dict, conversations, extraction_cache: dict) -> dict:
    qa_id, network_id, question, gold_answer = case["qa_id"], case["network_id"], case["question"], case["gold_answer"]
    gold_ids = case["gold_ids"]

    records, window_turns, extraction_failures = extract_records_for_case(case, conversations, extraction_cache)
    registry, decisions_log = build_versioned_cells(records, network_id)
    contexts = build_variant_contexts(window_turns, records, registry)

    answers = {name: cached_answer(question, {}, context) for name, context in contexts.items()}
    correctness = cached_judge_correctness(question, gold_answer, answers)
    temporal_attribution = cached_judge_temporal_attribution(question, gold_answer, answers)

    cited = {name: _cited_ids(answer) for name, answer in answers.items()}
    citation_metrics = {name: _set_metrics(cited[name], gold_ids) for name in VARIANTS}

    generation_failures = [name for name, answer in answers.items() if not answer.strip() or "insufficient" in answer.lower()[:200]]

    cell_chains = [
        {
            "cell_id": cell.cell_id, "viewpoint_owner": cell.viewpoint_owner, "subject": cell.subject,
            "chain": [{"operation": v.operation, "claim": v.record.claim, "closed": v.closed} for v in cell.versions],
            "observation_count": len(cell.observations),
        }
        for cell in registry.values()
    ]
    same_subject_cell_counts = Counter((cell.viewpoint_owner, cell.subject) for cell in registry.values())
    multi_cell_subjects = [key for key, count in same_subject_cell_counts.items() if count > 1]

    return {
        "qa_id": qa_id, "network_id": network_id, "question": question, "gold_answer": gold_answer,
        "gold_ids": sorted(gold_ids), "n_clusters": len(case["clusters"]), "n_records": len(records),
        "answers": answers, "cited_ids": {k: sorted(v) for k, v in cited.items()},
        "correctness": correctness, "temporal_attribution": temporal_attribution,
        "citation_metrics": citation_metrics,
        "extraction_failures": extraction_failures, "generation_failures": generation_failures,
        "cell_chains": cell_chains, "resolver_decisions": decisions_log,
        "multi_cell_subjects": [list(k) for k in multi_cell_subjects],
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def render_report(results: list[dict]) -> str:
    n = len(results)
    lines = [
        "# Stage 4.8 -- RAW vs FLAT vs VERSIONED, closed oracle-evidence component experiment",
        "",
        "**Methodological label**: oracle-evidence COMPONENT experiment. Evidence-anchor regions came from Q8 "
        "gold metadata; this is NOT honest end-to-end ingestion and is NOT comparable to official SocialMemBench "
        "numbers. All three variants share the same per-question evidence scope and the same generator/judge "
        "calls -- only the organization of that evidence differs. Primary comparison: VERSIONED vs FLAT.",
        "",
        f"- questions: {n}",
        "",
        "## Aggregate metrics",
        "",
        "| metric | RAW | FLAT | VERSIONED |",
        "|---|---|---|---|",
    ]

    correctness_avg = {name: sum(r["correctness"][name] for r in results) / n for name in VARIANTS}
    temporal_avg = {name: sum(r["temporal_attribution"][name]["temporal_correctness"] for r in results) / n for name in VARIANTS}
    attribution_avg = {name: sum(r["temporal_attribution"][name]["attribution_correctness"] for r in results) / n for name in VARIANTS}
    precision_avg = {name: sum(r["citation_metrics"][name]["precision"] for r in results) / n for name in VARIANTS}
    recall_avg = {name: sum(r["citation_metrics"][name]["recall"] for r in results) / n for name in VARIANTS}

    lines.append(f"| answer correctness | {correctness_avg['RAW']:.3f} | {correctness_avg['FLAT']:.3f} | {correctness_avg['VERSIONED']:.3f} |")
    lines.append(f"| temporal correctness | {temporal_avg['RAW']:.3f} | {temporal_avg['FLAT']:.3f} | {temporal_avg['VERSIONED']:.3f} |")
    lines.append(f"| attribution correctness | {attribution_avg['RAW']:.3f} | {attribution_avg['FLAT']:.3f} | {attribution_avg['VERSIONED']:.3f} |")
    lines.append(f"| citation precision | {precision_avg['RAW']:.3f} | {precision_avg['FLAT']:.3f} | {precision_avg['VERSIONED']:.3f} |")
    lines.append(f"| citation recall (vs gold anchors) | {recall_avg['RAW']:.3f} | {recall_avg['FLAT']:.3f} | {recall_avg['VERSIONED']:.3f} |")

    c_beats_b = sum(1 for r in results if r["correctness"]["VERSIONED"] > r["correctness"]["FLAT"])
    b_beats_c = sum(1 for r in results if r["correctness"]["FLAT"] > r["correctness"]["VERSIONED"])
    ties = n - c_beats_b - b_beats_c

    total_extraction_failures = sum(len(r["extraction_failures"]) for r in results)
    total_generation_failures = sum(len(r["generation_failures"]) for r in results)
    all_decisions = [d for r in results for d in r["resolver_decisions"]]
    decision_counts = Counter(d["decision"] for d in all_decisions)
    defaulted_count = sum(1 for d in all_decisions if d["defaulted"])
    multi_cell_cases = sum(1 for r in results if r["multi_cell_subjects"])

    lines += [
        "",
        "## Primary comparison: VERSIONED vs FLAT (per-question answer correctness)",
        "",
        f"- VERSIONED beats FLAT: {c_beats_b}/{n}",
        f"- FLAT beats VERSIONED: {b_beats_c}/{n}",
        f"- ties: {ties}/{n}",
        "",
        "## Failures",
        "",
        f"- extraction failures (clusters with zero valid records or empty context): {total_extraction_failures}",
        f"- generation failures (empty or 'insufficient evidence' answers, any variant): {total_generation_failures}",
        f"- resolver decisions: {dict(decision_counts)} (defaulted to NEW_CELL due to low confidence/malformed output: {defaulted_count})",
        f"- questions with >1 cell sharing the same (viewpoint_owner, subject) -- NOT a verdict on whether this is "
        f"correct, just a count of cases where linking decisions kept same-subject state in separate cells: {multi_cell_cases}",
        "",
        "## Per-question results", "",
    ]

    for i, r in enumerate(results, 1):
        lines.append(f"### {i}. {r['qa_id']} :: {r['network_id']}")
        lines.append(f"> {r['question']}")
        lines.append(f"- gold_answer: {r['gold_answer']}")
        lines.append(f"- gold_ids: {r['gold_ids']}")
        lines.append(f"- clusters: {r['n_clusters']}  records: {r['n_records']}")
        for name in VARIANTS:
            lines.append(
                f"- {name}: correctness={r['correctness'][name]:.1f} "
                f"temporal={r['temporal_attribution'][name]['temporal_correctness']:.0f} "
                f"attribution={r['temporal_attribution'][name]['attribution_correctness']:.0f} "
                f"precision={r['citation_metrics'][name]['precision']:.2f} "
                f"recall={r['citation_metrics'][name]['recall']:.2f} "
                f"cited={r['cited_ids'][name]}"
            )
        if r["cell_chains"]:
            lines.append("- cell chains used by VERSIONED:")
            for chain in r["cell_chains"]:
                ops = " -> ".join(f"{step['operation']}{'(closed)' if step['closed'] else ''}" for step in chain["chain"])
                lines.append(f"    {chain['cell_id']} ({chain['viewpoint_owner']}/{chain['subject']}): {ops}  observations={chain['observation_count']}")
        if r["extraction_failures"]:
            lines.append(f"- extraction_failures: {r['extraction_failures']}")
        if r["generation_failures"]:
            lines.append(f"- generation_failures: {r['generation_failures']}")
        if r["multi_cell_subjects"]:
            lines.append(f"- multi_cell_subjects: {r['multi_cell_subjects']}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    conversations, qa, _personas = load_data(DATA_DIR)
    selected = _select_questions(qa, PER_TYPE, SEED)
    q8_rows = [row for row in selected if row["query_type"] == "Q8"]
    assert len(q8_rows) == 20, f"expected 20 Q8 cases, got {len(q8_rows)}"

    cases = []
    for row in q8_rows:
        anchors = _json(row["evidence_anchors_json"], [])
        anchors = [a for a in anchors if a.get("turn_id")]
        gold_ids = {a["turn_id"] for a in anchors}
        clusters = build_session_clusters(anchors)
        cases.append({
            "qa_id": row["qa_id"], "network_id": row["network_id"], "question": row["question"],
            "gold_answer": row["answer"], "gold_ids": gold_ids, "clusters": clusters,
        })

    extraction_cache = build_extraction_cache(cases, conversations)

    print(f"Running {len(cases)} Q8 questions x 3 variants...", file=sys.stderr)
    results = []
    for i, case in enumerate(cases, 1):
        results.append(run_question(case, conversations, extraction_cache))
        print(f"  {i}/{len(cases)} {case['qa_id']}", file=sys.stderr)

    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    REPORT_MD.write_text(render_report(results))
    print(f"Results: {RESULTS_JSONL}", file=sys.stderr)
    print(f"Report: {REPORT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
