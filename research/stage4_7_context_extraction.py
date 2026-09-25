"""Stage 4.7 -- context-aware observation extraction, isolated from linking.

Stage 4.6 isolated LINKING from extraction and found co_cell_recall=0/12 even
with "oracle" extraction -- but manual review of that extraction (the pair
review sheets) found the real problem was upstream: extraction saw only the
bare anchor turn (no surrounding conversational context), so short replies,
pronouns, sarcasm, and ellipsis were extracted as bare, meaningless claims
("Fine.", "I hear you.", "this matters."); non-empty output was wrongly
treated as "usable"; and forcing the extractor to assign facet/topic_key
immediately caused vocabulary drift between independently-extracted sides
(e.g. "health_status" vs "knee_condition" for the SAME real topic) that
starved the candidate filter before either linker ever got a chance to
decide anything (Stage 4.6 decisions review: only 2/62 assertion
applications ever saw a non-empty candidate pool).

This module fixes ONLY extraction:
  1. Each anchor GROUP (old-side / new-side, exactly as split by Stage 4.5's
     _split_old_new -- UNCHANGED, reused as-is) now gets a local
     conversational context: every anchor turn in the group, plus up to 2
     turns before/after it in the SAME session, deduplicated across
     anchors, with anchor turns explicitly marked [ANCHOR].
  2. The extraction schema no longer asks for facet/topic_key/state_key/
     operation/memory_worthy/an existing cell reference, or any linking
     decision -- record_type is one of STATE/TRANSITION/CAUSE/REACTION,
     matching what memory work actually needs to distinguish (an observed
     state vs an explicit transition vs a cause vs someone else's reaction)
     without prematurely committing to a specific facet/topic slug.

Explicitly NOT done here: no linker (threshold or semantic) runs on this
output, no cells are created, no facet/topic_key is assigned, no
retrieval/answer generation, no Q6, no held-out QA. This module only
produces two things: (a) a cached, immutable set of Stage 4.7 records, and
(b) a human review sheet comparing them to the frozen Stage 4.6 baseline.
resolve_topic_slot, decide_state_operation, TOPIC_MATCH_THRESHOLD, and the
Stage 4.6 semantic resolver are not imported, not called, not modified.

Run:
    python3 -m research.stage4_7_context_extraction
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.socialmembench_pilot import _json, _parse_json_object, _select_questions, load_data
from research.stage4_5_audit import DATA_DIR, PER_TYPE, SEED, _split_old_new
from research.stage4_6_oracle_linking import ORACLE_EXTRACTION_CACHE, _validated_side_assertions
from research.versioned_memory_cells import _is_meta_value

SCHEMA_VERSION = "ctx_v1"
CONTEXT_EXTRACTION_CACHE = Path("/tmp/socialmembench_stage4_7_context_extraction.jsonl")
STRUCTURAL_REPORT_MD = Path("/tmp/socialmembench_stage4_7_structural_report.md")
EXTRACTION_REVIEW_MD = Path("/tmp/socialmembench_stage4_7_extraction_review.md")

VALID_RECORD_TYPES = {"STATE", "TRANSITION", "CAUSE", "REACTION"}
VALID_TEMPORAL_MODES = {"past", "current", "future", "habitual", "unknown"}

_TURN_NUM_RE = re.compile(r"t(\d+)$")

# The 3 pairs a human confirmed as genuine while reviewing
# /tmp/socialmembench_stage4_6_pair_review_v2.md, and the 9 (of 12 usable)
# Stage 4.6 cases the same review marked "No valid pair in extraction" --
# both sets given verbatim by the user, not re-derived here. Every qa_id
# below was cross-checked against the actual v1/v2 pair-review numbering
# (old_assertions then new_assertions, concatenated, numbered from 1) to
# confirm the referenced #s resolve to the described claims.
CONFIRMED_PAIRS = {
    "Q8_a9b0c1d216": {"who": "Seb", "pair": "#1 -> #3", "relation": "REVISE"},
    "Q8_f6g7h8i901": {"who": "Uncle Femi", "pair": "#2 -> #5", "relation": "RETRACT"},
    "Q8_ph9s4c1": {"who": "Diane", "pair": "#3 -> #6", "relation": "REVISE"},
}
NO_VALID_PAIR_QA_IDS = {
    "Q8_347b5cff", "Q8_b1c2d3e406", "Q8_b8c9dae029", "Q8_c2d3e4f5", "Q8_cbf238d6",
    "Q8_d4e5f613", "Q8_d4e5f616", "Q8_d5e6f7a8", "Q8_d7e8f9a002",
}


def _turn_num(turn_id: str) -> int:
    match = _TURN_NUM_RE.search(turn_id or "")
    return int(match.group(1)) if match else 0


def _session_frame_sorted(conversations, network_id: str, session_index: int):
    frame = conversations[
        (conversations.network_id == network_id) & (conversations.session_index == session_index)
    ].copy()
    frame["_order"] = frame.turn_id.map(_turn_num)
    return frame.sort_values("_order").reset_index(drop=True)


def build_group_context(conversations, network_id: str, anchor_ids: set[str]) -> tuple[str, set[str]]:
    """Every anchor turn in the group plus up to 2 turns before/after it in
    the SAME session, deduplicated across anchors, chronologically ordered,
    anchor turns marked [ANCHOR]. Returns (context_text, window_turn_ids) --
    window_turn_ids is every turn_id actually shown (the only ids an
    extraction call for this group could truthfully cite)."""
    lookup = conversations[
        (conversations.network_id == network_id) & (conversations.turn_id.isin(anchor_ids))
    ]
    session_by_turn = {row.turn_id: row.session_index for row in lookup.itertuples(index=False)}

    window_rows: dict[str, Any] = {}
    for turn_id in anchor_ids:
        session_index = session_by_turn.get(turn_id)
        if session_index is None:
            continue
        frame = _session_frame_sorted(conversations, network_id, session_index)
        idx_list = frame.index[frame.turn_id == turn_id].tolist()
        if not idx_list:
            continue
        pos = idx_list[0]
        window = frame.iloc[max(0, pos - 2): pos + 3]
        for _, row in window.iterrows():
            window_rows[row.turn_id] = row

    ordered = sorted(window_rows.values(), key=lambda r: (r.session_index, _turn_num(r.turn_id)))
    lines = []
    for row in ordered:
        marker = "[ANCHOR] " if row.turn_id in anchor_ids else ""
        lines.append(f"[[{row.turn_id}]] {marker}{row.timestamp} {row.speaker_display_name}: {row.message}")
    return "\n".join(lines), set(window_rows.keys())


def _context_extract_prompt(context_lines: str) -> str:
    return f"""You will see a short window of chat turns from ONE session -- some of them
are marked [ANCHOR] (turns specifically selected for review), the rest are
surrounding context (up to 2 turns before/after each anchor) included only to
help you understand pronouns, ellipsis, sarcasm, and short replies. You do not
know why these turns were selected, what question (if any) they answer, or what
relation (if any) is expected between them.

Extract atomic records from what these turns actually say. Each record has a
record_type:

- STATE: an observed value of some ongoing property (health, opinion/
  preference, intention, commitment, availability, role, location, social
  position, group norm, etc). Must include state_description and value.
- TRANSITION: the turn ITSELF explicitly expresses a change ("used to X now
  Y", "no longer", "started/stopped", "changed my mind", "this time I
  decided", "actually I now think"). Include state_description and, where the
  text actually supports it, from_value and to_value -- do NOT invent a side
  of the transition that isn't actually there.
- CAUSE: an event or explanation that plausibly caused some state to change.
  A CAUSE is NOT itself an old or new state version. Only set
  related_state_description if the text actually supports the connection.
- REACTION: another participant's (or the group's) reaction to a state or
  change. A REACTION is NOT a new state version of the subject being
  discussed.

Discipline:
- Use context ONLY to resolve pronouns/ellipsis/sarcasm/short replies -- every
  claim must be self-contained. Never leave a claim as a bare "Fine.", "I hear
  you.", "this matters." -- either resolve what it actually refers to using
  the context, or don't extract it.
- Do not turn every message into a memory record -- 0 items is a fine answer.
- Do not take sarcasm literally.
- Do not swap the person who holds an opinion with the person the opinion is
  about.
- A reaction from someone else is never a new state version of the original
  subject's state.
- A cause/explanation is never treated as an old state version.
- A stated plan and its later execution are NOT automatically a REVISE of
  each other -- only if the text shows the underlying state actually changed.
- Different events are not the same evolving state just because they involve
  the same person.
- Do not infer anything not supported by the visible turns.
- source_turn_ids must be copied EXACTLY from the turns shown, and every
  record must cite at least one turn marked [ANCHOR].

viewpoint_owner: whose view/knowledge/state/commitment this is.
subject: who or what the claim is actually about.
temporal_mode: one of past, current, future, habitual, unknown.
confidence: 0.0-1.0.

Do NOT include facet, topic_key, state_key, operation, memory_worthy, a
reference to an existing cell, or any NEW_CELL/OBSERVE/AUGMENT/REVISE/RETRACT
decision -- none of that is being asked for here.

Return JSON only:
{{"items":[{{"record_type":"STATE|TRANSITION|CAUSE|REACTION","claim":"...",
"viewpoint_owner":"...","subject":"...","state_description":"...","value":"...",
"temporal_mode":"unknown","from_value":null,"to_value":null,
"related_state_description":null,"source_turn_ids":["turn_id"],"confidence":0.0}}]}}

Turns:
{context_lines}"""


def _cache_key(network_id: str, anchor_ids: set[str], prompt_hash: str) -> str:
    payload = {
        "network_id": network_id, "source_ids": sorted(anchor_ids),
        "schema_version": SCHEMA_VERSION, "prompt_hash": prompt_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_context_extraction_cache(
    cases: list[dict], conversations,
) -> tuple[dict[str, dict], dict[tuple[str, str], str]]:
    """Returns (cache_by_key, key_by_case_side). Resumable, content-addressed
    -- a rerun with the same cases/context/schema/prompt makes zero new LLM
    calls. A changed prompt or SCHEMA_VERSION changes every cache_key, so
    stale entries are simply never looked up again (this file is append-only
    and immutable; old entries are harmless dead weight, never mutated)."""
    cached: dict[str, dict] = {}
    if CONTEXT_EXTRACTION_CACHE.exists():
        for line in CONTEXT_EXTRACTION_CACHE.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cached[row["cache_key"]] = row

    key_by_case_side: dict[tuple[str, str], str] = {}
    to_call: list[tuple[str, str, str, str, set[str]]] = []  # qa_id, side, key, prompt, window_ids
    seen_keys: set[str] = set()
    for case in cases:
        for side in ("old", "new"):
            ids = case[f"{side}_ids"]
            if not ids:
                continue
            context_text, window_ids = build_group_context(conversations, case["network_id"], ids)
            if not context_text:
                continue
            prompt = _context_extract_prompt(context_text)
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            key = _cache_key(case["network_id"], ids, prompt_hash)
            key_by_case_side[(case["qa_id"], side)] = key
            if key in cached or key in seen_keys:
                continue
            seen_keys.add(key)
            to_call.append((case["qa_id"], side, key, prompt, window_ids))

    if to_call:
        from llm.groq_client import get_chat_model

        print(f"Context-extracting {len(to_call)} anchor groups (Stage 4.7)...", file=sys.stderr)
        CONTEXT_EXTRACTION_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CONTEXT_EXTRACTION_CACHE.open("a") as out:
            for done, (qa_id, side, key, prompt, window_ids) in enumerate(to_call, 1):
                message = get_chat_model("fast", 0).invoke(prompt)
                parsed = _parse_json_object(message.content if isinstance(message.content, str) else "") or {}
                row = {
                    "cache_key": key, "qa_id": qa_id, "side": side, "schema_version": SCHEMA_VERSION,
                    "window_turn_ids": sorted(window_ids),
                    "raw_items": [item for item in parsed.get("items", []) if isinstance(item, dict)],
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                cached[key] = row
                if done % 10 == 0 or done == len(to_call):
                    print(f"  extracted {done}/{len(to_call)}", file=sys.stderr)
    return cached, key_by_case_side


# --------------------------------------------------------------------------
# Deterministic validation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextRecord:
    record_type: str
    claim: str
    viewpoint_owner: str
    subject: str
    state_description: Optional[str]
    value: Optional[str]
    temporal_mode: str
    from_value: Optional[str]
    to_value: Optional[str]
    related_state_description: Optional[str]
    source_turn_ids: tuple[str, ...]
    confidence: float
    observed_at: str


def _norm_temporal(value: str) -> str:
    lowered = (value or "").strip().lower()
    return lowered if lowered in VALID_TEMPORAL_MODES else "unknown"


def validate_context_record(
    raw: dict, valid_source_ids: set[str], anchor_ids: set[str], timestamp_by_turn: dict[str, str],
) -> tuple[Optional[ContextRecord], str]:
    """Returns (record_or_None, reason). reason == "ok" iff record is not
    None; otherwise it names exactly which rule rejected it, for the
    structural literal/meta/empty rejection counts."""
    record_type = str(raw.get("record_type") or "").strip().upper()
    if record_type not in VALID_RECORD_TYPES:
        return None, "unknown_record_type"

    source_ids = tuple(dict.fromkeys(
        str(source_id) for source_id in raw.get("source_turn_ids", []) if str(source_id) in valid_source_ids
    ))
    if not source_ids:
        return None, "no_valid_sources"
    if not (set(source_ids) & anchor_ids):
        return None, "no_anchor_turn_cited"

    claim = str(raw.get("claim") or "").strip()
    if not claim:
        return None, "empty_claim"

    state_description = raw.get("state_description")
    state_description = str(state_description).strip() or None if state_description else None
    value = raw.get("value")
    value = str(value).strip() if value not in (None, "") else None
    from_value = raw.get("from_value")
    from_value = str(from_value).strip() if from_value not in (None, "") else None
    to_value = raw.get("to_value")
    to_value = str(to_value).strip() if to_value not in (None, "") else None
    related_state_description = raw.get("related_state_description")
    related_state_description = str(related_state_description).strip() if related_state_description else None

    if record_type == "STATE":
        if not state_description or not value:
            return None, "state_missing_description_or_value"
        if _is_meta_value(value):
            return None, "meta_value_state"
    elif record_type == "TRANSITION":
        if not state_description and not from_value and not to_value:
            return None, "transition_missing_content"

    viewpoint_owner = str(raw.get("viewpoint_owner") or "").strip()
    subject = str(raw.get("subject") or "").strip()
    temporal_mode = _norm_temporal(str(raw.get("temporal_mode") or "unknown"))
    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(1.0, max(0.0, confidence))

    observed_at = max((timestamp_by_turn[s] for s in source_ids if s in timestamp_by_turn), default="")

    return ContextRecord(
        record_type=record_type, claim=claim, viewpoint_owner=viewpoint_owner, subject=subject,
        state_description=state_description, value=value, temporal_mode=temporal_mode,
        from_value=from_value, to_value=to_value, related_state_description=related_state_description,
        source_turn_ids=source_ids, confidence=confidence, observed_at=observed_at,
    ), "ok"


def validated_records_for_group(
    row: dict, conversations, network_id: str,
) -> tuple[list[ContextRecord], Counter]:
    window_ids = set(row["window_turn_ids"])
    anchor_ids = set(row.get("source_ids") or [])
    frame = conversations[(conversations.network_id == network_id) & (conversations.turn_id.isin(window_ids))]
    timestamp_by_turn = {str(r.turn_id): str(r.timestamp) for r in frame.itertuples(index=False)}
    records, reasons = [], Counter()
    for item in row.get("raw_items", []):
        record, reason = validate_context_record(item, window_ids, anchor_ids, timestamp_by_turn)
        reasons[reason] += 1
        if record is not None:
            records.append(record)
    return records, reasons


# --------------------------------------------------------------------------
# Structural comparison (Stage 4.6 frozen baseline vs Stage 4.7) -- counts
# only, NEVER a recall/usability claim (that needs the human review sheet).
# --------------------------------------------------------------------------

def summarize_baseline_a(cases: list[dict], oracle_cache: dict, conversations) -> dict[str, Any]:
    groups_processed = 0
    raw_total = 0
    valid_records = 0
    explicit_subject = 0
    explicit_viewpoint = 0
    cited_total = 0
    cited_valid_total = 0
    for case in cases:
        for side in ("old", "new"):
            ids = case[f"{side}_ids"]
            if not ids:
                continue
            row = oracle_cache.get((case["qa_id"], side))
            if row is None:
                continue
            groups_processed += 1
            raw_items = row.get("raw_items", [])
            raw_total += len(raw_items)
            for item in raw_items:
                if str(item.get("subject") or "").strip():
                    explicit_subject += 1
                if str(item.get("viewpoint_owner") or "").strip():
                    explicit_viewpoint += 1
                cited = [str(s) for s in item.get("source_turn_ids", [])]
                cited_total += len(cited)
                cited_valid_total += sum(1 for s in cited if s in ids)
            validated = _validated_side_assertions(case["qa_id"], side, case["network_id"], ids, raw_items, conversations)
            valid_records += len(validated)
    return {
        "groups_processed": groups_processed, "raw_records": raw_total, "valid_records": valid_records,
        "explicit_subject_rate": round(explicit_subject / raw_total, 3) if raw_total else 0.0,
        "explicit_viewpoint_owner_rate": round(explicit_viewpoint / raw_total, 3) if raw_total else 0.0,
        "source_id_validity_rate": round(cited_valid_total / cited_total, 3) if cited_total else 0.0,
        "by_type": "n/a (Stage 4.6 schema has no record_type)",
        "state_count": "n/a", "transition_count": "n/a",
        "literal_meta_empty_rejected": "n/a (Stage 4.6 validate_assertion has no meta-value/empty-claim check -- "
                                        "that check only existed at linking time in decide_state_operation)",
    }


def summarize_stage_b(
    cases: list[dict], context_cache: dict, key_by_case_side: dict, conversations,
) -> tuple[dict[str, Any], dict[tuple[str, str], list[ContextRecord]]]:
    groups_processed = 0
    raw_total = 0
    valid_records = 0
    explicit_subject = 0
    explicit_viewpoint = 0
    cited_total = 0
    cited_valid_total = 0
    by_type: Counter = Counter()
    reject_reasons: Counter = Counter()
    records_by_case_side: dict[tuple[str, str], list[ContextRecord]] = {}

    for case in cases:
        for side in ("old", "new"):
            ids = case[f"{side}_ids"]
            if not ids:
                continue
            key = key_by_case_side.get((case["qa_id"], side))
            row = context_cache.get(key) if key else None
            if row is None:
                continue
            groups_processed += 1
            raw_items = row.get("raw_items", [])
            raw_total += len(raw_items)
            for item in raw_items:
                if str(item.get("subject") or "").strip():
                    explicit_subject += 1
                if str(item.get("viewpoint_owner") or "").strip():
                    explicit_viewpoint += 1
                cited = [str(s) for s in item.get("source_turn_ids", [])]
                cited_total += len(cited)
                cited_valid_total += sum(1 for s in cited if s in set(row["window_turn_ids"]))

            records, reasons = validated_records_for_group(
                {**row, "source_ids": sorted(ids)}, conversations, case["network_id"],
            )
            records_by_case_side[(case["qa_id"], side)] = records
            valid_records += len(records)
            reject_reasons.update(reasons)
            for record in records:
                by_type[record.record_type] += 1

    reject_reasons.pop("ok", None)
    summary = {
        "groups_processed": groups_processed, "raw_records": raw_total, "valid_records": valid_records,
        "explicit_subject_rate": round(explicit_subject / raw_total, 3) if raw_total else 0.0,
        "explicit_viewpoint_owner_rate": round(explicit_viewpoint / raw_total, 3) if raw_total else 0.0,
        "source_id_validity_rate": round(cited_valid_total / cited_total, 3) if cited_total else 0.0,
        "by_type": dict(by_type),
        "state_count": by_type.get("STATE", 0), "transition_count": by_type.get("TRANSITION", 0),
        "cause_count": by_type.get("CAUSE", 0), "reaction_count": by_type.get("REACTION", 0),
        "rejected_by_reason": dict(reject_reasons),
    }
    return summary, records_by_case_side


def render_structural_report(summary_a: dict, summary_b: dict, new_llm_calls: int) -> str:
    lines = [
        "# Stage 4.7 -- structural comparison (Stage 4.6 frozen baseline vs Stage 4.7 context-aware)",
        "",
        "Structural counts ONLY -- no recall, no 'usable', no semantic-correctness claim. Those require "
        "the human review sheet.",
        "",
        f"- new LLM calls this run: {new_llm_calls}",
        f"- immutable cache: {CONTEXT_EXTRACTION_CACHE}",
        "",
        "| metric | A: Stage 4.6 (anchor-only, frozen) | B: Stage 4.7 (anchor +/-2 context) |",
        "|---|---|---|",
        f"| groups processed | {summary_a['groups_processed']} | {summary_b['groups_processed']} |",
        f"| raw records (pre-validation) | {summary_a['raw_records']} | {summary_b['raw_records']} |",
        f"| valid records (post-validation) | {summary_a['valid_records']} | {summary_b['valid_records']} |",
        f"| source-ID validity rate | {summary_a['source_id_validity_rate']} | {summary_b['source_id_validity_rate']} |",
        f"| fraction with explicit subject | {summary_a['explicit_subject_rate']} | {summary_b['explicit_subject_rate']} |",
        f"| fraction with explicit viewpoint_owner | {summary_a['explicit_viewpoint_owner_rate']} | {summary_b['explicit_viewpoint_owner_rate']} |",
        f"| STATE records | {summary_a['state_count']} | {summary_b['state_count']} |",
        f"| TRANSITION records | {summary_a['transition_count']} | {summary_b['transition_count']} |",
        f"| CAUSE records | n/a | {summary_b.get('cause_count', 0)} |",
        f"| REACTION records | n/a | {summary_b.get('reaction_count', 0)} |",
        f"| literal/meta/empty rejected | {summary_a['literal_meta_empty_rejected']} | "
        f"{summary_b['rejected_by_reason']} |",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Human review sheet
# --------------------------------------------------------------------------

def render_extraction_review(
    q8_rows: list[dict], cases: list[dict], oracle_cache: dict,
    context_cache: dict, key_by_case_side: dict, records_by_case_side: dict, conversations,
) -> str:
    lines = [
        "# Stage 4.7 -- extraction review sheet (all 20 Q8 dev cases, manual review only)",
        "",
        "Question shown here ONLY for this review -- never passed to the Stage 4.6 or Stage 4.7 extractor. "
        "No linker ran on the Stage 4.7 output below; nothing here has been auto-classified.",
        "",
        "**3 pairs confirmed genuine in the manual Stage 4.6 review "
        "(/tmp/socialmembench_stage4_6_pair_review_v2.md):** Seb #1->#3 REVISE, Uncle Femi #2->#5 RETRACT, "
        "Diane #3->#6 REVISE.",
        "",
        "**9 Stage 4.6 cases marked 'No valid pair in extraction'** -- see if Stage 4.7's context-aware "
        "extraction recovers the missed state that Stage 4.6 could not.",
        "",
    ]

    row_by_qa_id = {row["qa_id"]: row for row in q8_rows}
    case_by_qa_id = {case["qa_id"]: case for case in cases}

    for i, qa_id in enumerate(sorted(case_by_qa_id), 1):
        row = row_by_qa_id[qa_id]
        case = case_by_qa_id[qa_id]
        network_id = case["network_id"]

        badge = ""
        if qa_id in CONFIRMED_PAIRS:
            c = CONFIRMED_PAIRS[qa_id]
            badge = f"  **[CONFIRMED Stage 4.6 pair -- {c['who']} {c['pair']} {c['relation']}]**"
        elif qa_id in NO_VALID_PAIR_QA_IDS:
            badge = "  **[Stage 4.6: No valid pair in extraction]**"

        lines.append(f"## {i}. {qa_id} :: {network_id}{badge}")
        lines.append("")
        lines.append(f"**Question:** {row['question']}")
        lines.append("")

        old_ids, new_ids = case["old_ids"], case["new_ids"]
        old_a = _validated_side_assertions(qa_id, "old", network_id, old_ids, oracle_cache.get((qa_id, "old"), {}).get("raw_items", []), conversations)
        new_a = _validated_side_assertions(qa_id, "new", network_id, new_ids, oracle_cache.get((qa_id, "new"), {}).get("raw_items", []), conversations)
        numbered_a = list(enumerate(old_a + new_a, 1))

        for side, ids in (("old", old_ids), ("new", new_ids)):
            if not ids:
                continue
            key = key_by_case_side.get((qa_id, side))
            row_b = context_cache.get(key) if key else None
            context_text = "(no context available)"
            if row_b is not None:
                context_text, _ = build_group_context(conversations, network_id, ids)
            lines.append(f"### {side.upper()} side context")
            lines.append("```")
            lines.append(context_text)
            lines.append("```")

            side_frozen = old_a if side == "old" else new_a
            offset = 0 if side == "old" else len(old_a)
            lines.append(f"**Stage 4.6 frozen records ({side}):**")
            if side_frozen:
                for local_idx, assertion in enumerate(side_frozen, 1):
                    lines.append(f"  #{offset + local_idx}. {assertion.assertion_text}")
            else:
                lines.append("  (none)")
            lines.append("")

            v2_records = records_by_case_side.get((qa_id, side), [])
            lines.append(f"**Stage 4.7 records ({side}):**")
            if v2_records:
                for record in v2_records:
                    detail = f"[{record.record_type}] {record.claim}"
                    if record.state_description or record.value:
                        detail += f"  (state_description={record.state_description!r}, value={record.value!r})"
                    if record.from_value or record.to_value:
                        detail += f"  (from={record.from_value!r}, to={record.to_value!r})"
                    lines.append(f"  - {detail}")
                    lines.append(f"    source_turn_ids={list(record.source_turn_ids)}")
            else:
                lines.append("  (none)")
            lines.append("")

        lines.append("**Manual review:**")
        lines.append("- Old state recovered by v2: [ ] yes  [ ] no  [ ] unclear")
        lines.append("- New state recovered by v2: [ ] yes  [ ] no  [ ] unclear")
        lines.append("- Cause recovered by v2: [ ] yes  [ ] no  [ ] n/a")
        lines.append("- Attribution correct: [ ] yes  [ ] no")
        lines.append("- Provenance sufficient: [ ] yes  [ ] no")
        lines.append("- Spurious records present: [ ] yes  [ ] no")
        lines.append("- Candidate old record #__ -> new record #__")
        lines.append("- Expected relation: [ ] OBSERVE  [ ] AUGMENT  [ ] REVISE  [ ] RETRACT")
        lines.append("- No valid pair: [ ]")
        lines.append("- Comment: ____")
        lines.append("")
        lines.append("---")
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
        old_ids, new_ids, _unassigned, _degenerate = _split_old_new(anchors)
        cases.append({"qa_id": row["qa_id"], "network_id": row["network_id"], "old_ids": old_ids, "new_ids": new_ids})

    cache_before = set()
    if CONTEXT_EXTRACTION_CACHE.exists():
        cache_before = {json.loads(l)["cache_key"] for l in CONTEXT_EXTRACTION_CACHE.read_text().splitlines() if l.strip()}

    context_cache, key_by_case_side = build_context_extraction_cache(cases, conversations)
    new_llm_calls = len(set(context_cache) - cache_before)

    oracle_cache: dict[tuple[str, str], dict] = {}
    for line in ORACLE_EXTRACTION_CACHE.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            oracle_cache[(row["qa_id"], row["side"])] = row

    summary_a = summarize_baseline_a(cases, oracle_cache, conversations)
    summary_b, records_by_case_side = summarize_stage_b(cases, context_cache, key_by_case_side, conversations)

    STRUCTURAL_REPORT_MD.write_text(render_structural_report(summary_a, summary_b, new_llm_calls))
    EXTRACTION_REVIEW_MD.write_text(render_extraction_review(
        q8_rows, cases, oracle_cache, context_cache, key_by_case_side, records_by_case_side, conversations,
    ))

    print(f"New LLM calls: {new_llm_calls}", file=sys.stderr)
    print(f"Immutable cache: {CONTEXT_EXTRACTION_CACHE}", file=sys.stderr)
    print(f"Structural report: {STRUCTURAL_REPORT_MD}", file=sys.stderr)
    print(f"Extraction review: {EXTRACTION_REVIEW_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
