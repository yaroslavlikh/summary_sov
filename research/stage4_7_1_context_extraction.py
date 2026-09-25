"""Stage 4.7.1 -- minimal extractor-contract correction: separates EVIDENCE
from ANCHOR ASSOCIATION.

Stage 4.7's rule "every record must cite at least one [ANCHOR] turn" caused a
real, confirmed provenance bug found during manual review of
/tmp/socialmembench_stage4_7_extraction_review.md: the model attached the
anchor's turn_id to facts that were actually only stated by NEIGHBORING
context turns (e.g. Priyanka/Nadia's own remarks got tagged with Seb's anchor
turn_id; Ngozi/Tolu/Chidi's remarks got tagged with Uncle Femi's anchor
turn_id). That is false provenance -- source_turn_ids must always mean "this
turn actually contains text supporting this claim", never "this record is
about the same topic as this anchor."

The fix, structurally:
  - The model no longer emits source_turn_ids directly. It emits `evidence`
    (a list of {turn_id, quote} pairs -- which raw turns literally support
    the claim, with an exact quoted substring proving it) and
    `related_anchor_ids` (which selected anchor(s) this record helps
    interpret -- NOT provenance, never converted into source_turn_ids).
  - Deterministic validation checks each quote is a real, verifiable
    substring of that turn's actual message (after only whitespace
    normalization -- this does not prove semantic entailment, but it does
    guarantee the cited turn contains the text the model claims to rely on).
    source_turn_ids is then DERIVED server-side, exclusively from validated
    evidence -- the model's word is never trusted for provenance.
  - Extraction scope is tightened: a record is only emitted if it states/
    clarifies the anchor participant's state, is an explicit transition
    involving the anchor, is a cause of the anchor's state/transition, or is
    a reaction to it -- not every unrelated fact that happens to appear in
    the ±2 window.
  - TRANSITION is only allowed when the visible evidence explicitly supplies
    change language or both the old and new state are literally visible
    across the shown turns -- inventing a missing "from" side (e.g.
    from_value="healthy" out of "my knee went/locked") is now a validation
    reject, not just a prompt request.

Nothing else changes: STATE/TRANSITION/CAUSE/REACTION stays exactly as Stage
4.7 defined it; still no facet/topic_key/state_key/operation/linking
decision; versioned_memory_cells.py, the linker (threshold or semantic),
Stage 5, Q6, and held-out QA are untouched by this module.

Cache is a NEW, separate, append-only file
(/tmp/socialmembench_stage4_7_1_context_extraction.jsonl, schema_version
"ctx_v1_1") -- Stage 4.7's cache is never overwritten or read for reuse
(the contract changed, so its cached outputs are not valid inputs to this
stage's validator).

This module's __main__ runs ONLY a 3-case smoke test (Cam, Seb, Uncle Femi)
by design -- the full 20-case/40-group run is deliberately not wired up
here until the smoke output has been manually inspected.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.socialmembench_pilot import _json, _norm, _parse_json_object, _select_questions, load_data
from research.stage4_5_audit import DATA_DIR, PER_TYPE, SEED, _split_old_new
from research.versioned_memory_cells import _is_meta_value

SCHEMA_VERSION = "ctx_v1_1"
CONTEXT_EXTRACTION_CACHE = Path("/tmp/socialmembench_stage4_7_1_context_extraction.jsonl")
SMOKE_REPORT_MD = Path("/tmp/socialmembench_stage4_7_1_smoke_report.md")
SMOKE_QA_IDS = ("Q8_347b5cff", "Q8_a9b0c1d216", "Q8_f6g7h8i901")  # Cam, Seb, Uncle Femi

VALID_RECORD_TYPES = {"STATE", "TRANSITION", "CAUSE", "REACTION"}
VALID_TEMPORAL_MODES = {"past", "current", "future", "habitual", "unknown"}

_TURN_NUM_RE = re.compile(r"t(\d+)$")
_WS_RE = re.compile(r"\s+")


def _turn_num(turn_id: str) -> int:
    match = _TURN_NUM_RE.search(turn_id or "")
    return int(match.group(1)) if match else 0


def _norm_ws(text: str) -> str:
    return _WS_RE.sub(" ", (text or "")).strip()


def _session_frame_sorted(conversations, network_id: str, session_index: int):
    frame = conversations[
        (conversations.network_id == network_id) & (conversations.session_index == session_index)
    ].copy()
    frame["_order"] = frame.turn_id.map(_turn_num)
    return frame.sort_values("_order").reset_index(drop=True)


def build_group_context(conversations, network_id: str, anchor_ids: set[str]) -> tuple[str, set[str], dict[str, str]]:
    """Unchanged from Stage 4.7: union of every anchor's +/-2 window in the
    same session, deduplicated, chronologically ordered, anchor turns marked
    [ANCHOR]. Returns (context_text, window_turn_ids, message_by_turn) --
    message_by_turn is needed here (new) to verify evidence quotes."""
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
    message_by_turn = {}
    for row in ordered:
        marker = "[ANCHOR] " if row.turn_id in anchor_ids else ""
        lines.append(f"[[{row.turn_id}]] {marker}{row.timestamp} {row.speaker_display_name}: {row.message}")
        message_by_turn[row.turn_id] = row.message
    return "\n".join(lines), set(window_rows.keys()), message_by_turn


def _context_extract_prompt(context_lines: str, anchor_ids: list[str]) -> str:
    anchors_list = ", ".join(anchor_ids)
    return f"""You will see a short window of chat turns from ONE session. Turns marked
[ANCHOR] are the ones under review; the rest are up to 2 turns of surrounding
context per anchor, included ONLY to help you resolve pronouns, ellipsis,
sarcasm, and short replies -- NOT as a general invitation to mine every fact
in the window. You do not know why these anchors were selected, what question
(if any) they answer, or what relation (if any) is expected.

The anchor turn_ids in this window are: {anchors_list}

ONLY emit a record when it does at least one of these, about the ANCHOR
PARTICIPANT (the person speaking in an [ANCHOR] turn) or their state:
- states or clarifies that participant's state (health, opinion, intention,
  commitment, availability, role, location, social position, group norm...);
- expresses an explicit transition involving that participant;
- is a cause of that participant's state/transition;
- is a reaction (by someone else, or the group) to that participant's state
  or transition.

Do NOT extract unrelated logistics or independent facts merely because they
appear in the surrounding window (e.g. a neighboring message about an
unrelated meeting time must NOT be extracted just because it sits next to the
anchor, unless the text explicitly connects it to the anchor's state).

Each record has a record_type:
- STATE: an observed value of an ongoing property. Must include
  state_description and value.
- TRANSITION: ONLY when the visible evidence explicitly supplies change
  language ("used to X now Y", "no longer", "started/stopped", "changed my
  mind", "this time I decided", "actually I now think") OR both an old and a
  new state are literally visible across the shown turns. NEVER invent the
  missing side of a transition -- e.g. "my knee went/locked" alone is a
  STATE (value="locked"), NOT a TRANSITION with an invented
  from_value="healthy". If only the new value is explicit, emit STATE, not
  TRANSITION.
- CAUSE: an event/explanation that plausibly caused the anchor participant's
  state to change. Not itself an old/new state version. Set
  related_state_description only if the text actually supports the link.
- REACTION: another participant's (or the group's) reaction to the anchor
  participant's state/transition. Must cite the REACTING person's OWN turn
  as evidence (not the anchor's turn) -- a reaction is not a new state
  version of the anchor participant.

For EVERY record, provide:
- evidence: a list of {{"turn_id": "...", "quote": "..."}} -- turn_id must be
  one of the turn_ids shown above (verbatim), quote must be a short EXACT
  substring copied from that turn's actual message (this will be verified
  mechanically -- an invented or paraphrased quote gets the evidence entry
  dropped, and the whole record rejected if nothing valid remains). An
  evidence turn does NOT have to be an [ANCHOR] turn -- cite whichever turn
  actually contains the words you're relying on. A REACTION's evidence must
  be the reacting person's own turn, not the anchor's.
- related_anchor_ids: one or more of the anchor turn_ids listed above that
  this record helps interpret. This is NOT provenance -- it will never be
  used as a source_turn_id. Every record needs at least one, but it does
  NOT need to also appear in evidence.

viewpoint_owner: whose view/knowledge/state/commitment this is.
subject: who or what the claim is actually about.
temporal_mode: one of past, current, future, habitual, unknown.
confidence: 0.0-1.0.

Do NOT include facet, topic_key, state_key, operation, memory_worthy, a
reference to an existing cell, or any NEW_CELL/OBSERVE/AUGMENT/REVISE/RETRACT
decision.

0 items is a fine answer. Do not take sarcasm literally. Do not swap the
person who holds an opinion with the person the opinion is about.

Return JSON only:
{{"items":[{{"record_type":"STATE|TRANSITION|CAUSE|REACTION","claim":"...",
"viewpoint_owner":"...","subject":"...","state_description":"...","value":"...",
"temporal_mode":"unknown","from_value":null,"to_value":null,
"related_state_description":null,
"evidence":[{{"turn_id":"turn_id","quote":"exact short substring"}}],
"related_anchor_ids":["turn_id"],"confidence":0.0}}]}}

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
    cached: dict[str, dict] = {}
    if CONTEXT_EXTRACTION_CACHE.exists():
        for line in CONTEXT_EXTRACTION_CACHE.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cached[row["cache_key"]] = row

    key_by_case_side: dict[tuple[str, str], str] = {}
    to_call: list[tuple[str, str, str, str]] = []
    seen_keys: set[str] = set()
    for case in cases:
        for side in ("old", "new"):
            ids = case[f"{side}_ids"]
            if not ids:
                continue
            context_text, window_ids, _messages = build_group_context(conversations, case["network_id"], ids)
            if not context_text:
                continue
            prompt = _context_extract_prompt(context_text, sorted(ids))
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            key = _cache_key(case["network_id"], ids, prompt_hash)
            key_by_case_side[(case["qa_id"], side)] = key
            if key in cached or key in seen_keys:
                continue
            seen_keys.add(key)
            to_call.append((case["qa_id"], side, key, prompt))

    if to_call:
        from llm.groq_client import get_chat_model

        print(f"Context-extracting {len(to_call)} anchor groups (Stage 4.7.1)...", file=sys.stderr)
        CONTEXT_EXTRACTION_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CONTEXT_EXTRACTION_CACHE.open("a") as out:
            for done, (qa_id, side, key, prompt) in enumerate(to_call, 1):
                message = get_chat_model("fast", 0).invoke(prompt)
                parsed = _parse_json_object(message.content if isinstance(message.content, str) else "") or {}
                row = {
                    "cache_key": key, "qa_id": qa_id, "side": side, "schema_version": SCHEMA_VERSION,
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
class ContextRecordV2:
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
    evidence: tuple[dict, ...]  # validated {"turn_id":..., "quote":...} entries
    related_anchor_ids: tuple[str, ...]
    source_turn_ids: tuple[str, ...]  # DERIVED server-side from evidence, never model-trusted
    confidence: float
    observed_at: str


def _norm_temporal(value: str) -> str:
    lowered = (value or "").strip().lower()
    return lowered if lowered in VALID_TEMPORAL_MODES else "unknown"


def validate_context_record(
    raw: dict, window_turn_ids: set[str], anchor_ids: set[str], message_by_turn: dict[str, str],
    timestamp_by_turn: dict[str, str],
) -> tuple[Optional[ContextRecordV2], str]:
    """Returns (record_or_None, reason). Core Stage 4.7.1 fix: evidence and
    related_anchor_ids are validated SEPARATELY, and source_turn_ids is
    derived ONLY from validated evidence -- related_anchor_ids never
    contributes to provenance."""
    record_type = str(raw.get("record_type") or "").strip().upper()
    if record_type not in VALID_RECORD_TYPES:
        return None, "unknown_record_type"

    valid_evidence = []
    for entry in raw.get("evidence", []):
        if not isinstance(entry, dict):
            continue
        turn_id = str(entry.get("turn_id") or "")
        quote = str(entry.get("quote") or "")
        if turn_id not in window_turn_ids:
            continue
        if not quote.strip():
            continue
        normalized_quote = _norm_ws(quote)
        normalized_message = _norm_ws(message_by_turn.get(turn_id, ""))
        if not normalized_quote or normalized_quote not in normalized_message:
            continue
        valid_evidence.append({"turn_id": turn_id, "quote": quote.strip()})
    if not valid_evidence:
        return None, "no_valid_evidence"

    valid_related_anchor_ids = tuple(dict.fromkeys(
        str(a) for a in raw.get("related_anchor_ids", []) if str(a) in anchor_ids
    ))
    if not valid_related_anchor_ids:
        return None, "no_valid_related_anchor"

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

    source_turn_ids = tuple(dict.fromkeys(entry["turn_id"] for entry in valid_evidence))
    observed_at = max((timestamp_by_turn[t] for t in source_turn_ids if t in timestamp_by_turn), default="")

    return ContextRecordV2(
        record_type=record_type, claim=claim, viewpoint_owner=viewpoint_owner, subject=subject,
        state_description=state_description, value=value, temporal_mode=temporal_mode,
        from_value=from_value, to_value=to_value, related_state_description=related_state_description,
        evidence=tuple(valid_evidence), related_anchor_ids=valid_related_anchor_ids,
        source_turn_ids=source_turn_ids, confidence=confidence, observed_at=observed_at,
    ), "ok"


def dedupe_records(records: list[ContextRecordV2]) -> list[ContextRecordV2]:
    """Only exact/near-exact records are merged -- same record_type,
    normalized viewpoint_owner, normalized subject, normalized claim/state
    description, and same validated evidence turn IDs. NEVER merges on
    embedding similarity. related_anchor_ids is unioned on a true
    duplicate; every other field is kept from the first occurrence."""
    by_key: dict[tuple, ContextRecordV2] = {}
    order: list[tuple] = []
    for record in records:
        description_key = _norm(record.state_description or record.claim)
        key = (
            record.record_type, _norm(record.viewpoint_owner), _norm(record.subject),
            description_key, record.source_turn_ids,
        )
        if key in by_key:
            existing = by_key[key]
            merged_anchor_ids = tuple(dict.fromkeys(existing.related_anchor_ids + record.related_anchor_ids))
            by_key[key] = _dc_replace(existing, related_anchor_ids=merged_anchor_ids)
        else:
            by_key[key] = record
            order.append(key)
    return [by_key[key] for key in order]


def process_group(
    qa_id: str, side: str, network_id: str, anchor_ids: set[str], raw_items: list[dict],
    conversations,
) -> tuple[list[ContextRecordV2], list[tuple[dict, str]]]:
    """Returns (deduped_valid_records, [(raw_item, reject_reason), ...])."""
    context_text, window_ids, message_by_turn = build_group_context(conversations, network_id, anchor_ids)
    frame = conversations[(conversations.network_id == network_id) & (conversations.turn_id.isin(window_ids))]
    timestamp_by_turn = {str(r.turn_id): str(r.timestamp) for r in frame.itertuples(index=False)}

    accepted: list[ContextRecordV2] = []
    rejected: list[tuple[dict, str]] = []
    for item in raw_items:
        record, reason = validate_context_record(item, window_ids, anchor_ids, message_by_turn, timestamp_by_turn)
        if record is None:
            rejected.append((item, reason))
        else:
            accepted.append(record)
    return dedupe_records(accepted), rejected


# --------------------------------------------------------------------------
# Smoke report
# --------------------------------------------------------------------------

def render_smoke_report(
    q8_rows: list[dict], cases: list[dict], cache: dict, key_by_case_side: dict, conversations,
) -> str:
    lines = [
        "# Stage 4.7.1 -- 3-case smoke test (evidence/anchor-association separated)",
        "",
        "Question shown here ONLY for this review, never passed to the extractor. Every accepted record's "
        "source_turn_ids is DERIVED from validated evidence only -- related_anchor_ids is shown separately "
        "and is never provenance. Rejected raw items are shown with their rejection reason.",
        "",
    ]
    row_by_qa_id = {row["qa_id"]: row for row in q8_rows}
    case_by_qa_id = {case["qa_id"]: case for case in cases}

    for i, qa_id in enumerate(SMOKE_QA_IDS, 1):
        row = row_by_qa_id[qa_id]
        case = case_by_qa_id[qa_id]
        network_id = case["network_id"]
        lines.append(f"## {i}. {qa_id} :: {network_id}")
        lines.append("")
        lines.append(f"**Question:** {row['question']}")
        lines.append("")

        for side in ("old", "new"):
            ids = case[f"{side}_ids"]
            if not ids:
                continue
            key = key_by_case_side.get((qa_id, side))
            cache_row = cache.get(key) if key else None
            if cache_row is None:
                continue
            context_text, window_ids, _messages = build_group_context(conversations, network_id, ids)
            lines.append(f"### {side.upper()} side context (anchors: {sorted(ids)})")
            lines.append("```")
            lines.append(context_text)
            lines.append("```")

            accepted, rejected = process_group(qa_id, side, network_id, ids, cache_row.get("raw_items", []), conversations)

            lines.append(f"**Accepted records ({len(accepted)}):**")
            if accepted:
                for record in accepted:
                    lines.append(f"- [{record.record_type}] {record.claim}")
                    if record.state_description or record.value:
                        lines.append(f"    state_description={record.state_description!r}  value={record.value!r}")
                    if record.from_value or record.to_value:
                        lines.append(f"    from_value={record.from_value!r}  to_value={record.to_value!r}")
                    if record.related_state_description:
                        lines.append(f"    related_state_description={record.related_state_description!r}")
                    lines.append(f"    viewpoint_owner={record.viewpoint_owner!r}  subject={record.subject!r}")
                    lines.append("    evidence:")
                    for e in record.evidence:
                        lines.append(f"      - [[{e['turn_id']}]] \"{e['quote']}\"")
                    lines.append(f"    derived source_turn_ids: {list(record.source_turn_ids)}")
                    lines.append(f"    related_anchor_ids: {list(record.related_anchor_ids)}")
            else:
                lines.append("  (none)")
            lines.append("")

            lines.append(f"**Rejected raw items ({len(rejected)}):**")
            if rejected:
                for item, reason in rejected:
                    lines.append(f"- reason={reason}  record_type={item.get('record_type')!r}  claim={item.get('claim')!r}")
                    lines.append(f"    raw evidence: {item.get('evidence')}")
                    lines.append(f"    raw related_anchor_ids: {item.get('related_anchor_ids')}")
            else:
                lines.append("  (none)")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    conversations, qa, _personas = load_data(DATA_DIR)
    selected = _select_questions(qa, PER_TYPE, SEED)
    q8_rows = [row for row in selected if row["query_type"] == "Q8"]
    assert len(q8_rows) == 20, f"expected 20 Q8 cases, got {len(q8_rows)}"

    smoke_rows = [row for row in q8_rows if row["qa_id"] in SMOKE_QA_IDS]
    assert len(smoke_rows) == 3, f"expected 3 smoke cases, got {len(smoke_rows)}"

    cases = []
    for row in smoke_rows:
        anchors = _json(row["evidence_anchors_json"], [])
        old_ids, new_ids, _unassigned, _degenerate = _split_old_new(anchors)
        cases.append({"qa_id": row["qa_id"], "network_id": row["network_id"], "old_ids": old_ids, "new_ids": new_ids})

    cache_before = set()
    if CONTEXT_EXTRACTION_CACHE.exists():
        cache_before = {json.loads(l)["cache_key"] for l in CONTEXT_EXTRACTION_CACHE.read_text().splitlines() if l.strip()}

    cache, key_by_case_side = build_context_extraction_cache(cases, conversations)
    new_llm_calls = len(set(cache) - cache_before)

    SMOKE_REPORT_MD.write_text(render_smoke_report(smoke_rows, cases, cache, key_by_case_side, conversations))

    print(f"New LLM calls (smoke, 3 cases): {new_llm_calls}", file=sys.stderr)
    print(f"Cache: {CONTEXT_EXTRACTION_CACHE}", file=sys.stderr)
    print(f"Smoke report: {SMOKE_REPORT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
