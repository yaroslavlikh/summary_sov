"""Temporal episode prototype -- research-only, isolated.

Not a graph, not a portrait/static-summary system. Bottleneck analysis from
the semantic-key linking work: on clean oracle pairs the provenance-grounded
gate got 21/59 correct at precision=1.0, but on a streaming 20-case smoke
only 2-3 links out of 63 extracted records happened at all -- the bottleneck
is upstream, in generic extraction (`ContextRecordV2.state_description`)
producing unstable units of memory, not in the linking gate itself.

This prototype replaces that pipeline stage-for-stage with something
simpler and deliberately NOT identity-keyed by any generic field:

    raw message + local context
            v
    attributable MemoryEvent with validated provenance
            v
    retrieve top-3 live temporal episodes (embedding only -- no hard filter
    beyond network_id, no slot-key equality)
            v
    primary model: ATTACH_REVISE / ATTACH_REAFFIRM / ATTACH_AUGMENT / NEW_EPISODE
            v
    episode stores chronological immutable evidence events

An episode is not a graph node -- it is a small, provenance-backed
chronology of one evolving storyline (e.g. "Tigmen's opinion of Gordey: 12
May 'asshole' [[msg1]] -> 18 May 'idiot' [[msg2]] -> 2 June 'decent guy'
[[msg3]]"), with an immutable, append-only event list, at most 5 "hot"
(active-representation) events, and older evidence retained but excluded
from the active representation -- never deleted.

Old baseline VERSIONED (research/versioned_memory_cells.py,
research/stage4_8_versioned_experiment.py) and the semantic_key_*/
semantic_slot_* diagnostic runs are FROZEN reference points here -- this
module does not modify, re-tune, or re-run any of them. Production `/ask`,
`llm/graphs.py`, Telegram, the database, and the full SocialMemBench run are
untouched.

Uses `build_session_clusters` (stage4_8_versioned_experiment.py) and
`build_group_context` (stage4_7_1_context_extraction.py) read-only, purely
for turn-window construction -- their own extraction/validation logic is
NOT reused; this module defines its own MemoryEvent schema, prompt, and
validator from scratch, as instructed.

Run:
    python3 -m research.temporal_episode_prototype          # 3-case smoke (Seb, Uncle Femi, Diane)
    python3 -m research.temporal_episode_prototype --full   # all 20 Q8 dev cases -- materialization/linking only,
                                                              # still no answer generation, judges, held-out, or production
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from embeddings import embed, embed_batch
from research.socialmembench_pilot import _json, _parse_json_object, _select_questions, load_data
from research.stage4_5_audit import DATA_DIR, PER_TYPE, SEED
from research.stage4_7_1_context_extraction import build_group_context
from research.stage4_8_versioned_experiment import build_session_clusters

SCHEMA_VERSION = "temporal_episode_v2"  # bumped: evidence/quote contract replaces raw source_turn_ids trust
EVENT_CACHE = Path("/tmp/temporal_episode_event_cache.jsonl")
ATTACH_CACHE = Path("/tmp/temporal_episode_attach_cache.jsonl")
SMOKE_REPORT_MD = Path("/tmp/temporal_episode_smoke_report.md")

VALID_EVENT_TYPES = {"state", "opinion", "commitment", "decision", "relationship", "exception", "other"}
VALID_TEMPORAL_MODES = {"past", "current", "future", "habitual", "unknown"}
VALID_DECISIONS = {"ATTACH_REVISE", "ATTACH_REAFFIRM", "ATTACH_AUGMENT", "NEW_EPISODE"}
ATTACH_CONFIDENCE_FLOOR = 0.5
MAX_HOT_EVENTS = 5
TOP_K_CANDIDATES = 3
MAX_EVENTS_PER_SOURCE_TURN = 2

SMOKE_QA_IDS = ("Q8_a9b0c1d216", "Q8_f6g7h8i901", "Q8_ph9s4c1")  # Seb, Uncle Femi, Diane
FULL_DEV_REPORT_MD = Path("/tmp/temporal_episode_full20_report.md")


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().casefold()


# --------------------------------------------------------------------------
# MemoryEvent extraction
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    network_id: str
    event_text: str
    event_type: str
    viewpoint_owner: Optional[str]
    subject: Optional[str]
    evidence: tuple[dict, ...]  # validated {"turn_id": ..., "quote": ...} entries
    source_turn_ids: tuple[str, ...]  # DERIVED from validated evidence only, never model-trusted directly
    local_context_turn_ids: tuple[str, ...]
    temporal_mode: str
    observed_at: str  # server-derived from real turn timestamps, never trusted from the model


def _event_extraction_prompt(context_lines: str) -> str:
    return f"""You will see a window of chat turns from ONE session. Turns marked
[ANCHOR] were specifically selected for review; the rest are surrounding
context (up to 2 turns before/after each anchor), shown only to help you
resolve pronouns, ellipsis, sarcasm, and short replies. You do not know why
these turns were selected or what question (if any) they answer.

Extract only DURABLE, externally useful memory events -- a state, a change,
a commitment, a decision, or a relationship fact that someone could actually
need to recall later. Do NOT extract conversational fragments, agreement
markers, vague reactions, or generic self-reports -- for example "X is
right", "this matters", "I am thinking", "leave it to me", "okay", "sounds
good" are NOT events UNLESS they explicitly introduce or revise a concrete
decision/state that is stated in the source (in which case extract THAT
concrete content, not the filler phrase itself). 0 items is a fine answer,
and is expected for most turns -- do not force an event out of small talk.

At most 2 durable events per turn shown here (zero is normal; most turns
produce none).

Each event:
- event_text: a SELF-CONTAINED description (resolve pronouns/ellipsis using
  context -- never leave a bare filler phrase unresolved; if you cannot
  resolve it into something concrete, do not extract it).
- event_type: one of state, opinion, commitment, decision, relationship,
  exception, other.
- viewpoint_owner: who holds/asserts this (null if not an opinion/report).
- subject: who or what the event is about (may be null if unclear -- do not
  guess).
- evidence: one or more {{"turn_id": "...", "quote": "..."}} objects --
  turn_id copied EXACTLY from the turn_ids shown, quote a SHORT EXACT
  substring copied verbatim from that turn's actual message (this will be
  mechanically verified -- an invented or paraphrased quote is dropped, and
  the whole event is rejected if no valid quote remains). At least one
  required.
- local_context_turn_ids: any additional shown turns (anchor or context)
  that helped you interpret this event -- may overlap with evidence turn_ids,
  may be empty.
- temporal_mode: one of past, current, future, habitual, unknown.

Return JSON only:
{{"items":[{{"event_text":"...","event_type":"state|opinion|commitment|decision|relationship|exception|other",
"viewpoint_owner":"... or null","subject":"... or null",
"evidence":[{{"turn_id":"turn_id","quote":"exact short substring"}}],
"local_context_turn_ids":["turn_id"],"temporal_mode":"past|current|future|habitual|unknown"}}]}}

Turns:
{context_lines}"""


def _norm_temporal(value: str) -> str:
    lowered = (value or "").strip().lower()
    return lowered if lowered in VALID_TEMPORAL_MODES else "unknown"


_WS_RE = re.compile(r"\s+")
_GENERIC_FRAGMENT_RE = re.compile(
    r"^("
    r"(this|that|it)( is| matters| works| sounds good)?"
    r"|i(?:'m| am) (thinking|considering)"
    r"|leave it to me"
    r"|sounds good"
    r"|ok(ay|ey)?|alright|fine|sure|yes|yeah|yep|nah|no problem"
    r"|noted|got it|understood|i hear you"
    r"|[a-z' ]{1,20} is right"
    r")[.!]*$",
    re.IGNORECASE,
)


def _is_generic_fragment(event_text: str) -> bool:
    """Deterministic safety net behind the prompt instruction -- rejects
    ONLY when the entire event_text (not a substring of a longer, concrete
    claim) is one of the given filler/agreement-marker patterns. A longer
    sentence that happens to start with "okay" or contain "is right" as
    part of substantive content will NOT match (fullmatch), so this never
    punishes a real event for incidental phrasing."""
    normalized = _WS_RE.sub(" ", event_text or "").strip()
    return bool(_GENERIC_FRAGMENT_RE.fullmatch(normalized))


def _quote_is_valid_substring(quote: str, message: str) -> bool:
    normalized_quote = _WS_RE.sub(" ", (quote or "")).strip()
    if not normalized_quote:
        return False
    normalized_message = _WS_RE.sub(" ", (message or "")).strip()
    return normalized_quote in normalized_message


def _norm_turn_id(value: Any) -> str:
    """The extractor occasionally echoes a turn_id wrapped in the literal
    '[[...]]'/'[...]' citation-bracket characters it sees used elsewhere in
    this project's prompts (e.g. "cite as [[turn_id]]"), producing a STRING
    like "[grp_x_s06_t001]" instead of "grp_x_s06_t001" -- observed on the
    real smoke run (10 otherwise-valid Uncle Femi events wrongly rejected).
    Strips only literal leading/trailing bracket characters and whitespace;
    never invents or guesses an id that wasn't already present."""
    return str(value).strip().strip("[]").strip()


def _resolve_turn_id(value: Any, window_turn_ids: set[str]) -> str:
    """Restore a model-dropped namespace only when the window makes it unique."""
    turn_id = _norm_turn_id(value)
    if turn_id in window_turn_ids:
        return turn_id
    matches = [candidate for candidate in window_turn_ids if candidate.rsplit("|", 1)[-1] == turn_id]
    return matches[0] if len(matches) == 1 else turn_id


def validate_memory_event(
    raw: dict, window_turn_ids: set[str], timestamp_by_turn: dict[str, str], message_by_turn: dict[str, str],
    network_id: str, event_index: int,
) -> tuple[Optional[MemoryEvent], str, int, int]:
    """Returns (record_or_None, reason, evidence_attempted, evidence_valid)
    -- the last two counts feed the quote-valid provenance rate in the
    report regardless of whether the event as a whole is accepted."""
    event_text = str(raw.get("event_text") or "").strip()
    if not event_text:
        return None, "empty_event_text", 0, 0

    event_type = str(raw.get("event_type") or "").strip().lower()
    if event_type not in VALID_EVENT_TYPES:
        return None, "invalid_event_type", 0, 0

    raw_evidence = raw.get("evidence", [])
    evidence_attempted = len(raw_evidence) if isinstance(raw_evidence, list) else 0
    valid_evidence: list[dict] = []
    if isinstance(raw_evidence, list):
        for entry in raw_evidence:
            if not isinstance(entry, dict):
                continue
            turn_id = _resolve_turn_id(entry.get("turn_id"), window_turn_ids)
            quote = str(entry.get("quote") or "")
            if turn_id not in window_turn_ids:
                continue
            if not _quote_is_valid_substring(quote, message_by_turn.get(turn_id, "")):
                continue
            valid_evidence.append({"turn_id": turn_id, "quote": quote.strip()})
    if not valid_evidence:
        return None, "no_valid_evidence_quote", evidence_attempted, 0

    if _is_generic_fragment(event_text):
        return None, "generic_fragment_or_agreement_marker", evidence_attempted, len(valid_evidence)

    source_ids = tuple(dict.fromkeys(e["turn_id"] for e in valid_evidence))

    local_ids = tuple(dict.fromkeys(
        resolved for s in raw.get("local_context_turn_ids", [])
        if (resolved := _resolve_turn_id(s, window_turn_ids)) in window_turn_ids
    ))

    viewpoint_owner = raw.get("viewpoint_owner")
    viewpoint_owner = str(viewpoint_owner).strip() if viewpoint_owner else None
    subject = raw.get("subject")
    subject = str(subject).strip() if subject else None

    observed_at = max((timestamp_by_turn[t] for t in source_ids if t in timestamp_by_turn), default="")
    if not observed_at:
        return None, "no_resolvable_timestamp", evidence_attempted, len(valid_evidence)

    return MemoryEvent(
        event_id=f"{network_id}:evt{event_index}", network_id=network_id, event_text=event_text,
        event_type=event_type, viewpoint_owner=viewpoint_owner, subject=subject,
        evidence=tuple(valid_evidence), source_turn_ids=source_ids, local_context_turn_ids=local_ids,
        temporal_mode=_norm_temporal(str(raw.get("temporal_mode") or "unknown")), observed_at=observed_at,
    ), "ok", evidence_attempted, len(valid_evidence)


def _cache_key(network_id: str, anchor_ids: set[str], prompt_hash: str) -> str:
    payload = {"network_id": network_id, "anchor_ids": sorted(anchor_ids), "schema": SCHEMA_VERSION, "prompt_hash": prompt_hash}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_cache_dict(path: Path) -> dict[str, dict]:
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["cache_key"]] = row
    return cache


def _append_cache_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _call_llm(prompt: str) -> dict:
    from llm.groq_client import get_chat_model

    message = get_chat_model("fast", 0).invoke(prompt)
    return _parse_json_object(message.content if isinstance(message.content, str) else "") or {}


def _enforce_max_events_per_turn(events: list[MemoryEvent]) -> tuple[list[MemoryEvent], list[dict]]:
    """At most MAX_EVENTS_PER_SOURCE_TURN durable events may cite any given
    turn. Processes events in extraction order (stable, deterministic);
    an event is dropped only if keeping it would push ANY of its cited
    turns over the cap -- earlier events always win ties."""
    counts: dict[str, int] = defaultdict(int)
    kept: list[MemoryEvent] = []
    dropped: list[dict] = []
    for event in events:
        if any(counts[t] >= MAX_EVENTS_PER_SOURCE_TURN for t in event.source_turn_ids):
            dropped.append({"network_id": event.network_id, "reason": "exceeds_max_events_per_turn",
                             "event_id": event.event_id, "event_text": event.event_text})
            continue
        for t in event.source_turn_ids:
            counts[t] += 1
        kept.append(event)
    return kept, dropped


def extract_events_for_network(
    network_id: str, clusters: list[tuple[int, set[str]]], conversations: pd.DataFrame,
) -> tuple[list[MemoryEvent], list[dict], int, dict[str, int]]:
    """Runs extraction over every chronological session cluster for this
    network (built purely from QA gold anchor turn_ids for SCOPING which
    conversation regions to process -- never from question/answer/relevance
    text, which are never read here). Returns (valid_events, rejected_rows,
    new_llm_calls, quote_stats)."""
    event_cache = _load_cache_dict(EVENT_CACHE)
    events: list[MemoryEvent] = []
    rejected: list[dict] = []
    new_calls = 0
    event_index = 0
    evidence_attempted_total = 0
    evidence_valid_total = 0

    for _session_index, anchor_ids in clusters:
        context_text, window_ids, message_by_turn = build_group_context(conversations, network_id, anchor_ids)
        if not context_text:
            continue
        prompt = _event_extraction_prompt(context_text)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        key = _cache_key(network_id, anchor_ids, prompt_hash)
        row = event_cache.get(key)
        if row is None:
            output = _call_llm(prompt)
            row = {"cache_key": key, "network_id": network_id, "schema_version": SCHEMA_VERSION,
                   "raw_items": [item for item in output.get("items", []) if isinstance(item, dict)]}
            _append_cache_row(EVENT_CACHE, row)
            event_cache[key] = row
            new_calls += 1

        frame = conversations[(conversations.network_id == network_id) & (conversations.turn_id.isin(window_ids))]
        timestamp_by_turn = {str(r.turn_id): str(r.timestamp) for r in frame.itertuples(index=False)}
        for item in row.get("raw_items", []):
            record, reason, attempted, valid = validate_memory_event(
                item, window_ids, timestamp_by_turn, message_by_turn, network_id, event_index,
            )
            event_index += 1
            evidence_attempted_total += attempted
            evidence_valid_total += valid
            if record is None:
                rejected.append({"network_id": network_id, "reason": reason, "raw": item})
            else:
                events.append(record)

    events, cap_dropped = _enforce_max_events_per_turn(events)
    rejected.extend(cap_dropped)

    quote_stats = {"evidence_attempted": evidence_attempted_total, "evidence_valid": evidence_valid_total}
    return events, rejected, new_calls, quote_stats


# --------------------------------------------------------------------------
# TemporalEpisode + retrieval
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodeEvent:
    event: MemoryEvent
    decision: str  # create | attach_revise | attach_reaffirm | attach_augment


@dataclass
class TemporalEpisode:
    episode_id: str
    network_id: str
    viewpoint_owner: Optional[str]
    subject: Optional[str]
    events: list[EpisodeEvent] = field(default_factory=list)

    @property
    def active_events(self) -> list[EpisodeEvent]:
        """At most MAX_HOT_EVENTS most recent -- older evidence stays in
        `events` (never deleted), just excluded from the active/embedded
        representation and from what the attach prompt shows as history."""
        return self.events[-MAX_HOT_EVENTS:]

    @property
    def current_event(self) -> Optional[MemoryEvent]:
        return self.events[-1].event if self.events else None

    def active_representation_text(self) -> str:
        return " || ".join(ee.event.event_text for ee in self.active_events)

    def all_source_turn_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for ee in self.events:
            ids.extend(ee.event.source_turn_ids)
        return tuple(dict.fromkeys(ids))


def retrieve_top_candidates(
    new_event: MemoryEvent, episodes_in_network: list[TemporalEpisode], limit: int = TOP_K_CANDIDATES,
) -> list[TemporalEpisode]:
    """Embedding retrieves candidates only -- it never decides a merge. The
    ONLY hard filter is network_id (already applied by the caller passing
    only `episodes_in_network`); no conversation_id filter, no slot-key
    equality. Simple weighted-cosine combination of two soft signals:
    new-event-vs-episode's-active-representation, and new-event-vs-the
    episode's single most recent event (the most temporally relevant one)."""
    if not episodes_in_network:
        return []
    query_vec = np.asarray(embed(new_event.event_text), dtype=np.float64)
    active_texts = [ep.active_representation_text() for ep in episodes_in_network]
    recent_texts = [ep.current_event.event_text if ep.current_event else "" for ep in episodes_in_network]
    active_vecs = np.asarray(embed_batch(active_texts), dtype=np.float64)
    recent_vecs = np.asarray(embed_batch(recent_texts), dtype=np.float64)

    scores = 0.6 * _safe_cosine_batch(query_vec, active_vecs) + 0.4 * _safe_cosine_batch(query_vec, recent_vecs)
    order = np.argsort(-scores)[:limit]
    return [episodes_in_network[i] for i in order]


def _safe_cosine_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Stable cosine scores; invalid vectors can never enter top-k."""
    query = np.asarray(query, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(query_norm) or query_norm <= 0:
        return np.full(len(matrix), -np.inf)
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(row_norms) & (row_norms > 0)
    scores = np.full(len(matrix), -np.inf)
    # Do not use NumPy/Accelerate matmul here.  On macOS it emitted spurious
    # divide/overflow warnings for long-lived repeated small-vector calls even
    # when every float64 input and norm was finite.  Elementwise reduction is
    # the same dot product and avoids that BLAS path entirely.
    dots = np.sum(matrix[valid] * query[None, :], axis=1, dtype=np.float64)
    values = dots / (row_norms[valid] * query_norm)
    scores[valid] = np.where(np.isfinite(values), values, -np.inf)
    return scores


# --------------------------------------------------------------------------
# Attach decision
# --------------------------------------------------------------------------

def _attach_prompt(new_event: MemoryEvent, candidates: list[TemporalEpisode]) -> str:
    lines = []
    for index, episode in enumerate(candidates):
        recent = episode.active_events
        recent_lines = "\n".join(
            f"    [{ee.event.observed_at}] {ee.event.viewpoint_owner or '?'}: {ee.event.event_text}" for ee in recent
        )
        current = episode.current_event
        lines.append(
            f"[{index}] current: \"{current.event_text if current else ''}\" "
            f"(viewpoint_owner={episode.viewpoint_owner!r}, subject={episode.subject!r})\n"
            f"  recent history:\n{recent_lines}"
        )
    block = "\n".join(lines) if lines else "(none)"
    return f"""You do not know what question (if any) this relates to -- decide only
from the content below.

NEW event:
  text: "{new_event.event_text}"
  type: {new_event.event_type}
  viewpoint_owner: {new_event.viewpoint_owner!r}
  subject: {new_event.subject!r}
  observed_at: {new_event.observed_at}

Candidate existing episodes (same network, top {len(candidates)} by similarity):
{block}

Decide whether the NEW event continues one of these episodes, or starts a new one:
- ATTACH_REVISE: same episode/topic, but the value/content is replaced or
  contradicts the episode's current state.
- ATTACH_REAFFIRM: same episode/topic, restates essentially the same value.
- ATTACH_AUGMENT: same episode/topic, adds a compatible detail WITHOUT
  replacing the current value.
- NEW_EPISODE: a different topic/storyline, or no candidate is a confident
  match. If uncertain, choose NEW_EPISODE -- a missed link is recoverable,
  a false merge is not.

IMPORTANT attachment rule: an individual's opinion or action must NOT attach
to another person's individual episode merely because they share a broad
topic. Attaching a new event to a candidate episode held by a DIFFERENT
person (a different viewpoint_owner/subject) is allowed ONLY when the event
genuinely updates the SAME explicit person, a shared decision, a shared
plan, a relationship between them, or a group-level state -- not just
because the words are related.

Return JSON only:
{{"decision":"ATTACH_REVISE|ATTACH_REAFFIRM|ATTACH_AUGMENT|NEW_EPISODE",
"target_episode_index":0,"confidence":0.0,"rationale":"one short sentence"}}"""


def _parse_attach_decision(output: dict, num_candidates: int) -> dict:
    decision = str(output.get("decision") or "NEW_EPISODE").strip().upper()
    try:
        confidence = float(output.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    target_index = output.get("target_episode_index")
    try:
        target_index = int(target_index)
    except (TypeError, ValueError):
        target_index = None
    if target_index is not None and not (0 <= target_index < num_candidates):
        target_index = None

    defaulted = False
    if decision not in VALID_DECISIONS:
        decision, target_index, defaulted = "NEW_EPISODE", None, True
    elif decision != "NEW_EPISODE" and target_index is None:
        decision, defaulted = "NEW_EPISODE", True
    elif confidence < ATTACH_CONFIDENCE_FLOOR:
        decision, target_index, defaulted = "NEW_EPISODE", None, True

    return {
        "decision": decision, "target_episode_index": target_index, "confidence": confidence,
        "rationale": str(output.get("rationale") or ""), "defaulted": defaulted,
    }


def run_attach_decision(new_event: MemoryEvent, candidates: list[TemporalEpisode]) -> dict:
    if not candidates:
        return {"decision": "NEW_EPISODE", "target_episode_index": None, "confidence": 1.0,
                "rationale": "no candidate episodes", "defaulted": False}
    cache = _load_cache_dict(ATTACH_CACHE)
    prompt = _attach_prompt(new_event, candidates)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    key = "attach:" + prompt_hash
    row = cache.get(key)
    new_call = row is None
    if row is None:
        output = _call_llm(prompt)
        row = {"cache_key": key, "output": output}
        _append_cache_row(ATTACH_CACHE, row)
    parsed = _parse_attach_decision(row["output"], len(candidates))
    parsed["_new_llm_call"] = new_call
    return parsed


_DECISION_TO_EVENT_LABEL = {
    "ATTACH_REVISE": "attach_revise", "ATTACH_REAFFIRM": "attach_reaffirm", "ATTACH_AUGMENT": "attach_augment",
}


def _cross_speaker_identity(event: MemoryEvent) -> str:
    return _norm(event.viewpoint_owner or event.subject)


def _is_cross_speaker_attach(event: MemoryEvent, episode: TemporalEpisode) -> bool:
    """True only when BOTH identities are resolved (non-empty) and differ --
    diagnostic only (post-hoc reporting), the actual constraint is enforced
    by the attach prompt's rule text, not by this check."""
    event_identity = _cross_speaker_identity(event)
    episode_identity = _norm(episode.viewpoint_owner or episode.subject)
    return bool(event_identity) and bool(episode_identity) and event_identity != episode_identity


def materialize_episodes(
    events_by_network: dict[str, list[MemoryEvent]],
) -> tuple[dict[str, list[TemporalEpisode]], list[dict], int]:
    episodes_by_network: dict[str, list[TemporalEpisode]] = defaultdict(list)
    decisions_log: list[dict] = []
    new_llm_calls = 0

    for network_id, events in events_by_network.items():
        events_sorted = sorted(events, key=lambda e: e.observed_at)
        for event in events_sorted:
            candidates = retrieve_top_candidates(event, episodes_by_network[network_id])
            decision_info = run_attach_decision(event, candidates)
            new_llm_calls += int(decision_info.pop("_new_llm_call", False))

            decision = decision_info["decision"]
            target_index = decision_info["target_episode_index"]
            cross_speaker = False
            if decision == "NEW_EPISODE" or target_index is None:
                episode = TemporalEpisode(
                    episode_id=f"ep_{network_id}_{len(episodes_by_network[network_id]) + 1}",
                    network_id=network_id, viewpoint_owner=event.viewpoint_owner, subject=event.subject,
                    events=[EpisodeEvent(event=event, decision="create")],
                )
                episodes_by_network[network_id].append(episode)
                target_episode_id = episode.episode_id
            else:
                target_episode = candidates[target_index]
                cross_speaker = _is_cross_speaker_attach(event, target_episode)
                target_episode.events.append(EpisodeEvent(event=event, decision=_DECISION_TO_EVENT_LABEL[decision]))
                target_episode_id = target_episode.episode_id

            decisions_log.append({
                "network_id": network_id, "event_id": event.event_id, "event_text": event.event_text,
                "observed_at": event.observed_at, "source_turn_ids": list(event.source_turn_ids),
                "n_candidates_shown": len(candidates),
                "candidate_episode_ids": [c.episode_id for c in candidates],
                "decision": decision, "target_episode_id": target_episode_id,
                "confidence": decision_info["confidence"], "rationale": decision_info["rationale"],
                "defaulted": decision_info["defaulted"], "cross_speaker_attachment": cross_speaker,
            })

    return dict(episodes_by_network), decisions_log, new_llm_calls


# --------------------------------------------------------------------------
# Smoke test (2-3 cases only) -- post-hoc metrics use QA gold ONLY here,
# never passed to extraction/retrieval/attach above.
# --------------------------------------------------------------------------

def _provenance_is_valid(event: MemoryEvent, conversations: pd.DataFrame) -> bool:
    net_turns = set(conversations[conversations.network_id == event.network_id].turn_id.astype(str))
    return bool(event.source_turn_ids) and all(t in net_turns for t in event.source_turn_ids)


def run_dev_set(
    qa_ids: Optional[tuple[str, ...]], report_path: Path, run_label: str,
) -> None:
    """qa_ids=None runs all 20 frozen Q8 dev cases; otherwise restricts to
    the given subset (e.g. SMOKE_QA_IDS). Same extraction/retrieval/attach
    pipeline either way -- no code path differs by case count."""
    conversations, qa, _personas = load_data(DATA_DIR)
    selected = _select_questions(qa, PER_TYPE, SEED)
    q8_rows = [row for row in selected if row["query_type"] == "Q8"]
    assert len(q8_rows) == 20, f"expected 20 Q8 dev cases, got {len(q8_rows)}"
    run_rows = [row for row in q8_rows if qa_ids is None or row["qa_id"] in qa_ids]
    expected = 20 if qa_ids is None else len(qa_ids)
    assert len(run_rows) == expected, f"expected {expected} cases, got {len(run_rows)}"

    events_by_network: dict[str, list[MemoryEvent]] = defaultdict(list)
    all_rejected: list[dict] = []
    extraction_calls = 0
    case_by_network = {}
    evidence_attempted_total = 0
    evidence_valid_total = 0

    for row in run_rows:
        network_id = row["network_id"]
        case_by_network[network_id] = row["qa_id"]
        anchors = _json(row["evidence_anchors_json"], [])
        clusters = build_session_clusters(anchors)  # scoping ONLY -- question/answer/relevance never read below this line
        events, rejected, calls, quote_stats = extract_events_for_network(network_id, clusters, conversations)
        events_by_network[network_id].extend(events)
        all_rejected.extend(rejected)
        extraction_calls += calls
        evidence_attempted_total += quote_stats["evidence_attempted"]
        evidence_valid_total += quote_stats["evidence_valid"]

    episodes_by_network, decisions_log, attach_calls = materialize_episodes(dict(events_by_network))

    total_events = sum(len(v) for v in events_by_network.values())
    total_episodes = sum(len(v) for v in episodes_by_network.values())
    decision_counts = {}
    for d in decisions_log:
        decision_counts[d["decision"]] = decision_counts.get(d["decision"], 0) + 1
    chain_lengths = [len(ep.events) for eps in episodes_by_network.values() for ep in eps]
    provenance_valid = sum(1 for evs in events_by_network.values() for e in evs if _provenance_is_valid(e, conversations))
    quote_valid_rate = evidence_valid_total / evidence_attempted_total if evidence_attempted_total else float("nan")

    events_per_turn: dict[str, int] = defaultdict(int)
    for evs in events_by_network.values():
        for e in evs:
            for t in e.source_turn_ids:
                events_per_turn[t] += 1

    revise_links = [d for d in decisions_log if d["decision"] == "ATTACH_REVISE"]
    cross_speaker = [d for d in decisions_log if d["cross_speaker_attachment"]]

    lines = [
        f"# Temporal episode prototype -- {run_label}, tightened event contract",
        "",
        "QA question/answer/evidence_anchors 'relevance' text were NEVER passed to extraction, retrieval, or the "
        "attach decision -- gold evidence_anchors turn_ids/session_index were used ONLY to scope which "
        "conversation windows to process (same oracle-evidence-component discipline as every earlier stage). "
        "qa_id/gold labels appear below ONLY as post-hoc annotations. Retrieval, episode linking, confidence "
        "thresholds, and hot-history cap are UNCHANGED from the prior smoke run -- only the extraction contract "
        "(durability filter, evidence/quote provenance, per-turn cap) and the attach prompt's cross-speaker rule "
        "text changed.",
        "",
        f"- events extracted: {total_events}  (rejected: {len(all_rejected)})",
        f"- provenance-valid events: {provenance_valid}/{total_events}",
        f"- quote-valid provenance rate (evidence entries that were real substrings of their cited turn): "
        f"{evidence_valid_total}/{evidence_attempted_total} ({quote_valid_rate:.3f})",
        f"- episodes materialized: {total_episodes}",
        f"- attach/new distribution: {decision_counts}",
        f"- chain length distribution (events per episode): {sorted(chain_lengths, reverse=True)}",
        f"- new LLM calls: extraction={extraction_calls}, attach={attach_calls}, total={extraction_calls + attach_calls}",
        "",
        "## Event count per source turn", "",
    ]
    for turn_id, count in sorted(events_per_turn.items()):
        lines.append(f"- {turn_id}: {count}")
    lines += ["", f"## Not run in this {run_label.lower()}", ""]
    lines += [
        (f"- remaining Q8 dev cases ({len(run_rows)}/20 processed)" if len(run_rows) < 20 else "- all 20 Q8 dev cases processed"),
        "- answer generation / any generator prompt",
        "- Langfuse or any correctness/temporal/attribution judges",
        "- full SocialMemBench benchmark",
        "- held-out QA (never read)",
        "- any production code path (/ask, llm/graphs.py, Telegram, DB)",
        "",
        f"## All ATTACH_REVISE links ({len(revise_links)})", "",
    ]
    if revise_links:
        for d in revise_links:
            lines.append(f"- {d['event_id']} \"{d['event_text']}\" -> {d['target_episode_id']} "
                         f"(confidence={d['confidence']:.2f})  rationale: {d['rationale']}")
    else:
        lines.append("- none")
    lines += ["", f"## Cross-speaker attachments ({len(cross_speaker)})", ""]
    if cross_speaker:
        for d in cross_speaker:
            lines.append(f"- {d['event_id']} \"{d['event_text']}\" -> {d['target_episode_id']} "
                         f"decision={d['decision']} confidence={d['confidence']:.2f}  rationale: {d['rationale']}")
    else:
        lines.append("- none")
    lines += ["", "## Attach review cards (all decisions this run)", ""]
    for i, d in enumerate(decisions_log, 1):
        lines.append(f"### {i}. {d['event_id']} :: {d['network_id']} (qa_id={case_by_network.get(d['network_id'])}, post-hoc only)")
        lines.append(f"- event_text: {d['event_text']}")
        lines.append(f"- observed_at: {d['observed_at']}  source_turn_ids: {d['source_turn_ids']}")
        lines.append(f"- candidates shown: {d['n_candidates_shown']} ({d['candidate_episode_ids']})")
        lines.append(f"- decision: {d['decision']} -> {d['target_episode_id']}  confidence={d['confidence']:.2f} "
                     f"defaulted={d['defaulted']} cross_speaker={d['cross_speaker_attachment']}")
        lines.append(f"- rationale: {d['rationale']}")
        lines.append("")

    if all_rejected:
        lines.append("## Rejected raw items")
        lines.append("")
        for r in all_rejected:
            lines.append(f"- {r['network_id']}: reason={r['reason']} raw={r['raw']}")

    report_path.write_text("\n".join(lines))
    print("\n".join(lines))


def run_smoke_test() -> None:
    run_dev_set(SMOKE_QA_IDS, SMOKE_REPORT_MD, "SMOKE TEST (3 cases: Seb, Uncle Femi, Diane)")


def run_full_dev_set() -> None:
    run_dev_set(None, FULL_DEV_REPORT_MD, "FULL DEV SET (20 Q8 cases)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run all 20 Q8 dev cases instead of the 3-case smoke test.")
    args = parser.parse_args()
    if args.full:
        run_full_dev_set()
    else:
        run_smoke_test()
