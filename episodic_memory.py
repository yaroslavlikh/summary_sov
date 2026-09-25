"""Event memory with addressable sources, linked into versioned episodes --
the production port of research/temporal_episode_prototype.py (the method of
paper/preprint_memory_ru).

Replaces memory_facts as the write path. Memory is built query-independently
by a background worker (scheduler.py), never from a question:

    new messages, grouped into sessions (conversation_id, else calendar day)
        -> one extraction call per block of up to BLOCK_SIZE anchor messages,
           with earlier messages of the same session as context
        -> deterministic validation: every event needs >= 1 {message_id, quote}
           whose quote is a literal substring of that message; observed_at is
           taken from the real message dates, never from the model
        -> linking: top-3 episodes of the chat by
           0.6 cos(event, hot view) + 0.4 cos(event, last event),
           an LLM resolver picks NEW / REVISE / REAFFIRM / AUGMENT,
           anything invalid or below ATTACH_CONFIDENCE_FLOOR becomes NEW

An episode is append-only: events are never rewritten or deleted, the hot view
(last MAX_HOT_EVENTS events) is what gets embedded and shown, older events keep
their sources. In /ask (llm/graphs.py, behind MEMORY_EPISODES_IN_ASK) episodes
are ranked in the same vector pool as raw messages, and a selected episode
expands back into the source messages of its hot events.

What the paper does and does not license here: on EverMemBench the event index
improved delivery of annotated sources; a mean accuracy gain and an added gain
from linking into episodes were not established, and production uses different
models (Groq gpt-oss-20b, local MiniLM). Whether this helps /ask in this chat is
decided by tests/evals/compare_episodes.py under docs/EPISODES_ROLLOUT_PROTOCOL.md,
not assumed.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from config import get_timezone
from crypto_utils import decrypt, encrypt
from database.db import get_conn
from display_names import resolve_display_name
from embeddings import embed_batch, to_vector_literal
from participants import resolve_participant_key, resolve_subject_for_fact

SCHEMA_VERSION = "episodic_memory_v2"  # v2: self-containment rule for event_text

VALID_EVENT_TYPES = {"state", "opinion", "commitment", "decision", "relationship", "exception", "other"}
VALID_TEMPORAL_MODES = {"past", "current", "future", "habitual", "unknown"}
VALID_DECISIONS = {"ATTACH_REVISE", "ATTACH_REAFFIRM", "ATTACH_AUGMENT", "NEW_EPISODE"}
DECISION_TO_OPERATION = {
    "NEW_EPISODE": "create", "ATTACH_REVISE": "revise",
    "ATTACH_REAFFIRM": "reaffirm", "ATTACH_AUGMENT": "augment",
}
ATTACH_CONFIDENCE_FLOOR = 0.5
MAX_HOT_EVENTS = 5
TOP_K_CANDIDATES = 3
MAX_EVENTS_PER_MESSAGE = 2
BLOCK_SIZE = 40
CONTEXT_MESSAGES = 8
# A message is extracted only once it is this old: conversation_id can still
# grow for _CONVERSATION_GAP_SECONDS (handlers.py), so a fresh message would be
# processed without the replies that resolve it.
SETTLE_SECONDS = 15 * 60


# --------------------------------------------------------------------------
# Pure logic: prompts, validation, parsing, rendering (no DB, no model)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceMessage:
    row_id: int
    message_id: int
    author: str
    text: str
    message_date: Optional[int]
    replied_text: Optional[str] = None


@dataclass(frozen=True)
class MemoryEvent:
    event_text: str
    event_type: str
    viewpoint_owner: Optional[str]
    subject: Optional[str]
    temporal_mode: str
    evidence: tuple[dict, ...]            # validated {"message_id": int, "quote": str}
    source_message_ids: tuple[int, ...]   # derived from validated evidence only
    local_context_message_ids: tuple[int, ...]
    observed_at: Optional[int]            # max message_date of the sources, unix seconds


def _fmt_date(message_date: Optional[int]) -> str:
    if not message_date:
        return "дата неизвестна"
    return datetime.fromtimestamp(message_date, ZoneInfo(get_timezone())).strftime("%Y-%m-%d %H:%M")


def extraction_prompt(anchors: list[SourceMessage], context: list[SourceMessage]) -> str:
    lines = []
    for message, is_anchor in [(m, False) for m in context] + [(m, True) for m in anchors]:
        line = f"[[m{message.message_id}]] {'[ANCHOR] ' if is_anchor else ''}{_fmt_date(message.message_date)} {message.author}: {message.text}"
        if message.replied_text:
            line += f"  (в ответ на: {message.replied_text[:200]})"
        lines.append(line)
    turns = "\n".join(lines)
    return f"""You will see messages from ONE conversation of a Russian-speaking group chat.
Messages marked [ANCHOR] are the ones to process; the others are earlier
context, shown only to resolve pronouns, ellipsis, sarcasm and short replies.
You do not know what anyone will later ask about these messages.

Extract only DURABLE, externally useful memory events from the ANCHOR messages:
a state, a change, a commitment, a decision, an opinion or a relationship fact
that someone could actually need to recall later. Do NOT extract greetings,
reactions, agreement markers or small talk ("ок", "ахах", "го", "согласен") unless
they explicitly introduce or revise concrete content, in which case extract THAT
content. Zero events is a fine and frequent answer. At most {MAX_EVENTS_PER_MESSAGE}
events per anchor message.

Each event:
- event_text: ONE self-contained sentence IN RUSSIAN that will be read months later
  with no chat around it. NEVER start it with "Я", "Мы", "Мне", "У меня", "Но", "А",
  "Это", "Он", "Она", "Они": name the person, resolve every pronoun and ellipsis from
  the context, and keep the subject of the sentence explicit. Rewriting "Я не пропустил
  ни одной пары" (author Саша) as "Саша не пропустил ни одной пары" is required, not
  optional; an event that cannot be made concrete must not be extracted at all.
- event_type: one of state, opinion, commitment, decision, relationship, exception, other.
- viewpoint_owner: who holds or asserts this (the author of an opinion or report), or null.
- subject: who or what the event is about, named explicitly ("Саша", "релиз"), never
  "я"/"мы"/"себя", or null if unclear -- do not guess.
- evidence: one or more {{"message_id": "m123", "quote": "..."}} -- message_id copied
  exactly from an ANCHOR message, quote a SHORT substring copied verbatim from that
  message's text (mechanically verified; an invented or paraphrased quote is dropped,
  and the event is rejected if no valid quote remains).
- local_context_message_ids: other shown messages that helped you interpret it, may be empty.
- temporal_mode: one of past, current, future, habitual, unknown.

Return JSON only:
{{"items":[{{"event_text":"...","event_type":"...","viewpoint_owner":"... or null","subject":"... or null",
"evidence":[{{"message_id":"m123","quote":"exact short substring"}}],
"local_context_message_ids":["m120"],"temporal_mode":"..."}}]}}

Messages:
{turns}"""


_WS_RE = re.compile(r"\s+")
_GENERIC_FRAGMENT_RE = re.compile(
    r"^("
    r"(это|то|оно) (важно|работает|норм|верно)"
    r"|(ок(ей)?|окей|ага|угу|да|нет|норм|понял|поняла|понятно|согласен|согласна|принято|го|ладно|хорошо|давай)"
    r"|(ok(ay)?|sure|yes|no|got it|sounds good)"
    r")[.!]*$",
    re.IGNORECASE,
)


def _normalize_ws(value: str) -> str:
    return _WS_RE.sub(" ", value or "").strip()


def is_generic_fragment(event_text: str) -> bool:
    """Safety net behind the prompt: rejects only when the WHOLE event text is a
    filler or agreement marker, never a longer sentence that contains one."""
    return bool(_GENERIC_FRAGMENT_RE.fullmatch(_normalize_ws(event_text)))


_DANGLING_START_RE = re.compile(
    r"^(я|мы|мне|меня|мной|нам|нас|у меня|у нас|мой|моя|моё|мои|наш|наша|наши"
    r"|но|а|и|это|оно|он|она|они|там|тогда|тоже|ещё|еще)\b", re.IGNORECASE)
_NAMED_ENTITY_RE = re.compile(r"(?<!^)\b[A-ZА-ЯЁ][a-zа-яё]{2,}")


def is_not_self_contained(event_text: str) -> bool:
    """An event has to be readable months later with no chat around it. A text that
    opens with a bare pronoun or a conjunction and names nobody ("Я не пропустил ни
    одной пары", "Но скорее ближе к 8") fails that and is rejected rather than stored
    as a memory nobody can interpret. A sentence that starts the same way but does
    name someone ("Это правда, что Саша уехал") is kept."""
    text = _normalize_ws(event_text)
    return bool(_DANGLING_START_RE.match(text)) and not _NAMED_ENTITY_RE.search(text)


def _parse_message_id(value: Any) -> Optional[int]:
    match = re.fullmatch(r"\[*\s*m?(\d+)\s*\]*", str(value).strip())
    return int(match.group(1)) if match else None


def validate_event(
    raw: dict, anchors_by_id: dict[int, SourceMessage], shown_ids: set[int],
) -> tuple[Optional[MemoryEvent], str]:
    event_text = _normalize_ws(str(raw.get("event_text") or ""))
    if not event_text:
        return None, "empty_event_text"
    event_type = str(raw.get("event_type") or "").strip().lower()
    if event_type not in VALID_EVENT_TYPES:
        return None, "invalid_event_type"

    evidence: list[dict] = []
    for entry in raw.get("evidence") or []:
        if not isinstance(entry, dict):
            continue
        message_id = _parse_message_id(entry.get("message_id"))
        quote = _normalize_ws(str(entry.get("quote") or ""))
        source = anchors_by_id.get(message_id) if message_id is not None else None
        if source is None or not quote or quote not in _normalize_ws(source.text):
            continue
        if not any(e["message_id"] == message_id and e["quote"] == quote for e in evidence):
            evidence.append({"message_id": message_id, "quote": quote})
    if not evidence:
        return None, "no_valid_evidence_quote"
    if is_generic_fragment(event_text):
        return None, "generic_fragment"
    if is_not_self_contained(event_text):
        return None, "not_self_contained"

    source_ids = tuple(dict.fromkeys(e["message_id"] for e in evidence))
    local_ids = tuple(dict.fromkeys(
        mid for value in (raw.get("local_context_message_ids") or [])
        if (mid := _parse_message_id(value)) is not None and mid in shown_ids
    ))
    dates = [anchors_by_id[mid].message_date for mid in source_ids if anchors_by_id[mid].message_date]

    def _opt(value: Any) -> Optional[str]:
        text = _normalize_ws(str(value)) if value is not None else ""
        return text if text and text.lower() not in {"null", "none", "unknown"} else None

    temporal_mode = str(raw.get("temporal_mode") or "").strip().lower()
    return MemoryEvent(
        event_text=event_text, event_type=event_type,
        viewpoint_owner=_opt(raw.get("viewpoint_owner")), subject=_opt(raw.get("subject")),
        temporal_mode=temporal_mode if temporal_mode in VALID_TEMPORAL_MODES else "unknown",
        evidence=tuple(evidence), source_message_ids=source_ids, local_context_message_ids=local_ids,
        observed_at=max(dates) if dates else None,
    ), "ok"


def enforce_max_events_per_message(events: list[MemoryEvent]) -> tuple[list[MemoryEvent], int]:
    """Earlier events win: one is dropped only if it would push any of its source
    messages over MAX_EVENTS_PER_MESSAGE."""
    counts: dict[int, int] = {}
    kept, dropped = [], 0
    for event in events:
        if any(counts.get(mid, 0) >= MAX_EVENTS_PER_MESSAGE for mid in event.source_message_ids):
            dropped += 1
            continue
        for mid in event.source_message_ids:
            counts[mid] = counts.get(mid, 0) + 1
        kept.append(event)
    return kept, dropped


@dataclass
class EpisodeView:
    episode_id: int
    viewpoint_owner: Optional[str]
    subject: Optional[str]
    hot_events: list[dict] = field(default_factory=list)  # {event_text, observed_at, operation, owner, source_message_ids}


def attach_prompt(event: MemoryEvent, candidates: list[EpisodeView]) -> str:
    blocks = []
    for index, episode in enumerate(candidates):
        history = "\n".join(
            f"    [{_fmt_date(e['observed_at'])}] {e.get('viewpoint_owner') or '?'}: {e['event_text']}"
            for e in episode.hot_events
        )
        current = episode.hot_events[-1]["event_text"] if episode.hot_events else ""
        blocks.append(
            f"[{index}] current: \"{current}\" (viewpoint_owner={episode.viewpoint_owner!r}, subject={episode.subject!r})\n"
            f"  recent history:\n{history}"
        )
    return f"""You do not know what question (if any) this relates to -- decide only from the content below.

NEW event:
  text: "{event.event_text}"
  type: {event.event_type}
  viewpoint_owner: {event.viewpoint_owner!r}
  subject: {event.subject!r}
  observed_at: {_fmt_date(event.observed_at)}

Candidate existing episodes (same chat, top {len(candidates)} by similarity):
{chr(10).join(blocks)}

Decide whether the NEW event continues one of these episodes, or starts a new one:
- ATTACH_REVISE: same storyline, but the value is replaced or contradicts the current state.
- ATTACH_REAFFIRM: same storyline, restates essentially the same value.
- ATTACH_AUGMENT: same storyline, adds a compatible detail WITHOUT replacing the current value.
- NEW_EPISODE: a different storyline, or no candidate is a confident match. If uncertain,
  choose NEW_EPISODE -- a missed link is recoverable, a false merge is not.

An individual's opinion or action must NOT attach to another person's episode merely because
they share a broad topic. Attaching to an episode held by a DIFFERENT person is allowed ONLY when
the event genuinely updates the same explicit person, a shared decision, a shared plan, a
relationship between them, or a group-level state.

Return JSON only:
{{"decision":"ATTACH_REVISE|ATTACH_REAFFIRM|ATTACH_AUGMENT|NEW_EPISODE","target_episode_index":0,"confidence":0.0,"rationale":"one short sentence"}}"""


def parse_attach_decision(output: dict, num_candidates: int) -> dict:
    decision = str(output.get("decision") or "NEW_EPISODE").strip().upper()
    try:
        confidence = float(output.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        target = int(output.get("target_episode_index"))
    except (TypeError, ValueError):
        target = None
    if target is not None and not 0 <= target < num_candidates:
        target = None

    defaulted = False
    if decision not in VALID_DECISIONS:
        decision, target, defaulted = "NEW_EPISODE", None, True
    elif decision != "NEW_EPISODE" and target is None:
        decision, defaulted = "NEW_EPISODE", True
    elif decision != "NEW_EPISODE" and confidence < ATTACH_CONFIDENCE_FLOOR:
        decision, target, defaulted = "NEW_EPISODE", None, True
    if decision == "NEW_EPISODE":
        target = None
    return {"decision": decision, "target": target, "confidence": confidence,
            "rationale": str(output.get("rationale") or "")[:500], "defaulted": defaulted}


def link_text(hot_texts: list[str]) -> str:
    return " || ".join(hot_texts[-MAX_HOT_EVENTS:])


def index_text(viewpoint_owner: Optional[str], subject: Optional[str], hot_texts: list[str]) -> str:
    """The retrieval document of an episode (paper, appendix H): owner and subject
    prefix, then the hot view."""
    return f"Viewpoint owner: {viewpoint_owner or '?'}. Subject: {subject or '?'}. {link_text(hot_texts)}"


_OPERATION_LABELS = {"create": "", "revise": " (изменение)", "reaffirm": " (подтверждение)", "augment": " (дополнение)"}


def render_episode(episode: EpisodeView) -> str:
    """How the generator and the reranker see an episode: a navigational hint whose
    claims must be checked against the numbered source messages."""
    lines = [f"[ЭПИЗОД, чья позиция: {episode.viewpoint_owner or '?'}; о ком/чём: {episode.subject or '?'}]"]
    for event in episode.hot_events:
        lines.append(f"  {_fmt_date(event['observed_at'])}{_OPERATION_LABELS.get(event['operation'], '')}: {event['event_text']}")
    return "\n".join(lines)


def render_event(view: "EventView") -> str:
    """What the reranker and the generator see: the event itself, plus the rest of its
    episode as chronology. Claims here are a hint -- the sources below them are proof."""
    header = f"[ПАМЯТЬ, чья позиция: {view.viewpoint_owner or '?'}; о ком/чём: {view.subject or '?'}]"
    lines = [f"{header}\n  {_fmt_date(view.observed_at)}: {view.event_text}"]
    others = [e for e in (view.episode.hot_events if view.episode else []) if e["event_text"] != view.event_text]
    if others:
        lines.append("  эта же линия раньше/позже:")
        lines += [f"    {_fmt_date(e['observed_at'])}{_OPERATION_LABELS.get(e['operation'], '')}: {e['event_text']}"
                  for e in others]
    return "\n".join(lines)


def hot_source_message_ids(episode: EpisodeView) -> list[int]:
    return list(dict.fromkeys(mid for event in episode.hot_events for mid in event["source_message_ids"]))


def session_key(conversation_id: Optional[int], message_date: Optional[int], row_id: int) -> str:
    if conversation_id is not None:
        return f"conv:{conversation_id}"
    if message_date:
        return "day:" + datetime.fromtimestamp(message_date, ZoneInfo(get_timezone())).strftime("%Y-%m-%d")
    return f"rows:{row_id // BLOCK_SIZE}"


def parse_json_object(content: str) -> dict:
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


UNPARSABLE = "_unparsable"


def default_llm(prompt: str) -> dict:
    """Extraction and attach decisions: fast model, temperature 0, JSON mode.
    Groq rejects a JSON-mode generation it cannot validate (400
    json_validate_failed, seen on real blocks); that case is retried once without
    JSON mode and parsed leniently. An output that still is not JSON comes back as
    {UNPARSABLE: True} and is counted, not silently treated as "no events".
    Transport and rate-limit errors raise, so the worker stops and retries the
    block on the next tick instead of recording it."""
    from groq import BadRequestError
    from llm.groq_client import get_chat_model, tracing_config

    try:
        message = get_chat_model("fast", 0).bind(response_format={"type": "json_object"}).invoke(
            prompt, config=tracing_config())
    except BadRequestError as error:
        if "json_validate_failed" not in str(error):
            raise
        message = get_chat_model("fast", 0).invoke(prompt, config=tracing_config())
    parsed = parse_json_object(message.content if isinstance(message.content, str) else "")
    return parsed if parsed else {UNPARSABLE: True}


# --------------------------------------------------------------------------
# Database: sessions, storage, linking
# --------------------------------------------------------------------------

LEASE_SECONDS = 300


def _holder() -> str:
    import os
    import socket

    return f"{socket.gethostname()}:{os.getpid()}"


def _take_lease(chat_id: int, holder: str) -> bool:
    """One worker per chat, as a lease instead of a session advisory lock: a worker
    killed mid-block used to keep an advisory lock until Postgres reaped its
    connection, which blocked every later pass (seen in the backfill). A lease
    simply expires, and the holder renews it after each block."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO memory_worker_lease (chat_id, holder, expires_at)
               VALUES (%s, %s, now() + make_interval(secs => %s))
               ON CONFLICT (chat_id) DO UPDATE
                   SET holder = EXCLUDED.holder, expires_at = EXCLUDED.expires_at
                   WHERE memory_worker_lease.expires_at < now() OR memory_worker_lease.holder = EXCLUDED.holder
               RETURNING 1""",
            (chat_id, holder, LEASE_SECONDS),
        )
        taken = cursor.fetchone() is not None
        conn.commit()
    return taken


def _release_lease(chat_id: int, holder: str) -> None:
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memory_worker_lease WHERE chat_id = %s AND holder = %s", (chat_id, holder))
        conn.commit()


def _source_messages(rows) -> list[SourceMessage]:
    return [
        SourceMessage(row_id=row_id, message_id=message_id, author=resolve_display_name(username, user_name),
                      text=decrypt(text) or "", message_date=message_date,
                      replied_text=(lambda r: r if r and r != "Отмеченного сообщения нет" else None)(decrypt(replied)))
        for row_id, message_id, user_name, username, text, replied, message_date in rows
    ]


_MESSAGE_COLUMNS = "id, message_id, user_name, username, message, replied_message, message_date"


def pending_blocks(cursor, chat_id: int, now: Optional[int] = None, limit_rows: int = 2000) -> tuple[list[list[tuple]], Optional[int]]:
    """Settled, unprocessed messages after the chat's cursor, grouped by session and
    chunked into blocks. Stops at the first unsettled message so the cursor only
    ever moves over a contiguous processed prefix. Returns (blocks, new_cursor)."""
    now = now if now is not None else int(datetime.now(timezone.utc).timestamp())
    cursor.execute("SELECT last_row_id FROM memory_cursor WHERE chat_id = %s", (chat_id,))
    row = cursor.fetchone()
    start = row[0] if row else 0
    cursor.execute(
        f"""SELECT {_MESSAGE_COLUMNS}, conversation_id, is_bot FROM messages
            WHERE user_id = %s AND id > %s ORDER BY id ASC LIMIT %s""",
        (chat_id, start, limit_rows),
    )
    rows = cursor.fetchall()
    # Rows already inside a committed block are never extracted again, even if the
    # session they belong to has grown since and would now be cut differently.
    cursor.execute("SELECT DISTINCT unnest(row_ids) FROM memory_blocks WHERE chat_id = %s AND last_row_id > %s",
                   (chat_id, start))
    covered = {r[0] for r in cursor.fetchall()}
    settled, new_cursor = [], None
    for r in rows:
        message_date = r[6]
        if message_date is not None and now - message_date < SETTLE_SECONDS:
            break
        new_cursor = r[0]
        settled.append(r)
    sessions: dict[str, list[tuple]] = {}
    for r in settled:
        if r[8] or r[1] is None or r[0] in covered:  # bot answers and rows without a Telegram id are not sources
            continue
        sessions.setdefault(session_key(r[7], r[6], r[0]), []).append(r)
    blocks = []
    for key in sorted(sessions, key=lambda k: sessions[k][0][0]):
        members = sessions[key]
        for i in range(0, len(members), BLOCK_SIZE):
            blocks.append((key, members[i:i + BLOCK_SIZE]))
    return blocks, new_cursor


def _context_for_block(cursor, chat_id: int, key: str, first_row_id: int) -> list[tuple]:
    if key.startswith("conv:"):
        cursor.execute(
            f"""SELECT {_MESSAGE_COLUMNS} FROM messages WHERE user_id = %s AND is_bot = FALSE
                AND message_id IS NOT NULL AND conversation_id = %s AND id < %s ORDER BY id DESC LIMIT %s""",
            (chat_id, int(key[5:]), first_row_id, CONTEXT_MESSAGES),
        )
    else:
        cursor.execute(
            f"""SELECT {_MESSAGE_COLUMNS} FROM messages WHERE user_id = %s AND is_bot = FALSE
                AND message_id IS NOT NULL AND id < %s ORDER BY id DESC LIMIT %s""",
            (chat_id, first_row_id, CONTEXT_MESSAGES),
        )
    return cursor.fetchall()[::-1]


def _load_episode_views(cursor, episode_ids: list[int]) -> list[EpisodeView]:
    if not episode_ids:
        return []
    cursor.execute(
        """SELECT ep.id, ep.viewpoint_owner, ep.subject, ev.event_text, ev.viewpoint_owner, ev.observed_at,
                  ee.operation, ev.source_message_ids, ee.position
           FROM memory_episodes ep
           JOIN memory_episode_events ee ON ee.episode_id = ep.id
           JOIN memory_events ev ON ev.id = ee.event_id
           WHERE ep.id = ANY(%s) AND ee.position > ep.event_count - %s
           ORDER BY ep.id, ee.position""",
        (episode_ids, MAX_HOT_EVENTS),
    )
    views: dict[int, EpisodeView] = {}
    for ep_id, ep_owner, ep_subject, text, owner, observed_at, operation, sources, _pos in cursor.fetchall():
        view = views.setdefault(ep_id, EpisodeView(ep_id, decrypt(ep_owner), decrypt(ep_subject)))
        view.hot_events.append({"event_text": decrypt(text), "viewpoint_owner": decrypt(owner),
                                "observed_at": observed_at, "operation": operation,
                                "source_message_ids": list(sources)})
    return [views[i] for i in episode_ids if i in views]


def _candidate_episodes(cursor, chat_id: int, event_vector: str) -> list[int]:
    cursor.execute(
        """SELECT ep.id FROM memory_episodes ep JOIN memory_events last ON last.id = ep.last_event_id
           WHERE ep.chat_id = %s
           ORDER BY 0.6 * (1 - (ep.link_embedding <=> %s::vector)) + 0.4 * (1 - (last.embedding <=> %s::vector)) DESC
           LIMIT %s""",
        (chat_id, event_vector, event_vector, TOP_K_CANDIDATES),
    )
    return [r[0] for r in cursor.fetchall()]


def participant_keys(chat_id: int, event: MemoryEvent) -> tuple[Optional[str], Optional[str]]:
    """Resolve the free-text subject and viewpoint owner to real participants of this
    chat, so "кто такой Гордей" and "кто я" can query memory directly instead of
    hoping cosine similarity finds the right line. The name is resolved the same way
    memory_facts did it: who actually wrote the evidence wins over a name guess, and
    a genuinely ambiguous name stays unresolved rather than being attributed."""
    def resolve(hint):
        if not hint:
            return None
        try:
            resolved = resolve_subject_for_fact(chat_id, hint, list(event.source_message_ids))
        except Exception:
            resolved = None
        return resolved[0] if resolved else None

    return resolve(event.subject), resolve(event.viewpoint_owner)


def _store_and_link(cursor, chat_id: int, block_id: int, event: MemoryEvent, vector: list[float],
                    llm: Callable[[str], dict]) -> str:
    event_vector = to_vector_literal(vector)
    subject_key, owner_key = participant_keys(chat_id, event)
    cursor.execute(
        """INSERT INTO memory_events (chat_id, block_id, event_text, event_type, viewpoint_owner, subject,
               subject_key, owner_key, temporal_mode, observed_at, evidence, source_message_ids,
               local_context_message_ids, embedding, schema_version)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s) RETURNING id""",
        (chat_id, block_id, encrypt(event.event_text), event.event_type, encrypt(event.viewpoint_owner),
         encrypt(event.subject), subject_key, owner_key, event.temporal_mode, event.observed_at,
         encrypt(json.dumps(list(event.evidence), ensure_ascii=False)), list(event.source_message_ids),
         list(event.local_context_message_ids), event_vector, SCHEMA_VERSION),
    )
    event_id = cursor.fetchone()[0]

    candidates = _load_episode_views(cursor, _candidate_episodes(cursor, chat_id, event_vector))
    decision = (parse_attach_decision(llm(attach_prompt(event, candidates)), len(candidates)) if candidates
                else {"decision": "NEW_EPISODE", "target": None, "confidence": 1.0,
                      "rationale": "no candidate episodes", "defaulted": False})
    operation = DECISION_TO_OPERATION[decision["decision"]]

    if decision["target"] is None:
        cursor.execute(
            "INSERT INTO memory_episodes (chat_id, viewpoint_owner, subject) VALUES (%s, %s, %s) RETURNING id",
            (chat_id, encrypt(event.viewpoint_owner), encrypt(event.subject)),
        )
        episode = EpisodeView(cursor.fetchone()[0], event.viewpoint_owner, event.subject)
    else:
        episode = candidates[decision["target"]]
    cursor.execute("SELECT event_count FROM memory_episodes WHERE id = %s FOR UPDATE", (episode.episode_id,))
    position = cursor.fetchone()[0] + 1
    cursor.execute(
        """INSERT INTO memory_episode_events (episode_id, event_id, position, operation, confidence, rationale,
               defaulted, candidate_episode_ids)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (episode.episode_id, event_id, position, operation, decision["confidence"], encrypt(decision["rationale"]),
         decision["defaulted"], [c.episode_id for c in candidates]),
    )
    hot_texts = [e["event_text"] for e in episode.hot_events] + [event.event_text]
    link_vec, index_vec = embed_batch([link_text(hot_texts), index_text(episode.viewpoint_owner, episode.subject, hot_texts)])
    cursor.execute(
        """UPDATE memory_episodes SET last_event_id = %s, event_count = %s, link_embedding = %s::vector,
               index_embedding = %s::vector, updated_at = now() WHERE id = %s""",
        (event_id, position, to_vector_literal(link_vec), to_vector_literal(index_vec), episode.episode_id),
    )
    return operation


def process_block(chat_id: int, key: str, rows: list[tuple], llm: Callable[[str], dict] = default_llm) -> Optional[dict]:
    """One extraction call and its linking, committed atomically together with the
    block marker, so a crash or a rerun never duplicates events."""
    anchor_row_ids = [r[0] for r in rows]
    block_key = hashlib.sha256(json.dumps([SCHEMA_VERSION, chat_id, anchor_row_ids]).encode()).hexdigest()
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM memory_blocks WHERE chat_id = %s AND block_key = %s", (chat_id, block_key))
        if cursor.fetchone():
            return None
        anchors = _source_messages([r[:7] for r in rows])
        context = _source_messages(_context_for_block(cursor, chat_id, key, rows[0][0]))
        anchors_by_id = {m.message_id: m for m in anchors}
        shown_ids = set(anchors_by_id) | {m.message_id for m in context}

    output = llm(extraction_prompt(anchors, context))
    raw_items = [item for item in (output.get("items") or []) if isinstance(item, dict)]
    rejected: dict[str, int] = {"llm_output_unparsable": 1} if output.get(UNPARSABLE) else {}
    events = []
    for item in raw_items:
        event, reason = validate_event(item, anchors_by_id, shown_ids)
        if event is None:
            rejected[reason] = rejected.get(reason, 0) + 1
        else:
            events.append(event)
    events, capped = enforce_max_events_per_message(events)
    if capped:
        rejected["exceeds_max_events_per_message"] = capped
    events.sort(key=lambda e: (e.observed_at or 0))
    vectors = embed_batch([e.event_text for e in events]) if events else []

    operations: dict[str, int] = {}
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO memory_blocks (chat_id, block_key, session_key, first_row_id, last_row_id, row_ids,
                   message_count, raw_items, events, rejected, schema_version)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (chat_id, block_key) DO NOTHING RETURNING id""",
            (chat_id, block_key, key, anchor_row_ids[0], anchor_row_ids[-1], anchor_row_ids, len(rows), len(raw_items),
             len(events), json.dumps(rejected), SCHEMA_VERSION),
        )
        inserted = cursor.fetchone()
        if not inserted:
            conn.rollback()
            return None
        for event, vector in zip(events, vectors):
            operation = _store_and_link(cursor, chat_id, inserted[0], event, vector, llm)
            operations[operation] = operations.get(operation, 0) + 1
        conn.commit()
    return {"session": key, "messages": len(rows), "raw_items": len(raw_items), "events": len(events),
            "rejected": rejected, "operations": operations}


def run_for_chat(chat_id: int, llm: Callable[[str], dict] = default_llm, max_blocks: Optional[int] = None,
                 now: Optional[int] = None, log: Callable[[str], None] = print) -> dict:
    """Processes every settled block after the cursor, oldest first. Only one worker
    per chat at a time (advisory lock); on any error the cursor stays put and the
    already committed blocks are skipped on the next run."""
    totals = {"blocks": 0, "events": 0, "skipped_locked": False}
    holder = _holder()
    if not _take_lease(chat_id, holder):
        totals["skipped_locked"] = True
        return totals
    try:
        with get_conn() as conn:
            blocks, new_cursor = pending_blocks(conn.cursor(), chat_id, now=now)
        complete = max_blocks is None or len(blocks) <= max_blocks
        for key, rows in blocks[:max_blocks]:
            result = process_block(chat_id, key, rows, llm)
            _take_lease(chat_id, holder)  # renew: a long pass must not lose its own lease
            if result:
                totals["blocks"] += 1
                totals["events"] += result["events"]
                log(f"[memory] chat {chat_id} {key}: {result['messages']} msgs -> {result['events']} events "
                    f"{result['operations']} rejected={result['rejected']}")
        if complete and new_cursor is not None:
            with get_conn() as conn:
                conn.cursor().execute(
                    """INSERT INTO memory_cursor (chat_id, last_row_id) VALUES (%s, %s)
                       ON CONFLICT (chat_id) DO UPDATE SET last_row_id = GREATEST(memory_cursor.last_row_id, EXCLUDED.last_row_id),
                           updated_at = now()""",
                    (chat_id, new_cursor),
                )
                conn.commit()
    finally:
        _release_lease(chat_id, holder)
    return totals


# --------------------------------------------------------------------------
# Retrieval for /ask
# --------------------------------------------------------------------------

@dataclass
class EventView:
    """One retrieved event plus the episode it belongs to. The event is the unit of
    retrieval -- a short Russian sentence, comparable to a raw message in the same
    embedding space -- while the episode is what makes it readable: the chronology of
    how that line changed. Sources are the event's own, so expansion stays precise."""
    event_id: int
    event_text: str
    viewpoint_owner: Optional[str]
    subject: Optional[str]
    observed_at: Optional[int]
    operation: str
    source_message_ids: list[int]
    episode: Optional[EpisodeView] = None


def search_events(chat_id: int, query_vector: str, top_k: int = 8) -> list[tuple[int, float]]:
    """(event_id, cosine distance) in the same MiniLM space as messages.embedding."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, embedding <=> %s::vector AS distance FROM memory_events
               WHERE chat_id = %s ORDER BY distance LIMIT %s""",
            (query_vector, chat_id, top_k),
        )
        return [(r[0], float(r[1])) for r in cursor.fetchall()]


def search_events_by_participant(chat_id: int, participant_key: str, limit: int = 5) -> list[int]:
    """What memory holds about one person -- the direct answer to "кто такой X" / "кто
    я", which cosine similarity handles badly. Events where the person is the SUBJECT
    (what is known about them) come before ones where they merely spoke, because the
    latest thing someone said about a third party is rarely an answer to "who is X"."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id FROM memory_events WHERE chat_id = %s AND (subject_key = %s OR owner_key = %s)
               ORDER BY (subject_key = %s) DESC, observed_at DESC NULLS LAST LIMIT %s""",
            (chat_id, participant_key, participant_key, participant_key, limit),
        )
        return [r[0] for r in cursor.fetchall()]


def load_events(event_ids: list[int]) -> list[EventView]:
    if not event_ids:
        return []
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT ev.id, ev.event_text, ev.viewpoint_owner, ev.subject, ev.observed_at,
                      ev.source_message_ids, ee.operation, ee.episode_id
               FROM memory_events ev LEFT JOIN memory_episode_events ee ON ee.event_id = ev.id
               WHERE ev.id = ANY(%s)""",
            (list(event_ids),),
        )
        rows = {r[0]: r for r in cursor.fetchall()}
        episodes = {e.episode_id: e for e in
                    _load_episode_views(cursor, sorted({r[7] for r in rows.values() if r[7]}))}
    views = []
    for event_id in event_ids:
        row = rows.get(event_id)
        if not row:
            continue
        views.append(EventView(
            event_id=row[0], event_text=decrypt(row[1]), viewpoint_owner=decrypt(row[2]), subject=decrypt(row[3]),
            observed_at=row[4], operation=row[6] or "create", source_message_ids=list(row[5]),
            episode=episodes.get(row[7])))
    return views


def load_episodes(episode_ids: list[int]) -> list[EpisodeView]:
    with get_conn() as conn:
        return _load_episode_views(conn.cursor(), list(episode_ids))


def resolve_question_participant(chat_id: int, question: str, asker_name: Optional[str] = None) -> Optional[str]:
    """The participant a question is about: the asker for "кто я", otherwise the one
    name in the question that matches exactly one participant of this chat."""
    if re.search(r"\b(я|меня|мне|мой|моя|моё|мои)\b", question, re.IGNORECASE) and asker_name:
        resolved = resolve_participant_key(chat_id, asker_name)
        if resolved:
            return resolved[0]
    keys = {resolved[0] for word in re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", question)
            if (resolved := resolve_participant_key(chat_id, word))}
    return next(iter(keys)) if len(keys) == 1 else None


def memory_stats(chat_id: int) -> dict:
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""SELECT count(*), coalesce(sum(message_count), 0), coalesce(sum(raw_items), 0),
                                 coalesce(sum(events), 0) FROM memory_blocks WHERE chat_id = %s""", (chat_id,))
        blocks, messages, raw_items, events = cursor.fetchone()
        cursor.execute("SELECT rejected FROM memory_blocks WHERE chat_id = %s", (chat_id,))
        rejected: dict[str, int] = {}
        for (value,) in cursor.fetchall():
            for reason, n in (value if isinstance(value, dict) else json.loads(value or "{}")).items():
                rejected[reason] = rejected.get(reason, 0) + n
        cursor.execute("""SELECT ee.operation, count(*) FROM memory_episode_events ee
                          JOIN memory_episodes ep ON ep.id = ee.episode_id WHERE ep.chat_id = %s GROUP BY 1""", (chat_id,))
        operations = dict(cursor.fetchall())
        cursor.execute("SELECT count(*), coalesce(max(event_count), 0) FROM memory_episodes WHERE chat_id = %s", (chat_id,))
        episodes, longest = cursor.fetchone()
    return {"blocks": blocks, "messages": messages, "raw_items": raw_items, "events": events,
            "rejected": rejected, "operations": operations, "episodes": episodes, "longest_episode": longest}


if __name__ == "__main__":
    import sys

    target = int(sys.argv[1])
    run_for_chat(target, max_blocks=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    print(json.dumps(memory_stats(target), ensure_ascii=False, indent=2))
