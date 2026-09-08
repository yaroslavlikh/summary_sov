"""Offline pilot: semantic conversation disentanglement.

Scope (per the brief this was written against): a minimal, read-only,
sequential alternative to handlers._compute_conversation_id's fixed
time-gap heuristic, using the SAME 384-dim embeddings already stored on
`messages.embedding` -- no new model, no LLM call per assignment, no
production writes.

Two things this module deliberately keeps separate, because conflating
them was the exact mistake the current production heuristic makes:

- `conversation_id` -- a LOCAL episode: which messages belong to the same
  concrete exchange, right now, in this scope. Never merges with a
  semantically similar episode from a different time.
- semantic retrieval -- finding topically similar PAST episodes. Read-only,
  bounded, never rewrites conversation_id or merges episodes.

Everything here is a pure function of an in-memory message list plus
existing embeddings -- no Postgres, no network call, so it's unit-testable
without a DB and without loading the real embedding model.
"""
from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ---- Tunables (kept as plain constants, no config system, per the brief) ----

K_SUBSTANTIVE = 5                 # messages feeding a conversation's mean-embedding representation
SEMANTIC_JOIN_THRESHOLD = 0.55    # cosine similarity required to join an active conversation
ACTIVE_WINDOW_SECONDS = 30 * 60   # how long a conversation stays "active" for new assignments
ACTIVE_CONVERSATIONS_CONSIDERED = 5   # cap on recently-active conversations compared per message
MIN_SUBSTANTIVE_CHARS = 8         # floor (non-punctuation/emoji chars) for a message to carry semantic weight

_STRIP_RE = re.compile(r"[^\w]+", re.UNICODE)


def is_substantive(text: str) -> bool:
    """Crude, explainable heuristic: strips whitespace/punctuation/emoji and
    checks what's left has enough content to mean anything on its own.
    "да", "нет", "ага", a bare emoji, "." all fail this -- by design they
    must never single-handedly create a confident semantic association
    (see ConversationTracker.assign)."""
    stripped = _STRIP_RE.sub("", text)
    return len(stripped) >= MIN_SUBSTANTIVE_CHARS


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def normalized_mean(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    dim = len(vectors[0])
    mean = [sum(v[i] for v in vectors) / n for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in mean))
    if norm == 0.0:
        return mean
    return [x / norm for x in mean]


@dataclass
class Message:
    message_id: int
    chat_id: int
    thread_id: Optional[int]
    reply_to_message_id: Optional[int]
    author: str
    timestamp: float
    text: str
    embedding: list[float]


@dataclass
class ConversationEpisode:
    conversation_id: int
    chat_id: int
    thread_id: Optional[int]
    message_ids: list[int] = field(default_factory=list)
    last_timestamp: float = 0.0
    _substantive_embeddings: deque = field(default_factory=lambda: deque(maxlen=K_SUBSTANTIVE))

    def add(self, message: Message, substantive: bool) -> None:
        self.message_ids.append(message.message_id)
        self.last_timestamp = message.timestamp
        if substantive:
            self._substantive_embeddings.append(message.embedding)

    def representation(self) -> Optional[list[float]]:
        if not self._substantive_embeddings:
            return None
        return normalized_mean(list(self._substantive_embeddings))


@dataclass
class AssignmentResult:
    message_id: int
    conversation_id: int
    method: str  # "reply" | "semantic" | "continuity" | "fallback" | "new"
    score: Optional[float]
    low_confidence: bool
    candidates: list[tuple[int, Optional[float]]]


class ConversationTracker:
    """Sequential, stateful assignment -- feed messages in chronological
    order via `assign`. Pure in-memory state, no I/O."""

    def __init__(
        self,
        join_threshold: float = SEMANTIC_JOIN_THRESHOLD,
        active_window_seconds: float = ACTIVE_WINDOW_SECONDS,
        active_considered: int = ACTIVE_CONVERSATIONS_CONSIDERED,
    ):
        self.join_threshold = join_threshold
        self.active_window_seconds = active_window_seconds
        self.active_considered = active_considered
        self._episodes: dict[int, ConversationEpisode] = {}
        self._by_message_id: dict[int, int] = {}
        # scope -> [(conversation_id, last_timestamp), ...] most-recent-first
        self._active_by_scope: dict[tuple, list[int]] = {}
        # (scope, author) -> conversation_id of that author's last message
        self._author_last_conv: dict[tuple, int] = {}

    def all_episodes(self) -> list[ConversationEpisode]:
        return list(self._episodes.values())

    def assign(self, message: Message) -> AssignmentResult:
        scope = (message.chat_id, message.thread_id)
        substantive = is_substantive(message.text)

        # 1) Explicit reply always wins over any semantic signal -- but never
        # across a thread boundary (a cross-topic reply would be a Telegram
        # data anomaly, not a real signal to trust).
        reply_conv = self._by_message_id.get(message.reply_to_message_id) if message.reply_to_message_id else None
        if reply_conv is not None and self._episodes[reply_conv].thread_id == message.thread_id:
            self._record(scope, reply_conv, message, substantive)
            return AssignmentResult(message.message_id, reply_conv, "reply", None, False, [])

        # 2) Only recently-active conversations in the SAME (chat, thread)
        # scope are ever candidates -- this is what keeps different Telegram
        # topics from ever merging, and keeps an old similar conversation
        # from being silently resurrected instead of retrieved.
        candidate_ids = self._recent_active(scope, message.timestamp)
        candidates_scored: list[tuple[int, Optional[float]]] = []
        if substantive:
            for cid in candidate_ids:
                rep = self._episodes[cid].representation()
                if rep is not None:
                    candidates_scored.append((cid, cosine(message.embedding, rep)))

            scored_only = [(cid, s) for cid, s in candidates_scored if s is not None]
            if scored_only:
                best_id, best_score = max(scored_only, key=lambda t: t[1])
                if best_score >= self.join_threshold:
                    self._record(scope, best_id, message, substantive)
                    return AssignmentResult(message.message_id, best_id, "semantic", best_score, False, candidates_scored)
            new_id = self._new_episode(scope, message, substantive)
            return AssignmentResult(message.message_id, new_id, "new", None, False, candidates_scored)

        # 3) Short/semantically-empty message: never trust semantics alone.
        # Prefer participant continuity, else the single most-recently-active
        # conversation, else start a new (low-confidence either way) episode.
        continuity_id = self._author_last_conv.get((scope, message.author))
        if continuity_id is not None and continuity_id in candidate_ids:
            self._record(scope, continuity_id, message, substantive)
            return AssignmentResult(message.message_id, continuity_id, "continuity", None, True, [])

        if candidate_ids:
            fallback_id = candidate_ids[0]
            self._record(scope, fallback_id, message, substantive)
            return AssignmentResult(message.message_id, fallback_id, "fallback", None, True, [])

        new_id = self._new_episode(scope, message, substantive)
        return AssignmentResult(message.message_id, new_id, "new", None, True, [])

    def _recent_active(self, scope: tuple, now: float) -> list[int]:
        ids = self._active_by_scope.get(scope, [])
        fresh = [cid for cid in ids if now - self._episodes[cid].last_timestamp <= self.active_window_seconds]
        self._active_by_scope[scope] = fresh
        return fresh[: self.active_considered]

    def _touch_active(self, scope: tuple, conversation_id: int) -> None:
        ids = self._active_by_scope.setdefault(scope, [])
        if conversation_id in ids:
            ids.remove(conversation_id)
        ids.insert(0, conversation_id)

    def _record(self, scope: tuple, conversation_id: int, message: Message, substantive: bool) -> None:
        self._episodes[conversation_id].add(message, substantive)
        self._by_message_id[message.message_id] = conversation_id
        self._author_last_conv[(scope, message.author)] = conversation_id
        self._touch_active(scope, conversation_id)

    def _new_episode(self, scope: tuple, message: Message, substantive: bool) -> int:
        conversation_id = message.message_id
        episode = ConversationEpisode(conversation_id, message.chat_id, message.thread_id)
        self._episodes[conversation_id] = episode
        self._record(scope, conversation_id, message, substantive)
        return conversation_id


def search_similar_episodes(
    query_embedding: list[float],
    episodes: list[ConversationEpisode],
    top_k: int = 3,
    messages_per_episode: int = 5,
    max_total_messages: int = 15,
) -> list[dict]:
    """Read-only similarity search over PAST episode representations.
    Never mutates conversation_id or merges episodes -- a hit just returns a
    bounded slice of that episode's own message_ids for the caller to load
    and show as background context.

    For a deictic/reply-style query ("это правда?"), pass the embedding of
    the ANCHOR content (or the resolved conversation's own representation),
    not the bare follow-up text -- that's a caller-side choice, this
    function only ever compares whatever embedding it's given."""
    scored = []
    for ep in episodes:
        rep = ep.representation()
        if rep is not None:
            scored.append((ep, cosine(query_embedding, rep)))
    scored.sort(key=lambda t: -t[1])

    results = []
    budget = max_total_messages
    for ep, score in scored[:top_k]:
        if budget <= 0:
            break
        bounded = ep.message_ids[-messages_per_episode:][:budget]
        budget -= len(bounded)
        results.append({"conversation_id": ep.conversation_id, "score": score, "message_ids": bounded})
    return results


def time_gap_baseline(messages: list[Message], gap_seconds: float = 300) -> dict[int, int]:
    """Pure reimplementation of handlers._compute_conversation_id's actual
    production logic (reply inherits, else time-gap continuation, else new),
    for an apples-to-apples offline comparison against the same fixture.
    Faithfully reproduces its real behavior including NOT scoping "last
    message" by thread_id -- the shipped heuristic doesn't either."""
    conv_by_message_id: dict[int, int] = {}
    last_conv, last_message_id, last_ts = None, None, None
    result: dict[int, int] = {}
    for m in messages:
        conv_id = None
        if m.reply_to_message_id is not None and m.reply_to_message_id in conv_by_message_id:
            conv_id = conv_by_message_id[m.reply_to_message_id]
        if conv_id is None:
            if last_ts is not None and 0 <= m.timestamp - last_ts < gap_seconds:
                conv_id = last_conv if last_conv is not None else last_message_id
            else:
                conv_id = m.message_id
        conv_by_message_id[m.message_id] = conv_id
        last_conv, last_message_id, last_ts = conv_id, m.message_id, m.timestamp
        result[m.message_id] = conv_id
    return result


def semantic_pilot(messages: list[Message], **tracker_kwargs) -> tuple[dict[int, int], list[AssignmentResult], ConversationTracker]:
    tracker = ConversationTracker(**tracker_kwargs)
    results = [tracker.assign(m) for m in messages]
    return {r.message_id: r.conversation_id for r in results}, results, tracker
