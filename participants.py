"""Resolves who a memory fact is actually about among a chat's real
participants -- research/PROSPECTIVE_MEMORY_RETRIEVAL.md Stage 1 hardening
item #1.

Two entry points:

- `resolve_participant_key(chat_id, name_hint)` -- pure name matching, no
  special-casing. If a bare name genuinely matches more than one real
  participant (curated in DISPLAY_NAMES or not), that IS real ambiguity --
  this deliberately does NOT prefer a curated identity over an uncurated
  one just because it happens to be in a lookup table. Silently preferring
  "the Игорь we know about" over "some other real Игорь" would hide exactly
  the attribution-collapse failure mode this whole memory system exists to
  avoid.

- `resolve_subject_for_fact(chat_id, name_hint, source_message_ids)` -- the
  one memory_facts.upsert_state actually calls. Prefers deterministic
  signals grounded in who ACTUALLY wrote the source messages over name
  matching: if name_hint matches the real author of a source message
  (self-referential: "я"/"меня"/their own name), resolve to that author
  directly -- no ambiguity check needed, we know who wrote it. Else, if a
  source message is a reply and name_hint matches the reply target's real
  author, resolve to them. Only when neither signal applies does it fall
  back to plain name matching, which can still legitimately go ambiguous.
"""
import re

from database.db import get_conn
from display_names import DISPLAY_NAMES, DISPLAY_NAMES_BY_FIRST_NAME, resolve_display_name


def _normalize(s):
    # LLM-produced name hints sometimes use non-standard whitespace (e.g.
    # U+202F narrow no-break space) where a real chat name has a plain
    # space -- collapse any whitespace run to one regular space so those
    # still match instead of silently going "ambiguous"/unresolved.
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _participant_key(username, user_name):
    clean_username = (username or "").lstrip("@")
    return clean_username or _normalize(user_name)


def _name_candidates(username, user_name):
    """Every string that should count as "this is the same person" for
    exact-match purposes: raw first_name, username, resolved display name,
    and (only for curated identities, since an uncurated composite name
    doesn't exist) its individual words -- "Тигмен" alone should match
    "tigmen / Саша Тигмен" just as well as the full string."""
    clean_username = (username or "").lstrip("@")
    display = resolve_display_name(username, user_name)
    candidates = {_normalize(user_name), _normalize(display)}
    if clean_username:
        candidates.add(_normalize(clean_username))
    if clean_username in DISPLAY_NAMES:
        for part in display.split(" / "):
            candidates.add(_normalize(part))
            for word in part.split():
                if len(word) > 2:
                    candidates.add(_normalize(word))
    if user_name and _normalize(user_name) in DISPLAY_NAMES_BY_FIRST_NAME:
        candidates.add(_normalize(DISPLAY_NAMES_BY_FIRST_NAME[_normalize(user_name)]))
    return candidates


def _known_participants(chat_id):
    """Distinct (username, raw first_name) pairs actually seen posting in
    this chat -- ground truth of who's really here, not just whoever
    DISPLAY_NAMES happens to know about."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT username, user_name FROM messages WHERE user_id = %s AND is_bot = FALSE",
            (chat_id,),
        )
        return cursor.fetchall()


def resolve_participant_key(chat_id, name_hint):
    """Returns (participant_key, display_name) if name_hint unambiguously
    matches exactly one real participant of this chat, else None. Curated
    and uncurated identities are matched on equal footing -- see module
    docstring for why."""
    target = _normalize(name_hint)
    if not target:
        return None

    matches = {}
    for username, user_name in _known_participants(chat_id):
        key = _participant_key(username, user_name)
        if not key:
            continue
        if target in _name_candidates(username, user_name):
            matches[key] = resolve_display_name(username, user_name)

    if len(matches) == 1:
        return next(iter(matches.items()))
    return None


def resolve_subject_for_fact(chat_id, name_hint, source_message_ids):
    """Fact-specific resolution: prefers who actually wrote the evidence
    over guessing from a name string. Falls back to resolve_participant_key
    only when neither the author nor a reply-target signal applies."""
    target = _normalize(name_hint)
    if not target or not source_message_ids:
        return resolve_participant_key(chat_id, name_hint)

    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT message_id, username, user_name, reply_to_message_id FROM messages "
            "WHERE user_id = %s AND message_id = ANY(%s)",
            (chat_id, list(set(source_message_ids))),
        )
        source_rows = cursor.fetchall()

        reply_ids = [r[3] for r in source_rows if r[3] is not None]
        reply_authors = {}
        if reply_ids:
            cursor.execute(
                "SELECT message_id, username, user_name FROM messages WHERE user_id = %s AND message_id = ANY(%s)",
                (chat_id, list(set(reply_ids))),
            )
            reply_authors = {mid: (username, user_name) for mid, username, user_name in cursor.fetchall()}

    authors_matching = {}
    reply_targets_matching = {}
    for _, username, user_name, reply_to in source_rows:
        key = _participant_key(username, user_name)
        if key and target in _name_candidates(username, user_name):
            authors_matching[key] = resolve_display_name(username, user_name)

        if reply_to in reply_authors:
            r_username, r_user_name = reply_authors[reply_to]
            r_key = _participant_key(r_username, r_user_name)
            if r_key and target in _name_candidates(r_username, r_user_name):
                reply_targets_matching[r_key] = resolve_display_name(r_username, r_user_name)

    # Self-referential ("я"/their own name in a message they wrote
    # themselves) is the strongest possible signal -- we KNOW who wrote it.
    if len(authors_matching) == 1:
        return next(iter(authors_matching.items()))
    # A reply whose target's real identity matches the name -- still
    # grounded in actual message metadata, not a name guess.
    if len(reply_targets_matching) == 1:
        return next(iter(reply_targets_matching.items()))

    return resolve_participant_key(chat_id, name_hint)
