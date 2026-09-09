"""Resolves a free-text name (as an LLM tool call produces it -- "Миша",
"Михаил", "Мишаня") to a stable participant_key for a given chat --
research/PROSPECTIVE_MEMORY_RETRIEVAL.md Stage 1 hardening item #1.

participant_key is the Telegram username when known (stable and unique,
unlike a first_name that several people can share or one person can be
called several ways), else a normalized version of the raw first_name for
username-less senders.

Resolution only succeeds on an EXACT (case-insensitive, whitespace-
normalized) match against a known display name, username, or raw first_name
actually seen in this chat -- no fuzzy/diminutive matching in this version.
Zero or ambiguous (>1) matches return None; callers must treat that as
"don't write this automatically" (see memory_facts.upsert_state).
"""
from database.db import get_conn
from display_names import DISPLAY_NAMES, DISPLAY_NAMES_BY_FIRST_NAME, resolve_display_name


def _normalize(s):
    return (s or "").strip().lower()


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
    matches exactly one real participant of this chat, else None."""
    target = _normalize(name_hint)
    if not target:
        return None

    matches = {}  # participant_key -> display_name
    for username, user_name in _known_participants(chat_id):
        clean_username = (username or "").lstrip("@")
        key = clean_username or _normalize(user_name)
        if not key:
            continue

        candidates = {_normalize(user_name)}
        if clean_username:
            candidates.add(_normalize(clean_username))
        display = resolve_display_name(username, user_name)
        candidates.add(_normalize(display))
        # A composite display name like "tigmen / Саша Тигмен" should also
        # match on just "Тигмен" or "Саша Тигмен" -- split on the same " / "
        # separator the display names already use, not fuzzy matching.
        for part in display.split(" / "):
            candidates.add(_normalize(part))
        if user_name and _normalize(user_name) in DISPLAY_NAMES_BY_FIRST_NAME:
            candidates.add(_normalize(DISPLAY_NAMES_BY_FIRST_NAME[_normalize(user_name)]))

        if target in candidates:
            matches[key] = display

    if len(matches) == 1:
        return next(iter(matches.items()))
    return None
