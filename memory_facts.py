"""Prospective, provenance-bearing "state" memory -- see
research/PROSPECTIVE_MEMORY_RETRIEVAL.md (Stage 0/1, hardened). Separate
from chat_context/chat_moments on purpose: those two lose their source
message_ids at write time, so a derived fact can never be cited back to raw
evidence. A row here always carries source_message_ids, and an update is an
event (old row closed + superseded_by), not an overwrite -- so a later
contradicting fact doesn't erase what was true when.

Hardening (Stage 1 hardening PR): subject is resolved via
participants.resolve_subject_for_fact -- prefers who actually WROTE the
source messages (or who a reply targets) over guessing from a name string,
falling back to plain name matching only when neither signal applies.
Ambiguous or unknown subjects are refused, never silently written under a
guessed key -- including refusing to prefer a "known"/curated identity over
an uncurated one just because it's in a lookup table, since that would
recreate the exact attribution-collapse failure this system exists to
avoid. observed_at is computed server-side from source messages' real
message_date, never trusted from the model. state_key is restricted to a
fixed enum, so the model can't fragment "current_location"/"location"/
"city" into separate un-superseding buckets.
"""
from crypto_utils import decrypt, encrypt
from database.db import get_conn
from embeddings import embed, to_vector_literal
from participants import resolve_subject_for_fact

ALLOWED_STATE_KEYS = {
    "current_location", "work_study", "availability",
    "relationship", "preference", "plan",
}


def _validate_source_message_ids(cursor, chat_id, source_message_ids):
    ids = list(set(source_message_ids or []))
    if not ids:
        raise ValueError("memory_facts row requires at least one source_message_id")
    cursor.execute(
        "SELECT COUNT(*) FROM messages WHERE user_id = %s AND message_id = ANY(%s)",
        (chat_id, ids),
    )
    found = cursor.fetchone()[0]
    if found != len(ids):
        raise ValueError(
            f"source_message_ids not all found in chat {chat_id}: expected {len(ids)}, found {found}"
        )
    return ids


def _resolve_observed_at(cursor, chat_id, ids):
    """observed_at = max(source messages.message_date), computed here, not
    trusted from the model -- gives verifiable, automatic temporal ordering."""
    cursor.execute(
        "SELECT to_timestamp(MAX(message_date)) FROM messages WHERE user_id = %s AND message_id = ANY(%s)",
        (chat_id, ids),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def upsert_state(
    chat_id, subject_hint, state_key, kind, claim, retrieval_cues,
    source_message_ids, importance=1, expires_at=None,
):
    """Resolves subject_hint to a known participant of this chat and closes
    any prior active fact for (chat_id, participant_key, state_key), then
    inserts a new one. Raises ValueError (caller should report, not crash)
    if the subject can't be unambiguously resolved, if state_key isn't one
    of ALLOWED_STATE_KEYS (silently downgraded to None instead -- a fact
    that doesn't fit the fixed vocabulary still gets recorded, it just
    never supersedes anything), or if source_message_ids don't check out."""
    resolved = resolve_subject_for_fact(chat_id, subject_hint, source_message_ids)
    if resolved is None:
        raise ValueError(
            f"subject {subject_hint!r} doesn't unambiguously match a known participant of chat {chat_id} -- not written"
        )
    subject_key, subject_display = resolved

    if state_key not in ALLOWED_STATE_KEYS:
        state_key = None

    retrieval_text = claim if not retrieval_cues else claim + "\n" + "\n".join(retrieval_cues)
    embedding = embed(retrieval_text)

    with get_conn() as conn:
        cursor = conn.cursor()
        ids = _validate_source_message_ids(cursor, chat_id, source_message_ids)
        observed_at = _resolve_observed_at(cursor, chat_id, ids)

        prior_id = None
        if state_key:
            cursor.execute(
                "SELECT id FROM memory_facts WHERE chat_id = %s AND subject_key = %s "
                "AND state_key = %s AND active = TRUE",
                (chat_id, subject_key, state_key),
            )
            row = cursor.fetchone()
            prior_id = row[0] if row else None

        cursor.execute(
            """
            INSERT INTO memory_facts (
                chat_id, subject_key, subject_display, state_key, kind, claim, retrieval_text,
                embedding, importance, observed_at, expires_at, source_message_ids
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                chat_id, subject_key, subject_display, state_key, kind, encrypt(claim), encrypt(retrieval_text),
                to_vector_literal(embedding), importance, observed_at, expires_at, ids,
            ),
        )
        new_id = cursor.fetchone()[0]

        if prior_id:
            cursor.execute(
                "UPDATE memory_facts SET active = FALSE, superseded_by = %s WHERE id = %s",
                (new_id, prior_id),
            )
        conn.commit()
    return new_id


def _row_to_dict(row):
    (fact_id, subject_key, subject_display, state_key, kind, claim, importance, observed_at, source_message_ids) = row
    return {
        "id": fact_id, "subject_key": subject_key, "subject_display": subject_display,
        "state_key": state_key, "kind": kind, "claim": decrypt(claim), "importance": importance,
        "observed_at": observed_at, "source_message_ids": source_message_ids,
    }


_SELECT_COLUMNS = (
    "id, subject_key, subject_display, state_key, kind, claim, importance, observed_at, source_message_ids"
)


def get_active_facts(chat_id, subject_keys, limit=12):
    """Entity-state retrieval (research doc Sec 5.3) -- no cosine threshold,
    no lexical match required: pulls every currently-active fact for named
    subjects regardless of how differently the current question is worded."""
    if not subject_keys:
        return []
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT {_SELECT_COLUMNS} FROM memory_facts
            WHERE chat_id = %s AND subject_key = ANY(%s) AND active = TRUE
              AND (expires_at IS NULL OR expires_at > now())
            ORDER BY importance DESC, observed_at DESC NULLS LAST
            LIMIT %s
            """,
            (chat_id, list(subject_keys), limit),
        )
        return [_row_to_dict(row) for row in cursor.fetchall()]


def search_facts_by_vector(chat_id, query_embedding, top_k=8):
    """memory_vector branch (Sec 5.2) -- similarity over retrieval_text
    (claim + prospective cues), not just the claim's own wording."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT {_SELECT_COLUMNS} FROM memory_facts
            WHERE chat_id = %s AND active = TRUE
              AND (expires_at IS NULL OR expires_at > now())
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (chat_id, to_vector_literal(query_embedding), top_k),
        )
        return [_row_to_dict(row) for row in cursor.fetchall()]


def list_facts(chat_id, active_only=True):
    with get_conn() as conn:
        cursor = conn.cursor()
        query = f"SELECT {_SELECT_COLUMNS} FROM memory_facts WHERE chat_id = %s"
        if active_only:
            query += " AND active = TRUE"
        query += " ORDER BY id ASC"
        cursor.execute(query, (chat_id,))
        return [_row_to_dict(row) for row in cursor.fetchall()]
