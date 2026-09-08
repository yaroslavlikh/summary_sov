"""Prospective, provenance-bearing "state" memory -- see
research/PROSPECTIVE_MEMORY_RETRIEVAL.md (Stage 0/1). Separate from
chat_context/chat_moments on purpose: those two lose their source message_ids
at write time, so a derived fact can never be cited back to raw evidence. A
row here always carries source_message_ids, and an update is an event (old
row closed + superseded_by), not an overwrite -- so a later contradicting
fact doesn't erase what was true when.
"""
from crypto_utils import decrypt, encrypt
from database.db import get_conn
from embeddings import embed, to_vector_literal


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


def upsert_state(
    chat_id, subject_key, state_key, kind, claim, retrieval_cues,
    source_message_ids, importance=1, expires_at=None, observed_at=None,
):
    """Closes any prior active fact for (chat_id, subject_key, state_key) and
    inserts a new one. state_key may be None for one-off facts that aren't a
    single changeable state (a moment/group_lore-shaped fact) -- those never
    supersede anything, they just accumulate with provenance."""
    retrieval_text = claim if not retrieval_cues else claim + "\n" + "\n".join(retrieval_cues)
    embedding = embed(retrieval_text)

    with get_conn() as conn:
        cursor = conn.cursor()
        _validate_source_message_ids(cursor, chat_id, source_message_ids)

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
                chat_id, subject_key, state_key, kind, claim, retrieval_text,
                embedding, importance, observed_at, expires_at, source_message_ids
            ) VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                chat_id, subject_key, state_key, kind, encrypt(claim), encrypt(retrieval_text),
                to_vector_literal(embedding), importance, observed_at, expires_at,
                list(set(source_message_ids)),
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
    (fact_id, subject_key, state_key, kind, claim, importance, observed_at, source_message_ids) = row
    return {
        "id": fact_id, "subject_key": subject_key, "state_key": state_key, "kind": kind,
        "claim": decrypt(claim), "importance": importance, "observed_at": observed_at,
        "source_message_ids": source_message_ids,
    }


_SELECT_COLUMNS = "id, subject_key, state_key, kind, claim, importance, observed_at, source_message_ids"


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
