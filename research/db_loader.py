"""Read-only loader: real messages + their ALREADY-STORED embeddings from
production Postgres, converted into research.conversation_disentanglement's
plain Message objects. Never writes anything -- SELECT only, and reuses the
`embedding` column instead of recomputing (per the brief: no new model
calls, no live ingestion touched).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

from crypto_utils import decrypt
from database.db import get_conn
from display_names import resolve_display_name

from research.conversation_disentanglement import Message


def load_messages(chat_id: int, min_message_id: int = None, max_message_id: int = None) -> list[Message]:
    query = """
        SELECT message_id, message_thread_id, reply_to_message_id, user_name,
               username, message_date, message, embedding
        FROM messages
        WHERE user_id = %s AND message_date IS NOT NULL AND embedding IS NOT NULL
    """
    params: list = [chat_id]
    if min_message_id is not None:
        query += " AND message_id >= %s"
        params.append(min_message_id)
    if max_message_id is not None:
        query += " AND message_id <= %s"
        params.append(max_message_id)
    query += " ORDER BY message_id ASC"

    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

    out = []
    for message_id, thread_id, reply_to, user_name, username, message_date, message, embedding in rows:
        try:
            text = decrypt(message)
        except Exception:
            continue
        author = resolve_display_name(username, user_name)
        # pgvector comes back over psycopg2 as a literal "[0.1,0.2,...]" string
        # -- no vector type adapter is registered in this project.
        vector = [float(x) for x in embedding.strip("[]").split(",")] if isinstance(embedding, str) else list(embedding)
        out.append(Message(
            message_id=message_id,
            chat_id=chat_id,
            thread_id=thread_id,
            reply_to_message_id=reply_to,
            author=author,
            timestamp=float(message_date),
            text=text,
            embedding=vector,
        ))
    return out
