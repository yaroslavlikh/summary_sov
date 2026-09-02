from database.db import get_conn
from embeddings import to_vector_literal


def add_moment(chat_id, note, embedding):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_moments (chat_id, note, embedding) VALUES (%s, %s, %s::vector)",
            (chat_id, note, to_vector_literal(embedding)),
        )
        conn.commit()


def delete_all_moments(chat_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_moments WHERE chat_id = %s", (chat_id,))
        conn.commit()


def search_moments(chat_id, query_embedding, top_k=5):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT note FROM chat_moments
            WHERE chat_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (chat_id, to_vector_literal(query_embedding), top_k),
        )
        return [row[0] for row in cursor.fetchall()]
