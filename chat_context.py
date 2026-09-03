from crypto_utils import decrypt, encrypt
from database.db import get_conn


def add_note(chat_id, note, source='manual'):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_context (chat_id, note, source) VALUES (%s, %s, %s) RETURNING id",
            (chat_id, encrypt(note), source),
        )
        note_id = cursor.fetchone()[0]
        conn.commit()
    return note_id


def list_notes(chat_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, note FROM chat_context WHERE chat_id = %s ORDER BY id",
            (chat_id,),
        )
        return [(note_id, decrypt(note)) for note_id, note in cursor.fetchall()]


def remove_note(chat_id, note_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM chat_context WHERE chat_id = %s AND id = %s",
            (chat_id, note_id),
        )
        deleted = cursor.rowcount
        conn.commit()
    return deleted > 0


def delete_auto_notes(chat_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM chat_context WHERE chat_id = %s AND source = 'auto'",
            (chat_id,),
        )
        conn.commit()


def get_context_block(chat_id):
    notes = list_notes(chat_id)
    if not notes:
        return None
    return "\n".join(f"- {note}" for _, note in notes)
