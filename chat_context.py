import re

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


def upsert_portrait(chat_id, person, description):
    # Replaces that person's existing "Портрет: <person> — ..." note
    # wholesale (matched by decrypted prefix, since `note` is encrypted at
    # rest and can't be filtered with SQL LIKE) instead of appending --
    # unlike update_portrait's incremental one-fact-at-a-time appends during
    # /summary batch extraction, an explicit "запомни"/"измени" command from
    # a user is meant to set the description, not grow an unbounded list.
    prefix = f"Портрет: {person} —"
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, note FROM chat_context WHERE chat_id = %s AND source IN ('auto', 'live')",
            (chat_id,),
        )
        existing_id = next(
            (note_id for note_id, note in cursor.fetchall() if decrypt(note).startswith(prefix)),
            None,
        )
        # The model sometimes echoes the person's name again inside its own
        # description (having just seen the existing "Портрет: X —" note as
        # context) -- strip a redundant leading "X —"/"X -" so it doesn't
        # end up doubled after the prefix this function already adds.
        description = re.sub(rf"^{re.escape(person)}\s*[—-]\s*", "", description.strip())
        new_note = f"{prefix} {description}"
        if existing_id:
            cursor.execute(
                "UPDATE chat_context SET note = %s WHERE id = %s", (encrypt(new_note), existing_id)
            )
        else:
            cursor.execute(
                "INSERT INTO chat_context (chat_id, note, source) VALUES (%s, %s, 'live')",
                (chat_id, encrypt(new_note)),
            )
        conn.commit()
    return existing_id is not None


def get_context_block(chat_id):
    notes = list_notes(chat_id)
    if not notes:
        return None
    return "\n".join(f"- {note}" for _, note in notes)
