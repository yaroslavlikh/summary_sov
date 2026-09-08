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
    # Merges a new fact into that person's existing "Портрет: <person> — ..."
    # note (matched by decrypted prefix, since `note` is encrypted at rest
    # and can't be filtered with SQL LIKE) instead of either growing an
    # unbounded list of separate rows (like update_portrait's one-fact-per-row
    # appends during /summary batch extraction) or blindly replacing the
    # whole thing -- a blind replace silently drops earlier facts whenever
    # the caller's `description` is just the new addition on its own, which
    # is the common case here since the caller (an LLM call elsewhere) isn't
    # necessarily shown the prior text.
    prefix = f"Портрет: {person} —"
    # Strip a redundant leading "<person> —"/"<person> -" in case the caller
    # echoed the person's name into `description` itself.
    description = re.sub(rf"^{re.escape(person)}\s*[—-]\s*", "", description.strip())
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, note FROM chat_context WHERE chat_id = %s AND source IN ('auto', 'live')",
            (chat_id,),
        )
        existing = next(
            ((note_id, decrypt(note)) for note_id, note in cursor.fetchall() if decrypt(note).startswith(prefix)),
            None,
        )
        if existing:
            existing_id, existing_note = existing
            prior = existing_note[len(prefix):].strip()
            new_note = f"{prefix} {prior}; {description}" if prior else f"{prefix} {description}"
            cursor.execute(
                "UPDATE chat_context SET note = %s WHERE id = %s", (encrypt(new_note), existing_id)
            )
        else:
            new_note = f"{prefix} {description}"
            cursor.execute(
                "INSERT INTO chat_context (chat_id, note, source) VALUES (%s, %s, 'live')",
                (chat_id, encrypt(new_note)),
            )
        conn.commit()
    return existing is not None


def get_context_block(chat_id):
    notes = list_notes(chat_id)
    if not notes:
        return None
    return "\n".join(f"- {note}" for _, note in notes)
