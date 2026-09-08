"""Pull real historical /ask-style invocations (mention-triggered questions) out
of production `messages` -- these become the eval dataset's input items. We
don't hand-label ground truth: judges assess relevance live, given a wide
+-10 message window with real dates/authors, per case.
"""
import sys

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

from crypto_utils import decrypt
from database.db import get_conn
from display_names import resolve_display_name

BOT_USERNAME = "sov_summary_bot"
MENTION_TAG = f"@{BOT_USERNAME}"


def mine_cases(chat_id, limit=50):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, message, message_id, reply_to_message_id, user_name, username, message_date
            FROM messages
            WHERE user_id = %s AND is_bot = FALSE
            ORDER BY id DESC
            LIMIT 5000
            """,
            (chat_id,),
        )
        rows = cursor.fetchall()

    cases = []
    for row_id, message, message_id, reply_to, user_name, username, message_date in rows:
        try:
            plain = decrypt(message)
        except Exception:
            continue
        if MENTION_TAG.lower() not in plain.lower():
            continue

        idx = plain.lower().find(MENTION_TAG.lower())
        question = (plain[:idx] + plain[idx + len(MENTION_TAG):]).strip()
        if not question:
            continue

        cases.append({
            "chat_id": chat_id,
            "question": question,
            "asker_name": resolve_display_name(username, user_name),
            "replied_message_id": reply_to,
            "bot_username": BOT_USERNAME,
            "source_row_id": row_id,
            "source_message_id": message_id,
            "message_date": message_date,
        })
        if len(cases) >= limit:
            break

    return cases


if __name__ == "__main__":
    cases = mine_cases(-1002335227490, limit=30)
    print(f"Mined {len(cases)} real invocation cases")
    for c in cases[:10]:
        print(f"  [{c['asker_name']}] {c['question']!r} (reply={bool(c['replied_message_id'])}, date={c['message_date']})")
