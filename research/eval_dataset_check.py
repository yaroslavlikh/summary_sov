"""Read-only: for every item in the actual Langfuse eval dataset
(ask-pipeline-real-invocations) that has a reply anchor -- the only case
where conversation_id actually matters to /ask's _generate_answer window --
compare what PRODUCTION's conversation_id currently gives as context vs what
the semantic pilot would give, for the exact same real anchor message.

No writes anywhere. Doesn't touch chat_context/chat_moments/messages, doesn't
call /ask or run_eval.py, so it can't execute a memory_command either.
Usage: python3 -m research.eval_dataset_check
"""
import sys

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import config
from crypto_utils import decrypt
from database.db import get_conn
from langfuse import get_client

from research.conversation_disentanglement import semantic_pilot
from research.db_loader import load_messages

CHAT_ID = -1002335227490
DATASET_NAME = "ask-pipeline-real-invocations"


def baseline_window(anchor_message_id):
    """Reproduces the REAL production window query from llm/graphs.py
    _generate_answer -- conversation_id equality (or +-3 fallback) AND the
    is_bot filter (a row is only allowed in if it's genuinely content, or is
    the anchor itself) -- read-only. An earlier version of this function was
    missing the is_bot filter entirely, which let bot rows leak into the
    window that real production excludes; fixed after review."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.message_id, m.user_name, m.username, m.message, m.is_bot
            FROM messages m WHERE m.user_id = %s
              AND (m.is_bot = FALSE OR m.message_id = %s)
              AND EXISTS (
                SELECT 1 FROM messages anchor
                WHERE anchor.user_id = %s AND anchor.message_id = %s
                  AND (
                    (anchor.conversation_id IS NOT NULL AND m.conversation_id = anchor.conversation_id)
                    OR (anchor.conversation_id IS NULL AND m.message_id BETWEEN anchor.message_id - 3 AND anchor.message_id + 3)
                  )
            ) ORDER BY m.message_id ASC
            """,
            (CHAT_ID, anchor_message_id, CHAT_ID, anchor_message_id),
        )
        rows = cur.fetchall()
    out = []
    for mid, name, uname, msg, is_bot in rows:
        try:
            text = decrypt(msg)
        except Exception:
            text = "<decrypt fail>"
        out.append((mid, uname or name, text, is_bot))
    return out


def run():
    langfuse = get_client()
    dataset = langfuse.get_dataset(DATASET_NAME)
    anchor_cases = [item for item in dataset.items if item.input.get("replied_message_id")]
    print(f"{len(anchor_cases)} / {len(dataset.items)} eval cases have a reply anchor "
          f"(only these use conversation_id in production /ask)")

    print("\nLoading full real corpus + running semantic pilot once...")
    messages = load_messages(CHAT_ID)
    pilot_pred, pilot_results, tracker = semantic_pilot(messages)
    by_id = {m.message_id: m for m in messages}
    print(f"  {len(messages)} messages loaded")

    for item in anchor_cases:
        case = item.input
        anchor_mid = case["replied_message_id"]
        print("\n" + "=" * 100)
        print(f"{item.id}  |  {case['asker_name']}: {case['question']!r}  (reply -> {anchor_mid})")

        base_win = baseline_window(anchor_mid)
        print(f"\n  BASELINE window (production conversation_id / +-3 fallback): {len(base_win)} messages")
        for mid, author, text, is_bot in base_win:
            tag = " [BOT]" if is_bot else ""
            print(f"    [{mid}]{tag} {author}: {text[:70]!r}")

        if anchor_mid not in pilot_pred:
            print(f"\n  PILOT: anchor message_id={anchor_mid} not in loaded corpus "
                  f"(no message_date/embedding -- e.g. pre-migration history) -- pilot has nothing to say here.")
            continue

        pilot_conv_id = pilot_pred[anchor_mid]
        episode = tracker._episodes[pilot_conv_id]
        print(f"\n  PILOT window (episode {pilot_conv_id}): {len(episode.message_ids)} messages")
        for mid in episode.message_ids:
            m = by_id.get(mid)
            if m:
                print(f"    [{mid}] {m.author}: {m.text[:70]!r}")

        base_ids = {mid for mid, *_ in base_win}
        pilot_ids = set(episode.message_ids)
        only_baseline = base_ids - pilot_ids
        only_pilot = pilot_ids - base_ids
        print(f"\n  DIFF: only in baseline={sorted(only_baseline)}  only in pilot={sorted(only_pilot)}")


if __name__ == "__main__":
    run()
