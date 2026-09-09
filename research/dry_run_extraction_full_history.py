"""Stage 1 hardening PR, final step: dry-run extraction over the ENTIRE real
chat history, using write_state=collect (nothing hits memory_facts, real or
sandboxed production tables) so we can see exactly what the hardened
extraction would propose across every batch, without committing to it.

Runs entirely inside a sandbox schema (real messages copied in, same
pattern as earlier sidecar scripts) so update_portrait/add_chat_lore/
record_moment -- which have no dry-run guard of their own -- also can't
touch production chat_context/chat_moments.

Chunks history into ~180-message batches (matching what earlier sidecar
testing showed is a manageable extraction-prompt size), runs the real
build_summary_graph tool-calling loop on each, and reports every
upsert_state candidate the model proposed -- including ones that would have
been rejected (ambiguous subject, out-of-batch source) -- for manual review.

Usage: python3 -m research.dry_run_extraction_full_history
"""
import json
import sys

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import psycopg2
from psycopg2 import pool as pg_pool

import database.db as db
from config import get_database_url

CHAT_ID = -1002335227490
TEST_SCHEMA = "test_dry_run_extraction_full_history"
BATCH_SIZE = 180

_real_dsn = get_database_url()

print("Pulling full real history (still encrypted)...", file=sys.stderr)
with psycopg2.connect(_real_dsn) as real_conn:
    with real_conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, user_name, username, message, last_id, replied_message,
                   message_id, message_thread_id, reply_to_message_id, message_date,
                   conversation_id, is_bot
            FROM messages WHERE user_id = %s AND is_bot = FALSE
            ORDER BY id ASC
            """,
            (CHAT_ID,),
        )
        all_rows = cur.fetchall()
print(f"  {len(all_rows)} real human messages", file=sys.stderr)

_test_pool = pg_pool.ThreadedConnectionPool(1, 5, dsn=_real_dsn, options=f"-c search_path={TEST_SCHEMA},public")
db._get_pool = lambda: _test_pool

with psycopg2.connect(_real_dsn) as raw_conn:
    raw_conn.autocommit = True
    with raw_conn.cursor() as c:
        c.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE;")
        c.execute(f"CREATE SCHEMA {TEST_SCHEMA};")
print("Sandbox schema created.", file=sys.stderr)

try:
    from database.init_db import init_db
    init_db()
    print("init_db() OK", file=sys.stderr)

    with db.get_conn() as conn:
        cur = conn.cursor()
        for row in all_rows:
            cur.execute(
                """
                INSERT INTO messages (
                    id, user_id, user_name, username, message, last_id, replied_message,
                    message_id, message_thread_id, reply_to_message_id, message_date,
                    conversation_id, is_bot, search_vector
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('russian', ''))
                """,
                row,
            )
        conn.commit()
    print(f"Copied {len(all_rows)} rows into sandbox.", file=sys.stderr)

    from display_names import resolve_display_name
    from crypto_utils import decrypt
    import llm.graphs as g
    from participants import resolve_participant_key

    class _NullBot:
        def send_message(self, chat_id, text, parse_mode=None, message_thread_id=None):
            class M:
                message_id = None
            return M()

    candidates = []

    def make_collector(batch_no):
        def collect_to_json(chat_id, subject, state_key, kind, claim, retrieval_cues, source_message_ids):
            resolved = resolve_participant_key(chat_id, subject)
            candidates.append({
                "batch": batch_no,
                "subject_raw": subject,
                "resolved_key": resolved[0] if resolved else None,
                "resolved_display": resolved[1] if resolved else None,
                "ambiguous_or_unknown": resolved is None,
                "state_key": state_key,
                "kind": kind,
                "claim": claim,
                "retrieval_cues": retrieval_cues,
                "source_message_ids": source_message_ids,
            })
        return collect_to_json

    n_batches = (len(all_rows) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_no in range(n_batches):
        chunk = all_rows[batch_no * BATCH_SIZE : (batch_no + 1) * BATCH_SIZE]
        if not chunk:
            continue
        print(f"\n=== batch {batch_no + 1}/{n_batches} ({len(chunk)} messages) ===", file=sys.stderr)

        legend, lines, extraction_lines, batch_message_ids = {}, [], [], set()
        for idx, row in enumerate(chunk, start=1):
            (row_id, user_id, user_name, username, message, last_id, replied,
             msg_id, thread_id, reply_to, message_date, conv_id, is_bot) = row
            author = resolve_display_name(username, user_name)
            text = decrypt(message)
            lines.append(f"[{idx}] {author}: {text}")
            id_tag = f"message_id={msg_id}" if msg_id is not None else "message_id=unknown"
            extraction_lines.append(f"[{idx}] {{{id_tag}}} {author}: {text}")
            if msg_id is not None:
                batch_message_ids.add(msg_id)

        state = {
            "chat_id": CHAT_ID, "thread_id": None, "bot": _NullBot(),
            "prompt_body": "Новые сообщения:\n" + "\n".join(lines),
            "lines": lines, "extraction_lines": extraction_lines,
            "batch_message_ids": batch_message_ids,
            "legend": legend, "max_lines": 18,
            "newest_included_id": chunk[-1][0],
            "save_summary_state": lambda *a, **kw: None,
            "write_state": make_collector(batch_no + 1),
        }

        def _count_context():
            with db.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM chat_context WHERE chat_id = %s", (CHAT_ID,))
                ctx = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM chat_moments WHERE chat_id = %s", (CHAT_ID,))
                mom = cur.fetchone()[0]
            return ctx, mom

        before_ctx, before_mom = _count_context()
        try:
            # Bypass run_summary_graph's Langfuse tracing for this offline
            # research run -- a Langfuse read-timeout killed batch 4 outright
            # in the first attempt (broke the pooled Postgres connection
            # too), and tracing isn't needed to see what extraction proposes.
            graph = g.build_summary_graph(CHAT_ID, batch_message_ids, state["write_state"])
            graph.invoke(state, config={"recursion_limit": 20})
        except Exception as e:
            print(f"  batch {batch_no + 1} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        after_ctx, after_mom = _count_context()
        print(
            f"  batch {batch_no + 1}: +{after_ctx - before_ctx} chat_context, "
            f"+{after_mom - before_mom} chat_moments, "
            f"+{len([c for c in candidates if c['batch'] == batch_no + 1])} upsert_state candidates",
            file=sys.stderr,
        )

    print(f"\n\n=== TOTAL upsert_state candidates: {len(candidates)} ===")
    for c in candidates:
        print(json.dumps(c, ensure_ascii=False))

    resolved_ok = [c for c in candidates if not c["ambiguous_or_unknown"]]
    print(f"\nresolved (would be written): {len(resolved_ok)} / {len(candidates)}", file=sys.stderr)

    final_ctx, final_mom = _count_context()
    print(f"\nFinal sandbox chat_context rows: {final_ctx}, chat_moments rows: {final_mom}", file=sys.stderr)

    with open("/tmp/dry_run_candidates.json", "w") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    print("Saved to /tmp/dry_run_candidates.json", file=sys.stderr)

finally:
    with psycopg2.connect(_real_dsn) as raw_conn:
        raw_conn.autocommit = True
        with raw_conn.cursor() as c:
            c.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE;")
    print("Sandbox schema dropped.", file=sys.stderr)
