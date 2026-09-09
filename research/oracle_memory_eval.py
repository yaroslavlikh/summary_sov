"""Oracle-memory retrieval -- step 1 of the plan: (1) oracle-memory
retrieval, (2) a memory-dependent question dataset, (3) paired baseline vs
oracle-memory /ask comparison, (4) no-regression check against the
existing ask-pipeline-eval-run. This file is (1) plus the plumbing (3)
needs.

Uses the FROZEN 44 gold positive facts as a perfect-precision,
perfect-recall "oracle" memory store, deliberately isolating retrieval/
rerank/generation quality from extraction quality -- the forced-extractor
experiment already showed extraction alone needs more work (23% fully-
recovered recall), so testing retrieval on top of THAT would conflate two
separate questions. This is not a claim that extraction is solved; it's a
controlled experiment that assumes it away for now.

Runs entirely inside a sandbox Postgres schema (same pattern as
research/dry_run_extraction_full_history.py): real messages, chat_context
and chat_moments up to HISTORY_CUTOFF_MESSAGE_ID copied in unmodified
(ciphertext copied as-is -- crypto_utils doesn't depend on schema; only
search_vector is rebuilt, since that needs real plaintext, not ciphertext).
The 44 gold facts are inserted directly into memory_facts as the oracle
store, bypassing upsert_state's own resolution (subject_key is already
resolved and frozen -- this IS the oracle being granted, not something
under test). Never touches production.

Two graphs run against this SAME sandboxed environment, sharing every node
except the last one:
  - ask_baseline(): production's real, unmodified build_ask_graph_merged()
    pipeline nodes (today's actual /ask, as-is)
  - ask_oracle(): the same pipeline, except generate_answer also queries
    the oracle memory (memory_facts.search_facts_by_vector against the
    question's embedding) and injects results into group_context, exactly
    the way chat_moments already are -- the smallest possible diff from
    production between the two conditions.

MVP is vector-only retrieval (memory_facts.py's "memory_vector branch",
Sec 5.2 of the research doc). Entity-based retrieval (get_active_facts,
Sec 5.3) is a later ablation variant, not wired in here yet.

Usage: python3 -m research.oracle_memory_eval  (runs a small smoke test)
"""
import json
import sys

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import psycopg2
from psycopg2 import pool as pg_pool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

import config as _app_config  # noqa: triggers dotenv load -- named to avoid
# shadowing the "config" parameter LangGraph injects into node functions
# by name (see _generate_answer_oracle below).
import database.db as db
from config import get_database_url
from crypto_utils import decrypt, encrypt
from embeddings import embed, to_vector_literal
import llm.graphs as g
from llm.graphs import AskState
from memory_facts import search_facts_by_vector
from chat_moments import search_moments
from display_names import resolve_display_name
from llm.groq_client import get_chat_model
from llm.prompt import prompt_for_qa

CHAT_ID = -1002335227490
HISTORY_CUTOFF_MESSAGE_ID = 73100  # kept in sync with research/forced_extractor_eval.py
TEST_SCHEMA = "test_oracle_memory_eval"
GOLD_PATH = "/tmp/gold_facts.jsonl"


# --------------------------------------------------------------- sandbox --

def setup_sandbox():
    real_dsn = get_database_url()

    with psycopg2.connect(real_dsn) as real_conn:
        with real_conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, user_name, username, message, last_id, replied_message,
                       message_id, message_thread_id, reply_to_message_id, message_date,
                       conversation_id, is_bot, embedding::text
                FROM messages WHERE user_id = %s AND message_id <= %s
                ORDER BY id ASC
                """,
                (CHAT_ID, HISTORY_CUTOFF_MESSAGE_ID),
            )
            message_rows = cur.fetchall()
            cur.execute("SELECT id, chat_id, note, source FROM chat_context WHERE chat_id = %s", (CHAT_ID,))
            context_rows = cur.fetchall()
            cur.execute("SELECT id, chat_id, note, embedding::text FROM chat_moments WHERE chat_id = %s", (CHAT_ID,))
            moment_rows = cur.fetchall()
    print(f"Pulled {len(message_rows)} messages, {len(context_rows)} chat_context, {len(moment_rows)} chat_moments (real, read-only)", file=sys.stderr)

    test_pool = pg_pool.ThreadedConnectionPool(1, 5, dsn=real_dsn, options=f"-c search_path={TEST_SCHEMA},public")
    db._get_pool = lambda: test_pool

    with psycopg2.connect(real_dsn) as raw_conn:
        raw_conn.autocommit = True
        with raw_conn.cursor() as c:
            c.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE;")
            c.execute(f"CREATE SCHEMA {TEST_SCHEMA};")

    from database.init_db import init_db
    init_db()

    with db.get_conn() as conn:
        cur = conn.cursor()
        for row in message_rows:
            (rid, user_id, user_name, username, message, last_id, replied_message,
             message_id, message_thread_id, reply_to_message_id, message_date,
             conversation_id, is_bot, embedding_text) = row
            try:
                plaintext = decrypt(message)
            except Exception:
                plaintext = ""
            cur.execute(
                """
                INSERT INTO messages (
                    id, user_id, user_name, username, message, last_id, replied_message,
                    message_id, message_thread_id, reply_to_message_id, message_date,
                    conversation_id, is_bot, search_vector, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                          to_tsvector('russian', %s), %s::vector)
                """,
                (rid, user_id, user_name, username, message, last_id, replied_message,
                 message_id, message_thread_id, reply_to_message_id, message_date,
                 conversation_id, is_bot, plaintext, embedding_text),
            )
        for rid, chat_id, note, source in context_rows:
            cur.execute(
                "INSERT INTO chat_context (id, chat_id, note, source) VALUES (%s, %s, %s, %s)",
                (rid, chat_id, note, source),
            )
        for rid, chat_id, note, embedding_text in moment_rows:
            cur.execute(
                "INSERT INTO chat_moments (id, chat_id, note, embedding) VALUES (%s, %s, %s, %s::vector)",
                (rid, chat_id, note, embedding_text),
            )
        conn.commit()
    print("Sandbox populated with real data.", file=sys.stderr)

    gold = [json.loads(line) for line in open(GOLD_PATH)]
    positive = [x for x in gold if x["label"] == "positive"]
    with db.get_conn() as conn:
        cur = conn.cursor()
        for fact in positive:
            retrieval_text = fact["claim"]
            vec = embed(retrieval_text)
            cur.execute(
                """
                INSERT INTO memory_facts (
                    chat_id, subject_key, subject_display, state_key, kind, claim, retrieval_text,
                    embedding, importance, source_message_ids, active
                ) VALUES (%s, %s, %s, NULL, %s, %s, %s, %s::vector, 1, %s, TRUE)
                """,
                (CHAT_ID, fact["subject_key"], fact["subject"], fact["kind"],
                 encrypt(fact["claim"]), encrypt(retrieval_text), to_vector_literal(vec),
                 fact["source_message_ids"]),
            )
        conn.commit()
    print(f"Inserted {len(positive)} oracle facts into sandboxed memory_facts.", file=sys.stderr)
    return test_pool


# ------------------------------------------------------- oracle variant --

def _generate_answer_oracle(state: AskState, config: RunnableConfig):
    """Identical to llm.graphs._generate_answer except for the ORACLE
    MEMORY block -- the smallest possible diff from production between the
    baseline and oracle conditions."""
    match_ids = state.get("match_ids", set())
    if not match_ids:
        return {"answer": None, "window_rows": []}
    with db.get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT m.id, m.message_id, m.message_thread_id, m.user_name, m.username, m.message
            FROM messages m WHERE m.user_id = %s
              AND (m.is_bot = FALSE OR m.id = ANY(%s))
              AND EXISTS (
                SELECT 1 FROM messages anchor
                WHERE anchor.user_id = %s AND anchor.id = ANY(%s)
                  AND (
                    (anchor.conversation_id IS NOT NULL AND m.conversation_id = anchor.conversation_id)
                    OR (anchor.conversation_id IS NULL AND m.message_id BETWEEN anchor.message_id - 3 AND anchor.message_id + 3)
                  )
              ) ORDER BY m.message_id ASC
            """,
            (state["chat_id"], list(match_ids), state["chat_id"], list(match_ids)),
        )
        rows = cursor.fetchall()
    if not rows:
        return {"answer": None, "window_rows": []}
    legend = {}
    lines = []
    for index, (row_id, message_id, thread_id, user_name, username, text) in enumerate(rows, start=1):
        legend[index] = g._message_link(state["chat_id"], message_id, thread_id)
        anchor_tag = " [СООБЩЕНИЕ, НА КОТОРОЕ ОТВЕЧАЛИ]" if row_id == state.get("anchor_id") else ""
        lines.append(f"[{index}] {resolve_display_name(username, user_name)}: {decrypt(text)}{anchor_tag}")

    group_context = g._group_context(state["chat_id"])
    moments = search_moments(state["chat_id"], embed(state["question"]), top_k=5)
    if moments:
        group_context += "\nВозможно релевантные моменты из истории:\n" + "\n".join(f"- {m}" for m in moments)

    # ORACLE MEMORY -- the only substantive difference from production.
    facts = search_facts_by_vector(state["chat_id"], embed(state["question"]), top_k=8)
    if facts:
        group_context += "\nИзвестные факты из памяти о людях/группе (могут быть релевантны):\n" + "\n".join(
            f"- {f['claim']}" for f in facts
        )

    prompt = prompt_for_qa.format(
        question=state["question"], messages="\n".join(lines), group_context=group_context, asker_name=state["asker_name"]
    )
    answer = g._content(get_chat_model("primary", 0.3).invoke(prompt, config=config))
    return {
        "answer": g._format_citations(answer, legend) if answer else None,
        "answer_plain": answer or None,
        "window_rows": rows,
        "oracle_facts_used": [f["claim"] for f in facts],
    }


def _route_after_anchor(state):
    return "generate_answer" if state.get("anchor_id") else "classify_and_rewrite"


def build_ask_graph_baseline():
    """Exactly production's build_ask_graph_merged(), minus the Telegram-
    sending save_bot_answer node (not needed for eval -- read state["answer"]
    from the invoke() result directly)."""
    graph = StateGraph(AskState)
    graph.add_node("resolve_anchor", g._resolve_anchor)
    graph.add_node("classify_and_rewrite", g._classify_and_rewrite)
    graph.add_node("search_fts", g._search_fts)
    graph.add_node("search_vector", g._search_vector)
    graph.add_node("fuse_rrf", g._fuse_rrf)
    graph.add_node("rerank", g._rerank)
    graph.add_node("generate_answer", g._generate_answer)
    graph.add_edge(START, "resolve_anchor")
    graph.add_conditional_edges("resolve_anchor", _route_after_anchor)
    graph.add_edge("classify_and_rewrite", "search_fts")
    graph.add_edge("classify_and_rewrite", "search_vector")
    graph.add_edge("search_fts", "fuse_rrf")
    graph.add_edge("search_vector", "fuse_rrf")
    graph.add_edge("fuse_rrf", "rerank")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


def build_ask_graph_oracle():
    """Same as build_ask_graph_baseline() except generate_answer is the
    oracle-memory-augmented variant. Every other node is the SAME imported
    production function."""
    graph = StateGraph(AskState)
    graph.add_node("resolve_anchor", g._resolve_anchor)
    graph.add_node("classify_and_rewrite", g._classify_and_rewrite)
    graph.add_node("search_fts", g._search_fts)
    graph.add_node("search_vector", g._search_vector)
    graph.add_node("fuse_rrf", g._fuse_rrf)
    graph.add_node("rerank", g._rerank)
    graph.add_node("generate_answer", _generate_answer_oracle)
    graph.add_edge(START, "resolve_anchor")
    graph.add_conditional_edges("resolve_anchor", _route_after_anchor)
    graph.add_edge("classify_and_rewrite", "search_fts")
    graph.add_edge("classify_and_rewrite", "search_vector")
    graph.add_edge("search_fts", "fuse_rrf")
    graph.add_edge("search_vector", "fuse_rrf")
    graph.add_edge("fuse_rrf", "rerank")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


def ask(graph, question, asker_name="Ярик Лихачев", replied_message_id=None, thread_id=None):
    state: AskState = {
        "chat_id": CHAT_ID, "question": question, "asker_name": asker_name,
        "replied_message_id": replied_message_id, "bot_username": "sov_summary_bot",
        "thread_id": thread_id,
    }
    return graph.invoke(state, config={"recursion_limit": 20})


# --------------------------------------------------------------- smoke --

SMOKE_QUESTIONS = [
    "Что Ярик думает про Крым?",
    "Что Игорь говорил про Ксюшу?",
    "Пользуется ли Ярик VPN?",
]


def smoke_test():
    setup_sandbox()
    baseline_graph = build_ask_graph_baseline()
    oracle_graph = build_ask_graph_oracle()

    for q in SMOKE_QUESTIONS:
        print(f"\n=== {q} ===", file=sys.stderr)
        b = ask(baseline_graph, q)
        o = ask(oracle_graph, q)
        print(f"baseline: {b.get('answer_plain')}", file=sys.stderr)
        print(f"oracle:   {o.get('answer_plain')}", file=sys.stderr)
        if o.get("oracle_facts_used"):
            print(f"oracle facts injected: {o['oracle_facts_used']}", file=sys.stderr)


if __name__ == "__main__":
    smoke_test()
