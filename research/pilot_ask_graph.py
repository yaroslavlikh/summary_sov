"""Full /ask graph, identical to llm.graphs.build_ask_graph_merged, except
`generate_answer`'s window comes from the semantic pilot's conversation
episode instead of the production conversation_id SQL query. Every other
node (resolve_anchor, classify_and_rewrite, search_fts/vector, fuse_rrf,
rerank, save_bot_answer) is reused as-is, unmodified, imported directly from
llm.graphs -- so the ONLY variable between this and production is the
anchor-window source.
"""
import re
import sys

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

from langgraph.graph import END, START, StateGraph

from chat_moments import search_moments
from crypto_utils import decrypt
from database.db import get_conn
from display_names import resolve_display_name
from embeddings import embed
from llm.groq_client import get_chat_model
from llm.prompt import prompt_for_qa
from llm.graphs import (
    AskState, _classify_and_rewrite, _content, _format_citations, _fuse_rrf,
    _group_context, _message_link, _resolve_anchor, _rerank, _save_bot_answer,
    _search_fts, _search_vector,
)

from research.conversation_disentanglement import semantic_pilot
from research.db_loader import load_messages

CHAT_ID = -1002335227490

print("Building semantic pilot index over the full real corpus...", file=sys.stderr)
_messages_all = load_messages(CHAT_ID)
_pilot_pred, _pilot_results, _tracker = semantic_pilot(_messages_all)
print(f"  {len(_messages_all)} messages, pilot ready", file=sys.stderr)


def _window_rows(chat_id, message_ids):
    if not message_ids:
        return []
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, message_id, message_thread_id, user_name, username, message
            FROM messages WHERE user_id = %s AND message_id = ANY(%s)
            ORDER BY message_id ASC
            """,
            (chat_id, list(message_ids)),
        )
        return cur.fetchall()


def _generate_answer_pilot(state, config):
    match_ids = state.get("match_ids", set())
    if not match_ids:
        return {"answer": None, "window_rows": []}

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT message_id FROM messages WHERE id = ANY(%s)", (list(match_ids),))
        anchor_message_ids = [r[0] for r in cur.fetchall()]

    conv_ids = {_pilot_pred[mid] for mid in anchor_message_ids if mid in _pilot_pred}
    if not conv_ids:
        # pilot has no data for this anchor either (outside loaded corpus) --
        # same "nothing to answer from" outcome production would also hit.
        return {"answer": None, "window_rows": []}

    message_ids = []
    for conv_id in conv_ids:
        message_ids.extend(_tracker._episodes[conv_id].message_ids)
    rows = _window_rows(state["chat_id"], sorted(set(message_ids)))
    if not rows:
        return {"answer": None, "window_rows": []}

    legend, lines = {}, []
    for index, (row_id, message_id, thread_id, user_name, username, text) in enumerate(rows, start=1):
        legend[index] = _message_link(state["chat_id"], message_id, thread_id)
        anchor_tag = " [СООБЩЕНИЕ, НА КОТОРОЕ ОТВЕЧАЛИ]" if row_id == state.get("anchor_id") else ""
        lines.append(f"[{index}] {resolve_display_name(username, user_name)}: {decrypt(text)}{anchor_tag}")

    group_context = _group_context(state["chat_id"])
    moments = search_moments(state["chat_id"], embed(state["question"]), top_k=5)
    if moments:
        group_context += "\nВозможно релевантные моменты из истории:\n" + "\n".join(f"- {m}" for m in moments)
    prompt = prompt_for_qa.format(
        question=state["question"], messages="\n".join(lines), group_context=group_context, asker_name=state["asker_name"]
    )
    answer = _content(get_chat_model("primary", 0.3).invoke(prompt, config=config))
    return {
        "answer": _format_citations(answer, legend) if answer else None,
        "answer_plain": answer or None,
        "window_rows": rows,
    }


def build_ask_graph_pilot_window():
    graph = StateGraph(AskState)
    graph.add_node("resolve_anchor", _resolve_anchor)
    graph.add_node("classify_and_rewrite", _classify_and_rewrite)
    graph.add_node("search_fts", _search_fts)
    graph.add_node("search_vector", _search_vector)
    graph.add_node("fuse_rrf", _fuse_rrf)
    graph.add_node("rerank", _rerank)
    graph.add_node("generate_answer", _generate_answer_pilot)
    graph.add_node("save_bot_answer", _save_bot_answer)
    graph.add_edge(START, "resolve_anchor")
    graph.add_conditional_edges(
        "resolve_anchor", lambda s: "generate_answer" if s.get("anchor_id") else "classify_and_rewrite"
    )
    graph.add_edge("classify_and_rewrite", "search_fts")
    graph.add_edge("classify_and_rewrite", "search_vector")
    graph.add_edge("search_fts", "fuse_rrf")
    graph.add_edge("search_vector", "fuse_rrf")
    graph.add_edge("fuse_rrf", "rerank")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("generate_answer", "save_bot_answer")
    graph.add_edge("save_bot_answer", END)
    return graph.compile()


PILOT_ASK_GRAPH = build_ask_graph_pilot_window()


def run_pilot_ask_graph(state):
    from llm.graphs import _config
    return PILOT_ASK_GRAPH.invoke(state, config=_config())
