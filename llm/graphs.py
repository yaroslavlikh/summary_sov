from __future__ import annotations

import html
import json
import re
from typing import Annotated, Any, Callable, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from chat_context import add_note, get_context_block
from chat_moments import add_moment, search_moments
from crypto_utils import decrypt
from database.db import get_conn
from display_names import resolve_display_name
from embeddings import embed, to_vector_literal
from llm.groq_client import get_chat_model, tracing_config
from llm.prompt import (
    prompt_for_classify_and_rewrite,
    prompt_for_context_extraction,
    prompt_for_llm,
    prompt_for_qa,
    prompt_for_query_rewrite,
    prompt_for_rerank,
)


def _content(message: AIMessage) -> str:
    return message.content if isinstance(message.content, str) else ""


def _config() -> dict[str, Any]:
    return {**tracing_config(), "recursion_limit": 20}


def _group_context(chat_id: int) -> str:
    notes = get_context_block(chat_id)
    if not notes:
        return ""
    return (
        "\nКонтекст о группе (не сами сообщения переписки, а накопленные заметки "
        "про участников, их характерные черты, повторяющиеся шутки/темы):\n"
        f"{notes}\n"
    )


def _or_query(cursor, question: str) -> Optional[str]:
    cursor.execute("SELECT plainto_tsquery('russian', %s)::text", (question,))
    row = cursor.fetchone()
    if not row or not row[0]:
        return None
    aliases = {"rag": "раг", "раг": "rag", "sql": "скл"}
    lexemes = set(re.findall(r"'([^']+)'", row[0]))
    lexemes.update(aliases[lex.lower()] for lex in list(lexemes) if lex.lower() in aliases)
    return " | ".join(f"'{lex}'" for lex in lexemes) or None


def _rrf(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def _message_link(chat_id: int, message_id: Optional[int], thread_id: Optional[int]) -> Optional[str]:
    if not message_id or not str(chat_id).startswith("-100"):
        return None
    internal_id = str(chat_id)[4:]
    return (
        f"https://t.me/c/{internal_id}/{thread_id}/{message_id}"
        if thread_id
        else f"https://t.me/c/{internal_id}/{message_id}"
    )


def _format_citations(text: str, legend: dict[int, Optional[str]]) -> str:
    escaped = html.escape(text)
    seen = []
    for match in re.finditer(r"\[(\d+)\]", escaped):
        index = int(match.group(1))
        if index in legend and legend[index] and index not in seen:
            seen.append(index)
    remap = {old: new for new, old in enumerate(seen, start=1)}

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        url = legend.get(index)
        return f'<a href="{url}">[{remap[index]}]</a> ' if url else ""

    return "\n".join(line.rstrip() for line in re.sub(r"\[(\d+)\]", replace, escaped).split("\n"))


class AskState(TypedDict, total=False):
    chat_id: int
    question: str
    asker_name: str
    replied_message_id: Optional[int]
    bot_username: Optional[str]
    thread_id: Optional[int]
    bot: Any
    save_bot_answer: Callable[..., None]
    anchor_id: Optional[int]
    intent: str
    effective_question: str
    fts_ids: list[int]
    vector_ids: list[int]
    candidate_ids: list[int]
    match_ids: set[int]
    window_rows: list[tuple]
    answer: Optional[str]
    answer_plain: Optional[str]


def _resolve_anchor(state: AskState) -> AskState:
    anchor_id = None
    if state.get("replied_message_id"):
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM messages WHERE user_id = %s AND message_id = %s",
                (state["chat_id"], state["replied_message_id"]),
            )
            row = cursor.fetchone()
            anchor_id = row[0] if row else None
    return {"anchor_id": anchor_id, "match_ids": {anchor_id} if anchor_id else set()}


def _after_anchor(state: AskState) -> Literal["generate_answer", "classify_intent"]:
    return "generate_answer" if state.get("anchor_id") else "classify_intent"


def _classify_intent(state: AskState, config: RunnableConfig) -> AskState:
    prompt = (
        "Определи единственную категорию вопроса. Верни только одно значение: "
        "self_identity (про себя), deictic (неполная ссылка на контекст), "
        "self_contained (самодостаточный вопрос), other_person (про другого человека).\n"
        f"Вопрос: {state['question']}"
    )
    intent = _content(get_chat_model("fast", 0).invoke(prompt, config=config)).strip().lower()
    if intent not in {"self_identity", "deictic", "self_contained", "other_person"}:
        intent = "self_contained"
    return {"intent": intent}


def _classify_and_rewrite(state: AskState, config: RunnableConfig) -> AskState:
    """Merged classify_intent + rewrite_query into one LLM round-trip
    (speed experiment) -- same context/rules, just one call instead of two
    sequential ones."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_name, username, message FROM messages WHERE user_id = %s ORDER BY id DESC LIMIT 35",
            (state["chat_id"],),
        )
        recent = cursor.fetchall()
        cursor.execute(
            "SELECT user_name, username, message FROM messages WHERE user_id = %s AND is_bot = TRUE ORDER BY id DESC LIMIT 5",
            (state["chat_id"],),
        )
        bot_rows = cursor.fetchall()
    rows = recent + bot_rows
    context = "\n".join(
        f"{resolve_display_name(username, user_name)}: {decrypt(text)}"
        for user_name, username, text in rows
    )
    prompt = prompt_for_classify_and_rewrite.format(
        context=context, question=state["question"], asker_name=state["asker_name"],
    )
    raw = _content(get_chat_model("fast", 0).invoke(prompt, config=config)).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    intent, rewritten = "self_contained", state["question"]
    if match:
        try:
            parsed = json.loads(match.group(0))
            if parsed.get("intent") in {"self_identity", "deictic", "self_contained", "other_person"}:
                intent = parsed["intent"]
            rewritten = (parsed.get("rewritten_question") or state["question"]).strip()
        except Exception:
            pass
    return {"intent": intent, "effective_question": rewritten or state["question"]}


def _rewrite_query(state: AskState, config: RunnableConfig) -> AskState:
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_name, username, message FROM messages WHERE user_id = %s ORDER BY id DESC LIMIT 35",
            (state["chat_id"],),
        )
        recent = cursor.fetchall()
        cursor.execute(
            "SELECT user_name, username, message FROM messages WHERE user_id = %s AND is_bot = TRUE ORDER BY id DESC LIMIT 5",
            (state["chat_id"],),
        )
        bot_rows = cursor.fetchall()
    rows = recent + bot_rows
    context = "\n".join(
        f"{resolve_display_name(username, user_name)}: {decrypt(text)}"
        for user_name, username, text in rows
    )
    prompt = prompt_for_query_rewrite.format(
        context=context,
        question=state["question"],
        asker_name=state["asker_name"],
        intent=state["intent"],
    )
    rewritten = _content(get_chat_model("fast", 0.2).invoke(prompt, config=config)).strip()
    return {"effective_question": rewritten or state["question"]}


def _search_fts(state: AskState) -> AskState:
    with get_conn() as conn:
        cursor = conn.cursor()
        query = _or_query(cursor, state["effective_question"])
        if not query:
            return {"fts_ids": []}
        cursor.execute(
            """
            SELECT id FROM messages
            WHERE user_id = %s AND is_bot = FALSE AND search_vector @@ to_tsquery('russian', %s)
            ORDER BY ts_rank(search_vector, to_tsquery('russian', %s)) DESC LIMIT 8
            """,
            (state["chat_id"], query, query),
        )
        return {"fts_ids": [row[0] for row in cursor.fetchall()]}


def _search_vector(state: AskState) -> AskState:
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id FROM messages
            WHERE user_id = %s AND is_bot = FALSE AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector LIMIT 8
            """,
            (state["chat_id"], to_vector_literal(embed(state["effective_question"]))),
        )
        return {"vector_ids": [row[0] for row in cursor.fetchall()]}


def _fuse_rrf(state: AskState) -> AskState:
    candidate_ids = _rrf([state.get("fts_ids", []), state.get("vector_ids", [])])[:10]
    if not candidate_ids:
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM messages WHERE user_id = %s AND is_bot = FALSE ORDER BY id DESC LIMIT 10",
                (state["chat_id"],),
            )
            candidate_ids = [row[0] for row in cursor.fetchall()]
    return {"candidate_ids": candidate_ids}


def _after_fuse_skip_small(state: AskState) -> Literal["rerank", "skip_rerank"]:
    """Speed experiment: with <=2 candidates there's nothing meaningful to
    rerank -- skip the LLM call and use them as-is."""
    candidate_ids = state.get("candidate_ids") or []
    return "skip_rerank" if len(candidate_ids) <= 2 else "rerank"


def _match_ids_from_candidates(state: AskState) -> AskState:
    return {"match_ids": set(state.get("candidate_ids") or [])}


def _rerank(state: AskState, config: RunnableConfig) -> AskState:
    candidate_ids = state.get("candidate_ids", [])
    if not candidate_ids:
        return {"match_ids": set()}
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, user_name, username, message FROM messages
            WHERE user_id = %s AND id = ANY(%s) ORDER BY id ASC
            """,
            (state["chat_id"], candidate_ids),
        )
        rows = cursor.fetchall()
    match_ids = {row[0] for row in rows}
    if len(rows) > 1:
        numbered = "\n".join(
            f"[{row_id}] {resolve_display_name(username, user_name)}: {decrypt(text)}"
            for row_id, user_name, username, text in rows
        )
        prompt = prompt_for_rerank.format(question=state["effective_question"], messages=numbered)
        selected = [int(value) for value in re.findall(r"\d+", _content(get_chat_model("fast", 0).invoke(prompt, config=config)))]
        selected = [value for value in selected if value in match_ids][:5]
        if selected:
            match_ids = set(selected)
    return {"match_ids": match_ids}


def _generate_answer(state: AskState, config: RunnableConfig) -> AskState:
    match_ids = state.get("match_ids", set())
    if not match_ids:
        return {"answer": None, "window_rows": []}
    with get_conn() as conn:
        cursor = conn.cursor()
        # A row is only allowed into the window if it's genuinely content
        # (is_bot=FALSE) OR it's one of the actual matches itself (a reply
        # can legitimately anchor on the bot's own prior answer) -- this
        # closes the self-pollution case where the bot's own EARLIER,
        # unrelated answer sat in the +-3 neighborhood and got cited as an
        # independent source about someone.
        cursor.execute(
            """
            SELECT m.id, m.message_id, m.message_thread_id, m.user_name, m.username, m.message
            FROM messages m WHERE m.user_id = %s
              AND (m.is_bot = FALSE OR m.id = ANY(%s))
              AND EXISTS (
                SELECT 1 FROM messages anchor
                WHERE anchor.user_id = %s AND anchor.id = ANY(%s)
                  AND m.message_id BETWEEN anchor.message_id - 3 AND anchor.message_id + 3
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
        legend[index] = _message_link(state["chat_id"], message_id, thread_id)
        anchor_tag = " [СООБЩЕНИЕ, НА КОТОРОЕ ОТВЕЧАЛИ]" if row_id == state.get("anchor_id") else ""
        lines.append(f"[{index}] {resolve_display_name(username, user_name)}: {decrypt(text)}{anchor_tag}")

    group_context = _group_context(state["chat_id"])
    moments = search_moments(state["chat_id"], embed(state["question"]), top_k=5)
    if moments:
        group_context += "\nВозможно релевантные моменты из истории:\n" + "\n".join(f"- {moment}" for moment in moments)
    prompt = prompt_for_qa.format(
        question=state["question"], messages="\n".join(lines), group_context=group_context, asker_name=state["asker_name"]
    )
    answer = _content(get_chat_model("primary", 0.3).invoke(prompt, config=config))
    return {
        "answer": _format_citations(answer, legend) if answer else None,
        "answer_plain": answer or None,
        "window_rows": rows,
    }


def _save_bot_answer(state: AskState) -> AskState:
    bot = state["bot"]
    if not state.get("answer"):
        bot.send_message(state["chat_id"], "В истории чата не нашёл ответа на этот вопрос.", message_thread_id=state.get("thread_id"))
        return {}
    sent = bot.send_message(state["chat_id"], state["answer"], parse_mode="HTML", message_thread_id=state.get("thread_id"))
    state["save_bot_answer"](
        state["chat_id"], sent.message_id, state.get("bot_username"), re.sub(r"\s*\[\d+\]", "", state.get("answer_plain") or ""),
        getattr(sent, "message_thread_id", state.get("thread_id")), getattr(sent, "date", None),
    )
    return {}


def build_ask_graph():
    graph = StateGraph(AskState)
    graph.add_node("resolve_anchor", _resolve_anchor)
    graph.add_node("classify_intent", _classify_intent)
    graph.add_node("rewrite_query", _rewrite_query)
    graph.add_node("search_fts", _search_fts)
    graph.add_node("search_vector", _search_vector)
    graph.add_node("fuse_rrf", _fuse_rrf)
    graph.add_node("rerank", _rerank)
    graph.add_node("generate_answer", _generate_answer)
    graph.add_node("save_bot_answer", _save_bot_answer)
    graph.add_edge(START, "resolve_anchor")
    graph.add_conditional_edges("resolve_anchor", _after_anchor)
    graph.add_edge("classify_intent", "rewrite_query")
    graph.add_edge("rewrite_query", "search_fts")
    graph.add_edge("rewrite_query", "search_vector")
    graph.add_edge("search_fts", "fuse_rrf")
    graph.add_edge("search_vector", "fuse_rrf")
    graph.add_edge("fuse_rrf", "rerank")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("generate_answer", "save_bot_answer")
    graph.add_edge("save_bot_answer", END)
    return graph.compile()


def build_ask_graph_merged():
    """Speed variant A: classify_intent + rewrite_query merged into one
    LLM call (_classify_and_rewrite) instead of two sequential ones."""
    graph = StateGraph(AskState)
    graph.add_node("resolve_anchor", _resolve_anchor)
    graph.add_node("classify_and_rewrite", _classify_and_rewrite)
    graph.add_node("search_fts", _search_fts)
    graph.add_node("search_vector", _search_vector)
    graph.add_node("fuse_rrf", _fuse_rrf)
    graph.add_node("rerank", _rerank)
    graph.add_node("generate_answer", _generate_answer)
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


def build_ask_graph_merged_skip_rerank():
    """Speed variant A+B: merged classify+rewrite, plus skip rerank
    entirely when the candidate pool is too small (<=2) to need it."""
    graph = StateGraph(AskState)
    graph.add_node("resolve_anchor", _resolve_anchor)
    graph.add_node("classify_and_rewrite", _classify_and_rewrite)
    graph.add_node("search_fts", _search_fts)
    graph.add_node("search_vector", _search_vector)
    graph.add_node("fuse_rrf", _fuse_rrf)
    graph.add_node("rerank", _rerank)
    graph.add_node("skip_rerank", _match_ids_from_candidates)
    graph.add_node("generate_answer", _generate_answer)
    graph.add_node("save_bot_answer", _save_bot_answer)
    graph.add_edge(START, "resolve_anchor")
    graph.add_conditional_edges(
        "resolve_anchor", lambda s: "generate_answer" if s.get("anchor_id") else "classify_and_rewrite"
    )
    graph.add_edge("classify_and_rewrite", "search_fts")
    graph.add_edge("classify_and_rewrite", "search_vector")
    graph.add_edge("search_fts", "fuse_rrf")
    graph.add_edge("search_vector", "fuse_rrf")
    graph.add_conditional_edges("fuse_rrf", _after_fuse_skip_small)
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("skip_rerank", "generate_answer")
    graph.add_edge("generate_answer", "save_bot_answer")
    graph.add_edge("save_bot_answer", END)
    return graph.compile()


ASK_GRAPH = build_ask_graph()
ASK_GRAPH_MERGED = build_ask_graph_merged()
ASK_GRAPH_MERGED_SKIP_RERANK = build_ask_graph_merged_skip_rerank()


def run_ask_graph(state: AskState) -> AskState:
    return ASK_GRAPH.invoke(state, config=_config())


def run_ask_graph_merged(state: AskState) -> AskState:
    return ASK_GRAPH_MERGED.invoke(state, config=_config())


def run_ask_graph_merged_skip_rerank(state: AskState) -> AskState:
    return ASK_GRAPH_MERGED_SKIP_RERANK.invoke(state, config=_config())


class SummaryState(TypedDict, total=False):
    chat_id: int
    thread_id: Optional[int]
    bot: Any
    prompt_body: str
    lines: list[str]
    legend: dict[int, Optional[str]]
    max_lines: int
    newest_included_id: int
    save_summary_state: Callable[[int, str], None]
    answer: Optional[str]
    messages: Annotated[list, add_messages]


def _generate_summary(state: SummaryState, config: RunnableConfig) -> SummaryState:
    prompt = prompt_for_llm.format(max_lines=state["max_lines"], group_context=_group_context(state["chat_id"])) + state["prompt_body"]
    answer = _content(get_chat_model("primary", 0.9).invoke(prompt, config=config))
    return {"answer": answer or None}


def _route_summary(state: SummaryState) -> Literal["send_summary", "send_failure"]:
    return "send_summary" if state.get("answer") else "send_failure"


def _send_summary(state: SummaryState) -> SummaryState:
    rendered = _format_citations(state["answer"] or "", state["legend"])
    state["bot"].send_message(
        state["chat_id"], f"#summary\n\n{rendered}", parse_mode="HTML", message_thread_id=state.get("thread_id")
    )
    return {"messages": [HumanMessage(content=prompt_for_context_extraction.format(messages="\n".join(state["lines"])))]}


def _save_summary_state(state: SummaryState) -> SummaryState:
    state["save_summary_state"](state["newest_included_id"], re.sub(r"\s*\[\d+\]", "", state["answer"] or ""))
    return {}


def _send_summary_failure(state: SummaryState) -> SummaryState:
    state["bot"].send_message(state["chat_id"], "LLM решил послать вас с ответом", message_thread_id=state.get("thread_id"))
    return {}


def build_summary_graph(chat_id: int):
    @tool
    def update_portrait(person: str, addition: str, source: str) -> str:
        """Записать характерную черту или факт о человеке."""
        tag = "о себе" if source == "self" else "со слов других"
        add_note(chat_id, f"[О {person}, {tag}]: {addition}", source="live")
        return "Записано"

    @tool
    def add_chat_lore(note: str) -> str:
        """Записать общую шутку, тему или факт о группе."""
        add_note(chat_id, note, source="live")
        return "Записано"

    @tool
    def record_moment(note: str) -> str:
        """Записать яркий момент, мнение или шутку из переписки."""
        add_moment(chat_id, note, embed(note))
        return "Записано"

    tools = [update_portrait, add_chat_lore, record_moment]
    model = get_chat_model("fast", 0.3).bind_tools(tools)

    def extract_context(state: SummaryState, config: RunnableConfig) -> SummaryState:
        return {"messages": [model.invoke(state["messages"], config=config)]}

    graph = StateGraph(SummaryState)
    graph.add_node("generate_summary", _generate_summary)
    graph.add_node("send_summary", _send_summary)
    graph.add_node("save_summary_state", _save_summary_state)
    graph.add_node("send_failure", _send_summary_failure)
    graph.add_node("extract_context", extract_context)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "generate_summary")
    graph.add_conditional_edges("generate_summary", _route_summary)
    graph.add_edge("send_summary", "save_summary_state")
    graph.add_edge("save_summary_state", "extract_context")
    graph.add_conditional_edges("extract_context", tools_condition)
    graph.add_edge("tools", "extract_context")
    graph.add_edge("send_failure", END)
    return graph.compile()


def run_summary_graph(state: SummaryState) -> SummaryState:
    return build_summary_graph(state["chat_id"]).invoke(state, config=_config())
