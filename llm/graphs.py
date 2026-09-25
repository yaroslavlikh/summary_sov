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

from chat_context import add_note, get_context_block, upsert_portrait
from chat_moments import add_moment, search_moments
from config import episodes_in_ask_enabled
from crypto_utils import decrypt
from database.db import get_conn
from display_names import resolve_display_name
from embeddings import embed, to_vector_literal
from episodic_memory import (
    load_events,
    render_event,
    resolve_question_participant,
    search_events,
    search_events_by_participant,
)
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
    vector_ids: list[Any]
    memory_ids: list[int]
    candidate_ids: list[int]
    candidate_event_ids: list[int]
    match_ids: set[int]
    match_event_ids: list[int]
    episode_row_ids: list[int]
    context_lines: list[str]
    episode_context: str
    # None -> MEMORY_EPISODES_IN_ASK decides; evals set it explicitly per condition.
    use_episodes: Optional[bool]
    window_rows: list[tuple]
    answer: Optional[str]
    answer_plain: Optional[str]
    handled_as_memory: bool
    memory_reply: Optional[str]
    write_memory: Optional[Callable[[int, Optional[str], str], None]]


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


def _default_write_memory(chat_id: int, memory_target: Optional[str], memory_note: str) -> None:
    if memory_target:
        upsert_portrait(chat_id, memory_target, memory_note)
    else:
        add_note(chat_id, memory_note, source="live")


def _classify_and_rewrite(state: AskState, config: RunnableConfig) -> AskState:
    """Merged classify_intent + rewrite_query + memory-command detection into
    one LLM round-trip -- an explicit instruction like "запомни, что..." or
    "обращайся к X только как Y" (not a question at all) is recognized here
    too, piggybacking on the same call that already runs for every non-anchor
    message instead of costing a second one. A keyword pre-filter was tried
    first and dropped -- natural phrasings for this ("обращайся к X как Y",
    "зови меня Y") don't share a small fixed vocabulary, so keyword-matching
    missed real cases in production; the model call already happening here is
    reliable and free."""
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
    memory_target, memory_note = None, None
    if match:
        try:
            parsed = json.loads(match.group(0))
            if parsed.get("intent") in {
                "self_identity", "deictic", "self_contained", "other_person", "memory_command",
            }:
                intent = parsed["intent"]
            rewritten = (parsed.get("rewritten_question") or state["question"]).strip()
            memory_target = (parsed.get("memory_target") or "").strip() or None
            memory_note = (parsed.get("memory_note") or "").strip() or None
        except Exception:
            pass

    if intent == "memory_command" and memory_note:
        # The model is instructed to default memory_target to the asker for
        # self-referential commands ("подлизывайся КО МНЕ"), but doesn't
        # reliably follow that -- a real one landed as an ungrounded general
        # note instead of Игорь's portrait, meaning "suck up to him"
        # silently became "suck up to everyone". Backstop deterministically:
        # if the model left memory_target empty AND the original message is
        # clearly first-person, it's about the asker, not a stray group fact.
        if not memory_target and re.search(r"\b(мне|меня|мной|мой|моя|моё|мои)\b", state["question"], re.IGNORECASE):
            memory_target = state["asker_name"]
        write_memory = state.get("write_memory") or _default_write_memory
        write_memory(state["chat_id"], memory_target, memory_note)
        return {"intent": intent, "handled_as_memory": True, "memory_reply": "Записал"}

    return {"intent": intent, "effective_question": rewritten or state["question"], "handled_as_memory": False}


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
    if state.get("handled_as_memory"):
        return {"fts_ids": []}
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


_VECTOR_TOP_K = 8
_MEMORY_TOP_K = 5
_RESERVED_MEMORY_SLOTS = 2
# A memory hit has to be close in absolute terms AND not much worse than the best raw
# hit for the same question. Measured on 40 real questions: raw messages sit at a
# median cosine distance of .191, events at .326, so an unfiltered memory list rides
# into the pool on every question and displaces ~3 raw messages each time, most of
# them for an event that has nothing to do with the question.
_MEMORY_MAX_DISTANCE = 0.35
_MEMORY_MARGIN_OVER_RAW = 0.10
_MEMORY_PARTICIPANT_LIMIT = 2
# The reranker may keep at most this many memory entries, so raw matches are never
# displaced wholesale: in the first paired run a case where memory took every slot
# collapsed the context from 338 messages to 1 and the answer got worse.
_MAX_SELECTED_MEMORY = 2


def _episodes_on(state: AskState) -> bool:
    flag = state.get("use_episodes")
    return episodes_in_ask_enabled() if flag is None else bool(flag)


def _memory_key(event_id: int) -> str:
    return f"E{event_id}"


def _search_vector(state: AskState) -> AskState:
    if state.get("handled_as_memory"):
        return {"vector_ids": []}
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id FROM messages
            WHERE user_id = %s AND is_bot = FALSE AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector LIMIT %s
            """,
            (state["chat_id"], to_vector_literal(embed(state["effective_question"])), _VECTOR_TOP_K),
        )
        return {"vector_ids": [row[0] for row in cursor.fetchall()]}


def _search_memory(state: AskState) -> AskState:
    """Memory as its own ranked list, fused by rank rather than by raw distance.
    Derived text and a raw turn are not on one distance scale -- an event is a
    rewritten sentence, a message is what someone typed -- so merging them by cosine
    buried memory (episodes reached the pool for 5 of 40 real questions). Questions
    about a person also get that person's own events, which similarity handles badly."""
    if state.get("handled_as_memory") or not _episodes_on(state):
        return {"memory_ids": []}
    question = state["effective_question"]
    query_vector = to_vector_literal(embed(question))
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT embedding <=> %s::vector FROM messages
            WHERE user_id = %s AND is_bot = FALSE AND embedding IS NOT NULL
            ORDER BY 1 LIMIT 1
            """,
            (query_vector, state["chat_id"]),
        )
        row = cursor.fetchone()
    best_raw = float(row[0]) if row else 1.0
    ceiling = min(_MEMORY_MAX_DISTANCE, best_raw + _MEMORY_MARGIN_OVER_RAW)
    ranked = [event_id for event_id, distance in
              search_events(state["chat_id"], query_vector, _MEMORY_TOP_K) if distance <= ceiling]
    # A named participant is a direct signal, not a similarity guess, so their own
    # memory goes in front of the vector hits and is not subject to the ceiling.
    participant = resolve_question_participant(state["chat_id"], question, state.get("asker_name"))
    if participant:
        by_person = search_events_by_participant(state["chat_id"], participant, _MEMORY_PARTICIPANT_LIMIT)
        ranked = by_person + [event_id for event_id in ranked if event_id not in by_person]
    return {"memory_ids": ranked}


def _fuse_rrf(state: AskState) -> AskState:
    if state.get("handled_as_memory"):
        return {"candidate_ids": [], "candidate_event_ids": []}
    memory_ids = [_memory_key(event_id) for event_id in (state.get("memory_ids") or [])]
    fused = _rrf([state.get("fts_ids", []), state.get("vector_ids", []), memory_ids])[:10]
    # Raw retrieval is strong and memory is one list against two, so without a floor
    # it can be squeezed out of the pool entirely; the reranker still has to choose it.
    reserved = [key for key in memory_ids if key not in fused][:max(0, _RESERVED_MEMORY_SLOTS - sum(
        1 for key in fused if isinstance(key, str)))]
    fused = fused[:10 - len(reserved)] + reserved
    candidate_ids = [key for key in fused if isinstance(key, int)]
    candidate_event_ids = [int(key[1:]) for key in fused if isinstance(key, str)]
    if not fused:
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM messages WHERE user_id = %s AND is_bot = FALSE ORDER BY id DESC LIMIT 10",
                (state["chat_id"],),
            )
            candidate_ids = [row[0] for row in cursor.fetchall()]
    return {"candidate_ids": candidate_ids, "candidate_event_ids": candidate_event_ids}


def _after_fuse_skip_small(state: AskState) -> Literal["rerank", "skip_rerank"]:
    """Speed experiment: with <=2 candidates there's nothing meaningful to
    rerank -- skip the LLM call and use them as-is."""
    total = len(state.get("candidate_ids") or []) + len(state.get("candidate_event_ids") or [])
    return "skip_rerank" if total <= 2 else "rerank"


def _match_ids_from_candidates(state: AskState) -> AskState:
    return {"match_ids": set(state.get("candidate_ids") or []),
            "match_event_ids": list(state.get("candidate_event_ids") or [])}


def _rerank(state: AskState, config: RunnableConfig) -> AskState:
    """Raw messages and memory entries are reranked together; a kept memory entry
    later expands into the source messages it cites."""
    candidate_ids = state.get("candidate_ids", [])
    event_ids = state.get("candidate_event_ids") or []
    if not candidate_ids and not event_ids:
        return {"match_ids": set(), "match_event_ids": []}
    rows = []
    if candidate_ids:
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
    events = load_events(event_ids)
    match_ids = {row[0] for row in rows}
    match_event_ids = [event.event_id for event in events]
    if len(rows) + len(events) > 1:
        numbered = "\n".join(
            [f"[{row_id}] {resolve_display_name(username, user_name)}: {decrypt(text)}"
             for row_id, user_name, username, text in rows]
            + [f"[{_memory_key(event.event_id)}] {render_event(event)}" for event in events]
        )
        prompt = prompt_for_rerank.format(question=state["effective_question"], messages=numbered)
        reply = _content(get_chat_model("fast", 0).invoke(prompt, config=config))
        selected = []
        for prefix, number in re.findall(r"(E?)(\d+)", reply):
            key = (prefix, int(number))
            valid = int(number) in match_event_ids if prefix else int(number) in match_ids
            if valid and key not in selected:
                selected.append(key)
        selected = selected[:5]
        if selected:
            match_ids = {number for prefix, number in selected if not prefix}
            match_event_ids = [number for prefix, number in selected if prefix][:_MAX_SELECTED_MEMORY]
    return {"match_ids": match_ids, "match_event_ids": match_event_ids}


def _episode_source_row_ids(chat_id: int, events) -> list[int]:
    """Internal row ids of the messages a retrieved event cites. They become anchors
    for the same conversation-window expansion raw matches get: an event points at one
    short line, and without the exchange around it the generator loses what made that
    line meaningful. In the first paired run, expanding memory without a window
    collapsed one case's context from 338 messages to 1 and the answer got worse."""
    source_ids = list(dict.fromkeys(mid for event in events for mid in event.source_message_ids))
    if not source_ids:
        return []
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM messages WHERE user_id = %s AND is_bot = FALSE AND message_id = ANY(%s)",
            (chat_id, source_ids),
        )
        return [row[0] for row in cursor.fetchall()]


def _generate_answer(state: AskState, config: RunnableConfig) -> AskState:
    match_ids = set(state.get("match_ids") or set())
    episodes = load_events(state.get("match_event_ids") or [])
    memory_anchor_ids = set(_episode_source_row_ids(state["chat_id"], episodes))
    match_ids |= memory_anchor_ids
    if not match_ids:
        return {"answer": None, "window_rows": [], "episode_row_ids": []}
    with get_conn() as conn:
        cursor = conn.cursor()
        # A row is only allowed into the window if it's genuinely content
        # (is_bot=FALSE) OR it's one of the actual matches itself (a reply
        # can legitimately anchor on the bot's own prior answer) -- this
        # closes the self-pollution case where the bot's own EARLIER,
        # unrelated answer sat nearby and got cited as an independent
        # source about someone.
        #
        # The window itself is conversation_id equality when the anchor has
        # one (persisted at insert time via reply-chain inheritance / time-
        # gap continuation -- see _compute_conversation_id in
        # handlers/handlers.py), which only pulls in messages that are
        # actually part of the same exchange instead of whatever happens to
        # sit within +-3 message_ids of it. Anchors without a
        # conversation_id (pre-migration rows, or no known message_date)
        # fall back to that old +-3 window.
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
    episode_row_ids = [row[0] for row in rows if row[0] in memory_anchor_ids]
    if not rows:
        return {"answer": None, "window_rows": [], "episode_row_ids": []}
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
    episode_context = ""
    if episodes:
        episode_context = (
            "\nПамять чата — заранее извлечённые из переписки события с их историей изменений. "
            "Это подсказка, где и что искать, а не доказательство: исходные сообщения этих событий есть "
            "в пронумерованном списке ниже, отвечай и цитируй по ним.\n"
            + "\n".join(render_event(event) for event in episodes) + "\n"
        )
        group_context += episode_context
    prompt = prompt_for_qa.format(
        question=state["question"], messages="\n".join(lines), group_context=group_context, asker_name=state["asker_name"]
    )
    answer = _content(get_chat_model("primary", 0.3).invoke(prompt, config=config))
    return {
        "answer": _format_citations(answer, legend) if answer else None,
        "answer_plain": answer or None,
        "window_rows": rows,
        "episode_row_ids": episode_row_ids,
        "context_lines": lines,
        "episode_context": episode_context,
    }


def _save_bot_answer(state: AskState) -> AskState:
    bot = state["bot"]
    if state.get("handled_as_memory"):
        bot.send_message(state["chat_id"], state.get("memory_reply") or "Записал", message_thread_id=state.get("thread_id"))
        return {}
    if not state.get("answer"):
        bot.send_message(state["chat_id"], "В истории чата не нашёл ответа на этот вопрос.", message_thread_id=state.get("thread_id"))
        return {}
    sent = bot.send_message(state["chat_id"], state["answer"], parse_mode="HTML", message_thread_id=state.get("thread_id"))
    state["save_bot_answer"](
        state["chat_id"], sent.message_id, state.get("bot_username"), re.sub(r"\s*\[\d+\]", "", state.get("answer_plain") or ""),
        getattr(sent, "message_thread_id", state.get("thread_id")), getattr(sent, "date", None),
        state.get("replied_message_id"),
    )
    return {}


def build_ask_graph():
    graph = StateGraph(AskState)
    graph.add_node("resolve_anchor", _resolve_anchor)
    graph.add_node("classify_intent", _classify_intent)
    graph.add_node("rewrite_query", _rewrite_query)
    graph.add_node("search_fts", _search_fts)
    graph.add_node("search_vector", _search_vector)
    graph.add_node("search_memory", _search_memory)
    graph.add_node("fuse_rrf", _fuse_rrf)
    graph.add_node("rerank", _rerank)
    graph.add_node("generate_answer", _generate_answer)
    graph.add_node("save_bot_answer", _save_bot_answer)
    graph.add_edge(START, "resolve_anchor")
    graph.add_conditional_edges("resolve_anchor", _after_anchor)
    graph.add_edge("classify_intent", "rewrite_query")
    graph.add_edge("rewrite_query", "search_fts")
    graph.add_edge("rewrite_query", "search_vector")
    graph.add_edge("rewrite_query", "search_memory")
    graph.add_edge("search_fts", "fuse_rrf")
    graph.add_edge("search_vector", "fuse_rrf")
    graph.add_edge("search_memory", "fuse_rrf")
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
    graph.add_node("search_memory", _search_memory)
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
    graph.add_edge("classify_and_rewrite", "search_memory")
    graph.add_edge("search_fts", "fuse_rrf")
    graph.add_edge("search_vector", "fuse_rrf")
    graph.add_edge("search_memory", "fuse_rrf")
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
    graph.add_node("search_memory", _search_memory)
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
    graph.add_edge("classify_and_rewrite", "search_memory")
    graph.add_edge("search_fts", "fuse_rrf")
    graph.add_edge("search_vector", "fuse_rrf")
    graph.add_edge("search_memory", "fuse_rrf")
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
    extraction_lines: list[str]
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
    extraction_source = state.get("extraction_lines") or state["lines"]
    return {"messages": [HumanMessage(content=prompt_for_context_extraction.format(messages="\n".join(extraction_source)))]}


def _save_summary_state(state: SummaryState) -> SummaryState:
    state["save_summary_state"](state["newest_included_id"], re.sub(r"\s*\[\d+\]", "", state["answer"] or ""))
    return {}


def _send_summary_failure(state: SummaryState) -> SummaryState:
    state["bot"].send_message(state["chat_id"], "LLM решил послать вас с ответом", message_thread_id=state.get("thread_id"))
    return {}


def build_summary_graph(chat_id: int):
    # Changeable state of people (formerly the upsert_state tool writing
    # memory_facts) is now built by the query-independent episodic memory
    # worker (episodic_memory.py), not by this summary pass.
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
    graph = build_summary_graph(state["chat_id"])
    return graph.invoke(state, config=_config())
