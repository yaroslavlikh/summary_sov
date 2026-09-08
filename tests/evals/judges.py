"""LLM-judge evaluators for the /ask real-invocation benchmark.

Each judge gets real dates and author names in its context block (never
just bare message text) -- date/author fidelity was an explicit
requirement, not an afterthought, given how many incidents this session
traced back to exactly that being ignored.
"""
import json
import re

from crypto_utils import decrypt
from database.db import get_conn
from display_names import resolve_display_name
from llm.groq_client import get_chat_model, tracing_config


def _fmt_date(message_date):
    return f"дата неизвестна (старая запись до добавления message_date)" if not message_date else str(message_date)


def _context_window(chat_id, message_id, span=10):
    """+-span messages around message_id, WITH real author names and dates --
    the raw material every judge below reasons over."""
    if not message_id:
        return []
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT message_id, user_name, username, message, message_date, is_bot
            FROM messages
            WHERE user_id = %s AND message_id BETWEEN %s AND %s
            ORDER BY message_id ASC
            """,
            (chat_id, message_id - span, message_id + span),
        )
        rows = cursor.fetchall()
    out = []
    for msg_id, user_name, username, message, message_date, is_bot in rows:
        try:
            text = decrypt(message)
        except Exception:
            continue
        author = "БОТ (свой прошлый ответ)" if is_bot else resolve_display_name(username, user_name)
        out.append(f"[msg_id={msg_id}, {_fmt_date(message_date)}] {author}: {text}")
    return out


def _judge_llm(prompt):
    content = get_chat_model("fast", 0).invoke(prompt, config=tracing_config()).content
    content = content if isinstance(content, str) else ""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def agent_goal_evaluator(*, input, output, expected_output=None, **kwargs):
    """0.0-1.0 step 0.1: correctness + completeness of the final answer,
    judged against a wide +-10 window around every cited message (with real
    dates/authors), not just the bare answer text."""
    answer = output.get("answer") or "(нет ответа)"
    windows = "\n\n".join(
        "\n".join(_context_window(input["chat_id"], mid))
        for mid in (output.get("cited_message_ids") or [])[:5]
    ) or "(нет цитат — контекст берётся из окна кандидатов)"
    if not windows.strip("() нет—цитатконтекстберётсяизокнакандидатов"):
        windows = "\n".join(_context_window(input["chat_id"], output.get("first_candidate_message_id")))

    prompt = f"""Вопрос от {input['asker_name']}: {input['question']}

Ответ бота: {answer}

Реальный контекст переписки вокруг процитированных сообщений (с датами и именами авторов):
{windows}

Оцени ответ бота от 0.0 до 1.0 с шагом 0.1: правильность и полнота, с учётом того,
кто реально что сказал и когда (не приписывай слова не тому автору, не игнорируй,
если ответ явно устарел по дате). Верни ТОЛЬКО JSON: {{"score": 0.0-1.0, "comment": "почему"}}"""
    result = _judge_llm(prompt)
    if not result:
        return {"name": "agent_goal", "value": 0.0, "comment": "judge не вернул валидный JSON"}
    return {"name": "agent_goal", "value": float(result.get("score", 0.0)), "comment": result.get("comment", "")}


def search_precision_evaluator(*, input, output, expected_output=None, **kwargs):
    """Precision of the RAW candidate pool (fts_ids + vector_ids, before
    rerank) -- of what search found, how much is genuinely relevant/current
    given +-10 context and real dates around each candidate?"""
    candidate_ids = output.get("candidate_message_ids") or []
    if not candidate_ids:
        return []  # anchor case: no candidate pool to score, nothing to report

    blocks = []
    for mid in candidate_ids[:10]:
        window = _context_window(input["chat_id"], mid)
        blocks.append(f"--- Кандидат msg_id={mid} ---\n" + "\n".join(window))

    prompt = f"""Вопрос от {input['asker_name']}: {input['question']}

Ниже — кандидаты, которые поиск счёл релевантными, каждый с окном +-10 сообщений
вокруг (даты и авторы реальные). Оцени по каждому кандидату: он ДЕЙСТВИТЕЛЬНО
релевантен и актуален (не устарел по дате, не про другого человека/тему), или это
шум по случайному совпадению слов?

{chr(10).join(blocks)}

Верни ТОЛЬКО JSON: {{"relevant_count": N, "total": {len(candidate_ids[:10])}, "comment": "кратко почему"}}"""
    result = _judge_llm(prompt)
    if not result or not result.get("total"):
        return {"name": "search_precision", "value": 0.0, "comment": "judge не вернул валидный JSON"}
    precision = result.get("relevant_count", 0) / result["total"]
    return {"name": "search_precision", "value": round(precision, 2), "comment": result.get("comment", "")}


def rerank_recall_evaluator(*, input, output, expected_output=None, **kwargs):
    """Of the candidates search_precision judged relevant, what fraction did
    rerank actually KEEP into match_ids (not drop)?"""
    candidate_ids = set(output.get("candidate_message_ids") or [])
    match_ids = set(output.get("match_message_ids") or [])
    if not candidate_ids:
        return []  # anchor case: no candidate pool, nothing to score

    blocks = []
    for mid in list(candidate_ids)[:10]:
        window = _context_window(input["chat_id"], mid)
        blocks.append(f"--- msg_id={mid} (взят rerank'ом: {mid in match_ids}) ---\n" + "\n".join(window))

    prompt = f"""Вопрос от {input['asker_name']}: {input['question']}

Ниже кандидаты ДО reranking'а, с пометкой, оставил ли их rerank в финальном
наборе. Для каждого реши: он ДЕЙСТВИТЕЛЬНО был релевантен вопросу?

{chr(10).join(blocks)}

Верни ТОЛЬКО JSON: {{"truly_relevant_ids": [msg_id, ...], "kept_by_rerank_ids": [msg_id, ...], "comment": "..."}}"""
    result = _judge_llm(prompt)
    if not result:
        return {"name": "rerank_recall", "value": 0.0, "comment": "judge не вернул валидный JSON"}
    truly_relevant = set(result.get("truly_relevant_ids") or [])
    if not truly_relevant:
        return []  # judge found nothing genuinely relevant to check recall against
    kept_relevant = truly_relevant & match_ids
    recall = len(kept_relevant) / len(truly_relevant)
    return {"name": "rerank_recall", "value": round(recall, 2), "comment": result.get("comment", "")}


def citation_hit_rate_evaluator(*, input, output, expected_output=None, **kwargs):
    """Of the message_ids actually CITED in the final answer, what fraction
    point to genuinely correct evidentiary sources (not window-contamination
    noise, not the bot's own unrelated prior answer)?"""
    cited_ids = output.get("cited_message_ids") or []
    if not cited_ids:
        return []  # no citations in the answer, nothing to score

    blocks = []
    for mid in cited_ids:
        window = _context_window(input["chat_id"], mid)
        blocks.append(f"--- Процитировано: msg_id={mid} ---\n" + "\n".join(window))

    prompt = f"""Вопрос от {input['asker_name']}: {input['question']}
Ответ бота: {output.get('answer')}

Бот процитировал следующие сообщения как источники (окно +-10 вокруг каждого,
с реальными датами/авторами). Для каждой цитаты реши: это ДЕЙСТВИТЕЛЬНО то
сообщение, на котором основан ответ, или мимо (постороннее, взято по ошибке
из окна вокруг anchor'а, или это собственный прошлый ответ бота, выданный за
независимый источник)?

{chr(10).join(blocks)}

Верни ТОЛЬКО JSON: {{"correct_citations": N, "total": {len(cited_ids)}, "comment": "..."}}"""
    result = _judge_llm(prompt)
    if not result or not result.get("total"):
        return {"name": "citation_hit_rate", "value": 0.0, "comment": "judge не вернул валидный JSON"}
    hit_rate = result.get("correct_citations", 0) / result["total"]
    return {"name": "citation_hit_rate", "value": round(hit_rate, 2), "comment": result.get("comment", "")}


ALL_EVALUATORS = [
    agent_goal_evaluator,
    search_precision_evaluator,
    rerank_recall_evaluator,
    citation_hit_rate_evaluator,
]
