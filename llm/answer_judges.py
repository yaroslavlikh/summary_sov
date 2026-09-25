"""LLM judges of answer quality, shared by the offline benchmark and by online
sampling in production.

Two judges live here:

* `judge_answer` -- a rubric over the answer and the EXACT context the generator
  saw (numbered messages plus the episode block), returning five separate scores:
  faithfulness, answer relevance, attribution, temporal correctness and context
  sufficiency. Context sufficiency is the closest thing to the paper's evidence
  delivery metric that this chat allows: there are no gold source annotations
  here, so "was the support delivered" has to be judged, not computed.
* `judge_pairwise` -- blind A/B of two answers to the same question over the
  union of both contexts, run in BOTH orders; only an agreeing pair counts, so a
  position-biased judge produces a tie instead of a fake winner.

Both are judged by one model (JUDGE_MODEL_KIND, "primary" by default), which
also writes the answers -- a self-preference risk that applies equally to both
conditions being compared, and the reason a judged difference is evidence about
this pipeline, not a general claim.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from llm.groq_client import get_chat_model, tracing_config

RUBRIC_SCORES = ("faithfulness", "answer_relevance", "attribution", "temporal_correctness", "context_sufficiency")


def _judge_model():
    return get_chat_model(os.getenv("JUDGE_MODEL_KIND", "primary"), 0)


def _ask_judge(prompt: str) -> dict:
    message = _judge_model().invoke(prompt, config=tracing_config())
    content = message.content if isinstance(message.content, str) else ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _score(raw, name: str) -> Optional[float]:
    value = raw.get(name)
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "n/a", "na", "none"}):
        return None  # the judge says this dimension does not apply -- not a zero
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def judge_answer(question: str, asker_name: str, answer: Optional[str], context_lines: list[str],
                 episode_context: str = "") -> dict[str, dict]:
    """Rubric over the answer and the context it was actually generated from.
    Returns {score_name: {"value": float, "comment": str}} -- a dimension the judge
    marks inapplicable (no attribution in the answer, nothing temporal in the
    context) is left out instead of being scored zero."""
    if not (context_lines or episode_context):
        return {}
    prompt = f"""Ты — строгий оценщик ответов бота группового чата. Тебе дан вопрос, ответ бота и
РОВНО ТОТ контекст, который бот видел при генерации: пронумерованные сообщения переписки
(формат "[N] Автор: текст") и, возможно, блок эпизодов памяти — заранее извлечённых событий.

Вопрос (задаёт {asker_name}): {question}

Ответ бота: {answer or "(бот не ответил)"}

Контекст бота:
{chr(10).join(context_lines)}
{episode_context}

Оцени по пяти независимым шкалам от 0.0 до 1.0 (шаг 0.1). Если шкала к этому случаю
неприменима — верни для неё null, а не 0.
- faithfulness: каждое утверждение ответа следует из контекста. Логическая связь соседних
  сообщений — это НЕ выдумка; выдумка — факт, которого в контексте нет вообще. Честное
  "в истории чата не нашёл ответа" при контексте без ответа — это 1.0.
- answer_relevance: ответ отвечает именно на заданный вопрос, а не на соседний.
- attribution: слова, мнения и действия приписаны тем людям, которые их реально произнесли
  (null, если в ответе никому ничего не приписывается).
- temporal_correctness: если в контексте состояние менялось во времени, ответ отражает
  актуальное состояние или явно проговаривает изменение (null, если во времени ничего не менялось).
- context_sufficiency: хватало ли доставленного контекста, чтобы ответить на вопрос по существу.
  Это оценка ПОИСКА, а не ответа: если контекст нерелевантен вопросу — низкая оценка, даже
  если бот корректно отказался отвечать.

Верни ТОЛЬКО JSON:
{{"faithfulness": 0.0, "answer_relevance": 0.0, "attribution": 0.0, "temporal_correctness": 0.0,
"context_sufficiency": 0.0, "comment": "одно-два предложения, что решило оценку"}}"""
    raw = _ask_judge(prompt)
    comment = str(raw.get("comment") or "")[:500]
    return {name: {"value": value, "comment": comment}
            for name in RUBRIC_SCORES if (value := _score(raw, name)) is not None}


def _pairwise_once(question: str, asker_name: str, first: str, second: str, context: str) -> Optional[str]:
    prompt = f"""Ты — строгий оценщик. Два бота ответили на один вопрос по одной и той же переписке.

Вопрос (задаёт {asker_name}): {question}

Ответ A: {first or "(нет ответа)"}

Ответ B: {second or "(нет ответа)"}

Сообщения переписки, доступные обоим (объединение их контекстов):
{context}

Какой ответ лучше по совокупности: правильность по переписке, полнота, отсутствие выдумок,
прямота (без уклончивости) и корректность ссылок [N] на сообщения? Если разница несущественная —
честно верни "tie", не выдумывай победителя.

Верни ТОЛЬКО JSON: {{"winner": "A|B|tie", "reason": "одно предложение"}}"""
    winner = str(_ask_judge(prompt).get("winner") or "").strip().upper()
    return winner if winner in {"A", "B", "TIE"} else None


def judge_pairwise(question: str, asker_name: str, answer_a: Optional[str], answer_b: Optional[str],
                   context_lines: list[str]) -> dict:
    """Blind A/B in both orders. Result: +1 if A wins both orders, -1 if B does,
    0 on a tie or on disagreement between the orders (position bias)."""
    context = "\n".join(context_lines)
    forward = _pairwise_once(question, asker_name, answer_a or "", answer_b or "", context)
    reverse = _pairwise_once(question, asker_name, answer_b or "", answer_a or "", context)
    if forward is None or reverse is None:
        return {}
    if forward == "A" and reverse == "B":
        value, comment = 1.0, "A лучше в обоих порядках"
    elif forward == "B" and reverse == "A":
        value, comment = -1.0, "B лучше в обоих порядках"
    elif forward == reverse == "TIE":
        value, comment = 0.0, "ничья в обоих порядках"
    else:
        value, comment = 0.0, f"судья непоследователен между порядками ({forward}/{reverse}) — засчитано как ничья"
    return {"value": value, "comment": comment, "forward": forward, "reverse": reverse}
