"""Drafts one memory-dependent question per frozen gold positive fact --
step 2 of the plan: (1) oracle retrieval [done], (2) this, (3) paired
baseline vs oracle /ask, (4) no-regression check.

A "memory-dependent" question here means: a natural question a chat member
might plausibly ask, whose correct answer requires knowing the specific
gold fact. One question per ALL 44 positive facts, not a pre-filtered
subset -- picking ahead of time which facts are "truly memory-dependent"
based on a guess about what baseline retrieval struggles with would be
selection bias. The paired eval (step 3) is what actually reveals, per
question, whether oracle memory helps -- some will show no difference
(baseline's own message-window retrieval already finds the source, as the
oracle_memory_eval.py smoke test found for the Ксюша question), and that's
real, useful signal, not something to filter out beforehand.

Deliberately instructs the model to avoid copying the claim's own wording
-- a question that trivially reuses the claim's vocabulary would make FTS/
vector retrieval over raw messages artificially easy, undermining the
whole point of testing whether MEMORY (not just better message search)
helps.

This is a DRAFT for human review, like every other generated artifact this
research effort has produced -- not something to feed into the paired eval
until reviewed. Written to /tmp (gitignored pattern), not committed.

Usage: python3 -m research.build_memory_questions
"""
import json
import re
import sys

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import config  # noqa: triggers dotenv load
from llm.groq_client import get_chat_model

GOLD_PATH = "/tmp/gold_facts.jsonl"
OUT_PATH = "/tmp/memory_questions_draft.jsonl"
GROUP_SIZE = 15


def _judge_json(prompt):
    content = get_chat_model("fast", 0).invoke(prompt).content
    content = content if isinstance(content, str) else ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _prompt(facts):
    blocks = "\n".join(f"{f['id']}. [{f['kind']}, субъект: {f['subject'] or '(группа)'}] {f['claim']}" for f in facts)
    return f"""Вот список известных фактов из группового чата (реальные, проверенные людьми). Для КАЖДОГО факта придумай ОДИН естественный вопрос, который мог бы задать участник этого же чата, и ответ на который требует знания именно этого факта.

ВАЖНО: НЕ копируй формулировку факта в вопрос — другие слова/синонимы, разговорно, коротко.

КРИТИЧЕСКИ ВАЖНО — если субъект факта известен (не "(группа)"), НАЗЫВАЙ его по имени в вопросе и спрашивай про СОДЕРЖАНИЕ факта (мнение/состояние/событие/отношение), а НЕ переспрашивай "кто...?". Вопрос вида "Кто сделал X?" уместен ТОЛЬКО когда субъект факта — "(группа)" (не назван конкретный человек). Это правило нарушать нельзя.

Примеры ПРАВИЛЬНОГО стиля (субъект известен — называем его, спрашиваем про содержание):
- факт "Ярик считает Крым частью России" -> "Что Ярик думает про Крым?"
- факт "Ярик пользуется VPN Sota" -> "Ярик всё ещё сидит на впне?"
- факт "Тигмен является армянином" -> "Тигмен правда армянин?"
- факт "У Вани есть кент, который встречался с Ксюшей" -> "У Вани есть друг, который был с Ксюшей?"

Примеры НЕПРАВИЛЬНОГО стиля (запрещено, если субъект известен):
- "Кто считает Крым частью России?" -- НЕТ, субъект уже известен (Ярик), спроси про содержание, а не переспрашивай личность.
- "Кто пользуется VPN?" -- НЕТ, то же самое.

"Кто...?" разрешён только когда субъект = "(группа)", например факт "участники обсуждают отметку на лекции по QR" -> "Как вообще отмечаться на лекции?" (или "кто-то помнит как?").

ВАЖНО: вопрос должен быть самодостаточным (не "а он?", а "Ярик всё ещё пользуется впном?").

Факты:
{blocks}

Верни ТОЛЬКО JSON: {{"questions": [{{"id": 1, "question": "..."}}, ...]}} — по одному вопросу на КАЖДЫЙ факт из списка, id должны совпадать с номерами фактов выше."""


def build():
    gold = [json.loads(line) for line in open(GOLD_PATH)]
    positive = [g for g in gold if g["label"] == "positive"]
    print(f"{len(positive)} positive facts to draft questions for", file=sys.stderr)

    groups = [positive[i:i + GROUP_SIZE] for i in range(0, len(positive), GROUP_SIZE)]
    by_id = {}
    for group in groups:
        result = _judge_json(_prompt(group))
        if not result or not result.get("questions"):
            print(f"WARNING: group starting at #{group[0]['id']} produced no questions", file=sys.stderr)
            continue
        for item in result["questions"]:
            qid = item.get("id")
            question = (item.get("question") or "").strip()
            if isinstance(qid, int) and question:
                by_id[qid] = question

    missing = [f["id"] for f in positive if f["id"] not in by_id]
    if missing:
        print(f"WARNING: no question drafted for ids {missing}", file=sys.stderr)

    rows = []
    for f in positive:
        rows.append({
            "id": f["id"],
            "question": by_id.get(f["id"]),
            "gold_claim": f["claim"],
            "kind": f["kind"],
            "subject": f["subject"],
            "source_message_ids": f["source_message_ids"],
        })

    with open(OUT_PATH, "w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} draft questions to {OUT_PATH} ({len(missing)} missing)", file=sys.stderr)
    return rows


if __name__ == "__main__":
    build()
