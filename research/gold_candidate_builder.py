"""Fully read-only gold-dataset candidate builder.

Per explicit instruction: gold must NOT be built from production extraction
(upsert_state/free-choice tool-calling) -- that IS the thing under test, so
using it to generate its own ground truth is circular evaluation. Two
INDEPENDENT discovery paths instead:

  A. Existing chat_moments (644 of them) -> hybrid FTS+vector retrieval of
     supporting raw messages -> LLM verifies which candidates actually
     support the claim, and classifies support level.
  B. Raw history chunked into 30-50 message windows, scanned independently
     with a recall-oriented prompt across a WIDE category set (state,
     durable_person_fact, relationship, commitment, past_event, opinion,
     running_gag, group_lore) -- doesn't touch chat_moments at all, so it
     can surface facts moments never captured.

No writes anywhere -- memory_facts, chat_context, chat_moments are never
touched. Outputs a deduplicated Markdown review sheet for a human to
accept/edit/reject. No retrieval_cues or eval questions generated here --
just the claim and its minimal source set, per the brief.

Usage: python3 -m research.gold_candidate_builder
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import config  # noqa: triggers dotenv load
from crypto_utils import decrypt
from database.db import get_conn
from display_names import resolve_display_name
from embeddings import embed, to_vector_literal
from llm.graphs import _or_query, _rrf
from llm.groq_client import get_chat_model

CHAT_ID = -1002335227490
RAW_CHUNK_SIZE = 40
MAX_WORKERS = 5
KINDS = {
    "state", "durable_person_fact", "relationship", "commitment",
    "past_event", "opinion", "running_gag", "group_lore", "other",
}


# ---------------------------------------------------------------- shared --

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


def _local_context(chat_id, message_id, span=3):
    """A few messages around message_id, for the review sheet's benefit --
    not used for anything else, purely so a human reviewer doesn't have to
    dig through the DB to judge a candidate."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT message_id, user_name, username, message, is_bot
            FROM messages WHERE user_id = %s
              AND message_id BETWEEN %s AND %s
            ORDER BY message_id ASC
            """,
            (chat_id, message_id - span, message_id + span),
        )
        rows = cursor.fetchall()
    out = []
    for mid, name, uname, msg, is_bot in rows:
        try:
            text = decrypt(msg)
        except Exception:
            continue
        author = "БОТ" if is_bot else resolve_display_name(uname, name)
        out.append(f"[{mid}] {author}: {text}")
    return out


def _real_rows(chat_id, message_ids):
    if not message_ids:
        return []
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT message_id, user_name, username, message, message_date, is_bot "
            "FROM messages WHERE user_id = %s AND message_id = ANY(%s) ORDER BY message_id ASC",
            (chat_id, list(set(message_ids))),
        )
        rows = cursor.fetchall()
    out = []
    for mid, name, uname, msg, date, is_bot in rows:
        try:
            text = decrypt(msg)
        except Exception:
            continue
        author = "БОТ" if is_bot else resolve_display_name(uname, name)
        out.append({"message_id": mid, "author": author, "text": text, "date": date})
    return out


# ---------------------------------------------------------- path A: moments --

def _hybrid_candidates(chat_id, query_text, query_embedding, top_k=15):
    with get_conn() as conn:
        cursor = conn.cursor()
        fts_ids = []
        query = _or_query(cursor, query_text)
        if query:
            cursor.execute(
                """
                SELECT id, message_id FROM messages
                WHERE user_id = %s AND is_bot = FALSE AND search_vector @@ to_tsquery('russian', %s)
                ORDER BY ts_rank(search_vector, to_tsquery('russian', %s)) DESC LIMIT 15
                """,
                (chat_id, query, query),
            )
            fts_rows = cursor.fetchall()
        else:
            fts_rows = []
        cursor.execute(
            """
            SELECT id, message_id FROM messages
            WHERE user_id = %s AND is_bot = FALSE AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector LIMIT 15
            """,
            (chat_id, to_vector_literal(query_embedding)),
        )
        vector_rows = cursor.fetchall()

    by_internal_id = {row[0]: row[1] for row in fts_rows + vector_rows}
    fused = _rrf([[r[0] for r in fts_rows], [r[0] for r in vector_rows]])[:top_k]
    return [by_internal_id[i] for i in fused if i in by_internal_id]


def _verify_moment_prompt(claim, candidates):
    blocks = "\n".join(f"[msg_id={c['message_id']}, {c['date']}] {c['author']}: {c['text']}" for c in candidates)
    return f"""Ты проверяешь факт, ранее извлечённый из истории чата, на предмет того, насколько он подтверждён конкретными сообщениями.

Факт: "{claim}"

Кандидаты сообщений (могут подтверждать факт, а могут быть шумом по случайному совпадению слов):
{blocks}

Определи:
1. subject — о ком этот факт (имя человека, ровно как оно упомянуто в сообщениях; null, если факт не про конкретного человека, а про группу целиком).
2. kind — одно из: state, durable_person_fact, relationship, commitment, past_event, opinion, running_gag, group_lore, other.
3. minimal_source_message_ids — МИНИМАЛЬНЫЙ набор msg_id, которые реально подтверждают факт (не все кандидаты подряд, только релевантные).
4. support — одно из: fully_supported (полностью подтверждён), partially_supported (частично, требует домысливания), unsupported (кандидаты не подтверждают факт).
5. future_use_reason — одним предложением, зачем этот факт может понадобиться в будущем вопросе.

Верни ТОЛЬКО JSON: {{"subject": "...", "kind": "...", "minimal_source_message_ids": [123, 456], "support": "...", "future_use_reason": "..."}}"""


def _process_moment(chat_id, moment_id, claim, moment_embedding):
    candidate_ids = _hybrid_candidates(chat_id, claim, moment_embedding)
    if not candidate_ids:
        return None
    candidates = _real_rows(chat_id, candidate_ids)
    if not candidates:
        return None
    result = _judge_json(_verify_moment_prompt(claim, candidates))
    if not result or result.get("support") == "unsupported":
        return None
    sources = [mid for mid in (result.get("minimal_source_message_ids") or []) if mid in {c["message_id"] for c in candidates}]
    if not sources:
        return None
    return {
        "path": "A-moments", "moment_id": moment_id,
        "claim": claim, "kind": result.get("kind") or "other",
        "subject": result.get("subject"),
        "source_message_ids": sources,
        "support": result.get("support"),
        "future_use_reason": result.get("future_use_reason", ""),
    }


def discover_path_a(chat_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, note, embedding FROM chat_moments WHERE chat_id = %s", (chat_id,))
        rows = cursor.fetchall()
    print(f"Path A: {len(rows)} existing moments to verify against raw retrieval", file=sys.stderr)

    moments = []
    for moment_id, note, embedding in rows:
        try:
            claim = decrypt(note)
        except Exception:
            continue
        vector = [float(x) for x in embedding.strip("[]").split(",")] if isinstance(embedding, str) else list(embedding)
        moments.append((moment_id, claim, vector))

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process_moment, chat_id, mid, claim, emb): mid for mid, claim, emb in moments}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 50 == 0:
                print(f"  Path A progress: {done}/{len(moments)}", file=sys.stderr)
            try:
                r = future.result()
            except Exception as e:
                print(f"  Path A moment {futures[future]} failed: {e}", file=sys.stderr)
                continue
            if r:
                results.append(r)
    print(f"Path A: {len(results)} supported candidates out of {len(moments)} moments", file=sys.stderr)
    return results


# ------------------------------------------------------- path B: raw chunks --

def _recall_prompt(lines):
    return f"""Вот отрезок реальной переписки группового чата, сообщения занумерованы реальными message_id.

Твоя задача — найти факты, которые ДЕЙСТВИТЕЛЬНО стоит запомнить надолго: такие, что если через недели или месяцы кто-то спросит про них в чате, ответ был бы полезен. Это НЕ пересказ каждого сообщения подряд.

Строго ИСКЛЮЧАЙ:
- одноразовые технические реплики без долгосрочной ценности ("система использует Postgres", "фича теперь работает") — если это не устойчивый факт о самом проекте/группе, а просто рабочий комментарий по ходу дела;
- короткие реакции без содержания ("ахаха", "согласен", "хз");
- бессмысленный шум ("+1", стикеры, эмодзи без контекста);
- пересказ КАЖДОГО сообщения — это не цель, цель это отобрать по-настоящему ценное.

Включай только то, что похоже на настоящий факт про КОНКРЕТНОГО человека (устойчивая черта, локация, работа/учёба, отношения, обещание, важное прошлое событие, устойчивое мнение) или про ГРУППУ целиком (устойчивая внутренняя шутка/мем, который повторяется, общий факт про группу). Одно сообщение с проходной шуткой — это НЕ running_gag, только если видно что шутка/паттерн повторяется или явно станет отсылкой.

Категории (kind): state, durable_person_fact, relationship, commitment, past_event, opinion, running_gag, group_lore, other.

Для каждой находки: claim (сам факт одним предложением, ПО-РУССКИ, как самостоятельное утверждение, а не цитата), kind, subject (имя человека, как оно упомянуто в переписке, или null для group_lore), source_message_ids (реальные message_id, минимум один).

В этом отрезке из {len(lines)} сообщений обычно НЕТ более 3-5 действительно стоящих находок, часто 0. Не старайся заполнить список любой ценой.

Сообщения:
{chr(10).join(lines)}

Верни ТОЛЬКО JSON: {{"items": [{{"claim": "...", "kind": "...", "subject": "...", "source_message_ids": [123]}}]}} (пустой список items, если по-настоящему ценного ничего нет — это нормальный и частый результат)."""


def _process_chunk(chat_id, chunk_no, rows):
    lines = []
    valid_ids = set()
    for row_id, msg_id, user_name, username, message, is_bot in rows:
        try:
            text = decrypt(message)
        except Exception:
            continue
        author = "БОТ" if is_bot else resolve_display_name(username, user_name)
        lines.append(f"[message_id={msg_id}] {author}: {text}")
        if msg_id is not None:
            valid_ids.add(msg_id)

    result = _judge_json(_recall_prompt(lines))
    if not result or not result.get("items"):
        return []

    out = []
    for item in result["items"]:
        kind = item.get("kind") if item.get("kind") in KINDS else "other"
        sources = [mid for mid in (item.get("source_message_ids") or []) if mid in valid_ids]
        claim = (item.get("claim") or "").strip()
        if not claim or not sources:
            continue
        out.append({
            "path": "B-raw-chunk", "chunk": chunk_no,
            "claim": claim, "kind": kind, "subject": item.get("subject"),
            "source_message_ids": sources,
            "support": "fully_supported",  # model cited its own source at generation time
            "future_use_reason": "",
        })
    return out


def discover_path_b(chat_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, message_id, user_name, username, message, is_bot "
            "FROM messages WHERE user_id = %s ORDER BY id ASC",
            (chat_id,),
        )
        all_rows = cursor.fetchall()
    chunks = [all_rows[i:i + RAW_CHUNK_SIZE] for i in range(0, len(all_rows), RAW_CHUNK_SIZE)]
    print(f"Path B: {len(all_rows)} messages, {len(chunks)} chunks of ~{RAW_CHUNK_SIZE}", file=sys.stderr)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process_chunk, chat_id, i, chunk): i for i, chunk in enumerate(chunks)}
        done = 0
        for future in as_completed(futures):
            done += 1
            print(f"  Path B progress: {done}/{len(chunks)}", file=sys.stderr)
            try:
                items = future.result()
            except Exception as e:
                print(f"  Path B chunk {futures[future]} failed: {e}", file=sys.stderr)
                continue
            results.extend(items)
    print(f"Path B: {len(results)} raw candidates from {len(chunks)} chunks", file=sys.stderr)
    return results


# --------------------------------------------------------- dedup + select --

def _dedupe(candidates):
    """Semantic dedup on the claim text -- moments (path A) and raw-chunk
    scanning (path B) can independently surface the same real fact."""
    if not candidates:
        return []
    embeddings = [embed(c["claim"]) for c in candidates]
    kept = []
    kept_embeddings = []
    for cand, emb in zip(candidates, embeddings):
        is_dup = False
        for kept_emb in kept_embeddings:
            dot = sum(a * b for a, b in zip(emb, kept_emb))
            na = sum(a * a for a in emb) ** 0.5
            nb = sum(b * b for b in kept_emb) ** 0.5
            if na and nb and dot / (na * nb) > 0.92:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
            kept_embeddings.append(emb)
    return kept


def _select_best(candidates, target=70, max_per_subject=5):
    """Below target, there's no real competition for slots -- capping per
    subject there would just throw away good candidates for no benefit, so
    the cap only kicks in once the pool actually needs trimming."""
    support_rank = {"fully_supported": 0, "partially_supported": 1, "unsupported": 2}
    candidates = sorted(candidates, key=lambda c: support_rank.get(c.get("support"), 1))
    if len(candidates) <= target:
        return candidates
    selected = []
    per_subject = {}
    for c in candidates:
        subj = (c.get("subject") or "").strip().lower() or "_none_"
        if per_subject.get(subj, 0) >= max_per_subject:
            continue
        selected.append(c)
        per_subject[subj] = per_subject.get(subj, 0) + 1
        if len(selected) >= target:
            break
    return selected


# --------------------------------------------------------------- output --

def _render_markdown(candidates, chat_id):
    lines = ["# Gold candidate review sheet", "", f"Всего кандидатов на проверку: {len(candidates)}", ""]
    for i, c in enumerate(candidates, start=1):
        lines.append(f"### {i}.")
        lines.append(f"Claim: {c['claim']}")
        lines.append(f"Kind: {c['kind']}")
        lines.append(f"Subject: {c.get('subject') or '(группа/не указано)'}")
        lines.append(f"Discovery path: {c['path']}")
        lines.append("Sources:")
        for row in _real_rows(chat_id, c["source_message_ids"]):
            lines.append(f"- [{row['message_id']}] {row['author']}: {row['text'][:200]}")
        lines.append(f"Suggested support: {c.get('support')}")
        if c.get("future_use_reason"):
            lines.append(f"Зачем может пригодиться: {c['future_use_reason']}")
        lines.append("")
        lines.append("<details><summary>Локальный контекст вокруг первого источника</summary>")
        lines.append("")
        lines.append("```")
        for ctx_line in _local_context(chat_id, c["source_message_ids"][0]):
            lines.append(ctx_line)
        lines.append("```")
        lines.append("</details>")
        lines.append("")
        lines.append("Decision:")
        lines.append("- [ ] accept")
        lines.append("- [ ] edit")
        lines.append("- [ ] reject")
        lines.append("- [ ] ambiguous attribution")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def run():
    a = discover_path_a(CHAT_ID)
    b = discover_path_b(CHAT_ID)
    all_candidates = a + b
    print(f"\nTotal raw candidates before dedup: {len(all_candidates)}", file=sys.stderr)

    deduped = _dedupe(all_candidates)
    print(f"After dedup: {len(deduped)}", file=sys.stderr)

    selected = _select_best(deduped, target=70)
    print(f"Selected for review sheet: {len(selected)}", file=sys.stderr)

    with open("/tmp/gold_candidates_raw.json", "w") as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2, default=str)

    md = _render_markdown(selected, CHAT_ID)
    out_path = "/tmp/gold_review_sheet.md"
    with open(out_path, "w") as f:
        f.write(md)
    print(f"\nReview sheet written to {out_path} ({len(selected)} candidates)", file=sys.stderr)


if __name__ == "__main__":
    run()
