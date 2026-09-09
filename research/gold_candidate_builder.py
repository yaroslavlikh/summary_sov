"""Fully read-only gold-dataset candidate builder.

Per explicit instruction: gold must NOT be built from production extraction
(upsert_state/free-choice tool-calling) -- that IS the thing under test, so
using it to generate its own ground truth is circular evaluation. Two
INDEPENDENT discovery paths instead:

  A. Existing chat_moments (644 of them) -> hybrid FTS+vector retrieval of
     supporting raw messages -> verified against that evidence.
  B. Raw history chunked into ~40-message windows, scanned independently
     with a recall-oriented prompt across a WIDE category set (state,
     durable_person_fact, relationship, commitment, past_event, opinion,
     running_gag, group_lore) -- doesn't touch chat_moments at all, so it
     can surface facts moments never captured.

v2, after a human review of v1's 76-candidate output found real structural
bugs (see docs/CHANGELOG.md 2026-09-09 entries for the full list) -- fixed
here:

  - Bot messages can never be cited as evidence for a claim about a human
    (was: 21/76 candidates cited the bot's own prior output -- circular
    memory, the bot proving its own earlier generation). Path B's raw scan
    now excludes is_bot rows entirely at the SQL level; Path A's hybrid
    retrieval already excluded them.
  - Path B no longer hardcodes support="fully_supported" just because the
    model cited an ID that happens to exist. Every candidate from BOTH
    paths goes through one shared _verify() stage against real evidence
    (with surrounding local context, not just the cited line) before a
    support level is assigned.
  - New epistemic_status field (stated_by_subject / reported_by_other /
    observed_event / inferred) -- "Влад сказал что Ретивов умер" must not
    collapse into the objective claim "Ретивов умер"; reported_by_other
    forces the claim text to keep its attribution.
  - Compound claims (e.g. health + relatives + an unrelated event glued
    into one claim) are detected and split into atomic claims, each
    re-verified against its own minimal source subset.
  - Subject strings are resolved through participants.resolve_participant_key
    (read-only lookup, no writes) so "Саша Тигмен" / "Sasha Tigmen" /
    "tigmen / Саша Тигмен" collapse to one canonical identity for dedup and
    per-subject capping instead of silently multiplying past the cap.
  - A lightweight contradiction pass groups candidates by resolved subject
    and flags (does not auto-resolve) directly conflicting claims, e.g. two
    opposite opinions both marked fully_supported.

No writes anywhere -- memory_facts, chat_context, chat_moments are never
touched. Output (the Markdown review sheet and raw JSON) is real
chat-derived content -- names, health, politics, sexual content, self-harm
mentions -- so it's git-ignored (see .gitignore) and written to /tmp, never
committed. Outputs a deduplicated Markdown review sheet for a human to
accept/edit/reject. No retrieval_cues or eval questions generated here --
just the claim and its minimal source set, per the brief.

Usage: python3 -m research.gold_candidate_builder
"""
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import config  # noqa: triggers dotenv load
from crypto_utils import decrypt
from database.db import get_conn
from display_names import resolve_display_name
from embeddings import embed, to_vector_literal
from llm.graphs import _or_query, _rrf
from llm.groq_client import get_chat_model
from participants import resolve_subject_for_fact

CHAT_ID = -1002335227490
RAW_CHUNK_SIZE = 40
MAX_WORKERS = 5
MAX_SPLIT_DEPTH = 2
KINDS = {
    "state", "durable_person_fact", "relationship", "commitment",
    "past_event", "opinion", "running_gag", "group_lore", "other",
}
EPISTEMIC_STATUSES = {"stated_by_subject", "reported_by_other", "observed_event", "inferred"}
CONTRADICTION_KINDS = {"opinion", "durable_person_fact", "relationship", "past_event"}


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
    """A few messages (bot included) around message_id, purely for the
    review sheet's human-facing display -- never treated as evidence."""
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


def _local_context_multi(chat_id, message_ids, span=3):
    """Bot-inclusive local context around EACH of message_ids (not just the
    first), merged and deduped by message_id -- a multi-source candidate
    whose sources are far apart (e.g. two replies with a big gap) should
    show that gap to the reviewer instead of hiding it behind a window
    around only the first source."""
    ids = sorted(set(message_ids))
    if not ids:
        return []
    seen = {}
    with get_conn() as conn:
        cursor = conn.cursor()
        for mid in ids:
            cursor.execute(
                """
                SELECT message_id, user_name, username, message, is_bot
                FROM messages WHERE user_id = %s AND message_id BETWEEN %s AND %s
                ORDER BY message_id ASC
                """,
                (chat_id, mid - span, mid + span),
            )
            for row_mid, name, uname, msg, is_bot in cursor.fetchall():
                if row_mid in seen:
                    continue
                try:
                    text = decrypt(msg)
                except Exception:
                    continue
                author = "БОТ" if is_bot else resolve_display_name(uname, name)
                seen[row_mid] = f"[{row_mid}] {author}: {text}"
    return [seen[k] for k in sorted(seen)]


def _real_rows(chat_id, message_ids):
    """Only used to render final, already-verified source_message_ids as
    display rows -- those can never include bot messages by construction
    (Path A's retrieval and Path B's chunking both exclude is_bot), so no
    extra filter needed here."""
    if not message_ids:
        return []
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT message_id, user_name, username, message, message_date "
            "FROM messages WHERE user_id = %s AND message_id = ANY(%s) ORDER BY message_id ASC",
            (chat_id, list(set(message_ids))),
        )
        rows = cursor.fetchall()
    out = []
    for mid, name, uname, msg, date in rows:
        try:
            text = decrypt(msg)
        except Exception:
            continue
        out.append({"message_id": mid, "author": resolve_display_name(uname, name), "text": text, "date": date})
    return out


def _evidence_window(chat_id, message_ids, span=4):
    """Non-bot messages surrounding each of message_ids, deduped and
    sorted -- gives the verifier real conversational context instead of
    judging an isolated cited line out of context (e.g. "все с 1 сентября"
    collapsing into "все присоединились 1 сентября")."""
    ids = sorted(set(message_ids))
    if not ids:
        return []
    seen = {}
    with get_conn() as conn:
        cursor = conn.cursor()
        for mid in ids:
            cursor.execute(
                "SELECT message_id, user_name, username, message FROM messages "
                "WHERE user_id = %s AND is_bot = FALSE AND message_id BETWEEN %s AND %s "
                "ORDER BY message_id ASC",
                (chat_id, mid - span, mid + span),
            )
            for row_mid, name, uname, msg in cursor.fetchall():
                if row_mid in seen:
                    continue
                try:
                    text = decrypt(msg)
                except Exception:
                    continue
                seen[row_mid] = {"message_id": row_mid, "author": resolve_display_name(uname, name), "text": text}
    return [seen[k] for k in sorted(seen)]


# ---------------------------------------------------------- verification --

def _verify_prompt(claim, kind_hint, subject_hint, evidence_rows):
    blocks = "\n".join(f"[msg_id={r['message_id']}] {r['author']}: {r['text']}" for r in evidence_rows)
    return f"""Ты — строгий verifier в pipeline построения памяти чат-бота. Тебе дают предполагаемый факт и реальные сообщения чата (с окружающим контекстом, не только предположительно подтверждающую строку). НЕ доверяй тому, что факт верен, только потому что кандидаты рядом по теме — проверяй буквальное подтверждение.

Предполагаемый факт: "{claim}"
Предполагаемая категория: {kind_hint or "?"}
Предполагаемый субъект: {subject_hint or "(не указан / вся группа)"}

Сообщения (реальные, с контекстом вокруг):
{blocks}

ВАЖНО — эпистемический статус. Не путай "кто-то сказал X" с "X истинно":
- stated_by_subject — сам субъект написал это о себе (я/меня/моё);
- reported_by_other — ДРУГОЙ человек утверждает это о субъекте (пересказ, слух, чужие слова) — claim ОБЯЗАН сохранять атрибуцию, например "По словам Влада, Ретивов умер", а НЕ "Ретивов умер";
- observed_event — напрямую наблюдаемое в переписке событие, не требует человека-источника;
- inferred — вывод из паттерна поведения/переписки, никто явно не утверждал.

ВАЖНО — составные факты. Если предполагаемый факт склеивает НЕСКОЛЬКО разных, логически не связанных фактов (например здоровье + родственники + отдельное происшествие в одном предложении) — верни is_compound=true и раздели на атомарные части в split_into: список объектов вида {{"claim": "...", "source_message_ids": [123]}} (ключ ИМЕННО source_message_ids, не minimal_source_message_ids), каждая часть — один атомарный факт со своим минимальным набором msg_id из сообщений выше. Если is_compound=true, остальные поля можно оставить пустыми/приблизительными — они не используются.

Если факт не составной, определи:
1. subject
2. kind — одно из: state, durable_person_fact, relationship, commitment, past_event, opinion, running_gag, group_lore, other
3. epistemic_status — одно из: stated_by_subject, reported_by_other, observed_event, inferred
4. claim — финальная формулировка (по-русски), с сохранённой атрибуцией если epistemic_status=reported_by_other
5. minimal_source_message_ids — МИНИМАЛЬНЫЙ набор msg_id из сообщений выше, которые ДЕЙСТВИТЕЛЬНО и ЛИТЕРАЛЬНО подтверждают claim (не просто рядом по теме)
6. support — fully_supported (буквально подтверждён) / partially_supported (частично, есть домысливание) / unsupported (сообщения НЕ подтверждают claim, только показались похожими по теме)

Верни ТОЛЬКО JSON: {{"is_compound": false, "split_into": [], "subject": "...", "kind": "...", "epistemic_status": "...", "claim": "...", "minimal_source_message_ids": [123], "support": "..."}}"""


def _verify(chat_id, claim, kind_hint, subject_hint, evidence_rows, depth=0):
    """Returns a list of 0+ verified candidate dicts (0 or 1 normally, more
    if the claim was compound and got split). Never invents an id outside
    evidence_rows, never assigns support without actually checking."""
    if not evidence_rows:
        return []
    result = _judge_json(_verify_prompt(claim, kind_hint, subject_hint, evidence_rows))
    if not result:
        return []
    valid_ids = {r["message_id"] for r in evidence_rows}

    if result.get("is_compound") and depth < MAX_SPLIT_DEPTH:
        out = []
        for item in result.get("split_into") or []:
            sub_claim = (item.get("claim") or "").strip()
            raw_ids = item.get("source_message_ids") or item.get("minimal_source_message_ids") or []
            sub_ids = [mid for mid in raw_ids if mid in valid_ids]
            if not sub_claim or not sub_ids:
                continue
            sub_evidence = [r for r in evidence_rows if r["message_id"] in sub_ids]
            out.extend(_verify(chat_id, sub_claim, kind_hint, subject_hint, sub_evidence, depth=depth + 1))
        return out

    if result.get("support") == "unsupported":
        return []
    sources = [mid for mid in (result.get("minimal_source_message_ids") or []) if mid in valid_ids]
    if not sources:
        return []
    epistemic = result.get("epistemic_status")
    if epistemic not in EPISTEMIC_STATUSES:
        epistemic = "inferred"
    return [{
        "claim": (result.get("claim") or claim).strip(),
        "kind": result.get("kind") if result.get("kind") in KINDS else (kind_hint or "other"),
        "subject_raw": result.get("subject") or subject_hint,
        "epistemic_status": epistemic,
        "source_message_ids": sources,
        "support": result.get("support") if result.get("support") in {"fully_supported", "partially_supported"} else "partially_supported",
    }]


# ---------------------------------------------------------- path A: moments --

def _hybrid_candidates(chat_id, query_text, query_embedding, top_k=15):
    with get_conn() as conn:
        cursor = conn.cursor()
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


def _process_moment(chat_id, moment_id, claim, moment_embedding):
    candidate_ids = _hybrid_candidates(chat_id, claim, moment_embedding)
    if not candidate_ids:
        return []
    evidence_rows = _real_rows(chat_id, candidate_ids)
    if not evidence_rows:
        return []
    verified = _verify(chat_id, claim, None, None, evidence_rows)
    return [dict(v, path="A-moments", origin=f"moment:{moment_id}") for v in verified]


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
                results.extend(future.result())
            except Exception as e:
                print(f"  Path A moment {futures[future]} failed: {e}", file=sys.stderr)
    print(f"Path A: {len(results)} verified candidates out of {len(moments)} moments", file=sys.stderr)
    return results


# ------------------------------------------------------- path B: raw chunks --

def _recall_prompt(lines):
    return f"""Вот отрезок реальной переписки группового чата, сообщения занумерованы реальными message_id. Все сообщения здесь — от живых людей, сообщений бота тут нет.

Твоя задача — найти факты, которые ДЕЙСТВИТЕЛЬНО стоит запомнить надолго: такие, что если через недели или месяцы кто-то спросит про них в чате, ответ был бы полезен. Это НЕ пересказ каждого сообщения подряд.

Строго ИСКЛЮЧАЙ:
- одноразовые технические реплики без долгосрочной ценности ("система использует Postgres", "фича теперь работает") — если это не устойчивый факт о самом проекте/группе, а просто рабочий комментарий по ходу дела;
- короткие реакции без содержания ("ахаха", "согласен", "хз");
- бессмысленный шум ("+1", стикеры, эмодзи без контекста);
- пересказ КАЖДОГО сообщения — это не цель, цель это отобрать по-настоящему ценное.

Включай только то, что похоже на настоящий факт про КОНКРЕТНОГО человека (устойчивая черта, локация, работа/учёба, отношения, обещание, важное прошлое событие, устойчивое мнение) или про ГРУППУ целиком (устойчивая внутренняя шутка/мем, который повторяется, общий факт про группу). Одно сообщение с проходной шуткой — это НЕ running_gag, только если видно что шутка/паттерн повторяется или явно станет отсылкой.

ВАЖНО: один claim = ОДИН атомарный факт. Не склеивай несколько разных фактов ("здоровье + родственники + отдельное событие") в одно утверждение.

ВАЖНО: если факт — это пересказ ЧУЖИХ слов о ком-то (не сам человек написал о себе), формулируй claim с сохранением атрибуции ("По словам X, ..."), не превращай в объективное утверждение.

Категории (kind): state, durable_person_fact, relationship, commitment, past_event, opinion, running_gag, group_lore, other.

Для каждой находки: claim (сам факт одним предложением, ПО-РУССКИ), kind, subject (имя человека, как оно упомянуто в переписке, или null для group_lore), source_message_ids (реальные message_id, минимум один).

В этом отрезке из {len(lines)} сообщений обычно НЕТ более 3-5 действительно стоящих находок, часто 0. Не старайся заполнить список любой ценой.

Сообщения:
{chr(10).join(lines)}

Верни ТОЛЬКО JSON: {{"items": [{{"claim": "...", "kind": "...", "subject": "...", "source_message_ids": [123]}}]}} (пустой список items, если по-настоящему ценного ничего нет — это нормальный и частый результат)."""


def _process_chunk(chat_id, chunk_no, rows):
    """Proposal only -- no support is assigned here. rows already exclude
    is_bot at the SQL level (discover_path_b), so the bot's own prior
    output can never be cited as a source in the first place."""
    lines = []
    valid_ids = set()
    for msg_id, user_name, username, message in rows:
        try:
            text = decrypt(message)
        except Exception:
            continue
        lines.append(f"[message_id={msg_id}] {resolve_display_name(username, user_name)}: {text}")
        if msg_id is not None:
            valid_ids.add(msg_id)

    result = _judge_json(_recall_prompt(lines))
    if not result or not result.get("items"):
        return []

    proposals = []
    for item in result["items"]:
        kind = item.get("kind") if item.get("kind") in KINDS else "other"
        sources = [mid for mid in (item.get("source_message_ids") or []) if mid in valid_ids]
        claim = (item.get("claim") or "").strip()
        if not claim or not sources:
            continue
        proposals.append({
            "claim": claim, "kind": kind, "subject_raw": item.get("subject"),
            "proposed_source_message_ids": sources,
        })
    return proposals


def _verify_chunk_proposal(chat_id, proposal):
    evidence_rows = _evidence_window(chat_id, proposal["proposed_source_message_ids"])
    verified = _verify(chat_id, proposal["claim"], proposal["kind"], proposal["subject_raw"], evidence_rows)
    return [dict(v, path="B-raw-chunk", origin="raw-chunk") for v in verified]


def discover_path_b(chat_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT message_id, user_name, username, message "
            "FROM messages WHERE user_id = %s AND is_bot = FALSE ORDER BY id ASC",
            (chat_id,),
        )
        all_rows = cursor.fetchall()
    chunks = [all_rows[i:i + RAW_CHUNK_SIZE] for i in range(0, len(all_rows), RAW_CHUNK_SIZE)]
    print(f"Path B: {len(all_rows)} human messages, {len(chunks)} chunks of ~{RAW_CHUNK_SIZE}", file=sys.stderr)

    proposals = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process_chunk, chat_id, i, chunk): i for i, chunk in enumerate(chunks)}
        done = 0
        for future in as_completed(futures):
            done += 1
            print(f"  Path B propose progress: {done}/{len(chunks)}", file=sys.stderr)
            try:
                proposals.extend(future.result())
            except Exception as e:
                print(f"  Path B chunk {futures[future]} failed: {e}", file=sys.stderr)
    print(f"Path B: {len(proposals)} raw proposals from {len(chunks)} chunks, now verifying each", file=sys.stderr)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_verify_chunk_proposal, chat_id, p): i for i, p in enumerate(proposals)}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 20 == 0:
                print(f"  Path B verify progress: {done}/{len(proposals)}", file=sys.stderr)
            try:
                results.extend(future.result())
            except Exception as e:
                print(f"  Path B proposal {futures[future]} failed to verify: {e}", file=sys.stderr)
    print(f"Path B: {len(results)} verified candidates from {len(proposals)} proposals", file=sys.stderr)
    return results


# ------------------------------------------------------ subject resolution --

def _resolve_subjects(chat_id, candidates):
    """Read-only lookup (participants.resolve_subject_for_fact, no writes)
    -- the same source-author-first resolver upsert_state actually uses,
    NOT the plain name-matching resolve_participant_key: a self-report like
    Игорь's own message about himself should resolve deterministically to
    the author who wrote it even when "Игорь" alone is ambiguous among
    known participants. Falls back to plain name matching only when
    neither the author nor a reply-target signal applies, so it can only
    resolve MORE cases than resolve_participant_key, never fewer. Left
    unresolved (subject_key=None) when genuinely ambiguous or unmatched --
    surfaced to the reviewer via subject_raw, not guessed."""
    for c in candidates:
        raw = c.get("subject_raw")
        resolved = resolve_subject_for_fact(chat_id, raw, c.get("source_message_ids")) if raw else None
        c["subject_key"] = resolved[0] if resolved else None
        c["subject_display"] = resolved[1] if resolved else raw
    return candidates


# --------------------------------------------------------- contradictions --

def _contradiction_prompt(claims_with_idx):
    blocks = "\n".join(f"{i}. {c}" for i, c in claims_with_idx)
    return f"""Вот несколько утверждений об одном и том же человеке/теме, извлечённых из разных мест переписки:
{blocks}

Найди пары утверждений, которые ПРЯМО противоречат друг другу (взаимоисключающие факты или противоположные мнения по одному и тому же вопросу) — не просто разные темы.

Верни ТОЛЬКО JSON: {{"contradictions": [{{"a": 0, "b": 1, "reason": "..."}}]}} (пустой список, если противоречий нет)."""


def _flag_contradictions(candidates):
    groups = defaultdict(list)
    for c in candidates:
        if c["kind"] not in CONTRADICTION_KINDS:
            continue
        key = c.get("subject_key") or (c.get("subject_raw") or "").strip().lower()
        if not key:
            continue
        groups[key].append(c)

    for key, group in groups.items():
        if len(group) < 2:
            continue
        result = _judge_json(_contradiction_prompt([(i, g["claim"]) for i, g in enumerate(group)]))
        if not result:
            continue
        for pair in result.get("contradictions") or []:
            a, b, reason = pair.get("a"), pair.get("b"), pair.get("reason", "")
            if not isinstance(a, int) or not isinstance(b, int) or not (0 <= a < len(group)) or not (0 <= b < len(group)):
                continue
            group[a]["contradiction_note"] = f"⚠ противоречит другому кандидату того же субъекта: {reason}"
            group[b]["contradiction_note"] = f"⚠ противоречит другому кандидату того же субъекта: {reason}"
    return candidates


# --------------------------------------------------------- dedup + select --

def _dedupe(candidates):
    """Semantic dedup on the claim text -- moments (path A) and raw-chunk
    scanning (path B) can independently surface the same real fact. A near-
    duplicate is MERGED into the kept candidate (union of source_message_ids,
    both discovery_paths recorded, support upgraded if the new one is
    stronger) rather than discarded -- two independent paths agreeing on the
    same fact is a useful corroboration signal, not noise to throw away.

    Requires _resolve_subjects() to have already run: merging is gated on
    matching subject_key (or normalized subject_raw when unresolved) AND
    matching kind -- claim-embedding similarity ALONE is not safe across
    different people. A real incident: Ярик's "Крым России" and Тигмен's
    "Крым абсолютно точно часть России" are similar enough by embedding to
    have merged into one candidate under Ярик's name with Тигмен's message
    silently reassigned as if it were Ярик's -- exactly the attribution
    collapse this whole research effort exists to prevent. See
    tests/test_gold_candidate_builder.py for the regression test."""
    if not candidates:
        return []
    embeddings = [embed(c["claim"]) for c in candidates]
    kept = []
    kept_meta = []  # (embedding, subject_bucket, kind) parallel to kept
    for cand, emb in zip(candidates, embeddings):
        subj_bucket = cand.get("subject_key") or (cand.get("subject_raw") or "").strip().lower()
        merge_target = None
        for i, (kept_emb, kept_subj, kept_kind) in enumerate(kept_meta):
            if subj_bucket != kept_subj or cand["kind"] != kept_kind:
                continue
            dot = sum(a * b for a, b in zip(emb, kept_emb))
            na = sum(a * a for a in emb) ** 0.5
            nb = sum(b * b for b in kept_emb) ** 0.5
            if na and nb and dot / (na * nb) > 0.92:
                merge_target = kept[i]
                break
        if merge_target is None:
            cand.setdefault("discovery_paths", [cand["path"]])
            kept.append(cand)
            kept_meta.append((emb, subj_bucket, cand["kind"]))
            continue
        merge_target["source_message_ids"] = sorted(set(merge_target["source_message_ids"]) | set(cand["source_message_ids"]))
        merge_target.setdefault("discovery_paths", [merge_target["path"]])
        if cand["path"] not in merge_target["discovery_paths"]:
            merge_target["discovery_paths"].append(cand["path"])
        if cand.get("support") == "fully_supported":
            merge_target["support"] = "fully_supported"
    return kept


def _select_best(candidates, max_per_subject=8):
    """Orders by support, caps per resolved subject_key (falls back to
    normalized subject_raw) so one person can't dominate the sheet. No
    separate global-size target: with a pool this small the per-subject cap
    alone is enough -- a "trim to top N" step on top of it was producing a
    confusing off-by-one mismatch between the configured target and the
    actual sheet size (71 candidates, target=70)."""
    support_rank = {"fully_supported": 0, "partially_supported": 1}
    candidates = sorted(candidates, key=lambda c: support_rank.get(c.get("support"), 1))
    selected = []
    per_subject = {}
    for c in candidates:
        subj = c.get("subject_key") or (c.get("subject_raw") or "").strip().lower() or "_none_"
        if per_subject.get(subj, 0) >= max_per_subject:
            continue
        selected.append(c)
        per_subject[subj] = per_subject.get(subj, 0) + 1
    return selected


# --------------------------------------------------------------- output --

def _render_markdown(candidates, chat_id):
    import datetime

    lines = ["# Gold candidate review sheet (v2)", "", f"Всего кандидатов на проверку: {len(candidates)}", ""]
    for i, c in enumerate(candidates, start=1):
        lines.append(f"### {i}.")
        lines.append(f"Claim: {c['claim']}")
        lines.append(f"Kind: {c['kind']}")
        subject_display = c.get("subject_display") or "(группа/не указано)"
        if c.get("subject_raw") and not c.get("subject_key"):
            subject_display += "  ⚠ не резолвится однозначно к участнику чата"
        lines.append(f"Subject: {subject_display}")
        lines.append(f"Epistemic status: {c.get('epistemic_status')}")
        paths = c.get("discovery_paths") or [c["path"]]
        path_label = " + ".join(paths) + ("  (оба пути независимо нашли этот факт)" if len(paths) > 1 else "")
        lines.append(f"Discovery path: {path_label}")

        source_rows = _real_rows(chat_id, c["source_message_ids"])
        dates = [datetime.datetime.fromtimestamp(r["date"]).strftime("%Y-%m-%d %H:%M") for r in source_rows if r.get("date")]
        if dates:
            span = f"{min(dates)}" if min(dates) == max(dates) else f"{min(dates)} .. {max(dates)}"
            lines.append(f"Observed at (source message dates): {span}")

        lines.append("Sources:")
        for row in source_rows:
            lines.append(f"- [{row['message_id']}] {row['author']}: {row['text'][:200]}")
        lines.append(f"Suggested support: {c.get('support')}")
        if c.get("contradiction_note"):
            lines.append(f"{c['contradiction_note']}")
        lines.append("")
        lines.append("<details><summary>Локальный контекст вокруг источников</summary>")
        lines.append("")
        lines.append("```")
        for ctx_line in _local_context_multi(chat_id, c["source_message_ids"]):
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
        lines.append("Manual annotation (fill in during review):")
        lines.append("- memory_worthy: [ ] yes  [ ] no")
        lines.append("- assertion_mode: [ ] literal  [ ] joke  [ ] sarcasm  [ ] rumor  [ ] uncertain")
        lines.append("- temporal_scope: [ ] persistent  [ ] transient  [ ] event")
        lines.append("- subject_type: [ ] person  [ ] group  [ ] other")
        lines.append("- sources_sufficient: [ ] yes  [ ] no")
        lines.append("- edited_claim: ")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


VERIFIED_CACHE_PATH = "/tmp/gold_candidates_verified.json"


def run():
    """Full pipeline: runs the LLM discovery+verification+contradiction
    stages (costly, non-deterministic -- re-running it produces a different
    set of candidates each time). Subject resolution and contradiction
    flagging happen HERE, before the cache is written, so the cached file
    already carries subject_key/contradiction_note and reprocess_cached()
    never needs an LLM call to reproduce them. The freshly verified
    candidates are written ONCE to VERIFIED_CACHE_PATH and never touched
    again by this function or by reprocess_cached() -- that file is the
    immutable source of truth for everything downstream, specifically so a
    resolver/render/dedup fix doesn't force throwing away and re-paying for
    a real run."""
    a = discover_path_a(CHAT_ID)
    b = discover_path_b(CHAT_ID)
    all_candidates = a + b
    print(f"\nTotal verified candidates: {len(all_candidates)}", file=sys.stderr)

    _resolve_subjects(CHAT_ID, all_candidates)
    _flag_contradictions(all_candidates)

    with open(VERIFIED_CACHE_PATH, "w") as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2, default=str)
    print(f"Verified candidates (immutable cache) written to {VERIFIED_CACHE_PATH}", file=sys.stderr)

    reprocess_cached(VERIFIED_CACHE_PATH)


def reprocess_cached(input_path=VERIFIED_CACHE_PATH, apply_dedup=False, max_per_subject=100, out_prefix="/tmp/gold_review_sheet"):
    """Zero LLM calls -- only subject resolution (deterministic DB lookup,
    participants.resolve_subject_for_fact) and rendering against an already
    LLM-verified candidate cache. Exists so a code fix to the resolver,
    dedup, or the sheet layout doesn't require an expensive, non-
    deterministic full re-run of discovery+verification.

    apply_dedup defaults to False: for a pool this size a human doing full
    manual review can merge true duplicates by hand far more safely than an
    embedding-similarity heuristic can -- one already, once, silently
    reassigned one person's source message to another person's subject
    (see _dedupe's docstring and tests/test_gold_candidate_builder.py).
    Only turn it on for a corpus large enough that manual merging stops
    being practical; _dedupe itself now refuses to merge across different
    subject_key/kind, but that's a floor, not a guarantee of safety.

    Never overwrites input_path -- writes to out_prefix + ".md"/".json"."""
    with open(input_path) as f:
        candidates = json.load(f)

    _resolve_subjects(CHAT_ID, candidates)
    if apply_dedup:
        candidates = _dedupe(candidates)
    print(f"Reprocessing {len(candidates)} candidates from {input_path} (apply_dedup={apply_dedup}, 0 LLM calls)", file=sys.stderr)

    selected = _select_best(candidates, max_per_subject=max_per_subject)
    print(f"Selected for review sheet: {len(selected)}", file=sys.stderr)

    md = _render_markdown(selected, CHAT_ID)
    md_path = f"{out_prefix}.md"
    with open(md_path, "w") as f:
        f.write(md)

    json_path = f"{out_prefix}_processed.json"
    with open(json_path, "w") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2, default=str)

    print(f"Review sheet written to {md_path} ({len(selected)} candidates)", file=sys.stderr)
    return md_path


if __name__ == "__main__":
    if "--reprocess-cached" in sys.argv:
        path = VERIFIED_CACHE_PATH
        for arg in sys.argv[1:]:
            if arg.startswith("--input="):
                path = arg.split("=", 1)[1]
        reprocess_cached(path, apply_dedup="--dedup" in sys.argv)
    else:
        run()
