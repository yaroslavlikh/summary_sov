"""Forced structured-output extractor -- the mechanism
research/PROSPECTIVE_MEMORY_RETRIEVAL.md's roadmap called for once gold
was frozen: forced JSON output instead of free-choice tool-calling, 30-50
message batches instead of 180, temperature=0, deterministic validation
after generation instead of the model deciding whether to call a tool at
all. That free-choice mechanism was measured earlier (dry_run_extraction_
full_history.py) to produce 0-1 candidates across two identical runs --
this tests whether forcing structured output fixes that.

Deliberately ONE pass per batch, no separate verification stage -- that's
exactly the mechanism under test. The extraction prompt is intentionally
close to verbatim to gold-mining's Path B recall prompt in
research/gold_candidate_builder.py: that prompt was specifically tuned
during gold construction to stop over-triggering (v1 proposed a "fact" on
almost every message; the fix was an explicit durability bar and an
expectation of 0-5 findings per batch, not one per message). Reusing that
tested prompt isolates the actual variable this experiment measures: is a
cheap single forced-JSON pass -- without gold-mining's expensive separate
verify stage (local context, epistemic_status classification, compound
splitting) -- already good enough to recover what a human confirmed as
real, memory-worthy facts?

Uses gold's same wide kind taxonomy (state, durable_person_fact,
relationship, commitment, past_event, opinion, running_gag, group_lore),
not production's current narrow ALLOWED_STATE_KEYS in memory_facts.py --
extending that schema to match is a separate, later decision; this
measures the extraction MECHANISM, not the current schema's scope.

Read-only: no writes to memory_facts/chat_context/chat_moments. Evaluated
against the frozen gold (research/build_gold_facts.py's
/tmp/gold_facts.jsonl, checksum-pinned -- refuses to run if the file
changed since freeze) via source_message_id overlap, not semantic
similarity of claim text: two independent generations of the same fact
will basically never match verbatim, but they will cite the same evidence
if they're really about the same thing.

Usage: python3 -m research.forced_extractor_eval
"""
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import config  # noqa: triggers dotenv load
from crypto_utils import decrypt
from database.db import get_conn
from display_names import resolve_display_name
from llm.groq_client import get_chat_model
from participants import resolve_subject_for_fact

CHAT_ID = -1002335227490
BATCH_SIZE = 40
MAX_WORKERS = 5

GOLD_PATH = "/tmp/gold_facts.jsonl"
GOLD_CHECKSUM_PATH = "/tmp/gold_facts.sha256"
OUT_PATH = "/tmp/forced_extractor_output.json"
REPORT_PATH = "/tmp/forced_extractor_report.md"

KINDS = {
    "state", "durable_person_fact", "relationship", "commitment",
    "past_event", "opinion", "running_gag", "group_lore", "other",
}


# --------------------------------------------------------------- extract --

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


def _extraction_prompt(lines):
    return f"""Вот отрезок реальной переписки группового чата, сообщения занумерованы реальными message_id. Все сообщения здесь — от живых людей, сообщений бота тут нет.

Твоя задача — найти факты, которые ДЕЙСТВИТЕЛЬНО стоит запомнить надолго: такие, что если через недели или месяцы кто-то спросит про них в чате, ответ был бы полезен. Это НЕ пересказ каждого сообщения подряд.

Строго ИСКЛЮЧАЙ:
- одноразовые технические реплики без долгосрочной ценности — если это не устойчивый факт о самом проекте/группе, а просто рабочий комментарий по ходу дела;
- короткие реакции без содержания ("ахаха", "согласен", "хз");
- бессмысленный шум ("+1", стикеры, эмодзи без контекста);
- пересказ КАЖДОГО сообщения — это не цель, цель это отобрать по-настоящему ценное.

Включай только то, что похоже на настоящий факт про КОНКРЕТНОГО человека (устойчивая черта, локация, работа/учёба, отношения, обещание, важное прошлое событие, устойчивое мнение) или про ГРУППУ целиком (устойчивая внутренняя шутка/мем, который повторяется, общий факт про группу). Одно сообщение с проходной шуткой — это НЕ running_gag, только если видно что шутка/паттерн повторяется или явно станет отсылкой.

ВАЖНО: один claim = ОДИН атомарный факт. Не склеивай несколько разных фактов в одно утверждение.

ВАЖНО: если факт — это пересказ ЧУЖИХ слов о ком-то (не сам человек написал о себе), формулируй claim с сохранением атрибуции ("По словам X, ..." или "X написал, что считает..."), не превращай в объективное утверждение.

Категории (kind): state, durable_person_fact, relationship, commitment, past_event, opinion, running_gag, group_lore, other.

Для каждой находки: claim (сам факт одним предложением, ПО-РУССКИ), kind, subject (имя человека, как оно упомянуто в переписке, или null для group_lore), source_message_ids (реальные message_id, минимум один).

В этом отрезке из {len(lines)} сообщений обычно НЕТ более 3-5 действительно стоящих находок, часто 0. Не старайся заполнить список любой ценой.

Сообщения:
{chr(10).join(lines)}

Верни ТОЛЬКО JSON: {{"items": [{{"claim": "...", "kind": "...", "subject": "...", "source_message_ids": [123]}}]}} (пустой список items, если по-настоящему ценного ничего нет — это нормальный и частый результат)."""


def _extract_batch(chat_id, batch_no, rows):
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

    result = _judge_json(_extraction_prompt(lines))
    if not result or not result.get("items"):
        return []

    out = []
    for item in result["items"]:
        kind = item.get("kind") if item.get("kind") in KINDS else "other"
        sources = [mid for mid in (item.get("source_message_ids") or []) if mid in valid_ids]
        claim = (item.get("claim") or "").strip()
        if not claim or not sources:
            continue
        subject_raw = item.get("subject")
        resolved = resolve_subject_for_fact(chat_id, subject_raw, sources) if subject_raw else None
        out.append({
            "batch": batch_no,
            "claim": claim,
            "kind": kind,
            "subject_raw": subject_raw,
            "subject_key": resolved[0] if resolved else None,
            "subject_display": resolved[1] if resolved else subject_raw,
            "source_message_ids": sources,
        })
    return out


def run_extraction(chat_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT message_id, user_name, username, message FROM messages "
            "WHERE user_id = %s AND is_bot = FALSE ORDER BY id ASC",
            (chat_id,),
        )
        all_rows = cursor.fetchall()
    chunks = [all_rows[i:i + BATCH_SIZE] for i in range(0, len(all_rows), BATCH_SIZE)]
    print(f"Forced extractor: {len(all_rows)} human messages, {len(chunks)} batches of ~{BATCH_SIZE}", file=sys.stderr)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_extract_batch, chat_id, i, chunk): i for i, chunk in enumerate(chunks)}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 10 == 0:
                print(f"  progress: {done}/{len(chunks)}", file=sys.stderr)
            try:
                results.extend(future.result())
            except Exception as e:
                print(f"  batch {futures[future]} failed: {e}", file=sys.stderr)

    print(f"Forced extractor produced {len(results)} candidates from {len(chunks)} batches", file=sys.stderr)
    return results


# -------------------------------------------------------------- evaluate --

def _load_gold():
    with open(GOLD_CHECKSUM_PATH) as f:
        expected = f.read().split()[0]
    actual = hashlib.sha256(open(GOLD_PATH, "rb").read()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"gold_facts.jsonl checksum mismatch: expected {expected}, got {actual} -- "
            "the frozen gold file changed since it was checksummed, refusing to evaluate against it"
        )
    return [json.loads(line) for line in open(GOLD_PATH)]


def _matches(gold_fact, extracted):
    """Loose match: any shared source_message_id. Strict match: shared id
    AND matching resolved subject_key (only when both sides resolved one)
    -- id overlap alone can't be trusted when a batch legitimately covers
    several facts clustered around the same messages."""
    gold_ids = set(gold_fact["source_message_ids"])
    loose, strict = [], []
    for e in extracted:
        if gold_ids & set(e["source_message_ids"]):
            loose.append(e)
            gold_key, extracted_key = gold_fact.get("subject_key"), e.get("subject_key")
            if gold_key and extracted_key and gold_key == extracted_key:
                strict.append(e)
            elif not gold_key and not extracted_key:
                strict.append(e)
    return loose, strict


def evaluate(extracted, gold):
    positive = [g for g in gold if g["label"] == "positive"]
    rejected = [g for g in gold if g["label"] == "rejected_candidate"]
    multi_source = [g for g in positive if len(g["source_message_ids"]) > 1]

    pos_matches = [(g, *_matches(g, extracted)) for g in positive]
    multi_matches = [(g, *_matches(g, extracted)) for g in multi_source]
    rej_matches = [(g, *_matches(g, extracted)) for g in rejected]

    def recall(matches):
        return sum(1 for _, loose, _ in matches if loose) / len(matches) if matches else None

    def recall_strict(matches):
        return sum(1 for _, _, strict in matches if strict) / len(matches) if matches else None

    return {
        "n_extracted": len(extracted),
        "n_positive": len(positive),
        "n_positive_recovered_loose": sum(1 for _, loose, _ in pos_matches if loose),
        "n_positive_recovered_strict": sum(1 for _, _, strict in pos_matches if strict),
        "recall_loose": recall(pos_matches),
        "recall_strict": recall_strict(pos_matches),
        "n_multi_source": len(multi_source),
        "n_multi_source_recovered_loose": sum(1 for _, loose, _ in multi_matches if loose),
        "recall_multi_source_loose": recall(multi_matches),
        "n_rejected": len(rejected),
        "n_rejected_resurfaced_loose": sum(1 for _, loose, _ in rej_matches if loose),
        "missed_positive": [g for g, loose, _ in pos_matches if not loose],
        "pos_matches": pos_matches,
        "multi_matches": multi_matches,
        "rej_matches": rej_matches,
    }


def _render_report(stats):
    lines = ["# Forced extractor eval report", ""]
    lines.append(f"Extracted candidates (all batches): {stats['n_extracted']}")
    lines.append("")
    lines.append(f"## Recall on {stats['n_positive']} known positive facts")
    lines.append(f"- loose (source_message_id overlap): {stats['n_positive_recovered_loose']}/{stats['n_positive']} = {stats['recall_loose']:.0%}")
    lines.append(f"- strict (id overlap + matching subject_key): {stats['n_positive_recovered_strict']}/{stats['n_positive']} = {stats['recall_strict']:.0%}")
    lines.append("")
    lines.append(f"## Recall on {stats['n_multi_source']} multi-source positive facts")
    if stats["n_multi_source"]:
        lines.append(f"- loose: {stats['n_multi_source_recovered_loose']}/{stats['n_multi_source']} = {stats['recall_multi_source_loose']:.0%}")
    lines.append("")
    lines.append(f"## Rejected candidates resurfaced: {stats['n_rejected_resurfaced_loose']}/{stats['n_rejected']}")
    lines.append("(extractor proposed something overlapping a source the human already rejected once)")
    lines.append("")

    lines.append("## Missed positive facts (no source_message_id overlap at all)")
    for g in stats["missed_positive"]:
        lines.append(f"- #{g['id']} [{g['kind']}] {g['claim']}  (sources: {g['source_message_ids']})")
    lines.append("")

    lines.append("## Recovered positive facts, extractor's own wording")
    for g, loose, strict in stats["pos_matches"]:
        if not loose:
            continue
        tag = "STRICT" if strict else "loose-only"
        lines.append(f"- #{g['id']} [{tag}] gold: {g['claim']}")
        for m in loose:
            lines.append(f"    extractor: {m['claim']}  (subject_key={m['subject_key']}, sources={m['source_message_ids']})")
    lines.append("")

    resurfaced = [(g, loose) for g, loose, _ in stats["rej_matches"] if loose]
    if resurfaced:
        lines.append("## Resurfaced rejected candidates")
        for g, loose in resurfaced:
            lines.append(f"- #{g['id']} rejected: {g['claim']}")
            for m in loose:
                lines.append(f"    extractor: {m['claim']}  (sources={m['source_message_ids']})")

    return "\n".join(lines)


def run():
    gold = _load_gold()
    print(f"Loaded {len(gold)} frozen gold facts (checksum verified)", file=sys.stderr)

    extracted = run_extraction(CHAT_ID)
    with open(OUT_PATH, "w") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    stats = evaluate(extracted, gold)
    report = _render_report(stats)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\nrecall_loose={stats['recall_loose']:.0%} recall_strict={stats['recall_strict']:.0%} "
          f"multi_source_recall={stats['recall_multi_source_loose']}  "
          f"rejected_resurfaced={stats['n_rejected_resurfaced_loose']}/{stats['n_rejected']}", file=sys.stderr)
    print(f"Report: {REPORT_PATH}", file=sys.stderr)
    print(f"Raw extractor output: {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    run()
