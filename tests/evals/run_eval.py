"""Real-invocation benchmark for the /ask LangGraph pipeline.

Replays real historical bot-mention questions (mined from production
`messages`) through the CURRENT graph, read-only (no Telegram sends, no
writes back to `messages`), and scores four metrics per case via LLM
judges, pushed to Langfuse as a dataset run.

Usage: python3 tests/evals/run_eval.py [chat_id] [limit]
"""
import re
import sys
import time

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

from database.db import get_conn
from langfuse import get_client
from llm.graphs import run_ask_graph
from tests.evals.judges import ALL_EVALUATORS
from tests.evals.mine_cases import mine_cases


class _NullBot:
    """Eval runs are read-only against production: never actually send a
    Telegram message. save_bot_answer is stubbed separately (below) so no
    row gets written back to `messages` either -- an eval run must never
    pollute the corpus it's being scored against."""
    def send_message(self, chat_id, text, parse_mode=None, message_thread_id=None):
        class _Sent:
            message_id = None
            date = None
        return _Sent()


def _internal_ids_to_message_ids(chat_id, internal_ids):
    if not internal_ids:
        return []
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, message_id FROM messages WHERE user_id = %s AND id = ANY(%s)",
            (chat_id, list(internal_ids)),
        )
        by_id = dict(cursor.fetchall())
    return [by_id[i] for i in internal_ids if i in by_id and by_id[i] is not None]


def _cited_message_ids(answer_plain, window_rows):
    """[N] in answer_plain indexes into window_rows (1-based, same order
    _generate_answer built the legend in) -- map citation numbers back to
    real Telegram message_id for the judges."""
    if not answer_plain or not window_rows:
        return []
    ids = []
    for match in re.finditer(r"\[(\d+)\]", answer_plain):
        n = int(match.group(1))
        if 1 <= n <= len(window_rows):
            ids.append(window_rows[n - 1][1])  # (row_id, message_id, ...)
    return sorted(set(ids))


def ask_task(*, item, **kwargs):
    # item is a real Langfuse DatasetItem (pydantic model) here, not a plain
    # dict -- its case payload lives on the .input attribute, not item["input"].
    case = item.input
    state = {
        "bot": _NullBot(),
        "chat_id": case["chat_id"],
        "question": case["question"],
        "asker_name": case["asker_name"],
        "replied_message_id": case["replied_message_id"],
        "bot_username": case["bot_username"],
        "thread_id": None,
        "save_bot_answer": lambda *a, **kw: None,  # no-op: never write eval output back into messages
    }
    started = time.perf_counter()
    final = run_ask_graph(state)
    latency_seconds = round(time.perf_counter() - started, 2)

    candidate_message_ids = _internal_ids_to_message_ids(case["chat_id"], final.get("candidate_ids") or [])
    match_message_ids = _internal_ids_to_message_ids(case["chat_id"], final.get("match_ids") or [])
    window_rows = final.get("window_rows") or []
    cited_message_ids = _cited_message_ids(final.get("answer_plain"), window_rows)

    return {
        "answer": final.get("answer_plain") or "(нет ответа)",
        "anchor_id": final.get("anchor_id"),
        "intent": final.get("intent"),
        "candidate_message_ids": candidate_message_ids,
        "match_message_ids": match_message_ids,
        "cited_message_ids": cited_message_ids,
        "first_candidate_message_id": candidate_message_ids[0] if candidate_message_ids else None,
        "latency_seconds": latency_seconds,
    }


def latency_evaluator(*, input, output, expected_output=None, **kwargs):
    """Wall-clock time for the whole graph invocation (anchor path is near-
    instant; no-anchor path pays for rewrite+search+rerank+answer in
    sequence) -- pushed as a plain numeric score alongside the quality
    metrics, not judged by an LLM."""
    seconds = output.get("latency_seconds")
    if seconds is None:
        return []
    return {"name": "latency_seconds", "value": seconds, "comment": f"{seconds:.2f}s end-to-end"}


DATASET_NAME = "ask-pipeline-real-invocations"


def _sync_dataset(langfuse, chat_id, limit):
    """Create the dataset if it doesn't exist yet, and upsert mined cases as
    items -- each item's id is derived from the source DB row, so re-running
    the miner never creates duplicate items, it just fills in new ones."""
    try:
        langfuse.create_dataset(
            name=DATASET_NAME,
            description="Реальные вопросы боту (mention-triggered), замайненные из production messages.",
        )
    except Exception:
        pass  # already exists -- create_dataset isn't idempotent, get_dataset below is what matters

    cases = mine_cases(chat_id, limit=limit)
    for case in cases:
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=f"ask-row-{case['source_row_id']}",
            input=case,
            metadata={"chat_id": chat_id, "asker_name": case["asker_name"]},
        )
    return cases


def run(chat_id=-1002335227490, limit=20):
    langfuse = get_client()
    cases = _sync_dataset(langfuse, chat_id, limit)
    if not cases:
        print("Не нашёл ни одного реального вызова бота для этого чата.")
        return

    print(f"Датасет '{DATASET_NAME}' синхронизирован, {len(cases)} кейсов замайнено в этот прогон.")
    print(
        "ВАЖНО: reply_to_message_id/message_date -- новые колонки, у старых записей NULL, "
        "даже если сообщение реально было reply'ем -- почти все замайненные кейсы уйдут в "
        "no-anchor ветку графа независимо от того, был ли в реальности reply."
    )

    dataset = langfuse.get_dataset(DATASET_NAME)
    result = dataset.run_experiment(
        name="ask-pipeline-eval-run",
        description="Реплей реальных вопросов боту через текущий LangGraph /ask пайплайн, read-only.",
        task=ask_task,
        evaluators=ALL_EVALUATORS + [latency_evaluator],
        max_concurrency=3,  # Groq rate limits -- don't hammer them
    )

    print(f"\nГотово. {len(result.item_results)} кейсов обработано.")
    if getattr(result, "dataset_run_url", None):
        print(f"Смотри в Langfuse: {result.dataset_run_url}")

    by_metric = {}
    for item_result in result.item_results:
        for ev in item_result.evaluations:
            if ev.value is None:
                continue
            by_metric.setdefault(ev.name, []).append(ev.value)

    print("\nСредние по метрикам:")
    for name, values in by_metric.items():
        print(f"  {name}: {sum(values) / len(values):.2f} (n={len(values)})")


if __name__ == "__main__":
    chat_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1002335227490
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    run(chat_id, limit)
