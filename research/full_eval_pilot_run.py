"""Runs the FULL, standard ask-pipeline-eval-run -- all 20 real mined cases
from the ask-pipeline-real-invocations dataset, full search+rerank+generate
pipeline, scored by the SAME judges (tests/evals/judges.py) -- except
generate_answer's anchor window comes from the semantic pilot instead of
production's conversation_id SQL. Directly comparable in the Langfuse UI to
the existing "ask-pipeline-eval-run" runs (same run name, same dataset).

Read-only relative to messages/chat_context/chat_moments: save_bot_answer and
write_memory are both no-ops, exactly like tests/evals/run_eval.py.

Usage: python3 -m research.full_eval_pilot_run
"""
import re
import sys
import time

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

from database.db import get_conn
from langfuse import get_client

from tests.evals.judges import ALL_EVALUATORS
from research.pilot_ask_graph import run_pilot_ask_graph

DATASET_NAME = "ask-pipeline-real-invocations"


class _NullBot:
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
    if not answer_plain or not window_rows:
        return []
    ids = []
    for match in re.finditer(r"\[(\d+)\]", answer_plain):
        n = int(match.group(1))
        if 1 <= n <= len(window_rows):
            ids.append(window_rows[n - 1][1])
    return sorted(set(ids))


def ask_task(*, item, **kwargs):
    case = item.input
    state = {
        "bot": _NullBot(),
        "chat_id": case["chat_id"],
        "question": case["question"],
        "asker_name": case["asker_name"],
        "replied_message_id": case["replied_message_id"],
        "bot_username": case["bot_username"],
        "thread_id": None,
        "save_bot_answer": lambda *a, **kw: None,
        "write_memory": lambda *a, **kw: None,
    }
    started = time.perf_counter()
    final = run_pilot_ask_graph(state)
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
    seconds = output.get("latency_seconds")
    if seconds is None:
        return []
    return {"name": "latency_seconds", "value": seconds, "comment": f"{seconds:.2f}s end-to-end"}


def run():
    langfuse = get_client()
    dataset = langfuse.get_dataset(DATASET_NAME)
    print(f"Running full ask-pipeline-eval-run (semantic pilot window) on {len(dataset.items)} items...")

    result = dataset.run_experiment(
        name="ask-pipeline-eval-run",
        description="Полный /ask граф (search+rerank+generate) с semantic-pilot окном вместо conversation_id SQL в generate_answer -- read-only, save_bot_answer/write_memory no-op.",
        task=ask_task,
        evaluators=ALL_EVALUATORS + [latency_evaluator],
        max_concurrency=3,
    )

    print(f"\nГотово. {len(result.item_results)} кейсов обработано.")
    if getattr(result, "dataset_run_url", None):
        print(f"Смотри в Langfuse: {result.dataset_run_url}")

    by_metric = {}
    for item_result in result.item_results:
        for ev in item_result.evaluations:
            if ev.value is not None:
                by_metric.setdefault(ev.name, []).append(ev.value)

    print("\nСредние по метрикам:")
    for name, values in by_metric.items():
        print(f"  {name}: {sum(values) / len(values):.2f} (n={len(values)})")


if __name__ == "__main__":
    run()
