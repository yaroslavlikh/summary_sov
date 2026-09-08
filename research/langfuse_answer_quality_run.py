"""Real answer-quality comparison: generate an actual /ask answer using the
PRODUCTION window (conversation_id / +-3 fallback, exactly what's live right
now) vs the SEMANTIC PILOT's window, for the same anchor -- then score both
with the EXISTING Langfuse LLM judges (tests/evals/judges.py) that are
already used for the real ask-pipeline-eval-run, so this is directly
comparable to those runs, not a new ad-hoc metric.

This DOES call the real primary/fast Groq models (answer generation +
judges) -- contained to the ~3-5 anchor cases that have a resolvable window
in both variants, not the full 20-item dataset, and not the full /ask graph
(no search/rerank/anchor-resolution/memory_command -- just the answer-
generation step in isolation, fed an explicit message_id window).

Usage: python3 -m research.langfuse_answer_quality_run
"""
import re
import sys

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import config
from crypto_utils import decrypt
from database.db import get_conn
from display_names import resolve_display_name
from embeddings import embed
from langfuse import get_client

from chat_moments import search_moments
from llm.graphs import _config, _format_citations, _group_context, _message_link, _content
from llm.groq_client import get_chat_model
from llm.prompt import prompt_for_qa
from tests.evals.judges import ALL_EVALUATORS

from research.conversation_disentanglement import semantic_pilot
from research.db_loader import load_messages

CHAT_ID = -1002335227490
SOURCE_DATASET_NAME = "ask-pipeline-real-invocations"
# A dedicated small dataset holding ONLY the anchor cases where a window is
# resolvable in BOTH variants -- the existing judges (tests/evals/judges.py)
# have no "skip this item" protocol, so feeding them the other 17
# non-comparable items from the full dataset makes agent_goal average in a
# bunch of "bot said nothing to a real question" scores that have nothing to
# do with window quality. A separate dataset keeps the comparison honest.
COMPARISON_DATASET_NAME = "conversation-window-anchor-comparison"
RESOLVABLE_SOURCE_ROW_IDS = {46173, 46416, 46434}


def _window_rows(chat_id, message_ids, anchor_message_ids):
    # Mirrors llm/graphs.py _generate_answer's is_bot filter: a row is only
    # allowed in if it's genuinely content, or is one of the anchors itself.
    # Missing this let bot rows leak into the window for BOTH variants,
    # confounding the clustering-method comparison with an unrelated
    # content-filter difference -- fixed after review.
    if not message_ids:
        return []
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, message_id, message_thread_id, user_name, username, message
            FROM messages WHERE user_id = %s AND message_id = ANY(%s)
              AND (is_bot = FALSE OR message_id = ANY(%s))
            ORDER BY message_id ASC
            """,
            (chat_id, list(message_ids), list(anchor_message_ids)),
        )
        return cur.fetchall()


def _anchor_row_id(chat_id, anchor_message_id):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM messages WHERE user_id = %s AND message_id = %s", (chat_id, anchor_message_id))
        row = cur.fetchone()
        return row[0] if row else None


def _generate_answer_for_window(chat_id, question, asker_name, anchor_row_id, rows, config_):
    """Exact reproduction of llm/graphs.py _generate_answer's prompt-building
    and generation step, given an explicit set of window rows instead of a
    conversation_id-derived query -- so the two variants only differ in
    WHICH messages are shown to the model, nothing else about the pipeline."""
    if not rows:
        return {"answer_plain": None, "window_rows": []}
    legend, lines = {}, []
    for index, (row_id, message_id, thread_id, user_name, username, text) in enumerate(rows, start=1):
        legend[index] = _message_link(chat_id, message_id, thread_id)
        anchor_tag = " [СООБЩЕНИЕ, НА КОТОРОЕ ОТВЕЧАЛИ]" if row_id == anchor_row_id else ""
        lines.append(f"[{index}] {resolve_display_name(username, user_name)}: {decrypt(text)}{anchor_tag}")

    group_context = _group_context(chat_id)
    moments = search_moments(chat_id, embed(question), top_k=5)
    if moments:
        group_context += "\nВозможно релевантные моменты из истории:\n" + "\n".join(f"- {m}" for m in moments)
    prompt = prompt_for_qa.format(question=question, messages="\n".join(lines), group_context=group_context, asker_name=asker_name)
    answer = _content(get_chat_model("primary", 0.3).invoke(prompt, config=config_))
    return {"answer_plain": answer or None, "window_rows": rows}


def _cited_message_ids(answer_plain, window_rows):
    if not answer_plain or not window_rows:
        return []
    ids = []
    for match in re.finditer(r"\[(\d+)\]", answer_plain):
        n = int(match.group(1))
        if 1 <= n <= len(window_rows):
            ids.append(window_rows[n - 1][1])  # message_id
    return sorted(set(ids))


print("Loading full real corpus + running semantic pilot once...", file=sys.stderr)
_messages_all = load_messages(CHAT_ID)
_pilot_pred, _pilot_results, _tracker = semantic_pilot(_messages_all)
print(f"  {len(_messages_all)} messages loaded", file=sys.stderr)


def _make_task(variant):
    def task(*, item, **kwargs):
        case = item.input
        anchor_mid = case.get("replied_message_id")
        if not anchor_mid:
            return {"skip": True}

        if variant == "baseline":
            from research.eval_dataset_check import baseline_window
            bw = baseline_window(anchor_mid)
            if not bw:
                return {"skip": True}
            message_ids = [mid for mid, *_ in bw]
        else:
            conv_id = _pilot_pred.get(anchor_mid)
            if conv_id is None:
                return {"skip": True}
            message_ids = _tracker._episodes[conv_id].message_ids

        rows = _window_rows(CHAT_ID, message_ids, [anchor_mid])
        anchor_row_id = _anchor_row_id(CHAT_ID, anchor_mid)
        result = _generate_answer_for_window(
            CHAT_ID, case["question"], case["asker_name"], anchor_row_id, rows, _config(),
        )
        answer_plain = result["answer_plain"]
        window_rows = result["window_rows"]
        cited = _cited_message_ids(answer_plain, window_rows)
        return {
            "answer": answer_plain or "(нет ответа)",
            "anchor_id": anchor_row_id,
            "intent": None,
            "candidate_message_ids": [],
            "match_message_ids": [mid for _, mid, *_ in window_rows],
            "cited_message_ids": cited,
            "first_candidate_message_id": None,
            "window_size": len(window_rows),
        }
    return task


def _sync_comparison_dataset(langfuse):
    source = langfuse.get_dataset(SOURCE_DATASET_NAME)
    try:
        langfuse.create_dataset(
            name=COMPARISON_DATASET_NAME,
            description="Подмножество ask-pipeline-real-invocations: только anchor-кейсы, где окно "
                         "разрешимо и для baseline (production conversation_id), и для semantic pilot -- "
                         "остальные 17 не сравнимы (нет anchor'а вообще, либо anchor вне загруженного диапазона).",
        )
    except Exception:
        pass
    picked = [item for item in source.items if item.input.get("source_row_id") in RESOLVABLE_SOURCE_ROW_IDS]
    for item in picked:
        langfuse.create_dataset_item(
            dataset_name=COMPARISON_DATASET_NAME,
            id=f"anchor-cmp-{item.input['source_row_id']}",
            input=item.input,
        )
    return langfuse.get_dataset(COMPARISON_DATASET_NAME)


def run():
    langfuse = get_client()
    dataset = _sync_comparison_dataset(langfuse)
    print(f"Comparison dataset '{COMPARISON_DATASET_NAME}': {len(dataset.items)} items "
          f"(source_row_ids: {sorted(RESOLVABLE_SOURCE_ROW_IDS)})")

    results = {}
    for variant, run_name in [("baseline", "answer-quality-baseline-window"), ("pilot", "answer-quality-semantic-pilot-window")]:
        print(f"\n=== generating + judging: {variant} ===")
        result = dataset.run_experiment(
            name=run_name,
            description=f"Реальная генерация ответа /ask ({variant} window), оценено существующими judge'ами из tests/evals/judges.py.",
            task=_make_task(variant),
            evaluators=ALL_EVALUATORS,
            max_concurrency=2,
        )
        results[variant] = result
        print(f"  {len(result.item_results)} items processed")
        if getattr(result, "dataset_run_url", None):
            print(f"  {result.dataset_run_url}")

    print("\n=== summary ===")
    for variant, result in results.items():
        by_metric = {}
        for item_result in result.item_results:
            for ev in item_result.evaluations:
                if ev.value is not None:
                    by_metric.setdefault(ev.name, []).append(ev.value)
        print(f"  {variant}: " + ", ".join(f"{k}={sum(v)/len(v):.2f} (n={len(v)})" for k, v in by_metric.items()))


if __name__ == "__main__":
    run()
