"""Pushes the baseline-vs-pilot conversation-window comparison to Langfuse as
two real dataset runs (against the EXISTING ask-pipeline-real-invocations
dataset, no new items created) so they show up in the UI's run-compare view,
same as the judge-metric runs.

No LLM calls at all -- scoring is a deterministic, hand-verified
"does this window still contain a message we know is contamination"
check, built from the exact same real message_ids documented in
docs/eval_incidents.md #2 (self-pollution) and #8 (Лизка). Never touches
/ask, run_ask_graph, or anything that could execute a memory_command --
only SQL SELECT (baseline window) and the pure local pilot module.

Usage: python3 -m research.langfuse_disentanglement_run
"""
import sys

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import config
from database.db import get_conn
from langfuse import get_client

from research.conversation_disentanglement import semantic_pilot
from research.db_loader import load_messages
from research.eval_dataset_check import baseline_window

CHAT_ID = -1002335227490
DATASET_NAME = "ask-pipeline-real-invocations"

# Known-bad message_ids per case (source_row_id -> set), hand-verified against
# the real decrypted content -- see docs/eval_incidents.md #2 and #8, and
# research/REPORT.md section 10 for exactly how these were identified.
KNOWN_BAD_IDS = {
    # #8 Лизка: unrelated RAG/routing tech-discussion tangent that has
    # nothing to do with "а она существует?" but sits in the raw window.
    46173: {72465, 72466, 72467, 72468, 72469, 72470, 72471, 72472, 72473,
             72474, 72475, 72476, 72477, 72478, 72479, 72480, 72481, 72482, 72483},
    # #2 self-pollution (72718/72729) + #1 Наполовину (72743/72744) --
    # both anchor into the same real conversation_id=72717 blob.
    46416: {72718, 72729, 72743, 72744},
    46434: {72718, 72729, 72743, 72744},
}

print("Loading full real corpus + running semantic pilot once...", file=sys.stderr)
_messages = load_messages(CHAT_ID)
_pilot_pred, _pilot_results, _tracker = semantic_pilot(_messages)
_by_id = {m.message_id: m for m in _messages}
print(f"  {len(_messages)} messages loaded", file=sys.stderr)


def _baseline_task(*, item, **kwargs):
    case = item.input
    anchor_mid = case.get("replied_message_id")
    if not anchor_mid:
        return {"skip": True, "reason": "no reply anchor -- conversation_id not used for this case"}
    window = baseline_window(anchor_mid)
    if not window:
        return {"skip": True, "reason": "anchor not resolvable in production (message not found)"}
    return {
        "variant": "baseline_time_gap",
        "window_size": len(window),
        "message_ids": [mid for mid, *_ in window],
        "sample": " | ".join(text[:40] for _, _, text, _ in window[:5]),
    }


def _filter_is_bot(chat_id, message_ids, anchor_message_ids):
    # Mirrors llm/graphs.py _generate_answer's is_bot filter -- a row is
    # only allowed in if it's genuinely content, or is one of the anchors
    # itself. Missing this earlier let bot rows leak into the pilot window,
    # confounding the clustering-method comparison with an unrelated
    # content-filter difference -- fixed after review.
    if not message_ids:
        return []
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT message_id FROM messages WHERE user_id = %s AND message_id = ANY(%s) "
            "AND (is_bot = FALSE OR message_id = ANY(%s))",
            (chat_id, list(message_ids), list(anchor_message_ids)),
        )
        allowed = {r[0] for r in cur.fetchall()}
    return [mid for mid in message_ids if mid in allowed]


def _pilot_task(*, item, **kwargs):
    case = item.input
    anchor_mid = case.get("replied_message_id")
    if not anchor_mid:
        return {"skip": True, "reason": "no reply anchor -- conversation_id not used for this case"}
    conv_id = _pilot_pred.get(anchor_mid)
    if conv_id is None:
        return {"skip": True, "reason": "anchor not in loaded corpus (no message_date/embedding)"}
    episode = _tracker._episodes[conv_id]
    ids = _filter_is_bot(CHAT_ID, episode.message_ids, [anchor_mid])
    return {
        "variant": "semantic_pilot",
        "window_size": len(ids),
        "message_ids": ids,
        "sample": " | ".join((_by_id[m].text[:40] if m in _by_id else "?") for m in ids[:5]),
    }


def contamination_avoided_evaluator(*, input, output, expected_output=None, **kwargs):
    if output.get("skip"):
        return []
    bad = KNOWN_BAD_IDS.get(input.get("source_row_id"))
    if not bad:
        return []  # no hand-verified ground truth for this case -- don't fake a score
    present = set(output.get("message_ids") or []) & bad
    score = round(1 - len(present) / len(bad), 2)
    return {
        "name": "contamination_avoided",
        "value": score,
        "comment": f"known-bad ids still in window: {sorted(present) or 'none'}",
    }


def window_size_evaluator(*, input, output, expected_output=None, **kwargs):
    if output.get("skip"):
        return []
    return {"name": "window_size", "value": output["window_size"], "comment": ""}


def run():
    langfuse = get_client()
    dataset = langfuse.get_dataset(DATASET_NAME)

    print("\n=== running baseline (production time-gap heuristic) ===")
    baseline_result = dataset.run_experiment(
        name="conversation-window-baseline",
        description="Реальное окно /ask _generate_answer прямо сейчас (conversation_id / +-3 fallback), read-only, без LLM.",
        task=_baseline_task,
        evaluators=[contamination_avoided_evaluator, window_size_evaluator],
        max_concurrency=1,
    )
    print(f"  {len(baseline_result.item_results)} items processed")
    if getattr(baseline_result, "dataset_run_url", None):
        print(f"  {baseline_result.dataset_run_url}")

    print("\n=== running semantic pilot (offline, read-only) ===")
    pilot_result = dataset.run_experiment(
        name="conversation-window-semantic-pilot",
        description="research/conversation_disentanglement.py эпизод для того же anchor'а, read-only, без LLM.",
        task=_pilot_task,
        evaluators=[contamination_avoided_evaluator, window_size_evaluator],
        max_concurrency=1,
    )
    print(f"  {len(pilot_result.item_results)} items processed")
    if getattr(pilot_result, "dataset_run_url", None):
        print(f"  {pilot_result.dataset_run_url}")

    print("\n=== summary ===")
    for name, result in [("baseline", baseline_result), ("pilot", pilot_result)]:
        by_metric = {}
        for item_result in result.item_results:
            for ev in item_result.evaluations:
                if ev.value is not None:
                    by_metric.setdefault(ev.name, []).append(ev.value)
        print(f"  {name}: " + ", ".join(
            f"{k}={sum(v)/len(v):.2f} (n={len(v)})" for k, v in by_metric.items()
        ))


if __name__ == "__main__":
    run()
