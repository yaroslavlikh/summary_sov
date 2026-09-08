"""Clean, isolated latency measurement for the /ask graph -- sequential, no
evaluators, nothing else competing for the Groq account. This is what
run_eval.py's latency_seconds CANNOT give you cleanly, since evaluators for
other concurrent items share the same rate limit there.
"""
import sys
import time

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

from llm.graphs import run_ask_graph
from tests.evals.mine_cases import mine_cases
from tests.evals.run_eval import _NullBot


def measure(chat_id=-1002335227490, limit=15):
    cases = mine_cases(chat_id, limit=limit)
    if not cases:
        print("Нет реальных кейсов для замера.")
        return

    times = []
    for case in cases:
        state = {
            "bot": _NullBot(),
            "chat_id": case["chat_id"],
            "question": case["question"],
            "asker_name": case["asker_name"],
            "replied_message_id": case["replied_message_id"],
            "bot_username": case["bot_username"],
            "thread_id": None,
            "save_bot_answer": lambda *a, **kw: None,
        }
        started = time.perf_counter()
        run_ask_graph(state)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        anchor_note = "anchor" if case["replied_message_id"] else "no-anchor"
        print(f"  {elapsed:5.2f}s  [{anchor_note}]  {case['question']!r}")

    times.sort()
    n = len(times)
    avg = sum(times) / n
    p50 = times[n // 2]
    p95 = times[min(n - 1, int(n * 0.95))]
    print(f"\nn={n}  avg={avg:.2f}s  p50={p50:.2f}s  p95={p95:.2f}s  min={times[0]:.2f}s  max={times[-1]:.2f}s")


if __name__ == "__main__":
    chat_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1002335227490
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    measure(chat_id, limit)
