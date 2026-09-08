"""Compare baseline vs speed-optimized /ask graph variants on the SAME real
cases: latency (isolated, sequential, no evaluator contention) + quality
(same 4 judges as run_eval.py, called directly). Not pushed to Langfuse --
this is a fast internal A/B pass; the winning variant gets a proper
Langfuse-tracked run afterward via run_eval.py.
"""
import re
import sys
import time

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

from database.db import get_conn
from llm.graphs import run_ask_graph, run_ask_graph_merged, run_ask_graph_merged_skip_rerank
from tests.evals.judges import (
    agent_goal_evaluator,
    citation_hit_rate_evaluator,
    rerank_recall_evaluator,
    search_precision_evaluator,
)
from tests.evals.mine_cases import mine_cases
from tests.evals.run_eval import _NullBot, _cited_message_ids, _internal_ids_to_message_ids

VARIANTS = {
    "baseline (classify+rewrite separate, full rerank)": run_ask_graph,
    "A: merged classify+rewrite": run_ask_graph_merged,
    "A+B: merged + skip rerank <=2 candidates": run_ask_graph_merged_skip_rerank,
}

EVALUATORS = [agent_goal_evaluator, search_precision_evaluator, rerank_recall_evaluator, citation_hit_rate_evaluator]


def run_one(graph_fn, case):
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
    final = graph_fn(state)
    elapsed = time.perf_counter() - started

    candidate_message_ids = _internal_ids_to_message_ids(case["chat_id"], final.get("candidate_ids") or [])
    match_message_ids = _internal_ids_to_message_ids(case["chat_id"], final.get("match_ids") or [])
    window_rows = final.get("window_rows") or []
    cited_message_ids = _cited_message_ids(final.get("answer_plain"), window_rows)

    output = {
        "answer": final.get("answer_plain") or "(нет ответа)",
        "candidate_message_ids": candidate_message_ids,
        "match_message_ids": match_message_ids,
        "cited_message_ids": cited_message_ids,
        "first_candidate_message_id": candidate_message_ids[0] if candidate_message_ids else None,
    }
    return elapsed, output


def compare(chat_id=-1002335227490, n_cases=8):
    cases = mine_cases(chat_id, limit=n_cases)
    print(f"Сравниваю {len(VARIANTS)} варианта на {len(cases)} реальных кейсах...\n")

    for label, graph_fn in VARIANTS.items():
        print(f"=== {label} ===")
        latencies = []
        metric_values = {}
        for case in cases:
            elapsed, output = run_one(graph_fn, case)
            latencies.append(elapsed)
            for ev_fn in EVALUATORS:
                result = ev_fn(input=case, output=output)
                results = result if isinstance(result, list) else [result]
                for r in results:
                    if r and r.get("value") is not None:
                        metric_values.setdefault(r["name"], []).append(r["value"])

        avg_latency = sum(latencies) / len(latencies)
        print(f"  latency: avg={avg_latency:.2f}s  min={min(latencies):.2f}s  max={max(latencies):.2f}s")
        for name, values in metric_values.items():
            print(f"  {name}: {sum(values) / len(values):.2f} (n={len(values)})")
        print()


if __name__ == "__main__":
    chat_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1002335227490
    n_cases = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    compare(chat_id, n_cases)
