"""Does memory even reach the candidate pool? Retrieval-only probe, no generation.

Runs the real /ask retrieval nodes (FTS, vector, memory, RRF) over mined production
questions with the question used as-is -- no rewrite, no rerank, no answer -- so it
costs embeddings and SQL only. Answers two questions the paired benchmark cannot
answer cheaply: how often memory is in the pool at all, and what it displaces.

    python3 tests/evals/memory_retrieval_probe.py [chat_id] [limit]
"""
import sys
from collections import Counter

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import llm.graphs as graphs
from episodic_memory import load_events, render_event
from tests.evals.mine_cases import mine_cases


def probe(chat_id=-1002335227490, limit=40, show=6):
    cases = mine_cases(chat_id, limit=limit)
    in_pool = by_participant = displaced = 0
    ranks = Counter()
    examples = []
    for case in cases:
        state = {"chat_id": chat_id, "effective_question": case["question"],
                 "asker_name": case["asker_name"], "use_episodes": True}
        state.update(graphs._search_fts(state))
        state.update(graphs._search_vector(state))
        vector_only = dict(state, use_episodes=False)
        state.update(graphs._search_memory(state))
        fused = graphs._fuse_rrf(state)
        baseline = graphs._fuse_rrf({**vector_only, "memory_ids": []})
        memory_ids = fused["candidate_event_ids"]
        in_pool += bool(memory_ids)
        displaced += len(set(baseline["candidate_ids"]) - set(fused["candidate_ids"]))
        participant = graphs.resolve_question_participant(chat_id, case["question"], case["asker_name"])
        by_participant += bool(participant)
        for event_id in memory_ids:
            ranks[state["memory_ids"].index(event_id) + 1] += 1
        if memory_ids and len(examples) < show:
            examples.append((case["question"], participant, load_events(memory_ids[:1])[0]))

    print(f"вопросов: {len(cases)}")
    print(f"память прошла фильтр и попала в пул: {in_pool}/{len(cases)}")
    print(f"вопросов с распознанным участником: {by_participant}/{len(cases)}")
    print(f"вытеснено сырых сообщений из топ-10: {displaced} суммарно "
          f"({displaced / max(len(cases), 1):.2f} на вопрос)")
    print(f"ранги попавших записей памяти: {dict(sorted(ranks.items()))}")
    for question, participant, view in examples:
        print(f"\n— {question}  (участник: {participant or 'не распознан'})")
        print("  " + render_event(view).replace("\n", "\n  "))


if __name__ == "__main__":
    probe(int(sys.argv[1]) if len(sys.argv) > 1 else -1002335227490,
          int(sys.argv[2]) if len(sys.argv) > 2 else 40)
