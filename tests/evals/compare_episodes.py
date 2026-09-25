"""Paired benchmark: /ask with RAW retrieval vs RAW+EPISODES, on real questions.

Follows docs/EPISODES_ROLLOUT_PROTOCOL.md, which fixed the metrics, the judges
and the decision rule before this ever ran. Read-only against production: no
Telegram sends, no writes back to `messages`, no memory writes.

Per case the shared prefix (anchor, classify+rewrite, FTS) is executed once and
reused by both conditions, so the only thing that differs is the episodic layer.

    python3 tests/evals/compare_episodes.py [chat_id] [limit] [--no-langfuse]

Writes a JSON + Markdown report under tmp/evals/ (gitignored: real chat content)
and, unless --no-langfuse, pushes two dataset runs with every score to Langfuse.
"""
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import ask_metrics
import llm.graphs as graphs
from llm.answer_judges import judge_answer, judge_pairwise
from tests.evals.judges import (
    agent_goal_evaluator,
    citation_hit_rate_evaluator,
    rerank_recall_evaluator,
    search_precision_evaluator,
)
from tests.evals.mine_cases import mine_cases
from tests.evals.run_eval import _NullBot, _cited_message_ids, _internal_ids_to_message_ids

CONDITIONS = ("raw", "episodes")
BOOTSTRAP, SEED = 10_000, 20260917
PIPELINE_JUDGES = (agent_goal_evaluator, search_precision_evaluator, rerank_recall_evaluator, citation_hit_rate_evaluator)
OUT_DIR = Path("/Users/yaroslavlikh/summary_sov/tmp/evals")


def _shared_prefix(case):
    """Everything both conditions have in common, computed once per case."""
    state = {
        "bot": _NullBot(), "chat_id": case["chat_id"], "question": case["question"],
        "asker_name": case["asker_name"], "replied_message_id": case["replied_message_id"],
        "bot_username": case["bot_username"], "thread_id": None,
        "save_bot_answer": lambda *a, **kw: None, "write_memory": lambda *a, **kw: None,
    }
    state.update(graphs._resolve_anchor(state))
    if state.get("anchor_id"):
        return None, "anchor"
    state.update(graphs._classify_and_rewrite(state, config={}))
    if state.get("handled_as_memory"):
        return None, "memory_command"
    state.update(graphs._search_fts(state))
    return state, None


def _run_condition(prefix_state, use_episodes):
    state = {**prefix_state, "use_episodes": use_episodes}
    started = time.perf_counter()
    state.update(graphs._search_vector(state))
    state.update(graphs._search_memory(state))
    state.update(graphs._fuse_rrf(state))
    state.update(graphs._rerank(state, config={}))
    state.update(graphs._generate_answer(state, config={}))
    return state, time.perf_counter() - started


def _judge_output(case, state):
    """The dict shape tests/evals/judges.py expects."""
    window_rows = state.get("window_rows") or []
    return {
        "answer": state.get("answer_plain") or "(нет ответа)",
        "intent": state.get("intent"),
        "candidate_message_ids": _internal_ids_to_message_ids(case["chat_id"], state.get("candidate_ids") or []),
        "match_message_ids": _internal_ids_to_message_ids(case["chat_id"], state.get("match_ids") or []),
        "cited_message_ids": _cited_message_ids(state.get("answer_plain"), window_rows),
        "first_candidate_message_id": (window_rows[0][1] if window_rows else None),
    }


def _scores_for(case, state, condition, latency):
    scores = {name: {"value": value, "comment": ""}
              for name, value in ask_metrics.collect(state, condition, latency).items()}
    scores.update(judge_answer(case["question"], case["asker_name"], state.get("answer_plain"),
                               state.get("context_lines") or [], state.get("episode_context") or ""))
    output = _judge_output(case, state)
    for judge in PIPELINE_JUDGES:
        try:
            result = judge(input=case, output=output)
        except Exception as error:  # a judge failing is not a zero for the pipeline
            print(f"    судья {judge.__name__} упал: {error}")
            continue
        for item in (result if isinstance(result, list) else [result]):
            if item and item.get("value") is not None:
                scores[item["name"]] = {"value": float(item["value"]), "comment": item.get("comment", "")}
    return scores


def run_case(case):
    prefix, skipped = _shared_prefix(case)
    if skipped:
        return {"case": case, "skipped": skipped}
    result = {"case": case, "conditions": {}}
    states = {}
    for condition in CONDITIONS:
        state, latency = _run_condition(prefix, condition == "episodes")
        states[condition] = state
        result["conditions"][condition] = {
            "answer": state.get("answer_plain"),
            "scores": _scores_for(case, state, condition, latency),
        }
    context = sorted(set((states["raw"].get("context_lines") or []) + (states["episodes"].get("context_lines") or [])))
    pairwise = judge_pairwise(case["question"], case["asker_name"], states["episodes"].get("answer_plain"),
                              states["raw"].get("answer_plain"), context)
    if pairwise:
        result["conditions"]["episodes"]["scores"]["pairwise_vs_raw"] = pairwise
    return result


def _bootstrap_ci(values):
    rnd = random.Random(SEED)
    n = len(values)
    means = sorted(sum(values[rnd.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP)]


def summarize(results):
    paired = {}
    for result in results:
        if result.get("skipped"):
            continue
        raw, episodes = (result["conditions"][c]["scores"] for c in CONDITIONS)
        for name in set(raw) & set(episodes):
            paired.setdefault(name, []).append(episodes[name]["value"] - raw[name]["value"])
    summary = {}
    for name, deltas in sorted(paired.items()):
        lo, hi = _bootstrap_ci(deltas) if len(deltas) > 1 else (float("nan"), float("nan"))
        summary[name] = {
            "n": len(deltas), "delta": statistics.mean(deltas), "ci": [lo, hi],
            "better": sum(1 for d in deltas if d > 1e-9), "worse": sum(1 for d in deltas if d < -1e-9),
            "raw_mean": statistics.mean([r["conditions"]["raw"]["scores"][name]["value"]
                                         for r in results if not r.get("skipped")
                                         and name in r["conditions"]["raw"]["scores"]]),
            "episodes_mean": statistics.mean([r["conditions"]["episodes"]["scores"][name]["value"]
                                              for r in results if not r.get("skipped")
                                              and name in r["conditions"]["episodes"]["scores"]]),
        }
    pairwise = [r["conditions"]["episodes"]["scores"]["pairwise_vs_raw"]["value"] for r in results
                if not r.get("skipped") and "pairwise_vs_raw" in r["conditions"]["episodes"]["scores"]]
    if pairwise:
        lo, hi = _bootstrap_ci(pairwise) if len(pairwise) > 1 else (float("nan"), float("nan"))
        summary["pairwise_vs_raw"] = {
            "n": len(pairwise), "delta": statistics.mean(pairwise), "ci": [lo, hi],
            "better": sum(1 for v in pairwise if v > 0), "worse": sum(1 for v in pairwise if v < 0),
            "raw_mean": 0.0, "episodes_mean": statistics.mean(pairwise),
        }
    return summary


def decide(summary):
    """The rule of docs/EPISODES_ROLLOUT_PROTOCOL.md, applied mechanically."""
    def delta(name):
        return summary[name]["delta"] if name in summary else 0.0

    blockers = [
        ("faithfulness упал", delta("faithfulness") < -0.05),
        ("выдуманных цитат стало больше", delta("citations_invented") > 0.05),
        ("чаще не отвечает", delta("no_answer") > 0.05),
        ("парное сравнение против эпизодов", delta("pairwise_vs_raw") < -0.1),
    ]
    reasons = [name for name, hit in blockers if hit]
    if reasons:
        return {"enable": False, "verdict": "блокировано: " + ", ".join(reasons)}
    if delta("pairwise_vs_raw") > 0.1:
        return {"enable": True, "verdict": "парное сравнение в пользу эпизодов"}
    if delta("context_sufficiency") > 0.05:
        return {"enable": True, "verdict": "доставленного контекста стало достаточно чаще"}
    return {"enable": False, "verdict": "улучшение не подтверждено, флаг остаётся выключенным"}


def _markdown(summary, decision, results, memory, chat_id):
    skipped = [r["skipped"] for r in results if r.get("skipped")]
    lines = [
        f"# RAW vs RAW+EPISODES, чат {chat_id}", "",
        f"Кейсов сравнено: {sum(1 for r in results if not r.get('skipped'))}; "
        f"пропущено anchor: {skipped.count('anchor')}, memory_command: {skipped.count('memory_command')}.",
        f"Память чата: {memory}", "",
        f"**Решение: {decision['verdict']}** (флаг {'включаем' if decision['enable'] else 'остаётся выключенным'})", "",
        "| метрика | RAW | EPISODES | Δ | 95% CI | лучше/хуже | n |", "|---|---:|---:|---:|---|---:|---:|",
    ]
    for name, row in summary.items():
        lines.append(f"| {name} | {row['raw_mean']:.3f} | {row['episodes_mean']:.3f} | {row['delta']:+.3f} | "
                     f"[{row['ci'][0]:+.3f}; {row['ci'][1]:+.3f}] | {row['better']}/{row['worse']} | {row['n']} |")
    return "\n".join(lines) + "\n"


def _push_to_langfuse(results, chat_id, stamp):
    """Record the finished comparison as two dataset runs. Nothing is generated here:
    the task and the evaluators replay what this script already computed, so the runs
    in Langfuse are the same numbers as the local report."""
    from langfuse import get_client

    client = get_client()
    dataset_name = "ask-episodes-comparison"
    try:
        client.create_dataset(name=dataset_name, description="Парное сравнение RAW и RAW+EPISODES на реальных вопросах.")
    except Exception:
        pass
    by_id = {}
    for result in results:
        if result.get("skipped"):
            continue
        # Item ids are unique per project across datasets in Langfuse, and the /ask
        # eval dataset already owns the plain row ids, so these are namespaced.
        item_id = f"episodes-row-{result['case']['source_row_id']}"
        by_id[item_id] = result
        client.create_dataset_item(dataset_name=dataset_name, id=item_id, input=result["case"],
                                   metadata={"chat_id": chat_id})
    dataset = client.get_dataset(dataset_name)

    for condition in CONDITIONS:
        # The dataset can also hold items from earlier runs (ids are per project and
        # were namespaced later); those are skipped rather than failing the upload.
        def task(*, item, **_kwargs):
            result = by_id.get(item.id)
            return (result["conditions"][condition]["answer"] or "(нет ответа)") if result else "(не в этом прогоне)"

        def evaluator(*, input, output, metadata=None, **_kwargs):
            result = by_id.get(f"episodes-row-{input.get('source_row_id')}") if isinstance(input, dict) else None
            if not result:
                return []
            return [{"name": name, "value": float(score["value"]), "comment": str(score.get("comment") or "")[:500]}
                    for name, score in result["conditions"][condition]["scores"].items()]

        dataset.run_experiment(
            name=f"{condition}-{stamp}", task=task, evaluators=[evaluator], max_concurrency=4,
            description=f"/ask, условие {condition}; docs/EPISODES_ROLLOUT_PROTOCOL.md",
        )
    client.flush()
    return f"{dataset_name}: прогоны raw-{stamp} и episodes-{stamp}"


def main(chat_id=-1002335227490, limit=40, use_langfuse=True):
    import episodic_memory
    from embeddings import warm_up

    warm_up()  # load the embedding model once, before the workers start racing for it

    memory = episodic_memory.memory_stats(chat_id)
    if not memory["episodes"]:
        print("В этом чате ещё нет эпизодов — сначала построй память (episodic_memory.run_for_chat).")
        return
    cases = mine_cases(chat_id, limit=limit)
    workers = int(os.getenv("EVAL_WORKERS", "4"))
    print(f"Кейсов замайнено: {len(cases)}; память: {memory}; параллельно: {workers}")
    # Cases are independent (each replays its own question), so they run in parallel;
    # inside a case everything stays sequential and paired. Groq's limit is 250k
    # tokens/minute, far above what a handful of workers uses.
    done = [0]

    def run_logged(index_case):
        index, case = index_case
        try:
            result = run_case(case)
        except Exception as error:
            print(f"[{index}/{len(cases)}] кейс упал: {error}")
            return None
        done[0] += 1
        print(f"[{done[0]}/{len(cases)}] {result.get('skipped') or 'ok'}: {case['question'][:60]}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [r for r in pool.map(run_logged, enumerate(cases, start=1)) if r is not None]

    summary = summarize(results)
    decision = decide(summary)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"chat_id": chat_id, "stamp": stamp, "memory": memory, "summary": summary,
              "decision": decision, "results": results}
    (OUT_DIR / f"episodes_compare_{stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    markdown = _markdown(summary, decision, results, memory, chat_id)
    (OUT_DIR / f"episodes_compare_{stamp}.md").write_text(markdown)
    print("\n" + markdown)
    if use_langfuse:
        try:
            print(_push_to_langfuse(results, chat_id, stamp))
        except Exception as error:
            print(f"Не смог выложить прогон в Langfuse: {error}")
    print(f"Отчёт: {OUT_DIR}/episodes_compare_{stamp}.{{json,md}}")


def push_saved(report_path):
    """Upload a finished report to Langfuse (used when the push failed after a run)."""
    report = json.loads(open(report_path).read())
    print(_push_to_langfuse(report["results"], report["chat_id"], report["stamp"]))


if __name__ == "__main__":
    if "--push" in sys.argv:
        push_saved(sys.argv[sys.argv.index("--push") + 1])
        raise SystemExit(0)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(args[0]) if args else -1002335227490,
         int(args[1]) if len(args) > 1 else 40,
         "--no-langfuse" not in sys.argv)
