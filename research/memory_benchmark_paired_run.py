"""Paired baseline vs oracle-memory /ask benchmark -- step 3 of the plan:
(1) oracle retrieval [done], (2) memory-dependent question dataset [drafted,
44 questions], (3) this: paired run + judges, (4) no-regression check via
the existing ask-pipeline-eval-run [run separately, after this].

Sets up the sandbox and both graphs ONCE (research.oracle_memory_eval's
setup_sandbox / build_ask_graph_baseline / build_ask_graph_oracle -- not
rebuilt here), then runs the SAME 44 questions through both, pushed to
Langfuse as two named dataset-run experiments against one shared dataset
(same items, so the Langfuse UI can diff them side by side):

  - memory-benchmark-baseline
  - memory-benchmark-oracle

Two new judges (not the standing tests/evals/judges.py -- those score the
production ask-pipeline-eval-run against raw message windows; these score
against the FROZEN gold claim directly, which the standing judges don't
have access to):

  - memory_correctness: does the answer convey the SAME content as the
    known gold claim (not verbatim -- semantically)?
  - memory_attribution: only scored when the fact is fundamentally about
    someone's VIEW (kind=opinion) or is itself a reported_by_other claim --
    does the answer preserve that it's an attributed view, not present it
    as unattributed objective fact?

Also saves a local paired JSON (question, gold claim, both answers, oracle
facts retrieved) for direct inspection outside Langfuse -- real chat
content, /tmp only, not committed.

Per explicit instruction: questions are NOT polished/filtered before this
run -- it's the first diagnostic pass, and cases can't be cherry-picked out
after seeing results. The standing ask-pipeline-eval-run (production /ask,
unrelated to this memory experiment) should be run separately AFTER this,
purely as a no-regression check that adding memory-retrieval code paths
didn't change production's real behavior -- oracle memory is never wired
into that graph, so a regression there would mean something else broke.

Usage: python3 -m research.memory_benchmark_paired_run
"""
import json
import sys

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

from langfuse import get_client

from research.oracle_memory_eval import ask, build_ask_graph_baseline, build_ask_graph_oracle, setup_sandbox
from tests.evals.judges import _judge_llm

QUESTIONS_PATH = "/tmp/memory_questions_draft.jsonl"
GOLD_PATH = "/tmp/gold_facts.jsonl"
DATASET_NAME = "memory-benchmark-questions"
LOCAL_OUT_PATH = "/tmp/memory_benchmark_paired_results.json"
ASKER_NAME = "Ярик Лихачев"


# --------------------------------------------------------------- judges --

def memory_correctness_evaluator(*, input, output, expected_output=None, **kwargs):
    gold_claim = expected_output or input.get("gold_claim")
    answer = output.get("answer") or "(нет ответа)"
    prompt = f"""Известный факт (проверен человеком, это истина для этого чата): {gold_claim}

Вопрос, заданный боту: {input['question']}
Ответ бота: {answer}

Оцени от 0.0 до 1.0 с шагом 0.1: насколько ответ бота корректно передаёт СОДЕРЖАНИЕ известного факта (не дословно -- по смыслу, та же информация). 0.0 -- бот не знает / ответил неверно / ответил о другом. 1.0 -- ответ полностью и верно передаёт факт (перефразировка нормальна, если смысл сохранён).

Верни ТОЛЬКО JSON: {{"score": 0.0-1.0, "comment": "почему"}}"""
    result = _judge_llm(prompt)
    if not result:
        return []
    return {"name": "memory_correctness", "value": float(result.get("score", 0.0)), "comment": result.get("comment", "")}


def memory_attribution_evaluator(*, input, output, expected_output=None, **kwargs):
    """Only meaningful when the fact IS someone's view (opinion) or is
    itself a reported_by_other claim -- for a plain self-reported state/
    event fact, attribution isn't the point being tested."""
    kind = input.get("kind")
    epistemic_status = input.get("epistemic_status")
    if kind != "opinion" and epistemic_status != "reported_by_other":
        return []
    gold_claim = expected_output or input.get("gold_claim")
    answer = output.get("answer") or "(нет ответа)"
    subject = input.get("subject") or "(неизвестно)"
    prompt = f"""Известный факт: {gold_claim}
Это МНЕНИЕ/УТВЕРЖДЕНИЕ конкретного человека ({subject}), а не объективная истина.

Ответ бота на вопрос "{input['question']}": {answer}

Оцени: сохраняет ли ответ бота атрибуцию -- явно указывает, что это МНЕНИЕ/СЛОВА {subject} (например "по словам X", "X считает", "X написал, что..."), а не преподносит содержание как объективный факт без указания источника?

Если бот вообще не ответил (нет ответа/не нашёл) -- считай, что атрибуция не потеряна (нечего терять), но и не подтверждена: верни attribution_preserved=false с комментарием "нет ответа".

Верни ТОЛЬКО JSON: {{"attribution_preserved": true/false, "comment": "почему"}}"""
    result = _judge_llm(prompt)
    if not result:
        return []
    return {
        "name": "memory_attribution",
        "value": 1.0 if result.get("attribution_preserved") else 0.0,
        "comment": result.get("comment", ""),
    }


MEMORY_EVALUATORS = [memory_correctness_evaluator, memory_attribution_evaluator]


# --------------------------------------------------------------- dataset --

def _load_dataset_items():
    questions = [json.loads(line) for line in open(QUESTIONS_PATH)]
    gold_by_id = {g["id"]: g for g in (json.loads(line) for line in open(GOLD_PATH))}
    items = []
    for q in questions:
        if not q.get("question"):
            print(f"WARNING: question #{q['id']} has no drafted question, skipping", file=sys.stderr)
            continue
        gold = gold_by_id.get(q["id"], {})
        items.append({
            "id": q["id"],
            "question": q["question"],
            "gold_claim": q["gold_claim"],
            "kind": q["kind"],
            "subject": q["subject"],
            "epistemic_status": gold.get("epistemic_status"),
            "source_message_ids": q["source_message_ids"],
            "asker_name": ASKER_NAME,
        })
    return items


def _sync_dataset(langfuse, items):
    try:
        langfuse.create_dataset(
            name=DATASET_NAME,
            description="44 memory-dependent questions, one per frozen gold positive fact -- first diagnostic pass, not filtered.",
        )
    except Exception:
        pass
    for item in items:
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=f"memory-q-{item['id']}",
            input=item,
            expected_output=item["gold_claim"],
            metadata={"kind": item["kind"], "subject": item["subject"]},
        )


# ----------------------------------------------------------------- task --

def _make_task(graph, local_bucket):
    def task(*, item, **kwargs):
        case = item.input
        result = ask(graph, case["question"], asker_name=case.get("asker_name", ASKER_NAME))
        entry = {
            "id": case["id"],
            "question": case["question"],
            "gold_claim": case["gold_claim"],
            "answer": result.get("answer_plain") or "(нет ответа)",
            "oracle_facts_used": result.get("oracle_facts_used") or [],
        }
        local_bucket.append(entry)
        return {"answer": entry["answer"], "oracle_facts_used": entry["oracle_facts_used"]}
    return task


# ------------------------------------------------------------------ run --

def run():
    print("Setting up sandbox (real data, read-only copy) + both graphs...", file=sys.stderr)
    setup_sandbox()
    baseline_graph = build_ask_graph_baseline()
    oracle_graph = build_ask_graph_oracle()

    items = _load_dataset_items()
    print(f"{len(items)} questions loaded (not filtered -- first diagnostic pass)", file=sys.stderr)

    langfuse = get_client()
    _sync_dataset(langfuse, items)
    dataset = langfuse.get_dataset(DATASET_NAME)

    baseline_local, oracle_local = [], []

    print("\nRunning baseline (production /ask, no oracle memory)...", file=sys.stderr)
    baseline_result = dataset.run_experiment(
        name="memory-benchmark-baseline",
        description="Paired memory benchmark -- baseline (no oracle memory injection).",
        task=_make_task(baseline_graph, baseline_local),
        evaluators=MEMORY_EVALUATORS,
        max_concurrency=3,
    )

    print("\nRunning oracle (production /ask + oracle memory injection)...", file=sys.stderr)
    oracle_result = dataset.run_experiment(
        name="memory-benchmark-oracle",
        description="Paired memory benchmark -- oracle memory (44 frozen gold facts) injected into generate_answer.",
        task=_make_task(oracle_graph, oracle_local),
        evaluators=MEMORY_EVALUATORS,
        max_concurrency=3,
    )

    by_id_baseline = {e["id"]: e for e in baseline_local}
    by_id_oracle = {e["id"]: e for e in oracle_local}
    paired = []
    for item in items:
        b = by_id_baseline.get(item["id"], {})
        o = by_id_oracle.get(item["id"], {})
        paired.append({
            "id": item["id"], "question": item["question"], "gold_claim": item["gold_claim"],
            "kind": item["kind"], "subject": item["subject"],
            "baseline_answer": b.get("answer"), "oracle_answer": o.get("answer"),
            "oracle_facts_used": o.get("oracle_facts_used") or [],
        })
    with open(LOCAL_OUT_PATH, "w") as f:
        json.dump(paired, f, ensure_ascii=False, indent=2)
    print(f"\nLocal paired results: {LOCAL_OUT_PATH}", file=sys.stderr)

    def _summarize(result, label):
        by_metric = {}
        for item_result in result.item_results:
            for ev in item_result.evaluations:
                if ev.value is None:
                    continue
                by_metric.setdefault(ev.name, []).append(ev.value)
        print(f"\n{label}:", file=sys.stderr)
        for name, values in by_metric.items():
            print(f"  {name}: {sum(values) / len(values):.2f} (n={len(values)})", file=sys.stderr)
        if getattr(result, "dataset_run_url", None):
            print(f"  Langfuse: {result.dataset_run_url}", file=sys.stderr)

    _summarize(baseline_result, "memory-benchmark-baseline")
    _summarize(oracle_result, "memory-benchmark-oracle")


if __name__ == "__main__":
    run()
