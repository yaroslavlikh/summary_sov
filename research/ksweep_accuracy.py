"""k-sweep, phase 2: accuracy at k=20 for RAW and RAW+EVENTS on a stratified sample.

--dry-run makes no API call and constructs no API client. It rebuilds the real k=20 answer
prompts from the sealed phase-1 retrieval, counts their tokens, calibrates the chat-format
overhead and the mean output length against the sealed k=10 usage, and picks the largest
stratified sample whose estimated cost fits the target in research/ksweep_phase2_config.json.
The sample order is derived from the seed and written before any paid call exists.

--run freezes the approved sample in phase2/manifest.json before the first call, then answers
and judges with the official models, prompts, parameters and scoring, unchanged. Every paid
request passes through BudgetGuard, which wraps the client's create method: before a call it
checks spent + in-flight reservations + a conservative estimate of this call against the hard
guard, and afterwards books the actual cost from response.usage. Answers and judge verdicts
are cached by payload hash, so a resumed run never pays twice for a completed call.

    python3 -m research.ksweep_accuracy --dry-run
    python3 -m research.ksweep_accuracy --run          # only after explicit approval
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from research import evermembench_episode_run as base
from research import evermembench_parallel_resume as runner
from research import evermembench_temporal_budget_control as control
from research import final_sprint_evermembench as fs
from research import paper_benchmark_common as common
from research import ksweep_delivery as delivery
from research.ksweep_delivery import A_RUN, CATEGORY_NAMES
from research.paper_benchmark_common import (
    append_jsonl, exact_mcnemar, jsonl, log, seal_jsonl, sha256_bytes, sha256_file, verify_sealed_jsonl,
)

ROOT = fs.ROOT
PHASE1 = ROOT / ".research_runs" / "ksweep_v1_20260917_144836_MSK"
PHASE2 = PHASE1 / "phase2"
CONFIG = ROOT / "research" / "ksweep_phase2_config.json"
SEALED_RUN = {"RAW": fs.SOURCE, "RAW+EVENTS": A_RUN}
ANSWER_MODEL, JUDGE_MODEL = fs.base.ANSWER_MODEL, fs.base.JUDGE_MODEL


def usd(tokens_in: float, tokens_out: float, prices: dict) -> float:
    return (tokens_in * prices["input"] + tokens_out * prices["output"]) / 1_000_000


def allocation(n: int, sizes: dict[str, int]) -> dict[str, int]:
    """Proportional allocation by largest remainder; ties broken by category name."""
    total = sum(sizes.values())
    quotas = {name: n * size / total for name, size in sizes.items()}
    counts = {name: int(q) for name, q in quotas.items()}
    order = sorted(sizes, key=lambda name: (-(quotas[name] - counts[name]), name))
    for name in order[: n - sum(counts.values())]:
        counts[name] += 1
    return counts


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def dry_run() -> None:
    config = json.loads(CONFIG.read_text())
    k, k_ref = config["k"], config["k_reference"]
    conditions = config["conditions"]
    prices = config["prices_usd_per_million_tokens"]
    fee = config["fee_factor"]
    PHASE2.mkdir(parents=True, exist_ok=True)

    verify_sealed_jsonl(PHASE1 / "per_question.jsonl", PHASE1 / "per_question_meta.json", expected_rows=14400)
    phase1 = {(r["qa_key"], r["condition"], r["k"]): r for r in jsonl(PHASE1 / "per_question.jsonl")}
    data, _checks = fs.load_data()
    questions = {row["qa_key"]: row for row in data.questions}

    import tiktoken
    encoder = tiktoken.get_encoding(control.TOKEN_ENCODING)

    prompts: dict[tuple[str, str, int], dict] = {}
    token_mismatch = 0
    for qa, question in questions.items():
        for condition in conditions:
            for depth in (k_ref, k):
                cell = phase1[(qa, condition, depth)]
                docs = [data.docs_by_id[doc_id] for doc_id in cell["doc_ids"]]
                context = control.render_context(docs)
                if depth == k and len(encoder.encode(context)) != cell["context_tokens"]:
                    token_mismatch += 1
                prompt = fs.build_prompt(data.prompts, question, context)
                prompts[(qa, condition, depth)] = {
                    "key": fs.chat_key(prompt, fs.ANSWER_MAX_TOKENS),
                    "prompt_tokens": len(encoder.encode(prompt)),
                    "question_type": question["question_type"],
                }
    if token_mismatch:
        raise RuntimeError(f"{token_mismatch} rebuilt k={k} contexts differ from phase 1 token counts")

    # Calibration on sealed k=reference answer calls: chat overhead and output length.
    overhead: list[int] = []
    outputs: dict[tuple[str, str], list[int]] = defaultdict(list)
    matched = defaultdict(int)
    for condition in conditions:
        by_key = {prompts[(qa, condition, k_ref)]["key"]: prompts[(qa, condition, k_ref)] for qa in questions}
        for row in jsonl(SEALED_RUN[condition] / "api_usage.jsonl"):
            if row.get("phase") != "answer" or row["key"] not in by_key:
                continue
            item = by_key[row["key"]]
            overhead.append(row["input_tokens"] - item["prompt_tokens"])
            outputs[(condition, item["question_type"])].append(row["output_tokens"])
            matched[condition] += 1
    judge_calls: list[dict] = []
    for condition in conditions:
        judge_calls += [row for row in jsonl(SEALED_RUN[condition] / "api_usage.jsonl")
                        if row.get("phase") == "judge" and str(row.get("key", "")).startswith("judge:")]
    attempts_per_key = defaultdict(int)
    for row in judge_calls:
        attempts_per_key[row["key"]] += 1
    judge = {
        "paid_calls": len(judge_calls), "judged_rows": len(attempts_per_key),
        "mean_attempts": len(judge_calls) / len(attempts_per_key),
        "input_mean": float(np.mean([r["input_tokens"] for r in judge_calls])),
        "output_mean": float(np.mean([r["output_tokens"] for r in judge_calls])),
        "input_max": max(r["input_tokens"] for r in judge_calls),
        "output_max": max(r["output_tokens"] for r in judge_calls),
        "output_p95": percentile([r["output_tokens"] for r in judge_calls], 95),
    }
    calibration = {
        "answer_calls_matched": dict(matched),
        "overhead_tokens": {"mean": float(np.mean(overhead)), "min": min(overhead), "max": max(overhead)},
        "answer_output_tokens": {f"{c} / {t}": {"n": len(v), "mean": float(np.mean(v)), "p95": percentile(v, 95),
                                                "max": max(v)} for (c, t), v in sorted(outputs.items())},
        "judge": judge,
    }
    overhead_mean = calibration["overhead_tokens"]["mean"]
    judge_unit = fee * judge["mean_attempts"] * usd(judge["input_mean"], judge["output_mean"], prices[JUDGE_MODEL])

    # Per-question estimated cost at k, both conditions; a prompt already in a sealed cache is free.
    cost: dict[str, float] = {}
    cache_hits = 0
    for qa, question in questions.items():
        total = 0.0
        for condition in conditions:
            item = prompts[(qa, condition, k)]
            if item["key"] in data.cache:
                cache_hits += 1
            else:
                out = np.mean(outputs[(condition, item["question_type"])])
                total += fee * usd(item["prompt_tokens"] + overhead_mean, out, prices[ANSWER_MODEL])
            if item["question_type"] != "multiple_choice":
                total += judge_unit
        cost[qa] = total

    # Seeded order inside each category, fixed before any call and independent of cost.
    seed = config["sample"]["seed"]
    rng = np.random.default_rng(seed)
    by_category: dict[str, list[str]] = defaultdict(list)
    for qa, question in questions.items():
        by_category[CATEGORY_NAMES[(question["major"], question["minor"])]].append(qa)
    order = {name: [str(x) for x in rng.permutation(sorted(by_category[name]))] for name in sorted(by_category)}
    sizes = {name: len(keys) for name, keys in order.items()}
    order_record = {"seed": seed, "design": config["sample"]["design"], "category_sizes": sizes, "order": order}
    order_path = PHASE2 / "sample_order.json"
    order_text = json.dumps(order_record, indent=2, ensure_ascii=False) + "\n"
    if order_path.exists() and order_path.read_text() != order_text:
        raise RuntimeError("sample_order.json already exists and differs; refusing to redraw")
    order_path.write_text(order_text)

    def sample(n: int) -> list[str]:
        counts = allocation(n, sizes)
        return [qa for name in sorted(order) for qa in order[name][: counts[name]]]

    target = config["budget_usd"]["estimate_target"]
    curve = {n: sum(cost[qa] for qa in sample(n)) for n in range(1, len(questions) + 1)}
    n_max = max(n for n, value in curve.items() if value <= target)
    chosen = sample(n_max)
    counts = allocation(n_max, sizes)
    open_ended = sum(1 for qa in chosen if questions[qa]["question_type"] != "multiple_choice")

    answer_in = sum(fee * usd(prompts[(qa, c, k)]["prompt_tokens"] + overhead_mean, 0, prices[ANSWER_MODEL])
                    for qa in chosen for c in conditions)
    answer_out = sum(fee * usd(0, np.mean(outputs[(c, prompts[(qa, c, k)]["question_type"])]), prices[ANSWER_MODEL])
                     for qa in chosen for c in conditions)
    judge_total = open_ended * len(conditions) * judge_unit
    worst_answer = max(fee * usd(prompts[(qa, c, k)]["prompt_tokens"] + calibration["overhead_tokens"]["max"],
                                 fs.ANSWER_MAX_TOKENS, prices[ANSWER_MODEL]) for qa in chosen for c in conditions)
    worst_judge = fee * usd(judge["input_max"], judge["output_max"], prices[JUDGE_MODEL])

    sealed_verdicts = {c: {r["qa_key"] for r in jsonl(SEALED_RUN[c] / "results.jsonl") if r["condition"] == c}
                       for c in conditions}
    missing_k_ref = {c: sum(1 for qa in chosen if qa not in sealed_verdicts[c]) for c in conditions}
    result = {
        "api_calls": 0, "config_sha256": sha256_file(CONFIG), "phase1_manifest_sha256": sha256_file(PHASE1 / "manifest.json"),
        "sample_order_sha256": sha256_bytes(order_text.encode()),
        "k": k, "conditions": conditions, "prices": prices, "fee_factor": fee,
        "calibration": calibration,
        "k_prompts_already_in_sealed_cache": cache_hits,
        "estimate_target_usd": target, "n_selected": n_max,
        "allocation": counts, "open_ended_in_sample": open_ended, "multiple_choice_in_sample": n_max - open_ended,
        "estimated_usd": {"answer_input": answer_in, "answer_output": answer_out, "judge": judge_total,
                          "total": curve[n_max]},
        "planned_calls": {"answer": n_max * len(conditions),
                          "judge_rows": open_ended * len(conditions),
                          "judge_paid_calls_expected": open_ended * len(conditions) * judge["mean_attempts"]},
        "mean_prompt_tokens": {c: float(np.mean([prompts[(qa, c, k)]["prompt_tokens"] for qa in chosen])) for c in conditions},
        "guard_worst_single_call_usd": {"answer": worst_answer, "judge": worst_judge},
        "cost_curve_usd": {str(n): curve[n] for n in (100, 150, 200, 250, 300, n_max, n_max + 1) if n in curve},
        "sealed_k_reference_verdicts_missing": missing_k_ref,
        "sample_qa_keys": chosen,
    }
    (PHASE2 / "dry_run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "sample_qa_keys"}, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------
# Paid run
# --------------------------------------------------------------------------

class BudgetStop(RuntimeError):
    pass


class BudgetGuard:
    """Wraps client.chat.completions.create; forwards kwargs unchanged."""

    def __init__(self, create, *, cost, estimate, ceiling: float, spent: float = 0.0):
        self._create, self._cost, self._estimate = create, cost, estimate
        self.ceiling, self.spent, self.reserved = ceiling, spent, 0.0
        self.lock = threading.Lock()
        self.calls = 0

    def __call__(self, **kwargs):
        estimate = self._estimate(kwargs)
        with self.lock:
            if self.spent + self.reserved + estimate > self.ceiling:
                raise BudgetStop(f"guard: spent {self.spent:.4f} + reserved {self.reserved:.4f} + "
                                 f"next {estimate:.4f} > {self.ceiling:.2f}")
            self.reserved += estimate
        try:
            response = self._create(**kwargs)
        except Exception:
            with self.lock:
                self.reserved -= estimate
            raise
        usage = response.usage
        actual = self._cost(kwargs["model"], int(getattr(usage, "prompt_tokens", 0) or 0),
                            int(getattr(usage, "completion_tokens", 0) or 0))
        with self.lock:
            self.reserved -= estimate
            self.spent += actual
            self.calls += 1
        return response


def run(workers: int) -> None:
    config = json.loads(CONFIG.read_text())
    dry = json.loads((PHASE2 / "dry_run.json").read_text())
    k, k_ref, conditions = config["k"], config["k_reference"], config["conditions"]
    prices, fee = config["prices_usd_per_million_tokens"], config["fee_factor"]
    ceiling = config["budget_usd"]["hard_guard_accumulated_plus_next_call"]
    if sha256_file(CONFIG) != dry["config_sha256"]:
        raise SystemExit("config changed after the dry run; rerun --dry-run first")
    order_text = (PHASE2 / "sample_order.json").read_text()
    if sha256_bytes(order_text.encode()) != dry["sample_order_sha256"]:
        raise SystemExit("sample_order.json changed after the dry run")
    order_record = json.loads(order_text)
    n = dry["n_selected"]
    counts = allocation(n, order_record["category_sizes"])
    sample = [qa for name in sorted(order_record["order"]) for qa in order_record["order"][name][: counts[name]]]
    if sample != dry["sample_qa_keys"]:
        raise SystemExit("sample recomputed from seed differs from the dry run")

    manifest_path = PHASE2 / "manifest.json"
    frozen = {
        "experiment": "ksweep_v1", "phase": 2, "k": k, "k_reference": k_ref, "conditions": conditions,
        "sample_seed": order_record["seed"], "sample_design": order_record["design"], "n": n,
        "allocation": counts, "sample_qa_keys": sample, "sample_sha256": sha256_bytes("\n".join(sample).encode()),
        "config_sha256": dry["config_sha256"], "config": config,
        "dry_run_sha256": sha256_file(PHASE2 / "dry_run.json"),
        "sample_order_sha256": dry["sample_order_sha256"],
        "phase1_manifest_sha256": sha256_file(PHASE1 / "manifest.json"),
        "answer": {"model": ANSWER_MODEL, "provider": base.PROVIDER["answer"], "temperature": 0,
                   "max_tokens": fs.ANSWER_MAX_TOKENS, "postprocess": "evermembench_parallel_resume.official_answer"},
        "judge": {"model": JUDGE_MODEL, "provider": base.PROVIDER["judge"], "call": runner.JUDGE_CALL_VERSION,
                  "multiple_choice": "evermembench_parallel_resume.official_evaluator_mc (local, no API)"},
        "k_reference_verdicts": {"RAW": str(fs.SOURCE.parent.name), "RAW+EVENTS": str(A_RUN.parent.name)},
        "statistics": {"bootstrap": delivery.BOOTSTRAP, "seed": delivery.SEED, "ci": "percentile 2.5/97.5, paired question",
                       "mcnemar": "exact two-sided, paper_benchmark_common.exact_mcnemar"},
        "git": delivery.git_state(),
        "dependencies_sha256": {Path(m.__file__).name: sha256_file(Path(m.__file__))
                                for m in (fs, control, runner, base, common, delivery)},
    }
    frozen["dependencies_sha256"][Path(__file__).name] = sha256_file(Path(__file__).resolve())
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for field in ("sample_qa_keys", "sample_seed", "n", "config_sha256", "sample_order_sha256"):
            if existing[field] != frozen[field]:
                raise SystemExit(f"manifest field {field} differs from the frozen sample; refusing to continue")
    else:
        manifest_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")
    log(PHASE2, f"manifest frozen before calls: n={n}, sample sha256 {frozen['sample_sha256'][:16]}")

    data, _checks = fs.load_data()
    delivery.verify_a_run()
    questions = {row["qa_key"]: row for row in data.questions}
    phase1 = {(r["qa_key"], r["condition"], r["k"]): r for r in jsonl(PHASE1 / "per_question.jsonl")}

    import tiktoken
    encoder = tiktoken.get_encoding(control.TOKEN_ENCODING)
    overhead = int(dry["calibration"]["overhead_tokens"]["max"])
    judge_input_floor = int(dry["calibration"]["judge"]["input_max"])

    def cost(model: str, tokens_in: int, tokens_out: int) -> float:
        return fee * usd(tokens_in, tokens_out, prices[model])

    def estimate(kwargs: dict) -> float:
        tokens_in = sum(len(encoder.encode(str(m.get("content", "")))) for m in kwargs["messages"]) + overhead
        if kwargs["model"] == ANSWER_MODEL:
            return cost(ANSWER_MODEL, tokens_in, int(kwargs.get("max_tokens") or fs.ANSWER_MAX_TOKENS))
        return cost(kwargs["model"], max(2 * tokens_in, judge_input_floor), fs.JUDGE_OUTPUT_CEILING)

    router = fs.make_router(PHASE2, data)
    sealed_keys = set(data.cache)
    spent_before = sum(cost(r["model"], r["input_tokens"], r["output_tokens"])
                       for r in jsonl(PHASE2 / "api_usage.jsonl")) if (PHASE2 / "api_usage.jsonl").exists() else 0.0
    guard = BudgetGuard(router.client.chat.completions.create, cost=cost, estimate=estimate,
                        ceiling=ceiling, spent=spent_before)
    router.client.chat.completions.create = guard
    if router.client.chat.completions.create is not guard:
        raise SystemExit("budget guard is not installed on the client; refusing to call the API")
    stop = threading.Event()
    sink_lock = threading.Lock()

    def execute(items, work, path: Path, label: str) -> None:
        done = {(r["qa_key"], r["condition"]) for r in jsonl(path)} if path.exists() else set()
        pending = [item for item in items if (item[0], item[1]) not in done]
        log(PHASE2, f"{label}: {len(pending)} pending / {len(done)} done; spent {guard.spent:.4f}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(lambda it=item: None if stop.is_set() else work(*it)) for item in pending]
            for future in as_completed(futures):
                try:
                    row = future.result()
                except BudgetStop as error:
                    if not stop.is_set():
                        log(PHASE2, f"STOP {label}: {error}")
                    stop.set()
                    continue
                if row is not None:
                    with sink_lock:
                        append_jsonl(path, row)
        log(PHASE2, f"{label}: finished; spent {guard.spent:.4f}; stopped={stop.is_set()}")

    def answer(qa: str, condition: str) -> dict:
        question, cell = questions[qa], phase1[(qa, condition, k)]
        docs = [data.docs_by_id[doc_id] for doc_id in cell["doc_ids"]]
        context = control.render_context(docs)
        if len(encoder.encode(context)) != cell["context_tokens"]:
            raise RuntimeError(f"context differs from phase 1: {qa} {condition}")
        prompt = fs.build_prompt(data.prompts, question, context)
        key = fs.chat_key(prompt, fs.ANSWER_MAX_TOKENS)
        origin = "sealed_reuse:" + data.cache_origin[key] if key in sealed_keys else "new_or_resumed"
        raw_answer = router.chat(model=ANSWER_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
                                 max_tokens=fs.ANSWER_MAX_TOKENS, phase="answer", extra_body=base.PROVIDER["answer"])
        sources = control.unique_source_ids(docs)
        return {**{f: question[f] for f in ("qa_key", "topic", "id", "major", "minor", "question_type")},
                "condition": condition, "k": k, "answer": runner.official_answer(raw_answer, question["question_type"]),
                "answer_raw": raw_answer, "answer_cache": origin, "retrieved_ids": cell["doc_ids"],
                "exposed_source_ids": sources, "unique_source_count": len(sources),
                "context_tokens": cell["context_tokens"], "prompt_sha256": sha256_bytes(prompt.encode())}

    items = [(qa, condition) for qa in sample for condition in conditions]
    predictions_path = PHASE2 / "predictions.jsonl"
    execute(items, answer, predictions_path, "answers")
    predictions = {(r["qa_key"], r["condition"]): r for r in jsonl(predictions_path)} if predictions_path.exists() else {}
    if stop.is_set() or len(predictions) != len(items):
        write_partial(guard, predictions, None, "stopped during answers" if stop.is_set() else "incomplete answers")
        return
    failures = sum(1 for r in predictions.values() if r["answer"] in ("[EMPTY]", "[INVALID]", "[FAILED]"))
    if failures > max(3, fs.MAX_FAILURE_SHARE * len(predictions)):
        write_partial(guard, predictions, None, f"{failures} answer failure markers exceed the limit")
        return
    seal_jsonl(predictions_path, PHASE2 / "predictions_meta.json", expected_rows=len(items),
               identity_fields=("qa_key", "condition"),
               metadata={"phase": 2, "k": k, "manifest_sha256": sha256_file(manifest_path), "answer_failures": failures})

    gold = {row["qa_key"]: row for row in base.load_questions(include_gold=True)}  # only after sealed answers
    judge_prompts = base._prompts()["llm_judge"]

    def judge(qa: str, condition: str) -> dict:
        prediction, truth = predictions[(qa, condition)], gold[qa]
        meta: dict = {}
        if truth["question_type"] == "multiple_choice":
            correct = runner.official_evaluator_mc(prediction["answer"], truth["gold"])
            judge_text = "official exact multiple-choice comparison"
        else:
            prompt = judge_prompts["user_prompt"].format(question=truth["question"], golden_answer=truth["gold"],
                                                         generated_answer=prediction["answer"])
            call = runner.judge_chat(router, [{"role": "system", "content": judge_prompts["system_prompt"]},
                                              {"role": "user", "content": prompt}])
            correct, branch = runner.official_judge_label(call["content"])
            judge_text = call["content"]
            meta = {"judge_finish_reason": call["finish_reason"], "judge_attempts": call["attempts"], "judge_parse": branch}
        exposed, evidence = set(prediction["exposed_source_ids"]), set(truth["gold_source_ids"])
        return {"qa_key": qa, "topic": prediction["topic"], "major": prediction["major"], "minor": prediction["minor"],
                "question_type": prediction["question_type"], "condition": condition, "k": k, "correct": int(correct),
                "judge": judge_text, **meta,
                "evidence_recall": len(exposed & evidence) / len(evidence) if evidence else 1.0,
                "evidence_precision": len(exposed & evidence) / len(exposed) if exposed else 0.0}

    results_path = PHASE2 / "results.jsonl"
    execute(items, judge, results_path, "judge")
    results = {(r["qa_key"], r["condition"]): r for r in jsonl(results_path)} if results_path.exists() else {}
    if stop.is_set() or len(results) != len(items):
        write_partial(guard, predictions, results, "stopped during judging" if stop.is_set() else "incomplete verdicts")
        return
    seal_jsonl(results_path, PHASE2 / "results_meta.json", expected_rows=len(items), identity_fields=("qa_key", "condition"),
               metadata={"phase": 2, "k": k, "judge_call": runner.JUDGE_CALL_VERSION,
                         "predictions_sha256": sha256_file(predictions_path)})
    report(guard, sample, predictions, results, conditions, k, k_ref, prices, fee)


def rebuild_report() -> None:
    """Offline: re-render from sealed files; the actual spend is the value BudgetGuard booked during the run."""
    from types import SimpleNamespace
    config = json.loads(CONFIG.read_text())
    manifest = json.loads((PHASE2 / "manifest.json").read_text())
    verify_sealed_jsonl(PHASE2 / "predictions.jsonl", PHASE2 / "predictions_meta.json", expected_rows=2 * manifest["n"])
    verify_sealed_jsonl(PHASE2 / "results.jsonl", PHASE2 / "results_meta.json", expected_rows=2 * manifest["n"])
    previous = json.loads((PHASE2 / "summary.json").read_text())["spend"]
    booked = previous.get("actual_usd_with_fee", previous.get("guard_booked_usd"))
    predictions = {(r["qa_key"], r["condition"]): r for r in jsonl(PHASE2 / "predictions.jsonl")}
    results = {(r["qa_key"], r["condition"]): r for r in jsonl(PHASE2 / "results.jsonl")}
    report(SimpleNamespace(spent=booked), manifest["sample_qa_keys"], predictions, results, manifest["conditions"],
           manifest["k"], manifest["k_reference"], config["prices_usd_per_million_tokens"], config["fee_factor"])


def write_partial(guard: BudgetGuard, predictions: dict, results: dict | None, reason: str) -> None:
    status = {"complete": False, "reason": reason, "spent_usd_with_fee": guard.spent, "paid_calls": guard.calls,
              "predictions": len(predictions), "results": None if results is None else len(results)}
    (PHASE2 / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    log(PHASE2, f"partial results kept: {reason}")
    print(json.dumps(status, indent=2))


def report(guard, sample, predictions, results, conditions, k, k_ref, prices, fee) -> None:
    sealed = {"RAW": {r["qa_key"]: r for r in jsonl(fs.SOURCE / "results.jsonl") if r["condition"] == "RAW"},
              "RAW+EVENTS": {r["qa_key"]: r for r in jsonl(A_RUN / "results.jsonl")}}
    rows = []
    for qa in sample:
        for condition in conditions:
            rows.append({"qa_key": qa, "condition": f"{condition}@{k_ref}", "correct": int(sealed[condition][qa]["correct"])})
            rows.append({"qa_key": qa, "condition": f"{condition}@{k}", "correct": int(results[(qa, condition)]["correct"])})
    by = {(r["qa_key"], r["condition"]): r["correct"] for r in rows}
    rng = np.random.default_rng(delivery.SEED)
    indices = rng.integers(0, len(sample), size=(delivery.BOOTSTRAP, len(sample)), dtype=np.int32)

    def paired(left: str, right: str) -> dict:
        diff = 100.0 * np.asarray([by[(qa, left)] - by[(qa, right)] for qa in sample], dtype=np.float64)
        low, high = delivery.bootstrap_ci(diff, indices)
        mc = exact_mcnemar(rows, left, right)
        return {"left": left, "right": right, "delta_pp": float(diff.mean()), "ci95": [low, high],
                "left_only_correct": mc["left_only"], "right_only_correct": mc["right_only"],
                "discordant": mc["left_only"] + mc["right_only"], "mcnemar_p": mc["p_value"]}

    accuracy = {label: 100.0 * float(np.mean([by[(qa, label)] for qa in sample]))
                for label in sorted({r["condition"] for r in rows})}
    comparisons = [paired(f"RAW+EVENTS@{k_ref}", f"RAW@{k_ref}"), paired(f"RAW+EVENTS@{k}", f"RAW@{k}"),
                   paired(f"RAW@{k}", f"RAW@{k_ref}"), paired(f"RAW+EVENTS@{k}", f"RAW+EVENTS@{k_ref}")]
    usage = jsonl(PHASE2 / "api_usage.jsonl") if (PHASE2 / "api_usage.jsonl").exists() else []
    logged = sum(usd(r["input_tokens"], r["output_tokens"], prices[r["model"]]) for r in usage)
    caches = Counter(r["answer_cache"].split(":")[0] for r in predictions.values())
    parses = Counter(r.get("judge_parse", "multiple_choice") for r in results.values())
    new_answer_calls = sum(1 for r in predictions.values() if not r["answer_cache"].startswith("sealed_reuse"))
    logged_answer_calls = sum(1 for r in usage if r["phase"] == "answer")
    same_prompt = [qa for qa in sample
                   if predictions[(qa, conditions[0])]["prompt_sha256"] == predictions[(qa, conditions[1])]["prompt_sha256"]]
    spend = {
        "actual_usd_with_fee": guard.spent, "actual_usd_list": guard.spent / fee,
        "actual_basis": "BudgetGuard: response.usage of every create call, times config prices, times fee_factor",
        "api_usage_log_rows": len(usage), "api_usage_log_by_phase": dict(Counter(r["phase"] for r in usage)),
        "api_usage_log_usd_with_fee": logged * fee,
        "answer_calls_sent": new_answer_calls, "answer_calls_logged": logged_answer_calls,
        "answer_calls_unlogged": new_answer_calls - logged_answer_calls,
        "unlogged_reason": "identical payloads sent concurrently: CachedAPI.chat appends usage only for the call that stores the key first",
    }
    summary = {"n": len(sample), "accuracy_percent": accuracy, "comparisons": comparisons, "spend": spend,
               "answer_cache": dict(caches), "judge_parse": dict(parses),
               "judge_attempts_total": sum(r.get("judge_attempts", 0) for r in results.values()),
               "identical_k_prompt_both_conditions": len(same_prompt),
               "identical_answer_among_them": sum(1 for qa in same_prompt if predictions[(qa, conditions[0])]["answer_raw"]
                                                  == predictions[(qa, conditions[1])]["answer_raw"])}
    (PHASE2 / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    lines = ["# k-sweep, phase 2: accuracy at k=10 (sealed) and k=20", "",
             f"EverMemBench, stratified sample N = {len(sample)} "
             f"(sample seed {json.loads((PHASE2 / 'manifest.json').read_text())['sample_seed']}). "
             f"k=10 verdicts are the sealed ones; k=20 answers and verdicts are new. Paired question bootstrap, "
             f"{delivery.BOOTSTRAP:,} resamples, seed {delivery.SEED}, percentile 95% CI; exact McNemar.", "",
             "## Accuracy, %", "", "| condition | k=10 | k=20 |", "|---|---:|---:|"]
    for condition in conditions:
        lines.append(f"| {condition} | {accuracy[f'{condition}@{k_ref}']:.2f} | {accuracy[f'{condition}@{k}']:.2f} |")
    lines += ["", "## Paired differences", "",
              "| left − right | Δ, pp | 95% CI | left-only correct | right-only correct | discordant | McNemar p |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for c in comparisons:
        lines.append(f"| {c['left']} − {c['right']} | {c['delta_pp']:+.2f} | [{c['ci95'][0]:+.2f}; {c['ci95'][1]:+.2f}] | "
                     f"{c['left_only_correct']} | {c['right_only_correct']} | {c['discordant']} | {c['mcnemar_p']:.3f} |")
    lines += ["", "## Spend", "",
              f"- actual, from response.usage of every call (BudgetGuard): ${spend['actual_usd_with_fee']:.4f} with fee factor {fee}; "
              f"${spend['actual_usd_list']:.4f} at list price",
              f"- answer calls sent: {spend['answer_calls_sent']}; logged in api_usage.jsonl: {spend['answer_calls_logged']}; "
              f"unlogged: {spend['answer_calls_unlogged']} (identical payloads sent concurrently)",
              f"- api_usage.jsonl alone: {spend['api_usage_log_rows']} rows, ${spend['api_usage_log_usd_with_fee']:.4f} with fee",
              f"- answers reused from sealed caches: {caches.get('sealed_reuse', 0)}",
              f"- judge parse branches: {dict(parses)}; paid judge attempts: {summary['judge_attempts_total']}",
              "", "## Identical prompts", "",
              f"- questions whose k={k} prompt is byte-identical in both conditions (no event document in top-{k}): "
              f"{summary['identical_k_prompt_both_conditions']}; identical answer text among them: {summary['identical_answer_among_them']}"]
    (PHASE2 / "report_phase2.md").write_text("\n".join(lines) + "\n")
    (PHASE2 / "status.json").write_text(json.dumps({"complete": True, "spent_usd_with_fee": spend["actual_usd_with_fee"]},
                                                  indent=2) + "\n")
    log(PHASE2, "phase 2 complete")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true", help="rebuild the report offline from sealed phase-2 files")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.report:
        rebuild_report()
    else:
        run(args.workers)
