"""Final sprint experiment D: repair the 102 empty GroupMemBench gpt-5 answers.

Only the answers that came back empty are regenerated; the 1,388 valid answers and their
judgments are copied from the sealed run as immutable input. Same model (gpt-5), route (direct
OpenAI), messages, answer parsing and judge; only the completion budget of the failed requests is
raised (8,192, then one retry at 16,384 if the completion is still empty or cut by length).
Protocol: research/FINAL_EXPERIMENTS_PROTOCOL.md.

    python3 -m research.final_sprint_groupmembench_repair --preflight   # offline
    python3 -m research.final_sprint_groupmembench_repair --run         # after explicit GO
    python3 -m research.final_sprint_groupmembench_repair --report
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, RateLimitError

from research import evermembench_chronological_order_control as chrono
from research import groupmembench_answer_cap_rerun as reconstruct
from research import groupmembench_episode_run as base
from research.paper_benchmark_common import (
    MODEL_PRICES_USD_PER_M,
    CachedAPI,
    append_jsonl,
    exact_mcnemar,
    jsonl,
    log,
    paired_bootstrap,
    seal_jsonl,
    sha256_bytes,
    sha256_file,
    verify_sealed_jsonl,
)

SOURCE = base.RUN_DIR
RUN_DIR = base.ROOT / ".research_runs" / "final_sprint_D_groupmembench_repair_v2"
PREFLIGHT_DIR = base.ROOT / ".research_runs" / "final_sprint_preflight_v1"
PROTOCOL = base.ROOT / "research" / "FINAL_EXPERIMENTS_PROTOCOL.md"
RUNNER = Path(__file__).resolve()
ORIGINAL_BUDGET = 2048
REPAIR_BUDGETS = (8192, 16384)
SOURCE_FILES = ("predictions.jsonl", "predictions_meta.json", "results.jsonl", "results_meta.json",
                "episodes.jsonl", "episodes_meta.json", "openai_chat_cache.jsonl", "api_usage.jsonl")


# --------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------

def classify_original(answer: str, usage_row: dict | None) -> dict:
    output_tokens = int(usage_row["output_tokens"]) if usage_row else None
    exhausted = output_tokens is not None and output_tokens >= ORIGINAL_BUDGET
    return {
        "answer_empty": not (answer or "").strip(), "output_tokens": output_tokens,
        "max_completion_tokens": ORIGINAL_BUDGET,
        "diagnosis": ("completion budget exhausted by hidden reasoning; empty content scored incorrect"
                      if exhausted else "empty content below the completion budget"),
    }


def completion_ok(content: str, finish_reason: str) -> bool:
    return bool((content or "").strip()) and finish_reason != "length"


def request_key(messages: list[dict], budget: int) -> str:
    payload = {"model": base.MODEL, "messages": messages, "max_completion_tokens": budget}
    return "repair:" + sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())


# --------------------------------------------------------------------------
# Inputs (offline)
# --------------------------------------------------------------------------

def source_hashes() -> dict:
    return {name: sha256_file(SOURCE / name) for name in SOURCE_FILES}


def empty_jobs() -> tuple[list[dict], dict, list[dict]]:
    questions = reconstruct.target_questions(smoke=False)
    jobs, checks = reconstruct.prepare_jobs(questions)
    sealed = {(p["qa_key"], p["condition"]): p for p in jsonl(SOURCE / "predictions.jsonl")}
    usage = {row["key"]: row for row in jsonl(SOURCE / "api_usage.jsonl") if row.get("phase") == "answer"}
    empties = []
    for job in jobs:
        prediction = sealed[(job["question"]["qa_key"], job["condition"])]
        if reconstruct.is_empty(prediction["answer"]):
            key = reconstruct.chat_key(label="openai", model=base.MODEL, messages=job["messages"],
                                       temperature=reconstruct.TEMPERATURE, max_tokens=reconstruct.ORIGINAL_ANSWER_MAX_TOKENS)
            job["original"] = classify_original(prediction["answer"], usage.get(key))
            empties.append(job)
    checks["empty_by_condition"] = dict(Counter(job["condition"] for job in empties))
    return empties, checks, questions


def preflight() -> dict:
    empties, checks, _questions = empty_jobs()
    prices = MODEL_PRICES_USD_PER_M[base.MODEL]
    usage =[r for r in jsonl(SOURCE / "api_usage.jsonl") if r.get("phase") == "answer"]
    empty_inputs = []
    by_key = {r["key"]: r for r in usage}
    for job in empties:
        key = reconstruct.chat_key(label="openai", model=base.MODEL, messages=job["messages"],
                                   temperature=reconstruct.TEMPERATURE, max_tokens=reconstruct.ORIGINAL_ANSWER_MAX_TOKENS)
        empty_inputs.append(by_key[key]["input_tokens"])
    mean_input = sum(empty_inputs) / len(empty_inputs)
    judge_rows = [r for r in jsonl(SOURCE / "api_usage.jsonl") if r.get("phase") == "judge"]
    judge_in = sum(r["input_tokens"] for r in judge_rows) / len(judge_rows)
    judge_out = sum(r["output_tokens"] for r in judge_rows) / len(judge_rows)
    n = len(empties)
    cost = lambda i, o: (i * prices[0] + o * prices[1]) / 1_000_000
    report = {
        "checks": checks, "source_sha256": source_hashes(), "empty_answers": n,
        "original_diagnosis": dict(Counter(j["original"]["diagnosis"] for j in empties)),
        "original_output_tokens": dict(Counter(j["original"]["output_tokens"] for j in empties)),
        "new_answer_calls": n, "new_answer_retry_calls_ceiling": n, "new_judge_calls": n,
        "mean_answer_input_tokens": round(mean_input),
        "answers": {"expected_usd": cost(n * mean_input, n * 4000),
                    "ceiling_usd": cost(2 * n * mean_input, n * sum(REPAIR_BUDGETS))},
        "judge": {"mean_tokens": [round(judge_in), round(judge_out)], "expected_usd": cost(n * judge_in, n * judge_out),
                  "ceiling_usd": cost(n * judge_in, n * 2000)},
        "reused": "1,388 non-empty answers and their judgments copied from the sealed run; memory, retrieval and prompts unchanged",
    }
    report["expected_usd"] = report["answers"]["expected_usd"] + report["judge"]["expected_usd"]
    report["ceiling_usd"] = report["answers"]["ceiling_usd"] + report["judge"]["ceiling_usd"]
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    (PREFLIGHT_DIR / "preflight_groupmembench_repair.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


# --------------------------------------------------------------------------
# Execution (only after GO)
# --------------------------------------------------------------------------

def repair_call(client: OpenAI, job: dict) -> dict:
    """Direct gpt-5 call exactly as CachedAPI sends it, but recording finish reason and usage."""
    cache = {row["key"]: row for row in jsonl(RUN_DIR / "repair_cache.jsonl")}
    attempts = []
    for budget in REPAIR_BUDGETS:
        key = request_key(job["messages"], budget)
        row = cache.get(key)
        if row is None:
            started = time.perf_counter()
            response = client.chat.completions.create(model=base.MODEL, messages=job["messages"], max_completion_tokens=budget)
            choice, usage = response.choices[0], response.usage
            details = getattr(usage, "completion_tokens_details", None)
            row = {"key": key, "budget": budget, "content": choice.message.content or "",
                   "finish_reason": str(choice.finish_reason or ""),
                   "input_tokens": int(usage.prompt_tokens or 0), "output_tokens": int(usage.completion_tokens or 0),
                   "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0),
                   "duration_ms": (time.perf_counter() - started) * 1000}
            append_jsonl(RUN_DIR / "repair_cache.jsonl", row)
            append_jsonl(RUN_DIR / "api_usage.jsonl", {"key": key, "provider": "openai", "phase": "answer_repair",
                                                        "model": base.MODEL, "input_tokens": row["input_tokens"],
                                                        "output_tokens": row["output_tokens"], "duration_ms": row["duration_ms"]})
        attempts.append({k: row[k] for k in ("budget", "finish_reason", "input_tokens", "output_tokens", "reasoning_tokens")})
        if completion_ok(row["content"], row["finish_reason"]):
            return {"content": row["content"], "attempts": attempts, "status": "repaired"}
    return {"content": "", "attempts": attempts, "status": "still_failed"}


def run(workers: int) -> None:
    empties, checks, questions = empty_jobs()
    if len(empties) != 102:
        raise SystemExit(f"stop: expected 102 empty answers, found {len(empties)}")
    chrono.ensure_run_manifest(RUN_DIR, {
        "experiment": "D", "protocol_sha256": sha256_file(PROTOCOL), "runner_sha256": sha256_file(RUNNER),
        "source_sha256": source_hashes(), "checks": checks, "repair_budgets": list(REPAIR_BUDGETS),
        "model": base.MODEL, "route": "direct OpenAI API", "judge_max_tokens": reconstruct.JUDGE_MAX_TOKENS,
        "targets": sorted(f"{j['question']['qa_key']}|{j['condition']}" for j in empties),
    })
    load_dotenv(base.ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=5, timeout=600.0)
    done = {(r["qa_key"], r["condition"]) for r in jsonl(RUN_DIR / "repairs.jsonl")}
    pending = [j for j in empties if (j["question"]["qa_key"], j["condition"]) not in done]
    log(RUN_DIR, f"repair: {len(pending)} pending / {len(done)} done")

    def work(job: dict) -> dict:
        outcome = repair_call(client, job)
        reasoning, answer = base._split_final(outcome["content"])
        return {"qa_key": job["question"]["qa_key"], "condition": job["condition"], "original_error": job["original"],
                "retry_config": {"model": base.MODEL, "budgets": list(REPAIR_BUDGETS), "route": "direct OpenAI"},
                "attempts": outcome["attempts"], "status": outcome["status"], "final_completion": outcome["content"],
                "answer": answer, "agent_reasoning": reasoning}

    chrono.parallel(RUN_DIR, pending, work, lambda row: append_jsonl(RUN_DIR / "repairs.jsonl", row), workers=workers, label="D repair")
    seal_jsonl(RUN_DIR / "repairs.jsonl", RUN_DIR / "repairs_meta.json", expected_rows=len(empties),
               identity_fields=("qa_key", "condition"), metadata={"runner_sha256": sha256_file(RUNNER)})
    assemble_and_score(empties, questions)


def assemble_and_score(empties: list[dict], questions: list[dict]) -> None:
    repairs = {(r["qa_key"], r["condition"]): r for r in jsonl(RUN_DIR / "repairs.jsonl")}
    sealed_predictions = jsonl(SOURCE / "predictions.jsonl")
    sealed_results = {(r["qa_key"], r["condition"]): r for r in jsonl(SOURCE / "results.jsonl")}
    if not (RUN_DIR / "predictions_v2_meta.json").exists():
        for prediction in sealed_predictions:
            key = (prediction["qa_key"], prediction["condition"])
            row = dict(prediction)
            if key in repairs:
                row.update(answer=repairs[key]["answer"], agent_reasoning=repairs[key]["agent_reasoning"],
                           repair_status=repairs[key]["status"])
            else:
                row["repair_status"] = "original"
            append_jsonl(RUN_DIR / "predictions_v2.jsonl", row)
        seal_jsonl(RUN_DIR / "predictions_v2.jsonl", RUN_DIR / "predictions_v2_meta.json", expected_rows=len(sealed_predictions),
                   identity_fields=("qa_key", "condition"), metadata={"repairs_sha256": sha256_file(RUN_DIR / "repairs.jsonl")})
    judge_system = (base.OFFICIAL_ROOT / "prompts" / "hipporag_judge_system.txt").read_text().strip()
    gold = {row["qa_key"]: row for row in base.load_questions(base.DOMAINS, include_gold=True)}
    load_dotenv(base.ROOT / ".env")
    api = CachedAPI(RUN_DIR, api_key=os.environ["OPENAI_API_KEY"])
    done = {(r["qa_key"], r["condition"]) for r in jsonl(RUN_DIR / "results_v2.jsonl")}
    pending = [p for p in jsonl(RUN_DIR / "predictions_v2.jsonl") if (p["qa_key"], p["condition"]) not in done]

    def work(prediction: dict) -> dict:
        key = (prediction["qa_key"], prediction["condition"])
        if prediction["repair_status"] == "original":
            return {**sealed_results[key], "repair_status": "original"}
        truth = gold[prediction["qa_key"]]
        prompt = f"Question:\n{truth['question']}\n\nGold Answer:\n{truth['gold']}\n\nAgent Answer:\n{prediction['answer']}\n"
        output = api.chat(model=base.MODEL, messages=[{"role": "system", "content": judge_system}, {"role": "user", "content": prompt}],
                          temperature=reconstruct.TEMPERATURE, max_tokens=reconstruct.JUDGE_MAX_TOKENS, phase="judge")
        reasoning, final = base._split_final(output)
        judgment = base._parse_judgment(final)
        return {"qa_key": prediction["qa_key"], "domain": prediction["domain"], "qtype": prediction["qtype"],
                "condition": prediction["condition"], "correct": int(judgment is True),
                "verdict": "Correct" if judgment else "Incorrect", "judge_reasoning": reasoning, "judge_answer": final,
                "repair_status": prediction["repair_status"]}

    chrono.parallel(RUN_DIR, pending, work, lambda row: append_jsonl(RUN_DIR / "results_v2.jsonl", row), workers=8, label="D score")
    seal_jsonl(RUN_DIR / "results_v2.jsonl", RUN_DIR / "results_v2_meta.json", expected_rows=len(sealed_predictions),
               identity_fields=("qa_key", "condition"), metadata={"predictions_v2_sha256": sha256_file(RUN_DIR / "predictions_v2.jsonl")})
    render_report()


def render_report() -> Path:
    original = jsonl(SOURCE / "results.jsonl")
    repaired = jsonl(RUN_DIR / "results_v2.jsonl")
    repairs = jsonl(RUN_DIR / "repairs.jsonl")
    accuracy = lambda rows, c: sum(r["correct"] for r in rows if r["condition"] == c) / max(1, sum(1 for r in rows if r["condition"] == c))
    primary = lambda rows: [r for r in rows if r["domain"] in base.PRIMARY_DOMAINS and r["qtype"] in base.PRIMARY_QTYPES]

    def endpoint(rows: list[dict]) -> str:
        delta, low, high = paired_bootstrap(rows, "RAW+EPISODES", "RAW")
        test = exact_mcnemar(rows, "RAW+EPISODES", "RAW")
        return (f"{len(rows) // 2} | {100 * accuracy(rows, 'RAW'):.1f} | {100 * accuracy(rows, 'RAW+EPISODES'):.1f} | "
                f"{100 * delta:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}] | {test['left_only']} / {test['right_only']} | {test['p_value']:.4f}")

    lines = ["# Final sprint experiment D: GroupMemBench empty-answer repair (v2)", "",
             f"- source run `{SOURCE.name}` (read only); runner sha256 `{sha256_file(RUNNER)}`", "",
             "## Repair outcome", "", "| condition | original empty | repaired | still failed |", "|---|---:|---:|---:|"]
    for condition in base.CONDITIONS:
        rows = [r for r in repairs if r["condition"] == condition]
        lines.append(f"| {condition} | {len(rows)} | {sum(r['status'] == 'repaired' for r in rows)} | "
                     f"{sum(r['status'] == 'still_failed' for r in rows)} |")
    lines += ["", "## Endpoints", "", "| endpoint | answers | n | RAW % | RAW+EPISODES % | delta [95% CI] | EP-only / RAW-only | McNemar p |",
              "|---|---|---:|---:|---:|---|---:|---:|"]
    for label, select in (("primary: F+T, update+temporal", primary), ("all questions", lambda rows: rows)):
        lines.append(f"| {label} | original (protocol primary) | {endpoint(select(original))} |")
        lines.append(f"| {label} | repaired v2 | {endpoint(select(repaired))} |")
    report = RUN_DIR / "report.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("preflight", "run", "report"):
        modes.add_argument(f"--{mode}", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight:
            print(json.dumps(preflight(), indent=2, sort_keys=True))
        elif args.run:
            run(args.workers)
        else:
            print(render_report())
    except RateLimitError as error:
        raise SystemExit(f"rate limited; completed work is cached: {error}") from None
    except AuthenticationError as error:
        raise SystemExit(f"OpenAI authentication failed: {error}") from None


if __name__ == "__main__":
    main()
