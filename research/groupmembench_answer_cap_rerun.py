"""GroupMemBench amendment: regenerate all answers with an 8,192-token completion budget.

Copies retrieval from the sealed run, verifies that every reconstructed answer request matches
the sealed answer cache, then regenerates answers and judgments in a new run directory.
Amendment: research/GROUPMEMBENCH_ANSWER_CAP_AMENDMENT.md.

Run:
    python3 -m research.groupmembench_answer_cap_rerun --verify   # offline, no API
    python3 -m research.groupmembench_answer_cap_rerun --smoke    # first question per domain
    python3 -m research.groupmembench_answer_cap_rerun --all      # 745 questions x 2 conditions
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AuthenticationError, RateLimitError

from research import groupmembench_episode_run as base
from research.groupmembench_parallel_resume import _parallel
from research.paper_benchmark_common import (
    CachedAPI,
    SearchDoc,
    append_jsonl,
    exact_mcnemar,
    jsonl,
    log,
    paired_bootstrap,
    seal_jsonl,
    sha256_bytes,
    sha256_file,
    usage_summary,
    verify_episode_snapshot,
    verify_sealed_jsonl,
)


SOURCE_DIR = base.RUN_DIR
RUN_DIR = base.ROOT / ".research_runs" / "groupmembench_answer_cap_rerun_v1"
SMOKE_DIR = base.ROOT / ".research_runs" / "groupmembench_answer_cap_rerun_smoke_v1"
AMENDMENT = base.ROOT / "research" / "GROUPMEMBENCH_ANSWER_CAP_AMENDMENT.md"
RUNNER = Path(__file__).resolve()
ORIGINAL_ANSWER_MAX_TOKENS = 512    # CachedAPI budget for gpt-5: max(4 * 512, 2000) = 2048
RERUN_ANSWER_MAX_TOKENS = 2048      # CachedAPI budget for gpt-5: 4 * 2048 = 8192
JUDGE_MAX_TOKENS = 256
TEMPERATURE = .2
SOURCE_ROWS = 1490
SHARED_CATEGORIES = ("multi_hop", "knowledge_update", "temporal", "user_implicit", "term_ambiguity")


# --------------------------------------------------------------------------
# Request reconstruction (pure, unit-tested)
# --------------------------------------------------------------------------

def answer_messages(row: dict, docs: list[SearchDoc], agent_system: str) -> list[dict]:
    """Byte-identical to groupmembench_parallel_resume.run_inference."""
    passages = "\n\n".join(f"[{i}] {doc.rendered}" for i, doc in enumerate(docs, 1))
    asker = f"Asking user: {row['asking_user_id']}\n\n" if row["asking_user_id"] else ""
    prompt = (
        f"{asker}Question:\n{row['question']}\n\nRetrieved passages:\n{passages}\n\n"
        "Answer the question using the retrieved passages."
    )
    return [{"role": "system", "content": agent_system}, {"role": "user", "content": prompt}]


def chat_key(*, label: str, model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    """Key CachedAPI.chat assigns to a call without JSON output or extra body."""
    payload = {
        "provider": label, "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "json": False, "extra_body": None,
    }
    return "chat:" + sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())


def is_empty(answer: str) -> bool:
    return not (answer or "").strip()


# --------------------------------------------------------------------------
# Job preparation (offline)
# --------------------------------------------------------------------------

def verify_sources() -> None:
    verify_episode_snapshot(SOURCE_DIR, base._source_manifest(base.DOMAINS))
    verify_sealed_jsonl(SOURCE_DIR / "predictions.jsonl", SOURCE_DIR / "predictions_meta.json", expected_rows=SOURCE_ROWS)
    verify_sealed_jsonl(SOURCE_DIR / "results.jsonl", SOURCE_DIR / "results_meta.json", expected_rows=SOURCE_ROWS)


def target_questions(smoke: bool) -> list[dict]:
    rows = base.load_questions(base.DOMAINS, include_gold=False)
    if not smoke:
        return rows
    first: dict[str, dict] = {}
    for row in rows:
        first.setdefault(row["domain"], row)
    return list(first.values())


def prepare_jobs(questions: list[dict]) -> tuple[list[dict], dict]:
    verify_sources()
    _frame, raw_by_domain, raw_by_id = base.load_messages(base.DOMAINS)
    episodes = base._load_episode_docs(raw_by_id)
    docs_by_id = {doc.doc_id: doc for docs in raw_by_domain.values() for doc in docs}
    docs_by_id.update({doc.doc_id: doc for docs in episodes.values() for doc in docs})
    sealed = {(row["qa_key"], row["condition"]): row for row in jsonl(SOURCE_DIR / "predictions.jsonl")}
    sealed_keys = {json.loads(line)["key"] for line in open(SOURCE_DIR / "openai_chat_cache.jsonl")}
    agent_system = (base.OFFICIAL_ROOT / "prompts" / "hipporag_agent_system.txt").read_text().strip()

    jobs: list[dict] = []
    checks = {"requests": 0, "sealed_request_reproduced": 0, "sealed_empty_answers": 0}
    for row in questions:
        for condition in base.CONDITIONS:
            source = sealed[(row["qa_key"], condition)]
            docs = [docs_by_id[doc_id] for doc_id in source["retrieved_ids"]]
            exposed = list(dict.fromkeys(s for doc in docs for s in doc.source_ids))
            if exposed != source["exposed_source_ids"]:
                raise RuntimeError(f"sealed sources are not reproduced: {row['qa_key']} {condition}")
            messages = answer_messages(row, docs, agent_system)
            key = chat_key(label="openai", model=base.MODEL, messages=messages, temperature=TEMPERATURE,
                           max_tokens=ORIGINAL_ANSWER_MAX_TOKENS)
            if key not in sealed_keys:
                raise RuntimeError(f"sealed answer request is not reproduced: {row['qa_key']} {condition}")
            checks["requests"] += 1
            checks["sealed_request_reproduced"] += 1
            checks["sealed_empty_answers"] += int(is_empty(source["answer"]))
            jobs.append({"question": row, "condition": condition, "docs": docs, "messages": messages,
                         "retrieval_ms": source["retrieval_ms"]})
    return jobs, checks


def ensure_run_manifest(run_dir: Path, scope: list[str]) -> None:
    manifest = {
        "amendment": AMENDMENT.name, "amendment_sha256": sha256_file(AMENDMENT),
        "runner": RUNNER.name, "runner_sha256": sha256_file(RUNNER),
        "source_run": SOURCE_DIR.name,
        "source_predictions_sha256": sha256_file(SOURCE_DIR / "predictions.jsonl"),
        "source_results_sha256": sha256_file(SOURCE_DIR / "results.jsonl"),
        "source_episodes_sha256": sha256_file(SOURCE_DIR / "episodes.jsonl"),
        "official_commit": base.OFFICIAL_COMMIT, "model": base.MODEL,
        "answer_max_tokens": RERUN_ANSWER_MAX_TOKENS, "judge_max_tokens": JUDGE_MAX_TOKENS,
        "conditions": list(base.CONDITIONS), "scope_qa_keys": sorted(scope),
    }
    path = run_dir / "run_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError(f"run manifest mismatch (amendment, runner, sources or scope changed): {path}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


# --------------------------------------------------------------------------
# API phases
# --------------------------------------------------------------------------

def make_api(run_dir: Path) -> CachedAPI:
    load_dotenv(base.ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY отсутствует в .env")
    api = CachedAPI(run_dir, api_key=key)
    api.client.models.retrieve(base.MODEL)
    return api


def run_inference(run_dir: Path, questions: list[dict], api: CachedAPI, workers: int, *, smoke: bool) -> None:
    done = {(row["qa_key"], row["condition"]) for row in jsonl(run_dir / "predictions.jsonl")}
    jobs, checks = prepare_jobs(questions)
    log(run_dir, f"reconstruction verified: {json.dumps(checks, sort_keys=True)}")
    pending = [job for job in jobs if (job["question"]["qa_key"], job["condition"]) not in done]
    log(run_dir, f"inference: {len(pending)} pending / {len(done)} cached")

    def work(job: dict) -> dict:
        row, docs = job["question"], job["docs"]
        started = time.perf_counter()
        output = api.chat(model=base.MODEL, messages=job["messages"], temperature=TEMPERATURE,
                          max_tokens=RERUN_ANSWER_MAX_TOKENS, phase="answer")
        reasoning, answer = base._split_final(output)
        return {
            **row, "condition": job["condition"], "answer": answer, "agent_reasoning": reasoning,
            "empty_answer": is_empty(answer),
            "retrieved_ids": [doc.doc_id for doc in docs],
            "exposed_source_ids": list(dict.fromkeys(s for doc in docs for s in doc.source_ids)),
            "retrieval_ms": job["retrieval_ms"], "answer_wall_ms": (time.perf_counter() - started) * 1000,
        }

    _parallel(pending, work, lambda row: append_jsonl(run_dir / "predictions.jsonl", row),
              workers=workers, label="answer-cap inference")
    rows = jsonl(run_dir / "predictions.jsonl")
    empty = {c: sum(1 for r in rows if r["condition"] == c and r["empty_answer"]) for c in base.CONDITIONS}
    log(run_dir, f"empty answers after rerun: {json.dumps(empty, sort_keys=True)}")
    if smoke and any(empty.values()):
        raise SystemExit("smoke produced an empty answer at the 8,192-token budget; stop and amend")
    seal_jsonl(
        run_dir / "predictions.jsonl", run_dir / "predictions_meta.json",
        expected_rows=len(questions) * len(base.CONDITIONS), identity_fields=("qa_key", "condition"),
        metadata={
            "benchmark": "GroupMemBench", "official_commit": base.OFFICIAL_COMMIT,
            "amendment_sha256": sha256_file(AMENDMENT), "runner_sha256": sha256_file(RUNNER),
            "conditions": list(base.CONDITIONS), "answer_model": base.MODEL,
            "answer_max_tokens": RERUN_ANSWER_MAX_TOKENS, "empty_answers": empty,
        },
    )


def score(run_dir: Path, questions: list[dict], api: CachedAPI, workers: int) -> None:
    expected = len(questions) * len(base.CONDITIONS)
    verify_sealed_jsonl(run_dir / "predictions.jsonl", run_dir / "predictions_meta.json", expected_rows=expected)
    judge_system = (base.OFFICIAL_ROOT / "prompts" / "hipporag_judge_system.txt").read_text().strip()
    predictions = {(row["qa_key"], row["condition"]): row for row in jsonl(run_dir / "predictions.jsonl")}
    gold = {row["qa_key"]: row for row in base.load_questions(base.DOMAINS, include_gold=True)}
    done = {(row["qa_key"], row["condition"]) for row in jsonl(run_dir / "results.jsonl")}
    pending = [key for key in predictions if key not in done]
    log(run_dir, f"score: {len(pending)} pending / {len(done)} cached")

    def work(key: tuple[str, str]) -> dict:
        prediction = predictions[key]
        truth = gold[prediction["qa_key"]]
        prompt = (f"Question:\n{truth['question']}\n\nGold Answer:\n{truth['gold']}\n\n"
                  f"Agent Answer:\n{prediction['answer']}\n")
        output = api.chat(model=base.MODEL, messages=[{"role": "system", "content": judge_system},
                                                       {"role": "user", "content": prompt}],
                          temperature=TEMPERATURE, max_tokens=JUDGE_MAX_TOKENS, phase="judge")
        reasoning, final = base._split_final(output)
        judgment = base._parse_judgment(final)
        return {
            "qa_key": prediction["qa_key"], "domain": prediction["domain"], "qtype": prediction["qtype"],
            "condition": prediction["condition"], "correct": int(judgment is True),
            "verdict": "Correct" if judgment else "Incorrect", "judge_reasoning": reasoning, "judge_answer": final,
            "empty_judge": is_empty(final),
        }

    _parallel(pending, work, lambda row: append_jsonl(run_dir / "results.jsonl", row),
              workers=workers, label="answer-cap score")
    seal_jsonl(
        run_dir / "results.jsonl", run_dir / "results_meta.json",
        expected_rows=expected, identity_fields=("qa_key", "condition"),
        metadata={
            "benchmark": "GroupMemBench", "official_commit": base.OFFICIAL_COMMIT,
            "amendment_sha256": sha256_file(AMENDMENT), "runner_sha256": sha256_file(RUNNER),
            "conditions": list(base.CONDITIONS), "judge_model": base.MODEL,
            "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
        },
    )


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _accuracy(rows: list[dict], condition: str) -> float:
    values = [int(row["correct"]) for row in rows if row["condition"] == condition]
    return sum(values) / len(values) if values else float("nan")


def render_report(run_dir: Path) -> Path:
    rows = jsonl(run_dir / "results.jsonl")
    predictions = jsonl(run_dir / "predictions.jsonl")
    keys = {row["qa_key"] for row in rows}
    original = [row for row in jsonl(SOURCE_DIR / "results.jsonl") if row["qa_key"] in keys]
    original_predictions = [row for row in jsonl(SOURCE_DIR / "predictions.jsonl") if row["qa_key"] in keys]

    def endpoint(selected: list[dict]) -> str:
        delta, low, high = paired_bootstrap(selected, "RAW+EPISODES", "RAW")
        test = exact_mcnemar(selected, "RAW+EPISODES", "RAW")
        return (f"{len(selected) // 2} | {_accuracy(selected, 'RAW'):.3f} | {_accuracy(selected, 'RAW+EPISODES'):.3f} | "
                f"{delta:+.3f} [{low:+.3f}, {high:+.3f}] | {test['left_only']} / {test['right_only']} | {test['p_value']:.4f}")

    primary = lambda data: [r for r in data if r["domain"] in base.PRIMARY_DOMAINS and r["qtype"] in base.PRIMARY_QTYPES]
    shared = lambda data: [r for r in data if r["qtype"] in SHARED_CATEGORIES]
    lines = [
        "# GroupMemBench amendment: answers with an 8,192-token completion budget", "",
        f"- amendment: `{AMENDMENT.name}` sha256 `{sha256_file(AMENDMENT)}`",
        f"- runner sha256: `{sha256_file(RUNNER)}`",
        f"- source run (protocol primary result): `{SOURCE_DIR.name}`", "",
        "## Endpoints", "",
        "| endpoint | answers | n | RAW | RAW+EPISODES | delta [95% CI] | EP-only / RAW-only | McNemar p |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for label, select in (("primary: F+T, update+temporal", primary), ("five shared categories", shared),
                          ("all questions", lambda data: data)):
        lines.append(f"| {label} | original (protocol primary) | {endpoint(select(original))} |")
        lines.append(f"| {label} | 8,192-token budget | {endpoint(select(rows))} |")
    delta, low, high = paired_bootstrap(primary(rows), "RAW+EPISODES", "RAW")
    verdict = ("with an adequate completion budget, episodes improve the primary endpoint" if low > 0 else
               "with an adequate completion budget, episodes reduce accuracy on the primary endpoint" if high < 0 else
               "no difference is detected with an adequate completion budget")
    lines += ["", "## Interpretation rule fixed before new answers", "",
              f"- recomputed primary 95% CI [{low:+.3f}, {high:+.3f}]: **{verdict}**.",
              "- the original run remains the protocol's primary result.", "",
              "## Measurement health", "", "| condition | empty answers (original) | empty answers (rerun) | "
              "empty judge outputs (rerun) | accuracy original → rerun |", "|---|---:|---:|---:|---:|"]
    for condition in base.CONDITIONS:
        lines.append(
            f"| {condition} | {sum(1 for p in original_predictions if p['condition'] == condition and is_empty(p['answer']))} | "
            f"{sum(1 for p in predictions if p['condition'] == condition and p['empty_answer'])} | "
            f"{sum(1 for r in rows if r['condition'] == condition and r['empty_judge'])} | "
            f"{_accuracy(original, condition):.3f} → {_accuracy(rows, condition):.3f} |")
    lines += ["", "## Cells", "", "| domain | qtype | n | RAW | RAW+EPISODES | delta |", "|---|---|---:|---:|---:|---:|"]
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        cells[(row["domain"], row["qtype"])].append(row)
    for (domain, qtype), selected in sorted(cells.items()):
        raw, episodes = _accuracy(selected, "RAW"), _accuracy(selected, "RAW+EPISODES")
        lines.append(f"| {domain} | {qtype} | {len(selected) // 2} | {raw:.3f} | {episodes:.3f} | {episodes - raw:+.3f} |")
    usage = usage_summary(run_dir)
    lines += ["", "## Cost", "", f"- API calls: {usage['api_calls']}; estimated USD: {usage['estimated_usd_upper_bound']:.4f}",
              f"- usage by model and phase: `{json.dumps(usage['by_model'], sort_keys=True)}`"]
    report = run_dir / "report.md"
    report.write_text("\n".join(lines) + "\n")
    log(run_dir, f"report ready: {report}")
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def run(run_dir: Path, smoke: bool, answer_workers: int, judge_workers: int) -> None:
    questions = target_questions(smoke)
    ensure_run_manifest(run_dir, [row["qa_key"] for row in questions])
    api = make_api(run_dir)
    run_inference(run_dir, questions, api, answer_workers, smoke=smoke)
    score(run_dir, questions, api, judge_workers)
    render_report(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-workers", type=int, default=16)
    parser.add_argument("--judge-workers", type=int, default=16)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("verify", "smoke", "all", "report"):
        modes.add_argument(f"--{mode}", action="store_true")
    args = parser.parse_args()
    target_dir = SMOKE_DIR if args.smoke else RUN_DIR
    try:
        if args.verify:
            _jobs, checks = prepare_jobs(target_questions(smoke=False))
            print(json.dumps(checks, indent=2, sort_keys=True))
            return
        if args.report:
            print(render_report(RUN_DIR))
            return
        run(target_dir, args.smoke, args.answer_workers, args.judge_workers)
    except RateLimitError as error:
        log(target_dir, f"rate limit; completed work is cached: {error}")
        raise SystemExit(75) from None
    except AuthenticationError as error:
        raise SystemExit(f"OpenAI authentication failed: {error}") from None


if __name__ == "__main__":
    main()
