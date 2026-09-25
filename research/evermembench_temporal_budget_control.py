"""EverMemBench Temporal budget control: RAW+EPISODES versus equal-size raw contexts.

Exploratory control for the Temporal Duration effect of the sealed
evermembench_temporal_episodes_official_v1 run. New raw-only conditions are matched per
question to the context RAW+EPISODES actually exposed. The source run (memory, indexes,
question embeddings, predictions, results) is read only and nothing is rebuilt.
Protocol: research/EVERMEMBENCH_TEMPORAL_BUDGET_CONTROL_PROTOCOL.md.

Run:
    python3 -m research.evermembench_temporal_budget_control --verify   # offline, no API
    python3 -m research.evermembench_temporal_budget_control --smoke    # 5 questions
    python3 -m research.evermembench_temporal_budget_control --all      # 300 TP questions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
from dotenv import load_dotenv
from openai import AuthenticationError, RateLimitError

from research import evermembench_episode_run as base
from research import evermembench_parallel_resume as runner
from research.paper_benchmark_common import (
    EMBED_MODEL,
    CachedAPI,
    SearchDoc,
    append_jsonl,
    exact_mcnemar,
    jsonl,
    latency_summary,
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
RUN_DIR = base.ROOT / ".research_runs" / "evermembench_temporal_budget_control_v1"
SMOKE_DIR = base.ROOT / ".research_runs" / "evermembench_temporal_budget_control_smoke_v1"
PROTOCOL = base.ROOT / "research" / "EVERMEMBENCH_TEMPORAL_BUDGET_CONTROL_PROTOCOL.md"
RUNNER = Path(__file__).resolve()
TARGET_MINOR = "TP"
COUNT_MATCHED = "RAW-count-matched"
TOKEN_MATCHED = "RAW-token-matched"
EPISODES_RERUN = "RAW+EPISODES-rerun"
NEW_CONDITIONS = (COUNT_MATCHED, TOKEN_MATCHED, EPISODES_RERUN)
TOKEN_ENCODING = "o200k_base"
RANK_DEPTH = 400
EMBED_BATCH = 128
SOURCE_ROWS = 4800


# --------------------------------------------------------------------------
# Context construction (pure, unit-tested)
# --------------------------------------------------------------------------

def render_context(docs: list[SearchDoc]) -> str:
    """Byte-identical to evermembench_parallel_resume.run_inference."""
    return "\n".join(f"- {doc.rendered}" for doc in docs) or "(No memories retrieved)"


def unique_source_ids(docs: list[SearchDoc]) -> list[str]:
    return list(dict.fromkeys(source for doc in docs for source in doc.source_ids))


def select_count_matched(ranked: list[SearchDoc], count: int) -> list[SearchDoc]:
    chosen: list[SearchDoc] = []
    seen: set[str] = set()
    for doc in ranked:
        if len(chosen) >= count:
            break
        if doc.doc_id not in seen:
            seen.add(doc.doc_id)
            chosen.append(doc)
    return chosen


def select_token_matched(
    ranked: list[SearchDoc], target_tokens: int, count_tokens: Callable[[str], int],
) -> tuple[list[SearchDoc], int]:
    """Add documents in rank order while the rendered context stays <= target; include the
    first overshooting document only if it lands strictly closer to the target."""
    chosen: list[SearchDoc] = []
    current = 0
    seen: set[str] = set()
    for doc in ranked:
        if doc.doc_id in seen:
            continue
        candidate = chosen + [doc]
        tokens = count_tokens(render_context(candidate))
        if tokens <= target_tokens:
            chosen, current = candidate, tokens
            seen.add(doc.doc_id)
            continue
        if not chosen or abs(tokens - target_tokens) < abs(target_tokens - current):
            chosen, current = candidate, tokens
        break
    return chosen, current


# --------------------------------------------------------------------------
# Read-only access to the sealed source run
# --------------------------------------------------------------------------

def frozen_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise RuntimeError(f"sealed vector cache missing, refusing to recompute: {path}")
    return np.load(path)


def raw_index_matrix(embed_dir: Path, topic: str, docs: list[SearchDoc]) -> np.ndarray:
    """Same digest and filename as paper_benchmark_common.DenseIndex."""
    digest = sha256_bytes("\n".join(f"{doc.doc_id}\t{doc.index_text}" for doc in docs).encode())[:20]
    return frozen_npy(embed_dir / f"index-{topic}-raw-{digest}.npy")


def question_vectors(embed_dir: Path, texts: list[str]) -> np.ndarray:
    """Same batch keys as paper_benchmark_common.EmbeddingCache.all(batch_size=128)."""
    chunks = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        key = sha256_bytes(json.dumps({"model": EMBED_MODEL, "texts": batch}, ensure_ascii=False).encode())
        chunks.append(frozen_npy(embed_dir / f"batch-{key[:24]}.npy"))
    return np.vstack(chunks)


def rank_raw(matrix: np.ndarray, vector: np.ndarray, depth: int) -> tuple[np.ndarray, float]:
    """Same scoring and ordering as DenseIndex.search_vector."""
    started = time.perf_counter()
    scores = np.sum(matrix * vector[None, :], axis=1, dtype=np.float64)
    order = np.argsort(-scores)[:depth]
    return order, (time.perf_counter() - started) * 1000


def answer_chat_key(prompt: str) -> str:
    """Key CachedAPI.chat assigns to an answer call with the official parameters."""
    payload = {
        "provider": "openrouter", "model": base.ANSWER_MODEL,
        "messages": [{"role": "user", "content": prompt}], "temperature": 0,
        "max_tokens": 1000, "json": False, "extra_body": base.PROVIDER["answer"],
    }
    return "chat:" + sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())


def verify_sources() -> None:
    verify_episode_snapshot(SOURCE_DIR, base._source_manifest())
    verify_sealed_jsonl(SOURCE_DIR / "predictions.jsonl", SOURCE_DIR / "predictions_meta.json",
                        expected_rows=SOURCE_ROWS)
    verify_sealed_jsonl(SOURCE_DIR / "results.jsonl", SOURCE_DIR / "results_meta.json",
                        expected_rows=SOURCE_ROWS)


def ensure_run_manifest(run_dir: Path, scope: list[str]) -> None:
    manifest = {
        "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
        "runner": RUNNER.name, "runner_sha256": sha256_file(RUNNER),
        "source_run": SOURCE_DIR.name,
        "source_predictions_sha256": sha256_file(SOURCE_DIR / "predictions.jsonl"),
        "source_results_sha256": sha256_file(SOURCE_DIR / "results.jsonl"),
        "source_episodes_sha256": sha256_file(SOURCE_DIR / "episodes.jsonl"),
        "conditions": list(NEW_CONDITIONS), "scope_qa_keys": sorted(scope),
        "token_encoding": TOKEN_ENCODING, "rank_depth": RANK_DEPTH,
    }
    path = run_dir / "run_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError(f"run manifest mismatch (protocol, runner or scope changed): {path}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


# --------------------------------------------------------------------------
# Job preparation (offline)
# --------------------------------------------------------------------------

def target_questions(smoke: bool) -> list[dict]:
    rows = [row for row in base.load_questions(include_gold=False) if row["minor"] == TARGET_MINOR]
    if not smoke:
        return rows
    first_per_topic = {}
    for row in rows:
        first_per_topic.setdefault(row["topic"], row)
    return list(first_per_topic.values())


def prepare_jobs(questions: list[dict]) -> tuple[list[dict], dict]:
    import tiktoken

    verify_sources()
    encoder = tiktoken.get_encoding(TOKEN_ENCODING)
    count_tokens = lambda text: len(encoder.encode(text))
    _frame, raw_by_topic, raw_by_id = base.load_messages()
    episodes = runner._load_episode_docs(raw_by_id)
    docs_by_id = {doc.doc_id: doc for docs in raw_by_topic.values() for doc in docs}
    docs_by_id.update({doc.doc_id: doc for docs in episodes.values() for doc in docs})
    frozen = {(row["qa_key"], row["condition"]): row for row in jsonl(SOURCE_DIR / "predictions.jsonl")}
    source_cache = {json.loads(line)["key"] for line in open(SOURCE_DIR / "openrouter_chat_cache.jsonl")}
    prompts = base._prompts()["answer"]
    embed_dir = SOURCE_DIR / "embeddings"
    wanted = {row["qa_key"] for row in questions}
    all_questions = base.load_questions(include_gold=False)

    jobs: list[dict] = []
    checks = {"questions": 0, "raw_top10_matches_sealed": 0, "raw_prompt_in_sealed_cache": 0,
              "episodes_prompt_in_sealed_cache": 0, "rank_depth_exhausted": 0}
    for topic in base.BATCHES:
        topic_questions = [row for row in all_questions if row["topic"] == topic]
        if not any(row["qa_key"] in wanted for row in topic_questions):
            continue
        vectors = question_vectors(embed_dir, [row["question"] for row in topic_questions])
        matrix = raw_index_matrix(embed_dir, topic, raw_by_topic[topic])
        for row, vector in zip(topic_questions, vectors):
            if row["qa_key"] not in wanted:
                continue
            if row["question_type"] != "open_ended":
                raise RuntimeError(f"unexpected question type for {row['qa_key']}")
            order, retrieval_ms = rank_raw(matrix, vector, RANK_DEPTH)
            ranked = [raw_by_topic[topic][int(index)] for index in order]
            sealed_raw = frozen[(row["qa_key"], "RAW")]
            sealed_episodes = frozen[(row["qa_key"], "RAW+EPISODES")]
            if [doc.doc_id for doc in ranked[:10]] != sealed_raw["retrieved_ids"]:
                raise RuntimeError(f"raw ranking does not reproduce sealed RAW top-10: {row['qa_key']}")
            episode_docs = [docs_by_id[doc_id] for doc_id in sealed_episodes["retrieved_ids"]]
            build = lambda docs: prompts["open_ended"].format(context=render_context(docs), question=row["question"])
            if answer_chat_key(build(ranked[:10])) not in source_cache:
                raise RuntimeError(f"RAW prompt does not reproduce the sealed prompt: {row['qa_key']}")
            if answer_chat_key(build(episode_docs)) not in source_cache:
                raise RuntimeError(f"RAW+EPISODES prompt does not reproduce the sealed prompt: {row['qa_key']}")
            checks["questions"] += 1
            checks["raw_top10_matches_sealed"] += 1
            checks["raw_prompt_in_sealed_cache"] += 1
            checks["episodes_prompt_in_sealed_cache"] += 1

            episode_tokens = count_tokens(render_context(episode_docs))
            episode_sources = unique_source_ids(episode_docs)
            shared = {
                "raw10_context_tokens": count_tokens(render_context(ranked[:10])),
                "episodes_context_tokens": episode_tokens,
                "episodes_unique_sources": len(episode_sources),
            }

            started = time.perf_counter()
            count_docs = select_count_matched(ranked, len(episode_sources))
            count_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            token_docs, _ = select_token_matched(ranked, episode_tokens, count_tokens)
            token_ms = (time.perf_counter() - started) * 1000
            if len(token_docs) >= len(ranked):
                checks["rank_depth_exhausted"] += 1

            for condition, docs, selection_ms in (
                (COUNT_MATCHED, count_docs, count_ms),
                (TOKEN_MATCHED, token_docs, token_ms),
                (EPISODES_RERUN, episode_docs, 0.0),
            ):
                prompt = build(docs)
                jobs.append({
                    "question": row, "condition": condition, "docs": docs, "prompt": prompt,
                    "retrieval_ms": retrieval_ms, "selection_ms": selection_ms,
                    "context_tokens": count_tokens(render_context(docs)), **shared,
                })
    return jobs, checks


# --------------------------------------------------------------------------
# API phases
# --------------------------------------------------------------------------

def make_router(run_dir: Path) -> CachedAPI:
    load_dotenv(base.ROOT / ".env")
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise SystemExit("LLM_API_KEY отсутствует в .env")
    return CachedAPI(run_dir, api_key=key, base_url=os.getenv("LLM_BASE_URL", base.OPENROUTER_DEFAULT),
                     label="openrouter")


def run_inference(run_dir: Path, questions: list[dict], router: CachedAPI, workers: int) -> None:
    done = {(row["qa_key"], row["condition"]) for row in jsonl(run_dir / "predictions.jsonl")}
    jobs, checks = prepare_jobs(questions)
    log(run_dir, f"reconstruction verified: {json.dumps(checks, sort_keys=True)}")
    pending = [job for job in jobs if (job["question"]["qa_key"], job["condition"]) not in done]
    log(run_dir, f"inference: {len(pending)} pending / {len(done)} cached")

    def work(job: dict) -> dict:
        row, docs = job["question"], job["docs"]
        started = time.perf_counter()
        raw_answer = router.chat(
            model=base.ANSWER_MODEL, messages=[{"role": "user", "content": job["prompt"]}], temperature=0,
            max_tokens=1000, phase="answer", extra_body=base.PROVIDER["answer"],
        )
        sources = unique_source_ids(docs)
        return {
            **{field: row[field] for field in ("qa_key", "topic", "id", "major", "minor", "question", "question_type")},
            "condition": job["condition"],
            "answer": runner.official_answer(raw_answer, row["question_type"]), "answer_raw": raw_answer,
            "retrieved_ids": [doc.doc_id for doc in docs], "exposed_source_ids": sources,
            "unique_source_count": len(sources), "context_tokens": job["context_tokens"],
            "raw10_context_tokens": job["raw10_context_tokens"],
            "episodes_context_tokens": job["episodes_context_tokens"],
            "episodes_unique_sources": job["episodes_unique_sources"],
            "prompt_sha256": sha256_bytes(job["prompt"].encode()),
            "retrieval_ms": job["retrieval_ms"], "selection_ms": job["selection_ms"],
            "answer_wall_ms": (time.perf_counter() - started) * 1000,
        }

    runner._parallel(pending, work, lambda row: append_jsonl(run_dir / "predictions.jsonl", row),
                     workers=workers, label="inference")
    rows = jsonl(run_dir / "predictions.jsonl")
    seal_jsonl(
        run_dir / "predictions.jsonl", run_dir / "predictions_meta.json",
        expected_rows=len(questions) * len(NEW_CONDITIONS), identity_fields=("qa_key", "condition"),
        metadata={
            "benchmark": "EverMemBench-Dynamic", "protocol_sha256": sha256_file(PROTOCOL),
            "runner_sha256": sha256_file(RUNNER), "conditions": list(NEW_CONDITIONS),
            "answer_model": base.ANSWER_MODEL, "token_encoding": TOKEN_ENCODING,
            "answer_failure_markers": _failure_markers(rows),
        },
    )


def _failure_markers(rows: list[dict]) -> dict[str, dict[str, int]]:
    markers = ("[EMPTY]", "[INVALID]", "[FAILED]")
    return {condition: {marker: sum(1 for row in rows if row["condition"] == condition and row["answer"] == marker)
                        for marker in markers} for condition in NEW_CONDITIONS}


def score(run_dir: Path, questions: list[dict], router: CachedAPI, workers: int) -> None:
    expected = len(questions) * len(NEW_CONDITIONS)
    verify_sealed_jsonl(run_dir / "predictions.jsonl", run_dir / "predictions_meta.json", expected_rows=expected)
    predictions = {(row["qa_key"], row["condition"]): row for row in jsonl(run_dir / "predictions.jsonl")}
    gold = {row["qa_key"]: row for row in base.load_questions(include_gold=True)}
    prompts = base._prompts()["llm_judge"]
    done = {(row["qa_key"], row["condition"]) for row in jsonl(run_dir / "results.jsonl")}
    pending = [key for key in predictions if key not in done]
    log(run_dir, f"score: {len(pending)} pending / {len(done)} cached")

    def work(key: tuple[str, str]) -> dict:
        prediction = predictions[key]
        truth = gold[prediction["qa_key"]]
        prompt = prompts["user_prompt"].format(
            question=truth["question"], golden_answer=truth["gold"], generated_answer=prediction["answer"],
        )
        call = runner.judge_chat(router, [{"role": "system", "content": prompts["system_prompt"]},
                                          {"role": "user", "content": prompt}])
        correct, branch = runner.official_judge_label(call["content"])
        exposed = set(prediction["exposed_source_ids"])
        evidence = set(truth["gold_source_ids"])
        return {
            "qa_key": prediction["qa_key"], "topic": prediction["topic"], "major": prediction["major"],
            "minor": prediction["minor"], "question_type": prediction["question_type"],
            "condition": prediction["condition"], "correct": int(correct), "judge": call["content"],
            "judge_finish_reason": call["finish_reason"], "judge_attempts": call["attempts"], "judge_parse": branch,
            "evidence_recall": len(exposed & evidence) / len(evidence) if evidence else 1.0,
            "evidence_precision": len(exposed & evidence) / len(exposed) if exposed else 0.0,
        }

    runner._parallel(pending, work, lambda row: append_jsonl(run_dir / "results.jsonl", row),
                     workers=workers, label="score")
    seal_jsonl(
        run_dir / "results.jsonl", run_dir / "results_meta.json",
        expected_rows=expected, identity_fields=("qa_key", "condition"),
        metadata={
            "benchmark": "EverMemBench-Dynamic", "protocol_sha256": sha256_file(PROTOCOL),
            "runner_sha256": sha256_file(RUNNER), "conditions": list(NEW_CONDITIONS),
            "judge_model": base.JUDGE_MODEL, "judge_call": runner.JUDGE_CALL_VERSION,
            "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
        },
    )


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def render_report(run_dir: Path) -> Path:
    new_results = jsonl(run_dir / "results.jsonl")
    new_predictions = jsonl(run_dir / "predictions.jsonl")
    keys = {row["qa_key"] for row in new_results}
    sealed_results = [row for row in jsonl(SOURCE_DIR / "results.jsonl") if row["qa_key"] in keys]
    sealed_predictions = [row for row in jsonl(SOURCE_DIR / "predictions.jsonl") if row["qa_key"] in keys]
    rows = sealed_results + new_results
    tokens_by_key = {row["qa_key"]: row for row in new_predictions}
    predictions = sealed_predictions + new_predictions
    conditions = ("RAW", "RAW+EPISODES", COUNT_MATCHED, TOKEN_MATCHED, EPISODES_RERUN)

    lines = [
        "# EverMemBench Temporal budget control", "",
        f"- protocol: `{PROTOCOL.name}` sha256 `{sha256_file(PROTOCOL)}`",
        f"- runner sha256: `{sha256_file(RUNNER)}`",
        f"- source run: `{SOURCE_DIR.name}` (sealed RAW and RAW+EPISODES rows)",
        f"- questions: {len(keys)} Temporal Duration (`{TARGET_MINOR}`); tokens: `{TOKEN_ENCODING}`", "",
        "## Conditions", "",
        "| condition | n | accuracy | mean unique raw sources | mean context tokens | evidence recall | evidence precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        preds = [row for row in predictions if row["condition"] == condition]
        if condition == "RAW":
            tokens = [tokens_by_key[row["qa_key"]]["raw10_context_tokens"] for row in preds]
        elif condition == "RAW+EPISODES":
            tokens = [tokens_by_key[row["qa_key"]]["episodes_context_tokens"] for row in preds]
        else:
            tokens = [row["context_tokens"] for row in preds]
        lines.append(
            f"| {condition} | {len(selected)} | {_mean([row['correct'] for row in selected]):.3f} | "
            f"{_mean([len(row['exposed_source_ids']) for row in preds]):.1f} | {_mean(tokens):.0f} | "
            f"{_mean([row['evidence_recall'] for row in selected]):.3f} | "
            f"{_mean([row['evidence_precision'] for row in selected]):.3f} |"
        )

    def comparison(left: str, right: str, label: str) -> str:
        delta, low, high = paired_bootstrap(rows, left, right)
        test = exact_mcnemar(rows, left, right)
        return (f"| {label} | {left} − {right} | {delta:+.3f} | [{low:+.3f}, {high:+.3f}] | "
                f"{test['left_only']} / {test['right_only']} | {test['p_value']:.4f} |")

    lines += [
        "", "## Paired comparisons", "",
        "| role | difference | delta | 95% CI | left-only / right-only | McNemar p |",
        "|---|---|---:|---:|---:|---:|",
        comparison("RAW+EPISODES", TOKEN_MATCHED, "**primary**"),
        comparison("RAW+EPISODES", COUNT_MATCHED, "secondary"),
        comparison(TOKEN_MATCHED, "RAW", "secondary"),
        comparison(COUNT_MATCHED, "RAW", "secondary"),
        comparison(EPISODES_RERUN, "RAW+EPISODES", "calibration"),
        comparison(EPISODES_RERUN, TOKEN_MATCHED, "same-run check"),
    ]
    sealed_ep = {row["qa_key"]: row["correct"] for row in sealed_results if row["condition"] == "RAW+EPISODES"}
    rerun = {row["qa_key"]: row["correct"] for row in new_results if row["condition"] == EPISODES_RERUN}
    agreement = _mean([float(sealed_ep[key] == rerun[key]) for key in rerun])
    calibration_delta = _mean(list(rerun.values())) - _mean([sealed_ep[key] for key in rerun])
    noisy = abs(calibration_delta) >= 0.03 or agreement < 0.90
    delta, low, high = paired_bootstrap(rows, "RAW+EPISODES", TOKEN_MATCHED)
    verdict = ("episodes add accuracy beyond an equal-token raw context" if low > 0 else
               "an equal-token raw context is at least as accurate" if high < 0 else
               "the episode effect is not distinguishable from context size")
    lines += [
        "", "## Pre-registered interpretation", "",
        f"- primary 95% CI [{low:+.3f}, {high:+.3f}]: **{verdict}**.",
        f"- calibration: rerun − sealed RAW+EPISODES = {calibration_delta:+.3f}; per-question agreement {agreement:.1%}; "
        + ("cross-run comparisons are noise-limited, see the same-run check." if noisy else "within the pre-registered noise bounds."),
    ]

    usage = usage_summary(run_dir)
    latency = latency_summary([row for row in new_predictions])
    selection = {condition: _mean([row["selection_ms"] for row in new_predictions if row["condition"] == condition])
                 for condition in NEW_CONDITIONS}
    answer_ms = {condition: _mean([row["answer_wall_ms"] for row in new_predictions if row["condition"] == condition])
                 for condition in NEW_CONDITIONS}
    results_meta = json.loads((run_dir / "results_meta.json").read_text()) if (run_dir / "results_meta.json").exists() else {}
    judge_rows = [row for row in new_results if "judge_parse" in row]
    lines += [
        "", "## Cost and latency (new conditions only)", "",
        f"- API calls: {usage['api_calls']}; estimated USD (uncached list prices): {usage['estimated_usd_upper_bound']:.4f}",
        f"- usage by model: `{json.dumps(usage['by_model'], sort_keys=True)}`",
        f"- raw ranking latency: `{json.dumps(latency, sort_keys=True)}`",
        f"- context selection ms (mean): `{json.dumps(selection, sort_keys=True)}`",
        f"- answer wall ms (mean, includes cache hits): `{json.dumps(answer_ms, sort_keys=True)}`",
        f"- judge: parse `{json.dumps({b: sum(1 for r in judge_rows if r['judge_parse'] == b) for b in {r['judge_parse'] for r in judge_rows}}, sort_keys=True)}`, "
        f"re-requested {sum(1 for r in judge_rows if r['judge_attempts'] > 1)}",
        f"- results sealed: {bool(results_meta)}",
    ]
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
    router = make_router(run_dir)
    runner.check_routes(router)
    run_inference(run_dir, questions, router, answer_workers)
    score(run_dir, questions, router, judge_workers)
    render_report(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-workers", type=int, default=16)
    parser.add_argument("--judge-workers", type=int, default=32)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("verify", "smoke", "all", "report"):
        modes.add_argument(f"--{mode}", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify:
            jobs, checks = prepare_jobs(target_questions(smoke=False))
            by_condition: dict[str, list[dict]] = {}
            for job in jobs:
                by_condition.setdefault(job["condition"], []).append(job)
            print(json.dumps(checks, indent=2, sort_keys=True))
            first = next(iter(by_condition.values()))
            print(f"RAW top-10 context tokens mean: {_mean([j['raw10_context_tokens'] for j in first]):.0f}")
            print(f"RAW+EPISODES context tokens mean: {_mean([j['episodes_context_tokens'] for j in first]):.0f}; "
                  f"unique sources mean {_mean([j['episodes_unique_sources'] for j in first]):.1f}")
            for condition, items in by_condition.items():
                print(f"{condition}: docs mean {_mean([len(j['docs']) for j in items]):.1f}, "
                      f"unique sources mean {_mean([len(unique_source_ids(j['docs'])) for j in items]):.1f}, "
                      f"context tokens mean {_mean([j['context_tokens'] for j in items]):.0f}")
            return
        if args.report:
            print(render_report(RUN_DIR))
            return
        run(SMOKE_DIR if args.smoke else RUN_DIR, args.smoke, args.answer_workers, args.judge_workers)
    except RateLimitError as error:
        log(SMOKE_DIR if args.smoke else RUN_DIR, f"rate limit; completed work is cached: {error}")
        raise SystemExit(75) from None
    except AuthenticationError as error:
        raise SystemExit(f"API authentication failed: {error}") from None


if __name__ == "__main__":
    main()
