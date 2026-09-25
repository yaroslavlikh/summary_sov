"""Parallel resume for the sealed GroupMemBench answer and judge phases.

This changes only request scheduling.  Prompts, models, retrieval, parsing and
artifacts are identical to ``groupmembench_episode_run``.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import AuthenticationError, RateLimitError

from research import groupmembench_episode_run as base
from research.paper_benchmark_common import (
    EMBED_MODEL,
    TOP_K,
    DenseIndex,
    EmbeddingCache,
    append_jsonl,
    jsonl,
    log,
    seal_jsonl,
    sha256_file,
    verify_episode_snapshot,
    verify_sealed_jsonl,
)


RUNNER = Path(__file__).resolve()


def _parallel(jobs, work, write, *, workers: int, label: str) -> None:
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        try:
            for future in as_completed(futures):
                write(future.result())
                completed += 1
                if completed % 25 == 0 or completed == len(jobs):
                    log(base.RUN_DIR, f"{label}: {completed}/{len(jobs)} pending (workers={workers})")
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def run_inference(api, *, workers: int) -> None:
    domains = base.DOMAINS
    verify_episode_snapshot(base.RUN_DIR, base._source_manifest(domains))
    _frame, raw_by_domain, raw_by_id = base.load_messages(domains)
    episodes = base._load_episode_docs(raw_by_id)
    embeddings = EmbeddingCache(base.RUN_DIR, api)
    agent_system = (base.OFFICIAL_ROOT / "prompts" / "hipporag_agent_system.txt").read_text().strip()
    done = {(row["qa_key"], row["condition"]) for row in jsonl(base.RUN_DIR / "predictions.jsonl")}
    questions = base.load_questions(domains, include_gold=False)
    pending = [
        (row, condition)
        for row in questions
        for condition in base.CONDITIONS
        if (row["qa_key"], condition) not in done
    ]
    log(base.RUN_DIR, f"parallel inference: {len(pending)} pending / {len(done)} cached")

    jobs = []
    for domain in domains:
        selected = [(row, condition) for row, condition in pending if row["domain"] == domain]
        if not selected:
            continue
        indexes = {
            "RAW": DenseIndex(raw_by_domain[domain], embeddings, f"{domain}-raw"),
            "RAW+EPISODES": DenseIndex(
                raw_by_domain[domain] + episodes.get(domain, []), embeddings, f"{domain}-raw-episodes"
            ),
        }
        unique = {row["qa_key"]: row for row, _condition in selected}
        queries = {
            key: (f"{row['asking_user_id']} {row['question']}" if row["asking_user_id"] else row["question"])
            for key, row in unique.items()
        }
        vectors = embeddings.all(list(queries.values()))
        vector_by_key = dict(zip(queries, vectors))
        for row, condition in selected:
            docs, retrieval_ms = indexes[condition].search_vector(vector_by_key[row["qa_key"]], TOP_K)
            passages = "\n\n".join(f"[{i}] {doc.rendered}" for i, doc in enumerate(docs, 1))
            asker = f"Asking user: {row['asking_user_id']}\n\n" if row["asking_user_id"] else ""
            prompt = (
                f"{asker}Question:\n{row['question']}\n\nRetrieved passages:\n{passages}\n\n"
                "Answer the question using the retrieved passages."
            )
            jobs.append((row, condition, docs, retrieval_ms, prompt))

    def work(job):
        row, condition, docs, retrieval_ms, prompt = job
        started = time.perf_counter()
        output = api.chat(
            model=base.MODEL,
            messages=[{"role": "system", "content": agent_system}, {"role": "user", "content": prompt}],
            temperature=.2,
            max_tokens=512,
            phase="answer",
        )
        reasoning, answer = base._split_final(output)
        return {
            **row,
            "condition": condition,
            "answer": answer,
            "agent_reasoning": reasoning,
            "retrieved_ids": [doc.doc_id for doc in docs],
            "exposed_source_ids": list(dict.fromkeys(source for doc in docs for source in doc.source_ids)),
            "retrieval_ms": retrieval_ms,
            "answer_wall_ms": (time.perf_counter() - started) * 1000,
        }

    _parallel(
        jobs,
        work,
        lambda row: append_jsonl(base.RUN_DIR / "predictions.jsonl", row),
        workers=workers,
        label="parallel inference",
    )
    seal_jsonl(
        base.RUN_DIR / "predictions.jsonl",
        base.RUN_DIR / "predictions_meta.json",
        expected_rows=len(questions) * len(base.CONDITIONS),
        identity_fields=("qa_key", "condition"),
        metadata={
            "benchmark": "GroupMemBench",
            "official_commit": base.OFFICIAL_COMMIT,
            "conditions": list(base.CONDITIONS),
            "answer_model": base.MODEL,
            "embedding_model": EMBED_MODEL,
            "top_k": TOP_K,
            "episode_snapshot_sha256": sha256_file(base.RUN_DIR / "episodes.jsonl"),
            "execution_runner": RUNNER.name,
            "execution_runner_sha256": sha256_file(RUNNER),
            "workers": workers,
        },
    )


def score(api, *, workers: int) -> None:
    domains = base.DOMAINS
    judge_system = (base.OFFICIAL_ROOT / "prompts" / "hipporag_judge_system.txt").read_text().strip()
    expected = len(base.load_questions(domains, include_gold=False)) * len(base.CONDITIONS)
    verify_sealed_jsonl(
        base.RUN_DIR / "predictions.jsonl", base.RUN_DIR / "predictions_meta.json", expected_rows=expected
    )
    predictions = {
        (row["qa_key"], row["condition"]): row for row in jsonl(base.RUN_DIR / "predictions.jsonl")
    }
    gold = {row["qa_key"]: row for row in base.load_questions(domains, include_gold=True)}
    evaluation_manifest = base._evaluation_manifest(domains)
    manifest_path = base.RUN_DIR / "evaluation_input_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != evaluation_manifest:
        raise RuntimeError("evaluation input manifest mismatch")
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(evaluation_manifest, indent=2) + "\n")
    done = {(row["qa_key"], row["condition"]) for row in jsonl(base.RUN_DIR / "results.jsonl")}
    pending = [key for key in predictions if key not in done]
    log(base.RUN_DIR, f"parallel score: {len(pending)} pending / {len(done)} cached")

    def work(key):
        prediction = predictions[key]
        truth = gold[prediction["qa_key"]]
        prompt = (
            f"Question:\n{truth['question']}\n\nGold Answer:\n{truth['gold']}\n\n"
            f"Agent Answer:\n{prediction['answer']}\n"
        )
        output = api.chat(
            model=base.MODEL,
            messages=[{"role": "system", "content": judge_system}, {"role": "user", "content": prompt}],
            temperature=.2,
            max_tokens=256,
            phase="judge",
        )
        reasoning, final = base._split_final(output)
        judgment = base._parse_judgment(final)
        return {
            "qa_key": prediction["qa_key"],
            "domain": prediction["domain"],
            "qtype": prediction["qtype"],
            "condition": prediction["condition"],
            "correct": int(judgment is True),
            "verdict": "Correct" if judgment else "Incorrect",
            "judge_reasoning": reasoning,
            "judge_answer": final,
        }

    _parallel(
        pending,
        work,
        lambda row: append_jsonl(base.RUN_DIR / "results.jsonl", row),
        workers=workers,
        label="parallel score",
    )
    seal_jsonl(
        base.RUN_DIR / "results.jsonl",
        base.RUN_DIR / "results_meta.json",
        expected_rows=expected,
        identity_fields=("qa_key", "condition"),
        metadata={
            "benchmark": "GroupMemBench",
            "official_commit": base.OFFICIAL_COMMIT,
            "conditions": list(base.CONDITIONS),
            "judge_model": base.MODEL,
            "predictions_sha256": sha256_file(base.RUN_DIR / "predictions.jsonl"),
            "evaluation_input_manifest_sha256": sha256_file(manifest_path),
            "execution_runner": RUNNER.name,
            "execution_runner_sha256": sha256_file(RUNNER),
            "workers": workers,
        },
    )
    base.render_report(domains)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.status:
        base.status()
        return
    if not (args.inference or args.score or args.all):
        parser.error("choose --inference, --score, --all, or --status")
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    try:
        api = base.preflight(base.DOMAINS)
        if args.inference or args.all:
            run_inference(api, workers=args.workers)
        if args.score or args.all:
            score(api, workers=args.workers)
    except RateLimitError as error:
        log(base.RUN_DIR, f"rate limit; completed parallel work is cached: {error}")
        raise SystemExit(75) from None
    except AuthenticationError as error:
        raise SystemExit(f"OpenAI authentication failed: {error}") from None


if __name__ == "__main__":
    main()
