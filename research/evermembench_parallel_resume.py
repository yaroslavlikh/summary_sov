"""Parallel, resumable EverMemBench runner with unchanged benchmark semantics."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import AuthenticationError, RateLimitError

from research import evermembench_episode_run as base
from research.paper_benchmark_common import (
    EMBED_MODEL,
    EXTRACTION_MODEL,
    TOP_K,
    CachedAPI,
    DenseIndex,
    EmbeddingCache,
    append_jsonl,
    build_episode_snapshot,
    episode_doc,
    jsonl,
    log,
    seal_jsonl,
    sha256_bytes,
    sha256_file,
    verify_episode_snapshot,
    verify_sealed_jsonl,
)


RUNNER = Path(__file__).resolve()
SHARD_ROOT = base.RUN_DIR / "memory_shards"
JUDGE_CALL_VERSION = "official_evaluator_no_max_tokens_v1"
JUDGE_EMPTY_RETRIES = 5


def official_answerer_mc(response: str) -> str:
    """Port of the pinned official answerer._parse_mc_answer."""
    if response.startswith("[LLM_CALL_FAILED]"):
        return "[FAILED]"
    response = response.strip().upper()
    if not response:
        return "[EMPTY]"
    if len(response) == 1 and response in "ABCD":
        return response
    match = re.search(r"\b([ABCD])[.):,\s]", response)
    if match:
        return match.group(1)
    match = re.search(r"(?:answer|choice|option|select)[:\s]+([ABCD])\b", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if response[0] in "ABCD" and (len(response) == 1 or not response[1].isalpha()):
        return response[0]
    if response[-1] in "ABCD" and (len(response) == 1 or not response[-2].isalpha()):
        return response[-1]
    return "[INVALID]"


def official_answer(response: str, question_type: str) -> str:
    """Official answerer post-processing: MC letter or failure marker, open-ended text or [EMPTY]."""
    if question_type == "multiple_choice":
        return official_answerer_mc(response)
    if response.startswith("[LLM_CALL_FAILED]"):
        return "[FAILED]"
    return response.strip() if response else "[EMPTY]"


def official_evaluator_mc(generated: str, golden: str) -> bool:
    """Port of the pinned official evaluator._evaluate_mc and its _parse_mc_answer."""
    generated = generated.strip()
    if generated.startswith("[") and generated.endswith("]"):
        return False
    golden = golden.strip().upper()
    if len(golden) > 1 and golden[0] in "ABCD" and golden[1] in ".):":
        golden = golden[0]
    response = generated.upper()
    if not response:
        parsed = ""
    elif len(response) == 1 and response in "ABCD":
        parsed = response
    elif match := re.search(r"\b([ABCD])[.):,\s]", response):
        parsed = match.group(1)
    elif match := re.search(r"(?:answer|choice|option|select)[:\s]+([ABCD])\b", response, re.IGNORECASE):
        parsed = match.group(1).upper()
    elif response[0] in "ABCD" and (len(response) == 1 or not response[1].isalpha()):
        parsed = response[0]
    elif response[-1] in "ABCD" and (len(response) == 1 or not response[-2].isalpha()):
        parsed = response[-1]
    else:
        parsed = response
    return parsed == golden


def official_judge_label(content: str) -> tuple[bool, str]:
    """Port of the pinned official evaluator._parse_judge_response.

    Returns (is_correct, branch) so the report can show how many verdicts came
    from real JSON rather than the substring fallback.
    """
    try:
        if "```json" in content:
            start = content.find("```json") + 7
            content_json = content[start:content.find("```", start)].strip()
        elif "```" in content:
            start = content.find("```") + 3
            content_json = content[start:content.find("```", start)].strip()
        elif "{" in content and "}" in content:
            content_json = content[content.find("{"):content.rfind("}") + 1]
        else:
            content_json = content
        result = json.loads(content_json)
        label = result.get("label", "WRONG")
        if isinstance(label, dict):
            label = label.get("label", "WRONG")
        label = label.strip().upper() if isinstance(label, str) else "WRONG"
        return label == "CORRECT", "json"
    except (json.JSONDecodeError, KeyError):
        upper = content.upper()
        return "CORRECT" in upper and "WRONG" not in upper, "substring_fallback"
    except AttributeError:
        # Non-object JSON: the official evaluator raises here, exhausts retries and scores WRONG.
        return False, "non_object_json"


def judge_chat(router: CachedAPI, messages: list[dict]) -> dict:
    """Official judge request: temperature 0, no max_tokens, pinned provider.

    CachedAPI.chat always sends max_tokens, which the official evaluator never
    does; a thinking judge can spend a small cap on hidden reasoning and return
    empty content that the official parser scores WRONG. Empty or length-cut
    content is re-requested and never cached; if it persists the run stops.
    """
    payload = {
        "provider": router.label, "model": base.JUDGE_MODEL, "messages": messages,
        "temperature": 0, "extra_body": base.PROVIDER["judge"], "call": JUDGE_CALL_VERSION,
    }
    key = "judge:" + sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
    with router.lock:
        cached = router.values.get(key)
    if cached is not None:
        return {"content": str(cached), "finish_reason": "cached", "attempts": 0}
    finish_reason = ""
    for attempt in range(1, JUDGE_EMPTY_RETRIES + 1):
        started = time.perf_counter()
        response = router.client.chat.completions.create(
            model=base.JUDGE_MODEL, messages=messages, temperature=0, extra_body=base.PROVIDER["judge"],
        )
        duration_ms = (time.perf_counter() - started) * 1000
        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = str(choice.finish_reason or "")
        usage = response.usage
        with router.lock:
            append_jsonl(router.usage_path, {
                "key": key, "provider": router.label, "phase": "judge", "model": base.JUDGE_MODEL,
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "duration_ms": duration_ms, "finish_reason": finish_reason, "attempt": attempt,
            })
        if content.strip() and finish_reason != "length":
            with router.lock:
                if key not in router.values:
                    router.values[key] = content
                    append_jsonl(router.cache_path, {"key": key, "value": content, "finish_reason": finish_reason})
            return {"content": content, "finish_reason": finish_reason, "attempts": attempt}
    raise RuntimeError(
        f"judge returned empty or length-cut content {JUDGE_EMPTY_RETRIES} times "
        f"(last finish_reason={finish_reason!r}); refusing to score it as WRONG"
    )


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


def _shard_manifest(topic: str) -> dict:
    return {"source_manifest": base._source_manifest(), "memory_shard": topic}


def _build_memory_shard(topic: str) -> dict:
    """Process one topic chronologically; separate processes isolate mutable prototype globals."""
    load_dotenv(base.ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {"topic": topic, "status": "error", "error": "OPENAI_API_KEY missing"}
    shard_dir = SHARD_ROOT / topic
    shard_dir.mkdir(parents=True, exist_ok=True)
    try:
        direct = CachedAPI(shard_dir, api_key=key, label="openai")
        frame, _raw, _lookup = base.load_messages()
        frame = frame[frame.network_id.astype(str) == topic].copy()
        build_episode_snapshot(frame, shard_dir, direct, source_manifest=_shard_manifest(topic))
        return {"topic": topic, "status": "ready"}
    except RateLimitError as error:
        return {"topic": topic, "status": "rate_limit", "error": str(error)}
    except Exception as error:  # returned to the parent with the topic attached
        return {"topic": topic, "status": "error", "error": repr(error)}


def _merge_stats(metas: list[dict]) -> dict:
    totals = Counter()
    rejected = Counter()
    clusters = {}
    for meta in metas:
        stats = meta.get("stats", {})
        for key in (
            "networks", "sessions_processed", "events_total", "events_rejected",
            "episodes_total", "extraction_calls", "attach_calls", "total_new_calls",
            "evidence_attempted", "evidence_valid",
        ):
            totals[key] += int(stats.get(key, 0))
        rejected.update(stats.get("rejected_reasons", {}))
        clusters.update(stats.get("clusters_by_network", {}))
    return {**dict(totals), "rejected_reasons": dict(rejected), "clusters_by_network": clusters}


def _merge_memory_shards(*, memory_workers: int) -> None:
    episodes, decisions, metas, usage = [], [], [], []
    shard_hashes = {}
    for topic in base.BATCHES:
        shard_dir = SHARD_ROOT / topic
        meta = verify_episode_snapshot(shard_dir, _shard_manifest(topic))
        metas.append(meta)
        shard_rows = jsonl(shard_dir / "episodes.jsonl")
        episodes.extend(shard_rows)
        decisions.extend(jsonl(shard_dir / "attach_decisions.jsonl"))
        usage.extend(jsonl(shard_dir / "api_usage.jsonl"))
        shard_hashes[topic] = sha256_file(shard_dir / "episodes.jsonl")

    episode_payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in episodes)
    decision_payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions)
    base.RUN_DIR.mkdir(parents=True, exist_ok=True)
    (base.RUN_DIR / "episodes.jsonl").write_text(episode_payload)
    (base.RUN_DIR / "attach_decisions.jsonl").write_text(decision_payload)

    # Keep root preflight usage, then add each actually billed shard call exactly once.
    root_usage = jsonl(base.RUN_DIR / "api_usage.jsonl")
    usage_payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in root_usage + usage)
    (base.RUN_DIR / "api_usage.jsonl").write_text(usage_payload)

    frame, _raw, _lookup = base.load_messages()
    (base.RUN_DIR / "episodes_meta.json").write_text(json.dumps({
        "schema_version": "paper_external_temporal_episodes_v1",
        "memory_schema_version": metas[0]["memory_schema_version"],
        "extraction_and_attach_model": EXTRACTION_MODEL,
        "network_count": len(base.BATCHES),
        "session_count": int(frame[["network_id", "session_index"]].drop_duplicates().shape[0]),
        "turn_count": len(frame),
        "episode_count": len(episodes),
        "stats": _merge_stats(metas),
        "episodes_sha256": sha256_bytes(episode_payload.encode()),
        "qa_read": False,
        "source_manifest": base._source_manifest(),
        "execution_runner": RUNNER.name,
        "execution_runner_sha256": sha256_file(RUNNER),
        "memory_workers": memory_workers,
        "shard_sha256": shard_hashes,
    }, ensure_ascii=False, indent=2) + "\n")
    verify_episode_snapshot(base.RUN_DIR, base._source_manifest())


def prepare_memory(*, workers: int) -> None:
    if (base.RUN_DIR / "episodes.jsonl").exists():
        meta = verify_episode_snapshot(base.RUN_DIR, base._source_manifest())
        log(base.RUN_DIR, f"parallel memory: reuse sealed {meta['episode_count']} episodes")
        return
    log(base.RUN_DIR, f"parallel memory: {len(base.BATCHES)} topic shards (workers={workers}); QA unopened")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(base.BATCHES)), mp_context=context) as pool:
        results = list(pool.map(_build_memory_shard, base.BATCHES))
    for result in results:
        if result["status"] == "rate_limit":
            log(base.RUN_DIR, f"memory shard {result['topic']} rate limited; completed shards are cached")
            raise SystemExit(75)
        if result["status"] != "ready":
            raise RuntimeError(f"memory shard {result['topic']} failed: {result['error']}")
    _merge_memory_shards(memory_workers=workers)
    log(base.RUN_DIR, "parallel memory: sealed")


def _load_episode_docs(raw_by_id):
    result = {topic: [] for topic in base.BATCHES}
    for row in jsonl(base.RUN_DIR / "episodes.jsonl"):
        result[row["network_id"]].append(episode_doc(row, raw_by_id))
    return result


def run_inference(direct, router, *, workers: int) -> None:
    verify_episode_snapshot(base.RUN_DIR, base._source_manifest())
    _frame, raw_by_topic, raw_by_id = base.load_messages()
    episodes = _load_episode_docs(raw_by_id)
    embeddings = EmbeddingCache(base.RUN_DIR, direct)
    prompts = base._prompts()["answer"]
    done = {(row["qa_key"], row["condition"]) for row in jsonl(base.RUN_DIR / "predictions.jsonl")}
    questions = base.load_questions(include_gold=False)
    pending = [(row, condition) for row in questions for condition in base.CONDITIONS
               if (row["qa_key"], condition) not in done]
    log(base.RUN_DIR, f"parallel inference: {len(pending)} pending / {len(done)} cached")

    jobs = []
    for topic in base.BATCHES:
        selected = [(row, condition) for row, condition in pending if row["topic"] == topic]
        if not selected:
            continue
        indexes = {
            "RAW": DenseIndex(raw_by_topic[topic], embeddings, f"{topic}-raw"),
            "RAW+EPISODES": DenseIndex(
                raw_by_topic[topic] + episodes.get(topic, []), embeddings, f"{topic}-raw-episodes"
            ),
        }
        unique = {row["qa_key"]: row for row, _condition in selected}
        vectors = embeddings.all([row["question"] for row in unique.values()])
        vector_by_key = dict(zip(unique, vectors))
        for row, condition in selected:
            docs, retrieval_ms = indexes[condition].search_vector(vector_by_key[row["qa_key"]], TOP_K)
            context = "\n".join(f"- {doc.rendered}" for doc in docs) or "(No memories retrieved)"
            if row["question_type"] == "multiple_choice":
                prompt = prompts["multiple_choice"].format(
                    context=context, question=row["question"], options=base._option_text(row["options"]),
                )
            else:
                prompt = prompts["open_ended"].format(context=context, question=row["question"])
            jobs.append((row, condition, docs, retrieval_ms, prompt))

    def work(job):
        row, condition, docs, retrieval_ms, prompt = job
        started = time.perf_counter()
        raw_answer = router.chat(
            model=base.ANSWER_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
            max_tokens=1000, phase="answer", extra_body=base.PROVIDER["answer"],
        )
        return {
            **row, "condition": condition,
            "answer": official_answer(raw_answer, row["question_type"]), "answer_raw": raw_answer,
            "retrieved_ids": [doc.doc_id for doc in docs],
            "exposed_source_ids": list(dict.fromkeys(source for doc in docs for source in doc.source_ids)),
            "retrieval_ms": retrieval_ms, "answer_wall_ms": (time.perf_counter() - started) * 1000,
        }

    _parallel(jobs, work, lambda row: append_jsonl(base.RUN_DIR / "predictions.jsonl", row),
              workers=workers, label="parallel inference")
    predictions = jsonl(base.RUN_DIR / "predictions.jsonl")
    seal_jsonl(
        base.RUN_DIR / "predictions.jsonl", base.RUN_DIR / "predictions_meta.json",
        expected_rows=len(questions) * len(base.CONDITIONS), identity_fields=("qa_key", "condition"),
        metadata={
            "benchmark": "EverMemBench-Dynamic", "code_commit": base.CODE_COMMIT,
            "data_commit": base.DATA_COMMIT, "conditions": list(base.CONDITIONS),
            "answer_model": base.ANSWER_MODEL, "embedding_model": EMBED_MODEL, "top_k": TOP_K,
            "episode_snapshot_sha256": sha256_file(base.RUN_DIR / "episodes.jsonl"),
            "answer_postprocessing": "official answerer.py (_parse_mc_answer / [EMPTY] marker)",
            "answer_failure_markers": answer_failure_markers(predictions),
            "execution_runner": RUNNER.name, "execution_runner_sha256": sha256_file(RUNNER),
            "workers": workers,
        },
    )


def score(router, *, workers: int) -> None:
    prompts = base._prompts()["llm_judge"]
    questions = base.load_questions(include_gold=False)
    expected = len(questions) * len(base.CONDITIONS)
    verify_sealed_jsonl(base.RUN_DIR / "predictions.jsonl", base.RUN_DIR / "predictions_meta.json",
                        expected_rows=expected)
    predictions = {(row["qa_key"], row["condition"]): row
                   for row in jsonl(base.RUN_DIR / "predictions.jsonl")}
    gold = {row["qa_key"]: row for row in base.load_questions(include_gold=True)}
    evaluation_manifest = base._evaluation_manifest()
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
        if truth["question_type"] == "multiple_choice":
            correct = official_evaluator_mc(prediction["answer"], truth["gold"])
            rationale, judge_meta = "official exact multiple-choice comparison", {}
        else:
            prompt = prompts["user_prompt"].format(
                question=truth["question"], golden_answer=truth["gold"],
                generated_answer=prediction["answer"],
            )
            call = judge_chat(router, [{"role": "system", "content": prompts["system_prompt"]},
                                       {"role": "user", "content": prompt}])
            correct, parse_branch = official_judge_label(call["content"])
            rationale = call["content"]
            judge_meta = {
                "judge_finish_reason": call["finish_reason"], "judge_attempts": call["attempts"],
                "judge_parse": parse_branch,
            }
        exposed = set(prediction["exposed_source_ids"])
        evidence = set(truth["gold_source_ids"])
        return {
            "qa_key": prediction["qa_key"], "topic": prediction["topic"],
            "major": prediction["major"], "minor": prediction["minor"],
            "question_type": prediction["question_type"], "condition": prediction["condition"],
            "correct": int(correct), "judge": rationale, **judge_meta,
            "evidence_recall": len(exposed & evidence) / len(evidence) if evidence else 1.0,
            "evidence_precision": len(exposed & evidence) / len(exposed) if exposed else 0.0,
        }

    _parallel(pending, work, lambda row: append_jsonl(base.RUN_DIR / "results.jsonl", row),
              workers=workers, label="parallel score")
    seal_jsonl(
        base.RUN_DIR / "results.jsonl", base.RUN_DIR / "results_meta.json",
        expected_rows=expected, identity_fields=("qa_key", "condition"),
        metadata={
            "benchmark": "EverMemBench-Dynamic", "code_commit": base.CODE_COMMIT,
            "data_commit": base.DATA_COMMIT, "conditions": list(base.CONDITIONS),
            "judge_model": base.JUDGE_MODEL, "judge_call": JUDGE_CALL_VERSION,
            "judge_health": judge_health(jsonl(base.RUN_DIR / "results.jsonl")),
            "predictions_sha256": sha256_file(base.RUN_DIR / "predictions.jsonl"),
            "evaluation_input_manifest_sha256": sha256_file(manifest_path),
            "execution_runner": RUNNER.name, "execution_runner_sha256": sha256_file(RUNNER),
            "workers": workers,
        },
    )
    base.render_report()
    _append_measurement_health()


def answer_failure_markers(predictions: list[dict]) -> dict[str, dict[str, int]]:
    markers = ("[EMPTY]", "[INVALID]", "[FAILED]")
    return {
        condition: {
            marker: sum(1 for row in predictions if row["condition"] == condition and row["answer"] == marker)
            for marker in markers
        }
        for condition in base.CONDITIONS
    }


def judge_health(results: list[dict]) -> dict[str, dict]:
    health = {}
    for condition in base.CONDITIONS:
        judged = [row for row in results if row["condition"] == condition and "judge_parse" in row]
        health[condition] = {
            "open_ended_judged": len(judged),
            "parse": dict(Counter(row["judge_parse"] for row in judged)),
            "finish_reason": dict(Counter(row["judge_finish_reason"] for row in judged)),
            "re_requested": sum(1 for row in judged if row["judge_attempts"] > 1),
        }
    return health


def _append_measurement_health() -> None:
    markers = answer_failure_markers(jsonl(base.RUN_DIR / "predictions.jsonl"))
    health = judge_health(jsonl(base.RUN_DIR / "results.jsonl"))
    lines = [
        "", "## Measurement health", "",
        "Answer failure markers follow the official answerer and always score WRONG. Judge rows record "
        "finish_reason, attempts and which official parser branch produced the verdict; empty or "
        "length-cut judge output is re-requested instead of scored (see amendment "
        "evermembench_official_scoring_20260914).", "",
    ]
    for condition in base.CONDITIONS:
        lines.append(f"- {condition}: answer markers `{json.dumps(markers[condition])}`; "
                     f"judge `{json.dumps(health[condition], sort_keys=True)}`")
    with (base.RUN_DIR / "report.md").open("a") as report:
        report.write("\n".join(lines) + "\n")


def check_routes(router) -> None:
    checks = (
        (base.ANSWER_MODEL, base.PROVIDER["answer"]),
        (base.JUDGE_MODEL, base.PROVIDER["judge"]),
    )
    for model, provider in checks:
        result = router.chat(
            model=model, messages=[{"role": "user", "content": "Reply with OK only."}],
            temperature=0, max_tokens=8, phase="preflight", extra_body=provider,
        )
        if not result.strip():
            raise RuntimeError(f"empty preflight response from {model}")
        log(base.RUN_DIR, f"OpenRouter route OK: {model}")


def estimate_cost() -> None:
    print("EverMemBench expected uncached API budget (USD):")
    print("  OpenAI memory extraction+attachment: $3.00-$6.00")
    print("  OpenAI text-embedding-3-large:       $0.50-$1.20")
    print("  OpenRouter 4,800 GPT-4.1-mini answers: $3.00-$6.00")
    print("  OpenRouter 1,524 Gemini judges:        $0.50-$1.50")
    print("  Expected total: $7.00-$14.70; safe reserve: $18")
    print("  Multiple-choice scoring is local and costs $0.")


def status() -> None:
    base.status()
    for topic in base.BATCHES:
        shard = SHARD_ROOT / topic
        event_count = len(jsonl(shard / "event_extraction_cache.jsonl"))
        attach_count = len(jsonl(shard / "episode_attach_cache.jsonl"))
        ready = (shard / "episodes.jsonl").exists()
        print(f"memory_shard[{topic}]=ready:{ready} extraction_cache:{event_count} attach_cache:{attach_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-workers", type=int, default=5)
    parser.add_argument("--answer-workers", type=int, default=16)
    parser.add_argument("--judge-workers", type=int, default=32)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("all", "prepare-memory", "run-inference", "score", "preflight", "status", "estimate-cost"):
        modes.add_argument(f"--{mode}", action="store_true")
    args = parser.parse_args()
    if args.status:
        status(); return
    if args.estimate_cost:
        estimate_cost(); return
    if not 1 <= args.memory_workers <= len(base.BATCHES):
        parser.error(f"--memory-workers must be between 1 and {len(base.BATCHES)}")
    if not 1 <= args.answer_workers <= 64 or not 1 <= args.judge_workers <= 64:
        parser.error("answer/judge workers must be between 1 and 64")
    try:
        direct, router = base.preflight()
        check_routes(router)
        if args.preflight:
            return
        if args.all or args.prepare_memory:
            prepare_memory(workers=args.memory_workers)
        if args.all or args.run_inference:
            run_inference(direct, router, workers=args.answer_workers)
        if args.all or args.score:
            score(router, workers=args.judge_workers)
    except RateLimitError as error:
        log(base.RUN_DIR, f"rate limit; completed parallel work is cached: {error}")
        raise SystemExit(75) from None
    except AuthenticationError as error:
        raise SystemExit(f"API authentication failed: {error}") from None


if __name__ == "__main__":
    main()
