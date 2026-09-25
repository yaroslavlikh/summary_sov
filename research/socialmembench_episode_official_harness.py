"""Evaluate sealed TemporalEpisodes in the frozen SocialMemBench harness.

The three original conditions are read from the immutable official-v1 run.
Only RAW+EPISODES inference and scoring are new.  Retrieval, embeddings,
top-k, context packing, answer prompt, answer model, and judge are reused
verbatim from research.socialmembench_full_run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import AuthenticationError, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import socialmembench_full_run as official
from research.socialmembench_episode_full_run import _episode_candidate
from research.socialmembench_pilot import _json, build_raw_candidates


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / ".research_runs" / "frozen" / "socialmembench_full_official_v1_20260911_184037_MSK"
FROZEN_RUN = FROZEN / "run"
DATA_DIR = FROZEN / "dataset"
EPISODE_RUN = ROOT / ".research_runs" / "socialmembench_temporal_episodes_full_v1"
RUN_DIR = ROOT / ".research_runs" / "socialmembench_temporal_episodes_official_harness_v1"

EPISODES = EPISODE_RUN / "episodes.jsonl"
EPISODES_META = EPISODE_RUN / "episodes_meta.json"
API_CACHE = RUN_DIR / "openai_cache.jsonl"
EMBED_DIR = RUN_DIR / "embeddings"
PREDICTIONS = RUN_DIR / "episode_predictions.jsonl"
PREDICTIONS_META = RUN_DIR / "episode_predictions_meta.json"
RESULTS = RUN_DIR / "episode_results.jsonl"
REPORT = RUN_DIR / "report.md"
RUN_LOG = RUN_DIR / "run.log"

CONDITION = "RAW+EPISODES"
ALL_CONDITIONS = (*official.CONDITIONS, CONDITION)
SCHEMA_VERSION = "temporal_episodes_official_harness_v1"
ANSWER_WORKERS = official.ANSWER_WORKERS
JUDGE_WORKERS = official.JUDGE_WORKERS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()


def _log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, file=sys.stderr, flush=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a") as out:
        out.write(line + "\n")


def _verify_frozen_inputs() -> None:
    manifest = json.loads((FROZEN / "MANIFEST.json").read_text())
    required = (
        "dataset/conversations.parquet", "dataset/qa.parquet", "dataset/networks.parquet",
        "run/predictions.jsonl", "run/predictions_meta.json", "run/results.jsonl",
    )
    for relative in required:
        path = FROZEN / relative
        expected = manifest["files"][relative]["sha256"]
        if _sha256_file(path) != expected:
            raise RuntimeError(f"frozen input checksum mismatch: {relative}")
    episode_meta = json.loads(EPISODES_META.read_text())
    if _sha256_file(EPISODES) != episode_meta["episodes_sha256"]:
        raise RuntimeError("sealed episode snapshot checksum mismatch")


def _configure_api() -> None:
    official.RUN_DIR = RUN_DIR
    official.API_CACHE = API_CACHE
    official.EMBED_DIR = EMBED_DIR
    if not isinstance(official._api, official.CachedOpenAI) or official._api.cache_path != API_CACHE:
        official._api = official.CachedOpenAI(API_CACHE)


def preflight() -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY отсутствует в .env")
    _verify_frozen_inputs()
    _configure_api()
    official.api().client.models.retrieve(official.MODEL)
    _log("preflight OK: frozen baseline + sealed episodes verified")


def _episodes_by_network() -> dict[str, list]:
    result = defaultdict(list)
    for row in _jsonl(EPISODES):
        result[row["network_id"]].append(_episode_candidate(row))
    return dict(result)


def run_inference() -> None:
    conversations = pd.read_parquet(DATA_DIR / "conversations.parquet")
    qa = pd.read_parquet(DATA_DIR / "qa.parquet", columns=[
        "qa_id", "network_id", "query_type", "question", "answer_format", "options_json",
    ])
    raw_by_network, raw_by_source = build_raw_candidates(conversations)
    episodes_by_network = _episodes_by_network()
    session_index = {str(row.turn_id): int(row.session_index) for row in conversations.itertuples(index=False)}
    done = {row["qa_key"] for row in _jsonl(PREDICTIONS)}
    pending = [row for row in qa.itertuples(index=False)
               if official.qa_key(str(row.network_id), str(row.qa_id)) not in done]
    _log(f"inference: {len(pending)} pending / {len(done)} cached")
    if not pending:
        return

    indices = {}
    for number, network_id in enumerate(sorted(raw_by_network), 1):
        indices[network_id] = official.CosineIndex.build(
            raw_by_network[network_id] + episodes_by_network.get(network_id, []),
            f"{network_id}-raw-temporal-episodes",
        )
        if number % 10 == 0 or number == 43:
            _log(f"inference: embedded/indexed {number}/43 networks")

    question_vectors = []
    questions = [str(row.question) for row in pending]
    for start in range(0, len(questions), 256):
        question_vectors.extend(official._cached_embedding_batch(questions[start:start + 256]))

    def answer_one(item: tuple[int, Any]) -> dict:
        index, row = item
        network_id = str(row.network_id)
        retrieved = indices[network_id].search(question_vectors[index])
        context, exposed = official.pack_context(retrieved, raw_by_source, session_index)
        options = _json(row.options_json, {})
        answer = official.api().chat(
            official.answer_prompt(str(row.question), str(row.answer_format), options, context)
        )
        return {
            "qa_key": official.qa_key(network_id, str(row.qa_id)), "qa_id": str(row.qa_id),
            "network_id": network_id, "query_type": str(row.query_type),
            "answer_format": str(row.answer_format), "answer": answer,
            "retrieved_ids": [candidate.candidate_id for candidate in retrieved],
            "exposed_source_ids": exposed,
        }

    completed = 0
    indexed_pending = list(enumerate(pending))
    for offset in range(0, len(indexed_pending), ANSWER_WORKERS):
        failures = []
        with ThreadPoolExecutor(max_workers=ANSWER_WORKERS) as pool:
            futures = [pool.submit(answer_one, item) for item in indexed_pending[offset:offset + ANSWER_WORKERS]]
            for future in as_completed(futures):
                try:
                    _append_jsonl(PREDICTIONS, future.result())
                    completed += 1
                except Exception as error:
                    failures.append(error)
        if completed and (completed % 25 == 0 or completed == len(pending)):
            _log(f"inference: answered {completed}/{len(pending)} pending")
        if failures:
            raise failures[0]

    rows = _jsonl(PREDICTIONS)
    if len(rows) != 1031 or len({row["qa_key"] for row in rows}) != 1031:
        raise RuntimeError(f"episode predictions incomplete/duplicated: {len(rows)}")
    PREDICTIONS_META.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "condition": CONDITION,
        "model": official.MODEL, "temperature": 0,
        "embedding_model": official.EMBED_MODEL, "top_k": official.TOP_K,
        "answer_prompt": "research.socialmembench_full_run.answer_prompt",
        "episodes_sha256": _sha256_file(EPISODES),
        "predictions_sha256": _sha256_file(PREDICTIONS), "gold_read": False,
    }, indent=2))
    _log("inference sealed: 1,031 episode predictions")


def score() -> None:
    meta = json.loads(PREDICTIONS_META.read_text())
    if _sha256_file(PREDICTIONS) != meta["predictions_sha256"]:
        raise RuntimeError("episode prediction checksum mismatch")
    predictions = {row["qa_key"]: row for row in _jsonl(PREDICTIONS)}
    qa = pd.read_parquet(DATA_DIR / "qa.parquet")
    networks = pd.read_parquet(DATA_DIR / "networks.parquet", columns=["network_id", "tier"])
    tier_by_network = dict(zip(networks.network_id.astype(str), networks.tier.astype(str)))
    finished = {row["qa_key"] for row in _jsonl(RESULTS)}
    jobs = []
    for row in qa.itertuples(index=False):
        key = official.qa_key(str(row.network_id), str(row.qa_id))
        if key in finished:
            continue
        prediction = predictions[key]
        exposed = set(prediction["exposed_source_ids"])
        gold_ids = {str(a["turn_id"]) for a in _json(row.evidence_anchors_json, []) if a.get("turn_id")}
        base = {
            "qa_key": key, "qa_id": str(row.qa_id), "network_id": str(row.network_id),
            "tier": tier_by_network[str(row.network_id)], "query_type": str(row.query_type),
            "condition": CONDITION, "answer": prediction["answer"],
            "evidence_recall": len(exposed & gold_ids) / len(gold_ids) if gold_ids else 1.0,
            "evidence_precision": len(exposed & gold_ids) / len(exposed) if exposed else 0.0,
        }
        if str(row.answer_format) == "multiple_choice":
            value, rationale = official._mc_score(
                base["answer"], str(row.correct_option), _json(row.options_json, {})
            )
            _append_jsonl(RESULTS, {**base, "score": value, "rationale": rationale, "scoring": "exact_mc"})
        else:
            jobs.append((base, official.JUDGE_PROMPT.format(
                question=str(row.question), gold=str(row.answer), answer=base["answer"]
            )))
    _log(f"score: {len(jobs)} pending open-answer judges; MC scored exactly")

    def judge(job: tuple[dict, str]) -> dict:
        base, prompt = job
        parsed = official._parse_json_object(official.api().chat(prompt, json_output=True)) or {}
        try:
            value = min(1.0, max(0.0, float(parsed.get("score", 0.0))))
        except (TypeError, ValueError):
            value = 0.0
        return {**base, "score": value, "rationale": str(parsed.get("rationale", "")), "scoring": "llm_judge"}

    completed = 0
    for offset in range(0, len(jobs), JUDGE_WORKERS):
        failures = []
        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
            futures = [pool.submit(judge, job) for job in jobs[offset:offset + JUDGE_WORKERS]]
            for future in as_completed(futures):
                try:
                    _append_jsonl(RESULTS, future.result())
                    completed += 1
                except Exception as error:
                    failures.append(error)
        if completed and (completed % 50 == 0 or completed == len(jobs)):
            _log(f"score: judged {completed}/{len(jobs)} pending")
        if failures:
            raise failures[0]

    rows = _jsonl(RESULTS)
    if len(rows) != 1031 or len({row["qa_key"] for row in rows}) != 1031:
        raise RuntimeError(f"episode results incomplete/duplicated: {len(rows)}")
    REPORT.write_text(render_report(_jsonl(FROZEN_RUN / "results.jsonl") + rows))
    _log(f"final comparable report ready: {REPORT}")


def _mean(rows: list[dict], condition: str, field: str) -> float:
    values = [float(row[field]) for row in rows if row["condition"] == condition]
    return sum(values) / len(values)


def render_report(rows: list[dict]) -> str:
    lines = [
        "# SocialMemBench — TemporalEpisodes in the frozen official-v1 harness", "",
        "The RAW/FLAT/VERSIONED rows are immutable results from the frozen 2026-09-11 run. "
        "Only RAW+EPISODES is newly generated and scored.", "",
        f"- model: `{official.MODEL}`", f"- embeddings: `{official.EMBED_MODEL}`",
        f"- retrieval: one cosine top-{official.TOP_K} over each condition's corpus",
        "- answer prompt, judge prompt, context packing, provenance expansion: identical code path", "",
        "## Overall", "", "| condition | MeanQ | MeanN | 95% network CI | evidence precision | evidence recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ALL_CONDITIONS:
        net = official._network_means(rows, condition)
        lo, hi = official.bootstrap_network_ci(net)
        lines.append(f"| {condition} | {_mean(rows, condition, 'score'):.3f} | "
                     f"{sum(net.values())/len(net):.3f} | [{lo:.3f}, {hi:.3f}] | "
                     f"{_mean(rows, condition, 'evidence_precision'):.3f} | "
                     f"{_mean(rows, condition, 'evidence_recall'):.3f} |")
    lines += ["", "## Paired network deltas", ""]
    left = official._network_means(rows, CONDITION)
    for right_name in official.CONDITIONS:
        right = official._network_means(rows, right_name)
        deltas = [left[key] - right[key] for key in sorted(left)]
        lo, hi = official.bootstrap_paired_ci(rows, CONDITION, right_name)
        lines.append(f"- {CONDITION} vs {right_name}: {sum(d>0 for d in deltas)}W/"
                     f"{sum(d==0 for d in deltas)}T/{sum(d<0 for d in deltas)}L; "
                     f"delta={sum(deltas)/len(deltas):+.3f}; 95% CI [{lo:+.3f}, {hi:+.3f}]")
    lines += ["", "## By question type", "", "| type | n | RAW | FLAT | VERSIONED | EPISODES |",
              "|---|---:|---:|---:|---:|---:|"]
    for query_type in sorted({row["query_type"] for row in rows}):
        selected = [row for row in rows if row["query_type"] == query_type]
        n = len(selected) // len(ALL_CONDITIONS)
        values = [_mean(selected, condition, "score") for condition in ALL_CONDITIONS]
        lines.append(f"| {query_type} | {n} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += ["", "## Published paper reference", "", "| system | MeanQ |", "|---|---:|"]
    for name, value in official.PUBLISHED.items():
        lines.append(f"| {name} | {value:.3f} |")
    lines += ["", "## Protocol guardrails", "",
              "- The episode snapshot was constructed query-independently and sealed before QA inference.",
              "- Frozen inputs are checksum-verified on every process start.",
              "- This is an internally controlled adapter comparison, not an official leaderboard submission."]
    return "\n".join(lines) + "\n"


def status() -> None:
    print(f"run_dir={RUN_DIR}")
    print(f"episode_predictions={len(_jsonl(PREDICTIONS))}/1031")
    print(f"episode_scores={len(_jsonl(RESULTS))}/1031")
    print(f"report={REPORT if REPORT.exists() else 'not ready'}")


def run_all() -> None:
    run_inference()
    score()
    _log("DONE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group(required=True)
    phases.add_argument("--all", action="store_true")
    phases.add_argument("--run-inference", action="store_true")
    phases.add_argument("--score", action="store_true")
    phases.add_argument("--preflight", action="store_true")
    phases.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
        return
    try:
        preflight()
        if args.preflight:
            return
        if args.all:
            run_all()
        elif args.run_inference:
            run_inference()
        else:
            score()
    except RateLimitError as error:
        _log(f"rate limit; completed work is cached: {error}")
        raise SystemExit(75) from None
    except AuthenticationError as error:
        raise SystemExit(f"OpenAI authentication failed: {error}") from None


if __name__ == "__main__":
    main()
