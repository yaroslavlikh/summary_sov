"""Full SocialMemBench run for query-independent TemporalEpisode memory.

One resumable overnight command is printed by ``--status``.  The run is
isolated from every earlier SocialMemBench artifact and compares exactly two
conditions under the same retrieval/context budget:

    RAW               hybrid raw-turn retrieval
    RAW+EPISODES      the same raw retrieval plus TemporalEpisode retrieval

Memory construction never reads QA data.  Episodes are extracted from all 348
sessions in chronological order, linked with the frozen temporal-episode
prototype, and sealed before questions are opened.  Derived episodes are
expanded back to their validated raw provenance before answer generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AuthenticationError, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import query_independent_episode_pipeline as qi
from research import socialmembench_full_run as official
from research import stage4_9_hybrid_experiment as hybrid
from research import temporal_episode_prototype as tep
from research.socialmembench_pilot import Candidate, _json, _normalized, build_raw_candidates, ensure_data


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/tmp/socialmembench")
RUN_DIR = ROOT / ".research_runs" / "socialmembench_temporal_episodes_full_v1"

MODEL = official.MODEL
CONDITIONS = ("RAW", "RAW+EPISODES")
SCHEMA_VERSION = "socialmembench_temporal_episodes_full_v1"
ANSWER_WORKERS = 5
JUDGE_WORKERS = 10

EVENT_CACHE = RUN_DIR / "event_extraction_cache.jsonl"
ATTACH_CACHE = RUN_DIR / "episode_attach_cache.jsonl"
EPISODES = RUN_DIR / "episodes.jsonl"
EPISODES_META = RUN_DIR / "episodes_meta.json"
DECISIONS = RUN_DIR / "attach_decisions.jsonl"
API_CACHE = RUN_DIR / "openai_cache.jsonl"
USAGE = RUN_DIR / "api_usage.jsonl"
PREDICTIONS = RUN_DIR / "predictions.jsonl"
PREDICTIONS_META = RUN_DIR / "predictions_meta.json"
RESULTS = RUN_DIR / "results.jsonl"
REPORT = RUN_DIR / "report.md"
RUN_LOG = RUN_DIR / "run.log"
TIMING = RUN_DIR / "timing.json"
EMBED_DIR = RUN_DIR / "retrieval_embeddings"

# Public list prices used only for a reproducible estimate in the report.
CHAT_INPUT_USD_PER_M = 0.15
CHAT_OUTPUT_USD_PER_M = 0.60
EMBED_USD_PER_M = 0.02

_phase = "preflight"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str | None:
    return _sha256(path.read_bytes()) if path.exists() else None


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
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, file=sys.stderr, flush=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a") as out:
        out.write(line + "\n")


def _timing_mark(name: str) -> None:
    values = json.loads(TIMING.read_text()) if TIMING.exists() else {}
    values.setdefault(name, datetime.now(timezone.utc).isoformat())
    TIMING.write_text(json.dumps(values, indent=2))


class MeteredOpenAI(official.CachedOpenAI):
    """The existing content cache plus token accounting for the paper."""

    def __init__(self) -> None:
        super().__init__(API_CACHE)
        self.usage_keys = {row["key"] for row in _jsonl(USAGE)}

    def _record(self, key: str, kind: str, input_tokens: int, output_tokens: int = 0) -> None:
        if key in self.usage_keys:
            return
        self.usage_keys.add(key)
        _append_jsonl(USAGE, {
            "key": key, "phase": _phase, "kind": kind, "model": MODEL if kind == "chat" else official.EMBED_MODEL,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
        })

    def chat(self, prompt: str, *, json_output: bool = False) -> str:
        payload = {"model": MODEL, "temperature": 0, "json": json_output, "prompt": prompt}
        key = "chat:" + _sha256(json.dumps(payload, sort_keys=True).encode())
        with self.lock:
            cached = self.values.get(key)
        if cached is not None:
            return str(cached)
        kwargs: dict[str, Any] = {
            "model": MODEL, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        value = response.choices[0].message.content or ""
        usage = response.usage
        with self.lock:
            if key not in self.values:
                self.values[key] = value
                _append_jsonl(API_CACHE, {"key": key, "value": value})
                self._record(key, "chat", int(usage.prompt_tokens or 0), int(usage.completion_tokens or 0))
        return value

    def embeddings(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 1536), dtype=np.float32)
        key = "embedding:" + _sha256(json.dumps(texts, ensure_ascii=False).encode())
        response = self.client.embeddings.create(model=official.EMBED_MODEL, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        with self.lock:
            self._record(key, "embedding", int(response.usage.total_tokens or 0))
        return _normalized([item.embedding for item in ordered])


def api() -> MeteredOpenAI:
    if not isinstance(official._api, MeteredOpenAI):
        official._api = MeteredOpenAI()
    return official._api


class _OpenAIChatModel:
    def invoke(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content=api().chat(prompt, json_output=True))


def _configure_shared_modules() -> None:
    official.RUN_DIR = RUN_DIR
    official.API_CACHE = API_CACHE
    official.EMBED_DIR = RUN_DIR / "openai_embeddings"
    hybrid.EMBED_CACHE_DIR = EMBED_DIR
    qi.QI_EVENT_CACHE = EVENT_CACHE
    qi.QI_ATTACH_CACHE = ATTACH_CACHE
    tep.EVENT_CACHE = EVENT_CACHE
    tep.ATTACH_CACHE = ATTACH_CACHE
    import llm.groq_client as provider
    provider.get_chat_model = lambda *_args, **_kwargs: _OpenAIChatModel()


def preflight() -> None:
    global _phase
    _phase = "preflight"
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY отсутствует в .env")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ensure_data(DATA_DIR)
    _configure_shared_modules()
    api().client.models.retrieve(MODEL)
    _log("preflight OK; no benchmark phase was run")


def _episode_row(ep: tep.TemporalEpisode) -> dict:
    return {
        "episode_id": ep.episode_id, "network_id": ep.network_id,
        "viewpoint_owner": ep.viewpoint_owner, "subject": ep.subject,
        "events": [{"decision": ee.decision, "event": asdict(ee.event)} for ee in ep.events],
    }


def _episode_candidate(row: dict) -> Candidate:
    events = row["events"]
    active = events[-tep.MAX_HOT_EVENTS:]
    source_ids = tuple(dict.fromkeys(
        turn_id for item in events for turn_id in item["event"]["source_turn_ids"]
    ))
    latest = events[-1]["event"] if events else {}
    return Candidate(
        candidate_id=f"episode:{row['network_id']}:{row['episode_id']}",
        kind="derived:temporal_episode",
        text=" || ".join(item["event"]["event_text"] for item in active),
        source_ids=source_ids,
        asserted_by=(row["viewpoint_owner"],) if row.get("viewpoint_owner") else (),
        entities=(row["subject"],) if row.get("subject") else (),
        observed_at=str(latest.get("observed_at", "")),
    )


def prepare_memory() -> None:
    global _phase
    _phase = "memory"
    _timing_mark("memory_started_at")
    if EPISODES.exists() and EPISODES_META.exists():
        meta = json.loads(EPISODES_META.read_text())
        if _sha256_file(EPISODES) != meta.get("episodes_sha256"):
            raise RuntimeError("sealed episode snapshot checksum mismatch")
        _log(f"memory: reuse sealed {meta['episode_count']} episodes")
        return

    conversations = pd.read_parquet(DATA_DIR / "conversations.parquet")
    networks = tuple(sorted(conversations.network_id.astype(str).unique()))
    if len(networks) != 43 or conversations.session_id.nunique() != 348:
        raise RuntimeError("expected 43 networks / 348 sessions")
    episodes_by_network, events_by_network, decisions, stats = qi.run_qi_extraction_and_materialize(
        networks, conversations,
    )
    rows = [_episode_row(ep) for network_id in networks for ep in episodes_by_network.get(network_id, [])]
    payload = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n").encode()
    EPISODES.write_bytes(payload)
    DECISIONS.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in decisions) + "\n")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "memory_schema_version": tep.SCHEMA_VERSION,
        "extraction_and_attach_model": MODEL,
        "network_count": len(networks), "session_count": 348,
        "event_count": sum(len(items) for items in events_by_network.values()),
        "episode_count": len(rows), "stats": stats,
        "episodes_sha256": _sha256(payload), "qa_read": False,
    }
    EPISODES_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    _timing_mark("memory_completed_at")
    _log(f"memory sealed: {meta['event_count']} events -> {len(rows)} episodes")


def _episodes_by_network() -> dict[str, list[Candidate]]:
    meta = json.loads(EPISODES_META.read_text())
    if _sha256_file(EPISODES) != meta["episodes_sha256"]:
        raise RuntimeError("episode snapshot is not sealed")
    result: dict[str, list[Candidate]] = defaultdict(list)
    for row in _jsonl(EPISODES):
        result[row["network_id"]].append(_episode_candidate(row))
    return dict(result)


def run_inference() -> None:
    global _phase
    _phase = "inference"
    _timing_mark("inference_started_at")
    if not EPISODES_META.exists():
        raise RuntimeError("run --prepare-memory first")
    conversations = pd.read_parquet(DATA_DIR / "conversations.parquet")
    # No gold answer, correct option, evidence anchor, or temporal anchor is read here.
    qa = pd.read_parquet(DATA_DIR / "qa.parquet", columns=[
        "qa_id", "network_id", "query_type", "question", "answer_format", "options_json",
    ])
    raw_by_network, raw_by_source = build_raw_candidates(conversations)
    episodes_by_network = _episodes_by_network()
    done = {row["qa_key"] for row in _jsonl(PREDICTIONS)}
    pending = [row for row in qa.itertuples(index=False)
               if official.qa_key(str(row.network_id), str(row.qa_id)) not in done]
    _log(f"inference: {len(pending)} pending / {len(done)} cached")
    if not pending:
        return

    indices = {}
    for number, network_id in enumerate(sorted(raw_by_network), 1):
        indices[network_id] = hybrid.HybridIndex(
            raw_by_network[network_id], episodes_by_network.get(network_id, []), network_id, "temporal-episodes",
        )
        if number % 10 == 0 or number == 43:
            _log(f"inference: indexed {number}/43 networks")

    question_vectors = _normalized(__import__("embeddings").embed_batch([str(row.question) for row in pending]))

    def answer_one(item: tuple[int, Any]) -> dict:
        index, row = item
        network_id = str(row.network_id)
        qvec = question_vectors[index]
        raw_retrieved, episode_retrieved = indices[network_id].retrieve(str(row.question), qvec)
        pools = {"RAW": raw_retrieved, "RAW+EPISODES": raw_retrieved + episode_retrieved}
        answers, retrieved_ids, exposed_ids, context_words = {}, {}, {}, {}
        options = _json(row.options_json, {})
        for condition in CONDITIONS:
            ranked = hybrid.HybridIndex.rerank(qvec, pools[condition])
            context, packing = hybrid.pack_context(ranked, raw_by_source)
            answers[condition] = api().chat(
                official.answer_prompt(str(row.question), str(row.answer_format), options, context)
            )
            retrieved_ids[condition] = packing["candidate_ids"]
            exposed_ids[condition] = packing["source_ids"]
            context_words[condition] = packing["context_words"]
        return {
            "qa_key": official.qa_key(network_id, str(row.qa_id)), "qa_id": str(row.qa_id),
            "network_id": network_id, "query_type": str(row.query_type),
            "answer_format": str(row.answer_format), "answers": answers,
            "retrieved_ids": retrieved_ids, "exposed_source_ids": exposed_ids,
            "context_words": context_words,
        }

    completed = 0
    indexed_pending = list(enumerate(pending))
    for offset in range(0, len(indexed_pending), ANSWER_WORKERS):
        failures = []
        batch = indexed_pending[offset:offset + ANSWER_WORKERS]
        with ThreadPoolExecutor(max_workers=ANSWER_WORKERS) as pool:
            futures = [pool.submit(answer_one, item) for item in batch]
            for future in as_completed(futures):
                try:
                    _append_jsonl(PREDICTIONS, future.result())
                    completed += 1
                except Exception as error:
                    failures.append(error)
        if completed and (completed % 25 == 0 or completed == len(pending)):
            _log(f"inference: completed {completed}/{len(pending)} pending questions")
        if failures:
            raise failures[0]

    rows = _jsonl(PREDICTIONS)
    if len(rows) != 1031 or len({row["qa_key"] for row in rows}) != 1031:
        raise RuntimeError(f"predictions incomplete/duplicated: {len(rows)}")
    PREDICTIONS_META.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "model": MODEL, "temperature": 0,
        "retrieval": "hybrid raw + dense episodes + unified dense rerank",
        "raw_first_stage_k": hybrid.RAW_RETRIEVAL_LIMIT,
        "episode_first_stage_k": hybrid.MEMORY_RETRIEVAL_LIMIT,
        "context_word_budget": hybrid.CONTEXT_WORD_BUDGET,
        "conditions": CONDITIONS, "prediction_count": 1031,
        "predictions_sha256": _sha256_file(PREDICTIONS), "gold_read": False,
    }, indent=2))
    _timing_mark("inference_completed_at")
    _log("inference: sealed 1,031 paired predictions")


def score() -> None:
    global _phase
    _phase = "judge"
    _timing_mark("score_started_at")
    meta = json.loads(PREDICTIONS_META.read_text())
    if _sha256_file(PREDICTIONS) != meta.get("predictions_sha256"):
        raise RuntimeError("prediction checksum mismatch")
    predictions = {row["qa_key"]: row for row in _jsonl(PREDICTIONS)}
    qa = pd.read_parquet(DATA_DIR / "qa.parquet")
    networks = pd.read_parquet(DATA_DIR / "networks.parquet", columns=["network_id", "tier"])
    tier_by_network = dict(zip(networks.network_id.astype(str), networks.tier.astype(str)))
    finished = {(row["qa_key"], row["condition"]) for row in _jsonl(RESULTS)}
    jobs = []
    for row in qa.itertuples(index=False):
        key = official.qa_key(str(row.network_id), str(row.qa_id))
        prediction = predictions[key]
        options = _json(row.options_json, {})
        gold_ids = {str(a["turn_id"]) for a in _json(row.evidence_anchors_json, []) if a.get("turn_id")}
        for condition in CONDITIONS:
            if (key, condition) in finished:
                continue
            exposed = set(prediction["exposed_source_ids"][condition])
            base = {
                "qa_key": key, "qa_id": str(row.qa_id), "network_id": str(row.network_id),
                "tier": tier_by_network[str(row.network_id)], "query_type": str(row.query_type),
                "condition": condition, "answer": prediction["answers"][condition],
                "evidence_recall": len(exposed & gold_ids) / len(gold_ids) if gold_ids else 1.0,
                "evidence_precision": len(exposed & gold_ids) / len(exposed) if exposed else 0.0,
                "context_words": prediction["context_words"][condition],
            }
            if str(row.answer_format) == "multiple_choice":
                value, rationale = official._mc_score(base["answer"], str(row.correct_option), options)
                _append_jsonl(RESULTS, {**base, "score": value, "rationale": rationale, "scoring": "exact_mc"})
            else:
                jobs.append((base, official.JUDGE_PROMPT.format(
                    question=str(row.question), gold=str(row.answer), answer=base["answer"]
                )))
    _log(f"score: {len(jobs)} pending open-answer judge calls; MC scored exactly")

    def judge(job: tuple[dict, str]) -> dict:
        base, prompt = job
        parsed = official._parse_json_object(api().chat(prompt, json_output=True)) or {}
        try:
            value = min(1.0, max(0.0, float(parsed.get("score", 0.0))))
        except (TypeError, ValueError):
            value = 0.0
        return {**base, "score": value, "rationale": str(parsed.get("rationale", "")), "scoring": "llm_judge"}

    failures = []
    completed = 0
    for offset in range(0, len(jobs), JUDGE_WORKERS):
        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
            futures = [pool.submit(judge, job) for job in jobs[offset:offset + JUDGE_WORKERS]]
            for future in as_completed(futures):
                try:
                    _append_jsonl(RESULTS, future.result())
                    completed += 1
                except Exception as error:
                    failures.append(error)
        if completed and (completed % 50 == 0 or completed == len(jobs)):
            _log(f"score: judged {completed}/{len(jobs)} pending answers")
        if failures:
            raise failures[0]

    rows = _jsonl(RESULTS)
    expected = 1031 * len(CONDITIONS)
    if len(rows) != expected or len({(r["qa_key"], r["condition"]) for r in rows}) != expected:
        raise RuntimeError(f"results incomplete/duplicated: expected {expected}, got {len(rows)}")
    _timing_mark("score_completed_at")
    REPORT.write_text(render_report(rows))
    _log(f"final report ready: {REPORT}")


def _mean(rows: list[dict], condition: str, field: str) -> float:
    values = [float(row[field]) for row in rows if row["condition"] == condition]
    return sum(values) / len(values)


def _usage_summary() -> tuple[list[str], float]:
    rows = _jsonl(USAGE)
    lines = []
    total = 0.0
    for phase in ("memory", "inference", "judge"):
        selected = [r for r in rows if r.get("phase") == phase]
        input_tokens = sum(int(r.get("input_tokens", 0)) for r in selected)
        output_tokens = sum(int(r.get("output_tokens", 0)) for r in selected)
        chat_input = sum(int(r.get("input_tokens", 0)) for r in selected if r.get("kind") == "chat")
        embedding_input = sum(int(r.get("input_tokens", 0)) for r in selected if r.get("kind") == "embedding")
        cost = (chat_input * CHAT_INPUT_USD_PER_M + output_tokens * CHAT_OUTPUT_USD_PER_M
                + embedding_input * EMBED_USD_PER_M) / 1_000_000
        total += cost
        lines.append(f"| {phase} | {len(selected)} | {input_tokens:,} | {output_tokens:,} | ${cost:.3f} |")
    return lines, total


def render_report(rows: list[dict]) -> str:
    episode_meta = json.loads(EPISODES_META.read_text())
    usage_lines, total_cost = _usage_summary()
    lines = [
        "# SocialMemBench — full TemporalEpisode run", "",
        "Query-independent 43-network / 1,031-QA adapter run; not an official leaderboard submission.", "",
        f"- answer/judge/extraction model: `{MODEL}`",
        "- episode linking and query retrieval embeddings: `paraphrase-multilingual-MiniLM-L12-v2`",
        f"- retrieval: hybrid raw top-{hybrid.RAW_RETRIEVAL_LIMIT} + episode dense top-{hybrid.MEMORY_RETRIEVAL_LIMIT} "
        f"-> unified dense rerank -> {hybrid.CONTEXT_WORD_BUDGET}-word context",
        f"- memory: {episode_meta['event_count']} events -> {episode_meta['episode_count']} episodes", "",
        "## Overall", "", "| condition | MeanQ | MeanN | 95% network bootstrap CI | evidence precision | evidence recall | context words |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        net = official._network_means(rows, condition)
        lo, hi = official.bootstrap_network_ci(net)
        lines.append(
            f"| {condition} | {_mean(rows, condition, 'score'):.3f} | {sum(net.values())/len(net):.3f} | "
            f"[{lo:.3f}, {hi:.3f}] | {_mean(rows, condition, 'evidence_precision'):.3f} | "
            f"{_mean(rows, condition, 'evidence_recall'):.3f} | {_mean(rows, condition, 'context_words'):.1f} |"
        )
    left = official._network_means(rows, "RAW+EPISODES")
    right = official._network_means(rows, "RAW")
    deltas = [left[k] - right[k] for k in sorted(left)]
    lo, hi = official.bootstrap_paired_ci(rows, "RAW+EPISODES", "RAW")
    lines += ["", "## Paired network result", "", (
        f"- RAW+EPISODES vs RAW: {sum(d>0 for d in deltas)}W/{sum(d==0 for d in deltas)}T/"
        f"{sum(d<0 for d in deltas)}L networks; mean delta={sum(deltas)/len(deltas):+.3f}; "
        f"95% CI [{lo:+.3f}, {hi:+.3f}]"
    ), "", "## By question type", "", "| type | n | RAW | RAW+EPISODES |", "|---|---:|---:|---:|"]
    for query_type in sorted({r["query_type"] for r in rows}):
        selected = [r for r in rows if r["query_type"] == query_type]
        lines.append(f"| {query_type} | {len(selected)//2} | "
                     f"{sum(r['score'] for r in selected if r['condition']=='RAW')/(len(selected)//2):.3f} | "
                     f"{sum(r['score'] for r in selected if r['condition']=='RAW+EPISODES')/(len(selected)//2):.3f} |")
    lines += ["", "## API usage and estimated list-price cost", "",
              "| phase | new API calls | input tokens | output tokens | estimated USD |",
              "|---|---:|---:|---:|---:|", *usage_lines,
              f"| **total** |  |  |  | **${total_cost:.3f}** |", "",
              "Prices used: GPT-4o-mini $0.15/M input and $0.60/M output; text-embedding-3-small $0.02/M. "
              "Local MiniLM computation is not assigned a dollar price.", "",
              "## Published paper reference (MeanQ)", "", "| system | MeanQ |", "|---|---:|"]
    for name, value in official.PUBLISHED.items():
        lines.append(f"| {name} | {value:.3f} |")
    lines += ["", "## Guardrails", "",
              "- Memory was sealed before QA columns were opened (`qa_read=false` in episodes_meta.json).",
              "- Compare RAW and RAW+EPISODES directly; they share answerer, judge, raw retrieval, and budget.",
              "- Published rows are references from the paper, not reruns or an official leaderboard submission."]
    return "\n".join(lines) + "\n"


def status() -> None:
    memory = json.loads(EPISODES_META.read_text()) if EPISODES_META.exists() else None
    predictions = len(_jsonl(PREDICTIONS))
    results = len(_jsonl(RESULTS))
    print(f"run_dir={RUN_DIR}")
    print(f"memory={'sealed ' + str(memory['episode_count']) + ' episodes' if memory else 'not sealed'}")
    print(f"extracted_sessions_cached={len(_jsonl(EVENT_CACHE))}/348")
    print(f"attach_decisions_cached={len(_jsonl(ATTACH_CACHE))}")
    print(f"predictions={predictions}/1031")
    print(f"scores={results}/{1031 * len(CONDITIONS)}")
    print(f"report={REPORT if REPORT.exists() else 'not ready'}")


def run_all() -> None:
    _timing_mark("run_started_at")
    prepare_memory()
    run_inference()
    score()
    _timing_mark("run_completed_at")
    _log("DONE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group(required=True)
    phases.add_argument("--all", action="store_true")
    phases.add_argument("--prepare-memory", action="store_true")
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
        elif args.prepare_memory:
            prepare_memory()
        elif args.run_inference:
            run_inference()
        else:
            score()
    except RateLimitError as error:
        _log(f"rate limit; all completed calls are cached: {error}")
        raise SystemExit(75) from None
    except AuthenticationError as error:
        raise SystemExit(f"OpenAI authentication failed: {error}") from None


if __name__ == "__main__":
    main()
