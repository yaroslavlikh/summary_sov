"""EverMemBench Temporal ablation: linked episodes versus the same events unlinked.

Keeps the sealed derived events and their provenance but removes chronological linking:
every event becomes its own retrieval document. Compared with sealed RAW+EPISODES at an
equal token budget on the 300 Temporal Duration questions.
Protocol: research/EVERMEMBENCH_UNLINKED_EVENTS_CONTROL_PROTOCOL.md.

Run:
    python3 -m research.evermembench_unlinked_events_control --build-index   # OpenAI embeddings only
    python3 -m research.evermembench_unlinked_events_control --verify        # offline, no API
    python3 -m research.evermembench_unlinked_events_control --smoke         # 5 questions
    python3 -m research.evermembench_unlinked_events_control --all           # 300 TP questions
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AuthenticationError, RateLimitError

from research import evermembench_episode_run as base
from research import evermembench_parallel_resume as runner
from research import evermembench_temporal_budget_control as control
from research.paper_benchmark_common import (
    CachedAPI,
    EmbeddingCache,
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
    verify_sealed_jsonl,
)


SOURCE_DIR = base.RUN_DIR
CONTROL_FROZEN = (base.ROOT / ".research_runs" / "frozen"
                  / "evermembench_temporal_budget_control_v1_20260915_021134_MSK")
INDEX_DIR = base.ROOT / ".research_runs" / "evermembench_unlinked_events_index_v1"
RUN_DIR = base.ROOT / ".research_runs" / "evermembench_unlinked_events_control_v1"
SMOKE_DIR = base.ROOT / ".research_runs" / "evermembench_unlinked_events_control_smoke_v1"
PROTOCOL = base.ROOT / "research" / "EVERMEMBENCH_UNLINKED_EVENTS_CONTROL_PROTOCOL.md"
RUNNER = Path(__file__).resolve()
EVENTS_TOP10 = "RAW+EVENTS"
EVENTS_TOKEN = "RAW+EVENTS-token-matched"
NEW_CONDITIONS = (EVENTS_TOP10, EVENTS_TOKEN)
TOP_K = 10
RANK_DEPTH = 400
CONTROL_ROWS = 900


# --------------------------------------------------------------------------
# Unlinked event documents (pure, unit-tested)
# --------------------------------------------------------------------------

def event_doc(event: dict, raw_by_id: dict[str, SearchDoc]) -> SearchDoc:
    """One derived event with its own provenance, in the episode document format."""
    owner = event.get("viewpoint_owner") or "?"
    subject = event.get("subject") or "?"
    text = event["event_text"]
    sources = tuple(dict.fromkeys(event["source_turn_ids"]))
    lines = [f"[DERIVED EVENT / owner={owner} / subject={subject}]", text]
    for source_id in sources:
        source = raw_by_id.get(source_id)
        if source:
            lines.append(f"  [SOURCE {source_id}] {source.rendered}")
    return SearchDoc(
        doc_id=f"event:{event['network_id']}:{event['event_id']}",
        index_text=f"Viewpoint owner: {owner}. Subject: {subject}. {text}",
        rendered="\n".join(lines), source_ids=sources, kind="event",
    )


def events_by_topic(episodes_path: Path, raw_by_id: dict[str, SearchDoc]) -> dict[str, list[SearchDoc]]:
    """Every stored event, in sealed file order; never only an episode's hot window."""
    result: dict[str, list[SearchDoc]] = {topic: [] for topic in base.BATCHES}
    seen: set[str] = set()
    for row in jsonl(episodes_path):
        for item in row["events"]:
            doc = event_doc(item["event"], raw_by_id)
            if doc.doc_id in seen:
                raise RuntimeError(f"duplicate event id in sealed memory: {doc.doc_id}")
            seen.add(doc.doc_id)
            result[row["network_id"]].append(doc)
    return result


def docs_digest(docs: list[SearchDoc]) -> str:
    return sha256_bytes("\n".join(f"{doc.doc_id}\t{doc.index_text}" for doc in docs).encode())


# --------------------------------------------------------------------------
# Event index (built once, then read only)
# --------------------------------------------------------------------------

def build_event_index() -> dict:
    load_dotenv(base.ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY отсутствует в .env")
    control.verify_sources()
    _frame, _raw_by_topic, raw_by_id = base.load_messages()
    events = events_by_topic(SOURCE_DIR / "episodes.jsonl", raw_by_id)
    direct = CachedAPI(INDEX_DIR, api_key=key, label="openai")
    embeddings = EmbeddingCache(INDEX_DIR, direct)
    meta: dict = {"embedding_model": embeddings.model,
                  "source_episodes_sha256": sha256_file(SOURCE_DIR / "episodes.jsonl"), "topics": {}}
    for topic in base.BATCHES:
        docs = events[topic]
        digest = docs_digest(docs)
        path = INDEX_DIR / f"events-{topic}-{digest[:20]}.npy"
        if not path.exists():
            np.save(path, embeddings.all([doc.index_text for doc in docs]))
            log(INDEX_DIR, f"event index: topic {topic} embedded {len(docs)} events")
        meta["topics"][topic] = {"events": len(docs), "docs_sha256": digest,
                                 "matrix_file": path.name, "matrix_sha256": sha256_file(path)}
    meta_path = INDEX_DIR / "index_meta.json"
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != meta:
            raise RuntimeError("sealed event index changed")
    else:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    log(INDEX_DIR, "event index sealed")
    return meta


def load_event_matrix(topic: str, docs: list[SearchDoc]) -> np.ndarray:
    meta = json.loads((INDEX_DIR / "index_meta.json").read_text())["topics"][topic]
    if meta["docs_sha256"] != docs_digest(docs) or meta["events"] != len(docs):
        raise RuntimeError(f"event documents differ from the sealed event index: topic {topic}")
    path = INDEX_DIR / meta["matrix_file"]
    if sha256_file(path) != meta["matrix_sha256"]:
        raise RuntimeError(f"sealed event matrix checksum mismatch: {path}")
    matrix = np.load(path)
    if matrix.shape[0] != len(docs):
        raise RuntimeError(f"event matrix row count mismatch: topic {topic}")
    return matrix


# --------------------------------------------------------------------------
# Job preparation (offline)
# --------------------------------------------------------------------------

def verify_frozen_control() -> None:
    run = CONTROL_FROZEN / "run"
    verify_sealed_jsonl(run / "predictions.jsonl", run / "predictions_meta.json", expected_rows=CONTROL_ROWS)
    verify_sealed_jsonl(run / "results.jsonl", run / "results_meta.json", expected_rows=CONTROL_ROWS)


def prepare_jobs(questions: list[dict]) -> tuple[list[dict], dict]:
    import tiktoken

    control.verify_sources()
    verify_frozen_control()
    encoder = tiktoken.get_encoding(control.TOKEN_ENCODING)
    count_tokens = lambda text: len(encoder.encode(text))
    _frame, raw_by_topic, raw_by_id = base.load_messages()
    episodes = runner._load_episode_docs(raw_by_id)
    events = events_by_topic(SOURCE_DIR / "episodes.jsonl", raw_by_id)
    memory_meta = json.loads((SOURCE_DIR / "episodes_meta.json").read_text())
    total_events = sum(len(docs) for docs in events.values())
    if total_events != int(memory_meta["stats"]["events_total"]):
        raise RuntimeError(f"event documents {total_events} != sealed events {memory_meta['stats']['events_total']}")
    docs_by_id = {doc.doc_id: doc for docs in raw_by_topic.values() for doc in docs}
    docs_by_id.update({doc.doc_id: doc for docs in episodes.values() for doc in docs})
    frozen = {(row["qa_key"], row["condition"]): row for row in jsonl(SOURCE_DIR / "predictions.jsonl")}
    source_cache = {json.loads(line)["key"] for line in open(SOURCE_DIR / "openrouter_chat_cache.jsonl")}
    prompts = base._prompts()["answer"]
    embed_dir = SOURCE_DIR / "embeddings"
    wanted = {row["qa_key"] for row in questions}
    all_questions = base.load_questions(include_gold=False)

    jobs: list[dict] = []
    checks = {"questions": 0, "raw_top10_matches_sealed": 0, "episodes_prompt_in_sealed_cache": 0,
              "event_documents": total_events, "rank_depth_exhausted": 0}
    for topic in base.BATCHES:
        topic_questions = [row for row in all_questions if row["topic"] == topic]
        if not any(row["qa_key"] in wanted for row in topic_questions):
            continue
        vectors = control.question_vectors(embed_dir, [row["question"] for row in topic_questions])
        raw_matrix = control.raw_index_matrix(embed_dir, topic, raw_by_topic[topic])
        unified_docs = raw_by_topic[topic] + events[topic]
        unified_matrix = np.vstack([raw_matrix, load_event_matrix(topic, events[topic])])
        for row, vector in zip(topic_questions, vectors):
            if row["qa_key"] not in wanted:
                continue
            raw_order, _ = control.rank_raw(raw_matrix, vector, TOP_K)
            if [raw_by_topic[topic][int(i)].doc_id for i in raw_order] != frozen[(row["qa_key"], "RAW")]["retrieved_ids"]:
                raise RuntimeError(f"raw ranking does not reproduce sealed RAW top-10: {row['qa_key']}")
            episode_docs = [docs_by_id[doc_id] for doc_id in frozen[(row["qa_key"], "RAW+EPISODES")]["retrieved_ids"]]
            build = lambda docs: prompts["open_ended"].format(context=control.render_context(docs), question=row["question"])
            if control.answer_chat_key(build(episode_docs)) not in source_cache:
                raise RuntimeError(f"RAW+EPISODES prompt does not reproduce the sealed prompt: {row['qa_key']}")
            checks["questions"] += 1
            checks["raw_top10_matches_sealed"] += 1
            checks["episodes_prompt_in_sealed_cache"] += 1

            order, retrieval_ms = control.rank_raw(unified_matrix, vector, RANK_DEPTH)
            ranked = [unified_docs[int(index)] for index in order]
            episode_tokens = count_tokens(control.render_context(episode_docs))
            started = time.perf_counter()
            token_docs, _ = control.select_token_matched(ranked, episode_tokens, count_tokens)
            token_ms = (time.perf_counter() - started) * 1000
            if len(token_docs) >= len(ranked):
                checks["rank_depth_exhausted"] += 1
            shared = {
                "raw10_context_tokens": count_tokens(control.render_context([docs_by_id[i] for i in frozen[(row["qa_key"], "RAW")]["retrieved_ids"]])),
                "episodes_context_tokens": episode_tokens,
                "episodes_unique_sources": len(control.unique_source_ids(episode_docs)),
            }
            for condition, docs, selection_ms in ((EVENTS_TOP10, ranked[:TOP_K], 0.0),
                                                  (EVENTS_TOKEN, token_docs, token_ms)):
                jobs.append({
                    "question": row, "condition": condition, "docs": docs, "prompt": build(docs),
                    "retrieval_ms": retrieval_ms, "selection_ms": selection_ms,
                    "context_tokens": count_tokens(control.render_context(docs)), **shared,
                })
    return jobs, checks


def ensure_run_manifest(run_dir: Path, scope: list[str]) -> None:
    manifest = {
        "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
        "runner": RUNNER.name, "runner_sha256": sha256_file(RUNNER),
        "control_runner_sha256": sha256_file(control.RUNNER),
        "source_predictions_sha256": sha256_file(SOURCE_DIR / "predictions.jsonl"),
        "source_results_sha256": sha256_file(SOURCE_DIR / "results.jsonl"),
        "source_episodes_sha256": sha256_file(SOURCE_DIR / "episodes.jsonl"),
        "control_results_sha256": sha256_file(CONTROL_FROZEN / "run" / "results.jsonl"),
        "event_index_meta_sha256": sha256_file(INDEX_DIR / "index_meta.json"),
        "conditions": list(NEW_CONDITIONS), "scope_qa_keys": sorted(scope),
        "token_encoding": control.TOKEN_ENCODING, "rank_depth": RANK_DEPTH,
    }
    path = run_dir / "run_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError(f"run manifest mismatch (protocol, runner, index or scope changed): {path}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


# --------------------------------------------------------------------------
# API phases
# --------------------------------------------------------------------------

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
        sources = control.unique_source_ids(docs)
        return {
            **{field: row[field] for field in ("qa_key", "topic", "id", "major", "minor", "question", "question_type")},
            "condition": job["condition"],
            "answer": runner.official_answer(raw_answer, row["question_type"]), "answer_raw": raw_answer,
            "retrieved_ids": [doc.doc_id for doc in docs], "exposed_source_ids": sources,
            "unique_source_count": len(sources), "derived_docs": sum(1 for doc in docs if doc.kind == "event"),
            "context_tokens": job["context_tokens"], "raw10_context_tokens": job["raw10_context_tokens"],
            "episodes_context_tokens": job["episodes_context_tokens"],
            "episodes_unique_sources": job["episodes_unique_sources"],
            "prompt_sha256": sha256_bytes(job["prompt"].encode()),
            "retrieval_ms": job["retrieval_ms"], "selection_ms": job["selection_ms"],
            "answer_wall_ms": (time.perf_counter() - started) * 1000,
        }

    runner._parallel(pending, work, lambda row: append_jsonl(run_dir / "predictions.jsonl", row),
                     workers=workers, label="inference")
    rows = jsonl(run_dir / "predictions.jsonl")
    markers = ("[EMPTY]", "[INVALID]", "[FAILED]")
    seal_jsonl(
        run_dir / "predictions.jsonl", run_dir / "predictions_meta.json",
        expected_rows=len(questions) * len(NEW_CONDITIONS), identity_fields=("qa_key", "condition"),
        metadata={
            "benchmark": "EverMemBench-Dynamic", "protocol_sha256": sha256_file(PROTOCOL),
            "runner_sha256": sha256_file(RUNNER), "conditions": list(NEW_CONDITIONS),
            "answer_model": base.ANSWER_MODEL, "token_encoding": control.TOKEN_ENCODING,
            "event_index_meta_sha256": sha256_file(INDEX_DIR / "index_meta.json"),
            "answer_failure_markers": {c: {m: sum(1 for r in rows if r["condition"] == c and r["answer"] == m)
                                           for m in markers} for c in NEW_CONDITIONS},
        },
    )


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
    sealed_results = [r for r in jsonl(SOURCE_DIR / "results.jsonl") if r["qa_key"] in keys]
    sealed_predictions = [p for p in jsonl(SOURCE_DIR / "predictions.jsonl") if p["qa_key"] in keys]
    control_results = [r for r in jsonl(CONTROL_FROZEN / "run" / "results.jsonl")
                       if r["qa_key"] in keys and r["condition"] == control.TOKEN_MATCHED]
    control_predictions = [p for p in jsonl(CONTROL_FROZEN / "run" / "predictions.jsonl")
                           if p["qa_key"] in keys and p["condition"] == control.TOKEN_MATCHED]
    rows = sealed_results + control_results + new_results
    predictions = sealed_predictions + control_predictions + new_predictions
    shared = {row["qa_key"]: row for row in new_predictions}
    conditions = ("RAW", "RAW+EPISODES", control.TOKEN_MATCHED, EVENTS_TOP10, EVENTS_TOKEN)

    lines = [
        "# EverMemBench Temporal ablation: linked episodes versus unlinked events", "",
        f"- protocol: `{PROTOCOL.name}` sha256 `{sha256_file(PROTOCOL)}`",
        f"- runner sha256: `{sha256_file(RUNNER)}`",
        f"- event index meta sha256: `{sha256_file(INDEX_DIR / 'index_meta.json')}`",
        f"- sealed rows: RAW and RAW+EPISODES from `{SOURCE_DIR.name}`; "
        f"{control.TOKEN_MATCHED} from `{CONTROL_FROZEN.name}`",
        f"- questions: {len(keys)} Temporal Duration (`TP`); tokens: `{control.TOKEN_ENCODING}`", "",
        "## Conditions", "",
        "| condition | n | accuracy | mean unique raw sources | mean context tokens | mean derived docs | evidence recall | evidence precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        selected = [r for r in rows if r["condition"] == condition]
        preds = [p for p in predictions if p["condition"] == condition]
        if condition == "RAW":
            tokens = [shared[p["qa_key"]]["raw10_context_tokens"] for p in preds]
        elif condition == "RAW+EPISODES":
            tokens = [shared[p["qa_key"]]["episodes_context_tokens"] for p in preds]
        else:
            tokens = [p["context_tokens"] for p in preds]
        derived = [sum(1 for doc_id in p["retrieved_ids"] if not doc_id.startswith("raw:")) for p in preds]
        lines.append(
            f"| {condition} | {len(selected)} | {_mean([r['correct'] for r in selected]):.3f} | "
            f"{_mean([len(p['exposed_source_ids']) for p in preds]):.1f} | {_mean(tokens):.0f} | {_mean(derived):.1f} | "
            f"{_mean([r['evidence_recall'] for r in selected]):.3f} | {_mean([r['evidence_precision'] for r in selected]):.3f} |"
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
        comparison("RAW+EPISODES", EVENTS_TOKEN, "**primary**"),
        comparison("RAW+EPISODES", EVENTS_TOP10, "secondary"),
        comparison(EVENTS_TOKEN, control.TOKEN_MATCHED, "secondary"),
        comparison(EVENTS_TOP10, "RAW", "secondary"),
    ]
    delta, low, high = paired_bootstrap(rows, "RAW+EPISODES", EVENTS_TOKEN)
    verdict = ("chronological linking adds accuracy beyond the same unlinked events at an equal token budget" if low > 0 else
               "unlinked events at an equal token budget are at least as accurate" if high < 0 else
               "the linking effect is not distinguishable from event summaries alone")
    lines += ["", "## Interpretation rule fixed before new outcomes", "",
              f"- primary 95% CI [{low:+.3f}, {high:+.3f}]: **{verdict}**."]

    usage = usage_summary(run_dir)
    index_usage = usage_summary(INDEX_DIR)
    judge_rows = [r for r in new_results if "judge_parse" in r]
    lines += [
        "", "## Cost and latency (new work only)", "",
        f"- answer and judge API calls: {usage['api_calls']}; estimated USD: {usage['estimated_usd_upper_bound']:.4f}",
        f"- event index embedding calls: {index_usage['api_calls']}; estimated USD: {index_usage['estimated_usd_upper_bound']:.4f}",
        f"- usage by model: `{json.dumps(usage['by_model'], sort_keys=True)}`",
        f"- unified ranking latency: `{json.dumps(latency_summary(new_predictions), sort_keys=True)}`",
        f"- context selection ms (mean): `{json.dumps({c: _mean([p['selection_ms'] for p in new_predictions if p['condition'] == c]) for c in NEW_CONDITIONS}, sort_keys=True)}`",
        f"- judge: parse `{json.dumps({b: sum(1 for r in judge_rows if r['judge_parse'] == b) for b in {r['judge_parse'] for r in judge_rows}}, sort_keys=True)}`, "
        f"re-requested {sum(1 for r in judge_rows if r['judge_attempts'] > 1)}",
    ]
    report = run_dir / "report.md"
    report.write_text("\n".join(lines) + "\n")
    log(run_dir, f"report ready: {report}")
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def run(run_dir: Path, smoke: bool, answer_workers: int, judge_workers: int) -> None:
    questions = control.target_questions(smoke)
    ensure_run_manifest(run_dir, [row["qa_key"] for row in questions])
    router = control.make_router(run_dir)
    runner.check_routes(router)
    run_inference(run_dir, questions, router, answer_workers)
    score(run_dir, questions, router, judge_workers)
    render_report(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-workers", type=int, default=16)
    parser.add_argument("--judge-workers", type=int, default=4)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("build-index", "verify", "smoke", "all", "report"):
        modes.add_argument(f"--{mode}", action="store_true")
    args = parser.parse_args()
    target_dir = SMOKE_DIR if args.smoke else RUN_DIR
    try:
        if args.build_index:
            print(json.dumps(build_event_index(), indent=2))
            return
        if args.verify:
            jobs, checks = prepare_jobs(control.target_questions(smoke=False))
            print(json.dumps(checks, indent=2, sort_keys=True))
            for condition in NEW_CONDITIONS:
                items = [j for j in jobs if j["condition"] == condition]
                print(f"{condition}: docs {_mean([len(j['docs']) for j in items]):.1f}, "
                      f"derived docs {_mean([sum(1 for d in j['docs'] if d.kind == 'event') for j in items]):.1f}, "
                      f"unique sources {_mean([len(control.unique_source_ids(j['docs'])) for j in items]):.1f}, "
                      f"context tokens {_mean([j['context_tokens'] for j in items]):.0f} "
                      f"(RAW+EPISODES target {_mean([j['episodes_context_tokens'] for j in items]):.0f})")
            return
        if args.report:
            print(render_report(RUN_DIR))
            return
        run(target_dir, args.smoke, args.answer_workers, args.judge_workers)
    except RateLimitError as error:
        log(target_dir, f"rate limit; completed work is cached: {error}")
        raise SystemExit(75) from None
    except AuthenticationError as error:
        raise SystemExit(f"API authentication failed: {error}") from None


if __name__ == "__main__":
    main()
