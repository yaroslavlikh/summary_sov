"""Paper run: official EverMemBench-Dynamic RAW vs RAW+TemporalEpisodes."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import AuthenticationError, RateLimitError

from research.paper_benchmark_common import (
    EMBED_MODEL, EXTRACTION_MODEL, ROOT, TOP_K, CachedAPI, DenseIndex, EmbeddingCache,
    SearchDoc, append_jsonl, build_episode_snapshot, clustered_bootstrap, ensure_git_snapshot, episode_doc,
    exact_mcnemar, freeze_run, jsonl, latency_summary, log, paired_bootstrap,
    seal_jsonl, sha256_file, usage_summary, verify_episode_snapshot, verify_sealed_jsonl,
)


CODE_URL = "https://github.com/EverMind-AI/EverMemBench.git"
CODE_COMMIT = "e10b3d52f0e4cfc5c124ad406b5d95c59c73738b"
DATA_URL = "https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic"
DATA_COMMIT = "a6b210a32248e841967b7b64a64281d2ff3f669d"
OFFICIAL_CODE = ROOT / ".research_runs" / "official_sources" / "EverMemBench"
OFFICIAL_DATA = ROOT / ".research_runs" / "official_sources" / "EverMemBench-Dynamic"
RUN_DIR = ROOT / ".research_runs" / "evermembench_temporal_episodes_official_v1"
BATCHES = ("01", "02", "03", "04", "05")
CONDITIONS = ("RAW", "RAW+EPISODES")
ANSWER_MODEL = "openai/gpt-4.1-mini"
JUDGE_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_DEFAULT = "https://openrouter.ai/api/v1"
PROVIDER = {
    "answer": {"provider": {"order": ["openai"], "allow_fallbacks": False}},
    "judge": {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": False}},
}


def _turn_id(topic: str, date: str, group: str, message_index: object) -> str:
    return f"{topic}|{date}|{group}|{str(message_index)}"


def _source_manifest() -> dict:
    files = [OFFICIAL_CODE / "eval" / "config" / "pipeline.yaml",
             OFFICIAL_CODE / "eval" / "config" / "prompts.yaml"]
    files += [OFFICIAL_DATA / batch / "dialogue.json" for batch in BATCHES]
    return {
        "benchmark": "EverMemBench-Dynamic", "code_url": CODE_URL, "code_commit": CODE_COMMIT,
        "data_url": DATA_URL, "data_commit": DATA_COMMIT,
        "files": {
            (f"code/{path.relative_to(OFFICIAL_CODE)}" if path.is_relative_to(OFFICIAL_CODE)
             else f"data/{path.relative_to(OFFICIAL_DATA)}"): sha256_file(path)
            for path in files
        },
        "implementation": {
            name: sha256_file(ROOT / "research" / name) for name in (
                "evermembench_episode_run.py", "paper_benchmark_common.py",
                "query_independent_episode_pipeline.py", "temporal_episode_prototype.py",
                "PAPER_BENCHMARK_PROTOCOL.md",
            )
        },
    }


def _evaluation_manifest() -> dict:
    manifest = _source_manifest()
    manifest["question_files"] = {
        str((OFFICIAL_DATA / batch / f"qa_{batch}.json").relative_to(OFFICIAL_DATA)):
        sha256_file(OFFICIAL_DATA / batch / f"qa_{batch}.json")
        for batch in BATCHES
    }
    return manifest


def load_messages() -> tuple[pd.DataFrame, dict[str, list[SearchDoc]], dict[str, SearchDoc]]:
    frame_rows: list[dict] = []
    raw_by_topic: dict[str, list[SearchDoc]] = defaultdict(list)
    raw_by_id: dict[str, SearchDoc] = {}
    for topic in BATCHES:
        days = json.loads((OFFICIAL_DATA / topic / "dialogue.json").read_text())
        sessions = []
        for day in days:
            date = str(day["date"])
            for group, messages in day["dialogues"].items():
                if messages:
                    sessions.append((date, group, messages))
        for session_index, (date, group, messages) in enumerate(sorted(sessions), 1):
            for message in sorted(messages, key=lambda row: str(row.get("time", ""))):
                source_id = _turn_id(topic, date, group, message.get("message_index"))
                body = str(message.get("dialogue") or "").strip()
                stamp = str(message.get("time") or date)
                speaker = str(message.get("speaker") or "Unknown")
                rendered = f"[{stamp}][Group: {group}][Speaker: {speaker}]{body}"
                doc = SearchDoc(f"raw:{source_id}", body, rendered, (source_id,), "raw")
                if source_id in raw_by_id:
                    raise ValueError(f"duplicate EverMemBench source id: {source_id}")
                raw_by_id[source_id] = doc
                raw_by_topic[topic].append(doc)
                frame_rows.append({
                    "network_id": topic, "session_index": session_index, "turn_id": source_id,
                    "timestamp": stamp, "speaker_display_name": f"{speaker} ({group})", "message": body,
                })
    return pd.DataFrame(frame_rows), dict(raw_by_topic), raw_by_id


def _expand_message_indices(value: object) -> list[str]:
    result = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            result.extend(str(number) for number in range(int(match.group(1)), int(match.group(2)) + 1))
        elif part.isdigit():
            result.append(str(int(part)))
    return result


def load_questions(*, include_gold: bool) -> list[dict]:
    rows = []
    for topic in BATCHES:
        for raw in json.loads((OFFICIAL_DATA / topic / f"qa_{topic}.json").read_text()):
            parts = str(raw["id"]).split("_")
            row = {
                "qa_key": f"{topic}:{raw['id']}", "topic": topic, "id": str(raw["id"]),
                "major": parts[0] if parts else "Unknown", "minor": parts[1] if len(parts) > 1 else "Unknown",
                "question": str(raw["Q"]), "options": raw.get("options"),
                "question_type": "multiple_choice" if raw.get("options") else "open_ended",
            }
            if include_gold:
                evidence = []
                for reference in raw.get("R", []):
                    for index in _expand_message_indices(reference.get("message_index")):
                        evidence.append(_turn_id(topic, str(reference.get("date")), str(reference.get("group")), index))
                row.update(gold=str(raw.get("A") or ""), gold_source_ids=list(dict.fromkeys(evidence)))
            rows.append(row)
    return rows


def _prompts() -> dict:
    return yaml.safe_load((OFFICIAL_CODE / "eval" / "config" / "prompts.yaml").read_text())


def _option_text(options: dict | None) -> str:
    return "\n".join(f"{key}. {value}" for key, value in sorted((options or {}).items()))


def _parse_mc(response: str) -> str:
    value = (response or "").strip().upper()
    for pattern in (r"(?:ANSWER|OPTION)\s*[:\-]?\s*([ABCD])\b", r"^([ABCD])(?:[.)\s]|$)"):
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    if value and value[-1] in "ABCD" and (len(value) == 1 or not value[-2].isalpha()):
        return value[-1]
    return value


def preflight() -> tuple[CachedAPI, CachedAPI]:
    load_dotenv(ROOT / ".env")
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("LLM_API_KEY")
    if not openai_key:
        raise SystemExit("OPENAI_API_KEY отсутствует в .env (нужен для memory extraction и embeddings)")
    if not openrouter_key:
        raise SystemExit("LLM_API_KEY отсутствует в .env (нужен официальный OpenRouter answer/judge harness EverMemBench)")
    ensure_git_snapshot(OFFICIAL_CODE, CODE_URL, CODE_COMMIT)
    ensure_git_snapshot(OFFICIAL_DATA, DATA_URL, DATA_COMMIT)
    direct = CachedAPI(RUN_DIR, api_key=openai_key, label="openai")
    router = CachedAPI(RUN_DIR, api_key=openrouter_key,
                       base_url=os.getenv("LLM_BASE_URL", OPENROUTER_DEFAULT), label="openrouter")
    direct.client.models.retrieve(EXTRACTION_MODEL)
    direct.client.models.retrieve(EMBED_MODEL)
    router.client.models.list()
    log(RUN_DIR, f"preflight OK: EverMemBench code@{CODE_COMMIT[:12]} data@{DATA_COMMIT[:12]}")
    return direct, router


def prepare_memory(direct: CachedAPI) -> None:
    frame, _raw, _lookup = load_messages()
    sessions = frame[["network_id", "session_index"]].drop_duplicates().shape[0]
    log(RUN_DIR, f"memory: {len(frame)} messages / {sessions} day-group sessions; QA unopened")
    build_episode_snapshot(frame, RUN_DIR, direct, source_manifest=_source_manifest())
    log(RUN_DIR, "memory: sealed")


def _load_episode_docs(raw_by_id: dict[str, SearchDoc]) -> dict[str, list[SearchDoc]]:
    result: dict[str, list[SearchDoc]] = defaultdict(list)
    for row in jsonl(RUN_DIR / "episodes.jsonl"):
        result[row["network_id"]].append(episode_doc(row, raw_by_id))
    return dict(result)


def run_inference(direct: CachedAPI, router: CachedAPI) -> None:
    verify_episode_snapshot(RUN_DIR, _source_manifest())
    _frame, raw_by_topic, raw_by_id = load_messages()
    episodes = _load_episode_docs(raw_by_id)
    embeddings = EmbeddingCache(RUN_DIR, direct)
    prompts = _prompts()["answer"]
    done = {(row["qa_key"], row["condition"]) for row in jsonl(RUN_DIR / "predictions.jsonl")}
    pending = [(row, condition) for row in load_questions(include_gold=False) for condition in CONDITIONS
               if (row["qa_key"], condition) not in done]
    log(RUN_DIR, f"inference: {len(pending)} pending / {len(done)} cached")
    completed = 0
    for topic in BATCHES:
        topic_pending = [(row, condition) for row, condition in pending if row["topic"] == topic]
        if not topic_pending:
            continue
        indexes = {
            "RAW": DenseIndex(raw_by_topic[topic], embeddings, f"{topic}-raw"),
            "RAW+EPISODES": DenseIndex(raw_by_topic[topic] + episodes.get(topic, []), embeddings,
                                        f"{topic}-raw-episodes"),
        }
        unique_rows = {row["qa_key"]: row for row, _condition in topic_pending}
        vectors = embeddings.all([row["question"] for row in unique_rows.values()])
        vector_by_key = dict(zip(unique_rows, vectors))
        for row, condition in topic_pending:
            docs, retrieval_ms = indexes[condition].search_vector(vector_by_key[row["qa_key"]], TOP_K)
            context = "\n".join(f"- {doc.rendered}" for doc in docs) or "(No memories retrieved)"
            if row["question_type"] == "multiple_choice":
                prompt = prompts["multiple_choice"].format(
                    context=context, question=row["question"], options=_option_text(row["options"]),
                )
            else:
                prompt = prompts["open_ended"].format(context=context, question=row["question"])
            started = time.perf_counter()
            answer = router.chat(
                model=ANSWER_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
                max_tokens=1000, phase="answer", extra_body=PROVIDER["answer"],
            ).strip()
            append_jsonl(RUN_DIR / "predictions.jsonl", {
                **row, "condition": condition, "answer": answer,
                "retrieved_ids": [doc.doc_id for doc in docs],
                "exposed_source_ids": list(dict.fromkeys(s for doc in docs for s in doc.source_ids)),
                "retrieval_ms": retrieval_ms, "answer_wall_ms": (time.perf_counter() - started) * 1000,
            })
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                log(RUN_DIR, f"inference: {completed}/{len(pending)} pending")
        del indexes
    expected = len(load_questions(include_gold=False)) * len(CONDITIONS)
    seal_jsonl(
        RUN_DIR / "predictions.jsonl", RUN_DIR / "predictions_meta.json",
        expected_rows=expected, identity_fields=("qa_key", "condition"),
        metadata={
            "benchmark": "EverMemBench-Dynamic", "code_commit": CODE_COMMIT,
            "data_commit": DATA_COMMIT, "conditions": list(CONDITIONS),
            "answer_model": ANSWER_MODEL, "embedding_model": EMBED_MODEL, "top_k": TOP_K,
            "episode_snapshot_sha256": sha256_file(RUN_DIR / "episodes.jsonl"),
        },
    )


def score(router: CachedAPI) -> None:
    prompts = _prompts()["llm_judge"]
    expected = len(load_questions(include_gold=False)) * len(CONDITIONS)
    verify_sealed_jsonl(RUN_DIR / "predictions.jsonl", RUN_DIR / "predictions_meta.json",
                        expected_rows=expected)
    predictions = {(row["qa_key"], row["condition"]): row for row in jsonl(RUN_DIR / "predictions.jsonl")}
    gold = {row["qa_key"]: row for row in load_questions(include_gold=True)}
    evaluation_manifest = _evaluation_manifest()
    manifest_path = RUN_DIR / "evaluation_input_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != evaluation_manifest:
        raise RuntimeError("evaluation input manifest mismatch")
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(evaluation_manifest, indent=2) + "\n")
    done = {(row["qa_key"], row["condition"]) for row in jsonl(RUN_DIR / "results.jsonl")}
    pending = [key for key in predictions if key not in done]
    log(RUN_DIR, f"score: {len(pending)} pending / {len(done)} cached")
    for count, key in enumerate(pending, 1):
        prediction = predictions[key]
        truth = gold[prediction["qa_key"]]
        if truth["question_type"] == "multiple_choice":
            correct = _parse_mc(prediction["answer"]) == truth["gold"].strip().upper()
            rationale = "exact multiple-choice comparison"
        else:
            prompt = prompts["user_prompt"].format(
                question=truth["question"], golden_answer=truth["gold"],
                generated_answer=prediction["answer"],
            )
            output = router.chat(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": prompts["system_prompt"]},
                          {"role": "user", "content": prompt}],
                temperature=0, max_tokens=512, phase="judge", extra_body=PROVIDER["judge"],
            )
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                match = re.search(r'"label"\s*:\s*"(CORRECT|WRONG)"', output.upper())
                parsed = {"label": match.group(1)} if match else {}
            correct = str(parsed.get("label", "")).upper() == "CORRECT"
            rationale = output
        exposed = set(prediction["exposed_source_ids"])
        evidence = set(truth["gold_source_ids"])
        append_jsonl(RUN_DIR / "results.jsonl", {
            "qa_key": prediction["qa_key"], "topic": prediction["topic"],
            "major": prediction["major"], "minor": prediction["minor"],
            "question_type": prediction["question_type"], "condition": prediction["condition"],
            "correct": int(correct), "judge": rationale,
            "evidence_recall": len(exposed & evidence) / len(evidence) if evidence else 1.0,
            "evidence_precision": len(exposed & evidence) / len(exposed) if exposed else 0.0,
        })
        if count % 25 == 0 or count == len(pending):
            log(RUN_DIR, f"score: {count}/{len(pending)} pending")
    seal_jsonl(
        RUN_DIR / "results.jsonl", RUN_DIR / "results_meta.json",
        expected_rows=expected, identity_fields=("qa_key", "condition"),
        metadata={
            "benchmark": "EverMemBench-Dynamic", "code_commit": CODE_COMMIT,
            "data_commit": DATA_COMMIT, "conditions": list(CONDITIONS),
            "judge_model": JUDGE_MODEL,
            "predictions_sha256": sha256_file(RUN_DIR / "predictions.jsonl"),
            "evaluation_input_manifest_sha256": sha256_file(manifest_path),
        },
    )
    render_report()


def _mean(rows: list[dict], condition: str, field: str = "correct") -> float:
    values = [float(row[field]) for row in rows if row["condition"] == condition]
    return sum(values) / len(values) if values else float("nan")


def render_report() -> None:
    rows = jsonl(RUN_DIR / "results.jsonl")
    expected = len(load_questions(include_gold=False)) * len(CONDITIONS)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete results: {len(rows)}/{expected}")
    delta, lo, hi = paired_bootstrap(rows, "RAW+EPISODES", "RAW")
    topic_delta, topic_lo, topic_hi = clustered_bootstrap(rows, "RAW+EPISODES", "RAW", "topic")
    mcnemar = exact_mcnemar(rows, "RAW+EPISODES", "RAW")
    lines = [
        "# EverMemBench-Dynamic — RAW vs TemporalEpisodes", "",
        f"- official code: `{CODE_COMMIT}`; official data: `{DATA_COMMIT}`",
        f"- answer: `{ANSWER_MODEL}`; judge: `{JUDGE_MODEL}`; official prompts/provider pinning",
        f"- retrieval: `{EMBED_MODEL}`, unified cosine top-{TOP_K}",
        f"- memory builder: `{EXTRACTION_MODEL}`, query-independent day-group sessions", "",
        "## Primary endpoint (all 2,400 QA)", "",
        f"- RAW: {_mean(rows, 'RAW'):.3f}", f"- RAW+EPISODES: {_mean(rows, 'RAW+EPISODES'):.3f}",
        f"- paired delta: {delta:+.3f}, question bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]",
        f"- equal-topic delta: {topic_delta:+.3f}, topic-cluster bootstrap 95% CI [{topic_lo:+.3f}, {topic_hi:+.3f}]",
        f"- exact paired McNemar: EP-only={mcnemar['left_only']}, RAW-only={mcnemar['right_only']}, p={mcnemar['p_value']:.6f}",
        "", "## By batch and question form", "", "| slice | n | RAW | EPISODES | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    slices = [(f"topic {topic}", [r for r in rows if r["topic"] == topic]) for topic in BATCHES]
    slices += [(kind, [r for r in rows if r["question_type"] == kind])
               for kind in ("multiple_choice", "open_ended")]
    slices += [(f"major {major}", [r for r in rows if r["major"] == major])
               for major in sorted({r["major"] for r in rows})]
    slices += [(f"minor {minor}", [r for r in rows if r["minor"] == minor])
               for minor in sorted({r["minor"] for r in rows})]
    for label, selected in slices:
        raw, episode = _mean(selected, "RAW"), _mean(selected, "RAW+EPISODES")
        lines.append(f"| {label} | {len(selected)//2} | {raw:.3f} | {episode:.3f} | {episode-raw:+.3f} |")
    lines += ["", "## Retrieval provenance", "",
              f"- RAW evidence recall: {_mean(rows, 'RAW', 'evidence_recall'):.3f}",
              f"- EPISODES evidence recall: {_mean(rows, 'RAW+EPISODES', 'evidence_recall'):.3f}",
              f"- RAW evidence precision: {_mean(rows, 'RAW', 'evidence_precision'):.3f}",
              f"- EPISODES evidence precision: {_mean(rows, 'RAW+EPISODES', 'evidence_precision'):.3f}"]
    usage = usage_summary(RUN_DIR)
    latency = latency_summary(jsonl(RUN_DIR / "predictions.jsonl"))
    lines += ["", "## Efficiency artifacts", "", f"- API calls recorded: {usage['api_calls']}",
              f"- usage by model: `{json.dumps(usage['by_model'], sort_keys=True)}`",
              f"- estimated API cost (uncached-rate upper bound, USD): {usage['estimated_usd_upper_bound']:.4f}",
              f"- price snapshot: {usage['price_snapshot_date']}; {usage['pricing_assumption']}",
              f"- retrieval latency by condition: `{json.dumps(latency, sort_keys=True)}`",
              f"- episode snapshot bytes: {(RUN_DIR / 'episodes.jsonl').stat().st_size}", "",
              "## Protocol guardrails", "",
              "- Memory was sealed before QA files were opened.",
              "- Both conditions use the same official answer/judge prompts, models and top-k.",
              "- Every derived episode expands to mechanically validated raw evidence.",
              "- This is an adapter evaluation under the official harness, not a leaderboard submission."]
    (RUN_DIR / "report.md").write_text("\n".join(lines) + "\n")
    log(RUN_DIR, f"final report ready: {RUN_DIR / 'report.md'}")


def status() -> None:
    print(f"run_dir={RUN_DIR}")
    print("planned=51023 messages / 3570 extraction sessions / 2400 QA / 4800 answers / 1524 open-answer judges")
    print("attach_calls=unknown until extraction (approximately one per extracted event)")
    for name in ("episodes.jsonl", "predictions.jsonl", "results.jsonl", "report.md"):
        path = RUN_DIR / name
        print(f"{name}={'ready' if path.exists() else 'missing'}" +
              (f" rows={len(jsonl(path))}" if path.suffix == ".jsonl" and path.exists() else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("all", "prepare-memory", "run-inference", "score", "preflight", "status", "freeze"):
        modes.add_argument(f"--{mode}", action="store_true")
    args = parser.parse_args()
    if args.status:
        status(); return
    if args.freeze:
        destination = freeze_run(RUN_DIR, [Path(__file__), ROOT / "research" / "paper_benchmark_common.py",
                                           ROOT / "research" / "PAPER_BENCHMARK_PROTOCOL.md"])
        print(destination); return
    try:
        direct, router = preflight()
        if args.preflight:
            return
        if args.all or args.prepare_memory:
            prepare_memory(direct)
        if args.all or args.run_inference:
            run_inference(direct, router)
        if args.all or args.score:
            score(router)
    except RateLimitError as error:
        log(RUN_DIR, f"rate limit; completed work is cached: {error}")
        raise SystemExit(75) from None
    except AuthenticationError as error:
        raise SystemExit(f"API authentication failed: {error}") from None


if __name__ == "__main__":
    main()
