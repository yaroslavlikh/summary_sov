"""Paper run: official GroupMemBench dense RAW vs RAW+TemporalEpisodes.

The public dataset, prompts and harness revision are pinned. Memory is sealed
before question files are opened. The full run reports the filtered primary
domains separately from the two unfiltered robustness domains.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AuthenticationError, RateLimitError

from research.paper_benchmark_common import (
    EMBED_MODEL, EXTRACTION_MODEL, ROOT, TOP_K, CachedAPI, DenseIndex, EmbeddingCache,
    SearchDoc, append_jsonl, build_episode_snapshot, ensure_git_snapshot, episode_doc,
    exact_mcnemar, freeze_run, jsonl, latency_summary, log, paired_bootstrap,
    seal_jsonl, sha256_file, usage_summary, verify_episode_snapshot, verify_sealed_jsonl,
)


OFFICIAL_URL = "https://github.com/UCSB-NLP-Chang/GroupMemBench.git"
OFFICIAL_COMMIT = "e2682e01ff490acfe4fac2940159dce60307dfc9"
OFFICIAL_ROOT = ROOT / ".research_runs" / "official_sources" / "GroupMemBench"
RUN_DIR = ROOT / ".research_runs" / "groupmembench_temporal_episodes_official_v1"
MODEL = "gpt-5"
CONDITIONS = ("RAW", "RAW+EPISODES")
PRIMARY_DOMAINS = ("Finance", "Technology")
ROBUSTNESS_DOMAINS = ("Healthcare", "Manufacturing")
DOMAINS = PRIMARY_DOMAINS + ROBUSTNESS_DOMAINS
QTYPES = ("multi_hop", "knowledge_update", "temporal", "user_implicit", "term_ambiguity", "abstention")
PRIMARY_QTYPES = ("knowledge_update", "temporal")
CHUNK_SIZE = 40


def _official_path(domain: str) -> Path:
    return OFFICIAL_ROOT / "data" / "final" / domain / f"synthetic_domain_channels_rolevariants_{domain}.json"


def _questions_path(domain: str, qtype: str) -> Path:
    return OFFICIAL_ROOT / "questions" / domain / f"{qtype}.jsonl"


def _source_manifest(domains: tuple[str, ...]) -> dict:
    files = [OFFICIAL_ROOT / "prompts" / "hipporag_agent_system.txt",
             OFFICIAL_ROOT / "prompts" / "hipporag_judge_system.txt"]
    files += [_official_path(domain) for domain in domains]
    return {
        "benchmark": "GroupMemBench", "official_url": OFFICIAL_URL,
        "official_commit": OFFICIAL_COMMIT,
        "files": {str(path.relative_to(OFFICIAL_ROOT)): sha256_file(path) for path in files},
        "implementation": {
            name: sha256_file(ROOT / "research" / name) for name in (
                "groupmembench_episode_run.py", "paper_benchmark_common.py",
                "query_independent_episode_pipeline.py", "temporal_episode_prototype.py",
                "PAPER_BENCHMARK_PROTOCOL.md",
            )
        },
    }


def _evaluation_manifest(domains: tuple[str, ...]) -> dict:
    manifest = _source_manifest(domains)
    manifest["question_files"] = {
        str(_questions_path(domain, qtype).relative_to(OFFICIAL_ROOT)): sha256_file(_questions_path(domain, qtype))
        for domain in domains for qtype in QTYPES
    }
    return manifest


def load_messages(domains: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, list[SearchDoc]], dict[str, SearchDoc]]:
    frame_rows: list[dict] = []
    raw_by_domain: dict[str, list[SearchDoc]] = defaultdict(list)
    raw_by_id: dict[str, SearchDoc] = {}
    for domain in domains:
        channels = json.loads(_official_path(domain).read_text())
        chunks = []
        for channel, raw_messages in channels.items():
            messages = sorted(raw_messages, key=lambda row: (str(row.get("timestamp", "")), str(row.get("msg_node", ""))))
            by_phase: dict[str, list[dict]] = defaultdict(list)
            for message in messages:
                by_phase[str(message.get("phase_name") or "unknown")].append(message)
                body = html.unescape(str(message.get("content") or "")).strip()
                tags = [f"user={message.get('author') or '?'}"]
                for key, label in (("role", "speaker_role"), ("phase_name", "phase_name"),
                                   ("topic", "topic"), ("timestamp", "timestamp"),
                                   ("reply_to", "reply_to"), ("msg_node", "msg_node")):
                    if message.get(key) not in (None, ""):
                        tags.append(f"{label}={message[key]}")
                tags.insert(2, f"channel={channel}")
                rendered = f"[{' / '.join(tags)}]\n{body}"
                source_id = f"{domain}|{message['msg_node']}"
                doc = SearchDoc(f"raw:{domain}:{source_id}", body, rendered, (source_id,), "raw")
                raw_by_domain[domain].append(doc)
                if source_id in raw_by_id:
                    raise ValueError(f"duplicate GroupMemBench source id: {source_id}")
                raw_by_id[source_id] = doc
            for phase, phase_messages in by_phase.items():
                for start in range(0, len(phase_messages), CHUNK_SIZE):
                    part = phase_messages[start:start + CHUNK_SIZE]
                    chunks.append((str(part[0].get("timestamp", "")), channel, phase, part))
        for session_index, (_stamp, channel, _phase, part) in enumerate(
            sorted(chunks, key=lambda item: (item[0], item[1], item[2])), 1
        ):
            for message in part:
                frame_rows.append({
                    "network_id": domain, "session_index": session_index,
                    "turn_id": f"{domain}|{message['msg_node']}",
                    "timestamp": str(message.get("timestamp") or ""),
                    "speaker_display_name": f"{message.get('author') or '?'} ({message.get('role') or '?'}, {channel})",
                    "message": html.unescape(str(message.get("content") or "")).strip(),
                })
    return pd.DataFrame(frame_rows), dict(raw_by_domain), raw_by_id


def load_questions(domains: tuple[str, ...], *, include_gold: bool) -> list[dict]:
    rows = []
    for domain in domains:
        for qtype in QTYPES:
            for line in _questions_path(domain, qtype).read_text().splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                row = {
                    "qa_key": f"{domain}:{qtype}:{raw['id']}", "id": str(raw["id"]),
                    "domain": domain, "qtype": qtype, "question": str(raw["question"]),
                    "asking_user_id": str(raw.get("asking_user_id") or ""),
                }
                if include_gold:
                    row["gold"] = str(raw.get("answer") or "")
                rows.append(row)
    return rows


def _split_final(text: str) -> tuple[str, str]:
    lines = [line.rstrip() for line in (text or "").splitlines()]
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip().lower().startswith(("final:", "final answer:")):
            return "\n".join(lines[:index]).strip(), lines[index].split(":", 1)[1].strip()
    cleaned = [line.strip() for line in lines if line.strip()]
    return (text or "").strip(), cleaned[-1] if cleaned else ""


def _parse_judgment(text: str) -> bool | None:
    final = _split_final(text)[1].lower()
    if "incorrect" in final or "wrong" in final or "not correct" in final:
        return False
    if "correct" in final:
        return True
    return None


def preflight(domains: tuple[str, ...]) -> CachedAPI:
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY отсутствует в .env")
    ensure_git_snapshot(OFFICIAL_ROOT, OFFICIAL_URL, OFFICIAL_COMMIT)
    for domain in domains:
        if not _official_path(domain).exists():
            raise RuntimeError(f"official data missing: {domain}")
    api = CachedAPI(RUN_DIR, api_key=key)
    api.client.models.retrieve(MODEL)
    api.client.models.retrieve(EXTRACTION_MODEL)
    api.client.models.retrieve(EMBED_MODEL)
    log(RUN_DIR, f"preflight OK: GroupMemBench@{OFFICIAL_COMMIT[:12]}, domains={','.join(domains)}")
    return api


def prepare_memory(domains: tuple[str, ...], api: CachedAPI) -> None:
    frame, _raw, _lookup = load_messages(domains)
    sessions = frame[["network_id", "session_index"]].drop_duplicates().shape[0]
    log(RUN_DIR, f"memory: {len(frame)} messages / {sessions} bounded chunks; QA unopened")
    build_episode_snapshot(frame, RUN_DIR, api, source_manifest=_source_manifest(domains))
    log(RUN_DIR, "memory: sealed")


def _load_episode_docs(raw_by_id: dict[str, SearchDoc]) -> dict[str, list[SearchDoc]]:
    result: dict[str, list[SearchDoc]] = defaultdict(list)
    for row in jsonl(RUN_DIR / "episodes.jsonl"):
        result[row["network_id"]].append(episode_doc(row, raw_by_id))
    return dict(result)


def run_inference(domains: tuple[str, ...], api: CachedAPI) -> None:
    verify_episode_snapshot(RUN_DIR, _source_manifest(domains))
    _frame, raw_by_domain, raw_by_id = load_messages(domains)
    episodes = _load_episode_docs(raw_by_id)
    embeddings = EmbeddingCache(RUN_DIR, api)
    agent_system = (OFFICIAL_ROOT / "prompts" / "hipporag_agent_system.txt").read_text().strip()
    done = {(row["qa_key"], row["condition"]) for row in jsonl(RUN_DIR / "predictions.jsonl")}
    questions = load_questions(domains, include_gold=False)
    pending = [(row, condition) for row in questions for condition in CONDITIONS
               if (row["qa_key"], condition) not in done]
    log(RUN_DIR, f"inference: {len(pending)} pending / {len(done)} cached")
    completed = 0
    for domain in domains:
        domain_pending = [(row, condition) for row, condition in pending if row["domain"] == domain]
        if not domain_pending:
            continue
        indexes = {
            "RAW": DenseIndex(raw_by_domain[domain], embeddings, f"{domain}-raw"),
            "RAW+EPISODES": DenseIndex(raw_by_domain[domain] + episodes.get(domain, []), embeddings,
                                        f"{domain}-raw-episodes"),
        }
        unique_rows = {row["qa_key"]: row for row, _condition in domain_pending}
        query_by_key = {key: (f"{row['asking_user_id']} {row['question']}" if row["asking_user_id"] else row["question"])
                        for key, row in unique_rows.items()}
        vectors = embeddings.all(list(query_by_key.values()))
        vector_by_key = dict(zip(query_by_key, vectors))
        for row, condition in domain_pending:
            docs, retrieval_ms = indexes[condition].search_vector(vector_by_key[row["qa_key"]], TOP_K)
            passages = "\n\n".join(f"[{i}] {doc.rendered}" for i, doc in enumerate(docs, 1))
            asker = f"Asking user: {row['asking_user_id']}\n\n" if row["asking_user_id"] else ""
            user_prompt = (f"{asker}Question:\n{row['question']}\n\nRetrieved passages:\n{passages}\n\n"
                           "Answer the question using the retrieved passages.")
            started = time.perf_counter()
            output = api.chat(model=MODEL, messages=[{"role": "system", "content": agent_system},
                                                     {"role": "user", "content": user_prompt}],
                              temperature=.2, max_tokens=512, phase="answer")
            reasoning, answer = _split_final(output)
            append_jsonl(RUN_DIR / "predictions.jsonl", {
                **row, "condition": condition, "answer": answer, "agent_reasoning": reasoning,
                "retrieved_ids": [doc.doc_id for doc in docs],
                "exposed_source_ids": list(dict.fromkeys(s for doc in docs for s in doc.source_ids)),
                "retrieval_ms": retrieval_ms, "answer_wall_ms": (time.perf_counter() - started) * 1000,
            })
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                log(RUN_DIR, f"inference: {completed}/{len(pending)} pending")
        del indexes
    seal_jsonl(
        RUN_DIR / "predictions.jsonl", RUN_DIR / "predictions_meta.json",
        expected_rows=len(questions) * len(CONDITIONS), identity_fields=("qa_key", "condition"),
        metadata={
            "benchmark": "GroupMemBench", "official_commit": OFFICIAL_COMMIT,
            "conditions": list(CONDITIONS), "answer_model": MODEL,
            "embedding_model": EMBED_MODEL, "top_k": TOP_K,
            "episode_snapshot_sha256": sha256_file(RUN_DIR / "episodes.jsonl"),
        },
    )


def score(domains: tuple[str, ...], api: CachedAPI) -> None:
    judge_system = (OFFICIAL_ROOT / "prompts" / "hipporag_judge_system.txt").read_text().strip()
    expected = len(load_questions(domains, include_gold=False)) * len(CONDITIONS)
    verify_sealed_jsonl(RUN_DIR / "predictions.jsonl", RUN_DIR / "predictions_meta.json",
                        expected_rows=expected)
    predictions = {(row["qa_key"], row["condition"]): row for row in jsonl(RUN_DIR / "predictions.jsonl")}
    gold = {row["qa_key"]: row for row in load_questions(domains, include_gold=True)}
    evaluation_manifest = _evaluation_manifest(domains)
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
        user_prompt = (f"Question:\n{truth['question']}\n\nGold Answer:\n{truth['gold']}\n\n"
                       f"Agent Answer:\n{prediction['answer']}\n")
        output = api.chat(model=MODEL, messages=[{"role": "system", "content": judge_system},
                                                 {"role": "user", "content": user_prompt}],
                          temperature=.2, max_tokens=256, phase="judge")
        reasoning, final = _split_final(output)
        judgment = _parse_judgment(final)
        append_jsonl(RUN_DIR / "results.jsonl", {
            "qa_key": prediction["qa_key"], "domain": prediction["domain"],
            "qtype": prediction["qtype"], "condition": prediction["condition"],
            "correct": int(judgment is True), "verdict": "Correct" if judgment else "Incorrect",
            "judge_reasoning": reasoning, "judge_answer": final,
        })
        if count % 25 == 0 or count == len(pending):
            log(RUN_DIR, f"score: {count}/{len(pending)} pending")
    seal_jsonl(
        RUN_DIR / "results.jsonl", RUN_DIR / "results_meta.json",
        expected_rows=expected, identity_fields=("qa_key", "condition"),
        metadata={
            "benchmark": "GroupMemBench", "official_commit": OFFICIAL_COMMIT,
            "conditions": list(CONDITIONS), "judge_model": MODEL,
            "predictions_sha256": sha256_file(RUN_DIR / "predictions.jsonl"),
            "evaluation_input_manifest_sha256": sha256_file(manifest_path),
        },
    )
    render_report(domains)


def _accuracy(rows: list[dict], condition: str) -> float:
    values = [int(row["correct"]) for row in rows if row["condition"] == condition]
    return sum(values) / len(values) if values else float("nan")


def render_report(domains: tuple[str, ...]) -> None:
    rows = [row for row in jsonl(RUN_DIR / "results.jsonl") if row["domain"] in domains]
    expected = sum(1 for _ in load_questions(domains, include_gold=False)) * len(CONDITIONS)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete results: {len(rows)}/{expected}")
    primary = [row for row in rows if row["domain"] in PRIMARY_DOMAINS and row["qtype"] in PRIMARY_QTYPES]
    delta, lo, hi = paired_bootstrap(primary, "RAW+EPISODES", "RAW")
    mcnemar = exact_mcnemar(primary, "RAW+EPISODES", "RAW")
    lines = [
        "# GroupMemBench — RAW vs TemporalEpisodes", "",
        f"- official source: `{OFFICIAL_COMMIT}`", f"- answer/judge: `{MODEL}` (official prompts)",
        f"- retrieval: `{EMBED_MODEL}`, unified cosine top-{TOP_K}",
        f"- memory builder: `{EXTRACTION_MODEL}`, query-independent, chunk={CHUNK_SIZE}", "",
        "## Pre-registered primary endpoint", "",
        "Filtered Finance+Technology, knowledge_update+temporal combined.", "",
        f"- RAW: {_accuracy(primary, 'RAW'):.3f}",
        f"- RAW+EPISODES: {_accuracy(primary, 'RAW+EPISODES'):.3f}",
        f"- paired delta: {delta:+.3f}, question bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]",
        f"- exact paired McNemar: EP-only={mcnemar['left_only']}, RAW-only={mcnemar['right_only']}, p={mcnemar['p_value']:.6f}",
        "", "## All cells", "", "| split | domain | qtype | n | RAW | EPISODES | delta |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for domain in domains:
        split = "primary/filtered" if domain in PRIMARY_DOMAINS else "robustness/unfiltered"
        for qtype in QTYPES:
            selected = [row for row in rows if row["domain"] == domain and row["qtype"] == qtype]
            n = len(selected) // 2
            raw, episode = _accuracy(selected, "RAW"), _accuracy(selected, "RAW+EPISODES")
            lines.append(f"| {split} | {domain} | {qtype} | {n} | {raw:.3f} | {episode:.3f} | {episode-raw:+.3f} |")
    usage = usage_summary(RUN_DIR)
    latency = latency_summary(jsonl(RUN_DIR / "predictions.jsonl"))
    lines += ["", "## Efficiency artifacts", "", f"- API calls recorded: {usage['api_calls']}",
              f"- usage by model: `{json.dumps(usage['by_model'], sort_keys=True)}`",
              f"- estimated API cost (uncached-rate upper bound, USD): {usage['estimated_usd_upper_bound']:.4f}",
              f"- price snapshot: {usage['price_snapshot_date']}; {usage['pricing_assumption']}",
              f"- retrieval latency by condition: `{json.dumps(latency, sort_keys=True)}`",
              f"- episode snapshot bytes: {(RUN_DIR / 'episodes.jsonl').stat().st_size}", "",
              "## Interpretation guardrails", "",
              "- Finance/Technology are the confirmatory filtered split.",
              "- Healthcare/Manufacturing are unfiltered robustness data and are not pooled into the primary claim.",
              "- Memory was sealed before question files were opened; raw provenance is embedded in every derived passage.",
              "- This is an adapter evaluation under the official harness, not a leaderboard submission."]
    (RUN_DIR / "report.md").write_text("\n".join(lines) + "\n")
    log(RUN_DIR, f"final report ready: {RUN_DIR / 'report.md'}")


def status() -> None:
    print(f"run_dir={RUN_DIR}")
    print("planned=120000 messages / 3153 extraction chunks / 745 QA / 1490 answers / 1490 judges")
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
    domains = DOMAINS
    if args.status:
        status(); return
    if args.freeze:
        destination = freeze_run(RUN_DIR, [Path(__file__), ROOT / "research" / "paper_benchmark_common.py",
                                           ROOT / "research" / "PAPER_BENCHMARK_PROTOCOL.md"])
        print(destination); return
    try:
        api = preflight(domains)
        if args.preflight:
            return
        if args.all or args.prepare_memory:
            prepare_memory(domains, api)
        if args.all or args.run_inference:
            run_inference(domains, api)
        if args.all or args.score:
            score(domains, api)
    except RateLimitError as error:
        log(RUN_DIR, f"rate limit; completed work is cached: {error}")
        raise SystemExit(75) from None
    except AuthenticationError as error:
        raise SystemExit(f"OpenAI authentication failed: {error}") from None


if __name__ == "__main__":
    main()
