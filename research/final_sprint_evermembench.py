"""Final validation sprint on EverMemBench: experiments A, B1, B2 and C.

A   RAW+EVENTS on all 2,400 QA: unified dense retrieval over raw messages and the sealed
    individual event documents, top-10, exactly as the frozen unlinked-events control.
B1  RAW+EVENTS-token-matched-chrono-nodesc (300 TP): the frozen RAW+EVENTS-token-matched
    documents in the frozen chronological order, each event document shown only as its raw
    source messages (no event text, owner, subject or label).
B2  RAW+EPISODES-chrono (300 TP): the sealed RAW+EPISODES top-10 documents in global
    chronological order by earliest source timestamp; episode rendering unchanged.
C   HyDE-RAW (300 TP): gpt-4.1-mini writes a hypothetical chat passage from the question text
    alone; its text-embedding-3-large vector retrieves the raw top-10.

Protocol: research/FINAL_EXPERIMENTS_PROTOCOL.md. Nothing here calls an API before --run.

    python3 -m research.final_sprint_evermembench --preflight          # offline
    python3 -m research.final_sprint_evermembench --run A              # after explicit GO
    python3 -m research.final_sprint_evermembench --report A
    python3 -m research.final_sprint_evermembench --freeze A
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AuthenticationError, RateLimitError

from research import evermembench_chronological_order_control as chrono
from research import evermembench_episode_run as base
from research import evermembench_parallel_resume as runner
from research import evermembench_temporal_budget_control as control
from research import evermembench_unlinked_events_control as unlinked
from research import paper_benchmark_common as common
from research.paper_benchmark_common import (
    EMBED_MODEL,
    MODEL_PRICES_USD_PER_M,
    CachedAPI,
    EmbeddingCache,
    SearchDoc,
    append_jsonl,
    episode_doc,
    freeze_run,
    jsonl,
    log,
    seal_jsonl,
    sha256_bytes,
    sha256_file,
    usage_summary,
    verify_sealed_jsonl,
)


ROOT = base.ROOT
FROZEN = ROOT / ".research_runs" / "frozen"
SOURCE = FROZEN / "evermembench_temporal_episodes_official_v1_20260915_021550_MSK" / "run"
BUDGET = FROZEN / "evermembench_temporal_budget_control_v1_20260915_021134_MSK" / "run"
UNLINKED = FROZEN / "evermembench_unlinked_events_control_v1_20260915_022532_MSK" / "run"
CHRONO = FROZEN / "evermembench_chronological_order_control_v2_20260915_190741_MSK" / "run"
EVENT_INDEX = (FROZEN / "evermembench_unlinked_events_control_v1_20260915_022532_MSK" / "related"
               / "evermembench_unlinked_events_index_v1")
FROZEN_DIRS = tuple(path.parent for path in (SOURCE, BUDGET, UNLINKED, CHRONO))
PROTOCOL = ROOT / "research" / "FINAL_EXPERIMENTS_PROTOCOL.md"
AMENDMENT = ROOT / "research" / "EVERMEMBENCH_HYDE_TOKEN_MATCHED_AMENDMENT.md"
AMENDMENT_DIR = ROOT / ".research_runs" / "amendments" / "evermembench_hyde_token_matched_20260916"
RUNNER = Path(__file__).resolve()
PREFLIGHT_DIR = ROOT / ".research_runs" / "final_sprint_preflight_v1"
RUN_DIRS = {
    "A": ROOT / ".research_runs" / "final_sprint_A_events_full_v1",
    "B1": ROOT / ".research_runs" / "final_sprint_B1_events_nodesc_v1",
    "B2": ROOT / ".research_runs" / "final_sprint_B2_episodes_chrono_v1",
    "C": ROOT / ".research_runs" / "final_sprint_C_hyde_raw_v1",
    "C2": ROOT / ".research_runs" / "final_sprint_C2_hyde_token_matched_v1",
    "P1": ROOT / ".research_runs" / "final_sprint_P1_simple_prop_chrono_v1",
}
CONDITIONS = {"A": "RAW+EVENTS", "B1": "RAW+EVENTS-token-matched-chrono-nodesc",
              "B2": "RAW+EPISODES-chrono", "C": "HyDE-RAW",
              "C2": "HyDE-RAW-token-matched-chrono",
              "P1": "RAW+SIMPLE-PROP-token-matched-chrono"}
AMENDED = ("C2",)
FROZEN_ROWS = {SOURCE: 4800, BUDGET: 900, UNLINKED: 600, CHRONO: 900}
TOP_K = 10
ANSWER_MAX_TOKENS = 1000
HYDE_MAX_TOKENS = 200
HYDE_PROMPT = (
    "Write a short passage, two to four sentences, of the kind that could appear in a workplace "
    "team chat and would contain the answer to the question below. Write plausible chat content "
    "and invent specific details if needed. Output only the passage.\n\nQuestion: {question}"
)
FEE_FACTOR = 1.10
JUDGE_OUTPUT_CEILING = 1000
MAX_FAILURE_SHARE = 0.02


# --------------------------------------------------------------------------
# Rendering and keys (pure, unit-tested)
# --------------------------------------------------------------------------

def build_prompt(prompts: dict, row: dict, context: str) -> str:
    """Official answer prompt, byte-identical to evermembench_parallel_resume.run_inference."""
    if row["question_type"] == "multiple_choice":
        return prompts["multiple_choice"].format(context=context, question=row["question"],
                                                 options=base._option_text(row["options"]))
    return prompts["open_ended"].format(context=context, question=row["question"])


def render_without_description(docs: list[SearchDoc], raw_by_id: dict[str, SearchDoc]) -> str:
    """Raw documents unchanged; each derived document replaced by its raw source messages only."""
    lines: list[str] = []
    for doc in docs:
        if doc.kind == "raw":
            lines.append(f"- {doc.rendered}")
            continue
        lines.extend(f"- {raw_by_id[source].rendered}" for source in doc.source_ids if source in raw_by_id)
    return "\n".join(lines) or "(No memories retrieved)"


def hyde_prompt(question_text: str) -> str:
    """Uses the question text only: no gold, evidence, category, id or options."""
    return HYDE_PROMPT.format(question=question_text)


def token_matched_chrono(ranked: list[SearchDoc], target_tokens: int, raw_by_id: dict[str, SearchDoc],
                         count_tokens) -> list[SearchDoc]:
    """C2: ranked documents cut to the frozen event token budget, then presented in time order.

    Both steps reuse the frozen controls' rules unchanged, so the only difference from
    RAW+EVENTS-token-matched-chrono is which documents the ranking produced.
    """
    chosen, _ = control.select_token_matched(ranked, target_tokens, count_tokens)
    return [doc for _stamp, doc in chrono.chronological(chosen, raw_by_id)]


def chat_key(prompt: str, max_tokens: int) -> str:
    payload = {
        "provider": "openrouter", "model": base.ANSWER_MODEL, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": max_tokens, "json": False, "extra_body": base.PROVIDER["answer"],
    }
    return "chat:" + sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())


def list_price(model: str, input_tokens: float, output_tokens: float) -> float:
    prices = MODEL_PRICES_USD_PER_M[model]
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


# --------------------------------------------------------------------------
# Frozen inputs (offline)
# --------------------------------------------------------------------------

@dataclass
class Data:
    questions: list[dict]
    prompts: dict
    raw_by_topic: dict[str, list[SearchDoc]]
    raw_by_id: dict[str, SearchDoc]
    events: dict[str, list[SearchDoc]]
    docs_by_id: dict[str, SearchDoc]
    frozen: dict[tuple[str, str], dict]
    cache: dict[str, str]
    cache_origin: dict[str, str]


def verify_frozen() -> dict:
    counts = {}
    for directory in FROZEN_DIRS:
        lines = (directory / "SHA256SUMS").read_text().splitlines()
        for line in lines:
            digest, name = line.split("  ", 1)
            if sha256_file(directory / name) != digest:
                raise RuntimeError(f"frozen checksum mismatch: {directory.name}/{name}")
        counts[directory.name] = len(lines)
    for run, rows in FROZEN_ROWS.items():
        verify_sealed_jsonl(run / "predictions.jsonl", run / "predictions_meta.json", expected_rows=rows)
        verify_sealed_jsonl(run / "results.jsonl", run / "results_meta.json", expected_rows=rows)
    return counts


def load_data() -> tuple[Data, dict]:
    checks = {"frozen_sha256sums_verified": verify_frozen()}
    _frame, raw_by_topic, raw_by_id = base.load_messages()
    events = unlinked.events_by_topic(SOURCE / "episodes.jsonl", raw_by_id)
    docs_by_id = {doc.doc_id: doc for docs in raw_by_topic.values() for doc in docs}
    docs_by_id.update({doc.doc_id: doc for docs in events.values() for doc in docs})
    for row in jsonl(SOURCE / "episodes.jsonl"):
        doc = episode_doc(row, raw_by_id)
        docs_by_id[doc.doc_id] = doc
    frozen: dict[tuple[str, str], dict] = {}
    for run in (SOURCE, BUDGET, UNLINKED, CHRONO):
        for row in jsonl(run / "predictions.jsonl"):
            frozen[(row["qa_key"], row["condition"])] = row
    cache: dict[str, str] = {}
    origin: dict[str, str] = {}
    for run in (SOURCE, BUDGET, UNLINKED, CHRONO):
        for line in open(run / "openrouter_chat_cache.jsonl"):
            row = json.loads(line)
            if row["key"] not in cache:
                cache[row["key"]] = row["value"]
                origin[row["key"]] = run.parent.name
    data = Data(base.load_questions(include_gold=False), base._prompts()["answer"], raw_by_topic, raw_by_id,
                events, docs_by_id, frozen, cache, origin)
    checks["cache_entries_available"] = len(cache)
    return data, checks


def event_matrix(topic: str, docs: list[SearchDoc]) -> np.ndarray:
    meta = json.loads((EVENT_INDEX / "index_meta.json").read_text())["topics"][topic]
    if meta["docs_sha256"] != unlinked.docs_digest(docs) or meta["events"] != len(docs):
        raise RuntimeError(f"event documents differ from the sealed event index: {topic}")
    path = EVENT_INDEX / meta["matrix_file"]
    if sha256_file(path) != meta["matrix_sha256"]:
        raise RuntimeError(f"sealed event matrix checksum mismatch: {path}")
    return np.load(path)


def tp_questions(data: Data) -> list[dict]:
    rows = [row for row in data.questions if row["minor"] == "TP"]
    if len(rows) != 300:
        raise RuntimeError(f"expected 300 TP questions, found {len(rows)}")
    return rows


# --------------------------------------------------------------------------
# Jobs per experiment (offline, no gold)
# --------------------------------------------------------------------------

def jobs_a(data: Data, count_tokens) -> tuple[list[dict], dict]:
    embed_dir = SOURCE / "embeddings"
    checks = Counter()
    jobs = []
    for topic in base.BATCHES:
        topic_questions = [row for row in data.questions if row["topic"] == topic]
        vectors = control.question_vectors(embed_dir, [row["question"] for row in topic_questions])
        raw_matrix = control.raw_index_matrix(embed_dir, topic, data.raw_by_topic[topic])
        unified_docs = data.raw_by_topic[topic] + data.events[topic]
        unified_matrix = np.vstack([raw_matrix, event_matrix(topic, data.events[topic])])
        for row, vector in zip(topic_questions, vectors):
            raw_order, _ = control.rank_raw(raw_matrix, vector, TOP_K)
            raw_docs = [data.raw_by_topic[topic][int(i)] for i in raw_order]
            if [d.doc_id for d in raw_docs] != data.frozen[(row["qa_key"], "RAW")]["retrieved_ids"]:
                raise RuntimeError(f"raw top-10 not reproduced: {row['qa_key']}")
            raw_context = control.render_context(raw_docs)
            if chat_key(build_prompt(data.prompts, row, raw_context), ANSWER_MAX_TOKENS) not in data.cache:
                raise RuntimeError(f"sealed RAW prompt not reproduced: {row['qa_key']}")
            checks["raw_prompt_reproduced"] += 1
            order, retrieval_ms = control.rank_raw(unified_matrix, vector, TOP_K)
            docs = [unified_docs[int(i)] for i in order]
            if row["minor"] == "TP":
                if [d.doc_id for d in docs] != data.frozen[(row["qa_key"], "RAW+EVENTS")]["retrieved_ids"]:
                    raise RuntimeError(f"TP RAW+EVENTS top-10 differs from the frozen control: {row['qa_key']}")
                checks["tp_top10_matches_frozen_unlinked"] += 1
            checks["no_event_in_top10"] += int(all(d.kind == "raw" for d in docs))
            context = control.render_context(docs)
            jobs.append({"question": row, "docs": docs, "context": context, "prompt": build_prompt(data.prompts, row, context),
                         "retrieval_ms": retrieval_ms, "context_tokens": count_tokens(context),
                         "raw_context_tokens": count_tokens(raw_context)})
    checks["questions"] = len(jobs)
    return jobs, dict(checks)


def jobs_b1(data: Data, count_tokens) -> tuple[list[dict], dict]:
    checks = Counter()
    jobs = []
    for row in tp_questions(data):
        source = data.frozen[(row["qa_key"], "RAW+EVENTS-token-matched")]
        docs = [data.docs_by_id[i] for i in source["retrieved_ids"]]
        if sha256_bytes(build_prompt(data.prompts, row, control.render_context(docs)).encode()) != source["prompt_sha256"]:
            raise RuntimeError(f"frozen RAW+EVENTS-token-matched prompt not reproduced: {row['qa_key']}")
        ordered = [doc for _stamp, doc in chrono.chronological(docs, data.raw_by_id)]
        chrono_row = data.frozen[(row["qa_key"], "RAW+EVENTS-token-matched-chrono")]
        if [d.doc_id for d in ordered] != chrono_row["retrieved_ids"]:
            raise RuntimeError(f"chronological order differs from the frozen chrono control: {row['qa_key']}")
        context = render_without_description(ordered, data.raw_by_id)
        if "[DERIVED" in context:
            raise RuntimeError(f"derived label leaked into B1 context: {row['qa_key']}")
        checks["questions"] += 1
        checks["questions_with_event_docs"] += int(any(d.kind != "raw" for d in ordered))
        jobs.append({"question": row, "docs": ordered, "context": context, "prompt": build_prompt(data.prompts, row, context),
                     "retrieval_ms": 0.0, "context_tokens": count_tokens(context),
                     "reference_context_tokens": chrono_row["context_tokens"]})
    return jobs, dict(checks)


def jobs_p1(data: Data, count_tokens) -> tuple[list[dict], dict]:
    """P1: the SIMPLE-PROP index in place of the event index, under the controls that produced
    RAW+EVENTS-token-matched-chrono -- same per-question token budget, same chronological
    order. Answers the question B1 left open: is the Temporal Duration advantage carried by the
    event schema, or would a self-contained paraphrase with the same provenance get it too?
    """
    from research import simple_prop_baseline as prop

    checks = Counter()
    jobs = []
    props_by_topic: dict[str, list[SearchDoc]] = {topic: [] for topic in base.BATCHES}
    for row in jsonl(prop.RUN_DIR / "props.jsonl"):
        props_by_topic[row["network_id"]].append(prop.prop_doc(row, data.raw_by_id))
    embed_dir = SOURCE / "embeddings"
    for topic in base.BATCHES:
        # The sealed question-vector cache is keyed by the full per-topic batch, so the
        # vectors are read for every question of the topic and the TP rows selected by index.
        all_rows = [r for r in data.questions if r["topic"] == topic]
        all_vectors = control.question_vectors(embed_dir, [r["question"] for r in all_rows])
        tp_keys = {r["qa_key"] for r in tp_questions(data) if r["topic"] == topic}
        pairs = [(r, v) for r, v in zip(all_rows, all_vectors) if r["qa_key"] in tp_keys]
        if not pairs:
            continue
        topic_rows = [r for r, _ in pairs]
        vectors = [v for _, v in pairs]
        raw_matrix = control.raw_index_matrix(embed_dir, topic, data.raw_by_topic[topic])
        unified_docs = data.raw_by_topic[topic] + props_by_topic[topic]
        unified_matrix = np.vstack([raw_matrix, prop.prop_matrix(topic, props_by_topic[topic])])
        for row, vector in zip(topic_rows, vectors):
            reference = data.frozen[(row["qa_key"], "RAW+EVENTS-token-matched-chrono")]
            order, retrieval_ms = control.rank_raw(unified_matrix, vector, control.RANK_DEPTH)
            ranked = [unified_docs[int(i)] for i in order]
            ordered = token_matched_chrono(ranked, reference["context_tokens"], data.raw_by_id, count_tokens)
            context = control.render_context(ordered)
            checks["questions"] += 1
            checks["questions_with_prop_docs"] += int(any(d.kind == "prop" for d in ordered))
            jobs.append({"question": row, "docs": ordered, "context": context,
                         "prompt": build_prompt(data.prompts, row, context), "retrieval_ms": retrieval_ms,
                         "context_tokens": count_tokens(context),
                         "reference_context_tokens": reference["context_tokens"]})
    return jobs, dict(checks)


def jobs_b2(data: Data, count_tokens) -> tuple[list[dict], dict]:
    checks = Counter()
    jobs = []
    for row in tp_questions(data):
        source = data.frozen[(row["qa_key"], "RAW+EPISODES")]
        docs = [data.docs_by_id[i] for i in source["retrieved_ids"]]
        source_context = control.render_context(docs)
        if chat_key(build_prompt(data.prompts, row, source_context), ANSWER_MAX_TOKENS) not in data.cache:
            raise RuntimeError(f"sealed RAW+EPISODES prompt not reproduced: {row['qa_key']}")
        ordered = [doc for _stamp, doc in chrono.chronological(docs, data.raw_by_id)]
        context = control.render_context(ordered)
        chrono.verify_pure_permutation(docs, ordered, source_context, context)
        checks["questions"] += 1
        checks["order_changed"] += int([d.doc_id for d in ordered] != source["retrieved_ids"])
        jobs.append({"question": row, "docs": ordered, "context": context, "prompt": build_prompt(data.prompts, row, context),
                     "retrieval_ms": 0.0, "context_tokens": count_tokens(context),
                     "reference_context_tokens": count_tokens(source_context)})
    return jobs, dict(checks)


def hyde_jobs(data: Data) -> list[dict]:
    return [{"question": row, "prompt": hyde_prompt(row["question"])} for row in tp_questions(data)]


# --------------------------------------------------------------------------
# Preflight (offline)
# --------------------------------------------------------------------------

def answer_cost(new_jobs: list[dict], encoder, overhead: float, out_by_type: dict[str, float]) -> dict:
    input_tokens = sum(len(encoder.encode(job["prompt"])) + overhead for job in new_jobs)
    expected_output = sum(out_by_type[job["question"]["question_type"]] for job in new_jobs)
    model = base.ANSWER_MODEL
    return {
        "calls": len(new_jobs),
        "input_tokens": round(input_tokens),
        "expected_output_tokens": round(expected_output),
        "expected_usd": list_price(model, input_tokens, expected_output),
        "ceiling_usd": list_price(model, input_tokens, len(new_jobs) * ANSWER_MAX_TOKENS) * FEE_FACTOR,
    }


def judge_cost(calls: int, mean_input: float, mean_output: float) -> dict:
    model = base.JUDGE_MODEL
    return {
        "calls_ceiling": calls, "mean_input_tokens": round(mean_input), "mean_output_tokens": round(mean_output, 1),
        "expected_usd": list_price(model, calls * mean_input, calls * mean_output),
        "ceiling_usd": list_price(model, calls * mean_input * 1.2, calls * JUDGE_OUTPUT_CEILING) * FEE_FACTOR,
    }


def phase_means(run: Path, phase: str, model: str) -> tuple[float, float]:
    rows = [r for r in jsonl(run / "api_usage.jsonl") if r.get("phase") == phase and r["model"] == model]
    return (sum(r["input_tokens"] for r in rows) / len(rows), sum(r["output_tokens"] for r in rows) / len(rows))


def preflight() -> dict:
    import tiktoken

    data, load_checks = load_data()
    encoder = tiktoken.get_encoding(control.TOKEN_ENCODING)
    count_tokens = lambda text: len(encoder.encode(text))
    usage = {row["key"]: row for row in jsonl(SOURCE / "api_usage.jsonl") if row.get("phase") == "answer"}

    # Calibrate API overhead and output length on the sealed RAW prompts, reconstructed byte for byte.
    raw_jobs, a_checks = jobs_a(data, count_tokens)
    embed_dir = SOURCE / "embeddings"
    overhead, out_by_type = [], defaultdict(list)
    for topic in base.BATCHES:
        topic_questions = [row for row in data.questions if row["topic"] == topic]
        vectors = control.question_vectors(embed_dir, [row["question"] for row in topic_questions])
        raw_matrix = control.raw_index_matrix(embed_dir, topic, data.raw_by_topic[topic])
        for row, vector in zip(topic_questions, vectors):
            order, _ = control.rank_raw(raw_matrix, vector, TOP_K)
            prompt = build_prompt(data.prompts, row, control.render_context([data.raw_by_topic[topic][int(i)] for i in order]))
            usage_row = usage.get(chat_key(prompt, ANSWER_MAX_TOKENS))
            if usage_row:
                overhead.append(usage_row["input_tokens"] - count_tokens(prompt))
                out_by_type[row["question_type"]].append(usage_row["output_tokens"])
    mean_overhead = float(np.mean(overhead))
    out_means = {k: float(np.mean(v)) for k, v in out_by_type.items()}
    judge_in, judge_out = phase_means(SOURCE, "judge", base.JUDGE_MODEL)
    tp_judge_in, tp_judge_out = phase_means(CHRONO, "judge", base.JUDGE_MODEL)

    report: dict = {"load_checks": load_checks,
                    "calibration": {"api_input_overhead_tokens": round(mean_overhead, 2), "calibration_prompts": len(overhead),
                                    "mean_output_tokens_by_type": {k: round(v, 2) for k, v in out_means.items()},
                                    "judge_mean_tokens_all": [round(judge_in), round(judge_out, 1)],
                                    "judge_mean_tokens_tp": [round(tp_judge_in), round(tp_judge_out, 1)]},
                    "experiments": {}}

    for name, builder in (("A", None), ("B1", jobs_b1), ("B2", jobs_b2)):
        jobs, checks = (raw_jobs, a_checks) if builder is None else builder(data, count_tokens)
        new =[j for j in jobs if chat_key(j["prompt"], ANSWER_MAX_TOKENS) not in data.cache]
        reused = len(jobs) - len(new)
        reuse_origin = Counter(data.cache_origin[chat_key(j["prompt"], ANSWER_MAX_TOKENS)] for j in jobs
                               if chat_key(j["prompt"], ANSWER_MAX_TOKENS) in data.cache)
        answers = answer_cost(new, encoder, mean_overhead, out_means)
        open_new = sum(1 for j in new if j["question"]["question_type"] == "open_ended")
        means = (judge_in, judge_out) if name == "A" else (tp_judge_in, tp_judge_out)
        judges = judge_cost(open_new, *means)
        report["experiments"][name] = {
            "condition": CONDITIONS[name], "questions": len(jobs), "checks": checks,
            "answers_reused_from_cache": reused, "reuse_origin": dict(reuse_origin),
            "new_answer_calls": answers["calls"],
            "open_ended_questions": sum(1 for j in jobs if j["question"]["question_type"] == "open_ended"),
            "multiple_choice_scored_locally": sum(1 for j in jobs if j["question"]["question_type"] == "multiple_choice"),
            "new_judge_calls_ceiling": open_new, "answers": answers, "judge": judges,
            "mean_context_tokens": round(float(np.mean([j["context_tokens"] for j in jobs])), 1),
            "expected_usd": answers["expected_usd"] + judges["expected_usd"],
            "ceiling_usd": answers["ceiling_usd"] + judges["ceiling_usd"],
        }

    # C: generation calls, embeddings, then raw top-10 answers (prompt tokens estimated from sealed RAW TP prompts).
    hyde = hyde_jobs(data)
    hyde_new = [j for j in hyde if chat_key(j["prompt"], HYDE_MAX_TOKENS) not in data.cache]
    hyde_in = sum(count_tokens(j["prompt"]) + mean_overhead for j in hyde_new)
    hyde_expected_out = 90 * len(hyde_new)
    tp_raw = [j for j in raw_jobs if j["question"]["minor"] == "TP"]
    answer_in = float(np.mean([count_tokens(j["prompt"]) for j in tp_raw])) + mean_overhead
    hyde_texts_tokens = len(hyde_new) * 90
    c = {
        "condition": CONDITIONS["C"], "questions": len(hyde),
        "new_generation_calls": len(hyde_new), "new_embedding_calls": -(-len(hyde_new) // 128),
        "new_answer_calls": len(hyde_new), "new_judge_calls_ceiling": len(hyde_new),
        "generation": {"expected_usd": list_price(base.ANSWER_MODEL, hyde_in, hyde_expected_out),
                       "ceiling_usd": list_price(base.ANSWER_MODEL, hyde_in, len(hyde_new) * HYDE_MAX_TOKENS) * FEE_FACTOR},
        "embeddings": {"expected_usd": list_price(EMBED_MODEL, hyde_texts_tokens, 0),
                       "ceiling_usd": list_price(EMBED_MODEL, len(hyde_new) * HYDE_MAX_TOKENS, 0) * FEE_FACTOR},
        "answers": {"mean_input_tokens_estimate": round(answer_in),
                    "expected_usd": list_price(base.ANSWER_MODEL, len(hyde_new) * answer_in, len(hyde_new) * out_means["open_ended"]),
                    "ceiling_usd": list_price(base.ANSWER_MODEL, len(hyde_new) * answer_in * 1.5, len(hyde_new) * ANSWER_MAX_TOKENS) * FEE_FACTOR},
        "judge": judge_cost(len(hyde_new), tp_judge_in, tp_judge_out),
    }
    c["expected_usd"] = sum(c[k]["expected_usd"] for k in ("generation", "embeddings", "answers", "judge"))
    c["ceiling_usd"] = sum(c[k]["ceiling_usd"] for k in ("generation", "embeddings", "answers", "judge"))
    report["experiments"]["C"] = c

    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    (PREFLIGHT_DIR / "preflight_evermembench.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    return report


# --------------------------------------------------------------------------
# Execution (only after GO)
# --------------------------------------------------------------------------

def make_router(run_dir: Path, data: Data) -> CachedAPI:
    router = control.make_router(run_dir)
    for key, value in data.cache.items():
        router.values.setdefault(key, value)
    return router


def budget_guard(run_dir: Path, ceiling: float) -> None:
    spent = usage_summary(run_dir)["estimated_usd_upper_bound"] * FEE_FACTOR
    if spent > ceiling:
        raise SystemExit(f"stop: spend estimate {spent:.4f} exceeds the preflight ceiling {ceiling:.4f}")


def input_records(name: str, jobs: list[dict]) -> list[dict]:
    return [{"qa_key": j["question"]["qa_key"], "condition": CONDITIONS[name], "doc_ids": [d.doc_id for d in j["docs"]],
             "context_sha256": sha256_bytes(j["context"].encode()), "prompt_sha256": sha256_bytes(j["prompt"].encode()),
             "context_tokens": j["context_tokens"]} for j in jobs]


def manifest(name: str, inputs_sha256: str, checks: dict, ceiling: float) -> dict:
    amendment = {
        "amendment": AMENDMENT.name, "amendment_sha256": sha256_file(AMENDMENT),
        "hyde_generations_from": RUN_DIRS["C"].name,
        "hyde_queries_sha256": sha256_file(RUN_DIRS["C"] / "hyde_queries.jsonl"),
        "token_target": f"frozen {unlinked.EVENTS_TOKEN} context_tokens per question",
        "rank_depth": control.RANK_DEPTH,
    } if name in AMENDED else {}
    return {
        "experiment": name, "condition": CONDITIONS[name], "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
        **amendment,
        "runner": RUNNER.name, "runner_sha256": sha256_file(RUNNER),
        "dependencies_sha256": {Path(m.__file__).name: sha256_file(Path(m.__file__))
                                for m in (chrono, base, runner, control, unlinked, common)},
        "frozen_sha256sums": {d.name: sha256_file(d / "SHA256SUMS") for d in FROZEN_DIRS},
        "event_index_meta_sha256": sha256_file(EVENT_INDEX / "index_meta.json"),
        "inputs_sha256": inputs_sha256, "checks": checks, "ceiling_usd": ceiling,
        "answer_model": base.ANSWER_MODEL, "answer_provider": base.PROVIDER["answer"], "answer_max_tokens": ANSWER_MAX_TOKENS,
        "judge_model": base.JUDGE_MODEL, "judge_provider": base.PROVIDER["judge"], "judge_call": runner.JUDGE_CALL_VERSION,
        "top_k": TOP_K, "hyde_prompt": HYDE_PROMPT if name == "C" else None,
    }


def answer_phase(name: str, run_dir: Path, jobs: list[dict], router: CachedAPI, data: Data, workers: int, ceiling: float) -> None:
    existing = jsonl(run_dir / "predictions.jsonl")
    done = {row["qa_key"] for row in existing}
    pending = [job for job in jobs if job["question"]["qa_key"] not in done]
    log(run_dir, f"answers: {len(pending)} pending / {len(done)} done")
    completed = [0]

    def work(job: dict) -> dict:
        row = job["question"]
        key = chat_key(job["prompt"], ANSWER_MAX_TOKENS)
        reused = data.cache_origin.get(key)
        started = time.perf_counter()
        raw_answer = router.chat(model=base.ANSWER_MODEL, messages=[{"role": "user", "content": job["prompt"]}],
                                 temperature=0, max_tokens=ANSWER_MAX_TOKENS, phase="answer", extra_body=base.PROVIDER["answer"])
        completed[0] += 1
        if completed[0] % 100 == 0:
            budget_guard(run_dir, ceiling)
        sources = control.unique_source_ids(job["docs"])
        return {
            **{f: row[f] for f in ("qa_key", "topic", "id", "major", "minor", "question", "question_type")},
            "condition": CONDITIONS[name], "answer": runner.official_answer(raw_answer, row["question_type"]),
            "answer_raw": raw_answer, "answer_cache": f"reused:{reused}" if reused else "new",
            "retrieved_ids": [d.doc_id for d in job["docs"]], "exposed_source_ids": sources,
            "unique_source_count": len(sources), "derived_docs": sum(1 for d in job["docs"] if d.kind != "raw"),
            "context_tokens": job["context_tokens"], "prompt_sha256": sha256_bytes(job["prompt"].encode()),
            "retrieval_ms": job["retrieval_ms"], "answer_wall_ms": (time.perf_counter() - started) * 1000,
        }

    chrono.parallel(run_dir, pending, work, lambda row: append_jsonl(run_dir / "predictions.jsonl", row),
                    workers=workers, label=f"{name} answers")
    rows = jsonl(run_dir / "predictions.jsonl")
    failures = sum(1 for r in rows if r["answer"] in ("[EMPTY]", "[INVALID]", "[FAILED]"))
    if failures > max(3, MAX_FAILURE_SHARE * len(rows)):
        raise SystemExit(f"stop: {failures} answer failure markers exceed the protocol limit")
    seal_jsonl(run_dir / "predictions.jsonl", run_dir / "predictions_meta.json", expected_rows=len(jobs),
               identity_fields=("qa_key", "condition"),
               metadata={"experiment": name, "protocol_sha256": sha256_file(PROTOCOL), "runner_sha256": sha256_file(RUNNER),
                         "answer_failures": failures, "reused_answers": sum(1 for r in rows if r["answer_cache"] != "new")})


def judge_phase(name: str, run_dir: Path, expected: int, router: CachedAPI, workers: int, ceiling: float) -> None:
    verify_sealed_jsonl(run_dir / "predictions.jsonl", run_dir / "predictions_meta.json", expected_rows=expected)
    predictions = {row["qa_key"]: row for row in jsonl(run_dir / "predictions.jsonl")}
    gold = {row["qa_key"]: row for row in base.load_questions(include_gold=True)}  # only after sealed answers
    prompts = base._prompts()["llm_judge"]
    done = {row["qa_key"] for row in jsonl(run_dir / "results.jsonl")}
    pending = [key for key in predictions if key not in done]
    log(run_dir, f"score: {len(pending)} pending / {len(done)} done")
    completed = [0]

    def work(key: str) -> dict:
        prediction, truth = predictions[key], gold[key]
        meta: dict = {}
        started = time.perf_counter()
        if truth["question_type"] == "multiple_choice":
            correct = runner.official_evaluator_mc(prediction["answer"], truth["gold"])
            judge_text = "official exact multiple-choice comparison"
        else:
            prompt = prompts["user_prompt"].format(question=truth["question"], golden_answer=truth["gold"],
                                                   generated_answer=prediction["answer"])
            call = runner.judge_chat(router, [{"role": "system", "content": prompts["system_prompt"]},
                                              {"role": "user", "content": prompt}])
            correct, branch = runner.official_judge_label(call["content"])
            judge_text = call["content"]
            meta = {"judge_finish_reason": call["finish_reason"], "judge_attempts": call["attempts"], "judge_parse": branch}
            completed[0] += 1
            if completed[0] % 100 == 0:
                budget_guard(run_dir, ceiling)
        exposed, evidence = set(prediction["exposed_source_ids"]), set(truth["gold_source_ids"])
        return {
            "qa_key": key, "topic": prediction["topic"], "major": prediction["major"], "minor": prediction["minor"],
            "question_type": prediction["question_type"], "condition": prediction["condition"], "correct": int(correct),
            "judge": judge_text, **meta, "judge_wall_ms": (time.perf_counter() - started) * 1000,
            "evidence_recall": len(exposed & evidence) / len(evidence) if evidence else 1.0,
            "evidence_precision": len(exposed & evidence) / len(exposed) if exposed else 0.0,
        }

    chrono.parallel(run_dir, pending, work, lambda row: append_jsonl(run_dir / "results.jsonl", row),
                    workers=workers, label=f"{name} score")
    seal_jsonl(run_dir / "results.jsonl", run_dir / "results_meta.json", expected_rows=expected,
               identity_fields=("qa_key", "condition"),
               metadata={"experiment": name, "protocol_sha256": sha256_file(PROTOCOL), "runner_sha256": sha256_file(RUNNER),
                         "judge_model": base.JUDGE_MODEL, "judge_call": runner.JUDGE_CALL_VERSION,
                         "predictions_sha256": sha256_file(run_dir / "predictions.jsonl")})


def hyde_phase(run_dir: Path, data: Data, router: CachedAPI, workers: int, count_tokens) -> list[dict]:
    generation = hyde_jobs(data)
    done = {row["qa_key"]: row for row in jsonl(run_dir / "hyde_queries.jsonl")}
    pending = [job for job in generation if job["question"]["qa_key"] not in done]

    def work(job: dict) -> dict:
        text = router.chat(model=base.ANSWER_MODEL, messages=[{"role": "user", "content": job["prompt"]}], temperature=0,
                           max_tokens=HYDE_MAX_TOKENS, phase="hyde", extra_body=base.PROVIDER["answer"])
        if not text.strip():
            raise SystemExit(f"stop: empty HyDE generation for {job['question']['qa_key']}")
        return {"qa_key": job["question"]["qa_key"], "prompt_sha256": sha256_bytes(job["prompt"].encode()), "hypothetical": text}

    chrono.parallel(run_dir, pending, work, lambda row: append_jsonl(run_dir / "hyde_queries.jsonl", row),
                    workers=workers, label="C hyde")
    seal_jsonl(run_dir / "hyde_queries.jsonl", run_dir / "hyde_queries_meta.json", expected_rows=len(generation),
               identity_fields=("qa_key",), metadata={"model": base.ANSWER_MODEL, "prompt": HYDE_PROMPT, "max_tokens": HYDE_MAX_TOKENS})
    hypotheticals = {row["qa_key"]: row["hypothetical"] for row in jsonl(run_dir / "hyde_queries.jsonl")}

    load_dotenv(ROOT / ".env")
    embeddings = EmbeddingCache(run_dir, CachedAPI(run_dir, api_key=os.environ["OPENAI_API_KEY"]))
    questions = tp_questions(data)
    vectors = embeddings.all([hypotheticals[row["qa_key"]] for row in questions])
    jobs = []
    for topic in base.BATCHES:
        raw_matrix = control.raw_index_matrix(SOURCE / "embeddings", topic, data.raw_by_topic[topic])
        for row, vector in zip(questions, vectors):
            if row["topic"] != topic:
                continue
            order, retrieval_ms = control.rank_raw(raw_matrix, np.asarray(vector, dtype=np.float64), TOP_K)
            docs = [data.raw_by_topic[topic][int(i)] for i in order]
            context = control.render_context(docs)
            jobs.append({"question": row, "docs": docs, "context": context, "prompt": build_prompt(data.prompts, row, context),
                         "retrieval_ms": retrieval_ms, "context_tokens": count_tokens(context)})
    order = {row["qa_key"]: i for i, row in enumerate(questions)}
    jobs.sort(key=lambda j: order[j["question"]["qa_key"]])
    retrieval_path = run_dir / "hyde_retrieval.jsonl"
    if not retrieval_path.exists():
        for job in jobs:
            append_jsonl(retrieval_path, {"qa_key": job["question"]["qa_key"], "doc_ids": [d.doc_id for d in job["docs"]],
                                          "prompt_sha256": sha256_bytes(job["prompt"].encode())})
        seal_jsonl(retrieval_path, run_dir / "hyde_retrieval_meta.json", expected_rows=len(jobs), identity_fields=("qa_key",),
                   metadata={"embedding_model": EMBED_MODEL, "top_k": TOP_K})
    else:
        verify_sealed_jsonl(retrieval_path, run_dir / "hyde_retrieval_meta.json", expected_rows=len(jobs))
        sealed = {row["qa_key"]: row for row in jsonl(retrieval_path)}
        for job in jobs:
            if sealed[job["question"]["qa_key"]]["prompt_sha256"] != sha256_bytes(job["prompt"].encode()):
                raise RuntimeError(f"HyDE retrieval changed on resume: {job['question']['qa_key']}")
    return jobs


def sealed_hyde_generations() -> dict[str, str]:
    """Experiment C's generations, read only. C2 never generates its own passages."""
    source = RUN_DIRS["C"]
    verify_sealed_jsonl(source / "hyde_queries.jsonl", source / "hyde_queries_meta.json", expected_rows=300)
    return {row["qa_key"]: row["hypothetical"] for row in jsonl(source / "hyde_queries.jsonl")}


def c2_targets(data: Data) -> dict[str, int]:
    """Per-question token budget: the frozen RAW+EVENTS-token-matched context size."""
    return {row["qa_key"]: data.frozen[(row["qa_key"], unlinked.EVENTS_TOKEN)]["context_tokens"]
            for row in tp_questions(data)}


def c2_input_records(data: Data) -> list[dict]:
    """Frozen before the first call: which generation and which budget each question uses."""
    generations = sealed_hyde_generations()
    targets = c2_targets(data)
    return [{"qa_key": row["qa_key"], "condition": CONDITIONS["C2"],
             "hypothetical_sha256": sha256_bytes(generations[row["qa_key"]].encode()),
             "target_tokens": targets[row["qa_key"]]} for row in tp_questions(data)]


def c2_phase(run_dir: Path, data: Data, count_tokens) -> list[dict]:
    """Rank the sealed raw index with C's hypothetical passages, cut to the event budget, order by time."""
    generations = sealed_hyde_generations()
    targets = c2_targets(data)
    load_dotenv(ROOT / ".env")
    embeddings = EmbeddingCache(run_dir, CachedAPI(run_dir, api_key=os.environ["OPENAI_API_KEY"]))
    questions = tp_questions(data)
    vectors = dict(zip((row["qa_key"] for row in questions),
                       embeddings.all([generations[row["qa_key"]] for row in questions])))
    jobs, exhausted = [], 0
    for topic in base.BATCHES:
        raw_matrix = control.raw_index_matrix(SOURCE / "embeddings", topic, data.raw_by_topic[topic])
        for row in questions:
            if row["topic"] != topic:
                continue
            vector = np.asarray(vectors[row["qa_key"]], dtype=np.float64)
            order, retrieval_ms = control.rank_raw(raw_matrix, vector, control.RANK_DEPTH)
            ranked = [data.raw_by_topic[topic][int(i)] for i in order]
            docs = token_matched_chrono(ranked, targets[row["qa_key"]], data.raw_by_id, count_tokens)
            if len(docs) >= len(ranked):
                exhausted += 1
            context = control.render_context(docs)
            jobs.append({"question": row, "docs": docs, "context": context,
                         "prompt": build_prompt(data.prompts, row, context), "retrieval_ms": retrieval_ms,
                         "context_tokens": count_tokens(context), "target_tokens": targets[row["qa_key"]]})
    position = {row["qa_key"]: index for index, row in enumerate(questions)}
    jobs.sort(key=lambda job: position[job["question"]["qa_key"]])
    if exhausted:
        log(run_dir, f"rank depth {control.RANK_DEPTH} exhausted for {exhausted} questions")

    retrieval_path = run_dir / "c2_retrieval.jsonl"
    records = [{"qa_key": job["question"]["qa_key"], "doc_ids": [d.doc_id for d in job["docs"]],
                "context_tokens": job["context_tokens"], "target_tokens": job["target_tokens"],
                "prompt_sha256": sha256_bytes(job["prompt"].encode())} for job in jobs]
    if not retrieval_path.exists():
        for record in records:
            append_jsonl(retrieval_path, record)
        seal_jsonl(retrieval_path, run_dir / "c2_retrieval_meta.json", expected_rows=len(jobs),
                   identity_fields=("qa_key",),
                   metadata={"embedding_model": EMBED_MODEL, "rank_depth": control.RANK_DEPTH,
                             "rank_depth_exhausted": exhausted,
                             "token_target": f"frozen {unlinked.EVENTS_TOKEN} context_tokens"})
    else:
        verify_sealed_jsonl(retrieval_path, run_dir / "c2_retrieval_meta.json", expected_rows=len(jobs))
        sealed = {row["qa_key"]: row for row in jsonl(retrieval_path)}
        for record in records:
            if sealed[record["qa_key"]]["prompt_sha256"] != record["prompt_sha256"]:
                raise RuntimeError(f"C2 retrieval changed on resume: {record['qa_key']}")
    return jobs


def preflight_c2() -> dict:
    """Offline ceiling for the amendment: contexts are matched to the frozen event budget."""
    import tiktoken

    data, load_checks = load_data()
    encoder = tiktoken.get_encoding(control.TOKEN_ENCODING)
    count_tokens = lambda text: len(encoder.encode(text))
    calibration = json.loads((PREFLIGHT_DIR / "preflight_evermembench.json").read_text())["calibration"]
    overhead = calibration["api_input_overhead_tokens"]
    output_tokens = calibration["mean_output_tokens_by_type"]["open_ended"]
    judge_input, judge_output = calibration["judge_mean_tokens_tp"]

    questions = tp_questions(data)
    targets = c2_targets(data)
    # The prompt frame (template plus question) is exact; the context is matched to the frozen budget.
    input_tokens = sum(count_tokens(build_prompt(data.prompts, row, "")) + targets[row["qa_key"]] + overhead
                       for row in questions)
    calls = len(questions)
    answers = {
        "calls": calls, "input_tokens": round(input_tokens), "expected_output_tokens": round(calls * output_tokens),
        "expected_usd": list_price(base.ANSWER_MODEL, input_tokens, calls * output_tokens),
        "ceiling_usd": list_price(base.ANSWER_MODEL, input_tokens * 1.2, calls * ANSWER_MAX_TOKENS) * FEE_FACTOR,
    }
    judges = judge_cost(calls, judge_input, judge_output)
    embeddings = {
        "calls": -(-calls // 128),
        "expected_usd": list_price(EMBED_MODEL, calls * 90, 0),
        "ceiling_usd": list_price(EMBED_MODEL, calls * HYDE_MAX_TOKENS, 0) * FEE_FACTOR,
    }
    report = {
        "experiment": "C2", "condition": CONDITIONS["C2"],
        "amendment": AMENDMENT.name, "amendment_sha256": sha256_file(AMENDMENT),
        "runner_sha256": sha256_file(RUNNER), "load_checks": load_checks,
        "questions": calls, "new_generation_calls": 0,
        "new_embedding_calls": embeddings["calls"], "new_answer_calls": calls, "new_judge_calls_ceiling": calls,
        "mean_target_tokens": round(sum(targets.values()) / calls, 1),
        "answers": answers, "judge": judges, "embeddings": embeddings,
        "expected_usd": answers["expected_usd"] + judges["expected_usd"] + embeddings["expected_usd"],
        "ceiling_usd": answers["ceiling_usd"] + judges["ceiling_usd"] + embeddings["ceiling_usd"],
    }
    AMENDMENT_DIR.mkdir(parents=True, exist_ok=True)
    (AMENDMENT_DIR / "preflight_c2.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    return report


def run(name: str, workers: int, judge_workers: int) -> None:
    import tiktoken

    encoder = tiktoken.get_encoding(control.TOKEN_ENCODING)
    count_tokens = lambda text: len(encoder.encode(text))
    if name == "P1":
        ceiling = 2.50  # protocol: research/SIMPLE_PROP_BASELINE_PROTOCOL.md, amendment P1
    elif name in AMENDED:
        ceiling = json.loads((AMENDMENT_DIR / "preflight_c2.json").read_text())["ceiling_usd"]
    else:
        preflight_report = json.loads((PREFLIGHT_DIR / "preflight_evermembench.json").read_text())
        ceiling = preflight_report["experiments"][name]["ceiling_usd"]
    run_dir = RUN_DIRS[name]
    data, checks = load_data()
    if name == "C":
        gen_records = [{"qa_key": j["question"]["qa_key"], "prompt_sha256": sha256_bytes(j["prompt"].encode())} for j in hyde_jobs(data)]
        inputs_sha = chrono.write_or_verify_inputs(run_dir, gen_records)
    elif name == "C2":
        inputs_sha = chrono.write_or_verify_inputs(run_dir, c2_input_records(data))
    else:
        jobs, job_checks = {"A": jobs_a, "B1": jobs_b1, "B2": jobs_b2, "P1": jobs_p1}[name](data, count_tokens)
        checks.update(job_checks)
        inputs_sha = chrono.write_or_verify_inputs(run_dir, input_records(name, jobs))
    chrono.ensure_run_manifest(run_dir, manifest(name, inputs_sha, checks, ceiling))
    router = make_router(run_dir, data)
    if not any(r["phase"] == "before" for r in jsonl(run_dir / "billing_snapshots.jsonl")):
        chrono.billing_snapshot(run_dir, "before")
    chrono.check_routes(router, run_dir)
    if name == "C":
        jobs = hyde_phase(run_dir, data, router, workers, count_tokens)
    elif name == "C2":
        jobs = c2_phase(run_dir, data, count_tokens)
    answer_phase(name, run_dir, jobs, router, data, workers, ceiling)
    judge_phase(name, run_dir, len(jobs), router, judge_workers, ceiling)
    chrono.billing_snapshot(run_dir, "after")
    render_report(name)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

REFERENCES = {
    "A": ((SOURCE, ("RAW", "RAW+EPISODES")),),
    "B1": ((UNLINKED, ("RAW+EVENTS-token-matched",)), (CHRONO, ("RAW-token-matched-chrono", "RAW+EVENTS-token-matched-chrono")),
           (SOURCE, ("RAW+EPISODES",))),
    "B2": ((SOURCE, ("RAW", "RAW+EPISODES")), (CHRONO, ("RAW-token-matched-chrono", "RAW+EVENTS-token-matched-chrono"))),
    "C": ((SOURCE, ("RAW",)), (CHRONO, ("RAW-token-matched-chrono", "RAW+EVENTS-token-matched-chrono"))),
    "C2": ((SOURCE, ("RAW",)), (CHRONO, ("RAW-token-matched-chrono", "RAW+EVENTS-token-matched-chrono")),
           (RUN_DIRS["C"], ("HyDE-RAW",))),
    "P1": ((SOURCE, ("RAW",)), (CHRONO, ("RAW-token-matched-chrono", "RAW+EVENTS-token-matched-chrono")),
           (RUN_DIRS["B1"], ("RAW+EVENTS-token-matched-chrono-nodesc",))),
}
COMPARISONS = {
    "A": (("RAW+EVENTS", "RAW"), ("RAW+EVENTS", "RAW+EPISODES")),
    "B1": (("RAW+EVENTS-token-matched-chrono-nodesc", "RAW+EVENTS-token-matched-chrono"),
           ("RAW+EVENTS-token-matched-chrono-nodesc", "RAW-token-matched-chrono"),
           ("RAW+EVENTS-token-matched-chrono-nodesc", "RAW+EVENTS-token-matched")),
    "B2": (("RAW+EPISODES-chrono", "RAW+EPISODES"), ("RAW+EPISODES-chrono", "RAW+EVENTS-token-matched-chrono"),
           ("RAW+EPISODES-chrono", "RAW-token-matched-chrono")),
    "C": (("HyDE-RAW", "RAW"), ("RAW+EVENTS-token-matched-chrono", "HyDE-RAW"), ("RAW+EVENTS-token-matched-chrono", "RAW")),
    "C2": (("RAW+EVENTS-token-matched-chrono", "HyDE-RAW-token-matched-chrono"),
           ("HyDE-RAW-token-matched-chrono", "RAW-token-matched-chrono"),
           ("HyDE-RAW-token-matched-chrono", "HyDE-RAW"),
           ("HyDE-RAW-token-matched-chrono", "RAW")),
    "P1": (("RAW+SIMPLE-PROP-token-matched-chrono", "RAW+EVENTS-token-matched-chrono"),
           ("RAW+SIMPLE-PROP-token-matched-chrono", "RAW-token-matched-chrono"),
           ("RAW+SIMPLE-PROP-token-matched-chrono", "RAW+EVENTS-token-matched-chrono-nodesc")),
}


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def render_report(name: str) -> Path:
    run_dir = RUN_DIRS[name]
    new_results, new_predictions = jsonl(run_dir / "results.jsonl"), jsonl(run_dir / "predictions.jsonl")
    keys = {row["qa_key"] for row in new_results}
    results, predictions = list(new_results), list(new_predictions)
    for run, conditions in REFERENCES[name]:
        results += [r for r in jsonl(run / "results.jsonl") if r["qa_key"] in keys and r["condition"] in conditions]
        predictions += [p for p in jsonl(run / "predictions.jsonl") if p["qa_key"] in keys and p["condition"] in conditions]
    conditions = [CONDITIONS[name]] + [c for _run, cs in REFERENCES[name] for c in cs]

    def summary(condition: str, subset=lambda r: True) -> dict:
        res = [r for r in results if r["condition"] == condition and subset(r)]
        pre = [p for p in predictions if p["condition"] == condition and subset(p)]
        tokens = [p["context_tokens"] for p in pre if "context_tokens" in p]
        return {"n": len(res), "accuracy": _mean(r["correct"] for r in res),
                "evidence_recall": _mean(r["evidence_recall"] for r in res), "evidence_precision": _mean(r["evidence_precision"] for r in res),
                "mean_unique_sources": _mean(len(set(p["exposed_source_ids"])) for p in pre),
                "mean_context_tokens": _mean(tokens) if tokens else None,
                "mean_answer_ms": _mean(p["answer_wall_ms"] for p in pre if "answer_wall_ms" in p),
                "mean_retrieval_ms": _mean(p["retrieval_ms"] for p in pre if "retrieval_ms" in p)}

    out: dict = {"experiment": name, "questions": len(keys), "conditions": {c: summary(c) for c in conditions},
                 "comparisons": {f"{l} - {r}": chrono.compare(results, l, r) for l, r in COMPARISONS[name]},
                 "api_usage": usage_summary(run_dir), "billing": chrono.billing(run_dir),
                 "reused_answers": sum(1 for p in new_predictions if p.get("answer_cache", "new") != "new")}
    if name == "A":
        minors = sorted({r["minor"] for r in new_results})
        out["by_category"] = {m: {"conditions": {c: summary(c, lambda r, m=m: r["minor"] == m) for c in conditions},
                                  "comparison": chrono.compare([r for r in results if r["minor"] == m], "RAW+EVENTS", "RAW")}
                              for m in minors}
        out["non_tp"] = chrono.compare([r for r in results if r["minor"] != "TP"], "RAW+EVENTS", "RAW")
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=float) + "\n")

    pct = lambda v: "--" if v is None or v != v else f"{100 * v:.1f}"
    lines = [f"# Final sprint experiment {name}: {CONDITIONS[name]}", "",
             f"- protocol sha256 `{sha256_file(PROTOCOL)}`; runner sha256 `{sha256_file(RUNNER)}`",
             f"- questions: {len(keys)}; answers reused from sealed caches: {out['reused_answers']}",
             "- Temporal Duration is a post-hoc discovery slice; every TP result here is exploratory.", "",
             "## Conditions", "", "| condition | n | accuracy % | evidence recall % | evidence precision % | unique sources | context tokens | answer ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c, s in out["conditions"].items():
        tokens = "--" if s["mean_context_tokens"] is None else f"{s['mean_context_tokens']:.0f}"
        lines.append(f"| {c} | {s['n']} | {pct(s['accuracy'])} | {pct(s['evidence_recall'])} | {pct(s['evidence_precision'])} | "
                     f"{s['mean_unique_sources']:.1f} | {tokens} | {s['mean_answer_ms']:.0f} |")
    lines += ["", "## Paired comparisons (pp)", "",
              "| difference | delta | question 95% CI | McNemar p | left-only / right-only / both / neither | project 95% CI | per-project | sign-flip p |",
              "|---|---:|---|---:|---|---|---|---:|"]

    def comparison_line(label: str, c: dict) -> str:
        per = ", ".join(f"{k}: {100 * v['delta']:+.1f}" for k, v in c["projects"].items())
        return (f"| {label} | {100 * c['delta']:+.1f} | [{100 * c['question_ci95'][0]:+.1f}, {100 * c['question_ci95'][1]:+.1f}] | "
                f"{c['mcnemar_p']:.4f} | {c['left_only']} / {c['right_only']} / {c['both_correct']} / {c['both_wrong']} | "
                f"[{100 * c['project_ci95'][0]:+.1f}, {100 * c['project_ci95'][1]:+.1f}] | {per} | {c['sign_flip_p']:.4f} |")

    for label, c in out["comparisons"].items():
        lines.append(comparison_line(label, c))
    if name == "A":
        lines += ["", "## By official category (RAW+EVENTS vs RAW)", "",
                  "| category | n | RAW % | RAW+EPISODES % | RAW+EVENTS % | delta vs RAW | 95% CI | McNemar p |", "|---|---:|---:|---:|---:|---:|---|---:|"]
        for m, block in out["by_category"].items():
            cs, cmp_ = block["conditions"], block["comparison"]
            tag = " (post-hoc slice)" if m == "TP" else ""
            lines.append(f"| {m}{tag} | {cs['RAW+EVENTS']['n']} | {pct(cs['RAW']['accuracy'])} | {pct(cs['RAW+EPISODES']['accuracy'])} | "
                         f"{pct(cs['RAW+EVENTS']['accuracy'])} | {100 * cmp_['delta']:+.1f} | "
                         f"[{100 * cmp_['question_ci95'][0]:+.1f}, {100 * cmp_['question_ci95'][1]:+.1f}] | {cmp_['mcnemar_p']:.4f} |")
        lines += ["", comparison_line("RAW+EVENTS - RAW, all except TP", out["non_tp"])]
    usage, bill = out["api_usage"], out["billing"]
    lines += ["", "## Cost", "", f"- API calls: {usage['api_calls']}; list-price estimate USD {usage['estimated_usd_upper_bound']:.4f}",
              f"- by model: `{json.dumps(usage['by_model'], sort_keys=True)}`",
              f"- OpenRouter key usage delta: {bill['delta_usd']} ({bill['note']})"]
    report = run_dir / "report.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--judge-workers", type=int, default=2)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--preflight-c2", action="store_true")
    modes.add_argument("--run", choices=sorted(RUN_DIRS))
    modes.add_argument("--report", choices=sorted(RUN_DIRS))
    modes.add_argument("--freeze", choices=sorted(RUN_DIRS))
    args = parser.parse_args()
    try:
        if args.preflight:
            print(json.dumps(preflight()["experiments"], indent=2, sort_keys=True, default=float))
        elif args.preflight_c2:
            print(json.dumps(preflight_c2(), indent=2, sort_keys=True, default=float))
        elif args.run:
            run(args.run, args.workers, args.judge_workers)
        elif args.report:
            print(render_report(args.report))
        else:
            sources = [PROTOCOL, RUNNER] + [Path(m.__file__) for m in (chrono, base, runner, control, unlinked, common)]
            print(freeze_run(RUN_DIRS[args.freeze], sources))
    except RateLimitError as error:
        raise SystemExit(f"rate limited; completed work is cached: {error}") from None
    except AuthenticationError as error:
        raise SystemExit(f"API authentication failed: {error}") from None


if __name__ == "__main__":
    main()
