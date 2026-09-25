"""EverMemBench Temporal control v2: chronological order of the same retrieved context.

Copies the document sets of the frozen RAW-token-matched and RAW+EVENTS-token-matched
conditions and presents them in time order. The primary control is a pure permutation of
byte-identical documents; a day-offset rendering is a separate exploratory condition.
Compared with sealed RAW+EPISODES on the 300 frozen Temporal Duration questions.
Protocol: research/EVERMEMBENCH_CHRONOLOGICAL_ORDER_CONTROL_PROTOCOL_V2.md (supersedes v1,
which was never executed; see .research_runs/amendments/evermembench_chronological_control_v2_20260915).

Run:
    python3 -m research.evermembench_chronological_order_control --verify   # offline, no API
    python3 -m research.evermembench_chronological_order_control --smoke    # 5 questions
    python3 -m research.evermembench_chronological_order_control --all      # 300 TP questions
    python3 -m research.evermembench_chronological_order_control --report   # rebuild report
    python3 -m research.evermembench_chronological_order_control --freeze   # after --all
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import httpx
from openai import AuthenticationError, RateLimitError

from research import evermembench_episode_run as base
from research import evermembench_parallel_resume as runner
from research import evermembench_temporal_budget_control as control
from research import evermembench_unlinked_events_control as unlinked
from research import paper_benchmark_common as common
from research.paper_benchmark_common import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    CachedAPI,
    SearchDoc,
    append_jsonl,
    clustered_bootstrap,
    exact_mcnemar,
    freeze_run,
    jsonl,
    log,
    paired_bootstrap,
    seal_jsonl,
    sha256_bytes,
    sha256_file,
    usage_summary,
    verify_sealed_jsonl,
)


FROZEN = base.ROOT / ".research_runs" / "frozen"
SOURCE_FROZEN = FROZEN / "evermembench_temporal_episodes_official_v1_20260915_021550_MSK"
BUDGET_FROZEN = FROZEN / "evermembench_temporal_budget_control_v1_20260915_021134_MSK"
UNLINKED_FROZEN = FROZEN / "evermembench_unlinked_events_control_v1_20260915_022532_MSK"
FROZEN_DIRS = (SOURCE_FROZEN, BUDGET_FROZEN, UNLINKED_FROZEN)
FROZEN_ROWS = {BUDGET_FROZEN: 900, UNLINKED_FROZEN: 600}
LIVE_SOURCE_FILES = ("predictions.jsonl", "predictions_meta.json", "results.jsonl", "results_meta.json",
                     "episodes.jsonl", "episodes_meta.json")
RUN_DIR = base.ROOT / ".research_runs" / "evermembench_chronological_order_control_v2"
SMOKE_DIR = base.ROOT / ".research_runs" / "evermembench_chronological_order_control_smoke_v2"
AMENDMENT_DIR = base.ROOT / ".research_runs" / "amendments" / "evermembench_chronological_control_v2_20260915"
PROTOCOL = base.ROOT / "research" / "EVERMEMBENCH_CHRONOLOGICAL_ORDER_CONTROL_PROTOCOL_V2.md"
SUPERSEDED_PROTOCOL = base.ROOT / "research" / "EVERMEMBENCH_CHRONOLOGICAL_ORDER_CONTROL_PROTOCOL.md"
RUNNER = Path(__file__).resolve()
TESTS = base.ROOT / "tests" / "test_evermembench_chronological_control.py"

RAW_CHRONO = "RAW-token-matched-chrono"
EVENTS_CHRONO = "RAW+EVENTS-token-matched-chrono"
RAW_RELATIVE = "RAW-token-matched-chrono-relative"
NEW_CONDITIONS = (RAW_CHRONO, EVENTS_CHRONO, RAW_RELATIVE)
PURE_PERMUTATIONS = (RAW_CHRONO, EVENTS_CHRONO)
COPIED_FROM = {RAW_CHRONO: control.TOKEN_MATCHED, EVENTS_CHRONO: unlinked.EVENTS_TOKEN,
               RAW_RELATIVE: control.TOKEN_MATCHED}
EPISODES = "RAW+EPISODES"
REFERENCE_CONDITIONS = ("RAW", EPISODES, control.TOKEN_MATCHED, unlinked.EVENTS_TOKEN)
REPORT_ORDER = ("RAW", control.TOKEN_MATCHED, RAW_CHRONO, RAW_RELATIVE, unlinked.EVENTS_TOKEN, EVENTS_CHRONO, EPISODES)
ANSWER_MAX_TOKENS = 1000
EXPECTED_QUESTIONS = 300
QUESTIONS_PER_PROJECT = 60
ACCURACY_BANDS = (0.147, 0.185)
ANSWER_MARKERS = ("[EMPTY]", "[INVALID]", "[FAILED]")
STAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2})(?: (\d{2}:\d{2}:\d{2}))?\]")


# --------------------------------------------------------------------------
# Ordering and rendering (pure, unit-tested)
# --------------------------------------------------------------------------

def raw_timestamp(doc: SearchDoc) -> str:
    match = STAMP.match(doc.rendered)
    if not match:
        raise ValueError(f"raw document without a timestamp prefix: {doc.doc_id}")
    return f"{match.group(1)} {match.group(2) or '00:00:00'}"


def doc_timestamp(doc: SearchDoc, raw_by_id: dict[str, SearchDoc]) -> str:
    """A raw message's own time; for a derived document the earliest source message time."""
    if doc.kind == "raw":
        return raw_timestamp(doc)
    stamps = [raw_timestamp(raw_by_id[source]) for source in doc.source_ids if source in raw_by_id]
    if not stamps:
        raise ValueError(f"derived document without timestamped sources: {doc.doc_id}")
    return min(stamps)


def chronological(docs: list[SearchDoc], raw_by_id: dict[str, SearchDoc]) -> list[tuple[str, SearchDoc]]:
    """Ascending by timestamp, ties by document id; refuses non-unique sort keys."""
    items = sorted(((doc_timestamp(doc, raw_by_id), doc) for doc in docs), key=lambda item: (item[0], item[1].doc_id))
    keys = [(stamp, doc.doc_id) for stamp, doc in items]
    if len(set(keys)) != len(keys):
        raise ValueError("non-unique chronological sort key (duplicate document)")
    return items


def render_relative(items: list[tuple[str, SearchDoc]]) -> str:
    """Exploratory rendering: the same lines prefixed with day offsets from the earliest document."""
    if not items:
        return "(No memories retrieved)"
    first = date.fromisoformat(items[0][0][:10])
    return "\n".join(
        f"- [Day +{(date.fromisoformat(stamp[:10]) - first).days}] {doc.rendered}" for stamp, doc in items
    )


def blocks(docs: list[SearchDoc]) -> list[str]:
    return [f"- {doc.rendered}" for doc in docs]


def verify_pure_permutation(source_docs: list[SearchDoc], ordered_docs: list[SearchDoc],
                            source_context: str, ordered_context: str) -> None:
    """The ordered context must be exactly a permutation of the byte-identical frozen blocks."""
    source_ids = [doc.doc_id for doc in source_docs]
    ordered_ids = [doc.doc_id for doc in ordered_docs]
    if len(set(source_ids)) != len(source_ids) or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("duplicate document id")
    if Counter(source_ids) != Counter(ordered_ids):
        raise ValueError("document ids differ from the frozen condition")
    frozen = {doc.doc_id: doc for doc in source_docs}
    for doc in ordered_docs:
        original = frozen[doc.doc_id]
        if doc.rendered.encode() != original.rendered.encode() or doc.source_ids != original.source_ids:
            raise ValueError(f"document text or sources changed: {doc.doc_id}")
    if source_context != control.render_context(source_docs):
        raise ValueError("source context is not the frozen rendering")
    if ordered_context != control.render_context(ordered_docs):
        raise ValueError("ordered context is not the frozen rendering of the ordered documents")
    if sorted(blocks(source_docs)) != sorted(blocks(ordered_docs)):
        raise ValueError("ordered context is not a permutation of the frozen blocks")
    if Counter(control.unique_source_ids(source_docs)) != Counter(control.unique_source_ids(ordered_docs)):
        raise ValueError("exposed raw sources differ from the frozen condition")


def prompt_frame(prompt: str, context: str) -> tuple[str, str]:
    """Text of a prompt before and after its context."""
    index = prompt.index(context)
    return prompt[:index], prompt[index + len(context):]


def sign_flip_p(values: list[float]) -> float:
    """Exact two-sided sign-flip test over cluster means."""
    observed = abs(sum(values) / len(values))
    hits = sum(abs(sum(s * v for s, v in zip(signs, values)) / len(values)) >= observed - 1e-12
               for signs in itertools.product((1, -1), repeat=len(values)))
    return hits / 2 ** len(values)


def pending_jobs(jobs: list[dict], done: set[tuple[str, str]]) -> list[dict]:
    return [job for job in jobs if (job["question"]["qa_key"], job["condition"]) not in done]


def check_resumed_predictions(rows: list[dict], jobs: list[dict]) -> None:
    """Cached predictions must belong to the frozen inputs of this run."""
    expected = {(job["question"]["qa_key"], job["condition"]): sha256_bytes(job["prompt"].encode()) for job in jobs}
    for row in rows:
        key = (row["qa_key"], row["condition"])
        if key not in expected or row["prompt_sha256"] != expected[key]:
            raise RuntimeError(f"cached prediction does not match the frozen inputs: {key}")


def write_or_verify_inputs(run_dir: Path, records: list[dict]) -> str:
    path = run_dir / "inputs.jsonl"
    text = "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records)
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"frozen inputs differ from the reconstructed inputs: {path}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return sha256_file(path)


def ensure_run_manifest(run_dir: Path, manifest: dict) -> None:
    path = run_dir / "run_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError(f"run manifest mismatch (protocol, runner, inputs or scope changed): {path}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


# --------------------------------------------------------------------------
# Frozen inputs and job preparation (offline, no gold)
# --------------------------------------------------------------------------

def verify_frozen_inputs() -> dict:
    counts = {}
    for directory in FROZEN_DIRS:
        lines = (directory / "SHA256SUMS").read_text().splitlines()
        for line in lines:
            digest, name = line.split("  ", 1)
            if sha256_file(directory / name) != digest:
                raise RuntimeError(f"frozen checksum mismatch: {directory.name}/{name}")
        counts[directory.name] = len(lines)
    for name in LIVE_SOURCE_FILES:
        if sha256_file(base.RUN_DIR / name) != sha256_file(SOURCE_FROZEN / "run" / name):
            raise RuntimeError(f"live source file differs from its frozen copy: {name}")
    control.verify_sources()
    for directory, rows in FROZEN_ROWS.items():
        run = directory / "run"
        verify_sealed_jsonl(run / "predictions.jsonl", run / "predictions_meta.json", expected_rows=rows)
        verify_sealed_jsonl(run / "results.jsonl", run / "results_meta.json", expected_rows=rows)
    return counts


def frozen_scope() -> list[str]:
    return json.loads((BUDGET_FROZEN / "run" / "run_manifest.json").read_text())["scope_qa_keys"]


def prepare_jobs(questions: list[dict], *, full: bool) -> tuple[list[dict], dict]:
    import tiktoken

    checks: dict = {"frozen_sha256sums_verified": verify_frozen_inputs()}
    scope = frozen_scope()
    keys = [row["qa_key"] for row in questions]
    if full:
        per_project = Counter(row["topic"] for row in questions)
        if len(keys) != EXPECTED_QUESTIONS or sorted(keys) != sorted(scope) or \
                set(per_project.values()) != {QUESTIONS_PER_PROJECT}:
            raise RuntimeError(f"scope is not the frozen 300 questions: {len(keys)} {dict(per_project)}")
    elif not set(keys) <= set(scope):
        raise RuntimeError("smoke questions are outside the frozen scope")

    encoder = tiktoken.get_encoding(control.TOKEN_ENCODING)
    count_tokens = lambda text: len(encoder.encode(text))
    _frame, raw_by_topic, raw_by_id = base.load_messages()
    events = unlinked.events_by_topic(SOURCE_FROZEN / "run" / "episodes.jsonl", raw_by_id)
    docs_by_id = {doc.doc_id: doc for docs in raw_by_topic.values() for doc in docs}
    docs_by_id.update({doc.doc_id: doc for docs in events.values() for doc in docs})
    frozen = {(row["qa_key"], row["condition"]): row
              for row in jsonl(BUDGET_FROZEN / "run" / "predictions.jsonl") + jsonl(UNLINKED_FROZEN / "run" / "predictions.jsonl")}
    prompts = base._prompts()["answer"]

    checks.update({"questions": len(questions), "projects": dict(sorted(Counter(r["topic"] for r in questions).items())),
                   "frozen_prompt_reproduced": 0, "pure_permutation_verified": 0, "prompt_frame_unchanged": 0,
                   "timestamps_parsed": 0, "order_changed": {c: 0 for c in NEW_CONDITIONS},
                   "token_count_changed_by_permutation": {c: 0 for c in PURE_PERMUTATIONS}})
    jobs: list[dict] = []
    for row in questions:
        if row["question_type"] != "open_ended":
            raise RuntimeError(f"unexpected question type: {row['qa_key']}")
        build = lambda context: prompts["open_ended"].format(context=context, question=row["question"])
        chrono_ids: list[str] | None = None
        for condition in NEW_CONDITIONS:
            source = frozen[(row["qa_key"], COPIED_FROM[condition])]
            source_docs = [docs_by_id[doc_id] for doc_id in source["retrieved_ids"]]
            source_context = control.render_context(source_docs)
            source_prompt = build(source_context)
            if sha256_bytes(source_prompt.encode()) != source["prompt_sha256"]:
                raise RuntimeError(f"frozen {COPIED_FROM[condition]} prompt is not reproduced: {row['qa_key']}")
            if Counter(control.unique_source_ids(source_docs)) != Counter(source["exposed_source_ids"]):
                raise RuntimeError(f"frozen {COPIED_FROM[condition]} sources are not reproduced: {row['qa_key']}")
            checks["frozen_prompt_reproduced"] += 1
            items = chronological(source_docs, raw_by_id)
            checks["timestamps_parsed"] += len(items)
            ordered = [doc for _stamp, doc in items]
            if condition in PURE_PERMUTATIONS:
                context = control.render_context(ordered)
                verify_pure_permutation(source_docs, ordered, source_context, context)
                checks["pure_permutation_verified"] += 1
                if condition == RAW_CHRONO:
                    chrono_ids = [doc.doc_id for doc in ordered]
            else:
                context = render_relative(items)
                if [doc.doc_id for doc in ordered] != chrono_ids:
                    raise RuntimeError(f"relative condition order differs from {RAW_CHRONO}: {row['qa_key']}")
            prompt = build(context)
            if prompt_frame(prompt, context) != prompt_frame(source_prompt, source_context):
                raise RuntimeError(f"prompt changed outside the context: {row['qa_key']} {condition}")
            checks["prompt_frame_unchanged"] += 1
            changed = [doc.doc_id for doc in ordered] != source["retrieved_ids"]
            checks["order_changed"][condition] += int(changed)
            tokens = count_tokens(context)
            if condition in PURE_PERMUTATIONS and tokens != source["context_tokens"]:
                checks["token_count_changed_by_permutation"][condition] += 1
            jobs.append({
                "question": row, "condition": condition, "source_condition": COPIED_FROM[condition],
                "docs": ordered, "timestamps": [stamp for stamp, _doc in items], "context": context,
                "prompt": prompt, "context_tokens": tokens, "source_context_tokens": source["context_tokens"],
                "source_prompt_sha256": source["prompt_sha256"], "order_changed": changed,
            })
    return jobs, checks


def input_record(job: dict) -> dict:
    row = job["question"]
    return {
        "qa_key": row["qa_key"], "topic": row["topic"], "condition": job["condition"],
        "source_condition": job["source_condition"], "ordered_doc_ids": [doc.doc_id for doc in job["docs"]],
        "timestamps": job["timestamps"], "order_changed": job["order_changed"],
        "source_prompt_sha256": job["source_prompt_sha256"],
        "context_sha256": sha256_bytes(job["context"].encode()), "prompt_sha256": sha256_bytes(job["prompt"].encode()),
        "context_tokens": job["context_tokens"], "source_context_tokens": job["source_context_tokens"],
    }


def build_manifest(questions: list[dict], inputs_sha256: str, checks: dict) -> dict:
    return {
        "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
        "superseded_protocol": SUPERSEDED_PROTOCOL.name, "superseded_protocol_sha256": sha256_file(SUPERSEDED_PROTOCOL),
        "runner": RUNNER.name, "runner_sha256": sha256_file(RUNNER), "tests_sha256": sha256_file(TESTS),
        "frozen_sha256sums": {d.name: sha256_file(d / "SHA256SUMS") for d in FROZEN_DIRS},
        "frozen_checks": checks, "inputs_sha256": inputs_sha256,
        "conditions": list(NEW_CONDITIONS), "copied_from": COPIED_FROM, "primary": [EPISODES, RAW_CHRONO],
        "scope_qa_keys": sorted(row["qa_key"] for row in questions),
        "answer_model": base.ANSWER_MODEL, "answer_max_tokens": ANSWER_MAX_TOKENS, "answer_provider": base.PROVIDER["answer"],
        "judge_model": base.JUDGE_MODEL, "judge_call": runner.JUDGE_CALL_VERSION, "token_encoding": control.TOKEN_ENCODING,
        "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED},
    }


# --------------------------------------------------------------------------
# API phases (logs only to the control's own directory)
# --------------------------------------------------------------------------

def parallel(run_dir: Path, jobs, work, write, *, workers: int, label: str) -> None:
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        try:
            for future in as_completed(futures):
                write(future.result())
                completed += 1
                if completed % 25 == 0 or completed == len(jobs):
                    log(run_dir, f"{label}: {completed}/{len(jobs)} (workers={workers})")
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def check_routes(router: CachedAPI, run_dir: Path) -> None:
    for model, provider in ((base.ANSWER_MODEL, base.PROVIDER["answer"]), (base.JUDGE_MODEL, base.PROVIDER["judge"])):
        result = router.chat(model=model, messages=[{"role": "user", "content": "Reply with OK only."}],
                             temperature=0, max_tokens=8, phase="preflight", extra_body=provider)
        if not result.strip():
            raise RuntimeError(f"empty preflight response from {model}")
        log(run_dir, f"OpenRouter route OK: {model}")


def billing_snapshot(run_dir: Path, phase: str) -> None:
    """OpenRouter key usage in USD; the difference includes any concurrent use of the same key."""
    base_url = os.getenv("LLM_BASE_URL", base.OPENROUTER_DEFAULT).rstrip("/")
    record: dict = {"phase": phase, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "endpoint": f"{base_url}/key"}
    try:
        response = httpx.get(f"{base_url}/key", headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY', '')}"},
                             timeout=30)
        response.raise_for_status()
        record["usage_usd"] = response.json().get("data", {}).get("usage")
    except Exception as error:  # billing is reported, never required
        record["error"] = f"{type(error).__name__}: {error}"
    append_jsonl(run_dir / "billing_snapshots.jsonl", record)


def run_inference(run_dir: Path, questions: list[dict], jobs: list[dict], router: CachedAPI, workers: int) -> None:
    existing = jsonl(run_dir / "predictions.jsonl")
    check_resumed_predictions(existing, jobs)
    pending = pending_jobs(jobs, {(row["qa_key"], row["condition"]) for row in existing})
    log(run_dir, f"inference: {len(pending)} pending / {len(existing)} cached")

    def work(job: dict) -> dict:
        row, docs = job["question"], job["docs"]
        started = time.perf_counter()
        raw_answer = router.chat(
            model=base.ANSWER_MODEL, messages=[{"role": "user", "content": job["prompt"]}], temperature=0,
            max_tokens=ANSWER_MAX_TOKENS, phase="answer", extra_body=base.PROVIDER["answer"],
        )
        sources = control.unique_source_ids(docs)
        return {
            **{field: row[field] for field in ("qa_key", "topic", "id", "major", "minor", "question", "question_type")},
            "condition": job["condition"], "source_condition": job["source_condition"],
            "answer": runner.official_answer(raw_answer, row["question_type"]), "answer_raw": raw_answer,
            "retrieved_ids": [doc.doc_id for doc in docs], "exposed_source_ids": sources,
            "unique_source_count": len(sources), "order_changed": job["order_changed"],
            "context_tokens": job["context_tokens"], "source_context_tokens": job["source_context_tokens"],
            "context_sha256": sha256_bytes(job["context"].encode()), "prompt_sha256": sha256_bytes(job["prompt"].encode()),
            "answer_wall_ms": (time.perf_counter() - started) * 1000,
        }

    parallel(run_dir, pending, work, lambda row: append_jsonl(run_dir / "predictions.jsonl", row),
             workers=workers, label="inference")
    rows = jsonl(run_dir / "predictions.jsonl")
    seal_jsonl(
        run_dir / "predictions.jsonl", run_dir / "predictions_meta.json",
        expected_rows=len(questions) * len(NEW_CONDITIONS), identity_fields=("qa_key", "condition"),
        metadata={
            "benchmark": "EverMemBench-Dynamic", "protocol_sha256": sha256_file(PROTOCOL),
            "runner_sha256": sha256_file(RUNNER), "inputs_sha256": sha256_file(run_dir / "inputs.jsonl"),
            "conditions": list(NEW_CONDITIONS), "answer_model": base.ANSWER_MODEL, "token_encoding": control.TOKEN_ENCODING,
            "answer_failure_markers": {c: {m: sum(1 for r in rows if r["condition"] == c and r["answer"] == m)
                                           for m in ANSWER_MARKERS} for c in NEW_CONDITIONS},
        },
    )


def score(run_dir: Path, questions: list[dict], router: CachedAPI, workers: int) -> None:
    expected = len(questions) * len(NEW_CONDITIONS)
    verify_sealed_jsonl(run_dir / "predictions.jsonl", run_dir / "predictions_meta.json", expected_rows=expected)
    predictions = {(row["qa_key"], row["condition"]): row for row in jsonl(run_dir / "predictions.jsonl")}
    gold = {row["qa_key"]: row for row in base.load_questions(include_gold=True)}  # only after sealed inference
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
        started = time.perf_counter()
        call = runner.judge_chat(router, [{"role": "system", "content": prompts["system_prompt"]},
                                          {"role": "user", "content": prompt}])
        judge_ms = (time.perf_counter() - started) * 1000
        correct, branch = runner.official_judge_label(call["content"])
        exposed = set(prediction["exposed_source_ids"])
        evidence = set(truth["gold_source_ids"])
        return {
            "qa_key": prediction["qa_key"], "topic": prediction["topic"], "major": prediction["major"],
            "minor": prediction["minor"], "question_type": prediction["question_type"],
            "condition": prediction["condition"], "correct": int(correct), "judge": call["content"],
            "judge_finish_reason": call["finish_reason"], "judge_attempts": call["attempts"], "judge_parse": branch,
            "judge_wall_ms": judge_ms,
            "evidence_recall": len(exposed & evidence) / len(evidence) if evidence else 1.0,
            "evidence_precision": len(exposed & evidence) / len(exposed) if exposed else 0.0,
        }

    parallel(run_dir, pending, work, lambda row: append_jsonl(run_dir / "results.jsonl", row),
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
# Statistics and report
# --------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def compare(rows: list[dict], left: str, right: str) -> dict:
    selected = [row for row in rows if row["condition"] in (left, right)]
    delta, low, high = paired_bootstrap(selected, left, right)
    test = exact_mcnemar(selected, left, right)
    paired: dict[str, dict[str, int]] = defaultdict(dict)
    topic: dict[str, str] = {}
    for row in selected:
        paired[row["qa_key"]][row["condition"]] = int(row["correct"])
        topic[row["qa_key"]] = row["topic"]
    both = [v for v in paired.values() if left in v and right in v]
    projects: dict[str, dict] = {}
    for project in sorted(set(topic.values())):
        values = [paired[k] for k in paired if topic[k] == project]
        projects[project] = {"n": len(values), "left": _mean(v[left] for v in values),
                             "right": _mean(v[right] for v in values)}
        projects[project]["delta"] = projects[project]["left"] - projects[project]["right"]
    _, cluster_low, cluster_high = clustered_bootstrap(selected, left, right, "topic")
    deltas = [p["delta"] for p in projects.values()]
    return {
        "left": left, "right": right, "n": len(both),
        "accuracy_left": _mean(v[left] for v in both), "accuracy_right": _mean(v[right] for v in both),
        "delta": delta, "question_ci95": [low, high], "mcnemar_p": test["p_value"],
        "left_only": test["left_only"], "right_only": test["right_only"],
        "both_correct": sum(1 for v in both if v[left] and v[right]),
        "both_wrong": sum(1 for v in both if not v[left] and not v[right]),
        "projects": projects, "project_ci95": [cluster_low, cluster_high],
        "projects_positive": sum(d > 0 for d in deltas), "sign_flip_p": sign_flip_p(deltas),
    }


def _reference_rows(keys: set[str]) -> tuple[list[dict], list[dict]]:
    results, predictions = [], []
    sources = ((SOURCE_FROZEN, ("RAW", EPISODES)), (BUDGET_FROZEN, (control.TOKEN_MATCHED,)),
               (UNLINKED_FROZEN, (unlinked.EVENTS_TOKEN,)))
    for directory, conditions in sources:
        results += [r for r in jsonl(directory / "run" / "results.jsonl") if r["qa_key"] in keys and r["condition"] in conditions]
        predictions += [p for p in jsonl(directory / "run" / "predictions.jsonl") if p["qa_key"] in keys and p["condition"] in conditions]
    return results, predictions


def condition_summary(condition: str, results: list[dict], predictions: list[dict], budget: dict[str, dict]) -> dict:
    res = [r for r in results if r["condition"] == condition]
    pre = [p for p in predictions if p["condition"] == condition]
    if condition == "RAW":
        tokens = [budget[p["qa_key"]]["raw10_context_tokens"] for p in pre]
    elif condition == EPISODES:
        tokens = [budget[p["qa_key"]]["episodes_context_tokens"] for p in pre]
    else:
        tokens = [p["context_tokens"] for p in pre]
    return {
        "n": len(res), "accuracy": _mean(r["correct"] for r in res),
        "mean_context_tokens": _mean(tokens), "mean_unique_raw_sources": _mean(len(set(p["exposed_source_ids"])) for p in pre),
        "order_changed_share": _mean(float(p["order_changed"]) for p in pre) if condition in NEW_CONDITIONS else None,
        "evidence_recall": _mean(r["evidence_recall"] for r in res),
        "evidence_precision": _mean(r["evidence_precision"] for r in res),
        "mean_answer_wall_ms": _mean(p["answer_wall_ms"] for p in pre if "answer_wall_ms" in p),
        "mean_judge_wall_ms": _mean(r["judge_wall_ms"] for r in res) if res and "judge_wall_ms" in res[0] else None,
        "answer_failure_markers": {m: sum(1 for p in pre if p["answer"] == m) for m in ANSWER_MARKERS},
        "judge_parse": dict(Counter(r.get("judge_parse", "unrecorded") for r in res)),
        "judge_re_requested": sum(1 for r in res if int(r.get("judge_attempts", 0) or 0) > 1),
        "judge_finish_reasons": dict(Counter(str(r.get("judge_finish_reason", "unrecorded")) for r in res)),
    }


def billing(run_dir: Path) -> dict:
    rows = [r for r in jsonl(run_dir / "billing_snapshots.jsonl") if r.get("usage_usd") is not None]
    before = next((r for r in rows if r["phase"] == "before"), None)
    after = next((r for r in reversed(rows) if r["phase"] == "after"), None)
    return {"before_usd": before and before["usage_usd"], "after_usd": after and after["usage_usd"],
            "delta_usd": (after["usage_usd"] - before["usage_usd"]) if before and after else None,
            "note": "OpenRouter key usage; includes any concurrent use of the same key"}


def interpretation(summary: dict) -> dict:
    primary = summary["comparisons"]["primary"]
    low, high = primary["question_ci95"]
    accuracy = summary["conditions"][RAW_CHRONO]["accuracy"]
    linking = summary["comparisons"]["episodes_vs_events_chrono"]
    return {
        "primary_rule": ("the episode gain is not reproduced by chronological ordering of the equal-token raw context" if low > 0 else
                         "chronologically ordered raw context is more accurate than episodes" if high < 0 else
                         "episodes and chronologically ordered raw context are not distinguishable"),
        "raw_chrono_band": ("supports a contribution of event representation beyond context volume and order"
                            if accuracy < ACCURACY_BANDS[0] else
                            "ordering explains part of the effect; event abstraction may keep an independent contribution"
                            if accuracy < ACCURACY_BANDS[1] else
                            "an advantage of event representation over chronological presentation cannot be claimed"),
        "linking_rule": ("episode linking contributes beyond chronologically ordered unlinked events"
                         if linking["question_ci95"][0] > 0 else
                         "an independent contribution of episode linking is not established"),
    }


def build_summary(run_dir: Path) -> dict:
    new_results = jsonl(run_dir / "results.jsonl")
    new_predictions = jsonl(run_dir / "predictions.jsonl")
    keys = {row["qa_key"] for row in new_results}
    ref_results, ref_predictions = _reference_rows(keys)
    results, predictions = ref_results + new_results, ref_predictions + new_predictions
    budget = {p["qa_key"]: p for p in jsonl(BUDGET_FROZEN / "run" / "predictions.jsonl")
              if p["qa_key"] in keys and p["condition"] == control.TOKEN_MATCHED}
    comparisons = {
        "primary": compare(results, EPISODES, RAW_CHRONO),
        "ordering_raw": compare(results, RAW_CHRONO, control.TOKEN_MATCHED),
        "ordering_events": compare(results, EVENTS_CHRONO, unlinked.EVENTS_TOKEN),
        "episodes_vs_events_chrono": compare(results, EPISODES, EVENTS_CHRONO),
        "events_chrono_vs_raw_chrono": compare(results, EVENTS_CHRONO, RAW_CHRONO),
        "exploratory_day_offsets": compare(results, RAW_RELATIVE, RAW_CHRONO),
        "exploratory_episodes_vs_relative": compare(results, EPISODES, RAW_RELATIVE),
    }
    summary = {
        "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL), "runner_sha256": sha256_file(RUNNER),
        "questions": len(keys), "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED},
        "conditions": {c: condition_summary(c, results, predictions, budget) for c in REPORT_ORDER},
        "comparisons": comparisons,
        "api_usage_new_work": usage_summary(run_dir), "billing": billing(run_dir),
        "sign_flip_note": "five projects: the smallest attainable two-sided sign-flip p is 0.0625",
    }
    summary["interpretation"] = interpretation(summary)
    return summary


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}"


def _ci(pair: list[float]) -> str:
    return f"[{100 * pair[0]:+.1f}, {100 * pair[1]:+.1f}]"


def render_report(run_dir: Path) -> Path:
    summary = build_summary(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# EverMemBench Temporal control v2: chronological order of the same retrieved context", "",
        f"- protocol: `{PROTOCOL.name}` sha256 `{summary['protocol_sha256']}`",
        f"- runner sha256: `{summary['runner_sha256']}`; inputs sha256 `{sha256_file(run_dir / 'inputs.jsonl')}`",
        f"- references: RAW and RAW+EPISODES from `{SOURCE_FROZEN.name}`; {control.TOKEN_MATCHED} from "
        f"`{BUDGET_FROZEN.name}`; {unlinked.EVENTS_TOKEN} from `{UNLINKED_FROZEN.name}`",
        f"- questions: {summary['questions']} Temporal Duration; bootstrap {BOOTSTRAP_ITERATIONS} samples, seed {BOOTSTRAP_SEED}", "",
        "## Conditions", "",
        "| condition | n | accuracy % | context tokens | unique raw sources | order changed | evidence recall % | "
        "evidence precision % | answer ms | judge ms | answer markers | judge parse | judge re-requested |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for condition, c in summary["conditions"].items():
        role = " (primary control)" if condition == RAW_CHRONO else " (exploratory)" if condition == RAW_RELATIVE else ""
        lines.append(
            f"| {condition}{role} | {c['n']} | {_pct(c['accuracy'])} | {c['mean_context_tokens']:.0f} | "
            f"{c['mean_unique_raw_sources']:.1f} | {'--' if c['order_changed_share'] is None else f'{c['order_changed_share']:.3f}'} | "
            f"{_pct(c['evidence_recall'])} | {_pct(c['evidence_precision'])} | {c['mean_answer_wall_ms']:.0f} | "
            f"{'--' if c['mean_judge_wall_ms'] is None else f'{c['mean_judge_wall_ms']:.0f}'} | "
            f"`{json.dumps(c['answer_failure_markers'])}` | `{json.dumps(c['judge_parse'], sort_keys=True)}` | {c['judge_re_requested']} |"
        )
    labels = {
        "primary": "**primary**", "ordering_raw": "secondary: ordering, raw", "ordering_events": "secondary: ordering, events",
        "episodes_vs_events_chrono": "secondary: linking beyond ordered events",
        "events_chrono_vs_raw_chrono": "secondary: events beyond raw, both ordered",
        "exploratory_day_offsets": "exploratory: day offsets", "exploratory_episodes_vs_relative": "exploratory",
    }
    lines += ["", "## Paired comparisons (pp)", "",
              "| role | difference | left % | right % | delta | question 95% CI | McNemar p | left-only / right-only / both correct / both wrong | "
              "project 95% CI | per-project deltas | sign-flip p |",
              "|---|---|---:|---:|---:|---|---:|---|---|---|---:|"]
    for name, c in summary["comparisons"].items():
        per_project = ", ".join(f"{k}: {100 * v['delta']:+.1f}" for k, v in c["projects"].items())
        lines.append(
            f"| {labels[name]} | {c['left']} − {c['right']} | {_pct(c['accuracy_left'])} | {_pct(c['accuracy_right'])} | "
            f"{100 * c['delta']:+.1f} | {_ci(c['question_ci95'])} | {c['mcnemar_p']:.4f} | "
            f"{c['left_only']} / {c['right_only']} / {c['both_correct']} / {c['both_wrong']} | {_ci(c['project_ci95'])} | "
            f"{per_project} | {c['sign_flip_p']:.4f} |")
    lines += ["", f"Sign-flip note: {summary['sign_flip_note']}; the project bootstrap has five clusters and is descriptive.", "",
              "## Per-project accuracy (%)", "", "| project | " + " | ".join(REPORT_ORDER) + " |",
              "|---|" + "---:|" * len(REPORT_ORDER)]
    project_accuracy: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in _reference_rows({r["qa_key"] for r in jsonl(run_dir / "results.jsonl")})[0] + jsonl(run_dir / "results.jsonl"):
        project_accuracy[row["topic"]][row["condition"]].append(row["correct"])
    for project, values in sorted(project_accuracy.items()):
        lines.append(f"| {project} | " + " | ".join(_pct(_mean(values[c])) for c in REPORT_ORDER) + " |")
    rules = summary["interpretation"]
    usage, bill = summary["api_usage_new_work"], summary["billing"]
    lines += ["", "## Interpretation rules fixed before new outcomes", "",
              f"- primary 95% CI {_ci(summary['comparisons']['primary']['question_ci95'])}: **{rules['primary_rule']}**.",
              f"- {RAW_CHRONO} accuracy {_pct(summary['conditions'][RAW_CHRONO]['accuracy'])}%: **{rules['raw_chrono_band']}**.",
              f"- RAW+EPISODES − {EVENTS_CHRONO} 95% CI {_ci(summary['comparisons']['episodes_vs_events_chrono']['question_ci95'])}: "
              f"**{rules['linking_rule']}**.", "",
              "## Cost (new work in this directory)", "",
              f"- API calls: {usage['api_calls']}; list-price estimate USD {usage['estimated_usd_upper_bound']:.4f}",
              f"- by model: `{json.dumps(usage['by_model'], sort_keys=True)}`",
              f"- OpenRouter key usage: before {bill['before_usd']}, after {bill['after_usd']}, delta {bill['delta_usd']} ({bill['note']})"]
    report = run_dir / "report.md"
    report.write_text("\n".join(lines) + "\n")
    log(run_dir, f"report ready: {report}")
    return report


def render_health(run_dir: Path, questions: list[dict]) -> Path:
    """Smoke: technical health only; accuracy is deliberately not computed."""
    predictions, results = jsonl(run_dir / "predictions.jsonl"), jsonl(run_dir / "results.jsonl")
    health = {
        "questions": len(questions), "predictions": len(predictions), "results": len(results),
        "expected_rows": len(questions) * len(NEW_CONDITIONS),
        "answer_failure_markers": {c: {m: sum(1 for p in predictions if p["condition"] == c and p["answer"] == m)
                                       for m in ANSWER_MARKERS} for c in NEW_CONDITIONS},
        "judge_parse": dict(Counter(r["judge_parse"] for r in results)),
        "judge_re_requested": sum(1 for r in results if r["judge_attempts"] > 1),
        "judge_finish_reasons": dict(Counter(r["judge_finish_reason"] for r in results)),
        "api_usage": usage_summary(run_dir), "billing": billing(run_dir),
    }
    health["technically_clean"] = (
        health["predictions"] == health["results"] == health["expected_rows"]
        and not any(any(v.values()) for v in health["answer_failure_markers"].values())
        and set(health["judge_parse"]) == {"json"}
    )
    path = run_dir / "smoke_health.json"
    path.write_text(json.dumps(health, indent=2, sort_keys=True) + "\n")
    log(run_dir, f"smoke health: technically_clean={health['technically_clean']}")
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def run(run_dir: Path, smoke: bool, answer_workers: int, judge_workers: int) -> None:
    questions = control.target_questions(smoke)
    jobs, checks = prepare_jobs(questions, full=not smoke)
    inputs_sha256 = write_or_verify_inputs(run_dir, [input_record(job) for job in jobs])
    ensure_run_manifest(run_dir, build_manifest(questions, inputs_sha256, checks))
    log(run_dir, f"inputs frozen ({inputs_sha256}); checks {json.dumps(checks, sort_keys=True)}")
    router = control.make_router(run_dir)
    if not any(r["phase"] == "before" for r in jsonl(run_dir / "billing_snapshots.jsonl")):
        billing_snapshot(run_dir, "before")
    check_routes(router, run_dir)
    run_inference(run_dir, questions, jobs, router, answer_workers)
    score(run_dir, questions, router, judge_workers)
    billing_snapshot(run_dir, "after")
    if smoke:
        render_health(run_dir, questions)
    else:
        render_report(run_dir)


def freeze() -> Path:
    if not (RUN_DIR / "results_meta.json").exists() or not (SMOKE_DIR / "smoke_health.json").exists():
        raise SystemExit("freeze requires a sealed full run and a completed smoke")
    sources = [PROTOCOL, SUPERSEDED_PROTOCOL, RUNNER, TESTS, Path(control.__file__), Path(unlinked.__file__),
               Path(runner.__file__), Path(base.__file__), Path(common.__file__)]
    return freeze_run(RUN_DIR, sources, related_dirs=[SMOKE_DIR, AMENDMENT_DIR])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-workers", type=int, default=16)
    parser.add_argument("--judge-workers", type=int, default=4)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("verify", "smoke", "all", "report", "freeze"):
        modes.add_argument(f"--{mode}", action="store_true")
    args = parser.parse_args()
    target_dir = SMOKE_DIR if args.smoke else RUN_DIR
    try:
        if args.verify:
            jobs, checks = prepare_jobs(control.target_questions(smoke=False), full=True)
            print(json.dumps(checks, indent=2, sort_keys=True))
            for condition in NEW_CONDITIONS:
                items = [j for j in jobs if j["condition"] == condition]
                print(f"{condition}: context tokens {_mean(j['context_tokens'] for j in items):.1f} "
                      f"(frozen source {_mean(j['source_context_tokens'] for j in items):.1f}), "
                      f"unique sources {_mean(len(control.unique_source_ids(j['docs'])) for j in items):.2f}")
            return
        if args.report:
            print(render_report(RUN_DIR))
            return
        if args.freeze:
            print(freeze())
            return
        run(target_dir, args.smoke, args.answer_workers, args.judge_workers)
    except RateLimitError as error:
        log(target_dir, f"rate limit; completed work is cached: {error}")
        raise SystemExit(75) from None
    except AuthenticationError as error:
        raise SystemExit(f"API authentication failed: {error}") from None


if __name__ == "__main__":
    main()
