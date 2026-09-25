"""Full 43-network / 1,031-QA SocialMemBench run for this project's memory.

One resumable overnight command::

    caffeinate -i python3 -m research.socialmembench_full_run --all

The protocol matches the public paper where it matters for comparison:
GPT-4o-mini-2024-07-18 at temperature 0 for extraction, answering, and
judging; text-embedding-3-small cosine retrieval; top-k=10; five answer and
ten judge workers; exact MC parsing; question- and network-weighted reports.

The three conditions share raw turns, extraction outputs, embedding model,
answerer, judge, and retrieval rule:
  RAW               -- top-10 raw turns
  RAW+FLAT          -- one unified top-10 over raw turns + assertions
  RAW+VERSIONED     -- one unified top-10 over raw turns + versioned cells

Derived items always carry their source turn IDs and are expanded back to
raw evidence in the answer context.  This is our adapter, not an official
leaderboard submission.  It uses the benchmark's published representative
answer/judge prompts because the anonymous code mirror currently returns 401.
Paper: https://arxiv.org/html/2605.17789 (Appendices C and G).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.socialmembench_pilot import (
    Candidate,
    _cell_extract_prompt,
    _json,
    _normalized,
    _parse_json_object,
    _slot_resolution_prompt,
    build_cell_assertion_cache,
    build_raw_candidates,
    build_slot_resolution_cache,
    ensure_data,
    BASE_URL,
)
from research.stage5_heldout import build_memory_candidates


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/tmp/socialmembench")
RUN_DIR = ROOT / ".research_runs" / "socialmembench_full_official_v1"

MODEL = "gpt-4o-mini-2024-07-18"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 10
ANSWER_WORKERS = 5
JUDGE_WORKERS = 10
BOOTSTRAP_SEED = 20260911
BOOTSTRAP_ITERATIONS = 10_000
SCHEMA_VERSION = "full_official_v1"
CONDITIONS = ("RAW", "RAW+FLAT", "RAW+VERSIONED")

EXTRACTION_CACHE = RUN_DIR / "session_assertions.jsonl"
SLOT_CACHE = RUN_DIR / "slot_resolution.jsonl"
SNAPSHOT = RUN_DIR / "memory_snapshot.jsonl"
SNAPSHOT_META = RUN_DIR / "memory_snapshot_meta.json"
API_CACHE = RUN_DIR / "openai_cache.jsonl"
PREDICTIONS = RUN_DIR / "predictions.jsonl"
PREDICTIONS_META = RUN_DIR / "predictions_meta.json"
RESULTS = RUN_DIR / "results.jsonl"
REPORT = RUN_DIR / "report.md"
RUN_LOG = RUN_DIR / "run.log"
EMBED_DIR = RUN_DIR / "embeddings"


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


def qa_key(network_id: str, qa_id: str) -> str:
    """QA IDs are unique only within a social network in the public release."""
    return f"{network_id}:{qa_id}"


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, file=sys.stderr, flush=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a") as out:
        out.write(line + "\n")


class CachedOpenAI:
    """Thread-safe content-addressed OpenAI cache used by every LLM call."""

    def __init__(self, cache_path: Path = API_CACHE):
        self.client = OpenAI(max_retries=5, timeout=120.0)
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.values = {row["key"]: row["value"] for row in _jsonl(cache_path)}

    def chat(self, prompt: str, *, json_output: bool = False) -> str:
        payload = {"model": MODEL, "temperature": 0, "json": json_output, "prompt": prompt}
        key = "chat:" + _sha256(json.dumps(payload, sort_keys=True).encode())
        with self.lock:
            cached = self.values.get(key)
        if cached is not None:
            return str(cached)
        kwargs: dict[str, Any] = {
            "model": MODEL,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        value = response.choices[0].message.content or ""
        with self.lock:
            if key not in self.values:
                self.values[key] = value
                _append_jsonl(self.cache_path, {"key": key, "value": value})
        return value

    def embeddings(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 1536), dtype=np.float32)
        response = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        return _normalized([item.embedding for item in ordered])


_api: CachedOpenAI | None = None


def api() -> CachedOpenAI:
    global _api
    if _api is None:
        _api = CachedOpenAI()
    return _api


class _OpenAIChatModel:
    """Tiny compatibility shim for the existing query-independent extractor."""

    def invoke(self, prompt: str) -> SimpleNamespace:
        json_output = "JSON only" in prompt or "Return JSON" in prompt
        return SimpleNamespace(content=api().chat(prompt, json_output=json_output))


def _install_openai_extraction_provider() -> None:
    import llm.groq_client as provider

    provider.get_chat_model = lambda *_args, **_kwargs: _OpenAIChatModel()


def _preflight() -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY не найден. Добавь в .env строку OPENAI_API_KEY=... "
            "и повтори ту же команду. Ключ нужен для официально сопоставимых "
            "GPT-4o-mini answer/judge и text-embedding-3-small."
        )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ensure_data(DATA_DIR)
    networks_path = DATA_DIR / "networks.parquet"
    if not networks_path.exists():
        _log("data: downloading networks.parquet")
        urllib.request.urlretrieve(f"{BASE_URL}/networks.parquet", networks_path)
    try:
        _log("preflight: checking GPT-4o-mini model access")
        api().client.models.retrieve(MODEL)
        _log("preflight: checking text-embedding-3-small access")
        vector = _cached_embedding_batch(["SocialMemBench API preflight"])
    except (AuthenticationError, RateLimitError) as error:
        raise SystemExit(f"OpenAI API preflight failed: {error}") from None
    if vector.shape != (1, 1536) or not np.isfinite(vector).all():
        raise RuntimeError(f"embedding preflight returned unexpected shape/data: {vector.shape}")
    _log("preflight: chat + embeddings OK")


def _snapshot_rows(flat: list[Candidate], versioned: list[Candidate]) -> list[dict]:
    return (
        [{"corpus": "flat", **candidate.as_dict()} for candidate in flat]
        + [{"corpus": "versioned", **candidate.as_dict()} for candidate in versioned]
    )


def prepare_memory() -> None:
    if SNAPSHOT.exists() and SNAPSHOT_META.exists():
        meta = json.loads(SNAPSHOT_META.read_text())
        if _sha256_file(SNAPSHOT) != meta.get("snapshot_sha256"):
            raise RuntimeError("memory snapshot checksum mismatch")
        _log(f"memory: reuse sealed snapshot ({meta['flat_count']} flat, {meta['versioned_count']} cells)")
        return

    _install_openai_extraction_provider()
    api()  # initialize once before the extractor starts worker threads
    conversations = pd.read_parquet(DATA_DIR / "conversations.parquet")
    networks = sorted(conversations.network_id.astype(str).unique())
    if len(networks) != 43 or conversations.session_id.nunique() != 348:
        raise RuntimeError("dataset shape mismatch: expected 43 networks / 348 sessions")

    _log("memory: extracting 348 sessions query-independently (resumable)")
    extraction = build_cell_assertion_cache(conversations, EXTRACTION_CACHE)
    expected_pairs = {
        (str(row.network_id), str(row.session_id))
        for row in conversations[["network_id", "session_id"]].drop_duplicates().itertuples(index=False)
    }
    missing = expected_pairs - set(extraction)
    if missing:
        raise RuntimeError(f"extraction incomplete ({len(missing)} sessions missing); rerun the same command")

    _log("memory: resolving cross-session topic slots (resumable)")
    slots = build_slot_resolution_cache(extraction, SLOT_CACHE, set(networks))
    _log("memory: materializing flat assertions and chronological versioned cells")
    flat, versioned, stats = build_memory_candidates(conversations, networks, extraction, slots)
    rows = _snapshot_rows(flat, versioned)
    payload = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n").encode()
    SNAPSHOT.write_bytes(payload)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL,
        "network_count": len(networks),
        "session_count": len(expected_pairs),
        "flat_count": len(flat),
        "versioned_count": len(versioned),
        "build_stats": stats,
        "snapshot_sha256": _sha256(payload),
        "extraction_prompt_sha256": _sha256(_cell_extract_prompt("{SESSION}").encode()),
        "slot_prompt_sha256": _sha256(_slot_resolution_prompt([]).encode()),
        "qa_read": False,
    }
    SNAPSHOT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    _log(f"memory: sealed {len(flat)} flat assertions / {len(versioned)} versioned cells")


def _candidate_from_row(row: dict) -> Candidate:
    return Candidate(
        candidate_id=row["candidate_id"], kind=row["kind"], text=row["text"],
        source_ids=tuple(row["source_ids"]), asserted_by=tuple(row.get("asserted_by", [])),
        entities=tuple(row.get("entities", [])), observed_at=row.get("observed_at", ""),
    )


def _search_text(candidate: Candidate) -> str:
    if candidate.kind == "raw":
        return candidate.text
    return (
        f"About: {', '.join(candidate.entities)}. "
        f"Viewpoint owner: {', '.join(candidate.asserted_by)}. {candidate.text}"
    )


def _embedding_matrix(candidates: list[Candidate], label: str) -> np.ndarray:
    payload = "\n".join(f"{item.candidate_id}\t{_search_text(item)}" for item in candidates)
    digest = _sha256(payload.encode())[:16]
    path = EMBED_DIR / f"{label}-{digest}.npy"
    if path.exists():
        return np.load(path)
    chunks = []
    texts = [_search_text(item) for item in candidates]
    for start in range(0, len(texts), 256):
        chunks.append(_cached_embedding_batch(texts[start:start + 256]))
    matrix = np.vstack(chunks) if chunks else np.empty((0, 1536), dtype=np.float32)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    return matrix


def _cached_embedding_batch(texts: list[str]) -> np.ndarray:
    """Persist every API batch, not only completed network matrices."""
    digest = _sha256(json.dumps({"model": EMBED_MODEL, "texts": texts}, ensure_ascii=False).encode())[:24]
    path = EMBED_DIR / f"api-batch-{digest}.npy"
    if path.exists():
        return np.load(path)
    matrix = api().embeddings(texts)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    return matrix


@dataclass
class CosineIndex:
    candidates: list[Candidate]
    matrix: np.ndarray

    @classmethod
    def build(cls, candidates: list[Candidate], label: str) -> "CosineIndex":
        return cls(candidates, _embedding_matrix(candidates, label))

    def search(self, query_vector: np.ndarray, k: int = TOP_K) -> list[Candidate]:
        if not self.candidates:
            return []
        scores = np.sum(self.matrix * query_vector, axis=1)
        return [self.candidates[i] for i in np.argsort(-scores)[:k]]


def pack_context(
    candidates: list[Candidate], raw_by_source: dict[str, Candidate], session_index: dict[str, int],
) -> tuple[str, list[str]]:
    """Render top-level memories and expand every valid provenance edge."""
    lines: list[str] = []
    exposed: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.kind == "raw":
            source_id = candidate.source_ids[0]
            lines.append(
                f"[session {session_index.get(source_id, -1)}] [{candidate.observed_at}] "
                f"{candidate.text} (source_turn_id={source_id})"
            )
            if source_id not in seen:
                exposed.append(source_id)
                seen.add(source_id)
            continue
        lines.append(
            f"[DERIVED MEMORY] about={list(candidate.entities)}; "
            f"viewpoint_owner={list(candidate.asserted_by)}\n{candidate.text}"
        )
        for source_id in candidate.source_ids:
            raw = raw_by_source.get(source_id)
            if raw is None:
                continue
            lines.append(
                f"  [SOURCE session {session_index.get(source_id, -1)}] [{raw.observed_at}] "
                f"{raw.text} (source_turn_id={source_id})"
            )
            if source_id not in seen:
                exposed.append(source_id)
                seen.add(source_id)
    return "\n".join(lines), exposed


OPEN_ANSWER_PROMPT = """You are evaluating a social group AI memory system.
You have retrieved memories from a group chat and must answer an open-ended
question. Answer ONLY using the retrieved memories. Use no outside knowledge
or guesses.

Pay attention to WHO said what and WHEN. Attribution and timing matter.

Answering guidance by question type:
- Single preference (Q1): Name the person and state their preference. Be specific.
- Group decision (Q2): State the decision AND name any dissenter(s); implicit dissent counts.
- Multi-contact (Q3): Cover every group member; say "unclear" for those with no signal.
- Theory of mind (Q5): State BOTH the preference AND who revealed it, with the specific observable action that shows they already knew.
- Temporal shift (Q8): State the OLD preference, the NEW preference, and what triggered the change.

[MEMORIES]
{context}

[QUESTION]
{question}"""


def _option_lines(options: Any) -> str:
    if isinstance(options, dict):
        return "\n".join(f"{key}. {value}" for key, value in options.items())
    if isinstance(options, list):
        return "\n".join(
            f"{item.get('option')}. {item.get('name')}" if isinstance(item, dict) else str(item)
            for item in options
        )
    return ""


def answer_prompt(question: str, answer_format: str, options: Any, context: str) -> str:
    if answer_format == "multiple_choice":
        return OPEN_ANSWER_PROMPT.format(context=context, question=question) + (
            "\n\n[OPTIONS]\n" + _option_lines(options)
            + "\n\nReturn only the single correct option letter (for example: A)."
        )
    return OPEN_ANSWER_PROMPT.format(context=context, question=question)


def parse_choice(answer: str, valid: set[str] | None = None) -> str | None:
    valid = valid or set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    text = (answer or "").strip().upper()
    patterns = (
        r"^([A-Z])(?:[.)\s]|$)",
        r"(?:FINAL[_ ]OPTION|ANSWER|OPTION)(?:\s+IS)?\s*[:\-]?\s*([A-Z])\b",
        r"\bTHE ANSWER IS\s+([A-Z])\b",
    )
    found = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text))
    choices = [choice for choice in found if choice in valid]
    return choices[-1] if choices else None


def _load_memory() -> tuple[dict[str, list[Candidate]], dict[str, list[Candidate]]]:
    meta = json.loads(SNAPSHOT_META.read_text())
    if _sha256_file(SNAPSHOT) != meta["snapshot_sha256"]:
        raise RuntimeError("memory snapshot is not sealed")
    flat: dict[str, list[Candidate]] = defaultdict(list)
    versioned: dict[str, list[Candidate]] = defaultdict(list)
    for row in _jsonl(SNAPSHOT):
        candidate = _candidate_from_row(row)
        network_id = candidate.candidate_id.split(":", 2)[1]
        (flat if row["corpus"] == "flat" else versioned)[network_id].append(candidate)
    return dict(flat), dict(versioned)


def run_inference() -> None:
    if not SNAPSHOT.exists():
        raise RuntimeError("memory snapshot missing")
    api()  # initialize once before answer workers
    conversations = pd.read_parquet(DATA_DIR / "conversations.parquet")
    # Deliberately excludes gold answer, correct option, and evidence anchors.
    qa = pd.read_parquet(
        DATA_DIR / "qa.parquet",
        columns=["qa_id", "network_id", "query_type", "question", "answer_format", "options_json"],
    )
    if len(qa) != 1031 or qa.network_id.nunique() != 43:
        raise RuntimeError("QA shape mismatch: expected 1,031 questions / 43 networks")
    raw_by_network, raw_by_source = build_raw_candidates(conversations)
    flat_by_network, versioned_by_network = _load_memory()
    session_index = {str(row.turn_id): int(row.session_index) for row in conversations.itertuples(index=False)}

    done = {row["qa_key"] for row in _jsonl(PREDICTIONS)}
    pending = [
        row for row in qa.itertuples(index=False)
        if qa_key(str(row.network_id), str(row.qa_id)) not in done
    ]
    _log(f"inference: {len(pending)} pending / {len(done)} cached")
    if not pending:
        return

    indices: dict[str, dict[str, CosineIndex]] = {}
    for number, network_id in enumerate(sorted(raw_by_network), 1):
        raw = raw_by_network[network_id]
        indices[network_id] = {
            "RAW": CosineIndex.build(raw, f"{network_id}-raw"),
            "RAW+FLAT": CosineIndex.build(raw + flat_by_network.get(network_id, []), f"{network_id}-raw-flat"),
            "RAW+VERSIONED": CosineIndex.build(
                raw + versioned_by_network.get(network_id, []), f"{network_id}-raw-versioned"
            ),
        }
        if number % 10 == 0 or number == 43:
            _log(f"inference: embedded/indexed {number}/43 networks")

    questions = [str(row.question) for row in pending]
    query_vectors = []
    for start in range(0, len(questions), 256):
        query_vectors.append(_cached_embedding_batch(questions[start:start + 256]))
    query_matrix = np.vstack(query_vectors)

    def one(item: tuple[int, Any]) -> dict:
        index, row = item
        answers, retrieved_ids, exposed_ids = {}, {}, {}
        options = _json(row.options_json, {})
        for condition in CONDITIONS:
            retrieved = indices[str(row.network_id)][condition].search(query_matrix[index])
            context, exposed = pack_context(retrieved, raw_by_source, session_index)
            answers[condition] = api().chat(
                answer_prompt(str(row.question), str(row.answer_format), options, context)
            )
            retrieved_ids[condition] = [candidate.candidate_id for candidate in retrieved]
            exposed_ids[condition] = exposed
        return {
            "qa_key": qa_key(str(row.network_id), str(row.qa_id)),
            "qa_id": str(row.qa_id), "network_id": str(row.network_id),
            "query_type": str(row.query_type), "answer_format": str(row.answer_format),
            "answers": answers, "retrieved_ids": retrieved_ids, "exposed_source_ids": exposed_ids,
        }

    with ThreadPoolExecutor(max_workers=ANSWER_WORKERS) as pool:
        futures = {pool.submit(one, item): item[1].qa_id for item in enumerate(pending)}
        for number, future in enumerate(as_completed(futures), 1):
            _append_jsonl(PREDICTIONS, future.result())
            if number % 25 == 0 or number == len(futures):
                _log(f"inference: answered {number}/{len(futures)} pending questions")

    rows = _jsonl(PREDICTIONS)
    if len(rows) != 1031 or len({row["qa_key"] for row in rows}) != 1031:
        raise RuntimeError("predictions are incomplete or duplicated")
    PREDICTIONS_META.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "model": MODEL, "temperature": 0,
        "top_k": TOP_K, "conditions": CONDITIONS, "prediction_count": len(rows),
        "predictions_sha256": _sha256_file(PREDICTIONS), "gold_read": False,
    }, indent=2))
    _log("inference: sealed all 1,031 predictions")


JUDGE_PROMPT = """Score the generated answer against the gold answer on a 0.0–1.0 scale.

CRITICAL RULES:
1. A concise correct answer scores THE SAME as a verbose one.
2. Attributing a fact to the wrong person -> score <= 0.3.
3. "NOT ENOUGH INFORMATION" always scores 0.0.

General rubric:
- 1.0: Core fact correct, correct attribution, all parts present
- 0.7–0.9: Right answer but one minor part missing
- 0.4–0.6: Right direction but wrong detail or missing element
- 0.0–0.3: Wrong attribution, wrong fact, or no answer

Q-type rules:
- Q3: score = fraction of group members correctly recalled
- Q4: correct speaker + foil not named -> 1.0; foil also named -> 0.7; wrong speaker -> 0.0–0.3
- Q5: both preference AND observable evidence of prior knowledge required for 1.0; one part -> 0.5
- Q8: old state (+0.33) + new state (+0.33) + trigger (+0.33) = 1.0

Question: {question}
Gold answer: {gold}
Generated answer: {answer}

Output JSON only:
{{"score": 0.0, "rationale": "one sentence"}}"""


def _mc_score(answer: str, correct: str, options: Any) -> tuple[float, str]:
    if isinstance(options, dict):
        valid = {str(key).upper() for key in options}
    else:
        valid = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    parsed = parse_choice(answer, valid)
    score = float(parsed == str(correct).strip().upper())
    return score, f"parsed={parsed!r}; gold={correct!r}"


def score() -> None:
    meta = json.loads(PREDICTIONS_META.read_text())
    if _sha256_file(PREDICTIONS) != meta.get("predictions_sha256"):
        raise RuntimeError("predictions checksum mismatch; refusing to score")
    api()  # initialize once before judge workers
    predictions = {row["qa_key"]: row for row in _jsonl(PREDICTIONS)}
    qa = pd.read_parquet(DATA_DIR / "qa.parquet")
    networks = pd.read_parquet(DATA_DIR / "networks.parquet", columns=["network_id", "tier"])
    tier_by_network = dict(zip(networks.network_id, networks.tier))
    finished = {(row["qa_key"], row["condition"]): row for row in _jsonl(RESULTS)}
    jobs = []

    for row in qa.itertuples(index=False):
        row_key = qa_key(str(row.network_id), str(row.qa_id))
        prediction = predictions[row_key]
        options = _json(row.options_json, {})
        gold_ids = {
            str(anchor["turn_id"]) for anchor in _json(row.evidence_anchors_json, []) if anchor.get("turn_id")
        }
        for condition in CONDITIONS:
            key = (row_key, condition)
            if key in finished:
                continue
            base = {
                "qa_key": row_key,
                "qa_id": str(row.qa_id), "network_id": str(row.network_id),
                "tier": tier_by_network[str(row.network_id)], "query_type": str(row.query_type),
                "condition": condition, "answer": prediction["answers"][condition],
            }
            exposed = set(prediction["exposed_source_ids"][condition])
            base["evidence_recall"] = len(exposed & gold_ids) / len(gold_ids) if gold_ids else 1.0
            base["evidence_precision"] = len(exposed & gold_ids) / len(exposed) if exposed else 0.0
            if str(row.answer_format) == "multiple_choice":
                value, rationale = _mc_score(base["answer"], str(row.correct_option), options)
                base.update(score=value, rationale=rationale, scoring="exact_mc")
                _append_jsonl(RESULTS, base)
                finished[key] = base
            else:
                prompt = JUDGE_PROMPT.format(
                    question=str(row.question), gold=str(row.answer), answer=base["answer"]
                )
                jobs.append((key, base, prompt))

    _log(f"score: {len(jobs)} pending open-answer judge calls; MC already scored exactly")

    def judge(job: tuple[tuple[str, str], dict, str]) -> dict:
        _key, base, prompt = job
        parsed = _parse_json_object(api().chat(prompt, json_output=True)) or {}
        try:
            value = min(1.0, max(0.0, float(parsed.get("score", 0.0))))
        except (TypeError, ValueError):
            value = 0.0
        return {**base, "score": value, "rationale": str(parsed.get("rationale", "")), "scoring": "llm_judge"}

    completed = 0
    for offset in range(0, len(jobs), JUDGE_WORKERS):
        batch = jobs[offset:offset + JUDGE_WORKERS]
        first_error = None
        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
            futures = [pool.submit(judge, job) for job in batch]
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as error:
                    first_error = first_error or error
                    continue
                _append_jsonl(RESULTS, row)
                completed += 1
                if completed % 50 == 0 or completed == len(jobs):
                    _log(f"score: judged {completed}/{len(jobs)} pending answers")
        if first_error is not None:
            raise first_error

    rows = _jsonl(RESULTS)
    expected = 1031 * len(CONDITIONS)
    if len(rows) != expected or len({(row['qa_key'], row['condition']) for row in rows}) != expected:
        raise RuntimeError(f"results incomplete or duplicated: expected {expected}, got {len(rows)}")
    REPORT.write_text(render_report(rows))
    _log(f"score: final report ready at {REPORT}")


def _network_means(rows: list[dict], condition: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["condition"] == condition:
            buckets[row["network_id"]].append(float(row["score"]))
    return {network: sum(values) / len(values) for network, values in buckets.items()}


def bootstrap_network_ci(values: dict[str, float], seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    keys = sorted(values)
    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [values[keys[rng.randrange(len(keys))]] for _ in keys]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[250], means[9749]


def bootstrap_paired_ci(rows: list[dict], left: str, right: str) -> tuple[float, float]:
    left_values = _network_means(rows, left)
    right_values = _network_means(rows, right)
    deltas = {key: left_values[key] - right_values[key] for key in left_values.keys() & right_values.keys()}
    return bootstrap_network_ci(deltas)


PUBLISHED = {
    "full-context GPT-4o-mini": 0.369,
    "uncompressed raw retrieval": 0.345,
    "subject-mem": 0.321,
    "SMG": 0.213,
    "Graphiti": 0.178,
    "LangMem": 0.162,
    "Mem0": 0.143,
    "Cognee": 0.120,
}


def render_report(rows: list[dict]) -> str:
    by_condition = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    lines = [
        "# SocialMemBench — full 43-network run", "",
        "Official-compatible evaluation of this project's adapter (not an official leaderboard submission).",
        "", f"- QA: 1,031", f"- networks: 43", f"- model: `{MODEL}`",
        f"- embeddings: `{EMBED_MODEL}`", f"- retrieval: cosine top-{TOP_K}", "",
        "## Overall", "", "| condition | MeanQ | MeanN | network-bootstrap 95% CI | evidence recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        condition_rows = by_condition[condition]
        mean_q = sum(float(row["score"]) for row in condition_rows) / len(condition_rows)
        network_values = _network_means(rows, condition)
        mean_n = sum(network_values.values()) / len(network_values)
        lo, hi = bootstrap_network_ci(network_values)
        evidence = sum(float(row["evidence_recall"]) for row in condition_rows) / len(condition_rows)
        lines.append(f"| {condition} | {mean_q:.3f} | {mean_n:.3f} | [{lo:.3f}, {hi:.3f}] | {evidence:.3f} |")

    lines += ["", "## By question type", "", "| type | n | RAW | RAW+FLAT | RAW+VERSIONED |", "|---|---:|---:|---:|---:|"]
    for query_type in sorted({row["query_type"] for row in rows}):
        type_rows = [row for row in rows if row["query_type"] == query_type]
        n = len(type_rows) // len(CONDITIONS)
        values = []
        for condition in CONDITIONS:
            selected = [float(row["score"]) for row in type_rows if row["condition"] == condition]
            values.append(sum(selected) / len(selected))
        lines.append(f"| {query_type} | {n} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |")

    lines += ["", "## By network tier", "", "| tier | n QA | RAW | RAW+FLAT | RAW+VERSIONED |", "|---|---:|---:|---:|---:|"]
    for tier in ("small", "medium", "large"):
        tier_rows = [row for row in rows if row["tier"] == tier]
        n = len(tier_rows) // len(CONDITIONS)
        values = []
        for condition in CONDITIONS:
            selected = [float(row["score"]) for row in tier_rows if row["condition"] == condition]
            values.append(sum(selected) / len(selected))
        lines.append(f"| {tier} | {n} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |")

    lines += ["", "## Paired network-level deltas", ""]
    for right in ("RAW", "RAW+FLAT"):
        left_values = _network_means(rows, "RAW+VERSIONED")
        right_values = _network_means(rows, right)
        deltas = [left_values[key] - right_values[key] for key in sorted(left_values)]
        wins = sum(delta > 0 for delta in deltas)
        ties = sum(delta == 0 for delta in deltas)
        losses = sum(delta < 0 for delta in deltas)
        lo, hi = bootstrap_paired_ci(rows, "RAW+VERSIONED", right)
        lines.append(
            f"- RAW+VERSIONED vs {right}: {wins}W/{ties}T/{losses}L networks; "
            f"mean delta={sum(deltas)/len(deltas):+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]"
        )

    lines += ["", "## Published paper reference (MeanQ)", "", "These are paper values, not reruns in this repository.", "", "| condition | MeanQ |", "|---|---:|"]
    for name, value in PUBLISHED.items():
        lines.append(f"| {name} | {value:.3f} |")
    lines += [
        "", "## Interpretation guardrails", "",
        "- Compare our three rows directly: they are a controlled ablation sharing extraction and retrieval.",
        "- Comparison with published systems is informative but not an official submission until the authors' harness is accessible again.",
        "- Evidence recall is secondary and measures exposed raw provenance against gold anchors; it is not part of the paper's primary score.",
    ]
    return "\n".join(lines) + "\n"


def run_all() -> None:
    _log("START full SocialMemBench run; safe to rerun after any interruption")
    prepare_memory()
    run_inference()
    score()
    _log("DONE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group(required=True)
    phases.add_argument("--all", action="store_true", help="prepare memory, infer all QA, score, report")
    phases.add_argument("--prepare-memory", action="store_true")
    phases.add_argument("--run-inference", action="store_true")
    phases.add_argument("--score", action="store_true")
    args = parser.parse_args()
    try:
        _preflight()
        if args.all:
            run_all()
        elif args.prepare_memory:
            prepare_memory()
        elif args.run_inference:
            run_inference()
        else:
            score()
    except RateLimitError as error:
        raise SystemExit(
            "OpenAI rate limit reached. Progress is cached; rerun the same command "
            f"after the limit resets or your usage tier increases.\n{error}"
        ) from None


if __name__ == "__main__":
    main()
