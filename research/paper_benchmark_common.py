"""Small shared runtime for the two external TemporalEpisode benchmarks."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import sqlite3
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd
from openai import OpenAI

from research import query_independent_episode_pipeline as qi
from research import temporal_episode_prototype as tep


ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_MODEL = "gpt-4o-mini-2024-07-18"
EMBED_MODEL = "text-embedding-3-large"
TOP_K = 10
BOOTSTRAP_SEED = 20260912
BOOTSTRAP_ITERATIONS = 20_000
PRICE_SNAPSHOT_DATE = "2026-09-12"
# Standard uncached list prices in USD per million tokens.  Deliberately
# conservative: prompt-cache discounts are not subtracted from the estimate.
MODEL_PRICES_USD_PER_M = {
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "text-embedding-3-large": (0.13, 0.0),
    "gpt-5": (1.25, 10.0),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "google/gemini-3-flash-preview": (0.50, 3.00),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()


def seal_jsonl(
    path: Path, metadata_path: Path, *, expected_rows: int,
    identity_fields: tuple[str, ...], metadata: dict[str, Any],
) -> dict[str, Any]:
    """Checksum-lock a complete append-only artifact and reject duplicates."""
    rows = jsonl(path)
    if len(rows) != expected_rows:
        raise RuntimeError(f"incomplete {path.name}: {len(rows)}/{expected_rows}")
    identities = [tuple(row.get(field) for field in identity_fields) for row in rows]
    if len(set(identities)) != len(identities):
        raise RuntimeError(f"duplicate identities in {path.name}")
    sealed = {
        **metadata, "row_count": len(rows), "identity_fields": list(identity_fields),
        "sha256": sha256_file(path),
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing != sealed:
            raise RuntimeError(f"sealed metadata mismatch: {metadata_path}")
    else:
        metadata_path.write_text(json.dumps(sealed, ensure_ascii=False, indent=2) + "\n")
    return sealed


def verify_sealed_jsonl(path: Path, metadata_path: Path, *, expected_rows: int) -> dict[str, Any]:
    if not metadata_path.exists():
        raise RuntimeError(f"unsealed artifact: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("row_count") != expected_rows:
        raise RuntimeError(f"sealed row count mismatch: {metadata.get('row_count')} != {expected_rows}")
    if sha256_file(path) != metadata.get("sha256"):
        raise RuntimeError(f"sealed checksum mismatch: {path}")
    return metadata


def log(run_dir: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.log").open("a") as output:
        output.write(line + "\n")


def ensure_git_snapshot(path: Path, url: str, commit: str) -> Path:
    """Fetch exactly one public revision; never silently move with upstream."""
    if path.exists():
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, capture_output=True, check=True,
        ).stdout.strip()
        if actual != commit:
            raise RuntimeError(f"official source mismatch at {path}: {actual} != {commit}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "remote", "add", "origin", url], cwd=path, check=True)
    subprocess.run(["git", "fetch", "--depth", "1", "origin", commit], cwd=path, check=True)
    subprocess.run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=path, check=True)
    return path


class CachedAPI:
    """Content-addressed chat cache plus actual token/latency accounting."""

    def __init__(self, run_dir: Path, *, api_key: str, base_url: str | None = None, label: str = "openai"):
        self.run_dir = run_dir
        self.cache_path = run_dir / f"{label}_chat_cache.jsonl"
        self.usage_path = run_dir / "api_usage.jsonl"
        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 5, "timeout": 300.0}
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        self.client = OpenAI(**kwargs)
        self.label = label
        self.lock = threading.Lock()
        self.values = {row["key"]: row["value"] for row in jsonl(self.cache_path)}

    def chat(
        self, *, model: str, messages: list[dict], temperature: float,
        max_tokens: int, json_output: bool = False, phase: str,
        extra_body: dict | None = None,
    ) -> str:
        payload = {
            "provider": self.label, "model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "json": json_output,
            "extra_body": extra_body,
        }
        key = "chat:" + sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
        with self.lock:
            cached = self.values.get(key)
        if cached is not None:
            return str(cached)
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if model.lower().startswith(("gpt-5", "o1", "o3", "o4")):
            kwargs["max_completion_tokens"] = max(max_tokens * 4, 2000)
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}
        if extra_body:
            kwargs["extra_body"] = extra_body
        started = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        duration_ms = (time.perf_counter() - started) * 1000
        value = response.choices[0].message.content or ""
        usage = response.usage
        with self.lock:
            if key not in self.values:
                self.values[key] = value
                append_jsonl(self.cache_path, {"key": key, "value": value})
                append_jsonl(self.usage_path, {
                    "key": key, "provider": self.label, "phase": phase, "model": model,
                    "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "duration_ms": duration_ms,
                })
        return value


class EmbeddingCache:
    def __init__(self, run_dir: Path, api: CachedAPI, model: str = EMBED_MODEL):
        self.directory = run_dir / "embeddings"
        self.usage_path = run_dir / "api_usage.jsonl"
        self.api = api
        self.model = model

    def batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float64)
        key = sha256_bytes(json.dumps({"model": self.model, "texts": texts}, ensure_ascii=False).encode())
        path = self.directory / f"batch-{key[:24]}.npy"
        if path.exists():
            return np.load(path)
        started = time.perf_counter()
        response = self.api.client.embeddings.create(model=self.model, input=texts)
        duration_ms = (time.perf_counter() - started) * 1000
        matrix = np.asarray([row.embedding for row in sorted(response.data, key=lambda x: x.index)], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
        append_jsonl(self.usage_path, {
            "key": "embedding:" + key, "provider": self.api.label, "phase": "retrieval",
            "model": self.model, "input_tokens": int(response.usage.total_tokens or 0),
            "output_tokens": 0, "duration_ms": duration_ms,
        })
        return matrix

    def all(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        chunks = [self.batch(texts[start:start + batch_size]) for start in range(0, len(texts), batch_size)]
        return np.vstack(chunks) if chunks else np.empty((0, 0), dtype=np.float64)


@dataclass(frozen=True)
class SearchDoc:
    doc_id: str
    index_text: str
    rendered: str
    source_ids: tuple[str, ...]
    kind: str


class DenseIndex:
    def __init__(self, docs: list[SearchDoc], embeddings: EmbeddingCache, label: str):
        self.docs = docs
        digest = sha256_bytes("\n".join(f"{d.doc_id}\t{d.index_text}" for d in docs).encode())[:20]
        path = embeddings.directory / f"index-{label}-{digest}.npy"
        if path.exists():
            self.matrix = np.load(path)
        else:
            self.matrix = embeddings.all([doc.index_text for doc in docs])
            embeddings.directory.mkdir(parents=True, exist_ok=True)
            np.save(path, self.matrix)
        self.embeddings = embeddings

    def search(self, query: str, k: int = TOP_K) -> tuple[list[SearchDoc], float]:
        vector = self.embeddings.batch([query])[0]
        return self.search_vector(vector, k)

    def search_vector(self, vector: np.ndarray, k: int = TOP_K) -> tuple[list[SearchDoc], float]:
        started = time.perf_counter()
        scores = np.sum(self.matrix * vector[None, :], axis=1, dtype=np.float64)
        order = np.argsort(-scores)[:k]
        return [self.docs[int(i)] for i in order], (time.perf_counter() - started) * 1000


def episode_row(episode: tep.TemporalEpisode) -> dict:
    return {
        "episode_id": episode.episode_id, "network_id": episode.network_id,
        "viewpoint_owner": episode.viewpoint_owner, "subject": episode.subject,
        "events": [{"decision": item.decision, "event": asdict(item.event)} for item in episode.events],
    }


def episode_doc(row: dict, raw_by_id: dict[str, SearchDoc]) -> SearchDoc:
    active = row["events"][-tep.MAX_HOT_EVENTS:]
    all_source_ids = tuple(dict.fromkeys(
        source_id for item in row["events"] for source_id in item["event"]["source_turn_ids"]
    ))
    active_text = " || ".join(item["event"]["event_text"] for item in active)
    owner = row.get("viewpoint_owner") or "?"
    subject = row.get("subject") or "?"
    index_text = f"Viewpoint owner: {owner}. Subject: {subject}. {active_text}"
    lines = [f"[DERIVED TEMPORAL EPISODE / owner={owner} / subject={subject}]", active_text]
    for source_id in all_source_ids:
        source = raw_by_id.get(source_id)
        if source:
            lines.append(f"  [SOURCE {source_id}] {source.rendered}")
    return SearchDoc(
        doc_id=f"episode:{row['network_id']}:{row['episode_id']}", index_text=index_text,
        rendered="\n".join(lines), source_ids=all_source_ids, kind="episode",
    )


def build_episode_snapshot(
    frame: pd.DataFrame, run_dir: Path, api: CachedAPI, *, source_manifest: dict,
) -> Path:
    """Build and seal query-independent memory. This function never accepts QA."""
    snapshot = run_dir / "episodes.jsonl"
    metadata = run_dir / "episodes_meta.json"
    if snapshot.exists() and metadata.exists():
        verify_episode_snapshot(run_dir, source_manifest)
        return snapshot

    required = {"network_id", "session_index", "turn_id", "timestamp", "speaker_display_name", "message"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"episode input missing columns: {sorted(missing)}")
    if frame[["network_id", "turn_id"]].duplicated().any():
        raise ValueError("turn_id must be unique inside each network")

    originals = {
        "qi_event": qi.QI_EVENT_CACHE, "qi_attach": qi.QI_ATTACH_CACHE,
        "tep_event": tep.EVENT_CACHE, "tep_attach": tep.ATTACH_CACHE,
        "embed": tep.embed, "embed_batch": tep.embed_batch,
        "context": tep.build_group_context,
    }
    event_cache = run_dir / "event_extraction_cache.jsonl"
    attach_cache = run_dir / "episode_attach_cache.jsonl"
    qi.QI_EVENT_CACHE = event_cache
    qi.QI_ATTACH_CACHE = attach_cache
    tep.EVENT_CACHE = event_cache
    tep.ATTACH_CACHE = attach_cache

    # Same local embedding model and scores as the prototype, but memoized.
    # Without this, the prototype re-encodes every unchanged episode after
    # every event (quadratic work on 120k-message corpora).
    import embeddings as local_embeddings
    database = sqlite3.connect(run_dir / "local_embedding_cache.sqlite3")
    database.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
    raw_embed_batch = local_embeddings.embed_batch

    def cached_local_batch(texts: list[str]) -> list[list[float]]:
        keys = [sha256_bytes(text.encode()) for text in texts]
        found = {
            key: np.frombuffer(value, dtype=np.float32)
            for key, value in database.execute(
                f"SELECT key,value FROM vectors WHERE key IN ({','.join('?' for _ in keys)})", keys
            )
        } if keys else {}
        missing_texts = [text for text, key in zip(texts, keys) if key not in found]
        missing_keys = [key for key in keys if key not in found]
        if missing_texts:
            vectors = np.asarray(raw_embed_batch(missing_texts), dtype=np.float32)
            database.executemany("INSERT OR IGNORE INTO vectors(key,value) VALUES (?,?)", [
                (key, vector.tobytes()) for key, vector in zip(missing_keys, vectors)
            ])
            database.commit()
            found.update(dict(zip(missing_keys, vectors)))
        return [found[key].tolist() for key in keys]

    tep.embed_batch = cached_local_batch
    tep.embed = lambda text: cached_local_batch([text])[0]

    # Query-independent clusters always contain the complete bounded session;
    # render that session directly instead of rebuilding +/-2 windows once per
    # anchor turn (identical output, linear rather than quadratic preparation).
    session_frames = {
        (str(network), int(session)): group.sort_values(["timestamp", "turn_id"])
        for (network, session), group in frame.groupby(["network_id", "session_index"], sort=False)
    }
    turn_to_session = {
        (str(row.network_id), str(row.turn_id)): int(row.session_index)
        for row in frame.itertuples(index=False)
    }

    def full_session_context(_frame: pd.DataFrame, network: str, anchor_ids: set[str]):
        sessions = {turn_to_session[(str(network), str(turn))] for turn in anchor_ids}
        if len(sessions) != 1:
            raise ValueError("query-independent extraction cluster must be exactly one full session")
        selected = session_frames[(str(network), sessions.pop())]
        actual = set(selected.turn_id.astype(str))
        if actual != {str(turn) for turn in anchor_ids}:
            raise ValueError("query-independent extraction cluster is not the complete session")
        lines, message_by_turn = [], {}
        for row in selected.itertuples(index=False):
            turn_id = str(row.turn_id)
            message = str(row.message)
            lines.append(f"[[{turn_id}]] [ANCHOR] {row.timestamp} {row.speaker_display_name}: {message}")
            message_by_turn[turn_id] = message
        return "\n".join(lines), actual, message_by_turn

    tep.build_group_context = full_session_context

    class _Model:
        def invoke(self, prompt: str) -> SimpleNamespace:
            value = api.chat(
                model=EXTRACTION_MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=8192, json_output=True, phase="memory",
            )
            return SimpleNamespace(content=value)

    import llm.groq_client as provider
    original_provider = provider.get_chat_model
    provider.get_chat_model = lambda *_args, **_kwargs: _Model()

    networks = tuple(sorted(frame.network_id.astype(str).unique()))
    try:
        episodes, _events, decisions, stats = qi.run_qi_extraction_and_materialize(networks, frame)
    finally:
        qi.QI_EVENT_CACHE, qi.QI_ATTACH_CACHE = originals["qi_event"], originals["qi_attach"]
        tep.EVENT_CACHE, tep.ATTACH_CACHE = originals["tep_event"], originals["tep_attach"]
        tep.embed, tep.embed_batch = originals["embed"], originals["embed_batch"]
        tep.build_group_context = originals["context"]
        provider.get_chat_model = original_provider
        database.close()
    rows = [episode_row(ep) for network in networks for ep in episodes.get(network, [])]
    payload = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n").encode()
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(payload)
    (run_dir / "attach_decisions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in decisions) + "\n"
    )
    metadata.write_text(json.dumps({
        "schema_version": "paper_external_temporal_episodes_v1",
        "memory_schema_version": tep.SCHEMA_VERSION,
        "extraction_and_attach_model": EXTRACTION_MODEL,
        "network_count": len(networks),
        "session_count": int(frame[["network_id", "session_index"]].drop_duplicates().shape[0]),
        "turn_count": len(frame), "episode_count": len(rows), "stats": stats,
        "episodes_sha256": sha256_bytes(payload), "qa_read": False,
        "source_manifest": source_manifest,
    }, ensure_ascii=False, indent=2) + "\n")
    return snapshot


def verify_episode_snapshot(run_dir: Path, source_manifest: dict) -> dict[str, Any]:
    snapshot, metadata = run_dir / "episodes.jsonl", run_dir / "episodes_meta.json"
    if not snapshot.exists() or not metadata.exists():
        raise RuntimeError("sealed episode snapshot missing")
    meta = json.loads(metadata.read_text())
    if sha256_file(snapshot) != meta.get("episodes_sha256"):
        raise RuntimeError("sealed episode snapshot checksum mismatch")
    if meta.get("source_manifest") != source_manifest:
        raise RuntimeError("sealed episode snapshot source manifest mismatch")
    return meta


def exact_mcnemar(rows: list[dict], left: str, right: str) -> dict[str, Any]:
    paired: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        paired[row["qa_key"]][row["condition"]] = int(row["correct"])
    pairs = [value for value in paired.values() if left in value and right in value]
    left_only = sum(value[left] == 1 and value[right] == 0 for value in pairs)
    right_only = sum(value[left] == 0 and value[right] == 1 for value in pairs)
    discordant = left_only + right_only
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, i) for i in range(0, min(left_only, right_only) + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)
    return {"n": len(pairs), "left_only": left_only, "right_only": right_only, "p_value": p_value}


def paired_bootstrap(rows: list[dict], left: str, right: str) -> tuple[float, float, float]:
    paired: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        paired[row["qa_key"]][row["condition"]] = float(row["correct"])
    deltas = np.asarray([
        value[left] - value[right] for value in paired.values() if left in value and right in value
    ], dtype=np.float64)
    if not len(deltas):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(deltas, size=(BOOTSTRAP_ITERATIONS, len(deltas)), replace=True).mean(axis=1)
    return float(deltas.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def clustered_bootstrap(
    rows: list[dict], left: str, right: str, cluster_field: str,
) -> tuple[float, float, float]:
    paired: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        paired[row["qa_key"]][row["condition"]] = row
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for value in paired.values():
        if left in value and right in value:
            by_cluster[str(value[left][cluster_field])].append(
                float(value[left]["correct"]) - float(value[right]["correct"])
            )
    cluster_means = np.asarray([np.mean(values) for values in by_cluster.values()], dtype=np.float64)
    if not len(cluster_means):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(
        cluster_means, size=(BOOTSTRAP_ITERATIONS, len(cluster_means)), replace=True,
    ).mean(axis=1)
    return float(cluster_means.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def usage_summary(run_dir: Path) -> dict[str, Any]:
    rows = jsonl(run_dir / "api_usage.jsonl")
    by_model: dict[str, dict[str, float]] = defaultdict(lambda: {
        "calls": 0, "input_tokens": 0, "output_tokens": 0, "duration_ms": 0,
    })
    for row in rows:
        bucket = by_model[row["model"]]
        bucket["calls"] += 1
        bucket["input_tokens"] += int(row.get("input_tokens", 0))
        bucket["output_tokens"] += int(row.get("output_tokens", 0))
        bucket["duration_ms"] += float(row.get("duration_ms", 0))
    total_usd = 0.0
    for model, bucket in by_model.items():
        prices = MODEL_PRICES_USD_PER_M.get(model)
        if prices:
            bucket["estimated_usd"] = (
                bucket["input_tokens"] * prices[0] + bucket["output_tokens"] * prices[1]
            ) / 1_000_000
            total_usd += bucket["estimated_usd"]
        else:
            bucket["estimated_usd"] = None
    return {
        "api_calls": len(rows), "by_model": dict(by_model),
        "estimated_usd_upper_bound": total_usd,
        "price_snapshot_date": PRICE_SNAPSHOT_DATE,
        "pricing_assumption": "standard uncached list rates; excludes local compute",
    }


def latency_summary(predictions: list[dict]) -> dict[str, dict[str, float]]:
    result = {}
    for condition in sorted({str(row["condition"]) for row in predictions}):
        values = np.asarray([
            float(row.get("retrieval_ms", 0)) for row in predictions if row["condition"] == condition
        ])
        result[condition] = {
            "n": int(len(values)), "mean_ms": float(values.mean()),
            "p50_ms": float(np.quantile(values, .5)), "p95_ms": float(np.quantile(values, .95)),
        }
    return result


def freeze_run(run_dir: Path, source_files: Iterable[Path], related_dirs: Iterable[Path] = ()) -> Path:
    """Make a checksum-locked preservation copy after a completed run."""
    if not (run_dir / "report.md").exists():
        raise RuntimeError("cannot freeze an incomplete run (report.md missing)")
    stamp = time.strftime("%Y%m%d_%H%M%S_MSK")
    destination = ROOT / ".research_runs" / "frozen" / f"{run_dir.name}_{stamp}"
    if destination.exists():
        raise RuntimeError(f"freeze already exists: {destination}")
    shutil.copytree(run_dir, destination / "run")
    for related in related_dirs:
        shutil.copytree(related, destination / "related" / related.name)
    source_dir = destination / "source"
    source_dir.mkdir(parents=True)
    for source in source_files:
        shutil.copy2(source, source_dir / source.name)
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_run": str(run_dir),
        "files": {str(path.relative_to(destination)): {
            "size": path.stat().st_size, "sha256": sha256_file(path),
        } for path in files},
    }
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "SHA256SUMS").write_text("".join(
        f"{value['sha256']}  {name}\n" for name, value in manifest["files"].items()
    ))
    for path in sorted(destination.rglob("*"), reverse=True):
        mode = stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH
        if path.is_dir():
            mode |= stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
        os.chmod(path, mode)
    os.chmod(destination, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH |
             stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return destination
