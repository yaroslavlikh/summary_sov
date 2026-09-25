"""SIMPLE-PROP on SocialMemBench: independent replication of the schema-versus-paraphrase
decomposition measured on EverMemBench (research/simple_prop_baseline.py).

Same ablation, second benchmark, different extractor generation for the existing flat and
versioned baselines -- this run builds SIMPLE-PROP with the *current* extractor, so for the
first time on SocialMemBench the paraphrase and the event schema are produced by the same
pipeline and differ only in schema.

Reuses the EverMemBench SIMPLE-PROP prompt and validator unchanged (same self-contained
sentence, same literal-quote contract, same two-items-per-turn cap, no owner/subject/type/
temporal-mode, no episode linking). Retrieval follows the SocialMemBench harness rather than
the EverMemBench one: text-embedding-3-small, one cosine top-10 over the condition's corpus.
Gold is `evidence_anchors_json` turn ids, exactly as the sealed Social scoring reads it.

RAW and EVENTS are recomputed here through the same code path as the new PROP condition, so a
bug in this script shows up as a disagreement with the paper's published Social delivery
numbers before any new number is trusted.

    python3 -m research.simple_prop_social --preflight   # offline
    python3 -m research.simple_prop_social --run          # paid: extraction + embeddings
    python3 -m research.simple_prop_social --report
    python3 -m research.simple_prop_social --freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from research import socialmembench_full_run as official
from research import temporal_episode_prototype as tep
from research.simple_prop_baseline import (
    MAX_ITEMS_PER_TURN,
    PropUnit,
    _enforce_max_per_turn,
    _prec_rec,
    _prop_extraction_prompt,
    _session_context,
    validate_prop_unit,
)
from research.paper_benchmark_common import (
    CachedAPI,
    EmbeddingCache,
    MODEL_PRICES_USD_PER_M,
    SearchDoc,
    append_jsonl,
    freeze_run,
    jsonl,
    log,
    sha256_file,
    usage_summary,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/tmp/socialmembench")
RUN_DIR = ROOT / ".research_runs" / "simple_prop_social_v2"
INDEX_DIR = ROOT / ".research_runs" / "simple_prop_social_index_v2"
PROTOCOL = ROOT / "research" / "SIMPLE_PROP_BASELINE_PROTOCOL.md"
RUNNER = Path(__file__).resolve()
SEALED_EPISODES = (ROOT / ".research_runs" / "frozen"
                   / "socialmembench_temporal_episodes_official_harness_v1_20260912_122626_MSK"
                   / "related" / "socialmembench_temporal_episodes_full_v1" / "episodes.jsonl")
EXTRACTION_MODEL = official.MODEL
EMBED_MODEL = official.EMBED_MODEL
TOP_K = official.TOP_K
CEILING_USD = 1.50
SEED = 20260917
BOOTSTRAP = 10_000
WORKERS = 10
# Published Social delivery numbers this run must reproduce before anything new is trusted
# (manuscript §6.2: precision deltas against RAW, network bootstrap).
EXPECTED_EVENTS_MINUS_RAW_PRECISION_PP = 0.34  # RAW+EPISODES - RAW, as published


def load_social() -> tuple[pd.DataFrame, pd.DataFrame]:
    conversations = pd.read_parquet(DATA_DIR / "conversations.parquet")
    qa = pd.read_parquet(DATA_DIR / "qa.parquet")
    return conversations, qa


def raw_docs_by_network(conversations: pd.DataFrame) -> tuple[dict[str, list[SearchDoc]], dict[str, SearchDoc]]:
    by_network: dict[str, list[SearchDoc]] = defaultdict(list)
    by_id: dict[str, SearchDoc] = {}
    for row in conversations.itertuples(index=False):
        turn_id = str(row.turn_id)
        body = str(row.message)
        rendered = f"[{row.timestamp_label}][{row.speaker_display_name}] {body}"
        doc = SearchDoc(f"raw:{turn_id}", body, rendered, (turn_id,), "raw")
        by_network[str(row.network_id)].append(doc)
        by_id[turn_id] = doc
    return dict(by_network), by_id


def event_docs_by_network(raw_by_id: dict[str, SearchDoc]) -> dict[str, list[SearchDoc]]:
    """Every sealed Social event as its own document, in the same format the EverMemBench
    unlinked-events control uses."""
    result: dict[str, list[SearchDoc]] = defaultdict(list)
    for row in jsonl(SEALED_EPISODES):
        for item in row["events"]:
            event = item["event"]
            owner = event.get("viewpoint_owner") or "?"
            subject = event.get("subject") or "?"
            sources = tuple(dict.fromkeys(event["source_turn_ids"]))
            lines = [f"[DERIVED EVENT / owner={owner} / subject={subject}]", event["event_text"]]
            for source_id in sources:
                source = raw_by_id.get(source_id)
                if source:
                    lines.append(f"  [SOURCE {source_id}] {source.rendered}")
            result[str(event["network_id"])].append(SearchDoc(
                doc_id=f"event:{event['network_id']}:{event['event_id']}",
                index_text=f"Viewpoint owner: {owner}. Subject: {subject}. {event['event_text']}",
                rendered="\n".join(lines), source_ids=sources, kind="event",
            ))
    return dict(result)


def prop_doc(row: dict, raw_by_id: dict[str, SearchDoc]) -> SearchDoc:
    sources = tuple(dict.fromkeys(row["source_turn_ids"]))
    lines = ["[DERIVED PROPOSITION]", row["text"]]
    for source_id in sources:
        source = raw_by_id.get(source_id)
        if source:
            lines.append(f"  [SOURCE {source_id}] {source.rendered}")
    return SearchDoc(
        doc_id=f"prop:{row['network_id']}:{row['unit_id']}", index_text=row["text"],
        rendered="\n".join(lines), source_ids=sources, kind="prop",
    )


def build_session_prompts(conversations: pd.DataFrame) -> dict[str, list]:
    session_frames = {
        (str(network), int(session)): group.sort_values(["timestamp", "turn_id"])
        for (network, session), group in conversations.groupby(["network_id", "session_index"], sort=False)
    }
    result: dict[str, list] = defaultdict(list)
    for (network, session_index), frame in session_frames.items():
        anchor_ids = set(frame.turn_id.astype(str))
        context_lines, window_ids, message_by_turn = _session_context(frame, anchor_ids)
        result[network].append((session_index, window_ids, _prop_extraction_prompt(context_lines), message_by_turn))
    return dict(result)


def preflight() -> dict:
    conversations, qa = load_social()
    prompts = build_session_prompts(conversations)
    import tiktoken
    encoding = tiktoken.get_encoding("o200k_base")
    sessions = sum(len(v) for v in prompts.values())
    input_tokens = sum(len(encoding.encode(p)) for entries in prompts.values() for _, _, p, _ in entries)
    price_in, price_out = MODEL_PRICES_USD_PER_M[EXTRACTION_MODEL]
    assumed_output = 450
    extraction_usd = (input_tokens * price_in + sessions * assumed_output * price_out) / 1_000_000
    report = {
        "networks": len(prompts), "sessions": sessions, "messages": len(conversations),
        "questions": len(qa), "input_tokens": input_tokens,
        "extraction_expected_usd": round(extraction_usd, 4),
        "embedding_model": EMBED_MODEL, "ceiling_usd": CEILING_USD,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _extract_one(api: CachedAPI, network: str, session_index, window_ids, prompt, message_by_turn):
    output = api.chat(model=EXTRACTION_MODEL, messages=[{"role": "user", "content": prompt}],
                      temperature=0, max_tokens=8192, json_output=True, phase="extraction")
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", output, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
    items = [item for item in (parsed.get("items") or []) if isinstance(item, dict)]
    units, rejections = [], []
    attempted = valid = 0
    for local_index, item in enumerate(items):
        unit, reason, a, v = validate_prop_unit(
            item, window_ids, message_by_turn, network, int(session_index) * 100 + local_index)
        attempted += a
        valid += v
        (units if unit is not None else rejections).append(unit if unit is not None else reason)
    kept, dropped = _enforce_max_per_turn([u for u in units if u is not None])
    return kept, rejections, dropped, attempted, valid


def run_extraction() -> None:
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY отсутствует в .env")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    api = CachedAPI(RUN_DIR, api_key=key, label="openai")
    conversations, _qa = load_social()
    prompts = build_session_prompts(conversations)
    props_path, rejected_path = RUN_DIR / "props.jsonl", RUN_DIR / "rejected.jsonl"
    done = {row["network_id"] for row in jsonl(RUN_DIR / "networks_done.jsonl")} if (RUN_DIR / "networks_done.jsonl").exists() else set()
    lock = threading.Lock()

    for network, entries in prompts.items():
        if network in done:
            continue
        spend = usage_summary(RUN_DIR)["estimated_usd_upper_bound"]
        if spend > CEILING_USD:
            raise SystemExit(f"stopping before {network}: spend ${spend:.2f} over ceiling ${CEILING_USD:.2f}")
        stats = Counter()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(_extract_one, api, network, *entry) for entry in entries]
            for future in as_completed(futures):
                kept, rejections, dropped, attempted, valid = future.result()
                stats["evidence_attempted"] += attempted
                stats["evidence_valid"] += valid
                stats["rejected_exceeds_max_per_turn"] += dropped
                with lock:
                    for reason in rejections:
                        stats[f"rejected_{reason}"] += 1
                        append_jsonl(rejected_path, {"network_id": network, "reason": reason})
                    for unit in kept:
                        append_jsonl(props_path, {**asdict(unit), "unit_id": unit.unit_id, "network_id": unit.network_id})
                        stats["units_kept"] += 1
        append_jsonl(RUN_DIR / "networks_done.jsonl", {"network_id": network, "stats": dict(stats)})
    log(RUN_DIR, f"extraction done, spend ${usage_summary(RUN_DIR)['estimated_usd_upper_bound']:.4f}")


def build_index() -> dict:
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    conversations, _qa = load_social()
    _raw_by_network, raw_by_id = raw_docs_by_network(conversations)
    props: dict[str, list[SearchDoc]] = defaultdict(list)
    for row in jsonl(RUN_DIR / "props.jsonl"):
        props[row["network_id"]].append(prop_doc(row, raw_by_id))

    direct = CachedAPI(INDEX_DIR, api_key=key, label="openai")
    embeddings = EmbeddingCache(INDEX_DIR, direct, model=EMBED_MODEL)
    meta: dict = {"embedding_model": EMBED_MODEL, "networks": {}}
    for network in sorted(props):
        docs = props[network]
        digest = hashlib.sha256("\n".join(f"{d.doc_id}\t{d.index_text}" for d in docs).encode()).hexdigest()
        path = INDEX_DIR / f"props-{network}-{digest[:16]}.npy"
        if not path.exists():
            np.save(path, embeddings.all([doc.index_text for doc in docs]))
        meta["networks"][network] = {"docs": len(docs), "docs_sha256": digest,
                                     "matrix_file": path.name, "matrix_sha256": sha256_file(path)}
    (INDEX_DIR / "index_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(INDEX_DIR, f"prop index sealed: {sum(len(v) for v in props.values())} propositions")
    return meta


def _embed_texts(texts: list[str], directory: Path, label: str) -> np.ndarray:
    """Embeddings for RAW/EVENTS corpora and questions, cached by content in `directory`."""
    load_dotenv(ROOT / ".env")
    api = CachedAPI(directory, api_key=os.environ["OPENAI_API_KEY"], label="openai")
    cache = EmbeddingCache(directory, api, model=EMBED_MODEL)
    return cache.all(texts)


def evaluate() -> list[dict]:
    conversations, qa = load_social()
    raw_by_network, raw_by_id = raw_docs_by_network(conversations)
    events = event_docs_by_network(raw_by_id)
    props: dict[str, list[SearchDoc]] = defaultdict(list)
    for row in jsonl(RUN_DIR / "props.jsonl"):
        props[row["network_id"]].append(prop_doc(row, raw_by_id))

    rows = []
    for network in sorted(raw_by_network):
        network_qa = qa[qa.network_id == network]
        if not len(network_qa):
            continue
        questions = [str(r.question) for r in network_qa.itertuples(index=False)]
        q_vectors = _embed_texts(questions, INDEX_DIR, "questions")
        raw_docs = raw_by_network[network]
        raw_matrix = _embed_texts([d.index_text for d in raw_docs], INDEX_DIR, "raw")
        prop_docs_net = props.get(network, [])
        event_docs_net = events.get(network, [])
        prop_matrix = _embed_texts([d.index_text for d in prop_docs_net], INDEX_DIR, "prop") if prop_docs_net else np.empty((0, raw_matrix.shape[1]))
        event_matrix = _embed_texts([d.index_text for d in event_docs_net], INDEX_DIR, "event") if event_docs_net else np.empty((0, raw_matrix.shape[1]))

        unified_prop_docs = raw_docs + prop_docs_net
        unified_prop_matrix = np.vstack([raw_matrix, prop_matrix]) if len(prop_matrix) else raw_matrix
        unified_event_docs = raw_docs + event_docs_net
        unified_event_matrix = np.vstack([raw_matrix, event_matrix]) if len(event_matrix) else raw_matrix

        for qrow, vector in zip(network_qa.itertuples(index=False), q_vectors):
            gold = {str(a["turn_id"]) for a in official._json(qrow.evidence_anchors_json, []) if a.get("turn_id")}

            def top_sources(matrix, docs):
                scores = np.sum(matrix * vector[None, :], axis=1, dtype=np.float64)
                order = np.argsort(-scores)[:TOP_K]
                chosen = [docs[int(i)] for i in order]
                return set(s for doc in chosen for s in doc.source_ids)

            raw_p, raw_r = _prec_rec(top_sources(raw_matrix, raw_docs), gold)
            prop_p, prop_r = _prec_rec(top_sources(unified_prop_matrix, unified_prop_docs), gold)
            ev_p, ev_r = _prec_rec(top_sources(unified_event_matrix, unified_event_docs), gold)
            rows.append({
                "qa_key": f"{network}:{qrow.qa_id}", "network_id": network, "query_type": str(qrow.query_type),
                "raw_precision": raw_p, "raw_recall": raw_r, "prop_precision": prop_p, "prop_recall": prop_r,
                "events_precision": ev_p, "events_recall": ev_r,
            })
    return rows


def _network_bootstrap(rows: list[dict], field_a: str, field_b: str) -> dict:
    """Resample the 43 conversation networks, matching the manuscript's Social statistics."""
    by_network: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_network[r["network_id"]].append(100 * (r[field_a] - r[field_b]))
    means = np.asarray([np.mean(v) for v in by_network.values()], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    samples = rng.choice(means, size=(BOOTSTRAP, len(means)), replace=True).mean(axis=1)
    return {"delta_pp": float(means.mean()), "ci": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
            "networks": len(means), "positive_networks": int((means > 0).sum())}


def render_report() -> Path:
    rows = evaluate()

    def mean(field: str) -> float:
        return statistics.mean(r[field] for r in rows) * 100

    means = {
        "RAW": {"precision": mean("raw_precision"), "recall": mean("raw_recall")},
        "SIMPLE-PROP": {"precision": mean("prop_precision"), "recall": mean("prop_recall")},
        "EVENTS": {"precision": mean("events_precision"), "recall": mean("events_recall")},
    }
    comparisons = {
        "PROP - RAW (precision)": _network_bootstrap(rows, "prop_precision", "raw_precision"),
        "PROP - RAW (recall)": _network_bootstrap(rows, "prop_recall", "raw_recall"),
        "EVENTS - PROP (precision)": _network_bootstrap(rows, "events_precision", "prop_precision"),
        "EVENTS - PROP (recall)": _network_bootstrap(rows, "events_recall", "prop_recall"),
        "EVENTS - RAW (precision)": _network_bootstrap(rows, "events_precision", "raw_precision"),
    }
    total_units = sum(1 for _ in jsonl(RUN_DIR / "props.jsonl"))
    rejected = Counter(row["reason"] for row in jsonl(RUN_DIR / "rejected.jsonl"))
    summary = {"n_questions": len(rows), "means_percent": means, "comparisons_pp": comparisons,
               "prop_units_total": total_units, "rejected_reasons": dict(rejected),
               "cost": usage_summary(RUN_DIR)}
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    lines = ["# SIMPLE-PROP on SocialMemBench: schema vs self-contained paraphrase", "",
             f"- protocol sha256 `{sha256_file(PROTOCOL)}`; runner sha256 `{sha256_file(RUNNER)}`",
             f"- questions: {len(rows)}; SIMPLE-PROP units: {total_units}; embeddings: `{EMBED_MODEL}`", "",
             "## Means (%)", "", "| condition | precision | recall |", "|---|---:|---:|"]
    for cond, v in means.items():
        lines.append(f"| {cond} | {v['precision']:.2f} | {v['recall']:.2f} |")
    lines += ["", "## Paired comparisons (network bootstrap, pp)", "",
              "| comparison | delta | 95% CI | networks + |", "|---|---:|---|---:|"]
    for name, c in comparisons.items():
        lines.append(f"| {name} | {c['delta_pp']:+.2f} | [{c['ci'][0]:+.2f}; {c['ci'][1]:+.2f}] | "
                     f"{c['positive_networks']}/{c['networks']} |")
    lines += ["", "## Rejected items", "", f"{dict(rejected)}", "", "## Cost", "",
              f"- {usage_summary(RUN_DIR)['estimated_usd_upper_bound']:.4f} USD"]
    path = RUN_DIR / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(), indent=2))
    elif args.run:
        run_extraction()
        build_index()
        print(f"done, spend ${usage_summary(RUN_DIR)['estimated_usd_upper_bound']:.4f}")
    elif args.report:
        print(render_report())
    elif args.freeze:
        print(freeze_run(RUN_DIR, [RUNNER, PROTOCOL], related_dirs=[INDEX_DIR]))


if __name__ == "__main__":
    main()
