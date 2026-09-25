"""SIMPLE-PROP baseline: does the +3.21pp EverMemBench evidence-precision result survive
against a self-contained paraphrase with identical provenance, or is it explained by
self-containment alone?

Protocol: research/SIMPLE_PROP_BASELINE_PROTOCOL.md. Nothing here calls an API before --run.
Addresses the manuscript's own §10 gap, named independently by three reviews.

Same query-independent session clustering, model, call budget (at most 2 items/turn), and
quote-validation contract as the sealed EVENTS extraction; the only difference is the prompt
and the document schema -- one self-contained sentence, no owner/subject/type/temporal-mode
fields. Retrieval and evidence-delivery computation reuse the sealed RAW matrix and the sealed
per-question embedding cache at zero cost; only the new SIMPLE-PROP documents are embedded.
The RAW and EVENTS sides of the comparison are recomputed through the exact same ranking code
as the new PROP side and checked against the paper's own published numbers before anything
new is trusted.

    python3 -m research.simple_prop_baseline --preflight   # offline
    python3 -m research.simple_prop_baseline --run          # paid: extraction + embeddings
    python3 -m research.simple_prop_baseline --report
    python3 -m research.simple_prop_baseline --freeze
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from research import evermembench_episode_run as base
from research import evermembench_temporal_budget_control as control
from research import evermembench_unlinked_events_control as unlinked
from research import final_sprint_evermembench as fs
from research import query_independent_episode_pipeline as qi
from research import temporal_episode_prototype as tep
from research.ksweep_delivery import EXPECTED_K10
from research.paper_benchmark_common import (
    CachedAPI,
    EmbeddingCache,
    EXTRACTION_MODEL,
    EMBED_MODEL,
    MODEL_PRICES_USD_PER_M,
    SearchDoc,
    append_jsonl,
    freeze_run,
    jsonl,
    log,
    sha256_bytes,
    sha256_file,
    usage_summary,
)

ROOT = base.ROOT
RUN_DIR = ROOT / ".research_runs" / "simple_prop_baseline_v2"
INDEX_DIR = ROOT / ".research_runs" / "simple_prop_index_v2"
PROTOCOL = ROOT / "research" / "SIMPLE_PROP_BASELINE_PROTOCOL.md"
RUNNER = Path(__file__).resolve()
MAX_ITEMS_PER_TURN = 2
CEILING_USD = 6.00
SEED = 20260917
BOOTSTRAP = 10_000
CATEGORY_NAMES = {
    ("F", "MH"): "Multi-hop", ("F", "SH"): "Single-hop", ("F", "TP"): "Temporal Duration",
    ("MA", "C"): "Constraint", ("MA", "P"): "Proactivity", ("MA", "U"): "Update",
    ("P", "Skill"): "Skill", ("P", "Style"): "Style", ("P", "Title"): "Title",
}


def _prop_extraction_prompt(context_lines: str) -> str:
    return f"""You will see a window of chat turns from ONE session. Turns marked
[ANCHOR] were specifically selected for review; the rest are surrounding
context, shown only to help you resolve pronouns, ellipsis, sarcasm, and short replies.
You do not know why these turns were selected or what question (if any) they answer.

Extract only DURABLE, externally useful facts -- a state, a change, a commitment, a
decision, or a relationship fact that someone could actually need to recall later. Do NOT
extract conversational fragments, agreement markers, vague reactions, or generic
self-reports -- for example "X is right", "this matters", "I am thinking", "leave it to
me", "okay", "sounds good" are NOT facts UNLESS they explicitly introduce or revise a
concrete decision/state that is stated in the source (in which case extract THAT concrete
content, not the filler phrase itself). 0 items is a fine answer, and is expected for most
turns -- do not force a fact out of small talk.

At most {MAX_ITEMS_PER_TURN} durable facts per turn shown here (zero is normal; most turns
produce none).

Each fact:
- text: ONE self-contained sentence that resolves every pronoun and ellipsis using the
  context, and names people explicitly, so it can be understood months later with no chat
  around it. This is the ONLY thing you write about the fact -- do not add a category, a
  type, or say whose opinion it is; write it as a plain, complete statement.
- evidence: one or more {{"turn_id": "...", "quote": "..."}} objects -- turn_id copied
  EXACTLY from the turn_ids shown, quote a SHORT EXACT substring copied verbatim from that
  turn's actual message (this will be mechanically verified -- an invented or paraphrased
  quote is dropped, and the whole fact is rejected if no valid quote remains). At least one
  required.
- local_context_turn_ids: any additional shown turns that helped you interpret this fact;
  may overlap with evidence turn_ids, may be empty.

Return JSON only, with every turn id replaced by a real id copied from the [[...]] markers above:
{{"items":[{{"text":"...","evidence":[{{"turn_id":"<copy a real id from [[...]] above>","quote":"exact short substring"}}],
"local_context_turn_ids":["<copy a real id from [[...]] above>"]}}]}}

Turns:
{context_lines}"""


@dataclass(frozen=True)
class PropUnit:
    unit_id: str
    network_id: str
    text: str
    evidence: tuple[dict, ...]
    source_turn_ids: tuple[str, ...]
    local_context_turn_ids: tuple[str, ...]


def _session_context(session_frame: pd.DataFrame, anchor_ids: set[str]) -> tuple[str, set[str], dict[str, str]]:
    """Byte-identical formatting to build_episode_snapshot's full_session_context closure,
    which built the sealed EVENTS extraction's prompts."""
    actual = set(session_frame.turn_id.astype(str))
    if actual != {str(t) for t in anchor_ids}:
        raise ValueError("query-independent extraction cluster is not the complete session")
    lines, message_by_turn = [], {}
    for row in session_frame.itertuples(index=False):
        turn_id = str(row.turn_id)
        message = str(row.message)
        lines.append(f"[[{turn_id}]] [ANCHOR] {row.timestamp} {row.speaker_display_name}: {message}")
        message_by_turn[turn_id] = message
    return "\n".join(lines), actual, message_by_turn


def validate_prop_unit(
    raw: dict, window_turn_ids: set[str], message_by_turn: dict[str, str], network_id: str, unit_index: int,
) -> tuple[Optional[PropUnit], str, int, int]:
    """Same validator contract as temporal_episode_prototype.validate_memory_event, minus the
    event-schema fields: literal-quote evidence, generic-fragment rejection, nothing else."""
    text = str(raw.get("text") or "").strip()
    if not text:
        return None, "empty_text", 0, 0

    raw_evidence = raw.get("evidence", [])
    evidence_attempted = len(raw_evidence) if isinstance(raw_evidence, list) else 0
    valid_evidence: list[dict] = []
    if isinstance(raw_evidence, list):
        for entry in raw_evidence:
            if not isinstance(entry, dict):
                continue
            turn_id = tep._resolve_turn_id(entry.get("turn_id"), window_turn_ids)
            quote = str(entry.get("quote") or "")
            if turn_id not in window_turn_ids:
                continue
            if not tep._quote_is_valid_substring(quote, message_by_turn.get(turn_id, "")):
                continue
            valid_evidence.append({"turn_id": turn_id, "quote": quote.strip()})
    if not valid_evidence:
        return None, "no_valid_evidence_quote", evidence_attempted, 0

    if tep._is_generic_fragment(text):
        return None, "generic_fragment_or_agreement_marker", evidence_attempted, len(valid_evidence)

    source_ids = tuple(dict.fromkeys(e["turn_id"] for e in valid_evidence))
    local_ids = tuple(dict.fromkeys(
        resolved for s in raw.get("local_context_turn_ids", [])
        if (resolved := tep._resolve_turn_id(s, window_turn_ids)) in window_turn_ids
    ))
    return PropUnit(
        unit_id=f"{network_id}:prop{unit_index}", network_id=network_id, text=text,
        evidence=tuple(valid_evidence), source_turn_ids=source_ids, local_context_turn_ids=local_ids,
    ), "ok", evidence_attempted, len(valid_evidence)


def _enforce_max_per_turn(units: list[PropUnit]) -> tuple[list[PropUnit], int]:
    counts: dict[str, int] = defaultdict(int)
    kept: list[PropUnit] = []
    dropped = 0
    for unit in units:
        if any(counts[t] >= MAX_ITEMS_PER_TURN for t in unit.source_turn_ids):
            dropped += 1
            continue
        for t in unit.source_turn_ids:
            counts[t] += 1
        kept.append(unit)
    return kept, dropped


def build_session_prompts(frame: pd.DataFrame) -> dict[str, list[tuple[tuple, set, str, dict]]]:
    """(network_id -> [(cluster_key, anchor_ids, prompt, message_by_turn), ...]), fully built
    offline: used by both --preflight (token counting only) and --run (the real calls)."""
    session_frames = {
        (str(network), int(session)): group.sort_values(["timestamp", "turn_id"])
        for (network, session), group in frame.groupby(["network_id", "session_index"], sort=False)
    }
    result: dict[str, list] = {}
    for topic in base.BATCHES:
        entries = []
        for session_index, anchor_ids in qi.build_query_independent_clusters(frame, topic):
            session_frame = session_frames[(topic, int(session_index))]
            context_lines, window_ids, message_by_turn = _session_context(session_frame, anchor_ids)
            prompt = _prop_extraction_prompt(context_lines)
            entries.append((session_index, window_ids, prompt, message_by_turn))
        result[topic] = entries
    return result


def _tokenize(text: str, encoding) -> int:
    return len(encoding.encode(text))


def preflight() -> dict:
    frame, _raw_by_topic, _raw_by_id = base.load_messages()
    prompts_by_topic = build_session_prompts(frame)
    import tiktoken
    encoding = tiktoken.get_encoding("o200k_base")

    per_topic = {}
    total_input_tokens = 0
    total_calls = 0
    for topic, entries in prompts_by_topic.items():
        input_tokens = sum(_tokenize(prompt, encoding) for _, _, prompt, _ in entries)
        per_topic[topic] = {"sessions": len(entries), "input_tokens": input_tokens}
        total_input_tokens += input_tokens
        total_calls += len(entries)

    price_in, price_out = MODEL_PRICES_USD_PER_M[EXTRACTION_MODEL]
    # Conservative fixed assumption, documented rather than measured: extraction-only calls (no
    # linking/attach) run larger completions than the sealed run's blended average (142 tokens,
    # over extraction+attach combined, where attach outputs are short). Padded further by 50%.
    assumed_output_tokens_per_call = 450
    extraction_input_usd = total_input_tokens * price_in / 1_000_000
    extraction_output_usd = total_calls * assumed_output_tokens_per_call * price_out / 1_000_000
    extraction_usd = extraction_input_usd + extraction_output_usd

    embed_price = MODEL_PRICES_USD_PER_M[EMBED_MODEL][0]
    # Assume up to as many documents survive validation as the sealed EVENTS run had (16,859),
    # scaled by this run's session count relative to the sealed run's (this run has no separate
    # attach phase, so the session/call count itself is the only new ratio); padded further.
    assumed_docs = 17_000
    assumed_doc_tokens = 60  # a short self-contained sentence, well above typical length
    embedding_usd = assumed_docs * assumed_doc_tokens * embed_price / 1_000_000

    report = {
        "sessions_total": total_calls, "input_tokens_total": total_input_tokens,
        "per_topic": per_topic,
        "extraction_expected_usd": round(extraction_usd, 4),
        "embedding_expected_usd_upper_bound": round(embedding_usd, 4),
        "total_expected_usd": round(extraction_usd + embedding_usd, 4),
        "ceiling_usd": CEILING_USD,
        "assumptions": {
            "assumed_output_tokens_per_extraction_call": assumed_output_tokens_per_call,
            "assumed_surviving_documents": assumed_docs, "assumed_doc_tokens": assumed_doc_tokens,
        },
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def prop_doc(row: dict, raw_by_id: dict[str, SearchDoc]) -> SearchDoc:
    text = row["text"]
    sources = tuple(dict.fromkeys(row["source_turn_ids"]))
    lines = ["[DERIVED PROPOSITION]", text]
    for source_id in sources:
        source = raw_by_id.get(source_id)
        if source:
            lines.append(f"  [SOURCE {source_id}] {source.rendered}")
    return SearchDoc(
        doc_id=f"prop:{row['network_id']}:{row['unit_id']}", index_text=text,
        rendered="\n".join(lines), source_ids=sources, kind="prop",
    )


WORKERS = 30


def _extract_one_session(api: CachedAPI, topic: str, session_index, window_ids, prompt, message_by_turn):
    """Runs the chat call plus validation for one session, returns (kept_units, per-item
    rejection reasons, evidence counters) -- pure w.r.t. shared state, so it is safe to call
    from a thread pool; the caller does all file writes."""
    output = api.chat(model=EXTRACTION_MODEL, messages=[{"role": "user", "content": prompt}],
                      temperature=0, max_tokens=8192, json_output=True, phase="extraction")
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", output, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
    items = [item for item in (parsed.get("items") or []) if isinstance(item, dict)]
    session_units: list[PropUnit] = []
    rejections: list[str] = []
    evidence_attempted = evidence_valid = 0
    for local_index, item in enumerate(items):
        unit, reason, attempted, valid = validate_prop_unit(
            item, window_ids, message_by_turn, topic, session_index * 100 + local_index)
        evidence_attempted += attempted
        evidence_valid += valid
        if unit is None:
            rejections.append(reason)
        else:
            session_units.append(unit)
    kept, dropped = _enforce_max_per_turn(session_units)
    return kept, rejections, dropped, evidence_attempted, evidence_valid


def run_extraction() -> None:
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY отсутствует в .env")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    api = CachedAPI(RUN_DIR, api_key=key, label="openai")

    frame, _raw_by_topic, _raw_by_id = base.load_messages()
    prompts_by_topic = build_session_prompts(frame)
    props_path = RUN_DIR / "props.jsonl"
    rejected_path = RUN_DIR / "rejected.jsonl"
    write_lock = threading.Lock()
    completed_topics = {row["network_id"] for row in jsonl(RUN_DIR / "topics_done.jsonl")} if (RUN_DIR / "topics_done.jsonl").exists() else set()

    for topic, entries in prompts_by_topic.items():
        if topic in completed_topics:
            log(RUN_DIR, f"topic {topic}: already extracted, skipping")
            continue
        spend = usage_summary(RUN_DIR)["estimated_usd_upper_bound"]
        if spend > CEILING_USD:
            raise SystemExit(f"stopping before topic {topic}: spend ${spend:.2f} already over ceiling ${CEILING_USD:.2f}")
        stats = Counter()
        done = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(_extract_one_session, api, topic, *entry): entry[0] for entry in entries}
            for future in as_completed(futures):
                session_index = futures[future]
                kept, rejections, dropped, attempted, valid = future.result()
                stats["evidence_attempted"] += attempted
                stats["evidence_valid"] += valid
                stats["rejected_exceeds_max_per_turn"] += dropped
                with write_lock:
                    for reason in rejections:
                        stats[f"rejected_{reason}"] += 1
                        append_jsonl(rejected_path, {"network_id": topic, "session_index": session_index, "reason": reason})
                    for unit in kept:
                        append_jsonl(props_path, {**asdict(unit), "unit_id": unit.unit_id, "network_id": unit.network_id})
                        stats["units_kept"] += 1
                done += 1
                if done % 200 == 0:
                    log(RUN_DIR, f"topic {topic}: {done}/{len(entries)} sessions, "
                                 f"spend so far ${usage_summary(RUN_DIR)['estimated_usd_upper_bound']:.4f}")
        append_jsonl(RUN_DIR / "topics_done.jsonl", {"network_id": topic, "stats": dict(stats)})
        log(RUN_DIR, f"topic {topic}: {len(entries)} sessions -> {stats['units_kept']} units, "
                     f"spend so far ${usage_summary(RUN_DIR)['estimated_usd_upper_bound']:.4f}")


def build_index() -> dict:
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY отсутствует в .env")
    spend = usage_summary(RUN_DIR)["estimated_usd_upper_bound"]
    if spend > CEILING_USD:
        raise SystemExit(f"stopping before embeddings: extraction spend ${spend:.2f} already over ceiling ${CEILING_USD:.2f}")

    _frame, _raw_by_topic, raw_by_id = base.load_messages()
    props_by_topic: dict[str, list[SearchDoc]] = {topic: [] for topic in base.BATCHES}
    seen = set()
    for row in jsonl(RUN_DIR / "props.jsonl"):
        doc = prop_doc(row, raw_by_id)
        if doc.doc_id in seen:
            raise RuntimeError(f"duplicate prop id: {doc.doc_id}")
        seen.add(doc.doc_id)
        props_by_topic[row["network_id"]].append(doc)

    direct = CachedAPI(INDEX_DIR, api_key=key, label="openai")
    embeddings = EmbeddingCache(INDEX_DIR, direct)
    meta: dict = {"embedding_model": embeddings.model, "topics": {}}
    for topic in base.BATCHES:
        docs = props_by_topic[topic]
        digest = unlinked.docs_digest(docs)
        path = INDEX_DIR / f"props-{topic}-{digest[:20]}.npy"
        if not path.exists():
            np.save(path, embeddings.all([doc.index_text for doc in docs]))
            log(INDEX_DIR, f"prop index: topic {topic} embedded {len(docs)} propositions")
        meta["topics"][topic] = {"docs": len(docs), "docs_sha256": digest,
                                 "matrix_file": path.name, "matrix_sha256": sha256_file(path)}
    meta_path = INDEX_DIR / "index_meta.json"
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != meta:
            raise RuntimeError("sealed prop index changed")
    else:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    log(INDEX_DIR, "prop index sealed")
    return meta


def prop_matrix(topic: str, docs: list[SearchDoc]) -> np.ndarray:
    meta = json.loads((INDEX_DIR / "index_meta.json").read_text())["topics"][topic]
    if meta["docs_sha256"] != unlinked.docs_digest(docs) or meta["docs"] != len(docs):
        raise RuntimeError(f"prop documents differ from the sealed prop index: {topic}")
    path = INDEX_DIR / meta["matrix_file"]
    if sha256_file(path) != meta["matrix_sha256"]:
        raise RuntimeError(f"sealed prop matrix checksum mismatch: {path}")
    return np.load(path)


def _prec_rec(exposed: set, gold: set) -> tuple[float, float]:
    if not gold:
        return 0.0, 1.0
    precision = len(exposed & gold) / len(exposed) if exposed else 0.0
    recall = len(exposed & gold) / len(gold)
    return precision, recall


def evaluate() -> list[dict]:
    """Offline. Recomputes RAW and EVENTS through the exact same ranking code as the new PROP
    condition, so a bug in this script disagrees with the paper's own published numbers before
    anything new is trusted."""
    questions = base.load_questions(include_gold=True)
    _frame, raw_by_topic, raw_by_id = base.load_messages()
    props_by_topic: dict[str, list[SearchDoc]] = {topic: [] for topic in base.BATCHES}
    for row in jsonl(RUN_DIR / "props.jsonl"):
        props_by_topic[row["network_id"]].append(prop_doc(row, raw_by_id))
    events_by_topic = unlinked.events_by_topic(fs.SOURCE / "episodes.jsonl", raw_by_id)
    embed_dir = fs.SOURCE / "embeddings"

    rows = []
    for topic in base.BATCHES:
        topic_questions = [q for q in questions if q["topic"] == topic]
        vectors = control.question_vectors(embed_dir, [q["question"] for q in topic_questions])
        raw_matrix = control.raw_index_matrix(embed_dir, topic, raw_by_topic[topic])
        prop_matrix_topic = prop_matrix(topic, props_by_topic[topic])
        event_matrix_topic = fs.event_matrix(topic, events_by_topic[topic])
        unified_prop_docs = raw_by_topic[topic] + props_by_topic[topic]
        unified_prop_matrix = np.vstack([raw_matrix, prop_matrix_topic])
        unified_event_docs = raw_by_topic[topic] + events_by_topic[topic]
        unified_event_matrix = np.vstack([raw_matrix, event_matrix_topic])

        for q, vector in zip(topic_questions, vectors):
            gold = set(q["gold_source_ids"])
            raw_order, _ = control.rank_raw(raw_matrix, vector, 10)
            raw_exposed = set(control.unique_source_ids([raw_by_topic[topic][int(i)] for i in raw_order]))
            prop_order, _ = control.rank_raw(unified_prop_matrix, vector, 10)
            prop_exposed = set(control.unique_source_ids([unified_prop_docs[int(i)] for i in prop_order]))
            ev_order, _ = control.rank_raw(unified_event_matrix, vector, 10)
            ev_exposed = set(control.unique_source_ids([unified_event_docs[int(i)] for i in ev_order]))

            raw_p, raw_r = _prec_rec(raw_exposed, gold)
            prop_p, prop_r = _prec_rec(prop_exposed, gold)
            ev_p, ev_r = _prec_rec(ev_exposed, gold)
            rows.append({
                "qa_key": q["qa_key"], "topic": topic, "minor": q["minor"],
                "raw_precision": raw_p, "raw_recall": raw_r,
                "prop_precision": prop_p, "prop_recall": prop_r,
                "events_precision": ev_p, "events_recall": ev_r,
            })
    return rows


def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    samples = rng.choice(arr, size=(BOOTSTRAP, n), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def render_report() -> Path:
    rows = evaluate()
    n = len(rows)

    def mean(field: str) -> float:
        return statistics.mean(r[field] for r in rows) * 100

    sealed_raw = {"precision": mean("raw_precision"), "recall": mean("raw_recall")}
    sealed_events = {"precision": mean("events_precision"), "recall": mean("events_recall")}
    tolerance = 0.05
    for metric, expected in EXPECTED_K10["RAW"].items():
        if metric in sealed_raw and abs(sealed_raw[metric] - expected) > tolerance:
            raise RuntimeError(f"RAW reproduction mismatch: {metric} got {sealed_raw[metric]:.4f}, expected {expected}")
    for metric, expected in EXPECTED_K10["RAW+EVENTS"].items():
        if metric in sealed_events and abs(sealed_events[metric] - expected) > tolerance:
            raise RuntimeError(f"EVENTS reproduction mismatch: {metric} got {sealed_events[metric]:.4f}, expected {expected}")

    means = {
        "RAW": {"precision": mean("raw_precision"), "recall": mean("raw_recall")},
        "SIMPLE-PROP": {"precision": mean("prop_precision"), "recall": mean("prop_recall")},
        "EVENTS": {"precision": mean("events_precision"), "recall": mean("events_recall")},
    }

    def delta(field_a: str, field_b: str) -> dict:
        diffs = [100 * (r[field_a] - r[field_b]) for r in rows]
        lo, hi = _bootstrap_ci(diffs)
        return {"delta_pp": statistics.mean(diffs), "ci": [lo, hi], "n": len(diffs)}

    comparisons = {
        "PROP - RAW (precision)": delta("prop_precision", "raw_precision"),
        "PROP - RAW (recall)": delta("prop_recall", "raw_recall"),
        "EVENTS - PROP (precision)": delta("events_precision", "prop_precision"),
        "EVENTS - PROP (recall)": delta("events_recall", "prop_recall"),
        "EVENTS - RAW (precision, sanity check vs paper)": delta("events_precision", "raw_precision"),
    }

    by_project: dict[str, dict] = {}
    for topic in base.BATCHES:
        topic_rows = [r for r in rows if r["topic"] == topic]
        diffs = [100 * (r["prop_precision"] - r["raw_precision"]) for r in topic_rows]
        diffs_ev = [100 * (r["events_precision"] - r["prop_precision"]) for r in topic_rows]
        by_project[topic] = {"n": len(topic_rows), "prop_minus_raw": statistics.mean(diffs),
                             "events_minus_prop": statistics.mean(diffs_ev)}

    props_meta = {row["network_id"]: 0 for row in jsonl(RUN_DIR / "topics_done.jsonl")}
    total_units = sum(1 for _ in jsonl(RUN_DIR / "props.jsonl"))
    rejected_counts = Counter(row["reason"] for row in jsonl(RUN_DIR / "rejected.jsonl"))

    summary = {
        "n_questions": n, "means_percent": means, "comparisons_pp": comparisons,
        "by_project": by_project, "prop_units_total": total_units,
        "rejected_reasons": dict(rejected_counts),
        "reproduction_check": {"raw": sealed_raw, "events": sealed_events, "tolerance": tolerance},
        "cost": usage_summary(RUN_DIR),
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    def pct(v: float) -> str:
        return f"{v:.2f}"

    lines = ["# SIMPLE-PROP baseline: does the schema add anything beyond self-contained rewriting?", "",
             f"- protocol sha256 `{sha256_file(PROTOCOL)}`; runner sha256 `{sha256_file(RUNNER)}`",
             f"- questions: {n}; SIMPLE-PROP units extracted: {total_units}",
             f"- reproduction check: RAW precision {sealed_raw['precision']:.2f}% "
             f"(paper 21.29%), EVENTS precision {sealed_events['precision']:.2f}% (paper 24.50%) -- within tolerance", "",
             "## Means (%)", "", "| condition | precision | recall |", "|---|---:|---:|"]
    for cond, vals in means.items():
        lines.append(f"| {cond} | {pct(vals['precision'])} | {pct(vals['recall'])} |")
    lines += ["", "## Paired comparisons (pp)", "", "| comparison | delta | 95% CI | n |", "|---|---:|---|---:|"]
    for name, c in comparisons.items():
        lines.append(f"| {name} | {c['delta_pp']:+.2f} | [{c['ci'][0]:+.2f}; {c['ci'][1]:+.2f}] | {c['n']} |")
    lines += ["", "## Per-project (precision, pp)", "", "| project | n | PROP-RAW | EVENTS-PROP |", "|---|---:|---:|---:|"]
    for topic, v in by_project.items():
        lines.append(f"| {topic} | {v['n']} | {v['prop_minus_raw']:+.2f} | {v['events_minus_prop']:+.2f} |")
    lines += ["", "## Rejected extraction items", "", f"{dict(rejected_counts)}", "",
             "## Cost", "", f"- {usage_summary(RUN_DIR)}"]
    report_path = RUN_DIR / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    if args.preflight:
        report = preflight()
        print(json.dumps(report, indent=2))
    elif args.run:
        run_extraction()
        build_index()
        print(f"extraction + index done. cost so far: {usage_summary(RUN_DIR)['estimated_usd_upper_bound']:.4f}")
    elif args.report:
        path = render_report()
        print(path)
    elif args.freeze:
        destination = freeze_run(RUN_DIR, [RUNNER, PROTOCOL], related_dirs=[INDEX_DIR])
        print(destination)


if __name__ == "__main__":
    main()
