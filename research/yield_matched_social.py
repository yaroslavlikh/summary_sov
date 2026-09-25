"""Yield-matched control: is the SocialMemBench schema deficit representation or yield?

On SocialMemBench the event schema yields 639 accepted units against 1547 for SIMPLE-PROP from
the same 348 sessions under the same call budget and the same two-items-per-turn cap (40.1% vs
10.6% validator rejection). The paraphrase index is 2.4x larger, and it beats the schema on
delivery. That confound is untested anywhere in this literature: memory representations are
compared without controlling how many units survive validation.

This control removes the confound the only way that costs nothing: repeatedly subsample the
SIMPLE-PROP index down to the event index's size and recompute delivery. Everything else --
documents, embeddings, ranking code, gold, retrieval depth -- is reused unchanged from the
sealed v2 run, so no API call is made.

Reading rule, fixed before the run:
  * paraphrase still ahead at matched size -> the deficit is representation; the schema loses on
    its merits on this benchmark;
  * advantage disappears -> the deficit is yield, and the published comparison measures index
    size rather than representation;
  * partial -> the split is quantified by how much of the gap the matching removes.

    python3 -m research.yield_matched_social
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from research import simple_prop_social as sp
from research.paper_benchmark_common import SearchDoc, jsonl

REPEATS = 20
SEED = 20260921
OUT = sp.ROOT / ".research_runs" / "yield_matched_social_v1"


def _subsample(docs: list[SearchDoc], matrix: np.ndarray, size: int, rng) -> tuple[list[SearchDoc], np.ndarray]:
    if len(docs) <= size:
        return docs, matrix
    keep = np.sort(rng.choice(len(docs), size=size, replace=False))
    return [docs[int(i)] for i in keep], matrix[keep]


def main() -> None:
    conversations, qa = sp.load_social()
    raw_by_network, raw_by_id = sp.raw_docs_by_network(conversations)
    events = sp.event_docs_by_network(raw_by_id)
    props: dict[str, list[SearchDoc]] = defaultdict(list)
    for row in jsonl(sp.RUN_DIR / "props.jsonl"):
        props[row["network_id"]].append(sp.prop_doc(row, raw_by_id))

    # Cache every matrix once; the loop below only re-indexes rows of them.
    cache: dict[tuple[str, str], np.ndarray] = {}
    for network in sorted(raw_by_network):
        cache[(network, "raw")] = sp._embed_texts([d.index_text for d in raw_by_network[network]], sp.INDEX_DIR, "raw")
        for label, docs in (("prop", props.get(network, [])), ("event", events.get(network, []))):
            if docs:
                cache[(network, label)] = sp._embed_texts([d.index_text for d in docs], sp.INDEX_DIR, label)

    totals = {"prop_full": [], "events": [], "raw": []}
    matched_runs: list[dict] = []
    per_network_sizes = {n: (len(props.get(n, [])), len(events.get(n, []))) for n in sorted(raw_by_network)}

    def delivery(network: str, docs: list[SearchDoc], matrix: np.ndarray | None, questions) -> list[tuple[float, float]]:
        raw_docs = raw_by_network[network]
        raw_matrix = cache[(network, "raw")]
        pool_docs = raw_docs + docs
        pool_matrix = np.vstack([raw_matrix, matrix]) if docs else raw_matrix
        out = []
        for qrow, vector in questions:
            gold = {str(a["turn_id"]) for a in sp.official._json(qrow.evidence_anchors_json, []) if a.get("turn_id")}
            scores = np.sum(pool_matrix * vector[None, :], axis=1, dtype=np.float64)
            order = np.argsort(-scores)[:sp.TOP_K]
            exposed = {s for i in order for s in pool_docs[int(i)].source_ids}
            out.append(sp._prec_rec(exposed, gold))
        return out

    questions_by_network = {}
    for network in sorted(raw_by_network):
        network_qa = qa[qa.network_id == network]
        if not len(network_qa):
            continue
        vectors = sp._embed_texts([str(r.question) for r in network_qa.itertuples(index=False)], sp.INDEX_DIR, "questions")
        questions_by_network[network] = list(zip(network_qa.itertuples(index=False), vectors))

    # Full-size reference conditions.
    reference: dict[str, dict[str, list[float]]] = {k: defaultdict(list) for k in ("raw", "prop_full", "events")}
    for network, questions in questions_by_network.items():
        for label, docs in (("raw", []), ("prop_full", props.get(network, [])), ("events", events.get(network, []))):
            matrix = cache.get((network, "prop" if label == "prop_full" else "event")) if docs else None
            for precision, recall in delivery(network, docs, matrix, questions):
                reference[label]["precision"].append(precision)
                reference[label]["recall"].append(recall)

    # Matched-size paraphrase: subsample to the event count of the same network.
    for repeat in range(REPEATS):
        rng = np.random.default_rng(SEED + repeat)
        precisions, recalls = [], []
        for network, questions in questions_by_network.items():
            docs = props.get(network, [])
            target = len(events.get(network, []))
            if docs and target:
                kept, matrix = _subsample(docs, cache[(network, "prop")], target, rng)
            else:
                kept, matrix = ([], None)
            for precision, recall in delivery(network, kept, matrix, questions):
                precisions.append(precision)
                recalls.append(recall)
        matched_runs.append({"repeat": repeat, "precision": 100 * statistics.mean(precisions),
                             "recall": 100 * statistics.mean(recalls)})

    def mean(label: str, field: str) -> float:
        return 100 * statistics.mean(reference[label][field])

    matched_precision = [r["precision"] for r in matched_runs]
    matched_recall = [r["recall"] for r in matched_runs]
    summary = {
        "questions": len(reference["raw"]["precision"]),
        "units": {"events": sum(v[1] for v in per_network_sizes.values()),
                  "prop_full": sum(v[0] for v in per_network_sizes.values())},
        "precision": {
            "raw": mean("raw", "precision"), "events": mean("events", "precision"),
            "prop_full": mean("prop_full", "precision"),
            "prop_matched_mean": statistics.mean(matched_precision),
            "prop_matched_sd": statistics.pstdev(matched_precision),
            "prop_matched_min": min(matched_precision), "prop_matched_max": max(matched_precision),
        },
        "recall": {
            "raw": mean("raw", "recall"), "events": mean("events", "recall"),
            "prop_full": mean("prop_full", "recall"),
            "prop_matched_mean": statistics.mean(matched_recall),
            "prop_matched_sd": statistics.pstdev(matched_recall),
        },
        "repeats": REPEATS, "seed": SEED,
    }
    gap_full = summary["precision"]["events"] - summary["precision"]["prop_full"]
    gap_matched = summary["precision"]["events"] - summary["precision"]["prop_matched_mean"]
    summary["precision"]["gap_events_minus_prop_full"] = gap_full
    summary["precision"]["gap_events_minus_prop_matched"] = gap_matched
    summary["precision"]["share_of_gap_explained_by_yield"] = (
        (gap_matched - gap_full) / abs(gap_full) if gap_full else float("nan"))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (OUT / "matched_runs.json").write_text(json.dumps(matched_runs, indent=2) + "\n")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
