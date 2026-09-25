"""Yield-matched control on EverMemBench: the mirror of the SocialMemBench control.

On SocialMemBench the schema yields fewer units than SIMPLE-PROP (639 vs 1547) and loses on
delivery; matching the index size by subsampling the paraphrase removes the whole gap. On
EverMemBench the imbalance runs the other way -- the schema yields MORE units (16859 vs 16276,
3.5%) and wins by +0.91pp. If index size is what moves this metric, then part of the
EverMemBench advantage is bought by the same confound, and the honest comparison subsamples the
EVENTS index down to the paraphrase count.

Everything is reused from the sealed runs: documents, embeddings, ranking code, gold, depth.
No API call is made.

Reading rule, fixed before the run:
  * +0.91pp essentially unchanged at matched size -> the EverMemBench advantage is
    representation, and the two benchmarks differ in yield balance, not in what they say;
  * advantage shrinks toward zero -> the schema's edge on EverMemBench is also an index-size
    effect, and the paper's central claim about the schema does not survive;
  * partial -> the split is quantified by how much of the gap the matching removes.

Uncertainty is the same question-level bootstrap the manuscript already uses for this metric,
computed on the per-question difference averaged over subsampling repeats.

    python3 -m research.yield_matched_evermembench
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

from research import evermembench_episode_run as base
from research import evermembench_temporal_budget_control as control
from research import evermembench_unlinked_events_control as unlinked
from research import final_sprint_evermembench as fs
from research import simple_prop_baseline as spb
from research.paper_benchmark_common import SearchDoc, jsonl

REPEATS = 20
SEED = 20260921
BOOTSTRAP = 10000
OUT = spb.ROOT / ".research_runs" / "yield_matched_evermembench_v1"


def main() -> None:
    questions = base.load_questions(include_gold=True)
    _frame, raw_by_topic, raw_by_id = base.load_messages()
    props_by_topic: dict[str, list[SearchDoc]] = {topic: [] for topic in base.BATCHES}
    for row in jsonl(spb.RUN_DIR / "props.jsonl"):
        props_by_topic[row["network_id"]].append(spb.prop_doc(row, raw_by_id))
    events_by_topic = unlinked.events_by_topic(fs.SOURCE / "episodes.jsonl", raw_by_id)
    embed_dir = fs.SOURCE / "embeddings"

    sizes = {t: (len(events_by_topic[t]), len(props_by_topic[t])) for t in base.BATCHES}
    rows: dict[str, dict] = {}
    per_topic_keys: dict[str, list[str]] = {}

    for topic in base.BATCHES:
        topic_questions = [q for q in questions if q["topic"] == topic]
        per_topic_keys[topic] = [q["qa_key"] for q in topic_questions]
        vectors = control.question_vectors(embed_dir, [q["question"] for q in topic_questions])
        raw_matrix = control.raw_index_matrix(embed_dir, topic, raw_by_topic[topic])
        prop_matrix = spb.prop_matrix(topic, props_by_topic[topic])
        event_matrix = fs.event_matrix(topic, events_by_topic[topic])

        def delivery(docs: list[SearchDoc], matrix: np.ndarray | None) -> list[tuple[float, float]]:
            pool_docs = raw_by_topic[topic] + docs
            pool_matrix = np.vstack([raw_matrix, matrix]) if docs else raw_matrix
            out = []
            for q, vector in zip(topic_questions, vectors):
                order, _ = control.rank_raw(pool_matrix, vector, 10)
                exposed = set(control.unique_source_ids([pool_docs[int(i)] for i in order]))
                out.append(spb._prec_rec(exposed, set(q["gold_source_ids"])))
            return out

        full_prop = delivery(props_by_topic[topic], prop_matrix)
        full_event = delivery(events_by_topic[topic], event_matrix)

        target = len(props_by_topic[topic])
        matched = np.zeros((len(topic_questions), 2), dtype=np.float64)
        for repeat in range(REPEATS):
            rng = np.random.default_rng(SEED + repeat)
            docs = events_by_topic[topic]
            if len(docs) > target:
                keep = np.sort(rng.choice(len(docs), size=target, replace=False))
                kept_docs = [docs[int(i)] for i in keep]
                kept_matrix = event_matrix[keep]
            else:
                kept_docs, kept_matrix = docs, event_matrix
            matched += np.asarray(delivery(kept_docs, kept_matrix), dtype=np.float64)
        matched /= REPEATS

        for q, prop, event, match in zip(topic_questions, full_prop, full_event, matched):
            rows[q["qa_key"]] = {
                "topic": topic,
                "prop_precision": prop[0], "prop_recall": prop[1],
                "events_precision": event[0], "events_recall": event[1],
                "events_matched_precision": float(match[0]), "events_matched_recall": float(match[1]),
            }
        print(f"topic {topic}: events {sizes[topic][0]} -> {target} (prop), "
              f"precision {100*statistics.mean(x[0] for x in full_event):.2f} -> "
              f"{100*float(matched[:, 0].mean()):.2f}", flush=True)

    ordered = list(rows.values())

    def mean(field: str) -> float:
        return 100 * statistics.mean(r[field] for r in ordered)

    def paired(left: str, right: str) -> dict:
        diffs = np.asarray([100 * (r[left] - r[right]) for r in ordered], dtype=np.float64)
        rng = np.random.default_rng(SEED)
        samples = rng.choice(diffs, size=(BOOTSTRAP, len(diffs)), replace=True).mean(axis=1)
        return {"delta": float(diffs.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975))}

    summary = {
        "questions": len(ordered), "repeats": REPEATS, "seed": SEED,
        "units": {"events": sum(v[0] for v in sizes.values()),
                  "prop": sum(v[1] for v in sizes.values())},
        "per_topic_sizes": {t: {"events": v[0], "prop": v[1]} for t, v in sizes.items()},
        "precision": {"prop": mean("prop_precision"), "events": mean("events_precision"),
                      "events_matched": mean("events_matched_precision")},
        "recall": {"prop": mean("prop_recall"), "events": mean("events_recall"),
                   "events_matched": mean("events_matched_recall")},
        "paired": {
            "events_minus_prop_precision": paired("events_precision", "prop_precision"),
            "events_matched_minus_prop_precision": paired("events_matched_precision", "prop_precision"),
            "events_minus_prop_recall": paired("events_recall", "prop_recall"),
            "events_matched_minus_prop_recall": paired("events_matched_recall", "prop_recall"),
        },
    }
    per_project = {}
    for topic in base.BATCHES:
        sub = [rows[k] for k in per_topic_keys[topic]]
        per_project[topic] = {
            "events_minus_prop": 100 * statistics.mean(r["events_precision"] - r["prop_precision"] for r in sub),
            "events_matched_minus_prop": 100 * statistics.mean(r["events_matched_precision"] - r["prop_precision"] for r in sub),
        }
    summary["per_project_precision"] = per_project

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
