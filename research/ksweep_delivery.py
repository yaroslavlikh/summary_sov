"""k-sweep, phase 1: evidence delivery for RAW and RAW+EVENTS at k in {10, 20, 40}.

Offline. No API client is constructed: query and document vectors are read from the sealed
caches through evermembench_temporal_budget_control.frozen_npy, which refuses to recompute.
Sealed directories are only read; everything new goes to .research_runs/ksweep_v1_<ts>_MSK.

Ranking, provenance expansion, rendering and token counting are imported unchanged from the
runners that produced the sealed k=10 results. The run stops before any aggregate is written
if k=10 does not reproduce the sealed retrieval, prompts, per-question evidence metrics and
their means.

    python3 -m research.ksweep_delivery
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from research import evermembench_chronological_order_control as chrono
from research import evermembench_episode_run as base
from research import evermembench_parallel_resume as runner
from research import evermembench_temporal_budget_control as control
from research import evermembench_unlinked_events_control as unlinked
from research import final_sprint_evermembench as fs
from research import paper_benchmark_common as common
from research.paper_benchmark_common import (
    EMBED_MODEL,
    append_jsonl,
    jsonl,
    seal_jsonl,
    sha256_bytes,
    sha256_file,
    verify_sealed_jsonl,
)

ROOT = base.ROOT
KS = (10, 20, 40)
CONDITIONS = ("RAW", "RAW+EVENTS")
SEED = 20260917
BOOTSTRAP = 10_000
A_RUN = fs.FROZEN / "final_sprint_A_events_full_v1_20260915_233142_MSK" / "run"
EMBED_DIR = fs.SOURCE / "embeddings"
# Sealed k=10 means, as published: percent for precision/recall, counts per question otherwise.
EXPECTED_K10 = {
    "RAW": {"precision": 21.29, "recall": 8.08, "sources": 10.00, "gold_hits": 2.129},
    "RAW+EVENTS": {"precision": 24.50, "recall": 8.71, "sources": 9.47, "gold_hits": 2.245},
}
METRICS = ("precision", "recall", "gold_hits", "sources", "context_tokens")
CATEGORY_NAMES = {
    ("F", "MH"): "Multi-hop", ("F", "SH"): "Single-hop", ("F", "TP"): "Temporal Duration",
    ("MA", "C"): "Constraint", ("MA", "P"): "Proactivity", ("MA", "U"): "Update",
    ("P", "Skill"): "Skill", ("P", "Style"): "Style", ("P", "Title"): "Title",
}


def query_batch_paths(texts: list[str]) -> list[Path]:
    """Same batch keys as control.question_vectors, for the manifest."""
    paths = []
    for start in range(0, len(texts), control.EMBED_BATCH):
        batch = texts[start:start + control.EMBED_BATCH]
        key = sha256_bytes(json.dumps({"model": EMBED_MODEL, "texts": batch}, ensure_ascii=False).encode())
        paths.append(EMBED_DIR / f"batch-{key[:24]}.npy")
    return paths


def raw_index_path(topic: str, docs) -> Path:
    digest = sha256_bytes("\n".join(f"{doc.doc_id}\t{doc.index_text}" for doc in docs).encode())[:20]
    return EMBED_DIR / f"index-{topic}-raw-{digest}.npy"


def verify_a_run() -> None:
    for line in (A_RUN.parent / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if sha256_file(A_RUN.parent / name) != digest:
            raise RuntimeError(f"frozen checksum mismatch: {A_RUN.parent.name}/{name}")
    verify_sealed_jsonl(A_RUN / "predictions.jsonl", A_RUN / "predictions_meta.json", expected_rows=2400)
    verify_sealed_jsonl(A_RUN / "results.jsonl", A_RUN / "results_meta.json", expected_rows=2400)


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = [line for line in git("status", "--porcelain", "research").splitlines() if line]
    return {"head": git("rev-parse", "HEAD"), "research_uncommitted_entries": len(dirty),
            "note": "HEAD does not contain every script used; see dependencies_sha256"}


def bootstrap_ci(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    means = np.empty(indices.shape[0], dtype=np.float64)
    for start in range(0, indices.shape[0], 500):
        means[start:start + 500] = values[indices[start:start + 500]].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    stamp = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / ".research_runs" / f"ksweep_v1_{stamp}_MSK"
    run_dir.mkdir(parents=True, exist_ok=False)
    common.log(run_dir, "phase 1 start; no API client is constructed")

    verify_a_run()
    data, load_checks = fs.load_data()
    gold = {row["qa_key"]: set(row["gold_source_ids"]) for row in base.load_questions(include_gold=True)}
    sealed_pred = {
        "RAW": {qa: row for (qa, cond), row in data.frozen.items() if cond == "RAW"},
        "RAW+EVENTS": {row["qa_key"]: row for row in jsonl(A_RUN / "predictions.jsonl")},
    }
    sealed_res = {
        "RAW": {row["qa_key"]: row for row in jsonl(fs.SOURCE / "results.jsonl") if row["condition"] == "RAW"},
        "RAW+EVENTS": {row["qa_key"]: row for row in jsonl(A_RUN / "results.jsonl")},
    }

    import tiktoken
    encoder = tiktoken.get_encoding(control.TOKEN_ENCODING)

    mismatches: dict[str, list] = defaultdict(list)
    input_files: dict[str, str] = {}
    rows: list[dict] = []
    for topic in base.BATCHES:
        questions = [row for row in data.questions if row["topic"] == topic]
        texts = [row["question"] for row in questions]
        for path in query_batch_paths(texts):
            input_files[str(path.relative_to(ROOT))] = sha256_file(path)
        raw_docs = data.raw_by_topic[topic]
        index_path = raw_index_path(topic, raw_docs)
        input_files[str(index_path.relative_to(ROOT))] = sha256_file(index_path)
        vectors = control.question_vectors(EMBED_DIR, texts)
        raw_matrix = control.raw_index_matrix(EMBED_DIR, topic, raw_docs)
        unified_docs = raw_docs + data.events[topic]
        unified_matrix = np.vstack([raw_matrix, fs.event_matrix(topic, data.events[topic])])
        for question, vector in zip(questions, vectors):
            qa = question["qa_key"]
            for condition, docs_all, matrix in (("RAW", raw_docs, raw_matrix),
                                                 ("RAW+EVENTS", unified_docs, unified_matrix)):
                order, _ = control.rank_raw(matrix, vector, max(KS))
                for k in KS:
                    docs = [docs_all[int(i)] for i in order[:k]]
                    sources = control.unique_source_ids(docs)
                    context = control.render_context(docs)
                    exposed, evidence = set(sources), gold[qa]
                    hits = len(exposed & evidence)
                    row = {
                        "qa_key": qa, "topic": topic, "major": question["major"], "minor": question["minor"],
                        "condition": condition, "k": k, "sources": len(sources),
                        "context_tokens": len(encoder.encode(context)), "gold_hits": hits,
                        "precision": hits / len(exposed) if exposed else 0.0,
                        "recall": hits / len(evidence) if evidence else 1.0,
                        "derived_docs": sum(1 for doc in docs if doc.kind != "raw"),
                        "doc_ids": [doc.doc_id for doc in docs],
                    }
                    rows.append(row)
                    if k != 10:
                        continue
                    pred, res = sealed_pred[condition][qa], sealed_res[condition][qa]
                    if row["doc_ids"] != pred["retrieved_ids"]:
                        mismatches["retrieved_ids"].append((condition, qa))
                    if sources != pred["exposed_source_ids"]:
                        mismatches["exposed_source_ids"].append((condition, qa))
                    if row["precision"] != res["evidence_precision"] or row["recall"] != res["evidence_recall"]:
                        mismatches["evidence_metrics"].append(
                            (condition, qa, row["precision"], res["evidence_precision"], row["recall"], res["evidence_recall"]))
                    if condition == "RAW+EVENTS":
                        if row["context_tokens"] != pred["context_tokens"]:
                            mismatches["context_tokens"].append((condition, qa, row["context_tokens"], pred["context_tokens"]))
                    else:
                        prompt = fs.build_prompt(data.prompts, question, context)
                        if fs.chat_key(prompt, fs.ANSWER_MAX_TOKENS) not in data.cache:
                            mismatches["raw_prompt_not_in_sealed_cache"].append((condition, qa))

    for topic in base.BATCHES:
        meta = json.loads((fs.EVENT_INDEX / "index_meta.json").read_text())["topics"][topic]
        path = fs.EVENT_INDEX / meta["matrix_file"]
        input_files[str(path.relative_to(ROOT))] = sha256_file(path)

    by_key = {(r["qa_key"], r["condition"], r["k"]): r for r in rows}
    keys = sorted({r["qa_key"] for r in rows})

    def mean_of(condition: str, k: int, metric: str, subset: list[str] = keys) -> float:
        scale = 100.0 if metric in ("precision", "recall") else 1.0
        return scale * float(np.mean([by_key[(qa, condition, k)][metric] for qa in subset]))

    sanity = {"per_question_mismatches": {name: len(items) for name, items in mismatches.items()},
              "examples": {name: items[:5] for name, items in mismatches.items()}, "means": {}}
    means_ok = True
    for condition, expected in EXPECTED_K10.items():
        got = {"precision": round(mean_of(condition, 10, "precision"), 2),
               "recall": round(mean_of(condition, 10, "recall"), 2),
               "sources": round(mean_of(condition, 10, "sources"), 2),
               "gold_hits": round(mean_of(condition, 10, "gold_hits"), 3)}
        sanity["means"][condition] = {"expected": expected, "recomputed": got}
        means_ok &= all(got[name] == value for name, value in expected.items())
    sanity["passed"] = means_ok and not mismatches
    (run_dir / "sanity_k10.json").write_text(json.dumps(sanity, indent=2, ensure_ascii=False) + "\n")
    if not sanity["passed"]:
        common.log(run_dir, "k=10 sanity check FAILED; stopping before aggregates")
        print(json.dumps(sanity, indent=2, ensure_ascii=False))
        sys.exit(2)
    common.log(run_dir, "k=10 sanity check passed")

    for row in rows:
        append_jsonl(run_dir / "per_question.jsonl", row)
    seal_jsonl(run_dir / "per_question.jsonl", run_dir / "per_question_meta.json",
               expected_rows=len(keys) * len(CONDITIONS) * len(KS),
               identity_fields=("qa_key", "condition", "k"),
               metadata={"phase": 1, "ks": list(KS), "conditions": list(CONDITIONS)})

    # Groups and their bootstrap indices, drawn once in a fixed order and reused across k and metrics.
    groups: list[tuple[str, str, list[str]]] = [("all", "all 2400", keys)]
    for topic in base.BATCHES:
        groups.append(("project", topic, [qa for qa in keys if by_key[(qa, "RAW", 10)]["topic"] == topic]))
    for code, name in sorted(CATEGORY_NAMES.items(), key=lambda item: item[1]):
        groups.append(("category", name, [qa for qa in keys if (by_key[(qa, "RAW", 10)]["major"],
                                                                by_key[(qa, "RAW", 10)]["minor"]) == code]))
    rng = np.random.default_rng(SEED)
    indices = {label: rng.integers(0, len(subset), size=(BOOTSTRAP, len(subset)), dtype=np.int32)
               for _kind, label, subset in groups}

    summary: dict = {"means": {}, "deltas": {}, "dominance": {}}
    for k in KS:
        summary["means"][str(k)] = {
            condition: {metric: mean_of(condition, k, metric) for metric in (*METRICS, "derived_docs")}
            for condition in CONDITIONS
        }
        summary["deltas"][str(k)] = {}
        for kind, label, subset in groups:
            cell = {"kind": kind, "n": len(subset)}
            for metric in METRICS:
                scale = 100.0 if metric in ("precision", "recall") else 1.0
                values = scale * np.asarray([by_key[(qa, "RAW+EVENTS", k)][metric] - by_key[(qa, "RAW", k)][metric]
                                             for qa in subset], dtype=np.float64)
                low, high = bootstrap_ci(values, indices[label])
                cell[metric] = {"delta": float(values.mean()), "ci95": [low, high]}
            summary["deltas"][str(k)][label] = cell
        pairs = [(by_key[(qa, "RAW+EVENTS", k)], by_key[(qa, "RAW", k)]) for qa in keys]
        overall = summary["deltas"][str(k)]["all 2400"]
        summary["dominance"][str(k)] = {
            "mean_sources_events_le_raw": summary["means"][str(k)]["RAW+EVENTS"]["sources"]
                                         <= summary["means"][str(k)]["RAW"]["sources"],
            "mean_gold_hits_events_gt_raw": summary["means"][str(k)]["RAW+EVENTS"]["gold_hits"]
                                           > summary["means"][str(k)]["RAW"]["gold_hits"],
            "gold_hits_delta_ci_excludes_zero_above": overall["gold_hits"]["ci95"][0] > 0,
            "questions_events_more_sources": sum(1 for e, r in pairs if e["sources"] > r["sources"]),
            "questions_events_fewer_or_equal_sources_and_more_gold": sum(
                1 for e, r in pairs if e["sources"] <= r["sources"] and e["gold_hits"] > r["gold_hits"]),
            "questions_events_fewer_gold": sum(1 for e, r in pairs if e["gold_hits"] < r["gold_hits"]),
        }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_report(run_dir, summary, groups)

    dependencies = (Path(__file__).resolve(), Path(fs.__file__), Path(control.__file__), Path(base.__file__),
                    Path(unlinked.__file__), Path(common.__file__), Path(runner.__file__), Path(chrono.__file__))
    official = [base.OFFICIAL_CODE / "eval" / "config" / "prompts.yaml"]
    for topic in base.BATCHES:
        official += [base.OFFICIAL_DATA / topic / f"qa_{topic}.json", base.OFFICIAL_DATA / topic / "dialogue.json"]
    manifest = {
        "experiment": "ksweep_v1", "phase": 1, "created_msk": stamp, "git": git_state(),
        "config": {"ks": list(KS), "conditions": list(CONDITIONS), "bootstrap": BOOTSTRAP, "seed": SEED,
                   "ci": "percentile 2.5/97.5, paired question bootstrap; one index draw per group, "
                         "reused across k and metrics; groups drawn in order: all, projects, categories by name",
                   "token_encoding": control.TOKEN_ENCODING, "embedding_model": EMBED_MODEL,
                   "precision_empty_exposed": 0.0, "api_calls": 0},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "tiktoken": getattr(tiktoken, "__version__", "unknown")},
        "dependencies_sha256": {path.name: sha256_file(path) for path in dependencies},
        "frozen_sha256sums": {d.name: sha256_file(d / "SHA256SUMS") for d in (*fs.FROZEN_DIRS, A_RUN.parent)},
        "frozen_checks": load_checks,
        "event_index_meta_sha256": sha256_file(fs.EVENT_INDEX / "index_meta.json"),
        "vector_files_sha256": input_files,
        "official_inputs_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in official},
        "sanity_k10_sha256": sha256_file(run_dir / "sanity_k10.json"),
        "outputs_sha256": {name: sha256_file(run_dir / name)
                           for name in ("per_question.jsonl", "summary.json", "report_phase1.md")},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    common.log(run_dir, "phase 1 done")
    print(run_dir)


def fmt(cell: dict, digits: int) -> str:
    low, high = cell["ci95"]
    return f"{cell['delta']:+.{digits}f} [{low:+.{digits}f}; {high:+.{digits}f}]"


DIGITS = {"precision": 2, "recall": 2, "gold_hits": 3, "sources": 2, "context_tokens": 1}


def write_report(run_dir: Path, summary: dict, groups: list) -> None:
    lines = ["# k-sweep, phase 1: evidence delivery", "",
             "EverMemBench, 2,400 QA. Offline, 0 API calls. Precision and recall in percent; deltas in pp. "
             f"Paired question bootstrap, {BOOTSTRAP:,} resamples, seed {SEED}, percentile 95% CI.", "",
             "## Means by k and condition", "",
             "| k | condition | sources | context_tokens | gold hits | precision | recall | derived docs |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for k in KS:
        for condition in CONDITIONS:
            m = summary["means"][str(k)][condition]
            lines.append(f"| {k} | {condition} | {m['sources']:.2f} | {m['context_tokens']:.1f} | {m['gold_hits']:.3f} | "
                         f"{m['precision']:.2f} | {m['recall']:.2f} | {m['derived_docs']:.2f} |")
    lines += ["", "## RAW+EVENTS − RAW, all 2,400 QA", "",
              "| k | precision, pp | recall, pp | gold hits | sources | context_tokens |",
              "|---:|---|---|---|---|---|"]
    for k in KS:
        c = summary["deltas"][str(k)]["all 2400"]
        lines.append(f"| {k} | " + " | ".join(fmt(c[m], DIGITS[m]) for m in METRICS) + " |")
    lines += ["", "## Fewer-or-equal sources and more gold hits", "",
              "| k | mean sources E ≤ R | mean gold hits E > R | gold-hit Δ CI > 0 | "
              "QA with E more sources | QA with E ≤ sources and more gold | QA with E fewer gold |",
              "|---:|---|---|---|---:|---:|---:|"]
    for k in KS:
        d = summary["dominance"][str(k)]
        lines.append(f"| {k} | {d['mean_sources_events_le_raw']} | {d['mean_gold_hits_events_gt_raw']} | "
                     f"{d['gold_hits_delta_ci_excludes_zero_above']} | {d['questions_events_more_sources']} | "
                     f"{d['questions_events_fewer_or_equal_sources_and_more_gold']} | {d['questions_events_fewer_gold']} |")
    for k in KS:
        lines += ["", f"## RAW+EVENTS − RAW by project and category, k={k}", "",
                  "| group | n | precision, pp | recall, pp | gold hits | sources | context_tokens |",
                  "|---|---:|---|---|---|---|---|"]
        for kind, label, _subset in groups[1:]:
            c = summary["deltas"][str(k)][label]
            name = f"project {label}" if kind == "project" else label
            lines.append(f"| {name} | {c['n']} | " + " | ".join(fmt(c[m], DIGITS[m]) for m in METRICS) + " |")
    (run_dir / "report_phase1.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
