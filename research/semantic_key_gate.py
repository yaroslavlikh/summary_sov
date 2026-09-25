"""Natural-language semantic key retrieval plus a precision-oriented link gate.

This is a post-hoc diagnostic over frozen semantic-slot v2 artifacts.  It does
not rerun extraction, normalization, answer generation, or SocialMemBench.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

from research import semantic_slot_linking_v2 as v2
from research.semantic_slot_pairwise_resolver import BASE_DIR, _case_key, _load_cache, _slot_key, load_inputs


OUT_DIR = Path(".research_runs/semantic_key_gate_v1")
RANKINGS = OUT_DIR / "rankings.jsonl"
GATE_CACHE = OUT_DIR / "gate_cache.jsonl"
DECISIONS = OUT_DIR / "decisions.jsonl"
REPORT = OUT_DIR / "report.md"
CONFIG = OUT_DIR / "config.json"
RUN_LOG = OUT_DIR / "run.log"
MODEL = "fast"
SCHEMA_VERSION = "semantic_key_gate_v1"
TOP_K = 3
MIN_PRECISION = 0.95
MAX_NO_MATCH_FPR = 0.05


def _log(message: str) -> None:
    print(message, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a") as handle:
        handle.write(message + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_cache(path: Path, key: str, output: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"cache_key": key, "output": output}, ensure_ascii=False) + "\n")


def key_description(slot: v2.NormalizedSlot) -> str:
    owner = slot.viewpoint_owner or slot.state_holder
    topic = slot.topic_object or slot.state_holder
    if v2._norm(owner) == v2._norm(topic):
        return f"{owner}'s {slot.state_dimension} state. {slot.slot_question}"
    return f"{owner}'s {slot.state_dimension} concerning {topic}. {slot.slot_question}"


def build_rankings(slots: dict[str, v2.NormalizedSlot], retrieval: dict[str, dict]) -> list[dict]:
    old_by_network: dict[str, list[v2.NormalizedSlot]] = defaultdict(list)
    for slot in slots.values():
        if slot.side == "old":
            old_by_network[slot.network_id].append(slot)

    texts = sorted({key_description(slot) for slot in slots.values()})
    vectors = np.asarray(v2.embed_batch(texts), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    vector_by_text = dict(zip(texts, vectors))

    rows = []
    for case_key, base_row in retrieval.items():
        network_id, qa_id = base_row["network_id"], base_row["qa_id"]
        query = slots.get(_slot_key(network_id, qa_id, "new"))
        pool = old_by_network.get(network_id, [])
        if query is None or not pool:
            continue
        query_vector = vector_by_text[key_description(query)]
        scored = sorted(
            (
                float(query_vector @ vector_by_text[key_description(candidate)]),
                f"{candidate.case_qa_id}:old",
                key_description(candidate),
            )
            for candidate in pool
        )
        scored.reverse()
        rows.append({
            "case_key": case_key,
            "qa_id": qa_id,
            "network_id": network_id,
            "split": base_row["split"],
            "query_key": key_description(query),
            "gold_id": f"{qa_id}:old",
            "candidates": [
                {"candidate_id": candidate_id, "retrieval_score": score, "key_description": description, "rank": rank}
                for rank, (score, candidate_id, description) in enumerate(scored[:TOP_K], 1)
            ],
        })
    return rows


def _gate_prompt(query: v2.NormalizedSlot, candidates: list[tuple[dict, v2.NormalizedSlot]]) -> str:
    block = "\n".join(
        f"[{index}] KEY: {candidate_row['key_description']}\n"
        f"    EXISTING FACT: {candidate.state_text}"
        for index, (candidate_row, candidate) in enumerate(candidates)
    )
    return f"""Independently verify whether each existing memory key tracks the SAME evolving
state as the proposed NEW memory key.

Precision is more important than recall. MATCH only when holder/viewpoint,
semantic topic, and changeable property are the same. Values may be opposite
because opinions, plans, health, and decisions change over time. Wording and
taxonomy labels may differ: health/knee condition and plan/intention can still
refer to the same state. Return UNCERTAIN rather than guessing.

NEW KEY: {key_description(query)}
NEW FACT: {query.state_text}

EXISTING CANDIDATES:
{block}

Classify every index independently. Return JSON only:
{{"decisions":[{{"candidate_index":0,"verdict":"MATCH|NO_MATCH|UNCERTAIN",
"confidence":0.0,"rationale":"one short sentence"}}]}}"""


def normalize_gate_output(output: dict, candidate_count: int) -> dict[int, dict]:
    normalized = {}
    for item in output.get("decisions", []):
        try:
            index = int(item.get("candidate_index"))
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            continue
        verdict = str(item.get("verdict") or "").upper()
        if not 0 <= index < candidate_count or verdict not in {"MATCH", "NO_MATCH", "UNCERTAIN"} or index in normalized:
            continue
        normalized[index] = {
            "verdict": verdict,
            "confidence": confidence,
            "match_score": confidence if verdict == "MATCH" else 0.0,
            "rationale": str(item.get("rationale") or ""),
        }
    return normalized


def run_gate(rows: list[dict], slots: dict[str, v2.NormalizedSlot]) -> tuple[list[dict], int]:
    cache = _load_cache(GATE_CACHE)
    results, new_calls = [], 0
    for row in rows:
        candidates = [
            (candidate_row, slots[_slot_key(row["network_id"], candidate_row["candidate_id"].removesuffix(":old"), "old")])
            for candidate_row in row["candidates"]
        ]
        query = slots[_slot_key(row["network_id"], row["qa_id"], "new")]
        prompt = _gate_prompt(query, candidates)
        key = hashlib.sha256(f"{SCHEMA_VERSION}:{MODEL}:{prompt}".encode()).hexdigest()
        output = cache.get(key)
        if output is None:
            output = v2._call_llm(prompt, MODEL)
            cache[key] = output
            _append_cache(GATE_CACHE, key, output)
            new_calls += 1
            if new_calls % 20 == 0:
                _log(f"gate calls: {new_calls}")
        decisions = normalize_gate_output(output, len(candidates))
        enriched = []
        for index, candidate_row in enumerate(row["candidates"]):
            decision = decisions.get(index, {"verdict": "MISSING", "confidence": 0.0, "match_score": 0.0, "rationale": "missing decision"})
            enriched.append({**candidate_row, **decision})
        results.append({**row, "candidates": enriched})
    return results, new_calls


def select_match(row: dict, threshold: float) -> str | None:
    matches = [candidate for candidate in row["candidates"] if candidate["match_score"] >= threshold]
    if not matches:
        return None
    return max(matches, key=lambda candidate: (candidate["match_score"], candidate["retrieval_score"]))["candidate_id"]


def select_hybrid(row: dict, gate_threshold: float, score_threshold: float, margin_threshold: float) -> str | None:
    selected = select_match(row, gate_threshold)
    if selected is not None:
        return selected
    candidates = row["candidates"]
    if not candidates:
        return None
    margin = candidates[0]["retrieval_score"] - (candidates[1]["retrieval_score"] if len(candidates) > 1 else 0.0)
    if candidates[0]["retrieval_score"] >= score_threshold and margin >= margin_threshold:
        return candidates[0]["candidate_id"]
    return None


def metrics(rows: list[dict], threshold: float, confirmed_negatives: set[tuple[str, str, str]]) -> dict:
    counts = Counter()
    link_decisions = 0
    confirmed_no_match_queries = 0
    confirmed_no_match_fp = 0
    for row in rows:
        selected = select_match(row, threshold)
        gold = row["gold_id"]
        candidate_ids = {candidate["candidate_id"] for candidate in row["candidates"]}
        if selected == gold:
            counts["correct"] += 1
        elif selected is not None:
            counts["false_merge"] += 1
        elif gold in candidate_ids:
            counts["false_split"] += 1
        else:
            counts["retrieval_miss"] += 1
        link_decisions += selected is not None

        negative_candidates = [
            candidate for candidate in row["candidates"]
            if candidate["candidate_id"] != gold and (
                row["network_id"], f"{row['qa_id']}:new", candidate["candidate_id"]
            ) in confirmed_negatives
        ]
        if negative_candidates:
            confirmed_no_match_queries += 1
            confirmed_no_match_fp += any(candidate["match_score"] >= threshold for candidate in negative_candidates)

    n = len(rows)
    precision = counts["correct"] / link_decisions if link_decisions else 1.0
    return {
        "threshold": threshold,
        "n": n,
        "link_decisions": link_decisions,
        "precision": precision,
        "recall": counts["correct"] / n if n else 0.0,
        "correct": counts["correct"],
        "false_merge": counts["false_merge"],
        "false_split": counts["false_split"],
        "retrieval_miss": counts["retrieval_miss"],
        "confirmed_no_match_queries": confirmed_no_match_queries,
        "confirmed_no_match_fp": confirmed_no_match_fp,
        "confirmed_no_match_fpr": confirmed_no_match_fp / confirmed_no_match_queries if confirmed_no_match_queries else 0.0,
    }


def calibrate(rows: list[dict], confirmed_negatives: set[tuple[str, str, str]]) -> tuple[float, list[dict]]:
    thresholds = {0.000001, 1.000001}
    thresholds.update(candidate["match_score"] for row in rows for candidate in row["candidates"] if candidate["match_score"] > 0)
    table = [metrics(rows, threshold, confirmed_negatives) for threshold in sorted(thresholds)]
    feasible = [row for row in table if row["precision"] >= MIN_PRECISION and row["confirmed_no_match_fpr"] <= MAX_NO_MATCH_FPR]
    best = max(feasible, key=lambda row: (row["recall"], row["precision"], -row["threshold"]))
    return best["threshold"], table


def hybrid_metrics(rows: list[dict], gate_threshold: float, score_threshold: float, margin_threshold: float) -> dict:
    selected = [select_hybrid(row, gate_threshold, score_threshold, margin_threshold) for row in rows]
    correct = sum(candidate_id == row["gold_id"] for candidate_id, row in zip(selected, rows))
    links = sum(candidate_id is not None for candidate_id in selected)
    false_merges = links - correct
    return {
        "n": len(rows),
        "links": links,
        "correct": correct,
        "precision": correct / links if links else 1.0,
        "recall": correct / len(rows) if rows else 0.0,
        "false_merges": false_merges,
        "false_merge_rate": false_merges / len(rows) if rows else 0.0,
        "false_splits": sum(candidate_id is None and row["gold_id"] in {c["candidate_id"] for c in row["candidates"]}
                            for candidate_id, row in zip(selected, rows)),
        "retrieval_misses": sum(row["gold_id"] not in {c["candidate_id"] for c in row["candidates"]} for row in rows),
    }


def counterfactual_no_match_fpr(
    rows: list[dict], gate_threshold: float, score_threshold: float, margin_threshold: float,
    confirmed_negatives: set[tuple[str, str, str]],
) -> tuple[int, int, float]:
    total = false_positives = 0
    for row in rows:
        query_id = f"{row['qa_id']}:new"
        candidates = [
            candidate for candidate in row["candidates"]
            if (row["network_id"], query_id, candidate["candidate_id"]) in confirmed_negatives
        ]
        if not candidates:
            continue
        total += 1
        false_positives += select_hybrid(
            {"candidates": candidates}, gate_threshold, score_threshold, margin_threshold
        ) is not None
    return total, false_positives, false_positives / total if total else 0.0


def calibrate_hybrid(
    rows: list[dict], gate_threshold: float, confirmed_negatives: set[tuple[str, str, str]],
) -> tuple[float, float, list[dict]]:
    score_thresholds = sorted({row["candidates"][0]["retrieval_score"] for row in rows if row["candidates"]}) + [1.000001]
    margin_thresholds = sorted({
        row["candidates"][0]["retrieval_score"] - (row["candidates"][1]["retrieval_score"] if len(row["candidates"]) > 1 else 0.0)
        for row in rows if row["candidates"]
    }) + [1.000001]
    table = []
    for score_threshold in score_thresholds:
        for margin_threshold in margin_thresholds:
            result = hybrid_metrics(rows, gate_threshold, score_threshold, margin_threshold)
            no_match_n, no_match_fp, no_match_fpr = counterfactual_no_match_fpr(
                rows, gate_threshold, score_threshold, margin_threshold, confirmed_negatives,
            )
            table.append({
                **result,
                "score_threshold": score_threshold,
                "margin_threshold": margin_threshold,
                "counterfactual_no_match_n": no_match_n,
                "counterfactual_no_match_fp": no_match_fp,
                "counterfactual_no_match_fpr": no_match_fpr,
            })
    feasible = [
        row for row in table
        if row["precision"] >= MIN_PRECISION
        and row["false_merge_rate"] <= MAX_NO_MATCH_FPR
        and row["counterfactual_no_match_fpr"] <= MAX_NO_MATCH_FPR
    ]
    best = max(feasible, key=lambda row: (row["recall"], row["precision"], row["score_threshold"], row["margin_threshold"]))
    return best["score_threshold"], best["margin_threshold"], table


def bootstrap_delta(rows: list[dict], monolithic: dict[str, bool], iterations: int = 10000) -> tuple[float, float, float]:
    by_network = defaultdict(list)
    for row in rows:
        pair_key = _case_key(row["network_id"], row["qa_id"])
        by_network[row["network_id"]].append((int(row["hybrid_selected"] == row["gold_id"]), int(monolithic[pair_key])))
    networks = sorted(by_network)
    observed = sum(new - old for pairs in by_network.values() for new, old in pairs) / len(rows)
    rng = random.Random(20260912)
    samples = []
    for _ in range(iterations):
        sampled = [rng.choice(networks) for _ in networks]
        pairs = [pair for network in sampled for pair in by_network[network]]
        samples.append(sum(new - old for new, old in pairs) / len(pairs))
    samples.sort()
    return observed, samples[int(iterations * .025)], samples[int(iterations * .975) - 1]


def _write_manifest() -> None:
    files = {}
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS"}:
            relative = str(path.relative_to(OUT_DIR))
            files[relative] = {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "files": files}, indent=2) + "\n")
    (OUT_DIR / "SHA256SUMS").write_text("".join(f"{meta['sha256']}  {path}\n" for path, meta in files.items()))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log("verifying inputs and building natural-language key rankings")
    slots, base_retrieval, confirmed_negatives, _ = load_inputs()
    rankings = build_rankings(slots, base_retrieval)
    with RANKINGS.open("w") as handle:
        for row in rankings:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    gate_rows, new_calls = run_gate(rankings, slots)
    dev = [row for row in gate_rows if row["split"] == "dev"]
    evaluation = [row for row in gate_rows if row["split"] == "eval"]
    threshold, calibration = calibrate(dev, confirmed_negatives)
    dev_metrics = metrics(dev, threshold, confirmed_negatives)
    eval_metrics = metrics(evaluation, threshold, confirmed_negatives)
    score_threshold, margin_threshold, hybrid_calibration = calibrate_hybrid(dev, threshold, confirmed_negatives)
    dev_hybrid = hybrid_metrics(dev, threshold, score_threshold, margin_threshold)
    eval_hybrid = hybrid_metrics(evaluation, threshold, score_threshold, margin_threshold)
    dev_no_match = counterfactual_no_match_fpr(dev, threshold, score_threshold, margin_threshold, confirmed_negatives)
    eval_no_match = counterfactual_no_match_fpr(evaluation, threshold, score_threshold, margin_threshold, confirmed_negatives)
    decision_rows = [
        {
                **row,
                "gate_selected": select_match(row, threshold),
                "hybrid_selected": select_hybrid(row, threshold, score_threshold, margin_threshold),
        }
        for row in gate_rows
    ]
    with DECISIONS.open("w") as handle:
        for row in decision_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (OUT_DIR / "hybrid_calibration.json").write_text(json.dumps(hybrid_calibration, indent=2) + "\n")
    cache_total = len(_load_cache(GATE_CACHE))
    old_resolver = {
        _case_key(row["network_id"], row["qa_id"]): bool(row["correct_selection"])
        for row in _read_jsonl(BASE_DIR / "resolver_results.jsonl")
    }
    eval_decisions = [row for row in decision_rows if row["split"] == "eval"]
    delta, delta_low, delta_high = bootstrap_delta(eval_decisions, old_resolver)
    CONFIG.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "model": MODEL,
        "top_k": TOP_K,
        "threshold": threshold,
        "fallback_score_threshold": score_threshold,
        "fallback_margin_threshold": margin_threshold,
        "threshold_calibrated_on": "dev networks only",
        "minimum_dev_precision": MIN_PRECISION,
        "maximum_dev_confirmed_no_match_fpr": MAX_NO_MATCH_FPR,
        "new_calls_this_run": new_calls,
        "cached_calls_total": cache_total,
        "input": str(BASE_DIR),
    }, indent=2) + "\n")
    report = f"""# Natural-language semantic key + precision gate — diagnostic result

No extraction, normalization, answer generation, production code, or full benchmark was rerun.

## Setup

- top-k: {TOP_K}
- dev queries: {len(dev)}
- eval queries: {len(evaluation)}
- threshold calibrated on dev only: {threshold:.3f}
- gate calls in cache: {cache_total} (new this run: {new_calls})

## Dev calibration

- link precision: {dev_metrics['precision']:.3f}
- link recall: {dev_metrics['recall']:.3f} ({dev_metrics['correct']}/{dev_metrics['n']})
- false merges: {dev_metrics['false_merge']}/{dev_metrics['n']}
- false splits: {dev_metrics['false_split']}/{dev_metrics['n']}
- confirmed no-match FPR: {dev_metrics['confirmed_no_match_fpr']:.3f} ({dev_metrics['confirmed_no_match_fp']}/{dev_metrics['confirmed_no_match_queries']})

## Eval

- link precision: {eval_metrics['precision']:.3f}
- link recall: {eval_metrics['recall']:.3f} ({eval_metrics['correct']}/{eval_metrics['n']})
- false merges: {eval_metrics['false_merge']}/{eval_metrics['n']}
- false splits: {eval_metrics['false_split']}/{eval_metrics['n']}
- retrieval misses: {eval_metrics['retrieval_miss']}/{eval_metrics['n']}
- confirmed no-match FPR: {eval_metrics['confirmed_no_match_fpr']:.3f} ({eval_metrics['confirmed_no_match_fp']}/{eval_metrics['confirmed_no_match_queries']})

## Eval — precision gate plus embedding fallback

- fallback thresholds calibrated on dev only: score >= {score_threshold:.3f}, top1 margin >= {margin_threshold:.3f}
- dev precision: {dev_hybrid['precision']:.3f}; dev recall: {dev_hybrid['recall']:.3f} ({dev_hybrid['correct']}/{dev_hybrid['n']})
- dev counterfactual no-match FPR: {dev_no_match[2]:.3f} ({dev_no_match[1]}/{dev_no_match[0]})
- eval link precision: {eval_hybrid['precision']:.3f}
- eval link recall: {eval_hybrid['recall']:.3f} ({eval_hybrid['correct']}/{eval_hybrid['n']})
- eval false merges: {eval_hybrid['false_merges']}/{eval_hybrid['n']} ({eval_hybrid['false_merge_rate']:.3f})
- eval false splits: {eval_hybrid['false_splits']}/{eval_hybrid['n']}
- eval retrieval misses: {eval_hybrid['retrieval_misses']}/{eval_hybrid['n']}
- eval counterfactual no-match FPR: {eval_no_match[2]:.3f} ({eval_no_match[1]}/{eval_no_match[0]})

## Paired comparison

- monolithic resolver: 20/59 correct selections
- semantic-key precision gate: {eval_hybrid['correct']}/59 correct selections
- paired improvement: {delta:+.3f}
- network-bootstrap 95% CI: [{delta_low:+.3f}, {delta_high:+.3f}]

## Reference

- natural-key retrieval ceiling is the fraction whose gold OLD is in top-3;
- prior monolithic resolver selected correctly 20/59;
- prior strict pairwise resolver selected correctly 7/59.
"""
    REPORT.write_text(report)
    _log("complete")
    _write_manifest()
    print(report)


if __name__ == "__main__":
    main()
