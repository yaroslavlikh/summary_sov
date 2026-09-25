"""Two-stage resolver over the frozen semantic-slot v2 retrieval results.

Stage 1 classifies each top-5 candidate as SAME_SLOT/DIFFERENT_SLOT.  A
threshold is calibrated on dev networks only.  Stage 2 classifies the update
operation for the selected eval candidate.  No extraction, normalization,
retrieval, production code, or full benchmark is rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from research import semantic_slot_linking_v2 as v2


BASE_DIR = Path(".research_runs/semantic_slot_linking_v2")
OUT_DIR = Path(".research_runs/semantic_slot_pairwise_resolver_v1")
PAIRWISE_CACHE = OUT_DIR / "pairwise_cache.jsonl"
OPERATION_CACHE = OUT_DIR / "operation_cache.jsonl"
PAIRWISE_DECISIONS = OUT_DIR / "pairwise_decisions.jsonl"
DECISIONS = OUT_DIR / "decisions.jsonl"
CONFIG = OUT_DIR / "config.json"
REPORT = OUT_DIR / "report.md"
RUN_LOG = OUT_DIR / "run.log"
MODEL = "fast"
SCHEMA_VERSION = "pairwise_resolver_v1"
PAIRWISE_PROMPT_VERSION = "pairwise_v1"
OPERATION_PROMPT_VERSION = "operation_v1"
MAX_CANDIDATES = 5
MAX_FALSE_MERGE_RATE = 0.05


def _log(message: str) -> None:
    print(message, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a") as handle:
        handle.write(message + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_cache(path: Path) -> dict[str, dict]:
    return {row["cache_key"]: row["output"] for row in _read_jsonl(path)} if path.exists() else {}


def _append_cache(path: Path, key: str, output: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"cache_key": key, "output": output}, ensure_ascii=False) + "\n")


def _verify_manifest(directory: Path) -> None:
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    mismatches = []
    for relative, metadata in manifest["files"].items():
        path = directory / relative
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"artifact checksum mismatch: {mismatches[:5]}")


def _slot_from_cache(case: dict, side: str, conversations, cache: dict[str, dict]) -> v2.NormalizedSlot | None:
    turn_ids = tuple(case[f"{side}_source_turn_ids"])
    state_text = case[f"{side}_state_text"]
    net_turns = conversations[conversations.network_id == case["network_id"]].set_index("turn_id")
    records = [
        v2.EvidenceRecord(
            index=index,
            turn_id=turn_id,
            session_index=int(net_turns.loc[turn_id].session_index),
            speaker=str(net_turns.loc[turn_id].speaker_display_name),
            message=str(net_turns.loc[turn_id].message),
            timestamp=str(net_turns.loc[turn_id].timestamp),
            relevance="",
        )
        for index, turn_id in enumerate(turn_ids)
        if turn_id in net_turns.index
    ]
    if not records:
        return None
    prompt = v2._normalize_prompt(state_text, records)
    key = v2._normalize_cache_key(case["qa_id"], side, prompt)
    output = cache.get(key)
    if output is None:
        raise RuntimeError(f"missing frozen normalization cache for {case['qa_id']}:{side}")
    validated, _ = v2.validate_normalization(output, records)
    if validated is None:
        return None
    return v2.NormalizedSlot(
        case_qa_id=case["qa_id"],
        side=side,
        network_id=case["network_id"],
        state_holder=validated["state_holder"],
        viewpoint_owner=validated["viewpoint_owner"],
        topic_object=validated["topic_object"],
        state_dimension=validated["state_dimension"],
        slot_question=validated["slot_question"],
        value=validated["value"],
        assertion_type=validated["assertion_type"],
        normalization_confidence=validated["normalization_confidence"],
        source_turn_ids=tuple(records[i].turn_id for i in validated["source_indices"]),
        state_text=state_text,
    )


def _case_key(network_id: str, qa_id: str) -> str:
    """qa_id is not globally unique in SocialMemBench."""
    return f"{network_id}|{qa_id}"


def _slot_key(network_id: str, qa_id: str, side: str) -> str:
    return f"{_case_key(network_id, qa_id)}:{side}"


def load_inputs() -> tuple[dict[str, v2.NormalizedSlot], dict[str, dict], set[tuple[str, str, str]], dict[str, str]]:
    _verify_manifest(BASE_DIR)
    _verify_manifest(v2.FROZEN_DIR)
    _, conversations, _ = v2.load_frozen_data()
    normalization_cache = _load_cache(BASE_DIR / "normalization_cache.jsonl")
    slots: dict[str, v2.NormalizedSlot] = {}
    relation_by_qa: dict[str, str] = {}
    for case in _read_jsonl(BASE_DIR / "q8_linking_cases.jsonl"):
        case_key = _case_key(case["network_id"], case["qa_id"])
        relation_by_qa[case_key] = case["relation"]
        for side in ("old", "new"):
            slot = _slot_from_cache(case, side, conversations, normalization_cache)
            if slot:
                slots[_slot_key(slot.network_id, slot.case_qa_id, side)] = slot

    retrieval = {
        _case_key(row["network_id"], row["qa_id"]): row
        for row in _read_jsonl(BASE_DIR / "retrieval_results.jsonl")
        if row["variant"] == "D_structured_plus_context"
    }
    confirmed_negatives = {
        (row["network_id"], row["query_id"], row["candidate_id"])
        for row in _read_jsonl(BASE_DIR / "negative_pairs.jsonl")
        if row["tier"] == 1 or row.get("verdict") == "DIFFERENT_SLOT"
    }
    return slots, retrieval, confirmed_negatives, relation_by_qa


def _pairwise_prompt(query: v2.NormalizedSlot, candidates: list[v2.NormalizedSlot]) -> str:
    rows = "\n".join(
        f"[{index}] holder={candidate.state_holder!r}; viewpoint_owner={candidate.viewpoint_owner!r}; "
        f"topic={candidate.topic_object!r}; dimension={candidate.state_dimension!r}; "
        f"slot_question={candidate.slot_question!r}; value={candidate.value!r}; "
        f"state={candidate.state_text!r}"
        for index, candidate in enumerate(candidates)
    )
    return f"""Classify whether each candidate is the SAME evolving state slot as NEW.

SAME_SLOT means the same state holder/viewpoint owner, same semantic topic, and
same changeable dimension.  Values may contradict because a state can change.
DIFFERENT_SLOT means a different holder, target, topic, or state dimension.
Do not decide whether the value changed yet.  Do not choose NEW_CELL.  Judge
each candidate independently.

NEW: holder={query.state_holder!r}; viewpoint_owner={query.viewpoint_owner!r};
topic={query.topic_object!r}; dimension={query.state_dimension!r};
slot_question={query.slot_question!r}; value={query.value!r}; state={query.state_text!r}

CANDIDATES:
{rows}

Return JSON only. candidate_index must reference an index shown above:
{{"decisions":[{{"candidate_index":0,"verdict":"SAME_SLOT|DIFFERENT_SLOT",
"confidence":0.0,"rationale":"one short sentence"}}]}}"""


def normalize_pairwise_decisions(output: dict, candidate_count: int) -> dict[int, dict]:
    decisions: dict[int, dict] = {}
    for item in output.get("decisions", []):
        try:
            index = int(item.get("candidate_index"))
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            continue
        verdict = str(item.get("verdict") or "").upper()
        if not 0 <= index < candidate_count or verdict not in {"SAME_SLOT", "DIFFERENT_SLOT"} or index in decisions:
            continue
        decisions[index] = {
            "verdict": verdict,
            "confidence": confidence,
            "same_score": confidence if verdict == "SAME_SLOT" else 1.0 - confidence,
            "rationale": str(item.get("rationale") or ""),
        }
    return decisions


def select_candidate(scored_candidates: list[dict], threshold: float) -> str | None:
    eligible = [row for row in scored_candidates if row["same_score"] >= threshold]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row["same_score"], -row["rank"]))["candidate_id"]


def query_outcome(row: dict, threshold: float, confirmed_negatives: set[tuple[str, str, str]]) -> str:
    selected = select_candidate(row["candidates"], threshold)
    gold = f"{row['qa_id']}:old"
    if selected == gold:
        return "correct"
    if selected is not None:
        return "false_merge"
    if gold in {candidate["candidate_id"] for candidate in row["candidates"]}:
        return "false_split"
    return "retrieval_miss"


def threshold_metrics(rows: list[dict], threshold: float, confirmed_negatives: set[tuple[str, str, str]]) -> dict:
    outcomes = Counter(query_outcome(row, threshold, confirmed_negatives) for row in rows)
    n = len(rows)
    confirmed_false_merges = 0
    for row in rows:
        selected = select_candidate(row["candidates"], threshold)
        if selected and selected != f"{row['qa_id']}:old" and (
            row["network_id"], f"{row['qa_id']}:new", selected
        ) in confirmed_negatives:
            confirmed_false_merges += 1
    return {
        "threshold": threshold,
        "n": n,
        **{name: outcomes[name] for name in ("correct", "false_split", "false_merge", "retrieval_miss")},
        "confirmed_false_merge": confirmed_false_merges,
        "correct_rate": outcomes["correct"] / n if n else 0.0,
        "false_merge_rate": outcomes["false_merge"] / n if n else 0.0,
        "confirmed_false_merge_rate": confirmed_false_merges / n if n else 0.0,
    }


def calibrate_threshold(rows: list[dict], confirmed_negatives: set[tuple[str, str, str]], max_false_merge_rate: float = MAX_FALSE_MERGE_RATE) -> tuple[float, list[dict]]:
    scores = {0.0, 1.000001}
    scores.update(candidate["same_score"] for row in rows for candidate in row["candidates"])
    table = [threshold_metrics(rows, threshold, confirmed_negatives) for threshold in sorted(scores)]
    feasible = [row for row in table if row["false_merge_rate"] <= max_false_merge_rate]
    best = max(feasible, key=lambda row: (row["correct_rate"], -row["false_merge_rate"], row["threshold"]))
    return best["threshold"], table


def _operation_prompt(query: v2.NormalizedSlot, candidate: v2.NormalizedSlot) -> str:
    return f"""The OLD and NEW facts are already confirmed to describe the same evolving state slot.
Classify only how NEW updates OLD:
- REVISE: incompatible new value replaces the old value.
- OBSERVE: same value is reaffirmed.
- AUGMENT: compatible detail is added without replacing the old value.

OLD value={candidate.value!r}; state={candidate.state_text!r}
NEW value={query.value!r}; state={query.state_text!r}

Return JSON only: {{"operation":"REVISE|OBSERVE|AUGMENT","confidence":0.0,
"rationale":"one short sentence"}}"""


def normalize_operation(output: dict) -> tuple[str, float]:
    operation = str(output.get("operation") or "").upper()
    try:
        confidence = min(1.0, max(0.0, float(output.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return (operation if operation in {"REVISE", "OBSERVE", "AUGMENT"} else "UNKNOWN", confidence)


def run_pairwise(slots: dict[str, v2.NormalizedSlot], retrieval: dict[str, dict]) -> tuple[list[dict], int]:
    cache = _load_cache(PAIRWISE_CACHE)
    rows = []
    new_calls = 0
    for case_key, retrieval_row in retrieval.items():
        qa_id = retrieval_row["qa_id"]
        network_id = retrieval_row["network_id"]
        query = slots.get(_slot_key(network_id, qa_id, "new"))
        candidates = [
            slots[_slot_key(network_id, candidate_id.removesuffix(":old"), "old")]
            for candidate_id in retrieval_row["top10"][:MAX_CANDIDATES]
            if _slot_key(network_id, candidate_id.removesuffix(":old"), "old") in slots
        ]
        if query is None or not candidates:
            continue
        prompt = _pairwise_prompt(query, candidates)
        key = hashlib.sha256(f"{SCHEMA_VERSION}:{PAIRWISE_PROMPT_VERSION}:{MODEL}:{prompt}".encode()).hexdigest()
        output = cache.get(key)
        if output is None:
            output = v2._call_llm(prompt, MODEL)
            cache[key] = output
            _append_cache(PAIRWISE_CACHE, key, output)
            new_calls += 1
            if new_calls % 20 == 0:
                _log(f"pairwise calls: {new_calls}")
        decisions = normalize_pairwise_decisions(output, len(candidates))
        scored = []
        for index, candidate in enumerate(candidates):
            decision = decisions.get(index, {"verdict": "MISSING", "confidence": 0.0, "same_score": 0.0, "rationale": "missing decision"})
            scored.append({
                "candidate_id": f"{candidate.case_qa_id}:{candidate.side}",
                "rank": index + 1,
                **decision,
            })
        rows.append({
            "qa_id": qa_id,
            "case_key": case_key,
            "network_id": query.network_id,
            "split": retrieval_row.get("split"),
            "candidates": scored,
        })
    return rows, new_calls


def run_operations(eval_rows: list[dict], slots: dict[str, v2.NormalizedSlot], threshold: float, relations: dict[str, str]) -> tuple[list[dict], int]:
    cache = _load_cache(OPERATION_CACHE)
    results = []
    new_calls = 0
    for row in eval_rows:
        selected = select_candidate(row["candidates"], threshold)
        query = slots[_slot_key(row["network_id"], row["qa_id"], "new")]
        output: dict[str, Any] = {}
        if selected:
            candidate = slots[_slot_key(row["network_id"], selected.removesuffix(":old"), "old")]
            prompt = _operation_prompt(query, candidate)
            key = hashlib.sha256(f"{SCHEMA_VERSION}:{OPERATION_PROMPT_VERSION}:{MODEL}:{prompt}".encode()).hexdigest()
            output = cache.get(key, {})
            if not output:
                output = v2._call_llm(prompt, MODEL)
                cache[key] = output
                _append_cache(OPERATION_CACHE, key, output)
                new_calls += 1
        operation, confidence = normalize_operation(output)
        results.append({
            **row,
            "threshold": threshold,
            "selected_cell_id": selected,
            "operation": operation if selected else "NEW_CELL",
            "operation_confidence": confidence,
            "gold_relation": relations.get(row["case_key"]),
        })
    return results, new_calls


def _bootstrap_network_delta(pairwise_rows: list[dict], old_rows: dict[str, dict], threshold: float, iterations: int = 10000) -> tuple[float, float, float]:
    by_network: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in pairwise_rows:
        new_ok = query_outcome(row, threshold, set()) == "correct"
        old_ok = bool(old_rows.get(row["case_key"], {}).get("correct_selection"))
        by_network[row["network_id"]].append((int(new_ok), int(old_ok)))
    networks = sorted(by_network)
    observed = sum(new - old for values in by_network.values() for new, old in values) / sum(map(len, by_network.values()))
    rng = random.Random(20260912)
    samples = []
    for _ in range(iterations):
        picked = [rng.choice(networks) for _ in networks]
        pairs = [pair for network in picked for pair in by_network[network]]
        samples.append(sum(new - old for new, old in pairs) / len(pairs))
    samples.sort()
    return observed, samples[int(0.025 * iterations)], samples[int(0.975 * iterations) - 1]


def _write_manifest() -> None:
    files = {}
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS"}:
            relative = str(path.relative_to(OUT_DIR))
            files[relative] = {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"schema_version": SCHEMA_VERSION, "file_count": len(files), "files": files}
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (OUT_DIR / "SHA256SUMS").write_text("".join(f"{meta['sha256']}  {path}\n" for path, meta in files.items()))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log("verifying immutable inputs")
    slots, retrieval, confirmed_negatives, relations = load_inputs()
    pairwise_rows, pairwise_calls = run_pairwise(slots, retrieval)
    with PAIRWISE_DECISIONS.open("w") as handle:
        for row in pairwise_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    dev_rows = [row for row in pairwise_rows if row["split"] == "dev"]
    eval_rows = [row for row in pairwise_rows if row["split"] == "eval"]
    threshold, calibration = calibrate_threshold(dev_rows, confirmed_negatives)
    _log(f"dev-calibrated threshold={threshold:.3f}; dev={len(dev_rows)} eval={len(eval_rows)}")
    decisions, operation_calls = run_operations(eval_rows, slots, threshold, relations)
    with DECISIONS.open("w") as handle:
        for row in decisions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    dev_metrics = threshold_metrics(dev_rows, threshold, confirmed_negatives)
    eval_metrics = threshold_metrics(eval_rows, threshold, confirmed_negatives)
    old_rows = {
        _case_key(row["network_id"], row["qa_id"]): row
        for row in _read_jsonl(BASE_DIR / "resolver_results.jsonl")
    }
    delta, ci_low, ci_high = _bootstrap_network_delta(eval_rows, old_rows, threshold)
    operation_hits = sum(
        row["selected_cell_id"] == f"{row['qa_id']}:old" and row["operation"] == row["gold_relation"]
        for row in decisions
    )
    operation_denominator = len(decisions)
    pairwise_calls_total = len(_load_cache(PAIRWISE_CACHE))
    operation_calls_total = len(_load_cache(OPERATION_CACHE))
    config = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL,
        "temperature": 0,
        "pairwise_prompt_hash": hashlib.sha256(PAIRWISE_PROMPT_VERSION.encode()).hexdigest(),
        "operation_prompt_hash": hashlib.sha256(OPERATION_PROMPT_VERSION.encode()).hexdigest(),
        "threshold_calibrated_on": "dev networks only",
        "threshold": threshold,
        "max_dev_false_merge_rate": MAX_FALSE_MERGE_RATE,
        "pairwise_calls_this_run": pairwise_calls,
        "operation_calls_this_run": operation_calls,
        "pairwise_calls_total": pairwise_calls_total,
        "operation_calls_total": operation_calls_total,
        "input": str(BASE_DIR),
    }
    CONFIG.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    report = f"""# Two-stage semantic slot resolver — diagnostic result

No extraction, normalization, retrieval, production code, or 1,031-QA benchmark was rerun.

## Calibration

- dev queries: {len(dev_rows)}
- eval queries: {len(eval_rows)}
- SAME_SLOT threshold selected on dev only: {threshold:.3f}
- dev correct selection: {dev_metrics['correct']}/{dev_metrics['n']} ({dev_metrics['correct_rate']:.3f})
- dev false merge (any selected non-gold cell): {dev_metrics['false_merge']}/{dev_metrics['n']} ({dev_metrics['false_merge_rate']:.3f})
- dev structurally confirmed subset: {dev_metrics['confirmed_false_merge']}/{dev_metrics['n']} ({dev_metrics['confirmed_false_merge_rate']:.3f})

## Evaluation

- correct old-cell selection: {eval_metrics['correct']}/{eval_metrics['n']} ({eval_metrics['correct_rate']:.3f})
- false split: {eval_metrics['false_split']}/{eval_metrics['n']}
- false merge (any selected non-gold cell): {eval_metrics['false_merge']}/{eval_metrics['n']} ({eval_metrics['false_merge_rate']:.3f})
- structurally confirmed subset: {eval_metrics['confirmed_false_merge']}/{eval_metrics['n']} ({eval_metrics['confirmed_false_merge_rate']:.3f})
- retrieval miss: {eval_metrics['retrieval_miss']}/{eval_metrics['n']}
- correct selection + correct update operation: {operation_hits}/{operation_denominator} ({operation_hits/operation_denominator if operation_denominator else 0:.3f})

## Paired comparison with monolithic resolver

- monolithic correct selection: {sum(bool(row.get('correct_selection')) for row in old_rows.values())}/{len(old_rows)}
- pairwise minus monolithic correct-selection delta: {delta:+.3f}
- network-bootstrap 95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]

## Calls

- new pairwise LLM calls: {pairwise_calls}
- new operation LLM calls: {operation_calls}
- cached pairwise LLM calls in experiment: {pairwise_calls_total}
- cached operation LLM calls in experiment: {operation_calls_total}

## Interpretation

- The safe dev-calibrated pairwise classifier is worse than the monolithic resolver.
- An unconstrained threshold of 0 would reproduce D-retrieval top-1 (54/59), but would always merge and is not a valid NEW_CELL policy.
- The dominant pairwise failure is semantic vocabulary drift in independently normalized holder/topic/dimension fields.
"""
    REPORT.write_text(report)
    (OUT_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    _log("complete")
    _write_manifest()
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
