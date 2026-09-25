"""Factorized semantic-slot gate over the frozen 59-pair evaluation set.

Embedding retrieval is reused unchanged.  The LLM judges holder, viewpoint,
topic, and state dimension separately; values are explicitly not identity.
Threshold selection uses dev networks only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from research import semantic_key_gate as base
from research import semantic_slot_linking_v2 as v2
from research.semantic_slot_pairwise_resolver import _load_cache, _slot_key, load_inputs


OUT_DIR = Path(".research_runs/semantic_key_factorized_gate_v1")
CACHE = OUT_DIR / "gate_cache.jsonl"
DECISIONS = OUT_DIR / "decisions.jsonl"
CONFIG = OUT_DIR / "config.json"
REPORT = OUT_DIR / "report.md"
MODEL = "fast"
SCHEMA_VERSION = "semantic_key_factorized_gate_v1"


def factorized_prompt(query: v2.NormalizedSlot, candidates: list[tuple[dict, v2.NormalizedSlot]]) -> str:
    block = "\n".join(
        f"[{index}] KEY: {row['key_description']}\n"
        f"    holder={candidate.state_holder!r}; viewpoint={candidate.viewpoint_owner!r}; "
        f"topic={candidate.topic_object!r}; dimension={candidate.state_dimension!r}; "
        f"value={candidate.value!r}; fact={candidate.state_text!r}"
        for index, (row, candidate) in enumerate(candidates)
    )
    return f"""Determine whether each candidate tracks the SAME EVOLVING MEMORY SLOT as NEW.

This is identity of a changeable variable, NOT agreement between values.
Opposite or changed values are often positive evidence of the same slot:
- Tigmen's opinion of Gordey: 'bad' -> 'good' is SAME SLOT.
- Seb's knee condition: 'locked' -> '70% functional' is SAME SLOT.
- Marcus's menu size: 'nine courses' -> 'five courses' is SAME SLOT.
Different variables remain different even for the same person:
- Seb's knee condition vs Seb's plan to attend tomorrow is DIFFERENT.
- Marcus's menu size vs Marcus's emotional enthusiasm is DIFFERENT.

For every candidate assess five axes independently.  Use YES when wording differs
but the real referent/dimension is clearly the same.  Use UNCERTAIN only when the
evidence genuinely cannot decide, not merely because the values conflict.

NEW KEY: {base.key_description(query)}
NEW: holder={query.state_holder!r}; viewpoint={query.viewpoint_owner!r};
topic={query.topic_object!r}; dimension={query.state_dimension!r};
value={query.value!r}; fact={query.state_text!r}

CANDIDATES:
{block}

Return JSON only, exactly one comparison per candidate index:
{{"comparisons":[{{"candidate_index":0,
"same_holder":"YES|NO|UNCERTAIN",
"same_viewpoint":"YES|NO|NOT_APPLICABLE|UNCERTAIN",
"same_topic":"YES|NO|UNCERTAIN",
"same_dimension":"YES|NO|UNCERTAIN",
"same_slot":"YES|NO|UNCERTAIN",
"confidence":0.0,"rationale":"one short sentence"}}]}}"""


def normalize_factorized_output(output: dict, candidate_count: int) -> dict[int, dict]:
    normalized = {}
    valid = {"YES", "NO", "UNCERTAIN"}
    for item in output.get("comparisons", []):
        try:
            index = int(item.get("candidate_index"))
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            continue
        if not 0 <= index < candidate_count or index in normalized:
            continue
        holder = str(item.get("same_holder") or "").upper()
        viewpoint = str(item.get("same_viewpoint") or "").upper()
        topic = str(item.get("same_topic") or "").upper()
        dimension = str(item.get("same_dimension") or "").upper()
        same_slot = str(item.get("same_slot") or "").upper()
        if holder not in valid or viewpoint not in valid | {"NOT_APPLICABLE"} or topic not in valid or dimension not in valid or same_slot not in valid:
            continue
        matched = (
            holder == "YES" and viewpoint in {"YES", "NOT_APPLICABLE"}
            and topic == "YES" and dimension == "YES" and same_slot == "YES"
        )
        normalized[index] = {
            "same_holder": holder, "same_viewpoint": viewpoint,
            "same_topic": topic, "same_dimension": dimension,
            "same_slot": same_slot, "confidence": confidence,
            "match_score": confidence if matched else 0.0,
            "rationale": str(item.get("rationale") or ""),
        }
    return normalized


def run_gate(rows: list[dict], slots: dict[str, v2.NormalizedSlot]) -> tuple[list[dict], int]:
    cache = _load_cache(CACHE)
    results, calls = [], 0
    for row in rows:
        candidates = [
            (candidate_row, slots[_slot_key(row["network_id"], candidate_row["candidate_id"].removesuffix(":old"), "old")])
            for candidate_row in row["candidates"]
        ]
        query = slots[_slot_key(row["network_id"], row["qa_id"], "new")]
        prompt = factorized_prompt(query, candidates)
        key = hashlib.sha256(f"{SCHEMA_VERSION}:{MODEL}:{prompt}".encode()).hexdigest()
        output = cache.get(key)
        if output is None:
            output = v2._call_llm(prompt, MODEL)
            base._append_cache(CACHE, key, output)
            cache[key] = output
            calls += 1
            if calls % 20 == 0:
                print(f"factorized gate calls: {calls}", flush=True)
        decisions = normalize_factorized_output(output, len(candidates))
        results.append({
            **row,
            "candidates": [
                {**candidate, **decisions.get(index, {
                    "match_score": 0.0, "confidence": 0.0,
                    "same_holder": "MISSING", "same_viewpoint": "MISSING",
                    "same_topic": "MISSING", "same_dimension": "MISSING",
                    "same_slot": "MISSING", "rationale": "missing comparison",
                })}
                for index, candidate in enumerate(row["candidates"])
            ],
        })
    return results, calls


def _write_manifest() -> None:
    files = {}
    for path in sorted(OUT_DIR.iterdir()):
        if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS"}:
            files[path.name] = {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "files": files}, indent=2) + "\n")
    (OUT_DIR / "SHA256SUMS").write_text("".join(f"{meta['sha256']}  {name}\n" for name, meta in files.items()))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slots, retrieval, negatives, _ = load_inputs()
    rankings = base.build_rankings(slots, retrieval)
    rows, calls = run_gate(rankings, slots)
    dev = [row for row in rows if row["split"] == "dev"]
    evaluation = [row for row in rows if row["split"] == "eval"]
    threshold, calibration = base.calibrate(dev, negatives)
    dev_metrics = base.metrics(dev, threshold, negatives)
    eval_metrics = base.metrics(evaluation, threshold, negatives)
    with DECISIONS.open("w") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "selected": base.select_match(row, threshold)}, ensure_ascii=False) + "\n")
    CONFIG.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "model": MODEL, "top_k": base.TOP_K,
        "threshold": threshold, "calibrated_on": "dev networks only",
        "minimum_dev_precision": base.MIN_PRECISION,
        "maximum_dev_confirmed_no_match_fpr": base.MAX_NO_MATCH_FPR,
        "new_calls": calls,
    }, indent=2) + "\n")
    REPORT.write_text(f"""# Factorized semantic identity gate

Embedding retrieval is unchanged; this measures only the factorized final gate.
Threshold was selected on dev networks and then frozen for eval.

| split | n | precision | recall | correct | false merge | false split | retrieval miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev | {dev_metrics['n']} | {dev_metrics['precision']:.3f} | {dev_metrics['recall']:.3f} | {dev_metrics['correct']} | {dev_metrics['false_merge']} | {dev_metrics['false_split']} | {dev_metrics['retrieval_miss']} |
| eval | {eval_metrics['n']} | {eval_metrics['precision']:.3f} | {eval_metrics['recall']:.3f} | {eval_metrics['correct']} | {eval_metrics['false_merge']} | {eval_metrics['false_split']} | {eval_metrics['retrieval_miss']} |

- threshold: {threshold:.3f}
- eval confirmed no-match FPR: {eval_metrics['confirmed_no_match_fpr']:.3f}
- new LLM calls: {calls}
""")
    _write_manifest()
    print(REPORT)


if __name__ == "__main__":
    main()
