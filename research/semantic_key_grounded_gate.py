"""Provenance-grounded primary semantic-key gate on frozen linking pairs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from research import semantic_key_gate as base
from research import semantic_key_gate_primary as primary
from research import semantic_slot_linking_v2 as v2
from research.semantic_slot_pairwise_resolver import _load_cache, _slot_key, load_inputs


OUT_DIR = Path(".research_runs/semantic_key_grounded_gate_v1")
CACHE = OUT_DIR / "gate_cache.jsonl"
DECISIONS = OUT_DIR / "decisions.jsonl"
CONFIG = OUT_DIR / "config.json"
REPORT = OUT_DIR / "report.md"
SCHEMA_VERSION = "semantic_key_grounded_gate_v1"


def evidence_text(slot: v2.NormalizedSlot, turns: dict[str, dict]) -> str:
    rows = [turns[turn_id] for turn_id in slot.source_turn_ids if turn_id in turns]
    return "\n".join(
        f"[[{row['turn_id']}]] {row['timestamp']} {row['speaker']}: {row['message']}"
        for row in rows
    ) or "(no validated source available)"


def grounded_prompt(
    query: v2.NormalizedSlot, candidates: list[tuple[dict, v2.NormalizedSlot]], turns: dict[str, dict],
) -> str:
    block = "\n\n".join(
        f"[{index}] KEY: {row['key_description']}\nEXISTING FACT: {candidate.state_text}\n"
        f"VALIDATED EVIDENCE:\n{evidence_text(candidate, turns)}"
        for index, (row, candidate) in enumerate(candidates)
    )
    return f"""Decide whether each existing memory key tracks the SAME evolving state as NEW.
You have the original validated messages, not only summaries. Precision matters:
MATCH only for the same holder/viewpoint, semantic topic, and changeable property.
Values may be opposite because beliefs, plans, health, and decisions change.
Use the messages to resolve vague summaries, attribution, and referents. Do not
conflate two different roles, plans, feelings, or properties just because the
same person appears in both.

NEW KEY: {base.key_description(query)}
NEW FACT: {query.state_text}
VALIDATED NEW EVIDENCE:
{evidence_text(query, turns)}

EXISTING CANDIDATES:
{block}

Return JSON only, one judgement per shown index:
{{"decisions":[{{"candidate_index":0,"verdict":"MATCH|NO_MATCH|UNCERTAIN",
"confidence":0.0,"rationale":"one short sentence grounded in the evidence"}}]}}"""


def _turns() -> dict[str, dict]:
    _, conversations, _ = v2.load_frozen_data()
    return {
        str(row.turn_id): {"turn_id": str(row.turn_id), "timestamp": str(row.timestamp),
                           "speaker": str(row.speaker_display_name), "message": str(row.message)}
        for row in conversations.itertuples(index=False)
    }


def run_gate(rows: list[dict], slots: dict[str, v2.NormalizedSlot], turns: dict[str, dict]) -> tuple[list[dict], int]:
    cache = _load_cache(CACHE)
    output_rows, calls = [], 0
    for row in rows:
        candidates = [
            (candidate_row, slots[_slot_key(row["network_id"], candidate_row["candidate_id"].removesuffix(":old"), "old")])
            for candidate_row in row["candidates"]
        ]
        query = slots[_slot_key(row["network_id"], row["qa_id"], "new")]
        prompt = grounded_prompt(query, candidates, turns)
        key = hashlib.sha256(f"{SCHEMA_VERSION}:primary:{prompt}".encode()).hexdigest()
        parsed = cache.get(key)
        if parsed is None:
            parsed = primary._primary_only(prompt)
            base._append_cache(CACHE, key, parsed)
            cache[key] = parsed
            calls += 1
            if calls % 20 == 0:
                print(f"grounded gate calls: {calls}", flush=True)
        decisions = base.normalize_gate_output(parsed, len(candidates))
        output_rows.append({
            **row,
            "candidates": [
                {**candidate, **decisions.get(index, {"verdict": "MISSING", "confidence": 0.0,
                                                      "match_score": 0.0, "rationale": "missing decision"})}
                for index, candidate in enumerate(row["candidates"])
            ],
        })
    return output_rows, calls


def _manifest() -> None:
    files = {}
    for path in sorted(OUT_DIR.iterdir()):
        if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS"}:
            files[path.name] = {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "files": files}, indent=2) + "\n")
    (OUT_DIR / "SHA256SUMS").write_text("".join(f"{meta['sha256']}  {name}\n" for name, meta in files.items()))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slots, retrieval, negatives, _ = load_inputs()
    rows, calls = run_gate(base.build_rankings(slots, retrieval), slots, _turns())
    dev, evaluation = ([r for r in rows if r["split"] == split] for split in ("dev", "eval"))
    threshold, _ = base.calibrate(dev, negatives)
    dev_metrics, eval_metrics = base.metrics(dev, threshold, negatives), base.metrics(evaluation, threshold, negatives)
    with DECISIONS.open("w") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "selected": base.select_match(row, threshold)}, ensure_ascii=False) + "\n")
    CONFIG.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "model": "primary", "top_k": base.TOP_K,
                                  "threshold": threshold, "new_calls": calls,
                                  "calibrated_on": "dev networks only"}, indent=2) + "\n")
    REPORT.write_text(f"""# Provenance-grounded semantic-key gate

Same keys, retrieval candidates, and strict decision rule as the primary gate;
only validated source messages are newly visible to the gate.

| split | n | precision | recall | correct | false merge | false split | retrieval miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev | {dev_metrics['n']} | {dev_metrics['precision']:.3f} | {dev_metrics['recall']:.3f} | {dev_metrics['correct']} | {dev_metrics['false_merge']} | {dev_metrics['false_split']} | {dev_metrics['retrieval_miss']} |
| eval | {eval_metrics['n']} | {eval_metrics['precision']:.3f} | {eval_metrics['recall']:.3f} | {eval_metrics['correct']} | {eval_metrics['false_merge']} | {eval_metrics['false_split']} | {eval_metrics['retrieval_miss']} |

- threshold: {threshold:.3f}
- eval confirmed no-match FPR: {eval_metrics['confirmed_no_match_fpr']:.3f}
- new LLM calls: {calls}
""")
    _manifest()
    print(REPORT)


if __name__ == "__main__":
    main()
