"""Read-only review sheets for a human audit of EverMemBench events and attach decisions.

Writes empty verdict columns only; no model assigns any verdict. The sheets contain benchmark
text and stay under .research_runs (not tracked by git).

    python3 -m research.final_sprint_audit_sheets
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from research import evermembench_episode_run as base
from research.paper_benchmark_common import jsonl, sha256_file

ROOT = base.ROOT
SOURCE = ROOT / ".research_runs" / "frozen" / "evermembench_temporal_episodes_official_v1_20260915_021550_MSK" / "run"
OUT = ROOT / ".research_runs" / "final_sprint_audit_sheets_v1"
SEED = 20260915
PER_CELL = 5
EVENT_DECISIONS = ("create", "attach_revise", "attach_augment", "attach_reaffirm")
ATTACH_DECISIONS = ("NEW_EPISODE", "ATTACH_REVISE", "ATTACH_AUGMENT", "ATTACH_REAFFIRM")
EVENT_VERDICTS = ("paraphrase_valid", "owner_subject_correct", "temporal_grounding_correct",
                  "quote_entails_event", "provenance_sufficient", "reviewer_notes")
ATTACH_VERDICTS = ("same_episode", "decision_relation_correct", "false_merge", "false_split",
                   "provenance_sufficient", "reviewer_notes")


def stratified(items: list[dict], strata: tuple[str, ...], rng: random.Random) -> list[dict]:
    """PER_CELL items per (stratum, project, time bucket); a short cell borrows from the other bucket."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        cells[(item["stratum"], item["project"], item["time_bucket"])].append(item)
    chosen: list[dict] = []
    for stratum in strata:
        for project in base.BATCHES:
            picked: list[dict] = []
            for bucket in ("early", "late"):
                pool = sorted(cells[(stratum, project, bucket)], key=lambda x: x["id"])
                picked += rng.sample(pool, min(PER_CELL, len(pool)))
            if len(picked) < 2 * PER_CELL:
                rest = sorted((x for b in ("early", "late") for x in cells[(stratum, project, b)] if x not in picked), key=lambda x: x["id"])
                picked += rng.sample(rest, min(2 * PER_CELL - len(picked), len(rest)))
            chosen += picked
    return chosen


def sources_text(event: dict, raw_by_id: dict) -> str:
    return "\n".join(f"{s}: {raw_by_id[s].rendered}" for s in event["source_turn_ids"] if s in raw_by_id)


def main() -> None:
    rng = random.Random(SEED)
    _frame, _raw_by_topic, raw_by_id = base.load_messages()
    episodes = jsonl(SOURCE / "episodes.jsonl")
    decisions = jsonl(SOURCE / "attach_decisions.jsonl")
    event_meta, episode_of = {}, {}
    for episode in episodes:
        for item in episode["events"]:
            event_meta[item["event"]["event_id"]] = (item["event"], item["decision"])
            episode_of[item["event"]["event_id"]] = episode
    medians = {}
    for project in base.BATCHES:
        times = sorted(e["observed_at"] for e, _d in event_meta.values() if e["network_id"] == project)
        medians[project] = times[len(times) // 2]
    bucket = lambda e: "early" if e["observed_at"] < medians[e["network_id"]] else "late"

    event_items = [{"id": eid, "stratum": decision, "project": event["network_id"], "time_bucket": bucket(event)}
                   for eid, (event, decision) in event_meta.items()]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "events_review.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "stratum_decision", "project", "time_bucket", "event_id", "episode_id", "observed_at",
                         "event_type", "temporal_mode", "viewpoint_owner", "subject", "event_text", "evidence_quotes",
                         "source_messages", *EVENT_VERDICTS])
        for index, item in enumerate(stratified(event_items, EVENT_DECISIONS, rng), 1):
            event, decision = event_meta[item["id"]]
            quotes = "\n".join(f"{q['turn_id']}: {q['quote']}" for q in event["evidence"])
            writer.writerow([f"E{index:03d}", decision, item["project"], item["time_bucket"], item["id"],
                             episode_of[item["id"]]["episode_id"], event["observed_at"], event["event_type"], event["temporal_mode"],
                             event["viewpoint_owner"], event["subject"], event["event_text"], quotes,
                             sources_text(event, raw_by_id), *[""] * len(EVENT_VERDICTS)])

    position = defaultdict(dict)
    for index, row in enumerate(decisions):
        position[row["network_id"]][row["event_id"]] = index
    episodes_by_id = {e["episode_id"]: e for e in episodes}

    def history(episode_id: str, before: int, network: str) -> str:
        episode = episodes_by_id[episode_id]
        earlier = [i["event"] for i in episode["events"] if position[network].get(i["event"]["event_id"], 10**9) < before][-5:]
        return "\n".join(f"  [{e['observed_at']}] {e['viewpoint_owner'] or '?'}: {e['event_text']}" for e in earlier)

    attach_items = [{"id": row["event_id"], "stratum": row["decision"], "project": row["network_id"],
                     "time_bucket": bucket(event_meta[row["event_id"]][0]), "row": row}
                    for row in decisions if row["decision"] != "NEW_EPISODE" or row["n_candidates_shown"] > 0]
    with open(OUT / "attach_review.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "decision", "project", "time_bucket", "event_id", "observed_at", "new_event_owner",
                         "new_event_subject", "new_event_type", "new_event_text", "new_event_sources", "target_episode_id",
                         "candidate_episodes_at_decision_time", "confidence", *ATTACH_VERDICTS])
        for index, item in enumerate(stratified(attach_items, ATTACH_DECISIONS, rng), 1):
            row = item["row"]
            event, _decision = event_meta[row["event_id"]]
            before = position[row["network_id"]][row["event_id"]]
            candidates = "\n\n".join(
                f"{cid} (owner={episodes_by_id[cid]['viewpoint_owner']!r}, subject={episodes_by_id[cid]['subject']!r})\n"
                f"{history(cid, before, row['network_id'])}" for cid in row["candidate_episode_ids"])
            writer.writerow([f"A{index:03d}", row["decision"], row["network_id"], item["time_bucket"], row["event_id"],
                             event["observed_at"], event["viewpoint_owner"], event["subject"], event["event_type"], event["event_text"],
                             sources_text(event, raw_by_id), row["target_episode_id"], candidates, row["confidence"],
                             *[""] * len(ATTACH_VERDICTS)])

    (OUT / "README.md").write_text(
        "# Review sheets (not yet audited)\n\n"
        "Stratified samples for a human audit; every verdict column is empty. Do not call these results a human audit "
        "until a person has filled them in.\n\n"
        f"- source snapshot: `{SOURCE.parent.name}` (read only); seed {SEED}; {PER_CELL} items per decision x project x time bucket.\n"
        "- time bucket: event observed before (early) or after (late) the median observation time of its project.\n"
        "- attach sheet: NEW_EPISODE rows only where the arbiter was shown candidates; candidate histories are reconstructed "
        "at decision time from the sealed decision order; arbiter rationales are omitted to avoid anchoring.\n\n"
        "## events_review.csv verdicts (yes / no / unclear)\n\n"
        "- paraphrase_valid: event_text is a faithful self-contained paraphrase of the quoted sources.\n"
        "- owner_subject_correct: viewpoint owner and subject are the right people or objects.\n"
        "- temporal_grounding_correct: observed_at and temporal_mode match what the sources say.\n"
        "- quote_entails_event: the quotes alone support the event_text.\n"
        "- provenance_sufficient: the linked source messages are enough to verify the event.\n\n"
        "## attach_review.csv verdicts (yes / no / unclear)\n\n"
        "- same_episode: the new event belongs to the target (or, for NEW_EPISODE, to none of the candidates).\n"
        "- decision_relation_correct: REVISE / REAFFIRM / AUGMENT / NEW matches the relation.\n"
        "- false_merge: the event was attached to a different storyline.\n"
        "- false_split: a new episode was opened although a candidate continues the same storyline.\n"
        "- provenance_sufficient: sources of the new event and the candidate history suffice to decide.\n")
    manifest = {"seed": SEED, "per_cell": PER_CELL, "source_episodes_sha256": sha256_file(SOURCE / "episodes.jsonl"),
                "source_decisions_sha256": sha256_file(SOURCE / "attach_decisions.jsonl"),
                "files": {p.name: sha256_file(p) for p in sorted(OUT.iterdir()) if p.suffix in (".csv", ".md")}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ("events_review.csv", "attach_review.csv"):
        with open(OUT / name) as handle:
            rows = list(csv.DictReader(handle))
        key = "stratum_decision" if name.startswith("events") else "decision"
        print(name, len(rows), dict(sorted(__import__("collections").Counter(r[key] for r in rows).items())))


if __name__ == "__main__":
    main()
