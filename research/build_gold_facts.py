"""Builds gold_facts.jsonl from the human-annotated review sheet
(research/gold_review_sheet_annotated.md -- gitignored, real chat content:
names, health, politics, self-harm mentions).

accept + edit -> label positive. reject -> label rejected_candidate (kept,
not discarded -- useful later to check whether the forced extractor
proposes junk the human already rejected once, but deliberately NOT called
"hard_negative": a human rejecting a candidate doesn't mean "the extractor
must never produce this," just that this particular framing/claim didn't
hold up -- calling it a confirmed negative would overclaim). ambiguous
attribution -> excluded entirely, and so is #31 (marked "edit" but the
edit itself doesn't resolve the ambiguity -- unclear whose name is claimed
to come from an Indian barista).

On top of the human decisions, a hand-curated set of ACCEPTED opinion-kind
claims that stated a subjective view as if it were an objective fact get
their attribution fixed -- see ATTRIBUTION_FIXES. Found by scanning every
accepted opinion-kind claim for missing "считает"/"по словам"/"написал"
framing: 4 named directly by the user (Крым x2, Гордей, Ксюша), 3 more
found by extending the same check to the rest of the accepted opinion pool
(#19/#22 Israel opinions, #42 handcuffs opinion). All 11 changes (4 edits +
exclusion of #31 + 7 attribution fixes) were explicitly confirmed by the
user before freezing. Manual fix, not an LLM call -- consistent with the
"no more automated passes on this pool" decision from the review that
found the dedup attribution-collapse bug.

subject_key is re-derived here via participants.resolve_subject_for_fact
(deterministic DB lookup, no LLM, no writes) rather than trusted from the
sheet's display text, so gold_facts.jsonl is self-contained and doesn't
depend on the JSON cache's now-possibly-stale ordering.

Usage: python3 -m research.build_gold_facts
"""
import hashlib
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import config  # noqa: triggers dotenv load
from participants import resolve_subject_for_fact
from research.gold_candidate_builder import CHAT_ID

ANNOTATED_PATH = "/Users/yaroslavlikh/summary_sov/research/gold_review_sheet_annotated.md"
OUT_PATH = "/tmp/gold_facts.jsonl"
CHECKSUM_PATH = "/tmp/gold_facts.sha256"

EDITS = {
    8: "1 сентября 2026 года Ярик сообщил, что у него продолжаются летние каникулы.",
    20: "Ярик пошутил, что постарается не потерять любовь к текиле до зимы.",
    25: "Ярик предложил в будущем добавить боту чатовый режим.",
    34: "На 3 сентября 2026 года Ярик пользовался VPN Sota.",
}
EXCLUDE_DESPITE_EDIT = {31}

ATTRIBUTION_FIXES = {
    17: "Ярик написал, что считает Крым частью России.",
    18: "Тигмен написал, что считает Крым частью России.",
    19: "Тигмен написал, что, по его мнению, Израиль совершал военные преступления.",
    22: "Тигмен написал, что, по его мнению, Израиль не совершал военные преступления.",
    42: "Сахаи написал, что считает наручники из БК плохими.",
    44: "Ярик назвал Гордея хуесосом.",
    45: "Игорь написал, что Ксюша сейчас хороша.",
}
# #22 was labeled past_event by the verifier but its content ("Израиль не
# совершал военные преступления") is clearly the same kind of opinion as
# #19 -- corrected alongside the attribution fix.
KIND_FIXES = {22: "opinion"}

DECISION_LABELS = ["accept", "edit", "reject", "ambiguous attribution"]
CHECKED_MARKS = {"+", "x", "X", "-"}


def _parse_annotated(path):
    text = open(path).read()
    blocks = re.split(r"\n---\n", text)
    items = []
    for block in blocks:
        m = re.search(r"^### (\d+)\.", block, re.MULTILINE)
        if not m:
            continue
        num = int(m.group(1))

        def field(name, blk=block):
            fm = re.search(rf"^{name}: (.*)$", blk, re.MULTILINE)
            return fm.group(1).strip() if fm else None

        sources = []
        for sm in re.finditer(r"^- \[(\d+)\] ([^:]+): (.*)$", block, re.MULTILINE):
            sources.append({"message_id": int(sm.group(1)), "author": sm.group(2).strip(), "text": sm.group(3).strip()})

        decisions_found = []
        for label in DECISION_LABELS:
            for dm in re.finditer(rf"- \[(.)\] {re.escape(label)}", block):
                if dm.group(1) in CHECKED_MARKS:
                    decisions_found.append(label)

        items.append({
            "num": num,
            "claim": field("Claim"),
            "kind": field("Kind"),
            "subject": field("Subject"),
            "epistemic_status": field("Epistemic status"),
            "support": field("Suggested support"),
            "observed_at": field(r"Observed at \(source message dates\)"),
            "sources": sources,
            "decisions_found": decisions_found,
            "decision": decisions_found[0] if len(decisions_found) == 1 else None,
        })
    return items


def build():
    items = _parse_annotated(ANNOTATED_PATH)
    print(f"Parsed {len(items)} annotated items", file=sys.stderr)

    for item in items:
        if len(item["decisions_found"]) != 1:
            print(f"WARNING: item #{item['num']} has {len(item['decisions_found'])} decision marks: {item['decisions_found']}", file=sys.stderr)

    tally = Counter(i["decision"] for i in items)
    print(f"Decision tally: {dict(tally)}", file=sys.stderr)
    expected = {"accept": 40, "edit": 5, "reject": 24, "ambiguous attribution": 3}
    if dict(tally) != expected:
        print(f"WARNING: tally mismatch, expected {expected}", file=sys.stderr)

    label_map = {"accept": "positive", "edit": "positive", "reject": "rejected_candidate"}

    facts = []
    changed = []
    excluded = []
    for item in items:
        num = item["num"]
        if item["decision"] == "ambiguous attribution":
            excluded.append((num, item["claim"], "ambiguous attribution"))
            continue
        if num in EXCLUDE_DESPITE_EDIT:
            excluded.append((num, item["claim"], "edit doesn't resolve genuine ambiguity"))
            continue
        if item["decision"] not in ("accept", "edit", "reject"):
            print(f"WARNING: item #{num} has no single clear decision, skipped entirely", file=sys.stderr)
            continue

        original_claim = item["claim"]
        claim = original_claim
        if num in EDITS:
            changed.append((num, original_claim, EDITS[num], "manual edit (user-specified, confirmed)"))
            claim = EDITS[num]
        elif num in ATTRIBUTION_FIXES:
            changed.append((num, original_claim, ATTRIBUTION_FIXES[num], "attribution fix (confirmed)"))
            claim = ATTRIBUTION_FIXES[num]

        kind = KIND_FIXES.get(num, item["kind"])
        label = label_map[item["decision"]]

        subject_raw = (item["subject"] or "")
        subject_unresolved = "⚠" in subject_raw
        subject = subject_raw.split("⚠")[0].strip()
        source_ids = [s["message_id"] for s in item["sources"]]

        resolved = resolve_subject_for_fact(CHAT_ID, subject, source_ids) if subject else None
        subject_key = resolved[0] if resolved else None

        facts.append({
            "id": num,
            "label": label,
            "human_decision": item["decision"],
            "claim": claim,
            "original_claim": original_claim,
            "kind": kind,
            "subject": subject,
            "subject_key": subject_key,
            "subject_unresolved": subject_unresolved,
            "epistemic_status": item["epistemic_status"],
            "observed_at": item["observed_at"],
            "source_message_ids": source_ids,
            "sources": item["sources"],
        })

    with open(OUT_PATH, "w") as f:
        for fact in facts:
            f.write(json.dumps(fact, ensure_ascii=False) + "\n")

    checksum = hashlib.sha256(open(OUT_PATH, "rb").read()).hexdigest()
    with open(CHECKSUM_PATH, "w") as f:
        f.write(f"{checksum}  {OUT_PATH}\n")

    n_pos = sum(1 for f in facts if f["label"] == "positive")
    n_neg = sum(1 for f in facts if f["label"] == "rejected_candidate")
    multi_source = sum(1 for f in facts if f["label"] == "positive" and len(f["source_message_ids"]) > 1)
    n_subject_unresolved = sum(1 for f in facts if f["subject_unresolved"])
    n_subject_key_none = sum(1 for f in facts if f["subject_key"] is None and not f["subject_unresolved"])

    print(f"\nWrote {len(facts)} facts to {OUT_PATH}", file=sys.stderr)
    print(f"  positive: {n_pos} (of which multi-source: {multi_source})", file=sys.stderr)
    print(f"  rejected_candidate: {n_neg}", file=sys.stderr)
    print(f"  excluded: {len(excluded)} -- {[(n, reason) for n, _, reason in excluded]}", file=sys.stderr)
    print(f"  subject_unresolved (bot/group/third-party, not a real ambiguity): {n_subject_unresolved}", file=sys.stderr)
    if n_subject_key_none:
        print(f"  WARNING: {n_subject_key_none} facts have subject_key=None but subject_unresolved=False (genuine resolver miss, check manually)", file=sys.stderr)
    print(f"  checksum (sha256): {checksum}", file=sys.stderr)

    print("\n=== Changed claims (all confirmed) ===", file=sys.stderr)
    for num, old, new, reason in changed:
        print(f"#{num} [{reason}]", file=sys.stderr)
        print(f"  was: {old}", file=sys.stderr)
        print(f"  now: {new}", file=sys.stderr)

    return facts, changed


if __name__ == "__main__":
    build()
