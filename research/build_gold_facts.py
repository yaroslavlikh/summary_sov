"""Builds gold_facts.jsonl from the human-annotated review sheet
(research/gold_review_sheet_annotated.md -- gitignored, real chat content:
names, health, politics, self-harm mentions).

accept + edit -> positive facts. edit substitutes a hand-fixed claim text
from EDITS below -- the sheet's own "edited_claim" field was left blank in
the annotated copy, the user gave the 5 replacement texts directly in chat
instead. reject -> hard negatives (kept, not discarded -- useful later for
precision checks on the forced extractor). ambiguous attribution ->
excluded entirely.

On top of the human decisions, a hand-curated set of ACCEPTED opinion-kind
claims that stated a subjective view as if it were an objective fact get
their attribution fixed -- see ATTRIBUTION_FIXES. Found by scanning every
accepted opinion-kind claim for missing "считает"/"по словам"/"написал"
framing: 4 were named directly by the user (Крым x2, Гордей, Ксюша), 3 more
found by extending the same check to the rest of the accepted opinion pool
(#19/#22 Israel opinions, #42 handcuffs opinion). This is a manual fix, not
an LLM call -- consistent with the "no more automated passes on this pool"
decision from the review that found the dedup attribution-collapse bug.

Usage: python3 -m research.build_gold_facts
"""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

ANNOTATED_PATH = "/Users/yaroslavlikh/summary_sov/research/gold_review_sheet_annotated.md"
OUT_PATH = "/tmp/gold_facts.jsonl"

EDITS = {
    8: "1 сентября 2026 года Ярик сообщил, что у него продолжаются летние каникулы.",
    20: "Ярик пошутил, что постарается не потерять любовь к текиле до зимы.",
    25: "Ярик предложил в будущем добавить боту чатовый режим.",
    34: "На 3 сентября 2026 года Ярик пользовался VPN Sota.",
}
# #31 was marked "edit" but the user wants it dropped entirely as ambiguous
# (unclear whose name is claimed to come from an Indian barista).
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

    facts = []
    changed = []
    excluded_ambiguous = []
    for item in items:
        num = item["num"]
        if item["decision"] == "ambiguous attribution":
            excluded_ambiguous.append((num, item["claim"]))
            continue
        if num in EXCLUDE_DESPITE_EDIT:
            excluded_ambiguous.append((num, item["claim"]))
            continue
        if item["decision"] not in ("accept", "edit", "reject"):
            print(f"WARNING: item #{num} has no single clear decision, skipped entirely", file=sys.stderr)
            continue

        claim = item["claim"]
        if num in EDITS:
            changed.append((num, claim, EDITS[num], "manual edit (user-specified)"))
            claim = EDITS[num]
        elif num in ATTRIBUTION_FIXES:
            changed.append((num, claim, ATTRIBUTION_FIXES[num], "attribution fix (opinion stated as objective fact)"))
            claim = ATTRIBUTION_FIXES[num]

        kind = KIND_FIXES.get(num, item["kind"])
        label = "positive" if item["decision"] in ("accept", "edit") else "hard_negative"

        subject_raw = item["subject"] or ""
        subject_ambiguous = "⚠" in subject_raw
        subject = subject_raw.split("⚠")[0].strip()

        facts.append({
            "id": num,
            "label": label,
            "claim": claim,
            "kind": kind,
            "subject": subject,
            "subject_ambiguous": subject_ambiguous,
            "epistemic_status": item["epistemic_status"],
            "source_message_ids": [s["message_id"] for s in item["sources"]],
            "sources": item["sources"],
        })

    with open(OUT_PATH, "w") as f:
        for fact in facts:
            f.write(json.dumps(fact, ensure_ascii=False) + "\n")

    n_pos = sum(1 for f in facts if f["label"] == "positive")
    n_neg = sum(1 for f in facts if f["label"] == "hard_negative")
    multi_source = sum(1 for f in facts if f["label"] == "positive" and len(f["source_message_ids"]) > 1)

    print(f"\nWrote {len(facts)} facts to {OUT_PATH}", file=sys.stderr)
    print(f"  positive: {n_pos} (of which multi-source: {multi_source})", file=sys.stderr)
    print(f"  hard_negative: {n_neg}", file=sys.stderr)
    print(f"  excluded (ambiguous): {len(excluded_ambiguous)} -- {[n for n, _ in excluded_ambiguous]}", file=sys.stderr)

    print("\n=== Changed claims ===", file=sys.stderr)
    for num, old, new, reason in changed:
        print(f"#{num} [{reason}]", file=sys.stderr)
        print(f"  was: {old}", file=sys.stderr)
        print(f"  now: {new}", file=sys.stderr)

    return facts, changed


if __name__ == "__main__":
    build()
