"""Rebuilds honest forced-extractor eval metrics from ALREADY-CACHED
extractor output (/tmp/forced_extractor_output.json) and frozen gold
(/tmp/gold_facts.jsonl) -- no LLM calls, no DB queries, no re-running the
extractor. The first report (forced_extractor_eval.py's own evaluate())
overclaimed in two specific, confirmed ways:

  - "loose" (any shared source_message_id) was reported as if it were
    fact recall. It's source coverage: gold #67 (Ярик testing his bot)
    "matched" only because one extracted item happened to cite message
    72983 too -- while stating three entirely different facts (Igor's
    portrait, the bot's own description, Langfuse) about that message.
  - "strict" counted subject_key=None on both sides as a match. Two
    independent resolver misses agreeing they don't know the subject is
    not confirmation they agree on WHO the subject is (#13, #29, #40 were
    wrongly counted this way in the first report).

New, decomposed metrics -- deliberately kept as separate columns rather
than collapsed into one number, since they measure different things:

  - source_any: >=1 shared source_message_id (the old "loose", relabeled
    honestly -- this is a ceiling on recall, not recall itself)
  - source_all: some extracted candidate's source set is a SUPERSET of the
    gold fact's full source set (meaningful mainly for the 6 multi-source
    facts -- did the extractor actually pull in every piece of evidence,
    not just one)
  - subject_match: True only when BOTH sides resolved a non-None
    subject_key and they're equal. None==None is never a match.
  - claim_equivalent / attribution_preserved: hand-annotated for the 19
    source_any-matched pairs (MANUAL_REVIEW below) -- these need reading
    the actual sentences, not something a script can compute. Kept small
    and inspectable on purpose, not automated over all 44.

11/24 rejected_candidate resurfacing is kept as-is, unmodified, per the
instruction that it's a real, separate signal (not a precision claim).

Usage: python3 -m research.reevaluate_forced_extractor
"""
import json
import sys

EXTRACTED_PATH = "/tmp/forced_extractor_output.json"
GOLD_PATH = "/tmp/gold_facts.jsonl"
REPORT_PATH = "/tmp/forced_extractor_report_v2.md"

# Hand-reviewed for each of the 19 gold facts that had >=1 source_any
# match in the first run. claim_equivalent: does the extractor's own
# sentence assert the SAME proposition (not just share a topic/message)?
# attribution_preserved: for facts whose point IS who said/believes it,
# did the extractor's wording keep that framing (vs collapsing it into an
# unattributed objective statement)? None = not applicable (claim isn't
# fundamentally about attribution, e.g. a plain self-reported fact).
MANUAL_REVIEW = {
    13: {"claim_equivalent": True, "attribution_preserved": True,
         "note": "Minor typo only (Влад/Владa), same claim, attribution kept."},
    16: {"claim_equivalent": True, "attribution_preserved": None,
         "note": "Near-verbatim match."},
    17: {"claim_equivalent": True, "attribution_preserved": True,
         "note": "Same proposition, attribution kept (утверждает)."},
    18: {"claim_equivalent": True, "attribution_preserved": True,
         "note": "Same as #17 for Тигмен."},
    19: {"claim_equivalent": True, "attribution_preserved": True,
         "note": "Same proposition, attribution kept."},
    20: {"claim_equivalent": False, "attribution_preserved": None,
         "note": "Lost that it was a JOKE (пошутил -> обещает, changes assertion_mode) "
                 "and lost the explicit tequila referent (родной left untranslated)."},
    22: {"claim_equivalent": True, "attribution_preserved": True,
         "note": "Same proposition, attribution kept."},
    25: {"claim_equivalent": False, "attribution_preserved": None,
         "note": "Source is literally a question (\"...добавить чатовый режим...?\"); "
                 "extractor states it as a firm plan (планирует), overclaiming certainty."},
    29: {"claim_equivalent": True, "attribution_preserved": False,
         "note": "Same content, but dropped \"по словам Ярика\" -- presents his framing as objective fact."},
    33: {"claim_equivalent": True, "attribution_preserved": None,
         "note": "Exact match."},
    35: {"claim_equivalent": True, "attribution_preserved": None,
         "note": "Same claim."},
    40: {"claim_equivalent": True, "attribution_preserved": None,
         "note": "Same group-lore content; gold's own claim also lacks explicit attribution here."},
    43: {"claim_equivalent": False, "attribution_preserved": False,
         "note": "MISATTRIBUTION, not just missing attribution: source [72499] is Тигмен reporting "
                 "what Ваня allegedly said. Extractor's \"Ваня считает\" presents it as Ваня's own "
                 "direct statement -- worse than gold, an attribution-collapse case in its own right."},
    44: {"claim_equivalent": True, "attribution_preserved": True,
         "note": "Same proposition, attribution kept (По словам Ярика...)."},
    45: {"claim_equivalent": False, "attribution_preserved": False,
         "note": "Dropped attribution entirely (\"Ксюша сейчас хороша\" as bare fact) AND got the "
                 "subject wrong (Ксюша instead of Игорь) -- reproduces the exact epistemic-collapse "
                 "bug the gold review specifically fixed for this item."},
    46: {"claim_equivalent": True, "attribution_preserved": None,
         "note": "Same claim (кент/друг synonym)."},
    56: {"claim_equivalent": True, "attribution_preserved": True,
         "note": "Extractor added attribution (Тигмен считает) that gold's own accepted claim lacks -- "
                 "arguably better than gold here."},
    58: {"claim_equivalent": True, "attribution_preserved": None,
         "note": "Same running gag, faithfully reproduced."},
    67: {"claim_equivalent": False, "attribution_preserved": None,
         "note": "FALSE MATCH: gold is about Ярик testing/debugging his bot; extractor's 3 items "
                 "(from the same message 72983) are about Igor's portrait, the bot's own description, "
                 "and Langfuse -- entirely different facts that happen to share one source message."},
}


def _load():
    extracted = json.load(open(EXTRACTED_PATH))
    gold = [json.loads(line) for line in open(GOLD_PATH)]
    return extracted, gold


def _source_any(gold_fact, extracted):
    gold_ids = set(gold_fact["source_message_ids"])
    return [e for e in extracted if gold_ids & set(e["source_message_ids"])]


def _source_all(gold_fact, extracted):
    gold_ids = set(gold_fact["source_message_ids"])
    return [e for e in extracted if gold_ids <= set(e["source_message_ids"])]


def _subject_match(gold_fact, matches):
    gk = gold_fact.get("subject_key")
    for e in matches:
        ek = e.get("subject_key")
        if gk is not None and ek is not None and gk == ek:
            return True
    return False


def run():
    extracted, gold = _load()
    positive = [g for g in gold if g["label"] == "positive"]
    rejected = [g for g in gold if g["label"] == "rejected_candidate"]
    multi_source = [g for g in positive if len(g["source_message_ids"]) > 1]

    rows = []
    for g in positive:
        any_matches = _source_any(g, extracted)
        all_matches = _source_all(g, extracted)
        subj = _subject_match(g, any_matches)
        review = MANUAL_REVIEW.get(g["id"])
        rows.append({
            "id": g["id"], "claim": g["claim"], "kind": g["kind"],
            "n_sources": len(g["source_message_ids"]),
            "source_any": bool(any_matches), "source_all": bool(all_matches),
            "subject_match": subj,
            "claim_equivalent": review["claim_equivalent"] if review else None,
            "attribution_preserved": review["attribution_preserved"] if review else None,
            "note": review["note"] if review else None,
            "matches": any_matches,
        })

    n = len(positive)
    n_source_any = sum(1 for r in rows if r["source_any"])
    n_source_all = sum(1 for r in rows if r["source_all"])
    n_subject_match = sum(1 for r in rows if r["subject_match"])
    n_claim_equiv = sum(1 for r in rows if r["claim_equivalent"])
    n_attrib_needed = sum(1 for r in rows if r["attribution_preserved"] is not None)
    n_attrib_ok = sum(1 for r in rows if r["attribution_preserved"] is True)
    n_fully_recovered = sum(1 for r in rows if r["claim_equivalent"] and r["subject_match"])

    rej_rows = []
    for g in rejected:
        matches = _source_any(g, extracted)
        if matches:
            rej_rows.append({"id": g["id"], "claim": g["claim"], "matches": matches})

    multi_rows = [r for r in rows if r["n_sources"] > 1]

    lines = ["# Forced extractor eval report v2 (corrected, no re-run)", ""]
    lines.append("Metrics recomputed from the SAME cached extractor output as the first report -- "
                  "the extractor was not re-run. The first report's `loose`/`strict` numbers "
                  "overclaimed: `loose` measured source overlap, not fact recall, and `strict` "
                  "counted two unresolved subjects (None==None) as agreement.")
    lines.append("")
    lines.append(f"Positive gold facts: {n} | multi-source: {len(multi_source)} | rejected_candidate: {len(rejected)}")
    lines.append("")
    lines.append("## Decomposed metrics on 44 positive facts")
    lines.append(f"- source_any (>=1 shared source_message_id -- a CEILING, not recall): {n_source_any}/{n} = {n_source_any/n:.0%}")
    lines.append(f"- source_all (all gold sources covered by one extracted candidate): {n_source_all}/{n} = {n_source_all/n:.0%}")
    lines.append(f"- subject_match (both sides resolved, non-None, equal): {n_subject_match}/{n} = {n_subject_match/n:.0%}")
    lines.append(f"- claim_equivalent (hand-reviewed, only among the {n_source_any} source_any matches): {n_claim_equiv}/{n} = {n_claim_equiv/n:.0%}")
    lines.append(f"- attribution_preserved where applicable: {n_attrib_ok}/{n_attrib_needed} of the claims where attribution mattered")
    lines.append(f"- **fully recovered (claim_equivalent AND subject_match): {n_fully_recovered}/{n} = {n_fully_recovered/n:.0%}**")
    lines.append("")

    lines.append("## Multi-source facts (6 total)")
    for g in multi_source:
        r = next((x for x in rows if x["id"] == g["id"]), None)
        lines.append(f"- #{g['id']} [{len(g['source_message_ids'])} sources] source_any={r['source_any']} source_all={r['source_all']} "
                     f"subject_match={r['subject_match']} claim_equivalent={r['claim_equivalent']}")
        lines.append(f"    gold: {g['claim']}")
    n_multi_full = sum(1 for r in multi_rows if r["claim_equivalent"] and r["subject_match"])
    lines.append(f"Fully recovered (claim_equivalent AND subject_match): {n_multi_full}/{len(multi_source)} = {n_multi_full/len(multi_source):.0%}")
    lines.append("")

    lines.append("## Rejected candidates resurfaced (unchanged, separate signal, not precision)")
    lines.append(f"{len(rej_rows)}/{len(rejected)}")
    lines.append("")

    lines.append("## Per-fact detail (all 19 source_any matches, hand-reviewed)")
    for r in rows:
        if not r["source_any"]:
            continue
        lines.append(f"### #{r['id']} [{r['kind']}]")
        lines.append(f"gold: {r['claim']}")
        for m in r["matches"]:
            lines.append(f"extractor: {m['claim']}  (subject_key={m['subject_key']}, sources={m['source_message_ids']})")
        lines.append(f"source_all={r['source_all']} subject_match={r['subject_match']} "
                     f"claim_equivalent={r['claim_equivalent']} attribution_preserved={r['attribution_preserved']}")
        lines.append(f"note: {r['note']}")
        lines.append("")

    lines.append("## Missed entirely (no source_any match)")
    for g in positive:
        if not any(r["id"] == g["id"] and r["source_any"] for r in rows):
            lines.append(f"- #{g['id']} [{g['kind']}] {g['claim']}")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"source_any={n_source_any}/{n}={n_source_any/n:.0%}  source_all={n_source_all}/{n}={n_source_all/n:.0%}  "
          f"subject_match={n_subject_match}/{n}={n_subject_match/n:.0%}  claim_equivalent={n_claim_equiv}/{n}={n_claim_equiv/n:.0%}  "
          f"fully_recovered={n_fully_recovered}/{n}={n_fully_recovered/n:.0%}", file=sys.stderr)
    print(f"multi-source fully_recovered={n_multi_full}/{len(multi_source)}", file=sys.stderr)
    print(f"rejected resurfaced={len(rej_rows)}/{len(rejected)}", file=sys.stderr)
    print(f"Report: {REPORT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    run()
