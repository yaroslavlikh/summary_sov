"""Read-only, whole-corpus structural comparison: baseline time-gap heuristic
vs the semantic pilot, over EVERY real message in chat -1002335227490 that
has both message_date and embedding (not just the one hand-labeled incident
window in benchmark.py).

No ground truth exists for the full corpus (nobody hand-labeled 1000+
messages), so precision/recall/F1 are NOT computed here -- this is a
structural/qualitative scan: cluster size distributions, assignment method
mix, and a manual look at the largest baseline clusters (the ones most
likely to be silently contaminated, since baseline never splits a fast
back-and-forth no matter how many topics it actually covers).

Writes nothing. SELECT only. Usage: python3 -m research.full_corpus_scan
"""
import sys
from collections import Counter

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

from research.conversation_disentanglement import semantic_pilot, time_gap_baseline
from research.db_loader import load_messages

CHAT_ID = -1002335227490


def cluster_sizes(mapping):
    sizes = Counter(mapping.values())
    return sizes


def describe(name, sizes):
    values = list(sizes.values())
    print(f"\n--- {name} ---")
    print(f"  n_clusters={len(values)}  avg_size={sum(values)/len(values):.2f}  max_size={max(values)}")
    top = sizes.most_common(5)
    print(f"  top-5 largest clusters (conversation_id: size): {top}")
    hist = Counter()
    for v in values:
        bucket = "1" if v == 1 else ("2-3" if v <= 3 else ("4-9" if v <= 9 else ("10-19" if v <= 19 else "20+")))
        hist[bucket] += 1
    print(f"  size histogram: {dict(sorted(hist.items()))}")


def run():
    print(f"Loading ALL real messages for chat {CHAT_ID} (message_date + embedding present)...")
    messages = load_messages(CHAT_ID)
    print(f"  loaded {len(messages)} messages, span "
          f"{(messages[-1].timestamp - messages[0].timestamp) / 3600:.1f}h "
          f"({len(set(m.thread_id for m in messages))} distinct thread_id values)")

    baseline_pred = time_gap_baseline(messages)
    pilot_pred, pilot_results, tracker = semantic_pilot(messages)

    describe("baseline (<300s time-gap, current production)", cluster_sizes(baseline_pred))
    describe("semantic pilot", cluster_sizes(pilot_pred))

    method_counts = Counter(r.method for r in pilot_results)
    print(f"\npilot assignment method mix: {dict(method_counts)}")
    low_conf = sum(1 for r in pilot_results if r.low_confidence)
    print(f"pilot low_confidence assignments: {low_conf} / {len(pilot_results)}")

    # Manually inspect the largest baseline clusters -- a cluster this big,
    # spanning many minutes/topics without ever hitting a 300s gap, is
    # exactly the shape of the "Наполовину" bug. Show what pilot did with
    # the same messages, and print a topical sample so a human can eyeball
    # whether it's genuinely one conversation or several smashed together.
    by_id = {m.message_id: m for m in messages}
    baseline_sizes = cluster_sizes(baseline_pred)
    print("\n=== top 5 largest baseline clusters: contamination spot-check ===")
    for conv_id, size in baseline_sizes.most_common(5):
        members = sorted([mid for mid, c in baseline_pred.items() if c == conv_id])
        pilot_subclusters = Counter(pilot_pred[mid] for mid in members)
        span_min = (by_id[members[-1]].timestamp - by_id[members[0]].timestamp) / 60
        print(f"\n  baseline cluster {conv_id}: {size} messages, spans {span_min:.1f} min, "
              f"pilot split it into {len(pilot_subclusters)} sub-episode(s)")
        sample_ids = members[:3] + members[len(members)//2:len(members)//2+2] + members[-3:]
        seen = set()
        for mid in sample_ids:
            if mid in seen or mid not in by_id:
                continue
            seen.add(mid)
            m = by_id[mid]
            print(f"    [{mid}] {m.author}: {m.text[:70]!r}  (pilot -> {pilot_pred[mid]})")


if __name__ == "__main__":
    run()
