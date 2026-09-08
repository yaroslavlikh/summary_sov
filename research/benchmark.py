"""Read-only benchmark: current production time-gap heuristic vs the
semantic disentanglement pilot, on REAL production data (messages
72736-72815 of chat -1002335227490 -- the exact window covering the
"Наполовину" incident plus a genuinely-recurring topic ~26h later for the
retrieval check).

Writes nothing. Does not touch chat_context/chat_moments/messages. Does not
invoke /ask, run_eval.py, or anything that could execute a memory_command.

Usage: python3 -m research.benchmark
"""
import sys

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

from research.clustering_metrics import summarize
from research.conversation_disentanglement import search_similar_episodes, semantic_pilot, time_gap_baseline
from research.db_loader import load_messages

CHAT_ID = -1002335227490

# Hand-labeled ground truth for the incident window (72736-72751), built
# from the actual decrypted message content and real reply_to_message_id
# chain -- NOT an LLM judge. 72752 is excluded: its only reply target
# (72729) is outside this window, so honestly labeling it would be a guess.
#
# Labeling call worth flagging explicitly: 72749/72750 (Тигмен re-asking
# "кто я" ~85s after his first 72739/72740 exchange, no reply link) are
# labeled as a SEPARATE episode from 72739/72740, not a continuation --
# a debatable judgment call, not an objective fact.
GROUND_TRUTH = {
    72736: "IGOR_PORTRAIT", 72737: "IGOR_PORTRAIT", 72738: "IGOR_PORTRAIT",
    72739: "TIGMEN_IDENTITY_1", 72740: "TIGMEN_IDENTITY_1",
    72741: "IVJENIN_IDENTITY", 72742: "IVJENIN_IDENTITY",
    72743: "DEAD_LINKS", 72744: "DEAD_LINKS",
    72745: "TIGMEN_IDENTITY_1", 72746: "TIGMEN_IDENTITY_1", 72747: "TIGMEN_IDENTITY_1", 72748: "TIGMEN_IDENTITY_1",
    72749: "TIGMEN_IDENTITY_2", 72750: "TIGMEN_IDENTITY_2",
    72751: "MVP_COMMENT",
}


def run():
    print("Loading real messages 72736-72815 (read-only)...")
    messages = load_messages(CHAT_ID, min_message_id=72736, max_message_id=72815)
    incident_window = [m for m in messages if m.message_id <= 72751]
    print(f"  incident window: {len(incident_window)} messages, ground truth labels: {len(GROUND_TRUTH)}")

    baseline_pred = time_gap_baseline(incident_window)
    pilot_pred, pilot_results, tracker = semantic_pilot(incident_window)

    baseline_metrics = summarize(baseline_pred, GROUND_TRUTH)
    pilot_metrics = summarize(pilot_pred, GROUND_TRUTH)

    print("\n=== Real conversation_id assigned to the incident by production RIGHT NOW ===")
    import database.db as db  # noqa: local import, read-only check
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT conversation_id FROM messages WHERE user_id = %s AND message_id BETWEEN 72736 AND 72751",
            (CHAT_ID,),
        )
        print(f"  distinct conversation_id values in production: {[r[0] for r in cur.fetchall()]}")

    print("\n=== baseline (time-gap <300s, current production) vs semantic pilot ===")
    header = f"{'metric':<20} {'baseline':>10} {'pilot':>10}"
    print(header)
    print("-" * len(header))
    for key in ["precision", "recall", "f1", "contamination_rate", "erroneous_merges", "erroneous_splits", "avg_size", "max_size", "n_clusters"]:
        b, p = baseline_metrics[key], pilot_metrics[key]
        fmt = "{:>10.3f}" if isinstance(b, float) else "{:>10}"
        print(f"{key:<20} {fmt.format(b)} {fmt.format(p)}")

    print("\n=== per-message assignment (pilot) -- method + score + true label ===")
    by_id = {r.message_id: r for r in pilot_results}
    for m in incident_window:
        r = by_id[m.message_id]
        true_label = GROUND_TRUTH.get(m.message_id, "?")
        score = f"{r.score:.2f}" if r.score is not None else "  - "
        print(f"  {m.message_id} [{true_label:<18}] pred={r.conversation_id:<7} method={r.method:<10} score={score} low_conf={r.low_confidence} text={m.text[:45]!r}")

    print("\n=== retrieval check: dead-links episode (Sept 7) vs 'links working better' (Sept 8, ~26h later) ===")
    later = [m for m in messages if m.message_id > 72751 and "ссылк" in m.text.lower()]
    if not later:
        print("  no later 'ссылки' message found in loaded range -- skipping retrieval check")
    else:
        query_msg = later[0]
        print(f"  query message {query_msg.message_id}: {query_msg.text!r} ({(query_msg.timestamp - 72744):.0f}s... n/a, see below)")
        dead_links_conv_id = pilot_pred.get(72743)
        hits = search_similar_episodes(query_msg.embedding, tracker.all_episodes(), top_k=3)
        hit_ids = [h["conversation_id"] for h in hits]
        found = dead_links_conv_id in hit_ids
        print(f"  dead-links episode conversation_id = {dead_links_conv_id}")
        print(f"  top-3 retrieval hits: {[(h['conversation_id'], round(h['score'], 3)) for h in hits]}")
        print(f"  FOUND via retrieval: {found}")
        gap_hours = (query_msg.timestamp - 1788782997) / 3600
        print(f"  real time gap: {gap_hours:.1f} hours (well beyond the {30}-min active window, so it could only be found via retrieval, not by staying 'active')")


if __name__ == "__main__":
    run()
