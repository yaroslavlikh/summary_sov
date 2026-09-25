"""Query-independent temporal episode extraction -- full-corpus rollout, Step 1 (SMOKE ONLY).

Builds the SAME frozen TemporalEpisode memory (research/temporal_episode_prototype.py,
unmodified: MemoryEvent contract, validate_memory_event, retrieve_top_candidates,
run_attach_decision, materialize_episodes -- all reused verbatim) but feeds it
clusters built PURELY from conversation session boundaries, never from QA
evidence_anchors. This matches the SocialMemBench official protocol
(arXiv:2605.17789, Section 4.2): "Sessions are ingested in chronological order;
all queries are issued after the final session" -- i.e. memory construction must
be query-independent, not just "don't read the QA text."

Reuse trick (no frozen code touched): `extract_events_for_network(network_id,
clusters, conversations)` takes clusters as (session_index, anchor_turn_ids)
pairs and internally calls `build_group_context`, which unions a +/-2-turn
window around every id in anchor_turn_ids. Passing the FULL set of turn_ids in
a session as "anchor_ids" makes that union cover the ENTIRE session (turns
within one session are dense/consecutive, so neighboring +/-2 windows fully
overlap) -- confirmed against the actual corpus: sessions run 10-36 turns
(median 21), comfortably inside one extraction call's context. So zero lines
of the frozen extraction/context/validation code change; only what we pass in
as "clusters" changes, from QA-anchor-derived to session-derived.

Isolation: EVENT_CACHE / ATTACH_CACHE on the temporal_episode_prototype module
are monkey-patched to fresh paths (same precedent as EMBED_CACHE_DIR
elsewhere in this project) -- this step's cache never touches or is touched
by the earlier oracle-scoped dev-set caches.

SMOKE ONLY: a small, deliberately tier-diverse (small/medium/large) subset of
networks -- to measure real extraction+attach cost per network and eyeball
chain quality on full, unscoped sessions before committing to the full
43-network rollout. Does not scale beyond SMOKE_NETWORK_IDS without a
separate, explicit go-ahead call.

Run:
    python3 -m research.query_independent_episode_pipeline
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.socialmembench_pilot import load_data
from research.stage4_5_audit import DATA_DIR
from research import temporal_episode_prototype as tep

QI_EVENT_CACHE = Path("/tmp/qi_episode_event_cache.jsonl")
QI_ATTACH_CACHE = Path("/tmp/qi_episode_attach_cache.jsonl")
SMOKE_REPORT_MD = Path("/tmp/qi_episode_smoke_report.md")

# One network per tier boundary, chosen from the actual corpus size distribution
# (turns/network: min 50, median 178, max 302; members: 4-30) -- never chosen
# by looking at QA content, only by conversations.csv size/member-count stats.
SMOKE_NETWORK_IDS = (
    "grp_0d1e2f3a",  # small: 4 members, 50 turns, 5 sessions
    "grp_1a2b3c4d",  # small (upper end): 5 members, 101 turns, 5 sessions
    "grp_f2e3d4c5",  # medium: 9 members, 129 turns, 8 sessions
    "grp_a1b2c3d4",  # medium (upper end): 10 members, 188 turns, 8 sessions
    "grp_a6b7c8d9",  # large: 30 members, 290 turns, 10 sessions
)


def build_query_independent_clusters(conversations, network_id: str) -> list[tuple[int, set[str]]]:
    """Every session_index for this network becomes one cluster containing
    ALL of that session's turn_ids -- computed purely from conversation
    structure. QA/evidence_anchors are never read here or anywhere upstream
    of this call."""
    frame = conversations[conversations.network_id == network_id]
    by_session: dict[int, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        by_session[row.session_index].add(row.turn_id)
    return sorted(by_session.items())


def run_qi_extraction_and_materialize(
    network_ids: tuple[str, ...], conversations,
) -> tuple[dict, dict, list, dict]:
    """Returns (episodes_by_network, events_by_network, decisions_log, stats)."""
    tep.EVENT_CACHE = QI_EVENT_CACHE
    tep.ATTACH_CACHE = QI_ATTACH_CACHE

    events_by_network: dict[str, list] = defaultdict(list)
    all_rejected: list[dict] = []
    extraction_calls = 0
    evidence_attempted_total = 0
    evidence_valid_total = 0
    clusters_by_network: dict[str, list] = {}

    for network_id in network_ids:
        clusters = build_query_independent_clusters(conversations, network_id)
        clusters_by_network[network_id] = clusters
        events, rejected, calls, quote_stats = tep.extract_events_for_network(network_id, clusters, conversations)
        events_by_network[network_id].extend(events)
        all_rejected.extend(rejected)
        extraction_calls += calls
        evidence_attempted_total += quote_stats["evidence_attempted"]
        evidence_valid_total += quote_stats["evidence_valid"]

    episodes_by_network, decisions_log, attach_calls = tep.materialize_episodes(dict(events_by_network))

    stats = {
        "networks": len(network_ids),
        "sessions_processed": sum(len(c) for c in clusters_by_network.values()),
        "events_total": sum(len(v) for v in events_by_network.values()),
        "events_rejected": len(all_rejected),
        "rejected_reasons": _count_reasons(all_rejected),
        "episodes_total": sum(len(v) for v in episodes_by_network.values()),
        "extraction_calls": extraction_calls,
        "attach_calls": attach_calls,
        "total_new_calls": extraction_calls + attach_calls,
        "evidence_attempted": evidence_attempted_total,
        "evidence_valid": evidence_valid_total,
        "clusters_by_network": {k: len(v) for k, v in clusters_by_network.items()},
    }
    return dict(episodes_by_network), dict(events_by_network), decisions_log, stats


def _count_reasons(rejected: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in rejected:
        counts[r["reason"]] += 1
    return dict(counts)


def render_smoke_report(
    episodes_by_network: dict, events_by_network: dict, decisions_log: list[dict], stats: dict,
) -> str:
    lines = [
        "# Query-independent episode extraction -- SMOKE (5 tier-diverse networks)",
        "",
        "Clusters were built from conversation session_index alone -- no QA question, answer, or "
        "evidence_anchors were read at any point in this run. This is a cost/quality smoke test, "
        "NOT the full 43-network rollout and NOT a benchmark score.",
        "",
        f"- networks: {stats['networks']}  ({', '.join(SMOKE_NETWORK_IDS)})",
        f"- sessions processed: {stats['sessions_processed']}",
        f"- events extracted: {stats['events_total']}  (rejected: {stats['events_rejected']}, reasons: {stats['rejected_reasons']})",
        f"- quote-valid provenance rate: {stats['evidence_valid']}/{stats['evidence_attempted']}",
        f"- episodes materialized: {stats['episodes_total']}",
        f"- new LLM calls: extraction={stats['extraction_calls']}, attach={stats['attach_calls']}, "
        f"total={stats['total_new_calls']}",
        "",
        "## Cost per network (for full-43 extrapolation)",
        "",
        "| network_id | sessions | events | episodes | calls (extraction+attach est.) |",
        "|---|---|---|---|---|",
    ]
    decisions_by_network: dict[str, list] = defaultdict(list)
    for d in decisions_log:
        decisions_by_network[d["network_id"]].append(d)
    for network_id in SMOKE_NETWORK_IDS:
        n_sessions = stats["clusters_by_network"].get(network_id, 0)
        n_events = len(events_by_network.get(network_id, []))
        n_episodes = len(episodes_by_network.get(network_id, []))
        n_decisions = len(decisions_by_network.get(network_id, []))
        lines.append(f"| {network_id} | {n_sessions} | {n_events} | {n_episodes} | {n_sessions}+{n_decisions} |")

    full_corpus_sessions = 348
    full_corpus_networks = 43
    if stats["sessions_processed"]:
        extrapolated_extraction = stats["extraction_calls"] * full_corpus_sessions / stats["sessions_processed"]
        extrapolated_attach = stats["attach_calls"] * full_corpus_sessions / stats["sessions_processed"]
        lines += [
            "",
            f"## Extrapolation to all {full_corpus_networks} networks / {full_corpus_sessions} sessions",
            "",
            f"- linear-in-sessions estimate: extraction ~= {extrapolated_extraction:.0f} calls, "
            f"attach ~= {extrapolated_attach:.0f} calls, total ~= {extrapolated_extraction + extrapolated_attach:.0f} calls",
            "- attach calls scale with EVENTS not sessions, so this is a rough upper/lower sanity bound, "
            "not a precise forecast -- actual attach count depends on event density and episode fan-out, "
            "which this 5-network sample estimates only coarsely.",
        ]

    lines += ["", "## Chains per network", ""]
    for network_id in SMOKE_NETWORK_IDS:
        eps = episodes_by_network.get(network_id, [])
        lines.append(f"### {network_id} ({len(eps)} episodes)")
        for ep in sorted(eps, key=lambda e: e.episode_id):
            lines.append(f"- {ep.episode_id} (owner={ep.viewpoint_owner}, subject={ep.subject}, "
                         f"{len(ep.events)} event(s)):")
            for ee in ep.events:
                lines.append(f"    - [{ee.decision}] {ee.event.event_text}  (sources={list(ee.event.source_turn_ids)})")
        lines.append("")

    lines += [
        "## Not run in this smoke", "",
        f"- remaining {full_corpus_networks - stats['networks']}/{full_corpus_networks} networks",
        "- answer generation / any judge (official or generic)",
        "- retrieval evaluation (RAW vs RAW+EPISODES)",
        "- held-out QA (never read)",
        "- any production code path (/ask, llm/graphs.py, Telegram, DB)",
    ]
    return "\n".join(lines)


def main() -> None:
    conversations, _qa, _personas = load_data(DATA_DIR)
    episodes_by_network, events_by_network, decisions_log, stats = run_qi_extraction_and_materialize(
        SMOKE_NETWORK_IDS, conversations,
    )
    report = render_smoke_report(episodes_by_network, events_by_network, decisions_log, stats)
    SMOKE_REPORT_MD.write_text(report)
    print(f"networks={stats['networks']} sessions={stats['sessions_processed']} "
          f"events={stats['events_total']} episodes={stats['episodes_total']} "
          f"new_calls={stats['total_new_calls']}", file=sys.stderr)
    print(f"Report: {SMOKE_REPORT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
