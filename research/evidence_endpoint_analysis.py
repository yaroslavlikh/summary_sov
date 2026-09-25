"""Paired per-question analysis of the evidence-delivery endpoint.

Offline: reads sealed run artifacts only, issues no API calls. Produces
research/figures/evidence_endpoint_results.json, which paper/preprint_memory_ru
consumes so that no number in the manuscript is typed by hand.

Endpoint rationale: evidence_recall / evidence_precision are computed
mechanically from exposed source IDs and gold anchors. They pass through
neither the generator nor the judge, and are bit-identical across stochastic
reruns (verified below), whereas answer accuracy flips on ~4% of questions.
"""
from __future__ import annotations
import json, math, random, collections, statistics, pathlib

RUNS = pathlib.Path("/Users/yaroslavlikh/summary_sov/.research_runs")
OUT = pathlib.Path("/Users/yaroslavlikh/summary_sov/research/figures/evidence_endpoint_results.json")
BOOT, SEED = 20000, 20260912


def load(rel: str, cond: str, key: str = "qa_key") -> dict:
    d = {}
    for line in (RUNS / rel).open():
        r = json.loads(line)
        if r.get("condition") == cond:
            d[r[key]] = r
    return d


def boot_ci(x, B=BOOT, seed=SEED):
    rnd = random.Random(seed)
    n = len(x)
    bs = sorted(sum(x[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return bs[int(0.025 * B)], bs[int(0.975 * B)]


def wilcoxon_p(x):
    nz = [v for v in x if abs(v) > 1e-12]
    if not nz:
        return 1.0, 0
    order = sorted(range(len(nz)), key=lambda i: abs(nz[i]))
    rank = [0] * len(nz)
    for pos, i in enumerate(order):
        rank[i] = pos + 1
    W = sum(rank[i] for i in range(len(nz)) if nz[i] > 0)
    n = len(nz)
    z = (W - n * (n + 1) / 4) / math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    return math.erfc(abs(z) / math.sqrt(2)), n


def paired(A, B, field):
    ks = sorted(set(A) & set(B))
    x = [A[k][field] - B[k][field] for k in ks]
    m = sum(x) / len(x)
    lo, hi = boot_ci(x)
    p, nz = wilcoxon_p(x)
    return dict(n=len(ks), delta_pp=100 * m, ci=[100 * lo, 100 * hi], p=p,
                better=sum(1 for v in x if v > 1e-12),
                worse=sum(1 for v in x if v < -1e-12))


res: dict = {"_note": "all figures in percentage points; offline recomputation"}

# ---------------------------------------------------------------- EverMemBench
EV = "evermembench_temporal_episodes_official_v1/results.jsonl"
RAW = load(EV, "RAW")
EPI = load(EV, "RAW+EPISODES")
EVE = load("final_sprint_A_events_full_v1/results.jsonl", "RAW+EVENTS")

res["ever"] = {
    arm: {f: paired(D, RAW, f) for f in ("evidence_precision", "evidence_recall")}
    for arm, D in (("events", EVE), ("episodes", EPI))
}

# per-category precision delta, events - raw
cats = collections.defaultdict(list)
for k in sorted(set(EVE) & set(RAW)):
    cats[EVE[k]["minor"]].append(EVE[k]["evidence_precision"] - RAW[k]["evidence_precision"])
res["ever"]["events_precision_by_category"] = {}
for c, v in cats.items():
    lo, hi = boot_ci(v)
    res["ever"]["events_precision_by_category"][c] = dict(
        n=len(v), delta_pp=100 * sum(v) / len(v), ci=[100 * lo, 100 * hi])

# Pareto check: are fewer sources delivered?  (precision could rise mechanically)
def exposed(rel, cond):
    d = {}
    for line in (RUNS / rel).open():
        r = json.loads(line)
        if r["condition"] == cond:
            d[r["qa_key"]] = len(set(r["exposed_source_ids"]))
    return d

sRAW = exposed("evermembench_temporal_episodes_official_v1/predictions.jsonl", "RAW")
sEVE = exposed("final_sprint_A_events_full_v1/predictions.jsonl", "RAW+EVENTS")
sEPI = exposed("evermembench_temporal_episodes_official_v1/predictions.jsonl", "RAW+EPISODES")
ks = sorted(set(sRAW) & set(sEVE) & set(sEPI))
gold = lambda D, S, k: D[k]["evidence_precision"] * S[k]
gdiff = [gold(EVE, sEVE, k) - gold(RAW, sRAW, k) for k in ks]
lo, hi = boot_ci(gdiff)
res["ever"]["pareto"] = dict(
    sources={"raw": statistics.mean(sRAW[k] for k in ks),
             "events": statistics.mean(sEVE[k] for k in ks),
             "episodes": statistics.mean(sEPI[k] for k in ks)},
    gold_hits={"raw": statistics.mean(gold(RAW, sRAW, k) for k in ks),
               "events": statistics.mean(gold(EVE, sEVE, k) for k in ks),
               "episodes": statistics.mean(gold(EPI, sEPI, k) for k in ks)},
    gold_delta=statistics.mean(gdiff), gold_ci=[lo, hi],
    events_never_more_sources=all(sEVE[k] <= sRAW[k] for k in ks),
    questions_events_strictly_dominate=sum(
        1 for k in ks if gold(EVE, sEVE, k) > gold(RAW, sRAW, k) and sEVE[k] <= sRAW[k]))

# ------------------------------------------------------------- SocialMemBench
F = "frozen/socialmembench_full_official_v1_20260911_184037_MSK/run/results.jsonl"
E = ("frozen/socialmembench_temporal_episodes_official_harness_v1_20260912_122626_MSK"
     "/run/episode_results.jsonl")
sd = collections.defaultdict(dict)
for rel in (F, E):
    for line in (RUNS / rel).open():
        r = json.loads(line)
        sd[r["condition"]][r["qa_key"]] = r
res["social"] = {
    arm: {f: paired(sd[cond], sd["RAW"], f)
          for f in ("evidence_precision", "evidence_recall")}
    for arm, cond in (("flat", "RAW+FLAT"), ("versioned", "RAW+VERSIONED"),
                      ("episodes", "RAW+EPISODES"))
}

# ------------------------------------------------ HyDE control, Temporal slice
hd = collections.defaultdict(dict)
for rel in ("final_sprint_C_hyde_raw_v1/results.jsonl",
            "final_sprint_C2_hyde_token_matched_v1/results.jsonl",
            "evermembench_chronological_order_control_v2/results.jsonl"):
    for line in (RUNS / rel).open():
        r = json.loads(line)
        if "evidence_precision" in r:
            hd[r["condition"]][r["qa_key"]] = r
res["hyde"] = {
    "events_minus_hyde": {
        f: paired(hd["RAW+EVENTS-token-matched-chrono"],
                  hd["HyDE-RAW-token-matched-chrono"], f)
        for f in ("evidence_precision", "evidence_recall")},
    "hyde_minus_raw": {
        f: paired(hd["HyDE-RAW-token-matched-chrono"],
                  hd["RAW-token-matched-chrono"], f)
        for f in ("evidence_precision", "evidence_recall")},
}

# --------------------------------------------- noise floor: stochastic rerun
A = load("evermembench_temporal_budget_control_v1/results.jsonl", "RAW+EPISODES-rerun")
B = load(EV, "RAW+EPISODES")
ks = sorted(set(A) & set(B))
flips = [(A[k]["correct"], B[k]["correct"]) for k in ks]
res["noise_floor"] = dict(
    n=len(ks),
    accuracy_run=100 * sum(b for _, b in flips) / len(ks),
    accuracy_rerun=100 * sum(a for a, _ in flips) / len(ks),
    verdict_flips=sum(1 for a, b in flips if a != b),
    flips_1to0=sum(1 for a, b in flips if a == 0 and b == 1),
    flips_0to1=sum(1 for a, b in flips if a == 1 and b == 0),
    evidence_differences=sum(
        1 for k in ks if abs(A[k]["evidence_recall"] - B[k]["evidence_recall"]) > 1e-9
        or abs(A[k]["evidence_precision"] - B[k]["evidence_precision"]) > 1e-9))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(f"wrote {OUT}")
for k in ("ever", "social", "hyde", "noise_floor"):
    print("\n==", k)
    print(json.dumps(res[k], indent=2, ensure_ascii=False)[:900])
