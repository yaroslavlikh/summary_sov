"""Offline confirmatory check of the recall-precision frontier account.

Registered protocol: research/FRONTIER_MECHANISM_PREDICTION_PROTOCOL.md
sha256 0419cccb15940839fde5641471eb1cd64b61cf3a479b7e3028bc3558b04f8a59

Reads sealed frozen runs only. No API calls, no generations, no judgments.
"""
import hashlib
import json
import pathlib

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT / ".research_runs" / "frozen"
PROTOCOL = ROOT / "research" / "FRONTIER_MECHANISM_PREDICTION_PROTOCOL.md"
PROTOCOL_SHA = "0419cccb15940839fde5641471eb1cd64b61cf3a479b7e3028bc3558b04f8a59"

SOURCES = {
    "RAW": "evermembench_temporal_episodes_official_v1_20260915_021550_MSK",
    "RAW+EPISODES": "evermembench_temporal_episodes_official_v1_20260915_021550_MSK",
    "RAW+EVENTS": "final_sprint_A_events_full_v1_20260915_233142_MSK",
}
DISCOVERY = ("F", "TP")
NAMES = {
    ("F", "MH"): "Multi-hop", ("F", "SH"): "Single-hop", ("F", "TP"): "Temporal Duration",
    ("MA", "C"): "Constraint", ("MA", "P"): "Proactivity", ("MA", "U"): "Update",
    ("P", "Skill"): "Skill", ("P", "Style"): "Style", ("P", "Title"): "Title",
}
SEED, BOOT = 20260912, 20000


def load(condition):
    path = FROZEN / SOURCES[condition] / "run" / "results.jsonl"
    rows = {}
    for line in path.open():
        r = json.loads(line)
        if r["condition"] != condition:
            continue
        rows[r["qa_key"]] = r
    return rows


def mcnemar_exact(delta):
    """Two-sided exact McNemar on discordant pairs (+1 wins / -1 losses)."""
    b = int((delta > 0).sum())
    c = int((delta < 0).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(stats.binomtest(b, b + c, 0.5).pvalue)


def boot_ci(delta, seed=SEED, n=BOOT):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def dominance(re_, pe, rr, pr):
    if (re_ >= rr and pe >= pr) and (re_ > rr or pe > pr):
        return "A"
    if (rr >= re_ and pr >= pe) and (rr > re_ or pr > pe):
        return "B"
    return "C"


def f1(r, p):
    return 0.0 if (r + p) == 0 else 2 * r * p / (r + p)


def main():
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA, \
        "protocol file changed after registration"

    raw, events, episodes = load("RAW"), load("RAW+EVENTS"), load("RAW+EPISODES")
    keys = sorted(set(raw) & set(events) & set(episodes))
    analysis = [k for k in keys if (raw[k]["major"], raw[k]["minor"]) != DISCOVERY]
    print(f"protocol sha256 verified\nmatched qa_keys: {len(keys)}  "
          f"analysis set (TP excluded): {len(analysis)}\n")

    strata = {"A": [], "B": [], "C": []}
    for k in analysis:
        r, e = raw[k], events[k]
        s = dominance(e["evidence_recall"], e["evidence_precision"],
                      r["evidence_recall"], r["evidence_precision"])
        strata[s].append(e["correct"] - r["correct"])

    print("=== PRIMARY (question level, EVENTS - RAW, n=%d) ===" % len(analysis))
    print(f"{'stratum':9s}{'n':>6s}{'delta pp':>10s}{'95% CI':>20s}{'win/loss':>11s}{'McNemar':>10s}")
    primary = {}
    for s in "ABC":
        d = np.array(strata[s], dtype=float)
        if len(d) == 0:
            print(f"{s:9s}{0:6d}   empty")
            continue
        b, c, p = mcnemar_exact(d)
        lo, hi = boot_ci(d)
        primary[s] = dict(n=len(d), delta=d.mean() * 100, ci=(lo * 100, hi * 100),
                          wins=b, losses=c, p=p)
        print(f"{s:9s}{len(d):6d}{d.mean()*100:+10.2f}"
              f"{f'[{lo*100:+.2f}, {hi*100:+.2f}]':>20s}{f'{b}/{c}':>11s}{p:>10.4f}")

    ok_a = primary.get("A", {}).get("delta", 0) > 0 and primary.get("A", {}).get("p", 1) < .05
    ok_b = primary.get("B", {}).get("delta", 0) < 0
    ci_c = primary.get("C", {}).get("ci", (1, 1))
    ok_c = ci_c[0] <= 0 <= ci_c[1]
    falsified = primary.get("B", {}).get("delta", 0) > 0 and primary.get("B", {}).get("p", 1) < .05
    print(f"\nreading rule -> A positive & p<.05: {ok_a} | B negative: {ok_b} | "
          f"C contains 0: {ok_c}")
    print("VERDICT:", "CONFIRMED" if (ok_a and ok_b and ok_c)
          else ("FALSIFIED" if falsified else "NOT CONFIRMED"))

    print("\n=== SECONDARY (category level, 8 categories) ===")
    print(f"{'category':20s}{'n':>5s}{'dRecall':>9s}{'dPrec':>8s}{'dF1':>8s}{'dAcc pp':>9s}{'dom':>5s}{'agree':>7s}")
    df1s, daccs, agree = [], [], []
    for cat in sorted({(raw[k]['major'], raw[k]['minor']) for k in analysis}, key=lambda c: NAMES[c]):
        ks = [k for k in analysis if (raw[k]["major"], raw[k]["minor"]) == cat]
        rr = np.mean([raw[k]["evidence_recall"] for k in ks])
        pr = np.mean([raw[k]["evidence_precision"] for k in ks])
        re_ = np.mean([events[k]["evidence_recall"] for k in ks])
        pe = np.mean([events[k]["evidence_precision"] for k in ks])
        dacc = np.mean([events[k]["correct"] - raw[k]["correct"] for k in ks]) * 100
        dom = dominance(re_, pe, rr, pr)
        a = "" if dom == "C" else str((dom == "A") == (dacc > 0))
        if dom != "C":
            agree.append((dom == "A") == (dacc > 0))
        df1s.append(f1(re_, pe) - f1(rr, pr))
        daccs.append(dacc)
        print(f"{NAMES[cat]:20s}{len(ks):5d}{re_-rr:+9.4f}{pe-pr:+8.4f}"
              f"{f1(re_,pe)-f1(rr,pr):+8.4f}{dacc:+9.2f}{dom:>5s}{a:>7s}")

    if agree:
        k, m = sum(agree), len(agree)
        print(f"\nsign agreement: {k}/{m}, exact two-sided binomial "
              f"p={stats.binomtest(k, m, 0.5).pvalue:.4f}")
    rho, prho = stats.spearmanr(df1s, daccs)
    print(f"Spearman(dF1, dAcc) over {len(df1s)} categories: rho={rho:+.3f}, p={prho:.4f}")

    print("\n=== REFERENCE ONLY: discovery slice, excluded from every endpoint ===")
    tp = [k for k in keys if (raw[k]["major"], raw[k]["minor"]) == DISCOVERY]
    d = np.array([events[k]["correct"] - raw[k]["correct"] for k in tp], dtype=float)
    b, c, p = mcnemar_exact(d)
    print(f"Temporal Duration n={len(tp)} delta={d.mean()*100:+.2f} pp "
          f"wins/losses={b}/{c} p={p:.4f}")


if __name__ == "__main__":
    main()
