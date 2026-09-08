"""Clustering-quality metrics for comparing a predicted conversation_id
assignment against a small hand-labeled ground truth (message_id -> true
episode label). Deliberately NOT an LLM-judge -- these are closed-form,
deterministic set/pair counts over the labels the human supplied.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def pairwise_prf1(predicted: dict[int, int], true: dict[int, int]) -> dict:
    """Standard pairwise clustering precision/recall/F1: for every pair of
    labeled messages, do predicted and true agree on "same cluster"?"""
    ids = sorted(set(predicted) & set(true))
    tp = fp = fn = tn = 0
    for a, b in combinations(ids, 2):
        same_pred = predicted[a] == predicted[b]
        same_true = true[a] == true[b]
        if same_pred and same_true:
            tp += 1
        elif same_pred and not same_true:
            fp += 1
        elif not same_pred and same_true:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def contamination_rate(predicted: dict[int, int], true: dict[int, int]) -> float:
    """Fraction of labeled messages that share a predicted cluster with at
    least one message from a DIFFERENT true episode -- directly measures the
    anchor_window_contamination failure mode: an unrelated message pulled
    into the wrong conversation."""
    ids = sorted(set(predicted) & set(true))
    by_pred = defaultdict(list)
    for mid in ids:
        by_pred[predicted[mid]].append(mid)

    contaminated = set()
    for members in by_pred.values():
        true_labels = {true[m] for m in members}
        if len(true_labels) > 1:
            contaminated.update(members)
    return len(contaminated) / len(ids) if ids else 0.0


def erroneous_merges_and_splits(predicted: dict[int, int], true: dict[int, int]) -> dict:
    """merges: predicted clusters that mix >1 true episode (should have
    stayed separate). splits: true episodes scattered across >1 predicted
    cluster (should have stayed together)."""
    ids = sorted(set(predicted) & set(true))
    by_pred = defaultdict(set)
    by_true = defaultdict(set)
    for mid in ids:
        by_pred[predicted[mid]].add(true[mid])
        by_true[true[mid]].add(predicted[mid])
    merges = sum(1 for true_labels in by_pred.values() if len(true_labels) > 1)
    splits = sum(1 for pred_labels in by_true.values() if len(pred_labels) > 1)
    return {"erroneous_merges": merges, "erroneous_splits": splits}


def cluster_sizes(predicted: dict[int, int]) -> dict:
    sizes = defaultdict(int)
    for cid in predicted.values():
        sizes[cid] += 1
    values = list(sizes.values())
    return {
        "avg_size": sum(values) / len(values) if values else 0.0,
        "max_size": max(values) if values else 0,
        "n_clusters": len(values),
    }


def summarize(predicted: dict[int, int], true: dict[int, int]) -> dict:
    out = {}
    out.update(pairwise_prf1(predicted, true))
    out["contamination_rate"] = contamination_rate(predicted, true)
    out.update(erroneous_merges_and_splits(predicted, true))
    out.update(cluster_sizes(predicted))
    return out
