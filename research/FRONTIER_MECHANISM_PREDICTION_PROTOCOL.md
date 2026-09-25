# Frontier mechanism: confirmatory prediction on non-discovery questions

Registered **before** any of the numbers below were computed. Offline only: no API
calls, no new generations, no new judgments. Every input is an already-sealed frozen
run; this protocol only re-reads per-question fields that were written during those
runs.

## Why this exists

Section 7 of the preprint explains the Temporal Duration result with a
recall-precision frontier claim: accuracy tracks the joint position of delivered
gold evidence, not either coordinate alone. As stated there, this is a *description*
of four rows in Table 4 on the same 300 questions where the effect was discovered.
A description that only ever sees its discovery slice cannot be wrong, which is
exactly the objection the preprint currently cannot answer.

If the frontier claim is a real mechanism, it must also describe the questions where
events do **not** help. Those 2,100 questions are independent of the discovery slice
and already measured. This protocol turns the description into a falsifiable
prediction and fixes the reading rules before looking.

## Inputs (sealed, read-only)

| Condition | Frozen run | Rows |
|---|---|---|
| `RAW` | `evermembench_temporal_episodes_official_v1_20260915_021550_MSK` | 2400 |
| `RAW+EPISODES` | `evermembench_temporal_episodes_official_v1_20260915_021550_MSK` | 2400 |
| `RAW+EVENTS` | `final_sprint_A_events_full_v1_20260915_233142_MSK` | 2400 |

Per-question fields used, all written at run time: `qa_key`, `major`, `minor`,
`correct`, `evidence_recall`, `evidence_precision`.

`F/TP` (Temporal Duration, n=300) is the discovery slice and is **excluded from every
endpoint below**. Analysis set: the remaining 8 categories, n=2100. It is reported
separately for reference only and never pooled into a decision.

## Primary endpoint (question level, n=2100)

For each question, compare delivered evidence between `RAW` and `RAW+EVENTS` and
assign exactly one stratum:

- **A (events dominate):** `recall_E >= recall_R` and `precision_E >= precision_R`,
  with at least one strict.
- **B (raw dominates):** the same with the roles reversed.
- **C (mixed or identical):** everything else, including all ties.

Prediction, fixed now: the accuracy delta `EVENTS - RAW` is **positive in A**,
**negative in B**, and **not distinguishable from zero in C**.

Statistics: exact McNemar on discordant pairs within each stratum, plus a paired
question bootstrap (20,000 resamples, seed 20260912) for the delta and its 95% CI.

**Reading rule.** The mechanism is confirmed on independent material iff all three
hold: A is positive with McNemar p < .05; B is negative in direction; C's CI contains
zero. Any other pattern is recorded as *not confirmed*, and Section 7 keeps the
frontier as a description of the discovery slice only. A significant **positive**
delta in stratum B falsifies the claim outright and will be reported as such.

## Secondary endpoint (category level, 8 categories)

Per category, mean evidence recall and precision per condition, evidence F1 from
those means, and the accuracy delta. Two readings:

1. **Sign agreement.** Among categories where one condition dominates the other,
   does the accuracy delta share the sign of the dominance? Exact two-sided binomial
   against p=0.5.
2. **Spearman** between per-category `delta evidence F1` and `delta accuracy`.

**Power ceiling, stated before the fact.** With at most 8 decidable categories the
exact two-sided binomial reaches p=.0078 only at 8/8 and p=.070 at 7/8. Nothing below
8/8 can clear .05. The secondary endpoint therefore cannot confirm anything on its
own and is reported as supporting texture for the primary.

## Interpretation limits, fixed in advance

- Strata A/B/C are defined by a **post-treatment** variable. This is a
  mechanism-consistency test, not a causal estimate, and no causal language will be
  used for it in the paper.
- Confirming the prediction does **not** revive the average event gain, which stays
  null, nor does it rescue the Temporal Duration p-values from the Holm correction.
  It would only establish that the frontier account describes data it was not fitted
  to.
- `RAW+EPISODES` is carried through the same pipeline for completeness. Episodes
  remain a negative result and nothing here is intended to change that.

## What would make us drop the frontier account

A null or wrong-signed stratum A, or a significantly positive stratum B. In either
case Section 7's mechanism paragraph gets cut back to "events moved both coordinates
on this slice" with no explanatory claim attached.
