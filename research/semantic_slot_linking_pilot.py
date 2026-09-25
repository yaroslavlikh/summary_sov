"""Semantic slot linking pilot -- isolated diagnostic testing ONE hypothesis:
linking assertions by a stable, VALUE-FREE state slot ("what question does
this state answer") instead of surface claim similarity or exact
facet/topic_key beats the current linker's false-split rate.

Motivating finding (full official run, `.research_runs/socialmembench_full_official_v1/`,
frozen, read-only, NEVER modified here): 1,157 validated assertions
materialized into 1,148 versioned cells -- the current facet/topic_key-based
linker merged only ~9 assertions into existing cells. VERSIONED barely
differs from FLAT (MeanQ 0.367 vs 0.370) because almost nothing actually
gets linked. Manual review (`/tmp/socialmembench_stage4_6_pair_review_v2.md`,
already completed by a human -- read here, never regenerated) confirms real
same-state pairs exist (Seb's knee: REVISE, Uncle Femi's commitment: RETRACT,
Diane's approval-seeking: REVISE) that the old facet-based candidate filter
would never even surface as candidates, because facet/subject phrasing
("Seb's knee condition" vs "Seb's plan") drifts across independent
extractions of the same entity's different states.

This module does NOT redesign versioned_memory_cells.py, does NOT touch any
Stage 4/5/full-run cache, and does NOT run the 1,031-QA benchmark. It is a
bounded, read-only diagnostic over the Q8 dev networks that already have
completed manual review, restricted to their already-cached assertions in
the full official run's session_assertions.jsonl.

Architecture stays event-sourced/versioned, not a knowledge graph:
  1. Assertion events (raw, immutable, from session_assertions.jsonl -- never
     mutated here).
  2. SlotAssertion -- a normalized, value-free VIEW of each assertion (new
     subject_entity/state_dimension/slot_question fields only; viewpoint_
     owner/value/source_turn_ids/observed_at are passed through UNCHANGED
     from the original assertion, never re-derived by the LLM).
  3. Candidate retrieval -- pure, no LLM, ranks OTHER assertions in the same
     network by one of four representations (A/B/C/D ablation).
  4. A bounded top-5 resolver -- one more LLM call per query, choosing
     NEW/OBSERVE/REVISE/RETRACT against the top-5 candidates only.
  5. Diagnostic chain materialization for confirmed pairs only -- ordered
     state versions with per-version provenance, never deleting or mutating
     the underlying assertion events.

Run:
    python3 -m research.semantic_slot_linking_pilot
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from embeddings import embed_batch
from research.socialmembench_pilot import _norm, _parse_json_object
from research.stage4_5_audit import DATA_DIR

FULL_RUN_DIR = Path(".research_runs/socialmembench_full_official_v1")
SESSION_ASSERTIONS = FULL_RUN_DIR / "session_assertions.jsonl"
PAIR_REVIEW_MD = Path("/tmp/socialmembench_stage4_6_pair_review_v2.md")

SLOT_CACHE = Path("/tmp/socialmembench_semantic_slot_cache.jsonl")
RETRIEVAL_REPORT_MD = Path("/tmp/socialmembench_semantic_slot_retrieval_report.md")
RESOLVER_REPORT_MD = Path("/tmp/socialmembench_semantic_slot_resolver_report.md")
REVIEW_MD = Path("/tmp/socialmembench_semantic_slot_review.md")

SCHEMA_VERSION = "slot_v1"
NORMALIZE_CHUNK = 25
RETRIEVAL_VARIANTS = ("A_full_claim", "B_value_masked", "C_slot_question", "D_slot_plus_conversation")
SESSION_SOFT_WEIGHT = 0.15  # variant D: slot_question is primary, session context is a minor nudge

RECALL_AT_5_GATE = 0.85
FALSE_MERGE_RATE_GATE = 0.05
CHAIN_RECOVERY_GATE_NUM, CHAIN_RECOVERY_GATE_DEN = 8, 12
MAX_HOT_VERSIONS = 5


# --------------------------------------------------------------------------
# Step 1: inventory + gold labels (parsed from the ALREADY COMPLETED manual
# review sheet -- nothing here is auto-labeled).
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GoldPair:
    qa_id: str
    network_id: str
    who: str
    old_turn_id: str
    new_turn_id: str
    relation: str  # REVISE | RETRACT | OBSERVE | AUGMENT


@dataclass(frozen=True)
class GoldNoPair:
    qa_id: str
    network_id: str
    old_turn_id: Optional[str]
    new_turn_id: Optional[str]


_CARD_RE = re.compile(r"^## \d+\. (?P<qa_id>\S+) :: (?P<network_id>\S+)")
_ANCHOR_RE = re.compile(r"^### Anchor: (?P<turn_id>\S+) \(session \d+\) -- group: (?P<group>\S+)")
_PAIR_RE = re.compile(r"^- Valid same-state pair: assertion #(?P<a>\d+|_+) -> assertion #(?P<b>\d+|_+)")
_RELATION_RE = re.compile(r"^- Relation: (?:\[( |\+)\] (\w+)\s*)+")
_NO_PAIR_RE = re.compile(r"^- No valid pair in extraction: \[(?P<mark>.)\]")


def parse_manual_review(path: Path) -> tuple[list[GoldPair], list[GoldNoPair]]:
    """Parses the ALREADY-COMPLETED review sheet -- reads existing [+]
    checkboxes, never infers or fills in a verdict. Cases with neither a
    confirmed pair nor an explicit "No valid pair: [+]" mark are skipped
    (not silently treated as either label)."""
    if not path.exists():
        raise FileNotFoundError(f"required manual review artifact missing: {path}")
    lines = path.read_text().splitlines()

    pairs: list[GoldPair] = []
    no_pairs: list[GoldNoPair] = []
    qa_id = network_id = None
    old_turn = new_turn = None
    relation = None
    pair_indices: Optional[tuple[int, int]] = None
    no_pair_mark = None

    def flush():
        nonlocal qa_id, network_id, old_turn, new_turn, relation, pair_indices, no_pair_mark
        if qa_id is None:
            return
        if no_pair_mark == "+":
            no_pairs.append(GoldNoPair(qa_id=qa_id, network_id=network_id, old_turn_id=old_turn, new_turn_id=new_turn))
        elif pair_indices is not None and relation and old_turn and new_turn:
            pairs.append(GoldPair(
                qa_id=qa_id, network_id=network_id, who=network_id,
                old_turn_id=old_turn, new_turn_id=new_turn, relation=relation,
            ))
        qa_id = network_id = old_turn = new_turn = relation = None
        pair_indices = None
        no_pair_mark = None

    current_group = None
    for line in lines:
        card_match = _CARD_RE.match(line)
        if card_match:
            flush()
            qa_id, network_id = card_match.group("qa_id"), card_match.group("network_id")
            continue
        anchor_match = _ANCHOR_RE.match(line)
        if anchor_match:
            current_group = anchor_match.group("group")
            if current_group == "old":
                old_turn = anchor_match.group("turn_id")
            elif current_group == "new":
                new_turn = anchor_match.group("turn_id")
            continue
        pair_match = _PAIR_RE.match(line)
        if pair_match and pair_match.group("a").isdigit():
            pair_indices = (int(pair_match.group("a")), int(pair_match.group("b")))
            continue
        if line.startswith("- Relation:"):
            checked = re.findall(r"\[\+\]\s*(\w+)", line)
            if checked:
                relation = checked[0]
            continue
        no_pair_match = _NO_PAIR_RE.match(line)
        if no_pair_match:
            no_pair_mark = no_pair_match.group("mark")
            continue
    flush()
    return pairs, no_pairs


# --------------------------------------------------------------------------
# Step 1b: load bounded raw assertions from the FROZEN full official run.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RawAssertion:
    assertion_id: str
    network_id: str
    session_id: str
    viewpoint_owner: str
    subject: str
    facet: str
    assertion_text: str
    normalized_value: str
    modality: str
    source_turn_ids: tuple[str, ...]


def load_bounded_assertions(networks: set[str]) -> list[RawAssertion]:
    if not SESSION_ASSERTIONS.exists():
        raise FileNotFoundError(
            f"required immutable artifact missing: {SESSION_ASSERTIONS} -- "
            "not regenerating; restore from the full official run only."
        )
    out: list[RawAssertion] = []
    with SESSION_ASSERTIONS.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["network_id"] not in networks:
                continue
            for index, item in enumerate(row.get("raw_items", [])):
                source_ids = tuple(str(s) for s in item.get("source_turn_ids", []))
                if not source_ids:
                    continue
                out.append(RawAssertion(
                    assertion_id=f"{row['network_id']}:{row['session_id']}:{index}",
                    network_id=row["network_id"], session_id=row["session_id"],
                    viewpoint_owner=str(item.get("viewpoint_owner") or ""),
                    subject=str(item.get("subject") or ""), facet=str(item.get("facet") or ""),
                    assertion_text=str(item.get("assertion_text") or ""),
                    normalized_value=str(item.get("normalized_value") or ""),
                    modality=str(item.get("modality") or ""), source_turn_ids=source_ids,
                ))
    return out


def load_conversations(networks: set[str]) -> pd.DataFrame:
    conversations = pd.read_parquet(DATA_DIR / "conversations.parquet")
    return conversations[conversations.network_id.isin(networks)].copy()


# --------------------------------------------------------------------------
# Step 2: immutable, resumable, content-addressed slot-normalization cache.
# --------------------------------------------------------------------------

def _slot_normalize_prompt(items: list[RawAssertion]) -> str:
    rows = [
        f"[{i}] viewpoint_owner={item.viewpoint_owner!r} subject={item.subject!r} facet={item.facet!r} "
        f"value={item.normalized_value!r} source_turn_ids={list(item.source_turn_ids)!r}\n"
        f"    text: {item.assertion_text}"
        for i, item in enumerate(items)
    ]
    return f"""For each numbered assertion below, produce a VALUE-FREE state-slot view.
viewpoint_owner and value are already fixed and correct -- do not change them.
You are asked for exactly these NEW fields:

- subject_entity: the actual ENTITY the assertion is about -- a name, not a
  surface phrase. "Seb's knee condition" -> subject_entity="Seb" (the entity
  is Seb, not "Seb's knee"). If genuinely ambiguous who/what the entity is,
  leave subject_entity empty rather than guessing.
- state_dimension: a short label for WHICH property of that entity is being
  described (e.g. "knee condition", "location", "attitude toward Gordey",
  "commitment to event X"). Two assertions about the SAME entity but a
  DIFFERENT property must get a DIFFERENT state_dimension.
- slot_question: a natural-language question this state answers, phrased so
  the ANSWER would be the value -- e.g. "What is Seb's knee condition?",
  "How does Tigmen evaluate Gordey?". CRITICAL: slot_question must NOT
  contain the value itself, only the question.
- normalization_confidence: 0.0-1.0, your confidence subject_entity/
  state_dimension are correctly identified from the text alone.
- source_turn_ids: copy the EXACT source_turn_ids list shown for that
  assertion, verbatim, unchanged -- this is a consistency check, not a new
  judgment call.

Assertions:
{chr(10).join(rows)}

Return JSON only: {{"items":[{{"index":0,"subject_entity":"...","state_dimension":"...",
"slot_question":"...","normalization_confidence":0.0,"source_turn_ids":["..."]}}]}}"""


def _cache_key(assertion: RawAssertion, prompt_hash: str) -> str:
    payload = {
        "assertion_id": assertion.assertion_id, "schema_version": SCHEMA_VERSION, "prompt_hash": prompt_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_slot_cache(assertions: list[RawAssertion]) -> dict[str, dict]:
    cached: dict[str, dict] = {}
    if SLOT_CACHE.exists():
        for line in SLOT_CACHE.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cached[row["cache_key"]] = row

    chunks = [assertions[i:i + NORMALIZE_CHUNK] for i in range(0, len(assertions), NORMALIZE_CHUNK)]
    to_call = []
    for chunk in chunks:
        prompt = _slot_normalize_prompt(chunk)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        keys = [_cache_key(item, prompt_hash) for item in chunk]
        if all(key in cached for key in keys):
            continue
        to_call.append((chunk, prompt, keys))

    if to_call:
        from llm.groq_client import get_chat_model

        print(f"Normalizing {sum(len(c) for c, _, _ in to_call)} assertions in {len(to_call)} batches...", file=sys.stderr)
        SLOT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with SLOT_CACHE.open("a") as out:
            for done, (chunk, prompt, keys) in enumerate(to_call, 1):
                message = get_chat_model("fast", 0).invoke(prompt)
                parsed = _parse_json_object(message.content if isinstance(message.content, str) else "") or {}
                by_index = {item.get("index"): item for item in parsed.get("items", []) if isinstance(item, dict)}
                for local_index, (item, key) in enumerate(zip(chunk, keys)):
                    llm_out = by_index.get(local_index, {})
                    row = {"cache_key": key, "assertion_id": item.assertion_id, "schema_version": SCHEMA_VERSION, "llm_output": llm_out}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    cached[key] = row
                out.flush()
                if done % 5 == 0 or done == len(to_call):
                    print(f"  normalized {done}/{len(to_call)} batches", file=sys.stderr)
    return cached


# --------------------------------------------------------------------------
# Step 2b: deterministic validation of the normalization output.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SlotAssertion:
    assertion_id: str
    network_id: str
    viewpoint_owner: str
    subject_entity: str
    state_dimension: str
    slot_question: str
    value: str
    modality: str
    observed_at: str
    source_turn_ids: tuple[str, ...]
    normalization_confidence: float
    assertion_text: str


def validate_slot_normalization(
    raw: RawAssertion, llm_output: dict, observed_at: str,
) -> tuple[Optional[SlotAssertion], str]:
    echoed_sources = tuple(str(s) for s in llm_output.get("source_turn_ids", []))
    if set(echoed_sources) != set(raw.source_turn_ids):
        return None, "fabricated_or_altered_source_turn_ids"

    slot_question = str(llm_output.get("slot_question") or "").strip()
    if not slot_question:
        return None, "empty_slot_question"
    value = raw.normalized_value.strip()
    if value and _norm(value) in _norm(slot_question):
        return None, "slot_question_contains_value"

    subject_entity = str(llm_output.get("subject_entity") or "").strip()
    state_dimension = str(llm_output.get("state_dimension") or "").strip()
    if not state_dimension:
        return None, "missing_state_dimension"

    try:
        confidence = float(llm_output.get("normalization_confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    return SlotAssertion(
        assertion_id=raw.assertion_id, network_id=raw.network_id, viewpoint_owner=raw.viewpoint_owner,
        subject_entity=subject_entity, state_dimension=state_dimension, slot_question=slot_question,
        value=value, modality=raw.modality, observed_at=observed_at, source_turn_ids=raw.source_turn_ids,
        normalization_confidence=confidence, assertion_text=raw.assertion_text,
    ), "ok"


def normalize_all(
    assertions: list[RawAssertion], slot_cache: dict[str, dict], observed_at_by_turn: dict[str, str],
) -> tuple[list[SlotAssertion], list[tuple[str, str]]]:
    """Re-derives each assertion's cache key the same way build_slot_cache
    did, so this works whether the cache was just built or fully reused."""
    valid: list[SlotAssertion] = []
    rejected: list[tuple[str, str]] = []
    chunks = [assertions[i:i + NORMALIZE_CHUNK] for i in range(0, len(assertions), NORMALIZE_CHUNK)]
    for chunk in chunks:
        prompt_hash = hashlib.sha256(_slot_normalize_prompt(chunk).encode()).hexdigest()
        for item in chunk:
            key = _cache_key(item, prompt_hash)
            row = slot_cache.get(key)
            if row is None:
                rejected.append((item.assertion_id, "not_in_cache"))
                continue
            observed_at = max((observed_at_by_turn.get(t, "") for t in item.source_turn_ids), default="")
            record, reason = validate_slot_normalization(item, row["llm_output"], observed_at)
            if record is None:
                rejected.append((item.assertion_id, reason))
            else:
                valid.append(record)
    return valid, rejected


# --------------------------------------------------------------------------
# Step 3-4: candidate generation + retrieval-representation ablation
# --------------------------------------------------------------------------

def _identity_compatible(query: SlotAssertion, candidate: SlotAssertion) -> bool:
    if query.viewpoint_owner and candidate.viewpoint_owner and _norm(query.viewpoint_owner) != _norm(candidate.viewpoint_owner):
        return False
    if query.subject_entity and candidate.subject_entity and _norm(query.subject_entity) != _norm(candidate.subject_entity):
        return False
    return True


def candidate_pool(query: SlotAssertion, pool: list[SlotAssertion]) -> list[SlotAssertion]:
    """The ONLY hard filter is same network_id. Identity constraints only
    exclude a candidate when BOTH sides are resolved (non-empty) AND
    differ -- an unresolved side never causes an automatic drop."""
    return [
        c for c in pool
        if c.network_id == query.network_id
        and c.assertion_id != query.assertion_id
        and not (set(c.source_turn_ids) & set(query.source_turn_ids))
        and _identity_compatible(query, c)
    ]


_VALUE_MASK_RE_CACHE: dict[str, re.Pattern] = {}


def _mask_value(text: str, value: str) -> str:
    if not value:
        return text
    pattern = _VALUE_MASK_RE_CACHE.get(value)
    if pattern is None:
        pattern = re.compile(re.escape(value), re.IGNORECASE)
        _VALUE_MASK_RE_CACHE[value] = pattern
    return pattern.sub("[VALUE]", text)


def _variant_text(variant: str, item: SlotAssertion) -> str:
    if variant == "A_full_claim":
        return item.assertion_text
    if variant == "B_value_masked":
        return _mask_value(item.assertion_text, item.value)
    return item.slot_question  # C and D share the same base text; D adds a session blend on top


def _cosine_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return m @ q


@dataclass(frozen=True)
class RetrievalCase:
    label: str
    qa_id: str
    network_id: str
    query: SlotAssertion
    gold_target_ids: frozenset[str]  # empty for no-pair (negative) cases
    is_positive: bool


def run_retrieval_ablation(
    cases: list[RetrievalCase], slot_by_assertion: dict[str, SlotAssertion],
    pool_by_network: dict[str, list[SlotAssertion]], session_vec_by_key: dict[tuple[str, str], np.ndarray],
    turn_to_session: dict[str, tuple[str, str]],
) -> dict[str, dict]:
    text_embed_cache: dict[tuple[str, str], np.ndarray] = {}

    def embed_for(variant: str, items: list[SlotAssertion]) -> np.ndarray:
        texts = [_variant_text(variant, item) for item in items]
        key = (variant, hashlib.sha256("␟".join(texts).encode()).hexdigest())
        if key not in text_embed_cache:
            text_embed_cache[key] = np.asarray(embed_batch(texts), dtype=np.float32) if texts else np.zeros((0, 1))
        return text_embed_cache[key]

    results: dict[str, dict] = {}
    for variant in RETRIEVAL_VARIANTS:
        per_case = []
        for case in cases:
            pool = candidate_pool(case.query, pool_by_network.get(case.network_id, []))
            if not pool:
                per_case.append({"case": case, "rank": None, "empty_pool": True})
                continue
            query_vec = embed_for(variant, [case.query])[0]
            cand_vecs = embed_for(variant, pool)
            scores = _cosine_matrix(query_vec, cand_vecs)
            if variant == "D_slot_plus_conversation":
                session_key = turn_to_session.get(case.query.source_turn_ids[0])
                q_session_vec = session_vec_by_key.get(session_key) if session_key else None
                if q_session_vec is not None:
                    session_scores = np.array([
                        session_vec_by_key.get(turn_to_session.get(c.source_turn_ids[0]), np.zeros_like(q_session_vec)) @ q_session_vec
                        if turn_to_session.get(c.source_turn_ids[0]) else 0.0
                        for c in pool
                    ])
                    scores = (1 - SESSION_SOFT_WEIGHT) * scores + SESSION_SOFT_WEIGHT * session_scores
            order = np.argsort(-scores)
            ranked_ids = [pool[i].assertion_id for i in order]
            rank = None
            if case.gold_target_ids:
                for position, assertion_id in enumerate(ranked_ids, 1):
                    if assertion_id in case.gold_target_ids:
                        rank = position
                        break
            per_case.append({"case": case, "rank": rank, "empty_pool": False, "top5": ranked_ids[:5]})

        positives = [r for r in per_case if r["case"].is_positive]
        n_pos = len(positives)
        results[variant] = {
            "recall@1": sum(1 for r in positives if r["rank"] == 1) / n_pos if n_pos else float("nan"),
            "recall@3": sum(1 for r in positives if r["rank"] is not None and r["rank"] <= 3) / n_pos if n_pos else float("nan"),
            "recall@5": sum(1 for r in positives if r["rank"] is not None and r["rank"] <= 5) / n_pos if n_pos else float("nan"),
            "mrr": sum((1 / r["rank"]) if r["rank"] else 0.0 for r in positives) / n_pos if n_pos else float("nan"),
            "empty_pools_total": sum(1 for r in per_case if r["empty_pool"]),
            "n_positive": n_pos, "n_negative": len(per_case) - n_pos,
            "per_case": per_case,
        }
    return results


# --------------------------------------------------------------------------
# Step 5: bounded top-5 resolver (only run if the retrieval gate passes).
# --------------------------------------------------------------------------

def _resolver_prompt(query: SlotAssertion, candidates: list[SlotAssertion]) -> str:
    lines = []
    for letter, cand in zip("ABCDE", candidates):
        lines.append(
            f"[{letter}] cell_id={cand.assertion_id} viewpoint_owner={cand.viewpoint_owner!r} "
            f"subject_entity={cand.subject_entity!r} state_dimension={cand.state_dimension!r} "
            f"current_value={cand.value!r} observed_at={cand.observed_at}\n    \"{cand.assertion_text}\""
        )
    candidates_block = "\n".join(lines) if lines else "(no candidates)"
    return f"""Decide how this NEW assertion relates to existing candidate state cells for
the SAME network. You do not know what question or benchmark this relates to --
decide only from the content below.

NEW assertion:
  viewpoint_owner: {query.viewpoint_owner!r}
  subject_entity: {query.subject_entity!r}
  state_dimension: {query.state_dimension!r}
  slot_question: {query.slot_question!r}
  value: {query.value!r}
  observed_at: {query.observed_at}
  text: "{query.assertion_text}"

Candidate cells (top {len(candidates)}):
{candidates_block}

Choose exactly one operation:
- NEW: a different state, or no candidate is a confident match. If uncertain, choose NEW -- false merge is worse than false split.
- OBSERVE: the same state slot, same (or compatible, non-contradicting) value.
- REVISE: the same state slot, but the value genuinely changed to something incompatible with the candidate's current value, OR the text explicitly signals a change.
- RETRACT: the candidate's current value is explicitly withdrawn, with nothing new asserted in its place.

Return JSON only:
{{"selected_cell_id":"existing-assertion-id-or-null","operation":"NEW|OBSERVE|REVISE|RETRACT",
"same_viewpoint_owner":true,"same_subject_entity":true,"same_state_slot":true,
"value_relation":"same|compatible_extension|changed|contradiction|retraction|unrelated",
"confidence":0.0,"rationale":"one short sentence"}}"""


def run_resolver(query: SlotAssertion, candidates: list[SlotAssertion], resolver_cache: dict[str, dict]) -> dict:
    if not candidates:
        return {
            "selected_cell_id": None, "operation": "NEW", "same_viewpoint_owner": None, "same_subject_entity": None,
            "same_state_slot": None, "value_relation": "unrelated", "confidence": 1.0,
            "rationale": "no candidates", "defaulted": False,
        }
    prompt = _resolver_prompt(query, candidates)
    key = "resolver:" + hashlib.sha256(prompt.encode()).hexdigest()
    if key in resolver_cache:
        parsed = resolver_cache[key]
    else:
        from llm.groq_client import get_chat_model

        message = get_chat_model("fast", 0).invoke(prompt)
        parsed = _parse_json_object(message.content if isinstance(message.content, str) else "") or {}
        resolver_cache[key] = parsed
        with SLOT_CACHE.with_name(SLOT_CACHE.stem + "_resolver.jsonl").open("a") as f:
            f.write(json.dumps({"cache_key": key, "output": parsed}, ensure_ascii=False) + "\n")
    operation = str(parsed.get("operation") or "").strip().upper()
    if operation not in ("NEW", "OBSERVE", "REVISE", "RETRACT"):
        operation, defaulted = "NEW", True
    else:
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        defaulted = confidence < 0.5
        if defaulted:
            operation = "NEW"
    selected = parsed.get("selected_cell_id") if operation != "NEW" else None
    valid_ids = {c.assertion_id for c in candidates}
    if selected not in valid_ids:
        selected = None
        if operation != "NEW":
            operation, defaulted = "NEW", True
    return {**parsed, "operation": operation, "selected_cell_id": selected, "defaulted": defaulted}


def load_resolver_cache() -> dict[str, dict]:
    path = SLOT_CACHE.with_name(SLOT_CACHE.stem + "_resolver.jsonl")
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["cache_key"]] = row["output"]
    return cache


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def render_retrieval_report(ablation: dict[str, dict], gate_variant: str) -> str:
    lines = [
        "# Semantic slot linking pilot -- retrieval ablation (A/B/C/D)",
        "",
        "No LLM resolver used for this report -- pure embedding retrieval only. n is tiny (3 confirmed "
        "positive pairs) because that is the entire set of manually confirmed same-state pairs available; "
        "reported honestly, not smoothed over.",
        "",
        "| variant | recall@1 | recall@3 | recall@5 | MRR | empty pools | n_positive | n_negative |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for variant in RETRIEVAL_VARIANTS:
        r = ablation[variant]
        lines.append(
            f"| {variant} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | {r['recall@5']:.3f} | {r['mrr']:.3f} | "
            f"{r['empty_pools_total']} | {r['n_positive']} | {r['n_negative']} |"
        )
    lines += ["", f"## Success gate (recall@5 >= {RECALL_AT_5_GATE} on confirmed pairs)", ""]
    for variant in RETRIEVAL_VARIANTS:
        r = ablation[variant]
        passed = (r["recall@5"] >= RECALL_AT_5_GATE) if r["recall@5"] == r["recall@5"] else False
        lines.append(f"- {variant}: recall@5={r['recall@5']:.3f} -> {'PASS' if passed else 'FAIL'}")
    lines += ["", f"## Best variant selected for resolver stage: {gate_variant}", "", "## Per-case detail", ""]
    for variant in RETRIEVAL_VARIANTS:
        lines.append(f"### {variant}")
        for row in ablation[variant]["per_case"]:
            case = row["case"]
            lines.append(
                f"- {case.qa_id} ({case.label}) query={case.query.assertion_id} "
                f"gold_targets={sorted(case.gold_target_ids) or 'n/a (negative)'} "
                f"rank={row['rank']} empty_pool={row['empty_pool']} top5={row.get('top5')}"
            )
        lines.append("")
    return "\n".join(lines)


def render_resolver_report(resolver_rows: list[dict]) -> str:
    lines = [
        "# Semantic slot linking pilot -- bounded top-5 resolver", "",
        "Resolver never saw benchmark question/gold answer/evidence labels -- only assertion content and "
        "candidate cells.", "",
        "| qa_id | label | gold_relation | predicted_op | selected_matches_gold | confidence | defaulted |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in resolver_rows:
        lines.append(
            f"| {row['qa_id']} | {row['label']} | {row.get('gold_relation','n/a')} | {row['operation']} | "
            f"{row.get('selected_matches_gold','n/a')} | {row.get('confidence', 0.0):.2f} | {row['defaulted']} |"
        )
    lines += ["", "## Rationale detail", ""]
    for row in resolver_rows:
        lines.append(f"- {row['qa_id']} ({row['label']}): {row.get('rationale', '')}")
    return "\n".join(lines)


def classify_false_split(
    query_valid: bool, target_valid: bool, is_confirmed_pair: bool, same_cell: bool,
) -> Optional[bool]:
    """False split can only be declared when BOTH sides successfully
    normalized AND this is a manually confirmed same-state pair -- returns
    None (undeclarable) otherwise, never a default False. None == None (two
    unresolved/unusable sides) is explicitly NOT a match here."""
    if not (query_valid and target_valid and is_confirmed_pair):
        return None
    return not same_cell


def classify_false_merge(
    query_valid: bool, target_valid: bool, is_confirmed_different: bool, same_cell: bool,
) -> Optional[bool]:
    """Mirror of classify_false_split -- only declarable for a manually
    confirmed DIFFERENT-state pair, both sides valid. This pilot has no such
    confirmed-negative PAIRS (only confirmed absence-of-positive-pair cases),
    so in practice this always returns None here -- surfaced via the review
    sheet instead, never auto-declared."""
    if not (query_valid and target_valid and is_confirmed_different):
        return None
    return same_cell


def render_review_sheet(rows: list[dict]) -> str:
    lines = [
        "# Semantic slot linking pilot -- manual review needed (false-merge candidates)", "",
        "The 9 'No valid pair' cases give us confirmed ABSENCE of a known positive pair, not a confirmed "
        "pair of two specific assertions being definitely different states. Any resolver decision other than "
        "NEW on these queries is therefore a REVIEW CANDIDATE, not an auto-confirmed false merge.", "",
    ]
    for row in rows:
        lines.append(f"## {row['qa_id']} -- query {row['query_id']}")
        lines.append(f"- query text: {row['query_text']}")
        lines.append(f"- resolver operation: {row['operation']} (confidence={row.get('confidence',0):.2f})")
        lines.append(f"- selected candidate: {row.get('selected_cell_id')}")
        lines.append(f"- candidate text: {row.get('selected_text','')}")
        lines.append(f"- rationale: {row.get('rationale','')}")
        lines.append("- Decision: [ ] correct link  [ ] false merge  [ ] unclear")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Step 7: diagnostic chain materialization for confirmed pairs only.
# --------------------------------------------------------------------------

@dataclass
class ChainVersion:
    version_id: str
    operation: str
    assertion_id: str
    value: str
    observed_at: str
    source_turn_ids: tuple[str, ...]
    hot: bool = True


@dataclass
class DiagnosticChain:
    cell_id: str
    network_id: str
    viewpoint_owner: str
    subject_entity: str
    state_dimension: str
    versions: list[ChainVersion]


def materialize_chain(cell_id: str, old: SlotAssertion, new: SlotAssertion, operation: str) -> DiagnosticChain:
    """Ordered state versions, current separated from historical, source_
    turn_ids kept per version, nothing deleted -- old assertion events are
    read-only inputs, never mutated. At most MAX_HOT_VERSIONS stay 'hot';
    this pilot never exceeds that, so no version is cold here, but the
    field exists to define the format Stage 6+ would need."""
    versions = [
        ChainVersion("v1", "create", old.assertion_id, old.value, old.observed_at, old.source_turn_ids),
        ChainVersion("v2", operation.lower(), new.assertion_id, new.value, new.observed_at, new.source_turn_ids),
    ]
    for index, version in enumerate(versions):
        version.hot = (len(versions) - index) <= MAX_HOT_VERSIONS
    return DiagnosticChain(
        cell_id=cell_id, network_id=old.network_id, viewpoint_owner=old.viewpoint_owner,
        subject_entity=old.subject_entity or new.subject_entity, state_dimension=old.state_dimension,
        versions=versions,
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _assertions_citing(pool: list[SlotAssertion], turn_id: Optional[str]) -> list[SlotAssertion]:
    if not turn_id:
        return []
    return [a for a in pool if turn_id in a.source_turn_ids]


def main() -> None:
    print("Step 1: inventory + gold labels", file=sys.stderr)
    if not SESSION_ASSERTIONS.exists():
        print(f"MISSING immutable artifact: {SESSION_ASSERTIONS}", file=sys.stderr)
        raise SystemExit(1)
    if not PAIR_REVIEW_MD.exists():
        print(f"MISSING immutable artifact: {PAIR_REVIEW_MD}", file=sys.stderr)
        raise SystemExit(1)

    gold_pairs, gold_no_pairs = parse_manual_review(PAIR_REVIEW_MD)
    print(f"  confirmed pairs: {len(gold_pairs)}; no-valid-pair cases: {len(gold_no_pairs)}", file=sys.stderr)
    for gp in gold_pairs:
        print(f"    PAIR {gp.qa_id} :: {gp.network_id} :: {gp.who} :: {gp.relation}", file=sys.stderr)

    networks = {gp.network_id for gp in gold_pairs} | {gnp.network_id for gnp in gold_no_pairs}
    assertions = load_bounded_assertions(networks)
    print(f"  bounded raw assertions across {len(networks)} networks: {len(assertions)}", file=sys.stderr)

    conversations = load_conversations(networks)
    observed_at_by_turn = {str(r.turn_id): str(r.timestamp) for r in conversations.itertuples(index=False)}
    turn_to_session = {str(r.turn_id): (str(r.network_id), str(r.session_id)) for r in conversations.itertuples(index=False)}
    session_text: dict[tuple[str, str], str] = defaultdict(str)
    for r in conversations.itertuples(index=False):
        key = (str(r.network_id), str(r.session_id))
        session_text[key] += f" {r.message}"

    print("Step 2: slot normalization (cached, resumable)", file=sys.stderr)
    llm_calls_before = 0
    if SLOT_CACHE.exists():
        llm_calls_before = sum(1 for _ in SLOT_CACHE.open())
    slot_cache = build_slot_cache(assertions)
    llm_calls_after = sum(1 for _ in SLOT_CACHE.open()) if SLOT_CACHE.exists() else 0
    new_normalize_calls_rows = llm_calls_after - llm_calls_before

    slots, rejected = normalize_all(assertions, slot_cache, observed_at_by_turn)
    print(f"  normalized valid: {len(slots)}  rejected: {len(rejected)}", file=sys.stderr)
    slot_by_assertion = {s.assertion_id: s for s in slots}
    pool_by_network: dict[str, list[SlotAssertion]] = defaultdict(list)
    for s in slots:
        pool_by_network[s.network_id].append(s)

    print("  session embeddings (soft signal for variant D)", file=sys.stderr)
    session_keys = list(session_text.keys())
    session_vecs = np.asarray(embed_batch([session_text[k][:4000] for k in session_keys]), dtype=np.float32) if session_keys else np.zeros((0, 1))
    session_vec_by_key = {
        key: vec / (np.linalg.norm(vec) + 1e-12) for key, vec in zip(session_keys, session_vecs)
    }

    print("Step 3: smoke test on confirmed pairs", file=sys.stderr)
    positive_cases: list[RetrievalCase] = []
    for gp in gold_pairs:
        old_candidates = _assertions_citing(pool_by_network.get(gp.network_id, []), gp.old_turn_id)
        new_candidates = _assertions_citing(pool_by_network.get(gp.network_id, []), gp.new_turn_id)
        if not old_candidates or not new_candidates:
            print(f"  SKIP {gp.qa_id}: old or new side did not normalize successfully "
                  f"(old={len(old_candidates)} new={len(new_candidates)})", file=sys.stderr)
            continue
        gold_target_ids = frozenset(a.assertion_id for a in old_candidates)
        for query in new_candidates:
            positive_cases.append(RetrievalCase(
                label=gp.who, qa_id=gp.qa_id, network_id=gp.network_id, query=query,
                gold_target_ids=gold_target_ids, is_positive=True,
            ))
    print(f"  usable confirmed pairs (both sides normalized): {len({c.qa_id for c in positive_cases})}/{len(gold_pairs)}", file=sys.stderr)

    negative_cases: list[RetrievalCase] = []
    for gnp in gold_no_pairs:
        new_candidates = _assertions_citing(pool_by_network.get(gnp.network_id, []), gnp.new_turn_id)
        for query in new_candidates:
            negative_cases.append(RetrievalCase(
                label=gnp.network_id, qa_id=gnp.qa_id, network_id=gnp.network_id, query=query,
                gold_target_ids=frozenset(), is_positive=False,
            ))

    print("Step 4: retrieval ablation A/B/C/D on the bounded Q8 set", file=sys.stderr)
    all_cases = positive_cases + negative_cases
    ablation = run_retrieval_ablation(all_cases, slot_by_assertion, pool_by_network, session_vec_by_key, turn_to_session)
    best_variant = max(RETRIEVAL_VARIANTS, key=lambda v: (ablation[v]["recall@5"] if ablation[v]["recall@5"] == ablation[v]["recall@5"] else -1))
    RETRIEVAL_REPORT_MD.write_text(render_retrieval_report(ablation, best_variant))
    print(f"  retrieval report: {RETRIEVAL_REPORT_MD}", file=sys.stderr)

    print("Step 5: success gate check", file=sys.stderr)
    gate_recall = ablation[best_variant]["recall@5"]
    gate_passed = (gate_recall == gate_recall) and gate_recall >= RECALL_AT_5_GATE
    print(f"  best variant={best_variant} recall@5={gate_recall} gate={'PASS' if gate_passed else 'FAIL'}", file=sys.stderr)

    resolver_rows: list[dict] = []
    review_rows: list[dict] = []
    chains: list[DiagnosticChain] = []
    false_merge_flags = 0

    if gate_passed:
        print("Step 6: bounded top-5 resolver", file=sys.stderr)
        resolver_cache = load_resolver_cache()
        best = ablation[best_variant]
        by_case_id = {id(row["case"]): row for row in best["per_case"]}

        for row in best["per_case"]:
            case = row["case"]
            if row["empty_pool"]:
                continue
            top5_ids = row["top5"]
            candidates = [slot_by_assertion[cid] for cid in top5_ids]
            decision = run_resolver(case.query, candidates, resolver_cache)
            selected_matches_gold = (
                decision["selected_cell_id"] in case.gold_target_ids if case.is_positive else None
            )
            gold_relation = next((gp.relation for gp in gold_pairs if gp.qa_id == case.qa_id), None)
            resolver_rows.append({
                "qa_id": case.qa_id, "label": case.label, "operation": decision["operation"],
                "selected_cell_id": decision["selected_cell_id"], "confidence": decision.get("confidence", 0.0),
                "defaulted": decision["defaulted"], "rationale": decision.get("rationale", ""),
                "gold_relation": gold_relation, "selected_matches_gold": selected_matches_gold,
            })
            if not case.is_positive and decision["operation"] != "NEW":
                false_merge_flags += 1
                review_rows.append({
                    "qa_id": case.qa_id, "query_id": case.query.assertion_id, "query_text": case.query.assertion_text,
                    "operation": decision["operation"], "confidence": decision.get("confidence", 0.0),
                    "selected_cell_id": decision["selected_cell_id"],
                    "selected_text": slot_by_assertion[decision["selected_cell_id"]].assertion_text if decision["selected_cell_id"] in slot_by_assertion else "",
                    "rationale": decision.get("rationale", ""),
                })

        print("Step 7: diagnostic chain materialization", file=sys.stderr)
        for gp in gold_pairs:
            old_candidates = _assertions_citing(pool_by_network.get(gp.network_id, []), gp.old_turn_id)
            new_candidates = _assertions_citing(pool_by_network.get(gp.network_id, []), gp.new_turn_id)
            if not old_candidates or not new_candidates:
                continue
            row = next((r for r in resolver_rows if r["qa_id"] == gp.qa_id), None)
            if row is None or row["operation"] not in ("REVISE", "RETRACT", "OBSERVE"):
                continue
            chains.append(materialize_chain(f"diag_{gp.qa_id}", old_candidates[0], new_candidates[0], row["operation"]))

        RESOLVER_REPORT_MD.write_text(render_resolver_report(resolver_rows))
        print(f"  resolver report: {RESOLVER_REPORT_MD}", file=sys.stderr)
        if review_rows:
            REVIEW_MD.write_text(render_review_sheet(review_rows))
            print(f"  review sheet (possible false merges, NOT auto-confirmed): {REVIEW_MD}", file=sys.stderr)
    else:
        print("  retrieval gate FAILED -- resolver/materialization skipped per protocol", file=sys.stderr)

    print("Step 8: final summary", file=sys.stderr)
    chain_recovered = sum(
        1 for r in resolver_rows
        if r["gold_relation"] and r["operation"] == r["gold_relation"] and r["selected_matches_gold"]
    )
    summary = {
        "normalization_coverage": {"valid": len(slots), "rejected": len(rejected), "total": len(assertions)},
        "usable_confirmed_pairs": len({c.qa_id for c in positive_cases}),
        "retrieval_gate": {"variant": best_variant, "recall@5": gate_recall, "passed": gate_passed},
        "resolver_ran": gate_passed,
        "chain_recovery": f"{chain_recovered}/{len({c.qa_id for c in positive_cases}) if positive_cases else 0}",
        "false_merge_flags_for_review": false_merge_flags,
        "new_normalize_cache_rows_this_run": new_normalize_calls_rows,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
