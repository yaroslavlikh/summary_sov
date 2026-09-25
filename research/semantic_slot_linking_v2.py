"""Semantic slot linking pilot v2 -- automatic, no manual labeling.

v1 (research/semantic_slot_linking_pilot.py) failed statistically: 3 confirmed
pairs, 1 usable, recall@k measured on n=1. That is a diagnostic failure of
the EXPERIMENT SETUP, not evidence against value-free slot linking. v2 fixes
this by building the evaluation set automatically from the full 262-question
Q8 slice of SocialMemBench (QA gold is used ONLY to construct evaluation
labels -- never shown to the normalizer or the retrieval/resolver system
under test), and by fixing a real conceptual bug in v1's normalization
schema: subject_entity was overloaded for both "whose state this is" and
"what/who the state is about" (v1's Diane failure -- "fourth photograph" was
extracted as subject_entity when it should have been topic_object, with
Diane as state_holder).

Two measurements, kept separate as required:
  1. ORACLE-EXTRACTION LINKING (Phase 3-4) -- can slot-question/structured-
     slot retrieval find the correct OLD state among in-network candidates,
     when old/new states are GUARANTEED present (built from benchmark gold,
     not from what a session extractor happened to keep)?
  2. END-TO-END COVERAGE (Phase 5) -- separately, how much of this same gold
     state actually survives in the frozen full run's session_assertions.jsonl?
     Never multiplied together into a single end-to-end number.

This is a post-hoc diagnostic/development analysis over a frozen snapshot,
NOT a held-out leaderboard result and NOT mixed with the already-completed
full-benchmark numbers.

Read-only inputs (never written to):
  .research_runs/frozen/socialmembench_full_official_v1_20260911_184037_MSK/
  .research_runs/socialmembench_full_official_v1/   (also never touched)

All new artifacts:
  .research_runs/semantic_slot_linking_v2/

Run:
    python3 -m research.semantic_slot_linking_v2
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from embeddings import embed_batch
from research.socialmembench_pilot import _norm, _parse_json_object

FROZEN_DIR = Path(".research_runs/frozen/socialmembench_full_official_v1_20260911_184037_MSK")
OUT_DIR = Path(".research_runs/semantic_slot_linking_v2")

SCHEMA_VERSION = "slot_v2"
CONSTRUCTION_MODEL = "fast"
NORMALIZE_MODEL = "fast"
VERIFY_MODEL = "fast"
TEMPERATURE = 0

CONSTRUCTION_CACHE = OUT_DIR / "construction_cache.jsonl"
CONSTRUCTION_VERIFIER_CACHE = OUT_DIR / "construction_verifier_cache.jsonl"
NORMALIZATION_CACHE = OUT_DIR / "normalization_cache.jsonl"
TIER2_VERIFIER_CACHE = OUT_DIR / "tier2_verifier_cache.jsonl"
RESOLVER_CACHE = OUT_DIR / "resolver_cache.jsonl"

Q8_CASES_JSONL = OUT_DIR / "q8_linking_cases.jsonl"
Q8_AMBIGUOUS_JSONL = OUT_DIR / "q8_linking_ambiguous.jsonl"
Q8_META_JSON = OUT_DIR / "q8_linking_dataset_meta.json"
NORMALIZATION_REJECTED = OUT_DIR / "normalization_rejected.jsonl"
RETRIEVAL_RESULTS = OUT_DIR / "retrieval_results.jsonl"
NEGATIVE_PAIRS = OUT_DIR / "negative_pairs.jsonl"
RESOLVER_RESULTS = OUT_DIR / "resolver_results.jsonl"
EXTRACTION_COVERAGE = OUT_DIR / "extraction_coverage.jsonl"
CONFIG_JSON = OUT_DIR / "config.json"
REPORT_MD = OUT_DIR / "report.md"
RUN_LOG = OUT_DIR / "run.log"
README_MD = OUT_DIR / "README.md"

AMBIGUITY_CONFIDENCE_THRESHOLD = 0.6
CONSTRUCT_WORKERS = 8
NORMALIZE_CHUNK = 20
SPLIT_SEED = 20260912
DEV_FRACTION = 0.70
RETRIEVAL_VARIANTS = ("A_full_claim", "B_slot_question", "C_structured_slot", "D_structured_plus_context")
SESSION_SOFT_WEIGHT = 0.15
TIER2_SIMILARITY_THRESHOLD = 0.75

GATE_MIN_USABLE_PAIRS = 30
GATE_RECALL_AT_5 = 0.75
GATE_CANDIDATE_SURVIVAL = 0.90
GATE_FALSE_MERGE_AT_5 = 0.10


def _log(message: str) -> None:
    print(message, file=sys.stderr)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a") as f:
        f.write(message + "\n")


# --------------------------------------------------------------------------
# Frozen snapshot verification
# --------------------------------------------------------------------------

def verify_frozen_snapshot() -> dict:
    manifest = json.loads((FROZEN_DIR / "MANIFEST.json").read_text())
    mismatches = []
    for relpath, meta in manifest["files"].items():
        path = FROZEN_DIR / relpath
        if not path.exists():
            mismatches.append((relpath, "missing"))
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != meta["sha256"]:
            mismatches.append((relpath, f"sha256 mismatch: expected {meta['sha256']} got {actual}"))
    if mismatches:
        raise RuntimeError(f"frozen snapshot integrity check FAILED: {mismatches[:5]} (+{max(0,len(mismatches)-5)} more)")
    return manifest


def load_frozen_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qa = pd.read_parquet(FROZEN_DIR / "dataset" / "qa.parquet")
    conversations = pd.read_parquet(FROZEN_DIR / "dataset" / "conversations.parquet")
    networks = pd.read_parquet(FROZEN_DIR / "dataset" / "networks.parquet")
    return qa, conversations, networks


def _json_field(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and value != value):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


# --------------------------------------------------------------------------
# PHASE 1: automatic Q8 linking dataset construction
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceRecord:
    index: int
    turn_id: str
    session_index: int
    speaker: str
    message: str
    timestamp: str
    relevance: str


@dataclass(frozen=True)
class Q8LinkingCase:
    qa_id: str
    network_id: str
    state_holder: str
    viewpoint_owner: Optional[str]
    topic_object: str
    old_state_text: str
    new_state_text: str
    trigger_text: Optional[str]
    old_source_turn_ids: tuple[str, ...]
    new_source_turn_ids: tuple[str, ...]
    old_evidence_indices: tuple[int, ...]
    new_evidence_indices: tuple[int, ...]
    relation: str
    construction_confidence: float
    construction_notes: str
    verified: bool


def _build_evidence_records(row, conversations: pd.DataFrame) -> list[EvidenceRecord]:
    anchors = _json_field(row["evidence_anchors_json"], [])
    net_turns = conversations[conversations.network_id == row["network_id"]].set_index("turn_id")
    records = []
    for index, anchor in enumerate(anchors):
        turn_id = anchor.get("turn_id")
        if not turn_id or turn_id not in net_turns.index:
            continue
        turn = net_turns.loc[turn_id]
        records.append(EvidenceRecord(
            index=index, turn_id=turn_id, session_index=int(anchor.get("session_index", turn.session_index)),
            speaker=str(turn.speaker_display_name), message=str(turn.message), timestamp=str(turn.timestamp),
            relevance=str(anchor.get("relevance") or ""),
        ))
    return records


def _construction_prompt(question: str, answer: str, records: list[EvidenceRecord]) -> str:
    rows = "\n".join(
        f"[{r.index}] session={r.session_index} {r.speaker}: {r.message}\n    (why this is evidence: {r.relevance})"
        for r in records
    )
    return f"""This is a temporal-shift QA case from a social-memory benchmark. Use the
question, gold answer, and evidence records to construct a structured OLD-state
vs NEW-state description of the tracked change.

Question: {question}
Gold answer: {answer}

Evidence records (indexed):
{rows}

Return JSON only, referencing evidence ONLY by the [index] numbers shown above
(never invent or copy turn IDs -- indices only):

{{"state_holder":"the person/entity whose state changed",
"viewpoint_owner":"who holds this view, if it is an opinion, else null",
"topic_object":"what/who the state is about (may equal state_holder for self-states)",
"old_state_text":"self-contained description of the state BEFORE the change",
"new_state_text":"self-contained description of the state AFTER the change",
"trigger_text":"what caused the change, if evident, else null",
"old_evidence_indices":[0],
"new_evidence_indices":[1],
"relation":"REVISE|OBSERVE|AUGMENT|UNKNOWN",
"construction_confidence":0.0,
"construction_notes":"one short sentence"}}

relation semantics: REVISE = old value replaced by an incompatible new value;
OBSERVE = the same state reaffirmed, no real change; AUGMENT = a compatible
additional detail, not a replacement; UNKNOWN = cannot confidently separate
old/new from the evidence shown."""


def _verifier_prompt(question: str, answer: str, records: list[EvidenceRecord]) -> str:
    rows = "\n".join(f"[{r.index}] session={r.session_index} {r.speaker}: {r.message}" for r in records)
    return f"""Independently verify a temporal-shift QA case. Do NOT assume any prior answer.
From scratch, decide the OLD state, the NEW state, and the relation between them.

Question: {question}
Gold answer: {answer}

Evidence records (indexed):
{rows}

Return JSON only, indices only, never invented turn IDs:
{{"state_holder":"...","viewpoint_owner":"... or null","topic_object":"...",
"old_state_text":"...","new_state_text":"...","trigger_text":"... or null",
"old_evidence_indices":[0],"new_evidence_indices":[1],
"relation":"REVISE|OBSERVE|AUGMENT|UNKNOWN","construction_confidence":0.0,
"construction_notes":"..."}}"""


def _cache_get(path: Path, key: str) -> Optional[dict]:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["cache_key"] == key:
                return row["output"]
    return None


def _load_cache_dict(path: Path) -> dict[str, dict]:
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["cache_key"]] = row["output"]
    return cache


def _append_cache(path: Path, key: str, output: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"cache_key": key, "output": output}, ensure_ascii=False) + "\n")


def _call_llm(prompt: str, model: str = "fast") -> dict:
    from llm.groq_client import get_chat_model

    message = get_chat_model(model, TEMPERATURE).invoke(prompt)
    return _parse_json_object(message.content if isinstance(message.content, str) else "") or {}


def _resolve_indices(raw_indices: list, n_records: int) -> Optional[tuple[int, ...]]:
    try:
        indices = [int(i) for i in raw_indices]
    except (TypeError, ValueError):
        return None
    if any(i < 0 or i >= n_records for i in indices):
        return None
    return tuple(dict.fromkeys(indices))


def construct_q8_dataset(qa: pd.DataFrame, conversations: pd.DataFrame) -> tuple[list[Q8LinkingCase], list[dict], dict]:
    q8_rows = qa[qa.query_type == "Q8"].to_dict("records")
    total = len(q8_rows)
    _log(f"Phase 1: {total} Q8 rows found")

    construction_cache = _load_cache_dict(CONSTRUCTION_CACHE)
    verifier_cache = _load_cache_dict(CONSTRUCTION_VERIFIER_CACHE)

    stats = Counter()
    cases: list[Q8LinkingCase] = []
    ambiguous: list[dict] = []
    construction_calls = 0
    verifier_calls = 0

    for row in q8_rows:
        records = _build_evidence_records(row, conversations)
        if len(records) < 2:
            stats["missing_evidence"] += 1
            continue

        prompt = _construction_prompt(row["question"], row["answer"], records)
        key = "construct:" + hashlib.sha256((row["qa_id"] + prompt).encode()).hexdigest()
        output = construction_cache.get(key)
        if output is None:
            output = _call_llm(prompt, CONSTRUCTION_MODEL)
            construction_cache[key] = output
            _append_cache(CONSTRUCTION_CACHE, key, output)
            construction_calls += 1

        old_idx = _resolve_indices(output.get("old_evidence_indices", []), len(records))
        new_idx = _resolve_indices(output.get("new_evidence_indices", []), len(records))
        try:
            confidence = float(output.get("construction_confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        relation = str(output.get("relation") or "UNKNOWN").strip().upper()
        if relation not in ("REVISE", "OBSERVE", "AUGMENT", "UNKNOWN"):
            relation = "UNKNOWN"

        if old_idx is None or new_idx is None or not old_idx or not new_idx:
            stats["malformed"] += 1
            ambiguous.append({"qa_id": row["qa_id"], "network_id": row["network_id"], "reason": "malformed_indices", "primary": output})
            continue

        verified = True
        needs_verify = confidence < AMBIGUITY_CONFIDENCE_THRESHOLD or relation == "UNKNOWN"
        if needs_verify:
            vprompt = _verifier_prompt(row["question"], row["answer"], records)
            vkey = "verify:" + hashlib.sha256((row["qa_id"] + vprompt).encode()).hexdigest()
            voutput = verifier_cache.get(vkey)
            if voutput is None:
                voutput = _call_llm(vprompt, VERIFY_MODEL)
                verifier_cache[vkey] = voutput
                _append_cache(CONSTRUCTION_VERIFIER_CACHE, vkey, voutput)
                verifier_calls += 1
            v_relation = str(voutput.get("relation") or "UNKNOWN").strip().upper()
            v_old_idx = _resolve_indices(voutput.get("old_evidence_indices", []), len(records))
            v_new_idx = _resolve_indices(voutput.get("new_evidence_indices", []), len(records))
            agree_relation = v_relation == relation and relation != "UNKNOWN"
            agree_overlap = bool(v_old_idx and set(v_old_idx) & set(old_idx)) and bool(v_new_idx and set(v_new_idx) & set(new_idx))
            if not (agree_relation and agree_overlap):
                stats["construction_ambiguous"] += 1
                ambiguous.append({
                    "qa_id": row["qa_id"], "network_id": row["network_id"], "reason": "verifier_disagreement",
                    "primary": output, "verifier": voutput,
                })
                continue
            verified = True
        else:
            verified = False

        old_turn_ids = tuple(records[i].turn_id for i in old_idx)
        new_turn_ids = tuple(records[i].turn_id for i in new_idx)
        cases.append(Q8LinkingCase(
            qa_id=row["qa_id"], network_id=row["network_id"],
            state_holder=str(output.get("state_holder") or "").strip(),
            viewpoint_owner=(str(output.get("viewpoint_owner")).strip() if output.get("viewpoint_owner") else None),
            topic_object=str(output.get("topic_object") or "").strip(),
            old_state_text=str(output.get("old_state_text") or "").strip(),
            new_state_text=str(output.get("new_state_text") or "").strip(),
            trigger_text=(str(output.get("trigger_text")).strip() if output.get("trigger_text") else None),
            old_source_turn_ids=old_turn_ids, new_source_turn_ids=new_turn_ids,
            old_evidence_indices=old_idx, new_evidence_indices=new_idx,
            relation=relation, construction_confidence=confidence,
            construction_notes=str(output.get("construction_notes") or ""), verified=verified,
        ))
        stats["usable"] += 1

    meta = {
        "schema_version": SCHEMA_VERSION, "total_q8": total, "usable": stats["usable"],
        "ambiguous": stats["construction_ambiguous"], "missing_evidence": stats["missing_evidence"],
        "malformed": stats["malformed"], "relation_distribution": dict(Counter(c.relation for c in cases)),
        "construction_llm_calls_this_run": construction_calls, "verifier_llm_calls_this_run": verifier_calls,
        "model": CONSTRUCTION_MODEL, "temperature": TEMPERATURE,
    }
    _log(f"Phase 1 done: usable={stats['usable']} ambiguous={stats['construction_ambiguous']} "
         f"missing_evidence={stats['missing_evidence']} malformed={stats['malformed']}")
    return cases, ambiguous, meta


# --------------------------------------------------------------------------
# PHASE 2: corrected normalization schema
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedSlot:
    case_qa_id: str
    side: str  # old | new
    network_id: str
    state_holder: str
    viewpoint_owner: Optional[str]
    topic_object: str
    state_dimension: str
    slot_question: str
    value: str
    assertion_type: str
    normalization_confidence: float
    source_turn_ids: tuple[str, ...]
    state_text: str


_VALUE_LEAK_RE_CACHE: dict[str, re.Pattern] = {}


def _value_leaked(slot_question: str, value: str) -> bool:
    """Reject only a genuine value leak -- the value phrase itself appearing
    in the question -- not any incidental shared single word."""
    value_norm = _norm(value)
    if not value_norm or len(value_norm) < 3:
        return False
    pattern = _VALUE_LEAK_RE_CACHE.get(value_norm)
    if pattern is None:
        pattern = re.compile(re.escape(value_norm))
        _VALUE_LEAK_RE_CACHE[value_norm] = pattern
    return bool(pattern.search(_norm(slot_question)))


def _normalize_prompt(state_text: str, records: list[EvidenceRecord]) -> str:
    rows = "\n".join(f"[{r.index}] {r.speaker}: {r.message}" for r in records)
    return f"""Produce a value-free state-slot normalization of this state description.

State description: "{state_text}"

Supporting evidence (indexed, for grounding only):
{rows}

Distinguish these roles precisely -- do NOT conflate them:
- state_holder: WHOSE state/decision/condition this is (a person or entity).
- viewpoint_owner: who HOLDS this view, only if it is an opinion about someone/
  something else (often same as state_holder for self-reports; null otherwise).
- topic_object: WHAT or WHO the state is about (can equal state_holder for a
  self-state like health/location; must be a different entity for an opinion,
  e.g. "Tigmen said Gordey is an asshole" -> state_holder=Tigmen,
  viewpoint_owner=Tigmen, topic_object=Gordey). NEVER put a plan/object/choice
  phrase like "fourth photograph" into state_holder -- that belongs in
  topic_object or value, not state_holder.
- state_dimension: stable axis of change (opinion, health, location, plan,
  relationship, preference, commitment, availability, role, etc).
- slot_question: a value-free question this state answers -- must NOT contain
  the value itself.
- value: the concrete current value (kept OUT of slot_question).
- assertion_type: STATE (an observed value) | TRANSITION (explicit change
  language) | CAUSE (explains why a state changed) | REACTION (someone else's
  reaction to the state).

Return JSON only, source_record_indices referencing ONLY the [index] numbers
shown above (never invent IDs):
{{"state_holder":"...","viewpoint_owner":"... or null","topic_object":"...",
"state_dimension":"...","slot_question":"...","value":"...",
"assertion_type":"STATE|TRANSITION|CAUSE|REACTION","normalization_confidence":0.0,
"source_record_indices":[0]}}"""


def _normalize_cache_key(case_qa_id: str, side: str, prompt: str) -> str:
    return hashlib.sha256(json.dumps(
        {"case": case_qa_id, "side": side, "schema": SCHEMA_VERSION, "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()},
        sort_keys=True,
    ).encode()).hexdigest()


def validate_normalization(
    llm_output: dict, records: list[EvidenceRecord],
) -> tuple[Optional[dict], str]:
    raw_indices = llm_output.get("source_record_indices", [])
    indices = _resolve_indices(raw_indices, len(records))
    if indices is None:
        return None, "invalid_source_index"
    if not indices:
        return None, "no_source_indices"

    state_holder = str(llm_output.get("state_holder") or "").strip()
    if not state_holder:
        return None, "missing_state_holder"

    slot_question = str(llm_output.get("slot_question") or "").strip()
    if not slot_question:
        return None, "empty_slot_question"
    value = str(llm_output.get("value") or "").strip()
    if _value_leaked(slot_question, value):
        return None, "value_leaked_into_slot_question"

    assertion_type = str(llm_output.get("assertion_type") or "").strip().upper()
    if assertion_type not in ("STATE", "TRANSITION", "CAUSE", "REACTION"):
        return None, "invalid_assertion_type"

    try:
        confidence = float(llm_output.get("normalization_confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    return {
        "state_holder": state_holder,
        "viewpoint_owner": (str(llm_output.get("viewpoint_owner")).strip() if llm_output.get("viewpoint_owner") else None),
        "topic_object": str(llm_output.get("topic_object") or "").strip(),
        "state_dimension": str(llm_output.get("state_dimension") or "").strip(),
        "slot_question": slot_question, "value": value, "assertion_type": assertion_type,
        "normalization_confidence": confidence, "source_indices": indices,
    }, "ok"


def normalize_cases(cases: list[Q8LinkingCase], conversations: pd.DataFrame) -> tuple[list[NormalizedSlot], list[dict]]:
    cache = _load_cache_dict(NORMALIZATION_CACHE)
    slots: list[NormalizedSlot] = []
    rejected: list[dict] = []
    new_calls = 0

    for case in cases:
        for side, text, turn_ids in (
            ("old", case.old_state_text, case.old_source_turn_ids),
            ("new", case.new_state_text, case.new_source_turn_ids),
        ):
            net_turns = conversations[conversations.network_id == case.network_id].set_index("turn_id")
            records = [
                EvidenceRecord(index=i, turn_id=t, session_index=int(net_turns.loc[t].session_index),
                                speaker=str(net_turns.loc[t].speaker_display_name), message=str(net_turns.loc[t].message),
                                timestamp=str(net_turns.loc[t].timestamp), relevance="")
                for i, t in enumerate(turn_ids) if t in net_turns.index
            ]
            if not records:
                rejected.append({"qa_id": case.qa_id, "side": side, "reason": "no_valid_evidence_records"})
                continue
            prompt = _normalize_prompt(text, records)
            key = _normalize_cache_key(case.qa_id, side, prompt)
            output = cache.get(key)
            if output is None:
                output = _call_llm(prompt, NORMALIZE_MODEL)
                cache[key] = output
                _append_cache(NORMALIZATION_CACHE, key, output)
                new_calls += 1

            validated, reason = validate_normalization(output, records)
            if validated is None:
                rejected.append({"qa_id": case.qa_id, "side": side, "reason": reason})
                continue
            source_turn_ids = tuple(records[i].turn_id for i in validated["source_indices"])
            slots.append(NormalizedSlot(
                case_qa_id=case.qa_id, side=side, network_id=case.network_id,
                state_holder=validated["state_holder"], viewpoint_owner=validated["viewpoint_owner"],
                topic_object=validated["topic_object"], state_dimension=validated["state_dimension"],
                slot_question=validated["slot_question"], value=validated["value"],
                assertion_type=validated["assertion_type"], normalization_confidence=validated["normalization_confidence"],
                source_turn_ids=source_turn_ids, state_text=text,
            ))
    _log(f"Phase 2 done: normalized valid={len(slots)} rejected={len(rejected)} new_llm_calls={new_calls}")
    if rejected:
        NORMALIZATION_REJECTED.parent.mkdir(parents=True, exist_ok=True)
        with NORMALIZATION_REJECTED.open("w") as f:
            for row in rejected:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return slots, rejected


# --------------------------------------------------------------------------
# Deterministic network-level dev/eval split
# --------------------------------------------------------------------------

def split_networks(network_ids: list[str], seed: int = SPLIT_SEED, dev_fraction: float = DEV_FRACTION) -> tuple[set[str], set[str]]:
    ordered = sorted(set(network_ids))
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    cutoff = round(len(shuffled) * dev_fraction)
    dev = set(shuffled[:cutoff])
    eval_ = set(shuffled[cutoff:])
    return dev, eval_


# --------------------------------------------------------------------------
# PHASE 3: retrieval ablation (oracle-extraction linking)
# --------------------------------------------------------------------------

def _structured_text(slot: NormalizedSlot) -> str:
    parts = [slot.state_holder, slot.viewpoint_owner or "", slot.topic_object, slot.state_dimension, slot.slot_question]
    return " | ".join(p for p in parts if p)


def _variant_text(variant: str, slot: NormalizedSlot) -> str:
    if variant == "A_full_claim":
        return slot.state_text
    if variant == "B_slot_question":
        return slot.slot_question
    return _structured_text(slot)  # C and D share base text; D adds soft context on top


def run_retrieval_phase(
    old_slots_by_network: dict[str, list[NormalizedSlot]], new_slots: list[NormalizedSlot],
    session_vec_by_key: dict[tuple[str, str], np.ndarray], turn_to_session: dict[str, tuple[str, str]],
    split_label_by_network: dict[str, str],
) -> tuple[dict[str, dict], list[dict]]:
    embed_cache: dict[tuple[str, str], np.ndarray] = {}

    def embed_for(variant: str, items: list[NormalizedSlot]) -> np.ndarray:
        texts = [_variant_text(variant, item) for item in items]
        key = (variant, hashlib.sha256("␟".join(texts).encode()).hexdigest())
        if key not in embed_cache:
            embed_cache[key] = np.asarray(embed_batch(texts), dtype=np.float32) if texts else np.zeros((0, 1))
        return embed_cache[key]

    results: dict[str, dict] = {}
    rows_log: list[dict] = []
    hard_filter_loss = 0

    for variant in RETRIEVAL_VARIANTS:
        per_query = []
        for query in new_slots:
            # The candidate pool is EVERY old-state slot in the network,
            # INCLUDING the query's own gold OLD-state counterpart -- that is
            # exactly the correct target retrieval must find. (A prior bug
            # here filtered it out, silently forcing recall=0 for every
            # variant regardless of representation quality -- see
            # tests.test_semantic_slot_linking_v2.CandidatePoolIncludesGoldTargetTest.)
            pool = list(old_slots_by_network.get(query.network_id, []))
            gold_id = f"{query.case_qa_id}:old"
            if not pool:
                per_query.append({"query": query, "rank": None, "empty_pool": True, "top10": [], "pool_size": 0})
                continue
            query_vec = embed_for(variant, [query])[0]
            cand_vecs = embed_for(variant, pool)
            q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
            m = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12)
            scores = m @ q
            if variant == "D_structured_plus_context":
                session_key = turn_to_session.get(query.source_turn_ids[0]) if query.source_turn_ids else None
                q_sess = session_vec_by_key.get(session_key) if session_key else None
                if q_sess is not None:
                    sess_scores = np.array([
                        (session_vec_by_key.get(turn_to_session.get(c.source_turn_ids[0]), np.zeros_like(q_sess)) @ q_sess)
                        if c.source_turn_ids and turn_to_session.get(c.source_turn_ids[0]) else 0.0
                        for c in pool
                    ])
                    scores = (1 - SESSION_SOFT_WEIGHT) * scores + SESSION_SOFT_WEIGHT * sess_scores
            order = np.argsort(-scores)
            ranked = [f"{pool[i].case_qa_id}:{pool[i].side}" for i in order]
            rank = ranked.index(gold_id) + 1 if gold_id in ranked else None
            per_query.append({"query": query, "rank": rank, "empty_pool": False, "top10": ranked[:10], "pool_size": len(pool)})

            if variant == RETRIEVAL_VARIANTS[0]:
                gold = next((s for s in pool if f"{s.case_qa_id}:{s.side}" == gold_id), None)
                if gold and _norm(gold.state_holder) != _norm(query.state_holder):
                    hard_filter_loss += 1

        n = len(per_query)
        dev_n = sum(1 for r in per_query if split_label_by_network.get(r["query"].network_id) == "dev")
        eval_rows = [r for r in per_query if split_label_by_network.get(r["query"].network_id) == "eval"]
        dev_rows = [r for r in per_query if split_label_by_network.get(r["query"].network_id) == "dev"]

        def _metrics(rows: list[dict]) -> dict:
            m = len(rows)
            if m == 0:
                return {"n": 0, "recall@1": float("nan"), "recall@3": float("nan"), "recall@5": float("nan"),
                        "recall@10": float("nan"), "mrr": float("nan"), "survival": float("nan")}
            return {
                "n": m,
                "recall@1": sum(1 for r in rows if r["rank"] == 1) / m,
                "recall@3": sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 3) / m,
                "recall@5": sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 5) / m,
                "recall@10": sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 10) / m,
                "mrr": sum((1 / r["rank"]) if r["rank"] else 0.0 for r in rows) / m,
                "survival": sum(1 for r in rows if not r["empty_pool"]) / m,
            }

        results[variant] = {"overall": _metrics(per_query), "dev": _metrics(dev_rows), "eval": _metrics(eval_rows)}
        for r in per_query:
            rows_log.append({
                "variant": variant, "qa_id": r["query"].case_qa_id, "network_id": r["query"].network_id,
                "split": split_label_by_network.get(r["query"].network_id), "rank": r["rank"],
                "empty_pool": r["empty_pool"], "top10": r["top10"], "pool_size": r["pool_size"],
            })
    _log(f"Phase 3: hard-filter-by-state_holder would have lost {hard_filter_loss} gold candidates (diagnostic only, not applied)")
    for variant in results:
        results[variant]["hard_filter_state_holder_loss_diagnostic"] = hard_filter_loss
    return results, rows_log


# --------------------------------------------------------------------------
# PHASE 4: automatic negative construction + false-merge measurement
# --------------------------------------------------------------------------

def build_tier1_negatives(new_slots: list[NormalizedSlot], old_slots_by_network: dict[str, list[NormalizedSlot]]) -> list[dict]:
    negatives = []
    for query in new_slots:
        for candidate in old_slots_by_network.get(query.network_id, []):
            if candidate.case_qa_id == query.case_qa_id:
                continue
            holder_differs = query.state_holder and candidate.state_holder and _norm(query.state_holder) != _norm(candidate.state_holder)
            viewpoint_differs = (
                query.viewpoint_owner and candidate.viewpoint_owner and _norm(query.viewpoint_owner) != _norm(candidate.viewpoint_owner)
            )
            if holder_differs or viewpoint_differs:
                negatives.append({"tier": 1, "query_id": f"{query.case_qa_id}:new", "candidate_id": f"{candidate.case_qa_id}:{candidate.side}",
                                   "network_id": query.network_id})
    return negatives


def _tier2_verifier_prompt(query: NormalizedSlot, candidate: NormalizedSlot) -> str:
    return f"""Two state descriptions from the same social network. Do they describe the
SAME tracked state slot (the same evolving thing, e.g. the same person's same
opinion/health/plan) even if phrased differently, or DIFFERENT slots entirely?

A: state_holder={query.state_holder!r} topic_object={query.topic_object!r} "{query.state_text}"
B: state_holder={candidate.state_holder!r} topic_object={candidate.topic_object!r} "{candidate.state_text}"

Return JSON only: {{"verdict":"SAME_SLOT|DIFFERENT_SLOT|AMBIGUOUS","rationale":"one short sentence"}}"""


def build_tier2_negatives(
    new_slots: list[NormalizedSlot], old_slots_by_network: dict[str, list[NormalizedSlot]],
) -> tuple[list[dict], int]:
    cache = _load_cache_dict(TIER2_VERIFIER_CACHE)
    negatives = []
    new_calls = 0
    for query in new_slots:
        pool = old_slots_by_network.get(query.network_id, [])
        if not pool:
            continue
        q_vec = np.asarray(embed_batch([query.slot_question]), dtype=np.float32)[0]
        c_vecs = np.asarray(embed_batch([c.slot_question for c in pool]), dtype=np.float32)
        q = q_vec / (np.linalg.norm(q_vec) + 1e-12)
        m = c_vecs / (np.linalg.norm(c_vecs, axis=1, keepdims=True) + 1e-12)
        sims = m @ q
        for candidate, sim in zip(pool, sims):
            if candidate.case_qa_id == query.case_qa_id or sim < TIER2_SIMILARITY_THRESHOLD:
                continue
            prompt = _tier2_verifier_prompt(query, candidate)
            key = "tier2:" + hashlib.sha256(prompt.encode()).hexdigest()
            output = cache.get(key)
            if output is None:
                output = _call_llm(prompt, VERIFY_MODEL)
                cache[key] = output
                _append_cache(TIER2_VERIFIER_CACHE, key, output)
                new_calls += 1
            verdict = str(output.get("verdict") or "AMBIGUOUS").strip().upper()
            negatives.append({
                "tier": 2, "query_id": f"{query.case_qa_id}:new", "candidate_id": f"{candidate.case_qa_id}:{candidate.side}",
                "network_id": query.network_id, "similarity": float(sim), "verdict": verdict,
            })
    return negatives, new_calls


def _safe_metric(x: float) -> float:
    return x if x == x else -1.0


def select_best_variant(retrieval_results: dict[str, dict], split: str = "eval") -> str:
    """Primary: recall@5 (the registered gate metric). Ties -- all four
    variants tied at 0.949 recall@5 on the first real run -- are broken by
    recall@1, then MRR, NOT by array/iteration order. A prior version had no
    tie-break, so Python's stable max() silently picked "A_full_claim"
    (first in RETRIEVAL_VARIANTS) even when "D_structured_plus_context" had
    strictly better recall@1/MRR -- exactly the kind of ranking-metric
    error this recompute was asked to fix."""
    def _key(variant: str) -> tuple[float, float, float]:
        m = retrieval_results[variant][split]
        return (_safe_metric(m["recall@5"]), _safe_metric(m["recall@1"]), _safe_metric(m["mrr"]))

    return max(retrieval_results, key=_key)


def measure_false_merge(
    negatives: list[dict], retrieval_rows: list[dict], split_label_by_network: dict[str, str],
) -> dict[str, dict]:
    """Corrected version: split by eval/dev (previously combined, which hid
    which split drove the number), and pool-size-aware -- when a query's
    candidate pool has <=5 members, "top-5" is not a real filter (it is the
    whole pool by construction), so a negative appearing there is close to
    guaranteed regardless of representation quality. Reports raw fm@5
    alongside a pool_size>5-only fm@5 and a chance baseline (expected fm@5
    under a UNIFORM RANDOM ranking of the same pools) so the reader can see
    whether the observed rate beats chance, not just its raw value."""
    confirmed = [n for n in negatives if n["tier"] == 1 or (n["tier"] == 2 and n["verdict"] == "DIFFERENT_SLOT")]
    ambiguous_count = sum(1 for n in negatives if n["tier"] == 2 and n["verdict"] == "AMBIGUOUS")
    results = {}
    for variant in RETRIEVAL_VARIANTS:
        rows_by_query = {r["qa_id"]: r for r in retrieval_rows if r["variant"] == variant}

        def _rate_for(split: Optional[str], pool_size_gt: Optional[int] = None):
            rate1_hits = rate5_hits = evaluated = 0
            chance_at_5_sum = 0.0
            for neg in confirmed:
                qa_id = neg["query_id"].split(":")[0]
                row = rows_by_query.get(qa_id)
                if row is None:
                    continue
                if split is not None and split_label_by_network.get(row["network_id"]) != split:
                    continue
                pool_size = row["pool_size"]
                if pool_size_gt is not None and not (pool_size > pool_size_gt):
                    continue
                evaluated += 1
                top10 = row["top10"]
                if top10 and top10[0] == neg["candidate_id"]:
                    rate1_hits += 1
                if neg["candidate_id"] in top10[:5]:
                    rate5_hits += 1
                chance_at_5_sum += min(5, pool_size) / pool_size if pool_size else 0.0
            return {
                "false_merge_rate@1": rate1_hits / evaluated if evaluated else float("nan"),
                "false_merge_rate@5": rate5_hits / evaluated if evaluated else float("nan"),
                "chance_baseline_fm@5": chance_at_5_sum / evaluated if evaluated else float("nan"),
                "n_evaluated": evaluated,
            }

        results[variant] = {
            "combined": _rate_for(None), "eval": _rate_for("eval"), "dev": _rate_for("dev"),
            "combined_pool_gt5_only": _rate_for(None, pool_size_gt=5), "eval_pool_gt5_only": _rate_for("eval", pool_size_gt=5),
            "n_tier1": sum(1 for n in confirmed if n["tier"] == 1),
            "n_tier2_confirmed": sum(1 for n in confirmed if n["tier"] == 2),
            "n_tier2_ambiguous_excluded": ambiguous_count,
        }
    return results


# --------------------------------------------------------------------------
# PHASE 5: end-to-end extraction coverage (separate table, not multiplied)
# --------------------------------------------------------------------------

def measure_extraction_coverage(cases: list[Q8LinkingCase], frozen_session_assertions: Path) -> list[dict]:
    cited_turns_by_network: dict[str, set[str]] = defaultdict(set)
    subject_by_turn: dict[tuple[str, str], set[str]] = defaultdict(set)
    with frozen_session_assertions.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for item in row.get("raw_items", []):
                for turn_id in item.get("source_turn_ids", []):
                    cited_turns_by_network[row["network_id"]].add(turn_id)
                    subject_by_turn[(row["network_id"], turn_id)].add(_norm(str(item.get("subject") or "")))

    rows = []
    for case in cases:
        cited = cited_turns_by_network.get(case.network_id, set())
        old_covered = bool(set(case.old_source_turn_ids) & cited)
        new_covered = bool(set(case.new_source_turn_ids) & cited)
        old_subject_hit = any(
            _norm(case.state_holder) in subject_by_turn.get((case.network_id, t), set()) for t in case.old_source_turn_ids
        )
        rows.append({
            "qa_id": case.qa_id, "network_id": case.network_id,
            "old_covered": old_covered, "new_covered": new_covered, "both_covered": old_covered and new_covered,
            "old_subject_match": old_subject_hit,
        })
    return rows


# --------------------------------------------------------------------------
# PHASE 6: resolver (only if gates pass)
# --------------------------------------------------------------------------

def _resolver_prompt(query: NormalizedSlot, candidates: list[NormalizedSlot]) -> str:
    lines = []
    for letter, cand in zip("ABCDE", candidates):
        lines.append(
            f"[{letter}] cell_id={cand.case_qa_id}:{cand.side} state_holder={cand.state_holder!r} "
            f"topic_object={cand.topic_object!r} state_dimension={cand.state_dimension!r} value={cand.value!r}\n"
            f"    \"{cand.state_text}\""
        )
    block = "\n".join(lines) if lines else "(none)"
    return f"""Decide how this NEW state relates to existing candidate cells (same network).

NEW: state_holder={query.state_holder!r} topic_object={query.topic_object!r} "{query.state_text}"

Candidates:
{block}

Choose exactly one: NEW_CELL | OBSERVE | AUGMENT | REVISE.
If uncertain, choose NEW_CELL.
Return JSON only: {{"selected_cell_id":"id-or-null","operation":"NEW_CELL|OBSERVE|AUGMENT|REVISE",
"confidence":0.0,"rationale":"one short sentence"}}"""


RESOLVER_CONFIDENCE_FLOOR = 0.5


def _normalize_resolver_selection(raw_selected: Any, candidates: list[NormalizedSlot]) -> Optional[str]:
    """The resolver LLM sometimes returns the JSON STRING "null" instead of
    an actual null, and sometimes drops the ":old" suffix from an otherwise
    correct id (e.g. "Q8_n2d0e1f2" instead of "Q8_n2d0e1f2:old"). Both are
    real, observed failure modes (see run.log from the first resolver run --
    every "null"-string NEW_CELL decision was wrongly counted as a false
    merge because the raw string is truthy/not-None in Python). This
    normalizes both without ever inventing a candidate that wasn't shown."""
    if raw_selected in (None, "null", "None", "", "NULL"):
        return None
    raw_selected = str(raw_selected).strip()
    valid_ids = {f"{c.case_qa_id}:{c.side}" for c in candidates}
    if raw_selected in valid_ids:
        return raw_selected
    bare_matches = [f"{c.case_qa_id}:{c.side}" for c in candidates if c.case_qa_id == raw_selected]
    return bare_matches[0] if len(bare_matches) == 1 else None


def run_resolver_phase(
    new_slots: list[NormalizedSlot], old_slots_by_network: dict[str, list[NormalizedSlot]],
    retrieval_rows: list[dict], best_variant: str, gold_relation_by_qa_id: dict[str, str],
) -> list[dict]:
    cache = _load_cache_dict(RESOLVER_CACHE)
    rows_by_query = {r["qa_id"]: r for r in retrieval_rows if r["variant"] == best_variant}
    results = []
    for query in new_slots:
        row = rows_by_query.get(query.case_qa_id)
        if row is None or row["empty_pool"]:
            continue
        top5_ids = row["top10"][:5]
        pool = old_slots_by_network.get(query.network_id, [])
        by_id = {f"{s.case_qa_id}:{s.side}": s for s in pool}
        candidates = [by_id[i] for i in top5_ids if i in by_id]
        prompt = _resolver_prompt(query, candidates)
        key = "resolve:" + hashlib.sha256(prompt.encode()).hexdigest()
        output = cache.get(key)
        if output is None:
            output = _call_llm(prompt, "fast")
            cache[key] = output
            _append_cache(RESOLVER_CACHE, key, output)

        operation = str(output.get("operation") or "NEW_CELL").strip().upper()
        try:
            confidence = float(output.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        selected_cell_id = _normalize_resolver_selection(output.get("selected_cell_id"), candidates)
        defaulted = confidence < RESOLVER_CONFIDENCE_FLOOR
        if defaulted or operation not in ("NEW_CELL", "OBSERVE", "AUGMENT", "REVISE"):
            operation, selected_cell_id = "NEW_CELL", None
            defaulted = True

        gold_id = f"{query.case_qa_id}:old"
        correct_selection = selected_cell_id == gold_id
        gold_relation = gold_relation_by_qa_id.get(query.case_qa_id)
        # false_split: resolver said NEW_CELL (or otherwise picked the wrong
        # cell) when the gold candidate WAS actually retrievable in its pool.
        false_split = (not correct_selection) and (gold_id in top5_ids)
        # false_merge: resolver linked to a specific WRONG existing cell
        # (not null, not gold) rather than creating new / picking correctly.
        false_merge = (selected_cell_id is not None) and (not correct_selection)
        results.append({
            "qa_id": query.case_qa_id, "network_id": query.network_id, "gold_relation": gold_relation,
            "operation": operation, "selected_cell_id": selected_cell_id, "confidence": confidence,
            "defaulted": defaulted, "correct_selection": correct_selection,
            "revise_recall_hit": correct_selection and operation == "REVISE" and gold_relation == "REVISE",
            "false_split": false_split, "false_merge": false_merge,
        })
    return results


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main(force_resolver: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log("=== semantic_slot_linking_v2 run starting ===")
    _log("Verifying frozen snapshot integrity...")
    manifest = verify_frozen_snapshot()
    _log(f"  OK: {len(manifest['files'])} files match SHA256")

    qa, conversations, networks = load_frozen_data()
    tier_by_network = dict(zip(networks.network_id, networks.tier))

    cases, ambiguous, dataset_meta = construct_q8_dataset(qa, conversations)
    Q8_CASES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with Q8_CASES_JSONL.open("w") as f:
        for c in cases:
            f.write(json.dumps({**c.__dict__}, ensure_ascii=False, default=list) + "\n")
    with Q8_AMBIGUOUS_JSONL.open("w") as f:
        for row in ambiguous:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    all_networks = sorted({c.network_id for c in cases})
    dev_networks, eval_networks = split_networks(all_networks)
    dataset_meta["dev_networks"] = sorted(dev_networks)
    dataset_meta["eval_networks"] = sorted(eval_networks)
    dataset_meta["frozen_snapshot_checksum_ok"] = True
    Q8_META_JSON.write_text(json.dumps(dataset_meta, indent=2, ensure_ascii=False))

    slots, rejected = normalize_cases(cases, conversations)
    old_slots_by_network: dict[str, list[NormalizedSlot]] = defaultdict(list)
    new_slots: list[NormalizedSlot] = []
    for s in slots:
        if s.side == "old":
            old_slots_by_network[s.network_id].append(s)
        else:
            new_slots.append(s)

    conv_scope = conversations[conversations.network_id.isin(all_networks)]
    turn_to_session = {str(r.turn_id): (str(r.network_id), str(r.session_id)) for r in conv_scope.itertuples(index=False)}
    session_text: dict[tuple[str, str], str] = defaultdict(str)
    for r in conv_scope.itertuples(index=False):
        session_text[(str(r.network_id), str(r.session_id))] += f" {r.message}"
    session_keys = list(session_text.keys())
    session_vecs = np.asarray(embed_batch([session_text[k][:4000] for k in session_keys]), dtype=np.float32) if session_keys else np.zeros((0, 1))
    session_vec_by_key = {k: v / (np.linalg.norm(v) + 1e-12) for k, v in zip(session_keys, session_vecs)}

    split_label_by_network = {n: "dev" for n in dev_networks} | {n: "eval" for n in eval_networks}
    retrieval_results, retrieval_rows = run_retrieval_phase(
        old_slots_by_network, new_slots, session_vec_by_key, turn_to_session, split_label_by_network,
    )
    with RETRIEVAL_RESULTS.open("w") as f:
        for row in retrieval_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=list) + "\n")

    tier1 = build_tier1_negatives(new_slots, old_slots_by_network)
    tier2, tier2_calls = build_tier2_negatives(new_slots, old_slots_by_network)
    negatives = tier1 + tier2
    with NEGATIVE_PAIRS.open("w") as f:
        for row in negatives:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    false_merge = measure_false_merge(negatives, retrieval_rows, split_label_by_network)

    coverage_rows = measure_extraction_coverage(cases, FROZEN_DIR / "run" / "session_assertions.jsonl")
    with EXTRACTION_COVERAGE.open("w") as f:
        for row in coverage_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    eval_usable = sum(1 for s in new_slots if split_label_by_network.get(s.network_id) == "eval")
    best_variant = select_best_variant(retrieval_results)
    gate = {
        "usable_pairs_eval": eval_usable, "gate_min_usable_pairs": GATE_MIN_USABLE_PAIRS,
        "recall_at_5_eval": retrieval_results[best_variant]["eval"]["recall@5"], "gate_recall_at_5": GATE_RECALL_AT_5,
        "recall_at_1_eval": retrieval_results[best_variant]["eval"]["recall@1"],
        "mrr_eval": retrieval_results[best_variant]["eval"]["mrr"],
        "survival_eval": retrieval_results[best_variant]["eval"]["survival"], "gate_survival": GATE_CANDIDATE_SURVIVAL,
        "false_merge_at_5": false_merge[best_variant]["eval"]["false_merge_rate@5"], "gate_false_merge_at_5": GATE_FALSE_MERGE_AT_5,
        "false_merge_at_5_pool_gt5_only": false_merge[best_variant]["eval_pool_gt5_only"]["false_merge_rate@5"],
        "false_merge_chance_baseline_at_5": false_merge[best_variant]["eval"]["chance_baseline_fm@5"],
        "best_variant": best_variant,
    }
    passed = (
        gate["usable_pairs_eval"] >= GATE_MIN_USABLE_PAIRS
        and gate["recall_at_5_eval"] == gate["recall_at_5_eval"] and gate["recall_at_5_eval"] >= GATE_RECALL_AT_5
        and gate["survival_eval"] == gate["survival_eval"] and gate["survival_eval"] >= GATE_CANDIDATE_SURVIVAL
        and gate["false_merge_at_5"] == gate["false_merge_at_5"] and gate["false_merge_at_5"] <= GATE_FALSE_MERGE_AT_5
    )
    gate["passed"] = passed
    gate["resolver_forced"] = force_resolver and not passed
    _log(f"Gate check: {json.dumps(gate, indent=2)}")

    resolver_results = []
    if passed or force_resolver:
        if passed:
            _log("Gates PASSED -- running bounded resolver on eval split")
        else:
            _log("Gates FAILED (false_merge@5 gate) but resolver run was EXPLICITLY authorized by the user for this "
                 "step -- running on the 59 eval cases anyway. This is a manual override, not an automatic gate pass.")
        eval_new_slots = [s for s in new_slots if split_label_by_network.get(s.network_id) == "eval"]
        gold_relation_by_qa_id = {c.qa_id: c.relation for c in cases}
        resolver_results = run_resolver_phase(eval_new_slots, old_slots_by_network, retrieval_rows, best_variant, gold_relation_by_qa_id)
        with RESOLVER_RESULTS.open("w") as f:
            for row in resolver_results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        _log("Gates FAILED -- resolver and materialization skipped")

    config = {
        "schema_version": SCHEMA_VERSION, "construction_model": CONSTRUCTION_MODEL, "normalize_model": NORMALIZE_MODEL,
        "verify_model": VERIFY_MODEL, "temperature": TEMPERATURE, "split_seed": SPLIT_SEED, "dev_fraction": DEV_FRACTION,
        "dev_networks": sorted(dev_networks), "eval_networks": sorted(eval_networks),
        "gates": {"min_usable_pairs": GATE_MIN_USABLE_PAIRS, "recall_at_5": GATE_RECALL_AT_5,
                  "candidate_survival": GATE_CANDIDATE_SURVIVAL, "false_merge_at_5": GATE_FALSE_MERGE_AT_5},
        "frozen_snapshot": str(FROZEN_DIR), "frozen_snapshot_file_count": len(manifest["files"]),
        "tier2_similarity_threshold": TIER2_SIMILARITY_THRESHOLD,
        "gate_result": gate, "resolver_override_requested": force_resolver,
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2, ensure_ascii=False))

    report = render_report(
        dataset_meta, retrieval_results, false_merge, coverage_rows, gate, resolver_results, cases, tier_by_network,
    )
    REPORT_MD.write_text(report)
    _log("=== run complete ===")
    print(report)


def render_report(
    dataset_meta: dict, retrieval_results: dict, false_merge: dict, coverage_rows: list[dict],
    gate: dict, resolver_results: list[dict], cases: list[Q8LinkingCase], tier_by_network: dict,
) -> str:
    lines = [
        "# Semantic slot linking v2 -- diagnostic report",
        "",
        "Post-hoc diagnostic/development analysis over a frozen snapshot -- NOT a held-out leaderboard result, "
        "NOT mixed with the completed full-benchmark numbers.",
        "",
        f"- Q8 total: {dataset_meta['total_q8']}; usable: {dataset_meta['usable']}; "
        f"ambiguous: {dataset_meta['ambiguous']}; missing_evidence: {dataset_meta['missing_evidence']}; "
        f"malformed: {dataset_meta['malformed']}",
        f"- relation distribution: {dataset_meta['relation_distribution']}",
        f"- dev networks: {len(dataset_meta['dev_networks'])}; eval networks: {len(dataset_meta['eval_networks'])}",
        "",
        "## Retrieval ablation (eval split)", "",
        "| variant | n | recall@1 | recall@3 | recall@5 | recall@10 | MRR | survival |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in RETRIEVAL_VARIANTS:
        e = retrieval_results[v]["eval"]
        lines.append(f"| {v} | {e['n']} | {e['recall@1']:.3f} | {e['recall@3']:.3f} | {e['recall@5']:.3f} | "
                      f"{e['recall@10']:.3f} | {e['mrr']:.3f} | {e['survival']:.3f} |")
    lines += ["", "## Retrieval ablation (dev split, diagnostic only)", "",
              "| variant | n | recall@1 | recall@3 | recall@5 | recall@10 | MRR |", "|---|---|---|---|---|---|---|"]
    for v in RETRIEVAL_VARIANTS:
        d = retrieval_results[v]["dev"]
        lines.append(f"| {v} | {d['n']} | {d['recall@1']:.3f} | {d['recall@3']:.3f} | {d['recall@5']:.3f} | {d['recall@10']:.3f} | {d['mrr']:.3f} |")
    lines += [
        "", "## False-merge metrics -- EVAL split only (corrected: previously combined eval+dev, hiding which "
        "split drove the number)", "",
        "| variant | fm@1 | fm@5 | chance baseline fm@5 | n_evaluated |",
        "|---|---|---|---|---|",
    ]
    for v in RETRIEVAL_VARIANTS:
        fm = false_merge[v]["eval"]
        lines.append(f"| {v} | {fm['false_merge_rate@1']:.3f} | {fm['false_merge_rate@5']:.3f} | "
                      f"{fm['chance_baseline_fm@5']:.3f} | {fm['n_evaluated']} |")
    lines += [
        "", "### Pool-size-aware fm@5 (eval, restricted to queries whose candidate pool has >5 members -- where "
        "\"top-5\" is an actual cut, not the whole pool)", "",
        "| variant | fm@5 (pool>5 only) | n_evaluated |", "|---|---|---|",
    ]
    for v in RETRIEVAL_VARIANTS:
        fm = false_merge[v]["eval_pool_gt5_only"]
        lines.append(f"| {v} | {fm['false_merge_rate@5']:.3f} | {fm['n_evaluated']} |")
    lines += [
        "", "### Combined eval+dev (diagnostic only, larger n)", "",
        "| variant | fm@1 | fm@5 | chance baseline fm@5 | n_evaluated | tier1 | tier2_confirmed | tier2_ambiguous_excluded |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in RETRIEVAL_VARIANTS:
        c = false_merge[v]["combined"]
        lines.append(f"| {v} | {c['false_merge_rate@1']:.3f} | {c['false_merge_rate@5']:.3f} | "
                      f"{c['chance_baseline_fm@5']:.3f} | {c['n_evaluated']} | {false_merge[v]['n_tier1']} | "
                      f"{false_merge[v]['n_tier2_confirmed']} | {false_merge[v]['n_tier2_ambiguous_excluded']} |")

    old_cov = sum(1 for r in coverage_rows if r["old_covered"]) / len(coverage_rows) if coverage_rows else float("nan")
    new_cov = sum(1 for r in coverage_rows if r["new_covered"]) / len(coverage_rows) if coverage_rows else float("nan")
    both_cov = sum(1 for r in coverage_rows if r["both_covered"]) / len(coverage_rows) if coverage_rows else float("nan")
    subj_cov = sum(1 for r in coverage_rows if r["old_subject_match"]) / len(coverage_rows) if coverage_rows else float("nan")
    lines += [
        "", "## End-to-end extraction coverage vs frozen session_assertions.jsonl (SEPARATE from retrieval; not multiplied)", "",
        f"- n Q8 usable gold cases: {len(coverage_rows)}",
        f"- old-side coverage: {old_cov:.3f}",
        f"- new-side coverage: {new_cov:.3f}",
        f"- both-sides coverage: {both_cov:.3f}",
        f"- old-side correct-subject coverage: {subj_cov:.3f}",
        "",
        "## Success gates", "",
        f"- usable pairs (eval) >= {GATE_MIN_USABLE_PAIRS}: {gate['usable_pairs_eval']} -> "
        f"{'PASS' if gate['usable_pairs_eval'] >= GATE_MIN_USABLE_PAIRS else 'FAIL'}",
        f"- recall@5 (eval, best={gate['best_variant']}) >= {GATE_RECALL_AT_5}: {gate['recall_at_5_eval']:.3f} -> "
        f"{'PASS' if gate['recall_at_5_eval'] >= GATE_RECALL_AT_5 else 'FAIL'}",
        f"- candidate survival >= {GATE_CANDIDATE_SURVIVAL}: {gate['survival_eval']:.3f} -> "
        f"{'PASS' if gate['survival_eval'] >= GATE_CANDIDATE_SURVIVAL else 'FAIL'}",
        f"- false_merge@5 <= {GATE_FALSE_MERGE_AT_5}: {gate['false_merge_at_5']:.3f} -> "
        f"{'PASS' if gate['false_merge_at_5'] <= GATE_FALSE_MERGE_AT_5 else 'FAIL'}",
        f"- OVERALL: {'PASS -- resolver ran' if gate['passed'] else ('FAIL -- resolver run anyway (explicit user override, not an automatic gate pass)' if gate.get('resolver_forced') else 'FAIL -- resolver/materialization skipped')}",
    ]
    if resolver_results:
        n = len(resolver_results)
        correct = sum(1 for r in resolver_results if r["correct_selection"])
        n_revise_gold = sum(1 for r in resolver_results if r["gold_relation"] == "REVISE")
        revise_hits = sum(1 for r in resolver_results if r["revise_recall_hit"])
        false_split_n = sum(1 for r in resolver_results if r["false_split"])
        false_merge_n = sum(1 for r in resolver_results if r["false_merge"])
        defaulted_n = sum(1 for r in resolver_results if r["defaulted"])
        lines += [
            "", "## Resolver results (bounded top-5, eval split -- see run authorization note above the gate table)", "",
            f"- n resolved: {n}",
            f"- correct old-cell selection: {correct}/{n} ({correct/n:.3f})",
            f"- REVISE recall (gold=REVISE, resolver correctly selected + operation=REVISE): {revise_hits}/{n_revise_gold} "
            f"({revise_hits/n_revise_gold:.3f})" if n_revise_gold else "- REVISE recall: n/a (no REVISE gold cases)",
            f"- chain recovery (== correct old-cell selection here, since all gold cases are REVISE): {correct}/{n} ({correct/n:.3f})",
            f"- false split (gold candidate WAS retrievable but resolver picked NEW_CELL/wrong): {false_split_n}/{n} ({false_split_n/n:.3f})",
            f"- false merge (resolver linked to a specific wrong existing cell): {false_merge_n}/{n} ({false_merge_n/n:.3f})",
            f"- defaulted to NEW_CELL (low confidence or malformed output): {defaulted_n}/{n} ({defaulted_n/n:.3f})",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolver-override", action="store_true",
        help="Run the bounded top-5 resolver on the eval split even if the false-merge gate did not pass. "
             "Explicit user authorization required for this -- does not affect the gate computation or report, "
             "only whether the resolver phase actually executes.",
    )
    args = parser.parse_args()
    main(force_resolver=args.resolver_override)
