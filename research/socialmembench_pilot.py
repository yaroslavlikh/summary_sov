"""SocialMemBench pilot for heterogeneous memory retrieval.

This is deliberately isolated from production.  It downloads the pinned public
parquet release, builds one evidence-oracle memory store per social network,
retrieves raw turns and derived units, reranks them through the same interface,
expands derived provenance back to raw turns, and reports stage-level metrics.

The public parquet release does not contain the pre-QA planted-challenge records
described in the paper.  ``evidence_oracle`` therefore uses QA evidence-anchor
``relevance`` annotations (never the gold answer) and is an explicit diagnostic
upper bound, not an end-to-end ingestion result.

Cheap retrieval smoke test (no remote LLM calls):

    python3 -m research.socialmembench_pilot --mode retrieval --per-type 2

Paired 120-question pilot (remote rerank/answer/judge calls):

    python3 -m research.socialmembench_pilot --mode full --per-type 20 --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace as _dc_replace

from embeddings import embed, embed_batch
from research.versioned_memory_cells import (
    MODALITIES, SCOPE_TYPES, Assertion, CellKey, MemoryCell, MemoryObservation, MemoryStateVersion,
    apply_assertion, resolve_topic_slot, validate_assertion,
)


def _norm(value: str) -> str:
    return (value or "").strip().casefold()


REVISION = "ea7e4acf502df3eda484d56165ec594d6f4c138f"
BASE_URL = f"https://huggingface.co/datasets/anon4data/socialmembench/resolve/{REVISION}"
FILES = {
    "conversations": "conversations.parquet",
    "qa": "qa.parquet",
    "personas": "personas.parquet",
}
PILOT_TYPES = ("Q1", "Q2", "Q4", "Q5", "Q6", "Q8")
TYPE_NAMES = {
    "Q1": "single_contact",
    "Q2": "group_decision",
    "Q3": "multi_contact",
    "Q4": "attribution",
    "Q5": "theory_of_mind",
    "Q6": "norm_vs_individual",
    "Q7": "relationship",
    "Q8": "temporal_shift",
    "Q9": "departed_member",
}
VARIANTS = ("raw", "late_memory", "unified", "provenance")
EXTRACT_WORKERS = 5


@dataclass
class Candidate:
    candidate_id: str
    kind: str
    text: str
    source_ids: tuple[str, ...]
    asserted_by: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    observed_at: str = ""
    channels: set[str] = field(default_factory=set)
    score: float = 0.0
    role: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "text": self.text,
            "source_ids": list(self.source_ids),
            "asserted_by": list(self.asserted_by),
            "entities": list(self.entities),
            "observed_at": self.observed_at,
            "channels": sorted(self.channels),
            "score": self.score,
            "role": self.role,
        }


def _json(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def ensure_data(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, filename in FILES.items():
        path = data_dir / filename
        if not path.exists():
            print(f"Downloading {filename}...", file=sys.stderr)
            urllib.request.urlretrieve(f"{BASE_URL}/{filename}", path)
        paths[name] = path
    return paths


def load_data(data_dir: Path):
    paths = ensure_data(data_dir)
    conversations = pd.read_parquet(paths["conversations"])
    qa = pd.read_parquet(paths["qa"])
    personas = pd.read_parquet(paths["personas"])
    return conversations, qa, personas


def _mentions(text: str, names: list[str]) -> tuple[str, ...]:
    lowered = text.casefold()
    return tuple(sorted(name for name in names if name.casefold() in lowered))


def build_oracle_units(
    conversations: pd.DataFrame, qa: pd.DataFrame, personas: pd.DataFrame
) -> dict[str, list[Candidate]]:
    """Build a corpus-wide evidence oracle without putting gold answers in text."""
    turns = {
        (row.network_id, row.turn_id): row
        for row in conversations.itertuples(index=False)
    }
    names = defaultdict(list)
    for row in personas.itertuples(index=False):
        names[row.network_id].append(row.display_name)

    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for row in qa.itertuples(index=False):
        challenge = (row.source_challenge_id or "").strip() or row.qa_id
        bucket = buckets.setdefault(
            (row.network_id, challenge),
            {"types": set(), "sources": set(), "speakers": set(), "notes": []},
        )
        bucket["types"].add(TYPE_NAMES.get(row.query_type, row.query_type))
        for anchor in _json(row.evidence_anchors_json, []):
            turn_id = anchor.get("turn_id")
            if not turn_id or (row.network_id, turn_id) not in turns:
                raise ValueError(f"Unknown evidence anchor {row.network_id}/{turn_id} in {row.qa_id}")
            bucket["sources"].add(turn_id)
            speaker = anchor.get("speaker_display_name") or turns[(row.network_id, turn_id)].speaker_display_name
            if speaker:
                bucket["speakers"].add(speaker)
            note = (anchor.get("relevance") or "").strip()
            if note and note not in bucket["notes"]:
                bucket["notes"].append(note)

    by_network = defaultdict(list)
    for (network_id, challenge), bucket in buckets.items():
        sources = sorted(bucket["sources"])
        if not sources:
            continue
        source_rows = [turns[(network_id, turn_id)] for turn_id in sources]
        notes = bucket["notes"] or [f"{r.speaker_display_name}: {r.message}" for r in source_rows]
        text = " ".join(notes)
        entities = _mentions(text, names[network_id])
        observed_at = max((r.timestamp for r in source_rows), default="")
        by_network[network_id].append(Candidate(
            candidate_id=f"memory:{network_id}:{challenge}",
            kind="derived:" + "+".join(sorted(bucket["types"])),
            text=text,
            source_ids=tuple(sources),
            asserted_by=tuple(sorted(bucket["speakers"])),
            entities=entities,
            observed_at=observed_at,
        ))
    return dict(by_network)


def _validate_extracted_items(
    network_id: str, session_id: str, items: list[Any], source_rows: dict[str, Any]
) -> list[Candidate]:
    candidates = []
    for number, item in enumerate(items[:8]):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        source_ids = tuple(dict.fromkeys(
            str(source_id) for source_id in item.get("source_turn_ids", [])
            if str(source_id) in source_rows
        ))
        if not text or not source_ids:
            continue
        speakers = {source_rows[source_id].speaker_display_name for source_id in source_ids}
        asserted_by = tuple(
            name for name in item.get("asserted_by", [])
            if isinstance(name, str) and name in speakers
        ) or tuple(sorted(speakers))
        entities = tuple(
            name for name in item.get("entities", []) if isinstance(name, str) and name.strip()
        )
        kind = re.sub(r"[^a-z0-9_]+", "_", str(item.get("kind") or "fact").casefold()).strip("_")
        observed_at = max(str(source_rows[source_id].timestamp) for source_id in source_ids)
        candidates.append(Candidate(
            candidate_id=f"memory:{network_id}:{session_id}:{number}",
            kind=f"derived:{kind or 'fact'}",
            text=text,
            source_ids=source_ids,
            asserted_by=asserted_by,
            entities=entities,
            observed_at=observed_at,
        ))
    return candidates


def _extract_session_memory(network_id: str, session_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    from llm.groq_client import get_chat_model

    source_rows = {str(row.turn_id): row for row in frame.itertuples(index=False)}
    lines = [
        f"[[{row.turn_id}]] {row.timestamp} {row.speaker_display_name}: {row.message}"
        for row in frame.itertuples(index=False)
    ]
    prompt = f"""Compress this bounded multi-party chat session into durable memory.
Use ONLY the messages below; you do not know any future question or answer. Extract
0-8 atomic units that could matter weeks later: personal facts or states,
commitments, relationships, attributed opinions, group decisions or norms,
individual exceptions, recurring patterns, and temporal updates. Do not summarize
every utterance. Preserve who asserted what and whether it was a report, joke,
proposal, uncertainty, or settled decision. Never turn somebody's quote or report
into an unattributed objective fact. Each source_turn_id must be copied from the
session. Return JSON only:
{{"items":[{{"text":"durable claim","kind":"personal_fact|state|commitment|relationship|opinion|group_decision|group_norm|exception|running_pattern|temporal_update","source_turn_ids":["turn_id"],"asserted_by":["speaker"],"entities":["person or group"]}}]}}

Session:
{chr(10).join(lines)}"""
    message = get_chat_model("fast", 0).invoke(prompt)
    parsed = _parse_json_object(message.content if isinstance(message.content, str) else "") or {}
    candidates = _validate_extracted_items(
        network_id, session_id, parsed.get("items", []), source_rows
    )
    return {
        "network_id": network_id,
        "session_id": session_id,
        "items": [candidate.as_dict() for candidate in candidates],
    }


def build_extracted_units(conversations: pd.DataFrame, cache_path: Path) -> dict[str, list[Candidate]]:
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cached[(row["network_id"], str(row["session_id"]))] = row

    groups = [
        (str(network_id), str(session_id), frame)
        for (network_id, session_id), frame in conversations.groupby(["network_id", "session_id"], sort=True)
        if (str(network_id), str(session_id)) not in cached
    ]
    if groups:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Extracting memory from {len(groups)} chat sessions...", file=sys.stderr)
        with cache_path.open("a") as output, ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
            futures = {
                pool.submit(_extract_session_memory, network_id, session_id, frame): (network_id, session_id)
                for network_id, session_id, frame in groups
            }
            for done, future in enumerate(as_completed(futures), 1):
                key = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    print(f"  extraction failed for {key}: {error}", file=sys.stderr)
                    continue
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                cached[key] = row
                if done % 20 == 0 or done == len(groups):
                    print(f"  extracted {done}/{len(groups)} sessions", file=sys.stderr)

    by_network = defaultdict(list)
    selected_networks = set(conversations.network_id.astype(str))
    for (network_id, _), row in cached.items():
        if network_id not in selected_networks:
            continue
        for item in row["items"]:
            by_network[network_id].append(Candidate(**{
                **item,
                "source_ids": tuple(item["source_ids"]),
                "asserted_by": tuple(item.get("asserted_by", [])),
                "entities": tuple(item.get("entities", [])),
                "channels": set(item.get("channels", [])),
            }))
    return dict(by_network)


def _cell_extract_prompt(session_lines: str) -> str:
    # A plain f-string built fresh per call (not a module-level .format()
    # template) -- avoids a real bug that hit here first: an f-string with
    # a deferred {session_lines} placeholder, later re-templated via
    # .format(), breaks the moment the prompt ALSO contains literal braces
    # (the JSON schema example) that survive the f-string's own escaping
    # and then get misread as format fields on the second pass.
    return f"""Compress this bounded multi-party chat session into durable, ATTRIBUTABLE
memory assertions. Use ONLY the messages below; you do not know any future question,
answer, or any other session -- you cannot know whether something here updates a
belief from a different session, only whether it changes something ALREADY visible
in this same session.

Extract 0-8 atomic assertions. For each:

- viewpoint_owner: whose view/knowledge/state this is (for a plain self-report this
  is usually the same person as subject; for an opinion about someone else, or
  knowledge ABOUT someone else, viewpoint_owner is the person who holds that view/
  knowledge, not the person it's about).
- subject: the person, group, or decision the assertion is actually about.
- facet: a short normalized aspect, e.g. location, occupation, education,
  availability, attitude, preference, relationship, group_decision, group_norm,
  knows_about, commitment -- pick the closest fit, invent a short new one only if
  truly nothing fits.
- scope_type: one of {sorted(SCOPE_TYPES)}.
- assertion_text: the durable claim, IN RUSSIAN if the chat is in Russian, English
  otherwise. If this is someone's opinion or a report of someone else's words,
  the text MUST preserve that attribution (e.g. "Х says/believes that Y",
  "По словам X, Y") -- never state someone's opinion, guess, or report of a third
  party's words as if it were plain objective fact.
- topic_key: a short, normalized slug identifying the SPECIFIC state/topic this is
  about, e.g. "camping_weekend", "riverside_10k", "autumn_hikes", "group_warmth" --
  NOT just the facet. Two assertions about the same person and facet (e.g. both
  "attitude") but about DIFFERENT specific things (an opinion about a hike vs an
  opinion about breakfast eggs) MUST get different topic_key values -- a broad facet
  label alone is not enough identity, this is what actually distinguishes one
  evolving state from an unrelated one. If in doubt, pick a MORE specific topic_key,
  not a broader one -- a missed link is recoverable later, a false merge is not.
- normalized_value: a short stable label for the state/value if there is one
  (e.g. "negative", "positive", "Moscow"), or null if not applicable.
- modality: one of {sorted(MODALITIES)}.
- confidence: 0.0-1.0, how explicit/certain this assertion is in the source text.
- source_turn_ids: real turn_id values copied from the session below, at least one.
- effective_from / effective_to: ISO date the state applies from/until if the text
  gives one (e.g. a specific trip's dates), else empty string / null. Leave both
  empty for an ongoing, undated state.
- temporal_scope: one of ongoing, dated_event, recurring, unspecified.
- temporal_precision: one of exact, approximate, unknown.
- operation: your best guess among create/reaffirm/revise/retract based ONLY on
  what's visible in THIS session (e.g. someone explicitly correcting or retracting
  something they just said) -- null if you cannot tell from this session alone.
  Do not guess about anything from a different session; you cannot see other
  sessions and should not pretend to.

Do not summarize every utterance -- most of a normal conversation is not durable
memory. A one-off joke or insult is a joke/sarcasm modality assertion (if worth
keeping at all), not a durable attitude, unless the session shows it becoming a
repeated or settled characterization.

Return JSON only:
{{"items":[{{"viewpoint_owner":"...","subject":"...","facet":"...","topic_key":"...",
"scope_type":"...","assertion_text":"...","normalized_value":"...","modality":"...",
"confidence":0.0,"source_turn_ids":["turn_id"],"effective_from":"","effective_to":"",
"temporal_scope":"unspecified","temporal_precision":"unknown","operation":"create"}}]}}

Session:
{session_lines}"""


def _extract_session_assertions(network_id: str, session_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Query-independent: sees only this session's own turns, never QA
    question/answer/evidence_anchors, never another session, never
    production chat data. Forced JSON, temperature=0 -- no free-choice
    tool-calling, matching the earlier finding (research/forced_extractor_eval.py)
    that free-choice tool-calling on identical data produced 0-1 candidates
    across runs."""
    from llm.groq_client import get_chat_model

    lines = [
        f"[[{row.turn_id}]] {row.timestamp} {row.speaker_display_name}: {row.message}"
        for row in frame.itertuples(index=False)
    ]
    prompt = _cell_extract_prompt(chr(10).join(lines))
    message = get_chat_model("fast", 0).invoke(prompt)
    parsed = _parse_json_object(message.content if isinstance(message.content, str) else "") or {}
    return {
        "network_id": network_id, "session_id": session_id,
        "raw_items": [item for item in parsed.get("items", []) if isinstance(item, dict)],
    }


def build_cell_assertion_cache(conversations: pd.DataFrame, cache_path: Path) -> dict[tuple[str, str], dict]:
    """Phase 1: extraction only, cached per (network_id, session_id) --
    deliberately separate from build_extracted_units's flat-memory cache
    file (never overwrite that one, per the design doc's leakage/isolation
    rules). Extraction order across sessions doesn't matter (each session
    is independent), so this stays parallelized like build_extracted_units;
    the CHRONOLOGICAL replay that turns these into cells happens in
    materialize_cell_registries below, strictly sequential."""
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cached[(row["network_id"], str(row["session_id"]))] = row

    groups = [
        (str(network_id), str(session_id), frame)
        for (network_id, session_id), frame in conversations.groupby(["network_id", "session_id"], sort=True)
        if (str(network_id), str(session_id)) not in cached
    ]
    if groups:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Extracting cell assertions from {len(groups)} chat sessions...", file=sys.stderr)
        with cache_path.open("a") as output:
            completed = 0
            for offset in range(0, len(groups), EXTRACT_WORKERS):
                batch = groups[offset:offset + EXTRACT_WORKERS]
                first_error = None
                with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
                    futures = {
                        pool.submit(_extract_session_assertions, network_id, session_id, frame):
                        (network_id, session_id)
                        for network_id, session_id, frame in batch
                    }
                    for future in as_completed(futures):
                        key = futures[future]
                        try:
                            row = future.result()
                        except Exception as error:
                            first_error = first_error or error
                            print(f"  extraction failed for {key}: {error}", file=sys.stderr)
                            continue
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        cached[key] = row
                        completed += 1
                        if completed % 20 == 0 or completed == len(groups):
                            print(f"  extracted {completed}/{len(groups)} sessions", file=sys.stderr)
                if first_error is not None:
                    raise first_error
    return cached


# v0 note (frozen, no longer callable): the first materialize_cell_registries
# keyed cells purely by (network_id, viewpoint_owner, subject, facet, scope_type)
# -- manual review of that materialization (frozen: /tmp/*_v0_naive_facet.*)
# found this identity too coarse: broad facets like "attitude"/"commitment"
# silently merged UNRELATED states of the same person (an opinion about a
# hike "revised" into an opinion about eggs, because both were facet=attitude
# for the same person). Replaced below by a topic_key-scoped identity --
# validate_assertion now REQUIRES topic_key, so the old function is removed
# rather than kept around half-working; the frozen v0 artifacts are the
# historical record, not the old code path.

SLOT_RESOLUTION_WORKERS = 5


def _slot_resolution_prompt(items: list[dict]) -> str:
    rows = [
        f"[{index}] modality={item.get('modality')} confidence={item.get('confidence')} "
        f"observed_at={item.get('observed_at')}\n{item.get('assertion_text')}"
        for index, item in enumerate(items)
    ]
    return f"""These assertions were extracted independently about the SAME person and the
SAME broad facet label, so they might be about one evolving state, or might be
about completely unrelated specific topics that just happen to share that
facet. For EACH assertion (by its [index]), assign:

- topic_key: a short, normalized slug for the SPECIFIC state/topic, e.g.
  "camping_weekend", "riverside_10k", "autumn_hikes", "group_warmth". Give
  TWO assertions the SAME topic_key ONLY if they are genuinely about the same
  specific thing. If in doubt, use DIFFERENT topic_keys -- a missed link is
  recoverable later, a false merge is not (it would silently overwrite one
  real state with an unrelated one).
- temporal_scope: one of ongoing, dated_event, recurring, unspecified.
- temporal_precision: one of exact, approximate, unknown.
- effective_from / effective_to: ISO date if the text gives one, else empty.

Assertions:
{chr(10).join(rows)}

Return JSON only: {{"items":[{{"index":0,"topic_key":"...","temporal_scope":"unspecified",
"temporal_precision":"unknown","effective_from":"","effective_to":""}}]}}"""


def _resolve_slots_for_bucket(items: list[dict]) -> dict[int, dict]:
    from llm.groq_client import get_chat_model

    message = get_chat_model("fast", 0).invoke(_slot_resolution_prompt(items))
    parsed = _parse_json_object(message.content if isinstance(message.content, str) else "") or {}
    by_index = {}
    for entry in parsed.get("items", []):
        index = entry.get("index")
        if isinstance(index, int) and 0 <= index < len(items):
            by_index[index] = entry
    return by_index


def build_slot_resolution_cache(
    cache: dict[tuple[str, str], dict], slot_cache_path: Path,
    selected_networks: Optional[set[str]] = None,
) -> dict[tuple[str, str, int], dict]:
    """Resolves topic_key/temporal fields for every raw_item ALREADY cached
    by build_cell_assertion_cache, WITHOUT redoing any of the 315 per-session
    extraction calls. Groups raw items into coarse (network, owner, subject,
    facet, scope) buckets ACROSS ALL their sessions and runs ONE forced-JSON
    slot-resolution call per multi-item bucket -- singleton buckets (nothing
    to disambiguate from) get a stable content-hash topic_key, no LLM call
    needed. Separate, resumable cache file -- never touches the raw
    extraction cache or its file.

    selected_networks restricts which cache entries get bucketed at all --
    the raw `cache` dict returned by build_cell_assertion_cache can (harmlessly
    for correctness, but wastefully) hold sessions from OTHER networks left
    over from an earlier/broader run; materialize_cell_registries already
    filters its OWN output to the current selection, but without this filter
    here too, slot resolution would burn real LLM calls resolving buckets
    that the current run doesn't even use."""
    resolved: dict[tuple[str, str, int], dict] = {}
    if slot_cache_path.exists():
        for line in slot_cache_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                resolved[(row["network_id"], row["session_id"], row["item_index"])] = row["slot"]

    buckets: dict[tuple, list[tuple[str, str, int, dict]]] = defaultdict(list)
    for (network_id, session_id), row in cache.items():
        if selected_networks is not None and network_id not in selected_networks:
            continue
        for item_index, item in enumerate(row["raw_items"]):
            bucket_key = (
                network_id, _norm(item.get("viewpoint_owner", "")), _norm(item.get("subject", "")),
                _norm(item.get("facet", "")), _norm(item.get("scope_type") or "individual"),
            )
            buckets[bucket_key].append((network_id, session_id, item_index, item))

    pending: list[tuple[tuple, list]] = []
    for bucket_key, entries in buckets.items():
        unresolved = [e for e in entries if (e[0], e[1], e[2]) not in resolved]
        if not unresolved:
            continue
        if len(entries) == 1:
            network_id, session_id, item_index, item = entries[0]
            fallback = hashlib.sha256(
                (item.get("assertion_text", "") + "|".join(item.get("source_turn_ids", []))).encode()
            ).hexdigest()[:12]
            resolved[(network_id, session_id, item_index)] = {
                "topic_key": f"singleton_{fallback}", "temporal_scope": "unspecified",
                "temporal_precision": "unknown", "effective_from": "", "effective_to": "",
            }
        else:
            # Cap prompt size -- a very active (person, facet) bucket could
            # otherwise blow up one call; topic_key consistency is only
            # guaranteed within one call, so chunking is a real (documented,
            # not hidden) limitation, not silently ignored.
            for start in range(0, len(entries), 40):
                pending.append((bucket_key, entries[start:start + 40]))

    if pending:
        print(f"Resolving topic slots for {len(pending)} multi-assertion buckets...", file=sys.stderr)
        slot_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with slot_cache_path.open("a") as output, ThreadPoolExecutor(max_workers=SLOT_RESOLUTION_WORKERS) as pool:
            futures = {
                pool.submit(_resolve_slots_for_bucket, [e[3] for e in entries]): entries
                for _, entries in pending
            }
            for done, future in enumerate(as_completed(futures), 1):
                entries = futures[future]
                try:
                    by_index = future.result()
                except Exception as error:
                    print(f"  slot resolution failed for a bucket: {error}", file=sys.stderr)
                    continue
                for local_index, (network_id, session_id, item_index, item) in enumerate(entries):
                    entry = by_index.get(local_index) or {}
                    slot = {
                        "topic_key": _norm(str(entry.get("topic_key") or f"unresolved_{item_index}")),
                        "temporal_scope": entry.get("temporal_scope", "unspecified"),
                        "temporal_precision": entry.get("temporal_precision", "unknown"),
                        "effective_from": entry.get("effective_from") or "",
                        "effective_to": entry.get("effective_to") or "",
                    }
                    resolved[(network_id, session_id, item_index)] = slot
                    output.write(json.dumps({
                        "network_id": network_id, "session_id": session_id,
                        "item_index": item_index, "slot": slot,
                    }, ensure_ascii=False) + "\n")
                output.flush()
                if done % 20 == 0 or done == len(pending):
                    print(f"  resolved {done}/{len(pending)} buckets", file=sys.stderr)
    return resolved


def materialize_cell_registries(
    conversations: pd.DataFrame, cache: dict[tuple[str, str], dict],
    slot_resolution: dict[tuple[str, str, int], dict],
) -> tuple[dict[str, dict[CellKey, MemoryCell]], Counter, list[dict], dict[str, list[Assertion]]]:
    """Phase 2: pure Python + local (non-LLM) embeddings only. Replays
    cached, slot-resolved assertions in strict chronological (session_index)
    order per network. Beyond the exact (owner, subject, facet, scope, topic)
    key match, a SECOND, embedding-based check (resolve_topic_slot) merges a
    new topic_key into an existing one in the SAME coarse bucket only above
    a conservative similarity threshold -- catches the slot-resolution LLM
    pass using slightly different topic_key strings for what is really the
    same topic (e.g. a bucket got chunked into two calls), without risking a
    false merge of genuinely different topics. Every such embedding-based
    merge is recorded in the returned merge_events list -- these are
    precisely the "ambiguous same-slot decisions" that need human review,
    since they were NOT a simple exact topic_key match. Also returns every
    assertion that passed validate_assertion (topic_key already resolved),
    per network, REGARDLESS of whether item D's decide_state_operation went
    on to apply/observe/augment/reject it -- Stage 4.5's audits (A/B/C) need
    this "what did extraction actually produce" view independently of what
    the cell layer did with it, to tell extraction failures apart from
    materialization failures."""
    stats = Counter()
    merge_events: list[dict] = []
    validated_assertions: dict[str, list[Assertion]] = defaultdict(list)
    session_meta = (
        conversations[["network_id", "session_id", "session_index"]]
        .drop_duplicates()
        .astype({"network_id": str, "session_id": str})
    )
    order = {
        (row.network_id, row.session_id): row.session_index
        for row in session_meta.itertuples(index=False)
    }
    turns_by_session = {
        (str(network_id), str(session_id)): frame
        for (network_id, session_id), frame in conversations.groupby(["network_id", "session_id"], sort=True)
    }

    registries: dict[str, dict[CellKey, MemoryCell]] = defaultdict(dict)
    topic_embeddings: dict[tuple, list[tuple[str, Any]]] = defaultdict(list)
    selected_networks = set(conversations.network_id.astype(str))
    ordered_keys = sorted(
        (key for key in cache if key[0] in selected_networks),
        key=lambda key: (key[0], order.get(key, 0)),
    )
    for network_id, session_id in ordered_keys:
        frame = turns_by_session.get((network_id, session_id))
        if frame is None:
            continue
        valid_source_ids = {str(turn_id) for turn_id in frame.turn_id}
        # The model sometimes returns a short-form turn_id ("t000") instead
        # of copying the real one verbatim ("grp_x_s04_t000") even though the
        # prompt shows it in full -- found via a real ~40% rejection rate on
        # one session. The real id is a deterministic "<session_id>_<short>"
        # construction, not a guess, so reconstruct it rather than either
        # trusting an unverified id or silently losing real signal to a
        # formatting quirk.
        suffix_to_full = {full_id[len(session_id) + 1:]: full_id for full_id in valid_source_ids}
        timestamp_by_turn = {str(row.turn_id): str(row.timestamp) for row in frame.itertuples(index=False)}
        registry = registries[network_id]
        for item_index, raw_item in enumerate(cache[(network_id, session_id)]["raw_items"]):
            raw_item = dict(raw_item)
            raw_item["conversation_id"] = session_id
            normalized_ids = []
            for source_id in raw_item.get("source_turn_ids", []):
                source_id = str(source_id)
                if source_id in valid_source_ids:
                    normalized_ids.append(source_id)
                elif source_id in suffix_to_full:
                    normalized_ids.append(suffix_to_full[source_id])
            raw_item["source_turn_ids"] = normalized_ids
            # observed_at is derived server-side from real turn timestamps,
            # never trusted from the model -- same discipline as
            # memory_facts.py's _resolve_observed_at.
            raw_item["observed_at"] = (
                max(timestamp_by_turn[source_id] for source_id in normalized_ids)
                if normalized_ids else ""
            )
            slot = slot_resolution.get((network_id, session_id, item_index)) or {}
            raw_item["topic_key"] = slot.get("topic_key", "")
            raw_item["temporal_scope"] = slot.get("temporal_scope", "unspecified")
            raw_item["temporal_precision"] = slot.get("temporal_precision", "unknown")
            raw_item["effective_from"] = slot.get("effective_from") or raw_item["observed_at"]
            raw_item["effective_to"] = slot.get("effective_to") or None

            assertion = validate_assertion(raw_item, valid_source_ids, network_id)
            stats["raw_items"] += 1
            if assertion is None:
                stats["rejected"] += 1
                continue

            bucket_key = (
                network_id, _norm(assertion.viewpoint_owner), _norm(assertion.subject),
                _norm(assertion.facet), assertion.scope_type,
            )
            candidates = topic_embeddings[bucket_key]
            new_embedding = embed(assertion.topic_key)
            resolved_topic = resolve_topic_slot(assertion.topic_key, new_embedding, candidates)
            if resolved_topic is None:
                resolved_topic = assertion.topic_key
                topic_embeddings[bucket_key].append((assertion.topic_key, new_embedding))
                stats["topic_slot:new"] += 1
            elif resolved_topic != assertion.topic_key:
                stats["topic_slot:merged"] += 1
                merge_events.append({
                    "network_id": network_id, "session_id": session_id,
                    "viewpoint_owner": assertion.viewpoint_owner, "subject": assertion.subject,
                    "facet": assertion.facet, "scope_type": assertion.scope_type,
                    "proposed_topic_key": assertion.topic_key, "merged_into_topic_key": resolved_topic,
                    "assertion_text": assertion.assertion_text,
                })
            else:
                stats["topic_slot:exact"] += 1
            if resolved_topic != assertion.topic_key:
                assertion = _dc_replace(assertion, topic_key=resolved_topic)
            validated_assertions[network_id].append(assertion)

            result = apply_assertion(registry, assertion, recorded_at=raw_item["observed_at"])
            if result is None:
                # Stage 4.5 item D, rule 5: a content-free "X revised their
                # position" style assertion -- dropped, never adopted as
                # state. Counted separately from validate_assertion-time
                # rejections, which fail structurally (no text/sources),
                # not semantically (no new content).
                stats["rejected_meta_no_value"] += 1
                continue
            stats["applied"] += 1
            if isinstance(result, MemoryObservation):
                stats[f"op:{result.observation_kind}"] += 1
            else:
                stats[f"op:{result.operation}{':blocked' if result.blocked_by_guard else ''}"] += 1
    return dict(registries), stats, merge_events, dict(validated_assertions)


def build_raw_candidates(conversations: pd.DataFrame) -> tuple[dict[str, list[Candidate]], dict[str, Candidate]]:
    by_network = defaultdict(list)
    by_source = {}
    for row in conversations.itertuples(index=False):
        candidate = Candidate(
            candidate_id=f"raw:{row.turn_id}",
            kind="raw",
            text=f"{row.speaker_display_name}: {row.message}",
            source_ids=(row.turn_id,),
            asserted_by=(row.speaker_display_name,),
            entities=(row.speaker_display_name,),
            observed_at=row.timestamp,
        )
        by_network[row.network_id].append(candidate)
        by_source[row.turn_id] = candidate
    return dict(by_network), by_source


TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def _bm25_prepare(candidates: list[Candidate]) -> dict[str, Any]:
    docs = [_tokens(c.text) for c in candidates]
    frequencies = [Counter(doc) for doc in docs]
    df = Counter(token for doc in docs for token in set(doc))
    return {
        "docs": docs,
        "frequencies": frequencies,
        "df": df,
        "avg_len": sum(map(len, docs)) / max(len(docs), 1),
    }


def _bm25_rank(index: dict[str, Any], query: str, limit: int) -> list[int]:
    n = len(index["docs"])
    avg_len = index["avg_len"] or 1
    scores = []
    for i, (doc, frequencies) in enumerate(zip(index["docs"], index["frequencies"])):
        score = 0.0
        for token in set(_tokens(query)):
            freq = frequencies.get(token, 0)
            if not freq:
                continue
            df = index["df"][token]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += idf * freq * 2.5 / (freq + 1.5 * (1 - 0.75 + 0.75 * len(doc) / avg_len))
        scores.append((score, i))
    return [i for score, i in sorted(scores, reverse=True)[:limit] if score > 0]


def _normalized(values) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if not len(matrix):
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _cached_embeddings(candidates: list[Candidate], cache_dir: Path, label: str) -> np.ndarray:
    payload = "\n".join(f"{c.candidate_id}\t{c.text}" for c in candidates)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    path = cache_dir / f"{label}-{digest}.npy"
    if path.exists():
        return np.load(path)
    matrix = _normalized(embed_batch([c.text for c in candidates]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    return matrix


def _dense_rank(matrix: np.ndarray, query_vector: np.ndarray, limit: int) -> list[int]:
    if not len(matrix):
        return []
    # Accelerate's matmul emits bogus overflow warnings for some perfectly
    # finite float32 matrices on macOS; the equivalent elementwise dot does not.
    scores = np.sum(matrix * query_vector, axis=1)
    return np.argsort(-scores)[:limit].tolist()


def _rrf_ids(ranked: list[list[str]], limit: int) -> list[tuple[str, float, set[str]]]:
    scores = defaultdict(float)
    channels = defaultdict(set)
    for channel_no, ids in enumerate(ranked):
        channel = f"lane_{channel_no}"
        for rank, candidate_id in enumerate(ids, 1):
            scores[candidate_id] += 1 / (60 + rank)
            channels[candidate_id].add(channel)
    ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [(candidate_id, scores[candidate_id], channels[candidate_id]) for candidate_id in ordered]


class NetworkIndex:
    def __init__(self, raw: list[Candidate], memory: list[Candidate], cache_dir: Path, network_id: str):
        self.raw = raw
        self.memory = memory
        self.raw_by_id = {c.candidate_id: c for c in raw}
        self.memory_by_id = {c.candidate_id: c for c in memory}
        self.raw_bm25 = _bm25_prepare(raw)
        self.raw_vectors = _cached_embeddings(raw, cache_dir, f"{network_id}-raw")
        self.memory_vectors = _cached_embeddings(memory, cache_dir, f"{network_id}-memory")

    @staticmethod
    def _copy(candidate: Candidate, score: float, channels: set[str]) -> Candidate:
        return Candidate(**{
            **candidate.as_dict(),
            "source_ids": candidate.source_ids,
            "asserted_by": candidate.asserted_by,
            "entities": candidate.entities,
            "channels": set(channels),
            "score": score,
        })

    def retrieve(self, question: str, query_vector: np.ndarray) -> tuple[list[Candidate], list[Candidate]]:
        bm25 = [self.raw[i].candidate_id for i in _bm25_rank(self.raw_bm25, question, 10)]
        dense = [self.raw[i].candidate_id for i in _dense_rank(self.raw_vectors, query_vector, 10)]
        raw = [self._copy(self.raw_by_id[cid], score, channels)
               for cid, score, channels in _rrf_ids([bm25, dense], 12)]

        memory_dense = [self.memory[i].candidate_id for i in _dense_rank(self.memory_vectors, query_vector, 10)]
        query_names = {name.casefold() for name in re.findall(r"\b[A-Z][a-z]+\b", question)}
        entity = [c.candidate_id for c in self.memory
                  if query_names & {name.casefold() for name in c.entities}][:10]
        memory = [self._copy(self.memory_by_id[cid], score, channels)
                  for cid, score, channels in _rrf_ids([memory_dense, entity], 12)]
        return raw, memory


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group())
            return value if isinstance(value, dict) else None
        except Exception:
            return None


def llm_rerank(question: str, candidates: list[Candidate], limit: int = 8) -> list[Candidate]:
    if len(candidates) <= limit:
        return candidates
    from llm.groq_client import get_chat_model

    rows = []
    for i, candidate in enumerate(candidates, 1):
        rows.append(
            f"[C{i}] type={candidate.kind}; asserted_by={list(candidate.asserted_by)}; "
            f"entities={list(candidate.entities)}; observed_at={candidate.observed_at}\n{candidate.text}"
        )
    prompt = f"""Select at most {limit} evidence candidates needed to answer the question.
Raw and derived candidates must be judged by the SAME standard. Consider semantic
relevance, correct person attribution, temporal applicability, and whether removing
the candidate could change the answer. Do not prefer a candidate merely because it
is derived or raw. Return JSON only:
{{"selected":[{{"id":"C1","role":"direct_evidence|attribution|temporal_update|group_context|irrelevant"}}]}}

Question: {question}

Candidates:
{chr(10).join(rows)}"""
    message = get_chat_model("fast", 0).invoke(prompt)
    parsed = _parse_json_object(message.content if isinstance(message.content, str) else "") or {}
    chosen = []
    for item in parsed.get("selected", []):
        match = re.fullmatch(r"C(\d+)", str(item.get("id", "")))
        if not match:
            continue
        index = int(match.group(1)) - 1
        if 0 <= index < len(candidates) and candidates[index].candidate_id not in {c.candidate_id for c in chosen}:
            c = candidates[index]
            c.role = str(item.get("role", ""))
            chosen.append(c)
        if len(chosen) == limit:
            break
    return chosen or candidates[:limit]


def _unified_pool(raw: list[Candidate], memory: list[Candidate], limit: int = 24) -> list[Candidate]:
    by_id = {c.candidate_id: c for c in raw + memory}
    ranked = _rrf_ids(
        [[c.candidate_id for c in raw], [c.candidate_id for c in memory]],
        limit,
    )
    return [NetworkIndex._copy(by_id[cid], score, channels) for cid, score, channels in ranked]


def render_context(
    candidates: list[Candidate], raw_by_source: dict[str, Candidate], expand_provenance: bool
) -> tuple[str, set[str]]:
    lines, exposed, rendered_raw = [], set(), set()
    for candidate in candidates:
        if candidate.kind == "raw":
            source_id = candidate.source_ids[0]
            if source_id not in rendered_raw:
                lines.append(f"[[{source_id}]] {candidate.observed_at} {candidate.text}")
                exposed.add(source_id)
                rendered_raw.add(source_id)
            continue
        lines.append(
            f"[DERIVED {candidate.candidate_id}] type={candidate.kind}; "
            f"asserted_by={list(candidate.asserted_by)}; entities={list(candidate.entities)}\n{candidate.text}"
        )
        if expand_provenance:
            for source_id in candidate.source_ids:
                raw = raw_by_source[source_id]
                if source_id not in rendered_raw:
                    lines.append(f"  SOURCE [[{source_id}]] {raw.observed_at} {raw.text}")
                    exposed.add(source_id)
                    rendered_raw.add(source_id)
    return "\n".join(lines), exposed


def _option_lines(options: Any) -> list[str]:
    if isinstance(options, dict):
        return [f"{key}. {value}" for key, value in options.items()]
    if isinstance(options, list):
        return [
            f"{item['option']}. {item['name']}"
            if isinstance(item, dict) and "option" in item and "name" in item
            else str(item)
            for item in options
        ]
    return []


def _answer(question: str, options: Any, context: str) -> str:
    from llm.groq_client import get_chat_model

    option_lines = _option_lines(options)
    options_block = "\nOptions:\n" + "\n".join(option_lines) if option_lines else ""
    prompt = f"""Answer using only the supplied evidence. Preserve who said or
revealed each fact, distinguish group norms from individual exceptions, and respect
temporal updates. Cite raw evidence as [[turn_id]]. DERIVED items cannot themselves
be cited; cite only displayed raw/SOURCE turns. If evidence is insufficient, say so.
For a multiple-choice question, end with FINAL_OPTION: <letter>.

Question: {question}{options_block}

Evidence:
{context}"""
    message = get_chat_model("primary", 0).invoke(prompt)
    return message.content if isinstance(message.content, str) else ""


def _judge_open_answers(question: str, gold: str, answers: dict[str, str]) -> dict[str, float]:
    from llm.groq_client import get_chat_model

    rendered = "\n\n".join(f"{name}:\n{text}" for name, text in answers.items())
    prompt = f"""Score each answer independently against the gold answer from 0.0
to 1.0 in increments of 0.1. Ignore citation formatting when judging correctness.
Do not compare systems to each other. Return JSON only, mapping every system name
to a numeric score.

Question: {question}
Gold answer: {gold}

Answers:
{rendered}"""
    message = get_chat_model("primary", 0).invoke(prompt)
    parsed = _parse_json_object(message.content if isinstance(message.content, str) else "") or {}
    return {name: float(parsed.get(name, 0.0)) for name in answers}


def _cited_ids(answer: str) -> set[str]:
    return set(re.findall(r"\[\[([^\]]+)\]\]", answer or ""))


def _set_metrics(found: set[str], gold: set[str]) -> dict[str, float]:
    overlap = found & gold
    return {
        "recall": len(overlap) / len(gold) if gold else 1.0,
        "precision": len(overlap) / len(found) if found else 0.0,
    }


def _variant_metrics(
    selected: list[Candidate], exposed: set[str], gold: set[str], answer: str,
    valid_sources: set[str], context: str,
) -> dict[str, Any]:
    internal_sources = {source for candidate in selected for source in candidate.source_ids}
    cited = _cited_ids(answer)
    return {
        "selected_count": len(selected),
        "derived_selected": sum(c.kind != "raw" for c in selected),
        "candidate_evidence": _set_metrics(internal_sources, gold),
        "exposed_evidence": _set_metrics(exposed, gold),
        "citation_evidence": _set_metrics(cited, gold),
        "citation_validity": len(cited & valid_sources) / len(cited) if cited else 0.0,
        "context_words": len(_tokens(context)),
    }


def _select_questions(qa: pd.DataFrame, per_type: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected = []
    for query_type in PILOT_TYPES:
        rows = qa[qa.query_type == query_type].to_dict("records")
        rows = rng.sample(rows, min(per_type, len(rows))) if per_type else rows
        selected.extend(rows)
    return sorted(selected, key=lambda row: (row["query_type"], row["qa_id"]))


def _mc_score(answer: str, correct_option: str) -> float:
    matches = re.findall(r"FINAL_OPTION:\s*([A-Z])", answer or "", re.IGNORECASE)
    return float(bool(matches) and matches[-1].upper() == correct_option.upper())


def run_case(
    row: dict[str, Any], index: NetworkIndex, raw_by_source: dict[str, Candidate], mode: str,
    memory_source: str,
) -> dict[str, Any]:
    question = row["question"]
    qvec = _normalized([embed(question)])[0]
    raw, memory = index.retrieve(question, qvec)
    unified_pool = _unified_pool(raw, memory)

    if mode == "full":
        raw_selected = llm_rerank(question, raw)
        unified_selected = llm_rerank(question, unified_pool)
    else:
        raw_selected = raw[:8]
        unified_selected = unified_pool[:8]

    selections = {
        "raw": raw_selected,
        "late_memory": raw_selected + memory[:8],
        "unified": unified_selected,
        "provenance": unified_selected,
    }
    answers, contexts, exposed = {}, {}, {}
    options = _json(row.get("options_json"), {})
    for name, candidates in selections.items():
        context, source_ids = render_context(
            candidates, raw_by_source, expand_provenance=(name == "provenance")
        )
        contexts[name], exposed[name] = context, source_ids
        answers[name] = _answer(question, options, context) if mode == "full" else ""

    if mode == "full" and row["answer_format"] == "multiple_choice":
        correctness = {name: _mc_score(answer, row["correct_option"]) for name, answer in answers.items()}
    elif mode == "full":
        correctness = _judge_open_answers(question, row["answer"], answers)
    else:
        correctness = {name: None for name in VARIANTS}

    anchors = _json(row["evidence_anchors_json"], [])
    gold = {anchor["turn_id"] for anchor in anchors}
    valid_sources = set(raw_by_source)
    variants = {}
    for name in VARIANTS:
        variants[name] = {
            "selected": [candidate.as_dict() for candidate in selections[name]],
            "context": contexts[name],
            "answer": answers[name],
            "correctness": correctness[name],
            "metrics": _variant_metrics(
                selections[name], exposed[name], gold, answers[name], valid_sources, contexts[name]
            ),
        }
    return {
        "qa_id": row["qa_id"],
        "network_id": row["network_id"],
        "query_type": row["query_type"],
        "question": question,
        "gold_answer": row["answer"],
        "gold_source_ids": sorted(gold),
        "memory_source": memory_source,
        "oracle_warning": (
            "evidence_oracle uses QA anchor relevance; diagnostic upper bound, not learned ingestion"
            if memory_source == "oracle" else None
        ),
        "variants": variants,
    }


def summarize(rows: list[dict[str, Any]]) -> str:
    memory_source = rows[0].get("memory_source", "oracle") if rows else "unknown"
    description = (
        "> Evidence-oracle diagnostic: memory text uses QA anchor relevance metadata, never gold answers. "
        "This evaluates retrieval/reranking/provenance mechanics, not learned ingestion."
        if memory_source == "oracle" else
        "> Extracted-memory run: memory was generated from conversations alone, without QA questions, "
        "answers, evidence anchors, or production chat data."
    )
    lines = [
        "# SocialMemBench heterogeneous-memory pilot",
        "",
        description,
        "",
    ]
    for scope, scoped_rows in [("all", rows)] + [
        (query_type, [row for row in rows if row["query_type"] == query_type])
        for query_type in PILOT_TYPES
    ]:
        if not scoped_rows:
            continue
        lines.extend([
            f"## {scope} (n={len(scoped_rows)})", "",
            "| variant | candidate R/P | exposed R/P | citation R/P | valid cites | words | correctness |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for variant in VARIANTS:
            metrics = [row["variants"][variant]["metrics"] for row in scoped_rows]
            candidate_r = [m["candidate_evidence"]["recall"] for m in metrics]
            candidate_p = [m["candidate_evidence"]["precision"] for m in metrics]
            exposed_r = [m["exposed_evidence"]["recall"] for m in metrics]
            exposed_p = [m["exposed_evidence"]["precision"] for m in metrics]
            citation_r = [m["citation_evidence"]["recall"] for m in metrics]
            citation_p = [m["citation_evidence"]["precision"] for m in metrics]
            validity = [m["citation_validity"] for m in metrics]
            words = [m["context_words"] for m in metrics]
            correctness = [row["variants"][variant]["correctness"] for row in scoped_rows if row["variants"][variant]["correctness"] is not None]
            score = f"{sum(correctness)/len(correctness):.3f}" if correctness else "n/a"
            lines.append(
                f"| {variant} | {sum(candidate_r)/len(candidate_r):.3f}/{sum(candidate_p)/len(candidate_p):.3f} | "
                f"{sum(exposed_r)/len(exposed_r):.3f}/{sum(exposed_p)/len(exposed_p):.3f} | "
                f"{sum(citation_r)/len(citation_r):.3f}/{sum(citation_p)/len(citation_p):.3f} | "
                f"{sum(validity)/len(validity):.3f} | {sum(words)/len(words):.0f} | {score} |"
            )
        lines.append("")
    return "\n".join(lines)


def run(args) -> None:
    conversations, qa, personas = load_data(args.data_dir)
    selected = _select_questions(qa, args.per_type, args.seed)
    if args.max_cases:
        selected = selected[:args.max_cases]
    selected_networks = {row["network_id"] for row in selected}
    conversations = conversations[conversations.network_id.isin(selected_networks)]
    personas = personas[personas.network_id.isin(selected_networks)]

    raw_by_network, raw_by_source = build_raw_candidates(conversations)
    memory_by_network = (
        build_oracle_units(conversations, qa[qa.network_id.isin(selected_networks)], personas)
        if args.memory_source == "oracle"
        else build_extracted_units(conversations, args.memory_cache)
    )
    indexes = {
        network_id: NetworkIndex(
            raw_by_network[network_id], memory_by_network.get(network_id, []),
            args.data_dir / ".embedding_cache", network_id,
        )
        for network_id in selected_networks
    }

    completed = {}
    if args.resume and args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("memory_source", "oracle") != args.memory_source:
                    raise ValueError("Resume file was produced with a different --memory-source")
                completed[row["qa_id"]] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if completed else "w"
    with args.output.open(mode) as output:
        for number, row in enumerate(selected, 1):
            if row["qa_id"] in completed:
                continue
            print(f"[{number}/{len(selected)}] {row['qa_id']} {row['query_type']}", file=sys.stderr)
            result = run_case(
                row, indexes[row["network_id"]], raw_by_source, args.mode, args.memory_source
            )
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            completed[row["qa_id"]] = result

    ordered = [completed[row["qa_id"]] for row in selected if row["qa_id"] in completed]
    report = summarize(ordered)
    args.report.write_text(report)
    print(report, file=sys.stderr)
    print(f"Results: {args.output}\nReport: {args.report}", file=sys.stderr)


def _render_cell_review_sheet(
    genuine_chains: list[tuple[str, MemoryCell]], merge_events: list[dict],
) -> str:
    """Two sections, per the v1 correction's explicit requirement not to
    call every cell with >1 record a "real update chain":

    1. Genuine revise/retract chains -- cells with >=2 STATE versions
       (observations never count). Each state shows topic_key, temporal_scope,
       modality/confidence, and how many observations reinforce it.
    2. Ambiguous same-slot decisions -- every case where the embedding-based
       resolve_topic_slot merged a NEWLY proposed topic_key into an EXISTING
       one instead of an exact string match. These are exactly the risky
       merge calls that need a human's judgment, not the exact matches."""
    from research.versioned_memory_cells import state_chain

    lines = ["# Cell update-chain review sheet (v1, topic-scoped)", ""]
    lines.append(f"Genuine revise/retract chains: {len(genuine_chains)}")
    lines.append(f"Ambiguous same-slot (embedding-merged) decisions: {len(merge_events)}")
    lines.append("")
    lines.append("## Genuine revise/retract chains")
    lines.append("")
    for number, (network_id, cell) in enumerate(genuine_chains, 1):
        key = cell.key
        observations_by_state = Counter(o.state_version_id for o in cell.observations)
        lines.append(
            f"### {number}. {network_id} :: {key.viewpoint_owner} -> {key.subject} :: "
            f"{key.facet} / topic={key.topic_key} ({key.scope_type})"
        )
        for version in state_chain(cell):
            tags = []
            if version.blocked_by_guard:
                tags.append("BLOCKED")
            if version.version_id in cell.active_state_version_ids:
                tags.append("ACTIVE")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            n_obs = observations_by_state.get(version.version_id, 0)
            lines.append(
                f"- v{version.version_no} [{version.operation}{tag_str}] "
                f"modality={version.modality} conf={version.confidence} value={version.normalized_value!r} "
                f"observations={n_obs}"
            )
            lines.append(f'  "{version.assertion_text}"')
            lines.append(
                f"  temporal_scope={version.temporal_scope} effective_from={version.effective_from} "
                f"effective_to={version.effective_to} closed_at={version.closed_at}"
            )
            lines.append(f"  sources={list(version.source_turn_ids)}")
        lines.append("")
        lines.append("Decision:")
        lines.append("- [ ] chain correct (each link genuinely updates the same slot)")
        lines.append("- [ ] questionable (unrelated facts got linked, or a real update was missed)")
        lines.append("- [ ] wrong (should not have been linked at all)")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Ambiguous same-slot decisions (embedding-based merges)")
    lines.append("")
    for number, event in enumerate(merge_events, 1):
        lines.append(
            f"### {number}. {event['network_id']} :: {event['viewpoint_owner']} -> {event['subject']} :: "
            f"{event['facet']} ({event['scope_type']})"
        )
        lines.append(f"proposed_topic_key: {event['proposed_topic_key']!r}")
        lines.append(f"merged_into_topic_key: {event['merged_into_topic_key']!r}")
        lines.append(f'assertion_text: "{event["assertion_text"]}"')
        lines.append("")
        lines.append("Decision:")
        lines.append("- [ ] merge correct (genuinely the same topic)")
        lines.append("- [ ] false merge (should have stayed separate -- split it)")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def run_materialize(args) -> None:
    """Stage 4 (v1, topic-scoped): materialize cells for every network
    touched by the frozen 120-question selection (same _select_questions
    call as run(), so the corpus scope matches the existing flat-extraction
    results exactly). Reuses the ALREADY-cached raw session extraction
    (build_cell_assertion_cache) -- no new per-session LLM calls -- and adds
    a topic-slot resolution pass (build_slot_resolution_cache) before
    materializing. Reports honest, decomposed statistics (state cells vs
    observation-only cells vs genuine revise/retract chains vs blocked
    attempts vs suspected false merges) instead of calling every cell with
    more than one record a "real update chain"."""
    conversations, qa, _ = load_data(args.data_dir)
    selected = _select_questions(qa, args.per_type, args.seed)
    if args.max_cases:
        selected = selected[:args.max_cases]
    selected_networks = {row["network_id"] for row in selected}
    conversations = conversations[conversations.network_id.isin(selected_networks)]

    cache = build_cell_assertion_cache(conversations, args.cell_cache)
    slot_resolution = build_slot_resolution_cache(cache, args.slot_cache, selected_networks)
    registries, stats, merge_events, _validated_assertions = materialize_cell_registries(conversations, cache, slot_resolution)

    all_cells = [cell for registry in registries.values() for cell in registry.values()]
    state_chain_lengths = Counter(len(cell.state_versions) for cell in all_cells)
    genuine_chains = [
        (network_id, cell) for network_id, registry in registries.items()
        for cell in registry.values() if len(cell.state_versions) > 1
    ]
    observation_only_cells = sum(
        1 for cell in all_cells if len(cell.state_versions) == 1 and cell.observations
    )
    singleton_cells = sum(
        1 for cell in all_cells if len(cell.state_versions) == 1 and not cell.observations
    )
    retract_chains = sum(
        1 for cell in all_cells
        if any(v.operation == "retract" for v in cell.state_versions)
    )
    total_observations = sum(len(cell.observations) for cell in all_cells)
    cells_with_states = sum(1 for cell in all_cells if cell.state_versions)
    avg_observations_per_state = (
        total_observations / sum(len(cell.state_versions) for cell in all_cells)
        if any(cell.state_versions for cell in all_cells) else 0.0
    )

    # in-scope session count, distinct from the raw cache's total entries --
    # the cache can (harmlessly) hold sessions from networks outside THIS
    # run's selection, left over from an earlier/broader run; only count
    # what actually got materialized here.
    materialized_sessions = sum(
        1 for (network_id, _session_id) in cache if network_id in selected_networks
    )

    report_lines = [
        "# Cell materialization stats (v1, topic-scoped)", "",
        f"Cache entries total: {len(cache)}",
        f"Selected networks: {len(selected_networks)}",
        f"Selected/materialized sessions: {materialized_sessions}",
        f"Total cells: {len(all_cells)}",
        f"  state cells (>=1 state version): {cells_with_states}",
        f"  singleton cells (1 state, 0 observations): {singleton_cells}",
        f"  cells with observations only (1 state, >=1 observation): {observation_only_cells}",
        f"  genuine revise/retract chains (>1 state version): {len(genuine_chains)}",
        f"  cells containing a retract: {retract_chains}",
        f"Raw items: {stats['raw_items']}, applied: {stats['applied']}, rejected: {stats['rejected']}, "
        f"rejected (meta/content-free value, item D rule 5): {stats['rejected_meta_no_value']}",
        f"Total observations: {total_observations} (avg {avg_observations_per_state:.2f} per state version)",
        f"Suspected false merges (ambiguous, embedding-based): {stats['topic_slot:merged']} "
        f"(exact topic_key matches: {stats['topic_slot:exact']}, new topic slots: {stats['topic_slot:new']})",
        "", "## Operations", "",
    ]
    for op_key in sorted(k for k in stats if k.startswith("op:")):
        report_lines.append(f"- {op_key}: {stats[op_key]}")
    report_lines += ["", "## State-chain length distribution (state versions only, not observations)", ""]
    for length in sorted(state_chain_lengths):
        report_lines.append(f"- {length} state version(s): {state_chain_lengths[length]} cells")

    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text("\n".join(report_lines))
    print("\n".join(report_lines), file=sys.stderr)

    review = _render_cell_review_sheet(genuine_chains, merge_events)
    args.review_sheet.write_text(review)
    print(
        f"\nStats: {args.stats_output}\n"
        f"Review sheet ({len(genuine_chains)} genuine chains + {len(merge_events)} ambiguous decisions): {args.review_sheet}",
        file=sys.stderr,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("retrieval", "full", "materialize"), default="retrieval")
    parser.add_argument("--per-type", type=int, default=2, help="0 means every QA in the six pilot types")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--max-cases", type=int, default=0, help="global cap for a cheap smoke test")
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/socialmembench"))
    parser.add_argument("--memory-source", choices=("oracle", "extracted"), default="oracle")
    parser.add_argument(
        "--memory-cache", type=Path, default=Path("/tmp/socialmembench_extracted_memory.jsonl")
    )
    parser.add_argument(
        "--cell-cache", type=Path, default=Path("/tmp/socialmembench_cell_assertions.jsonl"),
        help="raw per-session extraction cache -- read/reused, never overwritten with new content by materialize",
    )
    parser.add_argument(
        "--slot-cache", type=Path, default=Path("/tmp/socialmembench_cell_slot_resolution.jsonl"),
        help="topic_key/temporal slot-resolution cache, separate from --cell-cache",
    )
    parser.add_argument(
        "--stats-output", type=Path, default=Path("/tmp/socialmembench_cell_stats_v1.md"),
        help="v1-suffixed by default so the frozen v0 stats file is never overwritten",
    )
    parser.add_argument(
        "--review-sheet", type=Path, default=Path("/tmp/socialmembench_cell_review_sheet_v1.md"),
        help="v1-suffixed by default so the frozen v0 review sheet is never overwritten",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/socialmembench_pilot_results.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("/tmp/socialmembench_pilot_report.md"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "materialize":
        run_materialize(args)
    else:
        run(args)
