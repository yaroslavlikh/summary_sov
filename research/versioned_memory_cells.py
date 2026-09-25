"""Versioned scoped memory cells -- v1, topic-scoped identity.

v0 (facet-only CellKey = network_id/viewpoint_owner/subject/facet/scope_type)
was materialized, manually reviewed, and found to have a real identity bug,
not a taxonomy problem: broad facets like "attitude" and "commitment" merged
UNRELATED semantic states of the same person into one evolving cell (an
opinion about a hike silently "revised" into an opinion about eggs, because
both were tagged facet=attitude for the same person). Confirmed example:
research/VERSIONED_MEMORY_CELLS.md's review notes / the frozen v0 artifacts
in /tmp/*_v0_naive_facet.*. This module is the fix: CellKey identity gains a
topic_key, and state-changing updates additionally require the new and
existing assertion's TEMPORAL windows to actually conflict before one can
close the other -- two non-overlapping plans about the same topic now
coexist instead of one silently overwriting the other.

Still deliberately NOT a knowledge graph: no generic node/edge type, no
relation-typed edges (CONTRADICTS/EVOLVED_FROM/SUPPORTS), no traversal. A
cell's evolution is explicit state (an ordered, immutable version list per
topic, each with operation/effective_from/effective_to/closed_at), not
graph structure. topic_key disambiguates identity; it is not a graph edge
to anything.

Core objects:
  - Assertion -- one atomic claim from ingestion, now carrying topic_key
    (which specific state slot this is about) and temporal fields
    (observed_at/effective_from/effective_to/temporal_scope/
    temporal_precision), not yet linked to any cell.
  - CellKey -- (network_id, viewpoint_owner, subject, facet, topic_key,
    scope_type), normalized. topic_key is what makes two assertions about
    the SAME evolving state rather than two unrelated facts that happen to
    share a person and a broad facet label.
  - MemoryStateVersion -- one immutable, append-only STATE record
    (operation in create/revise/retract). A cell can have more than one
    SIMULTANEOUSLY ACTIVE state version if their effective_from/effective_to
    windows genuinely don't overlap (e.g. two non-conflicting future plans
    under the same topic) -- state is not forced into a single "current"
    slot the way v0 did.
  - MemoryObservation -- a repeat of the SAME state (same topic, same
    value, overlapping window) is NOT a new state version and does not
    extend the evolution chain -- it attaches as an observation to the
    state version it reinforces (more source_turn_ids, higher confidence),
    exactly the "reaffirm must not increase chain depth" requirement.
  - MemoryCell -- a key plus its full state-version history, its attached
    observations, and which state version id(s) are currently active.

decide_state_operation is deterministic, not a separate LLM call: it
compares the new assertion's topic_key (already resolved to a cell key by
the ingestion layer, see resolve_topic_slot below) and temporal window
against each of the cell's ACTIVE state versions. A state version can only
be closed (revise/retract) by a new assertion whose window genuinely
overlaps it, or which explicitly replaces it (operation_hint=revise with
sufficient confidence) -- non-overlapping windows never compete, they both
stay active.

resolve_topic_slot decides, given a NEW topic_key string and its embedding,
whether it should merge into an EXISTING topic_key among cells that already
share (network_id, viewpoint_owner, subject, facet, scope_type), or stand
as its own new slot. This module never computes embeddings itself (no LLM/
embedding calls here at all) -- the caller (ingestion layer) supplies
precomputed vectors, keeping this module pure and fast to test. The default
is conservative: an uncertain match creates a NEW cell rather than merging
into an existing one -- false split (two cells that turn out to be the same
thing) is recoverable later; false merge (one state silently overwriting an
unrelated one) already caused the real bug that motivated this rewrite.

Deferred to Stage 5 (retrieval, needs the real corpus + an LLM for the slot
resolution pass over multi-assertion buckets): the actual ingestion wiring
and the cell-based benchmark variants. Tests for those live with that
stage, not here.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field, replace
from typing import NamedTuple, Optional, Sequence

SCOPE_TYPES = {"individual", "subgroup", "group", "relationship", "epistemic", "event"}
MODALITIES = {
    "asserted", "self_report", "reported_by_other", "opinion", "inferred",
    "uncertain", "joke", "sarcasm", "proposal", "commitment", "group_consensus", "dissent",
}
OPERATIONS = {"create", "reaffirm", "revise", "retract"}
TEMPORAL_SCOPES = {"ongoing", "dated_event", "recurring", "unspecified"}
TEMPORAL_PRECISIONS = {"exact", "approximate", "unknown"}

# Relative strength of modalities -- governs whether a NEW assertion is
# allowed to CLOSE an existing active state version. A lower-strength
# modality can never close a higher-strength one, regardless of whether
# their normalized_value differs. This guard only applies to state-changing
# operations (revise/retract); a mere observation never changes state, so
# there is nothing for a weak modality to damage there.
MODALITY_STRENGTH = {
    "joke": 0, "sarcasm": 0,
    "uncertain": 1, "proposal": 1, "inferred": 1,
    "opinion": 2, "dissent": 2, "reported_by_other": 2,
    "asserted": 3, "self_report": 3, "commitment": 3, "group_consensus": 3,
}
REVISE_CONFIDENCE_THRESHOLD = 0.5

# individual > subgroup/relationship/epistemic > group/event -- used only to
# ORDER candidate cells at retrieval time (Stage 5); ingestion never lets an
# individual-scope and a group-scope assertion collide, because scope_type
# is part of the cell key, so they are always different cells.
SCOPE_SPECIFICITY = {"individual": 3, "relationship": 2, "epistemic": 2, "subgroup": 2, "event": 1, "group": 1}

# Conservative on purpose: false split (two cells for the same real topic)
# is a retrieval-time inconvenience; false merge (one topic's state
# silently closing an unrelated topic's state) reproduces the exact bug
# this rewrite exists to fix. Raise, don't lower, if in doubt.
TOPIC_MATCH_THRESHOLD = 0.86

# Stage 4.5 item D: an assertion whose normalized_value is itself just a
# reference to "something changed" (no actual new content) can never become
# the current state -- e.g. "Derek revised his position" extracted as
# normalized_value="revised". Matched against the FULL normalized value
# only (not a substring search) so a legitimate value that merely contains
# one of these words ("new policy") is never caught by accident.
_META_VALUE_PATTERN = re.compile(
    r"^(revised|changed|updated|different|new)"
    r"(\s+(position|mind|opinion|stance|preference|choice))?$"
)


def _is_meta_value(value: Optional[str]) -> bool:
    if not value:
        return False
    return bool(_META_VALUE_PATTERN.match(_norm(value)))


def _norm(value: str) -> str:
    return (value or "").strip().casefold()


class CellKey(NamedTuple):
    network_id: str
    viewpoint_owner: str
    subject: str
    facet: str
    topic_key: str
    scope_type: str


def _cell_id(key: CellKey) -> str:
    return hashlib.sha256("|".join(key).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Assertion:
    """One atomic claim from ingestion, not yet linked to any cell. Has no
    field for QA question/answer/evidence-anchor data by construction --
    satisfies "ingestion API can't accept QA metadata" structurally, not by
    a runtime check."""
    network_id: str
    viewpoint_owner: str
    subject: str
    facet: str
    topic_key: str
    scope_type: str
    assertion_text: str
    modality: str
    confidence: float
    observed_at: str
    source_turn_ids: tuple[str, ...]
    normalized_value: Optional[str] = None
    effective_from: str = ""
    effective_to: Optional[str] = None
    temporal_scope: str = "unspecified"
    temporal_precision: str = "unknown"
    conversation_id: str = ""
    operation_hint: Optional[str] = None


def validate_assertion(raw: dict, valid_source_ids: set[str], network_id: str) -> Optional[Assertion]:
    """Mirrors the validation discipline already used for flat extraction
    in research/socialmembench_pilot.py: fabricated source_turn_ids are
    dropped, not trusted; an assertion left with zero real sources
    afterward is rejected outright. topic_key is REQUIRED (empty/missing ->
    rejected) -- a state-changing assertion with no resolved slot identity
    cannot be safely placed in the cell registry at all."""
    text = str(raw.get("assertion_text") or "").strip()
    source_ids = tuple(dict.fromkeys(
        str(source_id) for source_id in raw.get("source_turn_ids", [])
        if str(source_id) in valid_source_ids
    ))
    if not text or not source_ids:
        return None

    viewpoint_owner = str(raw.get("viewpoint_owner") or "").strip()
    subject = str(raw.get("subject") or "").strip()
    facet = str(raw.get("facet") or "").strip()
    topic_key = _norm(str(raw.get("topic_key") or ""))
    if not viewpoint_owner or not subject or not facet or not topic_key:
        return None

    scope_type = _norm(str(raw.get("scope_type") or "individual"))
    if scope_type not in SCOPE_TYPES:
        scope_type = "individual"
    modality = _norm(str(raw.get("modality") or "asserted"))
    if modality not in MODALITIES:
        modality = "asserted"
    operation_hint = _norm(str(raw["operation"])) if raw.get("operation") else None
    if operation_hint not in OPERATIONS:
        operation_hint = None
    temporal_scope = _norm(str(raw.get("temporal_scope") or "unspecified"))
    if temporal_scope not in TEMPORAL_SCOPES:
        temporal_scope = "unspecified"
    temporal_precision = _norm(str(raw.get("temporal_precision") or "unknown"))
    if temporal_precision not in TEMPORAL_PRECISIONS:
        temporal_precision = "unknown"

    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(1.0, max(0.0, confidence))

    normalized_value = raw.get("normalized_value")
    normalized_value = str(normalized_value).strip() if normalized_value not in (None, "") else None

    observed_at = str(raw.get("observed_at") or "")
    effective_from = str(raw.get("effective_from") or "") or observed_at
    effective_to = raw.get("effective_to")
    effective_to = str(effective_to).strip() if effective_to not in (None, "") else None

    return Assertion(
        network_id=network_id, viewpoint_owner=viewpoint_owner, subject=subject, facet=facet,
        topic_key=topic_key, scope_type=scope_type, assertion_text=text, modality=modality,
        confidence=confidence, observed_at=observed_at, source_turn_ids=source_ids,
        normalized_value=normalized_value, effective_from=effective_from, effective_to=effective_to,
        temporal_scope=temporal_scope, temporal_precision=temporal_precision,
        conversation_id=str(raw.get("conversation_id") or ""), operation_hint=operation_hint,
    )


@dataclass(frozen=True)
class MemoryStateVersion:
    """A genuine state change: create, revise, or retract. Reaffirming the
    same state is NEVER represented here -- see MemoryObservation."""
    version_id: str
    cell_id: str
    version_no: int
    operation: str  # create | revise | retract
    assertion_text: str
    normalized_value: Optional[str]
    modality: str
    confidence: float
    observed_at: str
    effective_from: str
    effective_to: Optional[str]
    temporal_scope: str
    temporal_precision: str
    recorded_at: str
    conversation_id: str
    source_turn_ids: tuple[str, ...]
    closed_at: Optional[str] = None
    superseded_by: Optional[str] = None
    # True when a modality/confidence guard prevented this attempt from
    # closing the state version it targeted -- still appended for
    # provenance/attribution, but never added to active_state_version_ids.
    blocked_by_guard: bool = False


@dataclass(frozen=True)
class MemoryObservation:
    """A repeat of an ALREADY-active state: same topic, non-conflicting
    value, overlapping window. Adds evidence without extending the
    evolution chain -- decide_state_operation is what decides an assertion
    is an observation rather than a new state version. observation_kind
    distinguishes two internal sub-decisions that both land here without
    ever entering the create/revise/retract evolution history: "observe"
    (a plain reaffirm -- same value, or evidence already counted toward the
    target state) vs "augment" (a compatible additional detail that must
    not be allowed to CLOSE the state it attaches to -- see decide_state_
    operation's rules 1/3)."""
    observation_id: str
    cell_id: str
    state_version_id: str
    assertion_text: str
    modality: str
    confidence: float
    observed_at: str
    recorded_at: str
    conversation_id: str
    source_turn_ids: tuple[str, ...]
    observation_kind: str = "observe"


@dataclass
class MemoryCell:
    cell_id: str
    key: CellKey
    state_versions: list[MemoryStateVersion] = field(default_factory=list)
    observations: list[MemoryObservation] = field(default_factory=list)
    active_state_version_ids: list[str] = field(default_factory=list)

    @property
    def active_states(self) -> list[MemoryStateVersion]:
        by_id = {v.version_id: v for v in self.state_versions}
        return [by_id[i] for i in self.active_state_version_ids if i in by_id]


def _temporal_windows_overlap(
    a_from: str, a_to: Optional[str], b_from: str, b_to: Optional[str],
) -> bool:
    """True if the two windows can't be ruled out as overlapping. Missing
    bounds are treated as open-ended (from="" -> -inf, to=None -> +inf), so
    two assertions with NO temporal information at all are conservatively
    treated as overlapping (matches the old, pre-topic-key default of
    "assume the same ongoing state unless told otherwise") -- only
    assertions that carry ACTUAL, non-overlapping date information get the
    new coexistence behavior. ISO8601-ish date strings compare correctly
    lexicographically."""
    a_lo = a_from or ""
    a_hi = a_to or "￿"
    b_lo = b_from or ""
    b_hi = b_to or "￿"
    return a_lo <= b_hi and b_lo <= a_hi


def decide_state_operation(
    cell: Optional[MemoryCell], assertion: Assertion,
) -> tuple[str, Optional[str], bool]:
    """Returns (decision, target_state_version_id, blocked_by_guard).
    decision is one of the internal vocabulary separate/observe/augment/
    revise/retract/reject (Stage 4.5 item D) -- apply_assertion is what
    collapses this down to an evolution-history operation (only create/
    revise/retract may ever appear there; observe/augment always become a
    MemoryObservation attached to the CURRENT state version, never a new
    one, and reject means the assertion is dropped, never applied at all).
    target is which EXISTING active state version this assertion reacts to
    (None for a fresh "separate", including a new non-overlapping temporal
    instance under an already-existing topic). Deterministic -- no LLM
    call; topic_key identity (which cell this even landed in) was already
    resolved upstream by resolve_topic_slot before this is called.

    Five deterministic invariants enforced here (Stage 4.5 item D), each
    found by manual review of the v1 materialization producing a chain that
    should NOT have been a plain revise:
      1. An assertion that shares a source_turn_id with the state it would
         revise cannot revise it -- the evidence is already counted toward
         that state, so this can only be a repeat or an elaboration of the
         SAME turn(s), never an independent later update.
      2. normalized_value is compared after normalization (casefold/strip)
         so a pure spelling/whitespace difference never masquerades as a
         value change.
      3. An assertion with no normalized_value at all (pure narrative
         detail, no structured value to compare) can never CLOSE a state
         that does have one -- it attaches as an augmenting observation.
      4. Given 1-3, a revise only fires when the (normalized) values differ
         at the surface (string) level, or the extractor explicitly flagged
         operation_hint="revise" with real confidence -- never a bare guess
         from surface value difference alone. IMPORTANT, honestly stated:
         "values differ after normalization" is a SURFACE-level string
         comparison, not a confirmed SEMANTIC incompatibility check. Two
         differently-worded assertions that mean the same thing (e.g.
         "needs volunteer" vs "one_more_volunteer") will still read as
         "differs" here and can still trigger a revise that is not a real
         state change. Detecting genuine semantic equivalence is explicitly
         DEFERRED, not implemented by this rule -- doing it properly needs
         either a calibrated embedding-similarity check on the dev corpus
         or a semantic resolver, neither of which this step adds.
      5. An assertion whose entire normalized_value is a meta reference to
         "something changed" ("revised", "changed position", ...) carries
         no actual new content and can never become the current state --
         rejected outright rather than silently adopted as truth. This
         applies unconditionally, including as the very FIRST assertion
         for a brand-new topic/cell (empty cell, no active state to compare
         against yet) -- a content-free assertion is still content-free
         whether or not anything already exists to revise.
    """
    if assertion.operation_hint != "retract" and _is_meta_value(assertion.normalized_value):
        return "reject", None, False

    active = cell.active_states if cell else []
    if not active:
        return "separate", None, False

    if assertion.operation_hint == "retract":
        target = _best_temporal_match(active, assertion) or active[-1]
        blocked = MODALITY_STRENGTH.get(assertion.modality, 2) < MODALITY_STRENGTH.get(target.modality, 2)
        return "retract", target.version_id, blocked

    for state in active:
        if not _temporal_windows_overlap(assertion.effective_from, assertion.effective_to,
                                          state.effective_from, state.effective_to):
            continue

        # Rule 1: evidence already counted toward this state can't revise it.
        shared_sources = set(assertion.source_turn_ids) & set(state.source_turn_ids)
        if shared_sources:
            new_sources = set(assertion.source_turn_ids) - set(state.source_turn_ids)
            return ("augment" if new_sources else "observe"), state.version_id, False

        # Rule 3: no structured value to compare -- additive detail only.
        if assertion.normalized_value is None:
            return "augment", state.version_id, False

        # Rule 2 + 4: SURFACE (string) comparison only -- NOT a confirmed
        # semantic-incompatibility check. See the rule-4 docstring above:
        # semantically-equivalent-but-differently-worded values will still
        # show up here as "differs" and can still cause a false revise.
        surface_value_differs = (
            state.normalized_value is not None
            and _norm(assertion.normalized_value) != _norm(state.normalized_value)
        )
        wants_revise = surface_value_differs or (
            assertion.operation_hint == "revise" and assertion.confidence >= REVISE_CONFIDENCE_THRESHOLD
        )
        if wants_revise:
            blocked = MODALITY_STRENGTH.get(assertion.modality, 2) < MODALITY_STRENGTH.get(state.modality, 2)
            return "revise", state.version_id, blocked
        return "observe", state.version_id, False

    # No active state shares an overlapping window -- a new, co-existing
    # temporal instance of the same topic, not a competitor to any of them.
    return "separate", None, False


def _best_temporal_match(active: list[MemoryStateVersion], assertion: Assertion) -> Optional[MemoryStateVersion]:
    for state in active:
        if _temporal_windows_overlap(assertion.effective_from, assertion.effective_to,
                                      state.effective_from, state.effective_to):
            return state
    return None


def apply_assertion(
    registry: dict[CellKey, MemoryCell], assertion: Assertion, *, recorded_at: str,
) -> Optional[MemoryStateVersion | MemoryObservation]:
    """Deterministically applies one validated, already topic-resolved
    assertion (mutates registry in place). Returns a new MemoryStateVersion
    (decision separate/revise/retract -> evolution-history operation
    create/revise/retract), a MemoryObservation (decision observe/augment,
    attached to the CURRENT state version, never extending the chain), or
    None (decision reject -- a content-free assertion that must never be
    adopted as state at all; the caller should count this distinctly from
    a validate_assertion-time rejection). Callers that only care about
    evolution chains should filter on isinstance(..., MemoryStateVersion)."""
    key = CellKey(
        assertion.network_id, _norm(assertion.viewpoint_owner), _norm(assertion.subject),
        _norm(assertion.facet), assertion.topic_key, assertion.scope_type,
    )
    cell = registry.get(key)
    decision, target_id, blocked = decide_state_operation(cell, assertion)

    if decision == "reject":
        # No registry mutation at all -- in particular, a reject on what
        # WOULD have been a brand-new cell's first assertion must not leave
        # a phantom empty MemoryCell (0 state_versions, 0 observations)
        # behind in the registry.
        return None

    if cell is None:
        cell = MemoryCell(cell_id=_cell_id(key), key=key)
        registry[key] = cell

    if decision in ("observe", "augment"):
        observation = MemoryObservation(
            observation_id=f"{cell.cell_id}:o{len(cell.observations) + 1}", cell_id=cell.cell_id,
            state_version_id=target_id, assertion_text=assertion.assertion_text,
            modality=assertion.modality, confidence=assertion.confidence, observed_at=assertion.observed_at,
            recorded_at=recorded_at, conversation_id=assertion.conversation_id,
            source_turn_ids=assertion.source_turn_ids, observation_kind=decision,
        )
        cell.observations.append(observation)
        # Reinforce the target state's own confidence/evidence without
        # extending the evolution chain -- replace it in place (same
        # version_id/version_no, richer source_turn_ids and confidence).
        for index, state in enumerate(cell.state_versions):
            if state.version_id == target_id:
                cell.state_versions[index] = replace(
                    state,
                    source_turn_ids=tuple(dict.fromkeys(state.source_turn_ids + assertion.source_turn_ids)),
                    confidence=max(state.confidence, assertion.confidence),
                )
                break
        return observation

    # Only separate/revise/retract ever reach the evolution history --
    # "separate" is the history-facing name "create" (item D's internal
    # decision vocabulary is deliberately finer-grained than the three
    # operations create/revise/retract that state_versions may contain).
    operation = "create" if decision == "separate" else decision
    version_no = len(cell.state_versions) + 1
    carried_value = None
    if blocked and target_id is not None:
        target = next((v for v in cell.state_versions if v.version_id == target_id), None)
        carried_value = target.normalized_value if target else assertion.normalized_value
    version = MemoryStateVersion(
        version_id=f"{cell.cell_id}:v{version_no}", cell_id=cell.cell_id, version_no=version_no,
        operation=operation,
        assertion_text=assertion.assertion_text,
        normalized_value=(carried_value if blocked else assertion.normalized_value),
        modality=assertion.modality, confidence=assertion.confidence, observed_at=assertion.observed_at,
        effective_from=assertion.effective_from, effective_to=assertion.effective_to,
        temporal_scope=assertion.temporal_scope, temporal_precision=assertion.temporal_precision,
        recorded_at=recorded_at, conversation_id=assertion.conversation_id,
        source_turn_ids=assertion.source_turn_ids, blocked_by_guard=blocked,
    )
    cell.state_versions.append(version)

    if blocked:
        return version

    if operation in ("revise", "retract") and target_id is not None:
        for index, state in enumerate(cell.state_versions):
            if state.version_id == target_id:
                cell.state_versions[index] = replace(state, closed_at=assertion.observed_at, superseded_by=version.version_id)
                break
        if target_id in cell.active_state_version_ids:
            cell.active_state_version_ids.remove(target_id)

    if operation != "retract":
        cell.active_state_version_ids.append(version.version_id)
    return version


def resolve_topic_slot(
    new_topic_key: str, new_embedding: Sequence[float],
    candidates: list[tuple[str, Sequence[float]]],
) -> Optional[str]:
    """Given a proposed topic_key + its embedding, and the (topic_key,
    embedding) pairs of EXISTING cells sharing the same (network_id,
    viewpoint_owner, subject, facet, scope_type), decides whether the new
    assertion belongs to one of those existing topics or is genuinely new.
    Pure function -- no embedding computation happens here, the caller
    supplies precomputed vectors. Exact topic_key string match short-
    circuits (cheap, unambiguous); otherwise the closest existing topic by
    cosine similarity is accepted only above TOPIC_MATCH_THRESHOLD. Returns
    None (keep new_topic_key as its own new slot) when nothing clears the
    bar -- false split over false merge."""
    existing_keys = {key for key, _ in candidates}
    if new_topic_key in existing_keys:
        return new_topic_key
    if not candidates:
        return None
    best_key, best_score = None, 0.0
    for topic_key, embedding in candidates:
        score = _cosine(new_embedding, embedding)
        if score > best_score:
            best_key, best_score = topic_key, score
    return best_key if best_score >= TOPIC_MATCH_THRESHOLD else None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def active_state_versions(cell: MemoryCell, active_depth: Optional[int]) -> list[MemoryStateVersion]:
    """The last active_depth STATE versions only (observations never count
    toward chain depth) -- active_depth=None means "all". Read-only view;
    never mutates or drops anything from cell.state_versions."""
    if active_depth is None or active_depth <= 0:
        return list(cell.state_versions)
    return cell.state_versions[-active_depth:]


@dataclass(frozen=True)
class HistoryDigest:
    text: str
    covers_version_ids: tuple[str, ...]
    source_turn_ids: tuple[str, ...]
    time_range: tuple[str, str]


def compute_history_digest(cell: MemoryCell, active_depth: Optional[int]) -> Optional[HistoryDigest]:
    """State versions older than active_depth, summarized but never
    discarded -- cell.state_versions keeps every one of them. Text is a
    deterministic concatenation here (no LLM call in this module); an
    ingestion pipeline may replace it with an LLM summary later without
    changing this function's contract."""
    if active_depth is None or active_depth <= 0 or len(cell.state_versions) <= active_depth:
        return None
    covered = cell.state_versions[:-active_depth]
    if not covered:
        return None
    source_ids = tuple(dict.fromkeys(source_id for version in covered for source_id in version.source_turn_ids))
    times = [version.observed_at for version in covered if version.observed_at]
    return HistoryDigest(
        text="; ".join(version.assertion_text for version in covered),
        covers_version_ids=tuple(version.version_id for version in covered),
        source_turn_ids=source_ids,
        time_range=(min(times) if times else "", max(times) if times else ""),
    )


def scope_specificity(scope_type: str) -> int:
    return SCOPE_SPECIFICITY.get(scope_type, 1)


def rank_by_scope_precedence(cells: list[MemoryCell]) -> list[MemoryCell]:
    """Individual-scope cells sort before group-scope ones for the same
    facet -- used at retrieval time to prefer a documented individual
    exception over a general group norm, WITHOUT a graph edge between them
    (they are simply two different cells, ordered by specificity)."""
    return sorted(cells, key=lambda cell: scope_specificity(cell.key.scope_type), reverse=True)


def state_chain(cell: MemoryCell) -> list[MemoryStateVersion]:
    """All state versions in chronological order -- trivial today (append-
    only list is already ordered), kept as a named accessor so retrieval
    code has one stable place to ask "give me this cell's evolution" rather
    than reaching into cell.state_versions directly."""
    return list(cell.state_versions)
