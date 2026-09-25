"""Tests for research/versioned_memory_cells.py v1 (topic-scoped identity) --
pure model, no DB, no LLM. Covers all 10 required cases from the v1
correction plus the modality-guard/attribution/scope-precedence properties
carried over from v0 (still valid behaviors of the new model)."""
import unittest

from research.versioned_memory_cells import (
    Assertion, CellKey, MemoryCell, MemoryObservation, MemoryStateVersion,
    active_state_versions, apply_assertion, compute_history_digest, decide_state_operation,
    rank_by_scope_precedence, resolve_topic_slot, state_chain, validate_assertion,
)


def _assertion(**overrides):
    base = dict(
        network_id="sov", viewpoint_owner="Тигмен", subject="Гордей", facet="attitude",
        topic_key="hike", scope_type="individual", assertion_text="Тигмен думает поход был плохим",
        modality="asserted", confidence=0.9, observed_at="2026-03-01", effective_from="2026-03-01",
        source_turn_ids=("t1",), normalized_value="negative",
    )
    base.update(overrides)
    return Assertion(**base)


class VersionedMemoryCellsTest(unittest.TestCase):
    # ---- 1. two attitude assertions about DIFFERENT objects get different cells ----
    def test_different_topic_creates_different_cell(self):
        registry = {}
        apply_assertion(registry, _assertion(topic_key="hike", normalized_value="negative"), recorded_at="t0")
        apply_assertion(registry, _assertion(
            topic_key="eggs", assertion_text="Тигмен думает яйца были хорошими",
            normalized_value="positive", source_turn_ids=("t2",),
        ), recorded_at="t1")
        self.assertEqual(len(registry), 2)
        keys = {key.topic_key for key in registry}
        self.assertEqual(keys, {"hike", "eggs"})
        # the hike opinion must NOT have been overwritten by the eggs opinion
        hike_cell = registry[CellKey("sov", "тигмен", "гордей", "attitude", "hike", "individual")]
        self.assertEqual(hike_cell.active_states[0].normalized_value, "negative")

    # ---- 2. two commitments to DIFFERENT events get different cells ----
    def test_different_events_create_different_commitment_cells(self):
        registry = {}
        apply_assertion(registry, _assertion(
            facet="commitment", topic_key="ridgeline_run", assertion_text="confirmed for Ridgeline run",
            modality="self_report", normalized_value="confirmed",
        ), recorded_at="t0")
        apply_assertion(registry, _assertion(
            facet="commitment", topic_key="camping_weekend", assertion_text="will attend camping weekend",
            modality="commitment", normalized_value="confirmed", source_turn_ids=("t2",),
        ), recorded_at="t1")
        self.assertEqual(len(registry), 2)

    # ---- 3. the SAME position repeated later creates an observation, not a new state ----
    def test_repeated_same_state_is_observation_not_new_version(self):
        registry = {}
        v1 = apply_assertion(registry, _assertion(), recorded_at="t0")
        result = apply_assertion(registry, _assertion(
            assertion_text="Тигмен снова говорит поход был плохим", observed_at="2026-03-05",
            effective_from="2026-03-05", source_turn_ids=("t2",),
        ), recorded_at="t1")
        self.assertIsInstance(result, MemoryObservation)
        cell = next(iter(registry.values()))
        self.assertEqual(len(cell.state_versions), 1)  # no new state version
        self.assertEqual(len(cell.observations), 1)

    # ---- 4. changed normalized_value on the SAME topic (overlapping window) creates a revise ----
    def test_value_change_same_topic_overlapping_window_is_revise(self):
        registry = {}
        v1 = apply_assertion(registry, _assertion(effective_from="2026-03-01", effective_to=None), recorded_at="t0")
        v2 = apply_assertion(registry, _assertion(
            assertion_text="Тигмен передумал, поход был хорошим", normalized_value="positive",
            effective_from="2026-03-01", effective_to=None, observed_at="2026-03-10", source_turn_ids=("t2",),
        ), recorded_at="t1")
        self.assertIsInstance(v2, MemoryStateVersion)
        self.assertEqual(v2.operation, "revise")
        cell = next(iter(registry.values()))
        closed_v1 = next(v for v in cell.state_versions if v.version_id == v1.version_id)
        self.assertIsNotNone(closed_v1.closed_at)
        self.assertEqual(closed_v1.superseded_by, v2.version_id)
        self.assertEqual([s.version_id for s in cell.active_states], [v2.version_id])

    # ---- 5. non-overlapping temporal windows coexist, neither closes the other ----
    def test_non_overlapping_windows_coexist(self):
        registry = {}
        v1 = apply_assertion(registry, _assertion(
            facet="availability", topic_key="trip_availability", normalized_value="busy",
            assertion_text="Nadia is busy in October", effective_from="2026-10-01", effective_to="2026-10-31",
        ), recorded_at="t0")
        v2 = apply_assertion(registry, _assertion(
            facet="availability", topic_key="trip_availability", normalized_value="free",
            assertion_text="Nadia is free in December", effective_from="2026-12-01", effective_to="2026-12-31",
            observed_at="2026-11-01", source_turn_ids=("t2",),
        ), recorded_at="t1")
        self.assertIsInstance(v2, MemoryStateVersion)
        self.assertEqual(v2.operation, "create")  # coexists, doesn't revise v1
        cell = next(iter(registry.values()))
        self.assertIsNone(cell.state_versions[0].closed_at)  # v1 never closed
        self.assertEqual(set(cell.active_state_version_ids), {v1.version_id, v2.version_id})

    # ---- 6. reaffirm (observation) does not increase evolution chain depth ----
    def test_observation_does_not_extend_chain(self):
        registry = {}
        apply_assertion(registry, _assertion(), recorded_at="t0")
        for i in range(5):
            apply_assertion(registry, _assertion(
                assertion_text=f"reaffirm {i}", observed_at=f"2026-03-{i+2:02d}",
                effective_from=f"2026-03-{i+2:02d}", source_turn_ids=(f"t{i+2}",),
            ), recorded_at=f"r{i}")
        cell = next(iter(registry.values()))
        self.assertEqual(len(cell.state_versions), 1)
        self.assertEqual(len(cell.observations), 5)

    # ---- 7. active depth counts only state versions, not observations ----
    def test_active_depth_counts_state_versions_only(self):
        registry = {}
        for i in range(7):
            apply_assertion(registry, _assertion(
                assertion_text=f"state {i}", normalized_value=f"value_{i}",
                observed_at=f"2026-0{(i % 9) + 1}-01", effective_from=f"2026-0{(i % 9) + 1}-01",
                source_turn_ids=(f"t{i}",),
            ), recorded_at=f"r{i}")
        apply_assertion(registry, _assertion(
            assertion_text="a mere reaffirm", normalized_value="value_6",
            observed_at="2026-09-01", effective_from="2026-01-01", source_turn_ids=("t9",),
        ), recorded_at="robs")
        cell = next(iter(registry.values()))
        self.assertEqual(len(cell.state_versions), 7)
        active = active_state_versions(cell, active_depth=5)
        self.assertEqual(len(active), 5)  # observation did not inflate this count

    # ---- 8. observation provenance is not lost -- merges into the reinforced state ----
    def test_observation_provenance_merges_into_state(self):
        registry = {}
        apply_assertion(registry, _assertion(source_turn_ids=("t1",)), recorded_at="t0")
        apply_assertion(registry, _assertion(
            assertion_text="reaffirm", observed_at="2026-03-05", effective_from="2026-03-05",
            source_turn_ids=("t2",),
        ), recorded_at="t1")
        cell = next(iter(registry.values()))
        state = cell.state_versions[0]
        self.assertEqual(set(state.source_turn_ids), {"t1", "t2"})

    # ---- 9. an uncertain topic match safely becomes a split, not a merge ----
    def test_uncertain_topic_match_defaults_to_split(self):
        # Below threshold -- resolve_topic_slot must refuse to merge.
        result = resolve_topic_slot(
            "new_topic", [1.0, 0.0, 0.0],
            candidates=[("existing_topic", [0.0, 1.0, 0.0])],  # orthogonal -- cosine 0.0
        )
        self.assertIsNone(result)

    def test_confident_topic_match_merges(self):
        result = resolve_topic_slot(
            "camping_weekend_v2", [1.0, 0.0, 0.0],
            candidates=[("camping_weekend", [0.99, 0.01, 0.0])],
        )
        self.assertEqual(result, "camping_weekend")

    def test_exact_topic_key_match_short_circuits(self):
        result = resolve_topic_slot(
            "camping_weekend", [1.0, 0.0, 0.0],
            candidates=[("camping_weekend", [0.0, 1.0, 0.0])],  # embedding irrelevant, string already matches
        )
        self.assertEqual(result, "camping_weekend")

    # ---- carried over from v0: modality guard blocks weak-modality revisions ----
    def test_joke_does_not_close_serious_state(self):
        registry = {}
        v1 = apply_assertion(registry, _assertion(modality="asserted"), recorded_at="t0")
        v2 = apply_assertion(registry, _assertion(
            assertion_text="в шутку сказал поход был отличным", normalized_value="positive",
            modality="joke", effective_from="2026-03-01", observed_at="2026-03-10", source_turn_ids=("t2",),
        ), recorded_at="t1")
        self.assertTrue(v2.blocked_by_guard)
        self.assertEqual(v2.normalized_value, "negative")  # carried over
        cell = next(iter(registry.values()))
        self.assertEqual(cell.active_state_version_ids, [v1.version_id])  # unchanged

    def test_reported_by_other_keeps_viewpoint_distinct_from_subject(self):
        registry = {}
        version = apply_assertion(registry, _assertion(
            viewpoint_owner="Саша", subject="Ваня", facet="opinion_of_person", topic_key="ksyusha_character",
            assertion_text="По словам Саши, Ваня сказал что Ксюша конченная", modality="reported_by_other",
        ), recorded_at="t0")
        cell = next(iter(registry.values()))
        self.assertEqual(cell.key.viewpoint_owner, "саша")
        self.assertEqual(cell.key.subject, "ваня")
        self.assertEqual(version.modality, "reported_by_other")

    def test_individual_scope_ranked_above_group_scope(self):
        individual = MemoryCell(cell_id="c1", key=CellKey("sov", "group", "вася", "policy", "contacting_council", "individual"))
        group = MemoryCell(cell_id="c2", key=CellKey("sov", "group", "group", "policy", "contacting_council", "group"))
        ranked = rank_by_scope_precedence([group, individual])
        self.assertEqual(ranked[0].cell_id, "c1")

    def test_state_chain_is_chronologically_ordered(self):
        registry = {}
        v1 = apply_assertion(registry, _assertion(), recorded_at="t0")
        v2 = apply_assertion(registry, _assertion(
            assertion_text="revise", normalized_value="positive", observed_at="2026-04-01",
            effective_from="2026-04-01", source_turn_ids=("t2",),
        ), recorded_at="t1")
        cell = next(iter(registry.values()))
        chain = state_chain(cell)
        self.assertEqual([v.version_id for v in chain], [v1.version_id, v2.version_id])

    def test_history_digest_unions_source_turn_ids(self):
        registry = {}
        for i in range(7):
            apply_assertion(registry, _assertion(
                assertion_text=f"state {i}", normalized_value=f"value_{i}",
                observed_at=f"2026-0{(i % 9) + 1}-01", effective_from=f"2026-0{(i % 9) + 1}-01",
                source_turn_ids=(f"t{i}",),
            ), recorded_at=f"r{i}")
        cell = next(iter(registry.values()))
        digest = compute_history_digest(cell, active_depth=5)
        self.assertIsNotNone(digest)
        self.assertEqual(set(digest.source_turn_ids), {"t0", "t1"})

    def test_fabricated_source_turn_id_is_dropped(self):
        valid_ids = {"t1"}
        accepted = validate_assertion({
            "viewpoint_owner": "Тигмен", "subject": "Гордей", "facet": "attitude", "topic_key": "hike",
            "assertion_text": "Тигмен считает поход плохим", "modality": "opinion",
            "normalized_value": "negative", "source_turn_ids": ["t1", "fake"],
        }, valid_ids, "sov")
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted.source_turn_ids, ("t1",))

        rejected = validate_assertion({
            "viewpoint_owner": "Тигмен", "subject": "Гордей", "facet": "attitude", "topic_key": "hike",
            "assertion_text": "Выдуманное", "source_turn_ids": ["fake_only"],
        }, valid_ids, "sov")
        self.assertIsNone(rejected)

    def test_missing_topic_key_is_rejected(self):
        rejected = validate_assertion({
            "viewpoint_owner": "Тигмен", "subject": "Гордей", "facet": "attitude",
            "assertion_text": "no topic key at all", "source_turn_ids": ["t1"],
        }, {"t1"}, "sov")
        self.assertIsNone(rejected)

    def test_decide_state_operation_is_pure_and_stateless(self):
        result_a = decide_state_operation(None, _assertion())
        result_b = decide_state_operation(None, _assertion())
        self.assertEqual(result_a, result_b)

    # ---- Stage 4.5 item D, rule 1: shared source_turn_id can never revise ----
    def test_shared_source_turn_id_cannot_revise_even_if_value_differs(self):
        registry = {}
        v1 = apply_assertion(registry, _assertion(source_turn_ids=("t1", "t2")), recorded_at="t0")
        result = apply_assertion(registry, _assertion(
            assertion_text="доп. деталь про тот же случай", normalized_value="positive",
            source_turn_ids=("t2",),  # subset already counted toward v1 -- no new evidence
        ), recorded_at="t1")
        self.assertIsInstance(result, MemoryObservation)
        self.assertEqual(result.observation_kind, "observe")
        cell = next(iter(registry.values()))
        self.assertEqual(len(cell.state_versions), 1)
        self.assertEqual(cell.state_versions[0].normalized_value, "negative")  # unchanged

    def test_shared_source_with_new_evidence_is_augment_not_revise(self):
        registry = {}
        apply_assertion(registry, _assertion(source_turn_ids=("t1",)), recorded_at="t0")
        result = apply_assertion(registry, _assertion(
            assertion_text="доп. деталь плюс новый источник", normalized_value="positive",
            source_turn_ids=("t1", "t2"),  # partial overlap: t1 already counted, t2 is new
        ), recorded_at="t1")
        self.assertIsInstance(result, MemoryObservation)
        self.assertEqual(result.observation_kind, "augment")
        cell = next(iter(registry.values()))
        self.assertEqual(len(cell.state_versions), 1)  # still did not close/replace v1

    # ---- Stage 4.5 item D, rule 2: spelling/whitespace is not a value change ----
    def test_normalized_value_spelling_difference_is_not_a_revise(self):
        registry = {}
        apply_assertion(registry, _assertion(normalized_value="Negative "), recorded_at="t0")
        result = apply_assertion(registry, _assertion(
            assertion_text="снова про то же", normalized_value=" NEGATIVE",
            source_turn_ids=("t2",),
        ), recorded_at="t1")
        self.assertIsInstance(result, MemoryObservation)
        cell = next(iter(registry.values()))
        self.assertEqual(len(cell.state_versions), 1)

    # ---- Stage 4.5 item D, rule 3: no structured value can't close a valued state ----
    def test_assertion_without_normalized_value_augments_not_revises(self):
        registry = {}
        apply_assertion(registry, _assertion(normalized_value="plan"), recorded_at="t0")
        result = apply_assertion(registry, _assertion(
            assertion_text="дополнительная деталь плана", normalized_value=None,
            source_turn_ids=("t2",),
        ), recorded_at="t1")
        self.assertIsInstance(result, MemoryObservation)
        self.assertEqual(result.observation_kind, "augment")
        cell = next(iter(registry.values()))
        self.assertEqual(len(cell.state_versions), 1)
        self.assertEqual(cell.state_versions[0].normalized_value, "plan")  # never closed

    # ---- Stage 4.5 item D, rule 5: content-free "X revised" is rejected ----
    def test_meta_value_assertion_is_rejected_not_adopted(self):
        registry = {}
        apply_assertion(registry, _assertion(normalized_value="tram_preferred"), recorded_at="t0")
        result = apply_assertion(registry, _assertion(
            assertion_text="Derek revised his position", normalized_value="revised",
            source_turn_ids=("t2",),
        ), recorded_at="t1")
        self.assertIsNone(result)
        cell = next(iter(registry.values()))
        self.assertEqual(len(cell.state_versions), 1)  # nothing adopted
        self.assertEqual(cell.state_versions[0].normalized_value, "tram_preferred")

    # ---- Stage 4.5 evaluation-fix D-a: rule 5 applies even with NO prior
    # active state (a brand-new cell's very first assertion) ----
    def test_meta_value_rejected_even_as_first_assertion_for_new_cell(self):
        registry = {}
        result = apply_assertion(registry, _assertion(
            assertion_text="Derek revised his position", normalized_value="revised",
        ), recorded_at="t0")
        self.assertIsNone(result)
        self.assertEqual(len(registry), 0)  # no cell was ever created


if __name__ == "__main__":
    unittest.main()
