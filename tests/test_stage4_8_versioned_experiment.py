"""Tests for research/stage4_8_versioned_experiment.py -- pure logic only
(cluster grouping, candidate filter, decision parsing, cell mutation,
cause/reaction attachment). No LLM calls."""
import unittest

from research.stage4_8_versioned_experiment import (
    Stage48Cell, apply_decision, build_session_clusters, build_versioned_cells,
    _candidate_filter, _parse_semantic_decision,
)
from research.stage4_7_1_context_extraction import ContextRecordV2


def _record(**overrides):
    base = dict(
        record_type="STATE", claim="Seb's knee is locked", viewpoint_owner="Seb", subject="Seb",
        state_description="knee condition", value="locked", temporal_mode="current",
        from_value=None, to_value=None, related_state_description=None,
        evidence=({"turn_id": "t2", "quote": "locked"},), related_anchor_ids=("t2",),
        source_turn_ids=("t2",), confidence=0.9, observed_at="2026-01-01T00:01:00",
    )
    base.update(overrides)
    return ContextRecordV2(**base)


class BuildSessionClustersTest(unittest.TestCase):
    def test_groups_by_session_index_recovering_middle_anchor(self):
        anchors = [
            {"turn_id": "s04_t012", "session_index": 4},
            {"turn_id": "s04_t018", "session_index": 4},  # previously "unassigned" middle anchor
            {"turn_id": "s06_t008", "session_index": 6},
        ]
        clusters = build_session_clusters(anchors)
        self.assertEqual([c[0] for c in clusters], [4, 6])
        self.assertEqual(clusters[0][1], {"s04_t012", "s04_t018"})
        self.assertEqual(clusters[1][1], {"s06_t008"})

    def test_single_session_is_one_cluster(self):
        anchors = [{"turn_id": "t1", "session_index": 1}, {"turn_id": "t2", "session_index": 1}]
        clusters = build_session_clusters(anchors)
        self.assertEqual(len(clusters), 1)


class CandidateFilterTest(unittest.TestCase):
    def test_matches_on_network_viewpoint_subject_only_no_facet(self):
        registry = {
            "c1": Stage48Cell(cell_id="c1", network_id="net1", viewpoint_owner="seb", subject="seb"),
            "c2": Stage48Cell(cell_id="c2", network_id="net1", viewpoint_owner="seb", subject="marcus"),
            "c3": Stage48Cell(cell_id="c3", network_id="net2", viewpoint_owner="seb", subject="seb"),
        }
        record = _record(viewpoint_owner="Seb", subject="Seb")
        candidates = _candidate_filter(registry, record, "net1")
        self.assertEqual([c.cell_id for c in candidates], ["c1"])


class ParseSemanticDecisionTest(unittest.TestCase):
    def test_valid_revise_with_target(self):
        decision, target, conf, reason, defaulted = _parse_semantic_decision(
            {"decision": "revise", "target": "A", "confidence": 0.9}, 1,
        )
        self.assertEqual(decision, "REVISE")
        self.assertEqual(target, 0)
        self.assertFalse(defaulted)

    def test_low_confidence_defaults_to_new_cell(self):
        decision, target, conf, reason, defaulted = _parse_semantic_decision(
            {"decision": "REVISE", "target": "A", "confidence": 0.1}, 1,
        )
        self.assertEqual(decision, "NEW_CELL")
        self.assertTrue(defaulted)


class ApplyDecisionTest(unittest.TestCase):
    def test_new_cell_creates_first_version(self):
        registry = {}
        apply_decision(registry, _record(), {"decision": "NEW_CELL", "target_cell_id": None}, "net1")
        self.assertEqual(len(registry), 1)
        cell = next(iter(registry.values()))
        self.assertEqual(len(cell.versions), 1)
        self.assertEqual(cell.active_version.operation, "create")

    def test_revise_closes_prior_and_appends(self):
        registry = {}
        apply_decision(registry, _record(), {"decision": "NEW_CELL", "target_cell_id": None}, "net1")
        cell_id = next(iter(registry))
        apply_decision(registry, _record(value="70%", source_turn_ids=("t8",)), {"decision": "REVISE", "target_cell_id": cell_id}, "net1")
        cell = registry[cell_id]
        self.assertEqual(len(cell.versions), 2)
        self.assertTrue(cell.versions[0].closed)
        self.assertFalse(cell.versions[1].closed)
        self.assertEqual(cell.active_version.record.value, "70%")

    def test_observe_does_not_extend_chain(self):
        registry = {}
        apply_decision(registry, _record(), {"decision": "NEW_CELL", "target_cell_id": None}, "net1")
        cell_id = next(iter(registry))
        apply_decision(registry, _record(source_turn_ids=("t9",)), {"decision": "OBSERVE", "target_cell_id": cell_id}, "net1")
        cell = registry[cell_id]
        self.assertEqual(len(cell.versions), 1)
        self.assertEqual(len(cell.observations), 1)

    def test_retract_closes_without_reactivating(self):
        registry = {}
        apply_decision(registry, _record(), {"decision": "NEW_CELL", "target_cell_id": None}, "net1")
        cell_id = next(iter(registry))
        apply_decision(registry, _record(source_turn_ids=("t9",)), {"decision": "RETRACT", "target_cell_id": cell_id}, "net1")
        cell = registry[cell_id]
        self.assertIsNone(cell.active_version)
        self.assertEqual(cell.versions[-1].operation, "retract")


class BuildVersionedCellsTest(unittest.TestCase):
    def test_cause_and_reaction_attach_to_matching_subject_cell(self):
        state = _record(record_type="STATE", viewpoint_owner="Seb", subject="Seb", source_turn_ids=("t1",))
        cause = _record(
            record_type="CAUSE", claim="Coaching caused the injury", viewpoint_owner="Seb", subject="Seb",
            state_description=None, value=None, related_state_description="knee condition", source_turn_ids=("t1",),
        )
        reaction = _record(
            record_type="REACTION", claim="Priyanka is worried", viewpoint_owner="Priyanka", subject="Seb",
            state_description=None, value=None, source_turn_ids=("t3",),
        )
        registry, decisions = build_versioned_cells([state, cause, reaction], "net1")
        self.assertEqual(len(registry), 1)
        cell = next(iter(registry.values()))
        self.assertEqual(len(cell.observations), 2)
        kinds = {o.kind for o in cell.observations}
        self.assertEqual(kinds, {"cause", "reaction"})
        # only STATE/TRANSITION go through the resolver
        self.assertEqual(len(decisions), 1)


if __name__ == "__main__":
    unittest.main()
