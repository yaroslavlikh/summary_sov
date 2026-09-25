"""Tests for research/stage4_6_oracle_linking.py -- pure logic only, no LLM
calls, no DB. apply_assertion_semantic's LLM-calling dependency
(run_semantic_resolver) is monkeypatched for the mutation-logic tests."""
import unittest
from unittest.mock import patch

from research.stage4_6_oracle_linking import (
    _evaluate_linked_cells, _parse_semantic_decision, _semantic_candidate_filter,
    apply_assertion_semantic,
)
from research.versioned_memory_cells import Assertion, CellKey, MemoryCell, MemoryStateVersion


def _assertion(**overrides):
    base = dict(
        network_id="sov", viewpoint_owner="Sam", subject="Sam", facet="health_status",
        topic_key="knee", scope_type="individual", assertion_text="Sam says his knee is fine",
        modality="self_report", confidence=0.9, observed_at="2026-01-01", effective_from="2026-01-01",
        source_turn_ids=("t1",), normalized_value="fine",
    )
    base.update(overrides)
    return Assertion(**base)


class ParseSemanticDecisionTest(unittest.TestCase):
    def test_valid_new_cell_no_target_needed(self):
        decision, target, conf, reason, defaulted = _parse_semantic_decision(
            {"decision": "NEW_CELL", "target": None, "confidence": 0.9, "reason": "different topic"}, 2,
        )
        self.assertEqual(decision, "NEW_CELL")
        self.assertIsNone(target)
        self.assertFalse(defaulted)

    def test_valid_revise_with_target(self):
        decision, target, conf, reason, defaulted = _parse_semantic_decision(
            {"decision": "revise", "target": "B", "confidence": 0.8, "reason": "state changed"}, 3,
        )
        self.assertEqual(decision, "REVISE")
        self.assertEqual(target, 1)
        self.assertFalse(defaulted)

    def test_invalid_decision_string_defaults_to_new_cell(self):
        decision, target, conf, reason, defaulted = _parse_semantic_decision(
            {"decision": "MAYBE", "target": "A", "confidence": 0.9}, 1,
        )
        self.assertEqual(decision, "NEW_CELL")
        self.assertIsNone(target)
        self.assertTrue(defaulted)

    def test_target_letter_out_of_candidate_range_defaults_to_new_cell(self):
        decision, target, conf, reason, defaulted = _parse_semantic_decision(
            {"decision": "REVISE", "target": "C", "confidence": 0.9}, 1,  # only 1 candidate, C is out of range
        )
        self.assertEqual(decision, "NEW_CELL")
        self.assertTrue(defaulted)

    def test_non_new_cell_without_target_defaults_to_new_cell(self):
        decision, target, conf, reason, defaulted = _parse_semantic_decision(
            {"decision": "OBSERVE", "target": None, "confidence": 0.9}, 2,
        )
        self.assertEqual(decision, "NEW_CELL")
        self.assertTrue(defaulted)

    def test_low_confidence_defaults_to_new_cell(self):
        decision, target, conf, reason, defaulted = _parse_semantic_decision(
            {"decision": "REVISE", "target": "A", "confidence": 0.2}, 1,
        )
        self.assertEqual(decision, "NEW_CELL")
        self.assertIsNone(target)
        self.assertTrue(defaulted)

    def test_malformed_confidence_treated_as_zero(self):
        decision, target, conf, reason, defaulted = _parse_semantic_decision(
            {"decision": "NEW_CELL", "confidence": "not_a_number"}, 0,
        )
        self.assertEqual(conf, 0.0)


class SemanticCandidateFilterTest(unittest.TestCase):
    def test_filters_by_full_bucket_key(self):
        registry = {}
        match_key = CellKey("sov", "sam", "sam", "health_status", "old_knee_topic", "individual")
        other_network = CellKey("other", "sam", "sam", "health_status", "x", "individual")
        other_subject = CellKey("sov", "sam", "jones", "health_status", "x", "individual")
        other_facet = CellKey("sov", "sam", "sam", "location", "x", "individual")
        for key in (match_key, other_network, other_subject, other_facet):
            registry[key] = MemoryCell(cell_id=key.topic_key, key=key)

        candidates = _semantic_candidate_filter(registry, _assertion())
        self.assertEqual([c.key for c in candidates], [match_key])


class EvaluateLinkedCellsTest(unittest.TestCase):
    def _cell_with_versions(self, versions):
        key = CellKey("sov", "sam", "sam", "health_status", "knee", "individual")
        cell = MemoryCell(cell_id="c1", key=key, state_versions=versions,
                           active_state_version_ids=[versions[-1].version_id] if versions else [])
        return cell

    def _version(self, version_no, source_ids, operation="create"):
        return MemoryStateVersion(
            version_id=f"c1:v{version_no}", cell_id="c1", version_no=version_no, operation=operation,
            assertion_text="x", normalized_value="v", modality="asserted", confidence=0.9,
            observed_at="2026-01-01", effective_from="2026-01-01", effective_to=None,
            temporal_scope="unspecified", temporal_precision="unknown", recorded_at="r",
            conversation_id="c", source_turn_ids=tuple(source_ids),
        )

    def test_co_cell_and_ordered_and_revision_recall_all_true(self):
        old_v = self._version(1, ["old1"], operation="create")
        new_v = self._version(2, ["new1"], operation="revise")
        cell = self._cell_with_versions([old_v, new_v])
        result = _evaluate_linked_cells([cell], {"old1"}, {"new1"})
        self.assertTrue(result["co_cell_recall"])
        self.assertTrue(result["ordered_chain_recall"])
        self.assertTrue(result["revision_recall"])
        self.assertEqual(result["new_state_operation"], "revise")

    def test_collapsed_into_one_version_is_not_ordered(self):
        # both old and new sources merged into the SAME single version --
        # co_cell_recall true, but there's no separate old->new ordering.
        merged_v = self._version(1, ["old1", "new1"], operation="create")
        cell = self._cell_with_versions([merged_v])
        result = _evaluate_linked_cells([cell], {"old1"}, {"new1"})
        self.assertTrue(result["co_cell_recall"])
        self.assertFalse(result["ordered_chain_recall"])
        self.assertFalse(result["revision_recall"])

    def test_different_cells_no_co_cell_recall(self):
        cell_a = self._cell_with_versions([self._version(1, ["old1"])])
        key_b = CellKey("sov", "sam", "sam", "health_status", "knee_v2", "individual")
        cell_b = MemoryCell(cell_id="c2", key=key_b, state_versions=[self._version(1, ["new1"])])
        result = _evaluate_linked_cells([cell_a, cell_b], {"old1"}, {"new1"})
        self.assertFalse(result["co_cell_recall"])

    def test_ordered_but_not_revise_operation_is_not_revision_recall(self):
        old_v = self._version(1, ["old1"], operation="create")
        new_v = self._version(2, ["new1"], operation="create")  # coexisting, not a revise
        cell = self._cell_with_versions([old_v, new_v])
        result = _evaluate_linked_cells([cell], {"old1"}, {"new1"})
        self.assertTrue(result["ordered_chain_recall"])
        self.assertFalse(result["revision_recall"])


class ApplyAssertionSemanticTest(unittest.TestCase):
    def test_new_cell_with_no_candidates_needs_no_llm_call(self):
        registry = {}
        version, info = apply_assertion_semantic(registry, _assertion(), recorded_at="t0")
        self.assertEqual(info["decision"], "NEW_CELL")
        self.assertFalse(info["defaulted"])
        self.assertEqual(len(registry), 1)

    _NEW_CELL_INFO = {
        "decision": "NEW_CELL", "target_cell_id": None, "confidence": 1.0,
        "reason": "no candidate cells", "defaulted": False, "candidates_seen": [],
    }

    @patch("research.stage4_6_oracle_linking.run_semantic_resolver")
    def test_revise_closes_prior_active_version_and_appends_new_one(self, mock_resolver):
        registry = {}
        # First call has an empty candidate pool (fresh registry) -- the
        # real run_semantic_resolver would short-circuit to NEW_CELL itself
        # without an LLM call; since the whole function is mocked here for
        # the SECOND call's controlled REVISE, the first call must be
        # scripted to match that same real short-circuit behavior.
        mock_resolver.side_effect = [dict(self._NEW_CELL_INFO)]
        v1, _ = apply_assertion_semantic(registry, _assertion(normalized_value="fine"), recorded_at="t0")
        target_cell_id = v1.cell_id
        mock_resolver.side_effect = [{
            "decision": "REVISE", "target_cell_id": target_cell_id, "confidence": 0.9,
            "reason": "knee got worse", "defaulted": False, "candidates_seen": [target_cell_id],
        }]
        v2, info = apply_assertion_semantic(registry, _assertion(
            topic_key="knee_pain", normalized_value="painful", source_turn_ids=("t2",),
        ), recorded_at="t1")
        self.assertEqual(info["decision"], "REVISE")
        cell = next(iter(registry.values()))
        self.assertEqual(len(cell.state_versions), 2)
        self.assertIsNotNone(cell.state_versions[0].closed_at)
        self.assertEqual(cell.active_state_version_ids, [v2.version_id])

    @patch("research.stage4_6_oracle_linking.run_semantic_resolver")
    def test_observe_attaches_without_extending_chain(self, mock_resolver):
        registry = {}
        mock_resolver.side_effect = [dict(self._NEW_CELL_INFO)]
        v1, _ = apply_assertion_semantic(registry, _assertion(), recorded_at="t0")
        mock_resolver.side_effect = [{
            "decision": "OBSERVE", "target_cell_id": v1.cell_id, "confidence": 0.9,
            "reason": "same state repeated", "defaulted": False, "candidates_seen": [v1.cell_id],
        }]
        result, info = apply_assertion_semantic(registry, _assertion(source_turn_ids=("t2",)), recorded_at="t1")
        cell = next(iter(registry.values()))
        self.assertEqual(len(cell.state_versions), 1)
        self.assertEqual(len(cell.observations), 1)

    @patch("research.stage4_6_oracle_linking.run_semantic_resolver")
    def test_retract_closes_active_version_without_reactivating(self, mock_resolver):
        registry = {}
        mock_resolver.side_effect = [dict(self._NEW_CELL_INFO)]
        v1, _ = apply_assertion_semantic(registry, _assertion(), recorded_at="t0")
        mock_resolver.side_effect = [{
            "decision": "RETRACT", "target_cell_id": v1.cell_id, "confidence": 0.9,
            "reason": "explicitly withdrawn", "defaulted": False, "candidates_seen": [v1.cell_id],
        }]
        v2, info = apply_assertion_semantic(registry, _assertion(source_turn_ids=("t2",)), recorded_at="t1")
        cell = next(iter(registry.values()))
        self.assertEqual(v2.operation, "retract")
        self.assertEqual(cell.active_state_version_ids, [])


if __name__ == "__main__":
    unittest.main()
