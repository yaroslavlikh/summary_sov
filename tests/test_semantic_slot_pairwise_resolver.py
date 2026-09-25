import unittest

from research.semantic_slot_pairwise_resolver import (
    calibrate_threshold,
    normalize_pairwise_decisions,
    query_outcome,
    select_candidate,
)


class PairwiseResolverTest(unittest.TestCase):
    def test_invalid_indices_are_ignored(self):
        output = {"decisions": [
            {"candidate_index": 0, "verdict": "SAME_SLOT", "confidence": 0.9},
            {"candidate_index": 9, "verdict": "SAME_SLOT", "confidence": 1},
            {"candidate_index": 1, "verdict": "MAYBE", "confidence": 1},
        ]}
        self.assertEqual(set(normalize_pairwise_decisions(output, 2)), {0})

    def test_different_confidence_becomes_same_probability(self):
        output = {"decisions": [{"candidate_index": 0, "verdict": "DIFFERENT_SLOT", "confidence": 0.8}]}
        self.assertAlmostEqual(normalize_pairwise_decisions(output, 1)[0]["same_score"], 0.2)

    def test_selection_uses_score_then_retrieval_rank(self):
        rows = [
            {"candidate_id": "a", "same_score": 0.8, "rank": 2},
            {"candidate_id": "b", "same_score": 0.8, "rank": 1},
        ]
        self.assertEqual(select_candidate(rows, 0.7), "b")
        self.assertIsNone(select_candidate(rows, 0.9))

    def test_outcomes_are_mutually_exclusive(self):
        row = {"qa_id": "Q1", "candidates": [
            {"candidate_id": "Q1:old", "same_score": 0.4, "rank": 1},
            {"candidate_id": "Q2:old", "same_score": 0.9, "rank": 2},
        ]}
        row["network_id"] = "net1"
        self.assertEqual(query_outcome(row, 0.8, {("net1", "Q1:new", "Q2:old")}), "false_merge")
        self.assertEqual(query_outcome(row, 0.8, set()), "false_merge")
        self.assertEqual(query_outcome(row, 0.95, set()), "false_split")

    def test_calibration_honors_false_merge_constraint(self):
        rows = [
            {"qa_id": "Q1", "candidates": [
                {"candidate_id": "Q1:old", "same_score": 0.8, "rank": 1},
                {"candidate_id": "Q2:old", "same_score": 0.9, "rank": 2},
            ]},
            {"qa_id": "Q3", "candidates": [
                {"candidate_id": "Q3:old", "same_score": 0.85, "rank": 1},
            ]},
        ]
        for row in rows:
            row["network_id"] = "net1"
        threshold, _ = calibrate_threshold(rows, {("net1", "Q1:new", "Q2:old")}, max_false_merge_rate=0.0)
        self.assertGreater(threshold, 0.9)


if __name__ == "__main__":
    unittest.main()
