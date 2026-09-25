import unittest

from research.semantic_key_gate import bootstrap_delta, calibrate, calibrate_hybrid, counterfactual_no_match_fpr, key_description, metrics, normalize_gate_output, select_hybrid, select_match
from research.semantic_slot_linking_v2 import NormalizedSlot


def _slot(holder="Tigmen", topic="Gordey", dimension="opinion"):
    return NormalizedSlot(
        case_qa_id="Q1", side="new", network_id="net", state_holder=holder,
        viewpoint_owner=holder, topic_object=topic, state_dimension=dimension,
        slot_question=f"How does {holder} evaluate {topic}?", value="good",
        assertion_type="STATE", normalization_confidence=1,
        source_turn_ids=("t1",), state_text="fact",
    )


class SemanticKeyGateTest(unittest.TestCase):
    def test_key_excludes_value(self):
        self.assertNotIn("good", key_description(_slot()))

    def test_gate_indices_are_bounded(self):
        output = {"decisions": [
            {"candidate_index": 0, "verdict": "MATCH", "confidence": .9},
            {"candidate_index": 3, "verdict": "MATCH", "confidence": 1},
        ]}
        self.assertEqual(set(normalize_gate_output(output, 2)), {0})

    def test_uncertain_never_links(self):
        output = {"decisions": [{"candidate_index": 0, "verdict": "UNCERTAIN", "confidence": 1}]}
        self.assertEqual(normalize_gate_output(output, 1)[0]["match_score"], 0)

    def test_selection_prefers_match_confidence_then_retrieval(self):
        row = {"candidates": [
            {"candidate_id": "a", "match_score": .9, "retrieval_score": .7},
            {"candidate_id": "b", "match_score": .9, "retrieval_score": .8},
        ]}
        self.assertEqual(select_match(row, .8), "b")

    def test_calibration_respects_precision_and_no_match_fpr(self):
        rows = [
            {"qa_id": "Q1", "network_id": "net", "gold_id": "Q1:old", "candidates": [
                {"candidate_id": "Q1:old", "match_score": .95, "retrieval_score": .9},
                {"candidate_id": "Q2:old", "match_score": .8, "retrieval_score": .8},
            ]},
            {"qa_id": "Q3", "network_id": "net", "gold_id": "Q3:old", "candidates": [
                {"candidate_id": "Q4:old", "match_score": .9, "retrieval_score": .9},
            ]},
        ]
        negatives = {("net", "Q1:new", "Q2:old"), ("net", "Q3:new", "Q4:old")}
        threshold, _ = calibrate(rows, negatives)
        result = metrics(rows, threshold, negatives)
        self.assertGreaterEqual(result["precision"], .95)
        self.assertLessEqual(result["confirmed_no_match_fpr"], .05)

    def test_hybrid_falls_back_only_with_score_and_margin(self):
        row = {"candidates": [
            {"candidate_id": "a", "match_score": 0, "retrieval_score": .9},
            {"candidate_id": "b", "match_score": 0, "retrieval_score": .7},
        ]}
        self.assertEqual(select_hybrid(row, .8, .85, .1), "a")
        self.assertIsNone(select_hybrid(row, .8, .95, .1))
        self.assertIsNone(select_hybrid(row, .8, .85, .3))

    def test_hybrid_calibration_keeps_precision_constraint(self):
        rows = [
            {"qa_id": "Q1", "network_id": "net", "gold_id": "Q1:old", "candidates": [
                {"candidate_id": "Q1:old", "match_score": 0, "retrieval_score": .95},
                {"candidate_id": "x", "match_score": 0, "retrieval_score": .5},
            ]},
            {"qa_id": "Q2", "network_id": "net", "gold_id": "Q2:old", "candidates": [
                {"candidate_id": "x", "match_score": 0, "retrieval_score": .7},
                {"candidate_id": "Q2:old", "match_score": 0, "retrieval_score": .6},
            ]},
        ]
        negatives = {("net", "Q1:new", "x"), ("net", "Q2:new", "x")}
        score, margin, _ = calibrate_hybrid(rows, .5, negatives)
        result = __import__("research.semantic_key_gate", fromlist=["hybrid_metrics"]).hybrid_metrics(rows, .5, score, margin)
        self.assertGreaterEqual(result["precision"], .95)
        self.assertLessEqual(counterfactual_no_match_fpr(rows, .5, score, margin, negatives)[2], .05)

    def test_bootstrap_delta_uses_composite_case_identity(self):
        rows = [
            {"network_id": "a", "qa_id": "same", "gold_id": "same:old", "hybrid_selected": "same:old"},
            {"network_id": "b", "qa_id": "same", "gold_id": "same:old", "hybrid_selected": None},
        ]
        delta, low, high = bootstrap_delta(rows, {"a|same": False, "b|same": False}, iterations=100)
        self.assertEqual(delta, .5)
        self.assertLessEqual(low, delta)
        self.assertGreaterEqual(high, delta)


if __name__ == "__main__":
    unittest.main()
