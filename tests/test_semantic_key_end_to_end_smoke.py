import unittest

from research.semantic_key_end_to_end_smoke import (
    select_link, validate_key_items,
)
from research.stage4_7_1_context_extraction import ContextRecordV2


def _record(value="locked"):
    return ContextRecordV2(
        record_type="STATE", claim=f"Seb's knee is {value}", viewpoint_owner="Seb",
        subject="Seb", state_description="knee condition", value=value,
        temporal_mode="current", from_value=None, to_value=None,
        related_state_description=None, evidence=({"turn_id": "t1", "quote": value},),
        related_anchor_ids=("t1",), source_turn_ids=("t1",), confidence=1.0,
        observed_at="2026-01-01",
    )


class KeyValidationTest(unittest.TestCase):
    def test_accepts_value_free_key(self):
        output = {"items": [{
            "record_index": 0, "state_holder": "Seb", "viewpoint_owner": None,
            "topic_object": "Seb's knee", "state_dimension": "condition",
            "slot_question": "What is the condition of Seb's knee?",
        }]}
        slots, rejected = validate_key_items(output, [_record()], "q")
        self.assertIn(0, slots)
        self.assertFalse(rejected)

    def test_rejects_value_in_key_question(self):
        output = {"items": [{
            "record_index": 0, "state_holder": "Seb", "viewpoint_owner": None,
            "topic_object": "Seb's knee", "state_dimension": "condition",
            "slot_question": "Why is Seb's knee locked?",
        }]}
        slots, rejected = validate_key_items(output, [_record()], "q")
        self.assertFalse(slots)
        self.assertEqual(rejected["value_leak"], 1)


class PrecisionLinkSelectionTest(unittest.TestCase):
    def test_explicit_gate_match_wins(self):
        candidates = [
            {"candidate_id": "a", "retrieval_score": .7},
            {"candidate_id": "b", "retrieval_score": .69},
        ]
        output = {"decisions": [
            {"candidate_index": 0, "verdict": "NO_MATCH", "confidence": .99},
            {"candidate_index": 1, "verdict": "MATCH", "confidence": .96},
        ]}
        self.assertEqual(select_link(candidates, output), ("b", "llm_gate"))

    def test_embedding_similarity_never_overrides_gate(self):
        candidates = [
            {"candidate_id": "a", "retrieval_score": .999},
            {"candidate_id": "b", "retrieval_score": .100},
        ]
        self.assertEqual(select_link(candidates, {"decisions": []}), (None, "new_cell"))
        no_match = {"decisions": [{
            "candidate_index": 0, "verdict": "NO_MATCH", "confidence": 1.0,
        }]}
        self.assertEqual(select_link(candidates, no_match), (None, "new_cell"))


if __name__ == "__main__":
    unittest.main()
