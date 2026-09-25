import unittest

from research.semantic_key_grounded_gate import evidence_text
from research.semantic_slot_linking_v2 import NormalizedSlot


class GroundedGateTest(unittest.TestCase):
    def test_evidence_only_uses_validated_source_ids(self):
        slot = NormalizedSlot("q", "new", "n", "A", None, "A", "state", "question", "value", "STATE", 1.0, ("t1", "missing"), "fact")
        rendered = evidence_text(slot, {"t1": {"turn_id": "t1", "timestamp": "now", "speaker": "A", "message": "hello"}})
        self.assertIn("[[t1]]", rendered)
        self.assertNotIn("missing", rendered)


if __name__ == "__main__":
    unittest.main()
