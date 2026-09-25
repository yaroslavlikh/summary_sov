import unittest

from research.socialmembench_pilot import Candidate
from research.stage4_7_1_context_extraction import ContextRecordV2
from research.stage4_9_hybrid_experiment import pack_context, rebuild_registry


def _record(claim="old", source="t1"):
    return ContextRecordV2(
        record_type="STATE", claim=claim, viewpoint_owner="Seb", subject="Seb",
        state_description="knee condition", value=claim, temporal_mode="current",
        from_value=None, to_value=None, related_state_description=None,
        evidence=({"turn_id": source, "quote": claim},), related_anchor_ids=(source,),
        source_turn_ids=(source,), confidence=1.0, observed_at=source,
    )


class FrozenRegistryTest(unittest.TestCase):
    def test_replays_cached_decisions_without_resolver(self):
        records = [_record("old", "t1"), _record("new", "t2")]
        decisions = [
            {"claim": "old", "decision": "NEW_CELL", "target_cell_id": None},
            {"claim": "new", "decision": "REVISE", "target_cell_id": "cell_1"},
        ]
        registry = rebuild_registry(records, "net", decisions)
        self.assertEqual([v.operation for v in registry["cell_1"].versions], ["create", "revise"])


class PackContextTest(unittest.TestCase):
    def test_memory_is_expanded_to_raw_source_and_deduplicated(self):
        raw = Candidate("raw:t1", "raw", "Seb: old", ("t1",), observed_at="2026")
        memory = Candidate("cell:1", "derived:versioned_cell", "old -> new", ("t1",), entities=("Seb",))
        context, info = pack_context([memory, raw], {"t1": raw}, budget=100)
        self.assertIn("[DERIVED cell:1]", context)
        self.assertEqual(context.count("[[t1]]"), 1)
        self.assertEqual(info["source_ids"], ["t1"])

    def test_derived_unit_is_skipped_when_whole_provenance_does_not_fit(self):
        raw = Candidate("raw:t1", "raw", "Seb: " + "word " * 20, ("t1",), observed_at="2026")
        memory = Candidate("cell:1", "derived:versioned_cell", "state", ("t1",))
        context, info = pack_context([memory], {"t1": raw}, budget=5)
        self.assertEqual(context, "")
        self.assertEqual(info["memory_candidate_ids"], [])
        self.assertLessEqual(info["context_words"], 5)


if __name__ == "__main__":
    unittest.main()
