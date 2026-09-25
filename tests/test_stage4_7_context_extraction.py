"""Tests for research/stage4_7_context_extraction.py -- pure validation
logic + cache-key/cache-hit behavior. No LLM quality is tested here (that
needs the human review sheet), only the deterministic rules and the
caching contract."""
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from research.stage4_7_context_extraction import (
    _cache_key, build_context_extraction_cache, build_group_context, validate_context_record,
)


class ValidateContextRecordTest(unittest.TestCase):
    def setUp(self):
        self.valid_ids = {"t1", "t2", "t3"}
        self.anchor_ids = {"t2"}
        self.timestamps = {"t1": "2026-01-01T00:00:00", "t2": "2026-01-01T00:01:00", "t3": "2026-01-01T00:02:00"}

    def test_fabricated_source_id_is_dropped(self):
        record, reason = validate_context_record({
            "record_type": "STATE", "claim": "Sam's knee is fine", "state_description": "knee",
            "value": "fine", "source_turn_ids": ["t2", "fake_id"],
        }, self.valid_ids, self.anchor_ids, self.timestamps)
        self.assertEqual(reason, "ok")
        self.assertEqual(record.source_turn_ids, ("t2",))

    def test_record_with_no_valid_sources_is_rejected(self):
        record, reason = validate_context_record({
            "record_type": "STATE", "claim": "x", "state_description": "y", "value": "z",
            "source_turn_ids": ["fake_only"],
        }, self.valid_ids, self.anchor_ids, self.timestamps)
        self.assertIsNone(record)
        self.assertEqual(reason, "no_valid_sources")

    def test_record_citing_only_context_turns_not_the_anchor_is_rejected(self):
        # t1/t3 are valid, real turns -- but neither is the anchor (t2).
        record, reason = validate_context_record({
            "record_type": "STATE", "claim": "x", "state_description": "y", "value": "z",
            "source_turn_ids": ["t1", "t3"],
        }, self.valid_ids, self.anchor_ids, self.timestamps)
        self.assertIsNone(record)
        self.assertEqual(reason, "no_anchor_turn_cited")

    def test_state_without_value_is_rejected(self):
        record, reason = validate_context_record({
            "record_type": "STATE", "claim": "Sam's knee", "state_description": "knee status",
            "value": None, "source_turn_ids": ["t2"],
        }, self.valid_ids, self.anchor_ids, self.timestamps)
        self.assertIsNone(record)
        self.assertEqual(reason, "state_missing_description_or_value")

    def test_state_without_description_is_rejected(self):
        record, reason = validate_context_record({
            "record_type": "STATE", "claim": "Sam's knee", "state_description": None,
            "value": "fine", "source_turn_ids": ["t2"],
        }, self.valid_ids, self.anchor_ids, self.timestamps)
        self.assertIsNone(record)
        self.assertEqual(reason, "state_missing_description_or_value")

    def test_content_free_meta_state_is_rejected(self):
        record, reason = validate_context_record({
            "record_type": "STATE", "claim": "Derek revised his position", "state_description": "position",
            "value": "revised", "source_turn_ids": ["t2"],
        }, self.valid_ids, self.anchor_ids, self.timestamps)
        self.assertIsNone(record)
        self.assertEqual(reason, "meta_value_state")

    def test_transition_with_no_content_at_all_is_rejected(self):
        record, reason = validate_context_record({
            "record_type": "TRANSITION", "claim": "something changed", "state_description": None,
            "from_value": None, "to_value": None, "source_turn_ids": ["t2"],
        }, self.valid_ids, self.anchor_ids, self.timestamps)
        self.assertIsNone(record)
        self.assertEqual(reason, "transition_missing_content")

    def test_transition_with_only_to_value_is_accepted(self):
        record, reason = validate_context_record({
            "record_type": "TRANSITION", "claim": "Hiro no longer eats fish", "state_description": None,
            "from_value": None, "to_value": "vegetarian", "source_turn_ids": ["t2"],
        }, self.valid_ids, self.anchor_ids, self.timestamps)
        self.assertEqual(reason, "ok")
        self.assertEqual(record.to_value, "vegetarian")

    def test_cause_does_not_require_value(self):
        record, reason = validate_context_record({
            "record_type": "CAUSE", "claim": "Seb hurt his knee during coaching",
            "related_state_description": "knee status", "source_turn_ids": ["t2"],
        }, self.valid_ids, self.anchor_ids, self.timestamps)
        self.assertEqual(reason, "ok")
        self.assertIsNone(record.value)

    def test_reaction_does_not_require_value(self):
        record, reason = validate_context_record({
            "record_type": "REACTION", "claim": "Priyanka expresses concern about Seb's knee",
            "source_turn_ids": ["t2"],
        }, self.valid_ids, self.anchor_ids, self.timestamps)
        self.assertEqual(reason, "ok")
        self.assertIsNone(record.value)

    def test_unknown_record_type_is_rejected(self):
        record, reason = validate_context_record({
            "record_type": "OPINION", "claim": "x", "source_turn_ids": ["t2"],
        }, self.valid_ids, self.anchor_ids, self.timestamps)
        self.assertIsNone(record)
        self.assertEqual(reason, "unknown_record_type")

    def test_observed_at_comes_from_real_timestamp_not_llm(self):
        record, reason = validate_context_record({
            "record_type": "STATE", "claim": "x", "state_description": "y", "value": "z",
            "source_turn_ids": ["t1", "t2"], "observed_at": "2099-01-01T00:00:00",  # LLM-supplied, must be ignored
        }, self.valid_ids, self.anchor_ids, self.timestamps)
        self.assertEqual(reason, "ok")
        # server-derived: max real timestamp among the (validated) cited sources
        self.assertEqual(record.observed_at, "2026-01-01T00:01:00")


class CacheKeyTest(unittest.TestCase):
    def test_different_prompt_hash_yields_different_key(self):
        key_a = _cache_key("net1", {"t1", "t2"}, "hash_a")
        key_b = _cache_key("net1", {"t1", "t2"}, "hash_b")
        self.assertNotEqual(key_a, key_b)

    def test_different_network_id_yields_different_key(self):
        key_a = _cache_key("net1", {"t1"}, "hash_a")
        key_b = _cache_key("net2", {"t1"}, "hash_a")
        self.assertNotEqual(key_a, key_b)

    def test_same_inputs_yield_same_key(self):
        key_a = _cache_key("net1", {"t1", "t2"}, "hash_a")
        key_b = _cache_key("net1", {"t2", "t1"}, "hash_a")  # set order shouldn't matter
        self.assertEqual(key_a, key_b)


class CachedRerunTest(unittest.TestCase):
    def setUp(self):
        self.conversations = pd.DataFrame([
            {"network_id": "net1", "session_id": "net1_s01", "session_index": 1, "turn_id": "net1_s01_t000",
             "timestamp": "2026-01-01T00:00:00", "speaker_display_name": "Sam", "message": "hi"},
            {"network_id": "net1", "session_id": "net1_s01", "session_index": 1, "turn_id": "net1_s01_t001",
             "timestamp": "2026-01-01T00:01:00", "speaker_display_name": "Sam", "message": "knee is fine"},
            {"network_id": "net1", "session_id": "net1_s01", "session_index": 1, "turn_id": "net1_s01_t002",
             "timestamp": "2026-01-01T00:02:00", "speaker_display_name": "Priya", "message": "good to hear"},
        ])
        self.cases = [{"qa_id": "Q_test", "network_id": "net1", "old_ids": {"net1_s01_t001"}, "new_ids": set()}]

    def test_cache_hit_makes_zero_llm_calls(self):
        context_text, window_ids = build_group_context(self.conversations, "net1", {"net1_s01_t001"})
        from research.stage4_7_context_extraction import _context_extract_prompt
        import hashlib

        prompt = _context_extract_prompt(context_text)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        key = _cache_key("net1", {"net1_s01_t001"}, prompt_hash)

        with patch("research.stage4_7_context_extraction.CONTEXT_EXTRACTION_CACHE") as mock_path:
            import json as json_module
            precomputed_line = json_module.dumps({
                "cache_key": key, "qa_id": "Q_test", "side": "old", "schema_version": "ctx_v1",
                "window_turn_ids": sorted(window_ids), "raw_items": [],
            })
            mock_path.exists.return_value = True
            mock_path.read_text.return_value = precomputed_line

            with patch("llm.groq_client.get_chat_model") as mock_model:
                cache, key_by_case_side = build_context_extraction_cache(self.cases, self.conversations)
                mock_model.assert_not_called()

        self.assertIn(("Q_test", "old"), key_by_case_side)
        self.assertEqual(key_by_case_side[("Q_test", "old")], key)
        self.assertIn(key, cache)


if __name__ == "__main__":
    unittest.main()
