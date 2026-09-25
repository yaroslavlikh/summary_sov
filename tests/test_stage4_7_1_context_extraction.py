"""Tests for research/stage4_7_1_context_extraction.py -- the evidence/
anchor-association separation fix. Pure validation + dedup logic only, no
LLM calls, no quality judgment of the extractor's output."""
import unittest

from research.stage4_7_1_context_extraction import (
    ContextRecordV2, _cache_key, dedupe_records, validate_context_record,
)


class ValidateContextRecordTest(unittest.TestCase):
    def setUp(self):
        self.window_ids = {"t1", "t2", "t3"}
        self.anchor_ids = {"t2"}
        self.messages = {
            "t1": "should be able to make it 14th",
            "t2": "knee completely went this afternoon, like proper locked up after coaching",
            "t3": "oh no! are you ok right now?",
        }
        self.timestamps = {"t1": "2026-01-01T00:00:00", "t2": "2026-01-01T00:01:00", "t3": "2026-01-01T00:02:00"}

    def _raw(self, **overrides):
        base = dict(
            record_type="STATE", claim="Seb's knee is locked", state_description="knee condition",
            value="locked", viewpoint_owner="Seb", subject="Seb",
            evidence=[{"turn_id": "t2", "quote": "proper locked up"}],
            related_anchor_ids=["t2"],
        )
        base.update(overrides)
        return base

    def test_valid_record_is_accepted(self):
        record, reason = validate_context_record(self._raw(), self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertEqual(reason, "ok")
        self.assertEqual(record.source_turn_ids, ("t2",))

    def test_evidence_with_unknown_turn_id_is_dropped_but_record_survives_if_other_evidence_valid(self):
        record, reason = validate_context_record(self._raw(
            evidence=[{"turn_id": "fake_turn", "quote": "x"}, {"turn_id": "t2", "quote": "proper locked up"}],
        ), self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertEqual(reason, "ok")
        self.assertEqual(record.source_turn_ids, ("t2",))
        self.assertEqual(len(record.evidence), 1)

    def test_evidence_quote_not_present_in_turn_text_is_dropped(self):
        record, reason = validate_context_record(self._raw(
            evidence=[{"turn_id": "t2", "quote": "completely healthy and pain-free"}],
        ), self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertIsNone(record)
        self.assertEqual(reason, "no_valid_evidence")

    def test_record_with_zero_valid_evidence_is_rejected(self):
        record, reason = validate_context_record(self._raw(evidence=[]), self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertIsNone(record)
        self.assertEqual(reason, "no_valid_evidence")

    def test_quote_matches_after_whitespace_normalization_only(self):
        record, reason = validate_context_record(self._raw(
            evidence=[{"turn_id": "t2", "quote": "  proper   locked\nup  "}],
        ), self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertEqual(reason, "ok")

    def test_related_anchor_id_outside_actual_anchors_is_dropped(self):
        record, reason = validate_context_record(self._raw(
            related_anchor_ids=["t3"],  # t3 is a real turn but NOT an anchor
        ), self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertIsNone(record)
        self.assertEqual(reason, "no_valid_related_anchor")

    def test_related_anchor_ids_never_become_source_turn_ids(self):
        # Evidence cites t3 (a real, non-anchor turn); related_anchor_ids
        # points at t2 (the anchor) to say "this helps interpret t2" -- but
        # t2 must NOT show up in source_turn_ids just because it's related.
        record, reason = validate_context_record(self._raw(
            record_type="REACTION", claim="Priyanka is worried", viewpoint_owner="Priyanka", subject="Seb",
            state_description=None, value=None,
            evidence=[{"turn_id": "t3", "quote": "are you ok right now"}],
            related_anchor_ids=["t2"],
        ), self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertEqual(reason, "ok")
        self.assertEqual(record.source_turn_ids, ("t3",))
        self.assertEqual(record.related_anchor_ids, ("t2",))

    def test_state_without_value_is_rejected(self):
        record, reason = validate_context_record(self._raw(value=None), self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertIsNone(record)
        self.assertEqual(reason, "state_missing_description_or_value")

    def test_meta_value_state_is_rejected(self):
        record, reason = validate_context_record(self._raw(
            claim="Derek revised his position", state_description="position", value="revised",
        ), self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertIsNone(record)
        self.assertEqual(reason, "meta_value_state")

    def test_transition_with_no_content_is_rejected(self):
        record, reason = validate_context_record(self._raw(
            record_type="TRANSITION", state_description=None, value=None, from_value=None, to_value=None,
        ), self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertIsNone(record)
        self.assertEqual(reason, "transition_missing_content")

    def test_transition_with_only_to_value_does_not_invent_from_value(self):
        record, reason = validate_context_record(self._raw(
            record_type="TRANSITION", state_description=None, value=None, from_value=None, to_value="locked",
        ), self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertEqual(reason, "ok")
        self.assertIsNone(record.from_value)
        self.assertEqual(record.to_value, "locked")

    def test_cause_does_not_require_value(self):
        record, reason = validate_context_record(self._raw(
            record_type="CAUSE", claim="Coaching caused the injury", state_description=None, value=None,
            related_state_description="knee condition",
        ), self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertEqual(reason, "ok")
        self.assertIsNone(record.value)

    def test_reaction_does_not_require_value(self):
        record, reason = validate_context_record(self._raw(
            record_type="REACTION", claim="Priyanka is concerned", state_description=None, value=None,
        ), self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertEqual(reason, "ok")

    def test_observed_at_from_real_timestamp_not_llm(self):
        record, reason = validate_context_record(self._raw(observed_at="2099-01-01T00:00:00"),
                                                   self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertEqual(reason, "ok")
        self.assertEqual(record.observed_at, "2026-01-01T00:01:00")

    def test_unknown_record_type_is_rejected(self):
        record, reason = validate_context_record(self._raw(record_type="OPINION"), self.window_ids, self.anchor_ids, self.messages, self.timestamps)
        self.assertIsNone(record)
        self.assertEqual(reason, "unknown_record_type")


def _record(**overrides):
    base = dict(
        record_type="STATE", claim="Seb's knee is locked", viewpoint_owner="Seb", subject="Seb",
        state_description="knee condition", value="locked", temporal_mode="current",
        from_value=None, to_value=None, related_state_description=None,
        evidence=({"turn_id": "t2", "quote": "locked up"},), related_anchor_ids=("t2",),
        source_turn_ids=("t2",), confidence=0.9, observed_at="2026-01-01T00:01:00",
    )
    base.update(overrides)
    return ContextRecordV2(**base)


class DedupeRecordsTest(unittest.TestCase):
    def test_exact_duplicate_across_anchors_is_merged_with_unioned_related_anchor_ids(self):
        r1 = _record(related_anchor_ids=("t2",))
        r2 = _record(related_anchor_ids=("t5",))  # same everything else, different anchor tag
        result = dedupe_records([r1, r2])
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0].related_anchor_ids), {"t2", "t5"})

    def test_different_state_description_is_not_merged(self):
        r1 = _record(state_description="knee condition")
        r2 = _record(state_description="availability for hike")
        result = dedupe_records([r1, r2])
        self.assertEqual(len(result), 2)

    def test_different_source_turn_ids_is_not_merged(self):
        r1 = _record(source_turn_ids=("t2",))
        r2 = _record(source_turn_ids=("t9",))
        result = dedupe_records([r1, r2])
        self.assertEqual(len(result), 2)

    def test_different_viewpoint_owner_is_not_merged(self):
        r1 = _record(viewpoint_owner="Seb")
        r2 = _record(viewpoint_owner="Priyanka")
        result = dedupe_records([r1, r2])
        self.assertEqual(len(result), 2)

    def test_case_and_whitespace_only_difference_in_description_is_merged(self):
        r1 = _record(state_description="Knee Condition", related_anchor_ids=("t2",))
        r2 = _record(state_description="  knee condition  ", related_anchor_ids=("t7",))
        result = dedupe_records([r1, r2])
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0].related_anchor_ids), {"t2", "t7"})


class CacheKeyTest(unittest.TestCase):
    def test_different_prompt_hash_yields_different_key(self):
        self.assertNotEqual(_cache_key("net1", {"t1"}, "hash_a"), _cache_key("net1", {"t1"}, "hash_b"))

    def test_different_network_yields_different_key(self):
        self.assertNotEqual(_cache_key("net1", {"t1"}, "hash_a"), _cache_key("net2", {"t1"}, "hash_a"))

    def test_schema_version_change_changes_the_key(self):
        import research.stage4_7_1_context_extraction as mod
        key_before = _cache_key("net1", {"t1"}, "hash_a")
        original = mod.SCHEMA_VERSION
        try:
            mod.SCHEMA_VERSION = "ctx_v1_2"
            key_after = _cache_key("net1", {"t1"}, "hash_a")
        finally:
            mod.SCHEMA_VERSION = original
        self.assertNotEqual(key_before, key_after)


if __name__ == "__main__":
    unittest.main()
