"""Tests for research/semantic_slot_linking_pilot.py -- pure normalization/
candidate-filter/chain logic only. No LLM calls, no held-out or benchmark
data touched."""
import unittest
from unittest.mock import patch

from research.semantic_slot_linking_pilot import (
    RawAssertion, SlotAssertion, _cache_key, build_slot_cache, candidate_pool,
    classify_false_merge, classify_false_split, materialize_chain, validate_slot_normalization,
)


def _raw(**overrides):
    base = dict(
        assertion_id="net1:s01:0", network_id="net1", session_id="s01",
        viewpoint_owner="Tigmen", subject="Gordey", facet="attitude",
        assertion_text="Tigmen said Gordey is an asshole.", normalized_value="asshole",
        modality="opinion", source_turn_ids=("t1",),
    )
    base.update(overrides)
    return RawAssertion(**base)


def _slot(**overrides):
    base = dict(
        assertion_id="net1:s01:0", network_id="net1", viewpoint_owner="tigmen", subject_entity="gordey",
        state_dimension="attitude/evaluation", slot_question="How does Tigmen evaluate Gordey?",
        value="asshole", modality="opinion", observed_at="2026-01-01T00:00:00",
        source_turn_ids=("t1",), normalization_confidence=0.9, assertion_text="Tigmen said Gordey is an asshole.",
    )
    base.update(overrides)
    return SlotAssertion(**base)


class ValidateSlotNormalizationTest(unittest.TestCase):
    def test_slot_question_must_not_contain_value(self):
        raw = _raw()
        llm_out = {
            "subject_entity": "Gordey", "state_dimension": "attitude",
            "slot_question": "Is Gordey an asshole according to Tigmen?",  # leaks the value
            "normalization_confidence": 0.9, "source_turn_ids": ["t1"],
        }
        record, reason = validate_slot_normalization(raw, llm_out, "2026-01-01T00:00:00")
        self.assertIsNone(record)
        self.assertEqual(reason, "slot_question_contains_value")

    def test_valid_slot_question_is_accepted(self):
        raw = _raw()
        llm_out = {
            "subject_entity": "Gordey", "state_dimension": "attitude/evaluation",
            "slot_question": "How does Tigmen evaluate Gordey?",
            "normalization_confidence": 0.9, "source_turn_ids": ["t1"],
        }
        record, reason = validate_slot_normalization(raw, llm_out, "2026-01-01T00:00:00")
        self.assertEqual(reason, "ok")
        self.assertEqual(record.observed_at, "2026-01-01T00:00:00")

    def test_fabricated_source_turn_ids_are_rejected(self):
        raw = _raw(source_turn_ids=("t1",))
        llm_out = {
            "subject_entity": "Gordey", "state_dimension": "attitude", "slot_question": "How is Gordey seen?",
            "normalization_confidence": 0.9, "source_turn_ids": ["t1", "t_invented"],
        }
        record, reason = validate_slot_normalization(raw, llm_out, "2026-01-01T00:00:00")
        self.assertIsNone(record)
        self.assertEqual(reason, "fabricated_or_altered_source_turn_ids")

    def test_dropped_source_turn_id_is_also_rejected(self):
        raw = _raw(source_turn_ids=("t1", "t2"))
        llm_out = {
            "subject_entity": "Gordey", "state_dimension": "attitude", "slot_question": "How is Gordey seen?",
            "normalization_confidence": 0.9, "source_turn_ids": ["t1"],  # silently dropped t2
        }
        record, reason = validate_slot_normalization(raw, llm_out, "2026-01-01T00:00:00")
        self.assertIsNone(record)
        self.assertEqual(reason, "fabricated_or_altered_source_turn_ids")

    def test_observed_at_comes_from_raw_conversation_data_not_llm(self):
        raw = _raw()
        llm_out = {
            "subject_entity": "Gordey", "state_dimension": "attitude", "slot_question": "How is Gordey seen?",
            "normalization_confidence": 0.9, "source_turn_ids": ["t1"], "observed_at": "2099-01-01T00:00:00",
        }
        record, reason = validate_slot_normalization(raw, llm_out, "2026-03-05T10:00:00")
        self.assertEqual(reason, "ok")
        self.assertEqual(record.observed_at, "2026-03-05T10:00:00")


class CandidatePoolTest(unittest.TestCase):
    def test_same_owner_entity_slot_different_value_stays_candidate(self):
        query = _slot(assertion_id="q", value="good person", source_turn_ids=("t9",))
        candidate = _slot(assertion_id="c", value="asshole", source_turn_ids=("t1",))
        pool = candidate_pool(query, [candidate])
        self.assertEqual([c.assertion_id for c in pool], ["c"])

    def test_different_state_dimension_is_not_pre_merged_out_of_pool(self):
        query = _slot(
            assertion_id="q", state_dimension="location", slot_question="Where is Gordey?", value="Moscow",
            source_turn_ids=("t9",),
        )
        same_dim = _slot(assertion_id="c1", state_dimension="attitude/evaluation", source_turn_ids=("t1",))
        pool = candidate_pool(query, [same_dim])
        # Different slot is still a retrievable candidate (ranking, not a hard filter, decides relevance) --
        # candidate_pool must never pre-merge or silently drop it.
        self.assertEqual([c.assertion_id for c in pool], ["c1"])

    def test_different_resolved_owner_is_excluded(self):
        query = _slot(assertion_id="q", viewpoint_owner="tigmen")
        candidate = _slot(assertion_id="c", viewpoint_owner="priya")
        pool = candidate_pool(query, [candidate])
        self.assertEqual(pool, [])

    def test_different_resolved_subject_entity_is_excluded(self):
        query = _slot(assertion_id="q", subject_entity="gordey")
        candidate = _slot(assertion_id="c", subject_entity="priya")
        pool = candidate_pool(query, [candidate])
        self.assertEqual(pool, [])

    def test_unresolved_identity_is_not_automatically_excluded(self):
        query = _slot(assertion_id="q", subject_entity="", source_turn_ids=("t9",))  # unresolved
        candidate = _slot(assertion_id="c", subject_entity="gordey", source_turn_ids=("t1",))
        pool = candidate_pool(query, [candidate])
        self.assertEqual([c.assertion_id for c in pool], ["c"])

    def test_facet_is_not_part_of_the_candidate_model_at_all(self):
        # SlotAssertion has no facet field -- top-k cannot depend on exact
        # facet match because the field doesn't exist past normalization.
        self.assertNotIn("facet", SlotAssertion.__dataclass_fields__)

    def test_same_network_is_the_only_hard_filter_besides_identity(self):
        query = _slot(assertion_id="q", network_id="net1")
        other_network = _slot(assertion_id="c", network_id="net2")
        pool = candidate_pool(query, [other_network])
        self.assertEqual(pool, [])


class CacheKeyTest(unittest.TestCase):
    def test_different_prompt_hash_changes_cache_key(self):
        raw = _raw()
        self.assertNotEqual(_cache_key(raw, "hash_a"), _cache_key(raw, "hash_b"))

    def test_cached_rerun_makes_zero_llm_calls(self):
        raw = _raw()
        with patch("research.semantic_slot_linking_pilot._slot_normalize_prompt", return_value="PROMPT") as mock_prompt:
            import hashlib
            prompt_hash = hashlib.sha256(b"PROMPT").hexdigest()
            key = _cache_key(raw, prompt_hash)
            precomputed = {"cache_key": key, "assertion_id": raw.assertion_id, "schema_version": "slot_v1", "llm_output": {}}
            with patch("research.semantic_slot_linking_pilot.SLOT_CACHE") as mock_path:
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = __import__("json").dumps(precomputed)
                with patch("llm.groq_client.get_chat_model") as mock_model:
                    build_slot_cache([raw])
                    mock_model.assert_not_called()


class FalseSplitMergeDefinitionTest(unittest.TestCase):
    def test_false_split_requires_both_sides_valid_and_confirmed_pair(self):
        self.assertIsNone(classify_false_split(query_valid=False, target_valid=True, is_confirmed_pair=True, same_cell=False))
        self.assertIsNone(classify_false_split(query_valid=True, target_valid=True, is_confirmed_pair=False, same_cell=False))

    def test_false_split_declared_only_with_full_evidence(self):
        result = classify_false_split(query_valid=True, target_valid=True, is_confirmed_pair=True, same_cell=False)
        self.assertTrue(result)
        result_ok = classify_false_split(query_valid=True, target_valid=True, is_confirmed_pair=True, same_cell=True)
        self.assertFalse(result_ok)

    def test_false_merge_requires_both_sides_valid_and_confirmed_different(self):
        self.assertIsNone(classify_false_merge(query_valid=True, target_valid=False, is_confirmed_different=True, same_cell=True))
        self.assertIsNone(classify_false_merge(query_valid=True, target_valid=True, is_confirmed_different=False, same_cell=True))

    def test_none_equals_none_is_not_a_match(self):
        # Two totally unresolved/unusable sides must never resolve to a
        # confirmed split or merge verdict.
        self.assertIsNone(classify_false_split(False, False, False, same_cell=False))
        self.assertIsNone(classify_false_merge(False, False, False, same_cell=True))


class ChainMaterializationTest(unittest.TestCase):
    def test_both_assertions_survive_relinking_with_original_provenance(self):
        old = _slot(assertion_id="old1", value="asshole", source_turn_ids=("t1",), observed_at="2026-01-01T00:00:00")
        new = _slot(assertion_id="new1", value="good person", source_turn_ids=("t9",), observed_at="2026-02-01T00:00:00")
        chain = materialize_chain("cell1", old, new, "REVISE")
        assertion_ids = {v.assertion_id for v in chain.versions}
        self.assertEqual(assertion_ids, {"old1", "new1"})
        by_id = {v.assertion_id: v for v in chain.versions}
        self.assertEqual(by_id["old1"].source_turn_ids, ("t1",))
        self.assertEqual(by_id["new1"].source_turn_ids, ("t9",))
        self.assertEqual(chain.versions[0].operation, "create")
        self.assertEqual(chain.versions[1].operation, "revise")


if __name__ == "__main__":
    unittest.main()
