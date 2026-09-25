"""Tests for research/semantic_slot_linking_v2.py -- pure validation/schema
logic only. No LLM calls, no frozen snapshot writes."""
import unittest
from unittest.mock import patch

from research.semantic_slot_linking_v2 import (
    EvidenceRecord, _normalize_cache_key, _resolve_indices, _value_leaked,
    validate_normalization,
)


def _records(*messages):
    return tuple(
        EvidenceRecord(index=i, turn_id=f"t{i}", session_index=1, speaker="X", message=msg,
                        timestamp="2026-01-01T00:00:00", relevance="")
        for i, msg in enumerate(messages)
    )


class DianeFourthPhotographTest(unittest.TestCase):
    """The exact bug v1 had: state_holder must be the PERSON (Diane), never
    the object phrase ("fourth photograph")."""

    def test_correct_schema_separates_holder_from_object(self):
        records = _records("okay okay I hear you. I was going to pick the fourth one. Can I submit both?")
        output = {
            "state_holder": "Diane", "viewpoint_owner": None, "topic_object": "fourth photograph",
            "state_dimension": "plan", "slot_question": "Which photograph is Diane planning to select?",
            "value": "fourth photograph", "assertion_type": "STATE", "normalization_confidence": 0.9,
            "source_record_indices": [0],
        }
        validated, reason = validate_normalization(output, records)
        self.assertEqual(reason, "ok")
        self.assertEqual(validated["state_holder"], "Diane")
        self.assertNotEqual(validated["state_holder"], "fourth photograph")

    def test_the_old_bug_pattern_would_still_validate_structurally_but_is_wrong_content(self):
        # The validator checks STRUCTURE, not semantic correctness (that's
        # exactly why v1's bug slipped through) -- this documents that the
        # fix is in prompt/schema separation, not a content-level check the
        # validator could reasonably enforce.
        records = _records("I was going to pick the fourth one.")
        buggy_output = {
            "state_holder": "fourth photograph", "viewpoint_owner": None, "topic_object": "",
            "state_dimension": "plan", "slot_question": "What is being selected?",
            "value": "fourth photograph", "assertion_type": "STATE", "normalization_confidence": 0.9,
            "source_record_indices": [0],
        }
        validated, reason = validate_normalization(buggy_output, records)
        self.assertEqual(reason, "ok")  # structurally valid -- content quality is a model-quality concern


class SebKneeTest(unittest.TestCase):
    def test_health_state_normalization(self):
        records = _records("knee completely went this afternoon, like proper locked up after coaching")
        output = {
            "state_holder": "Seb", "viewpoint_owner": None, "topic_object": "Seb",
            "state_dimension": "health", "slot_question": "What is the condition of Seb's knee?",
            "value": "locked", "assertion_type": "STATE", "normalization_confidence": 0.95,
            "source_record_indices": [0],
        }
        validated, reason = validate_normalization(output, records)
        self.assertEqual(reason, "ok")
        self.assertEqual(validated["state_holder"], "Seb")
        self.assertEqual(validated["state_dimension"], "health")


class TigmenGordeyTest(unittest.TestCase):
    def test_opinion_separates_viewpoint_owner_from_topic_object(self):
        records = _records("Tigmen said Gordey is an asshole.")
        output = {
            "state_holder": "Tigmen", "viewpoint_owner": "Tigmen", "topic_object": "Gordey",
            "state_dimension": "opinion", "slot_question": "How does Tigmen evaluate Gordey?",
            "value": "asshole", "assertion_type": "STATE", "normalization_confidence": 0.9,
            "source_record_indices": [0],
        }
        validated, reason = validate_normalization(output, records)
        self.assertEqual(reason, "ok")
        self.assertEqual(validated["viewpoint_owner"], "Tigmen")
        self.assertEqual(validated["topic_object"], "Gordey")
        self.assertNotEqual(validated["topic_object"], validated["state_holder"] + " is wrong")  # sanity


class InvalidSourceIndexTest(unittest.TestCase):
    def test_out_of_range_index_is_rejected(self):
        records = _records("only one record")
        output = {
            "state_holder": "X", "viewpoint_owner": None, "topic_object": "Y", "state_dimension": "d",
            "slot_question": "q?", "value": "v", "assertion_type": "STATE", "normalization_confidence": 0.9,
            "source_record_indices": [0, 5],  # 5 does not exist
        }
        validated, reason = validate_normalization(output, records)
        self.assertIsNone(validated)
        self.assertEqual(reason, "invalid_source_index")

    def test_empty_indices_is_rejected(self):
        records = _records("a")
        output = {
            "state_holder": "X", "viewpoint_owner": None, "topic_object": "Y", "state_dimension": "d",
            "slot_question": "q?", "value": "v", "assertion_type": "STATE", "normalization_confidence": 0.9,
            "source_record_indices": [],
        }
        validated, reason = validate_normalization(output, records)
        self.assertIsNone(validated)
        self.assertEqual(reason, "no_source_indices")


class DeterministicIdInheritanceTest(unittest.TestCase):
    def test_resolve_indices_never_trusts_out_of_range_values(self):
        self.assertIsNone(_resolve_indices([0, 1, 99], 3))
        self.assertEqual(_resolve_indices([2, 0, 0], 3), (2, 0))  # dedup, order preserved, in-range

    def test_source_turn_ids_come_from_records_not_llm_text(self):
        records = _records("a", "b", "c")
        output = {
            "state_holder": "X", "viewpoint_owner": None, "topic_object": "Y", "state_dimension": "d",
            "slot_question": "q?", "value": "v", "assertion_type": "STATE", "normalization_confidence": 0.9,
            "source_record_indices": [2],
        }
        validated, reason = validate_normalization(output, records)
        self.assertEqual(reason, "ok")
        # The LLM never provides turn_ids at all -- only indices; the caller
        # (normalize_cases) maps validated["source_indices"] -> real turn_ids.
        self.assertEqual(validated["source_indices"], (2,))
        self.assertEqual(records[2].turn_id, "t2")


class ValueLeakageTest(unittest.TestCase):
    def test_value_inside_slot_question_is_rejected(self):
        records = _records("x")
        output = {
            "state_holder": "Seb", "viewpoint_owner": None, "topic_object": "Seb", "state_dimension": "health",
            "slot_question": "Is Seb's knee locked?", "value": "locked", "assertion_type": "STATE",
            "normalization_confidence": 0.9, "source_record_indices": [0],
        }
        validated, reason = validate_normalization(output, records)
        self.assertIsNone(validated)
        self.assertEqual(reason, "value_leaked_into_slot_question")

    def test_incidental_shared_word_is_not_treated_as_leakage(self):
        # "good" appears in both value and question by coincidence of common
        # words -- must not be flagged unless the full value phrase leaks.
        records = _records("x")
        output = {
            "state_holder": "Tigmen", "viewpoint_owner": "Tigmen", "topic_object": "Gordey",
            "state_dimension": "opinion", "slot_question": "How does Tigmen evaluate Gordey these days?",
            "value": "a genuinely good and thoughtful person", "assertion_type": "STATE",
            "normalization_confidence": 0.9, "source_record_indices": [0],
        }
        validated, reason = validate_normalization(output, records)
        self.assertEqual(reason, "ok")


class MissingStateHolderTest(unittest.TestCase):
    def test_empty_state_holder_is_rejected(self):
        records = _records("x")
        output = {
            "state_holder": "", "viewpoint_owner": None, "topic_object": "Y", "state_dimension": "d",
            "slot_question": "q?", "value": "v", "assertion_type": "STATE", "normalization_confidence": 0.9,
            "source_record_indices": [0],
        }
        validated, reason = validate_normalization(output, records)
        self.assertIsNone(validated)
        self.assertEqual(reason, "missing_state_holder")


class InvalidAssertionTypeTest(unittest.TestCase):
    def test_unknown_assertion_type_is_rejected(self):
        records = _records("x")
        output = {
            "state_holder": "X", "viewpoint_owner": None, "topic_object": "Y", "state_dimension": "d",
            "slot_question": "q?", "value": "v", "assertion_type": "OPINION", "normalization_confidence": 0.9,
            "source_record_indices": [0],
        }
        validated, reason = validate_normalization(output, records)
        self.assertIsNone(validated)
        self.assertEqual(reason, "invalid_assertion_type")


class CacheKeySensitivityTest(unittest.TestCase):
    def test_different_side_changes_key(self):
        self.assertNotEqual(
            _normalize_cache_key("qa1", "old", "PROMPT"), _normalize_cache_key("qa1", "new", "PROMPT"),
        )

    def test_different_prompt_changes_key(self):
        self.assertNotEqual(
            _normalize_cache_key("qa1", "old", "PROMPT_A"), _normalize_cache_key("qa1", "old", "PROMPT_B"),
        )

    def test_different_qa_id_changes_key(self):
        self.assertNotEqual(
            _normalize_cache_key("qa1", "old", "PROMPT"), _normalize_cache_key("qa2", "old", "PROMPT"),
        )


class CachedRerunTest(unittest.TestCase):
    def test_cached_normalization_makes_zero_llm_calls(self):
        import hashlib
        import json as json_module

        from research.semantic_slot_linking_v2 import Q8LinkingCase, normalize_cases
        import pandas as pd

        case = Q8LinkingCase(
            qa_id="Q8_x", network_id="net1", state_holder="Seb", viewpoint_owner=None, topic_object="Seb",
            old_state_text="Seb's knee is fine", new_state_text="Seb's knee is locked", trigger_text=None,
            old_source_turn_ids=("t0",), new_source_turn_ids=("t0",),
            old_evidence_indices=(0,), new_evidence_indices=(0,),
            relation="REVISE", construction_confidence=0.9, construction_notes="", verified=False,
        )
        conversations = pd.DataFrame([{
            "network_id": "net1", "session_id": "s1", "session_index": 1, "turn_id": "t0",
            "timestamp": "2026-01-01T00:00:00", "speaker_display_name": "Seb", "message": "knee is locked",
        }])

        with patch("research.semantic_slot_linking_v2._normalize_prompt", return_value="PROMPT"):
            key_old = _normalize_cache_key("Q8_x", "old", "PROMPT")
            key_new = _normalize_cache_key("Q8_x", "new", "PROMPT")
            precomputed_output = {
                "state_holder": "Seb", "viewpoint_owner": None, "topic_object": "Seb", "state_dimension": "health",
                "slot_question": "What is the condition of Seb's knee?", "value": "fine",
                "assertion_type": "STATE", "normalization_confidence": 0.9, "source_record_indices": [0],
            }
            cache_lines = "\n".join(
                json_module.dumps({"cache_key": k, "output": precomputed_output}) for k in (key_old, key_new)
            )
            with patch("research.semantic_slot_linking_v2.NORMALIZATION_CACHE") as mock_path:
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = cache_lines
                with patch("llm.groq_client.get_chat_model") as mock_model:
                    slots, rejected = normalize_cases([case], conversations)
                    mock_model.assert_not_called()
        self.assertEqual(len(slots), 2)


class CandidatePoolIncludesGoldTargetTest(unittest.TestCase):
    """Regression test for a real bug: the retrieval candidate pool must
    include the query's own gold OLD-state slot -- excluding it silently
    forces recall=0 for every representation, which is exactly what
    happened on the first real run of this module."""

    def test_gold_old_slot_is_retrievable_for_its_own_query(self):
        from research.semantic_slot_linking_v2 import NormalizedSlot, run_retrieval_phase

        def _slot(case_qa_id, side, slot_question, state_text):
            return NormalizedSlot(
                case_qa_id=case_qa_id, side=side, network_id="net1", state_holder="Seb",
                viewpoint_owner=None, topic_object="Seb", state_dimension="health",
                slot_question=slot_question, value="v", assertion_type="STATE",
                normalization_confidence=0.9, source_turn_ids=("t1",), state_text=state_text,
            )

        old = _slot("Q8_x", "old", "What is the condition of Seb's knee?", "Seb's knee is fine")
        new = _slot("Q8_x", "new", "What is the condition of Seb's knee?", "Seb's knee is locked")
        distractor_old = _slot("Q8_y", "old", "Where does Priya live?", "Priya moved to Berlin")

        old_slots_by_network = {"net1": [old, distractor_old]}
        results, rows = run_retrieval_phase(
            old_slots_by_network, [new], session_vec_by_key={}, turn_to_session={},
            split_label_by_network={"net1": "eval"},
        )
        row = next(r for r in rows if r["variant"] == "A_full_claim")
        # The correct gold candidate must at least be PRESENT in the pool
        # (rank is not None) -- proves it was never excluded from candidacy.
        self.assertIsNotNone(row["rank"])
        self.assertIn("Q8_x:old", row["top10"])


class SelectBestVariantTest(unittest.TestCase):
    """Regression test: recall@5 ties must NOT silently resolve to the
    first variant by array order -- recall@1 then MRR must break the tie."""

    def _results(self, recall5, recall1, mrr):
        return {"eval": {"recall@5": recall5, "recall@1": recall1, "mrr": mrr}}

    def test_recall5_tie_breaks_on_recall1(self):
        from research.semantic_slot_linking_v2 import select_best_variant

        retrieval_results = {
            "A_full_claim": self._results(0.949, 0.847, 0.895),
            "D_structured_plus_context": self._results(0.949, 0.915, 0.929),
        }
        self.assertEqual(select_best_variant(retrieval_results), "D_structured_plus_context")

    def test_strictly_better_recall5_wins_regardless_of_recall1(self):
        from research.semantic_slot_linking_v2 import select_best_variant

        retrieval_results = {
            "A_full_claim": self._results(0.90, 0.99, 0.99),
            "B_slot_question": self._results(0.95, 0.10, 0.10),
        }
        self.assertEqual(select_best_variant(retrieval_results), "B_slot_question")

    def test_nan_metrics_never_win(self):
        from research.semantic_slot_linking_v2 import select_best_variant

        nan = float("nan")
        retrieval_results = {
            "A_full_claim": self._results(nan, nan, nan),
            "B_slot_question": self._results(0.5, 0.5, 0.5),
        }
        self.assertEqual(select_best_variant(retrieval_results), "B_slot_question")


class MeasureFalseMergeSplitTest(unittest.TestCase):
    def test_eval_and_dev_negatives_are_measured_separately(self):
        from research.semantic_slot_linking_v2 import measure_false_merge

        negatives = [
            {"tier": 1, "query_id": "Q_eval:new", "candidate_id": "Q_other:old", "network_id": "net_eval"},
            {"tier": 1, "query_id": "Q_dev:new", "candidate_id": "Q_other2:old", "network_id": "net_dev"},
        ]
        retrieval_rows = [
            {"variant": "A_full_claim", "qa_id": "Q_eval", "network_id": "net_eval", "rank": 1,
             "empty_pool": False, "top10": ["Q_eval:old", "Q_other:old"], "pool_size": 2},
            {"variant": "A_full_claim", "qa_id": "Q_dev", "network_id": "net_dev", "rank": 1,
             "empty_pool": False, "top10": ["Q_dev:old"], "pool_size": 1},  # Q_other2 never appears here
        ]
        split_label = {"net_eval": "eval", "net_dev": "dev"}
        result = measure_false_merge(negatives, retrieval_rows, split_label)
        eval_fm = result["A_full_claim"]["eval"]
        dev_fm = result["A_full_claim"]["dev"]
        self.assertEqual(eval_fm["n_evaluated"], 1)
        self.assertEqual(eval_fm["false_merge_rate@5"], 1.0)  # Q_other:old IS in Q_eval's top10[:5]
        self.assertEqual(dev_fm["n_evaluated"], 1)
        self.assertEqual(dev_fm["false_merge_rate@5"], 0.0)  # Q_other2:old is NOT in Q_dev's top10

    def test_pool_size_gt5_filter_excludes_small_pools(self):
        from research.semantic_slot_linking_v2 import measure_false_merge

        negatives = [{"tier": 1, "query_id": "Q1:new", "candidate_id": "Q2:old", "network_id": "net1"}]
        retrieval_rows = [{"variant": "A_full_claim", "qa_id": "Q1", "network_id": "net1", "rank": 1,
                            "empty_pool": False, "top10": ["Q1:old", "Q2:old"], "pool_size": 2}]
        result = measure_false_merge(negatives, retrieval_rows, {"net1": "eval"})
        self.assertEqual(result["A_full_claim"]["eval_pool_gt5_only"]["n_evaluated"], 0)  # pool_size=2, not >5
        self.assertEqual(result["A_full_claim"]["eval"]["n_evaluated"], 1)

    def test_chance_baseline_reflects_pool_size(self):
        from research.semantic_slot_linking_v2 import measure_false_merge

        negatives = [{"tier": 1, "query_id": "Q1:new", "candidate_id": "Q2:old", "network_id": "net1"}]
        # pool_size=2 -> chance of a specific candidate landing in top-5 is 100% (min(5,2)/2)
        retrieval_rows = [{"variant": "A_full_claim", "qa_id": "Q1", "network_id": "net1", "rank": 1,
                            "empty_pool": False, "top10": ["Q1:old", "Q2:old"], "pool_size": 2}]
        result = measure_false_merge(negatives, retrieval_rows, {"net1": "eval"})
        self.assertAlmostEqual(result["A_full_claim"]["eval"]["chance_baseline_fm@5"], 1.0)


class NormalizeResolverSelectionTest(unittest.TestCase):
    """Regression test for a real bug: the resolver LLM sometimes returns
    the JSON STRING "null" (truthy, not-None in Python) instead of an
    actual null, and this was previously counted as a real selection --
    causing every NEW_CELL decision that used the string "null" to be
    wrongly flagged as a false merge (0/59 correct, 59/59 false_merge on
    the first real resolver run)."""

    def _candidate(self, qa_id):
        from research.semantic_slot_linking_v2 import NormalizedSlot
        return NormalizedSlot(
            case_qa_id=qa_id, side="old", network_id="net1", state_holder="X", viewpoint_owner=None,
            topic_object="Y", state_dimension="d", slot_question="q?", value="v", assertion_type="STATE",
            normalization_confidence=0.9, source_turn_ids=("t0",), state_text="s",
        )

    def test_string_null_is_treated_as_no_selection(self):
        from research.semantic_slot_linking_v2 import _normalize_resolver_selection
        self.assertIsNone(_normalize_resolver_selection("null", [self._candidate("Q1")]))
        self.assertIsNone(_normalize_resolver_selection(None, [self._candidate("Q1")]))

    def test_exact_id_is_accepted(self):
        from research.semantic_slot_linking_v2 import _normalize_resolver_selection
        candidates = [self._candidate("Q8_x")]
        self.assertEqual(_normalize_resolver_selection("Q8_x:old", candidates), "Q8_x:old")

    def test_truncated_id_missing_side_suffix_is_reconstructed_when_unique(self):
        from research.semantic_slot_linking_v2 import _normalize_resolver_selection
        candidates = [self._candidate("Q8_n2d0e1f2")]
        self.assertEqual(_normalize_resolver_selection("Q8_n2d0e1f2", candidates), "Q8_n2d0e1f2:old")

    def test_unrecognized_id_never_invents_a_candidate(self):
        from research.semantic_slot_linking_v2 import _normalize_resolver_selection
        candidates = [self._candidate("Q8_x")]
        self.assertIsNone(_normalize_resolver_selection("Q8_totally_unknown", candidates))


class ResolverPhaseFlagsTest(unittest.TestCase):
    def test_low_confidence_defaults_to_new_cell(self):
        from unittest.mock import patch as _patch

        from research.semantic_slot_linking_v2 import NormalizedSlot, run_resolver_phase

        query = NormalizedSlot(
            case_qa_id="Q1", side="new", network_id="net1", state_holder="Seb", viewpoint_owner=None,
            topic_object="Seb", state_dimension="health", slot_question="q?", value="v",
            assertion_type="STATE", normalization_confidence=0.9, source_turn_ids=("t1",), state_text="x",
        )
        old = NormalizedSlot(
            case_qa_id="Q1", side="old", network_id="net1", state_holder="Seb", viewpoint_owner=None,
            topic_object="Seb", state_dimension="health", slot_question="q?", value="v0",
            assertion_type="STATE", normalization_confidence=0.9, source_turn_ids=("t0",), state_text="y",
        )
        retrieval_rows = [{"variant": "A_full_claim", "qa_id": "Q1", "network_id": "net1", "rank": 1,
                            "empty_pool": False, "top10": ["Q1:old"], "pool_size": 1}]
        with _patch("research.semantic_slot_linking_v2._load_cache_dict", return_value={}), \
             _patch("research.semantic_slot_linking_v2._append_cache"), \
             _patch("research.semantic_slot_linking_v2._call_llm", return_value={
                 "selected_cell_id": "Q1:old", "operation": "REVISE", "confidence": 0.2, "rationale": "x",
             }):
            results = run_resolver_phase([query], {"net1": [old]}, retrieval_rows, "A_full_claim", {"Q1": "REVISE"})
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["defaulted"])
        self.assertEqual(results[0]["operation"], "NEW_CELL")
        self.assertIsNone(results[0]["selected_cell_id"])
        self.assertFalse(results[0]["correct_selection"])

    def test_string_null_new_cell_decision_is_not_flagged_as_false_merge(self):
        from unittest.mock import patch as _patch

        from research.semantic_slot_linking_v2 import NormalizedSlot, run_resolver_phase

        query = NormalizedSlot(
            case_qa_id="Q1", side="new", network_id="net1", state_holder="Seb", viewpoint_owner=None,
            topic_object="Seb", state_dimension="health", slot_question="q?", value="v",
            assertion_type="STATE", normalization_confidence=0.9, source_turn_ids=("t1",), state_text="x",
        )
        old = NormalizedSlot(
            case_qa_id="Q1", side="old", network_id="net1", state_holder="Seb", viewpoint_owner=None,
            topic_object="Seb", state_dimension="health", slot_question="q?", value="v0",
            assertion_type="STATE", normalization_confidence=0.9, source_turn_ids=("t0",), state_text="y",
        )
        retrieval_rows = [{"variant": "A_full_claim", "qa_id": "Q1", "network_id": "net1", "rank": 1,
                            "empty_pool": False, "top10": ["Q1:old"], "pool_size": 1}]
        with _patch("research.semantic_slot_linking_v2._load_cache_dict", return_value={}), \
             _patch("research.semantic_slot_linking_v2._append_cache"), \
             _patch("research.semantic_slot_linking_v2._call_llm", return_value={
                 # This is exactly the observed real output shape: high
                 # confidence, valid operation, but selected_cell_id is the
                 # JSON STRING "null".
                 "selected_cell_id": "null", "operation": "NEW_CELL", "confidence": 0.95, "rationale": "x",
             }):
            results = run_resolver_phase([query], {"net1": [old]}, retrieval_rows, "A_full_claim", {"Q1": "REVISE"})
        self.assertFalse(results[0]["defaulted"])
        self.assertEqual(results[0]["operation"], "NEW_CELL")
        self.assertIsNone(results[0]["selected_cell_id"])
        self.assertFalse(results[0]["false_merge"])  # no selection was actually made -- must not count as a merge


if __name__ == "__main__":
    unittest.main()
