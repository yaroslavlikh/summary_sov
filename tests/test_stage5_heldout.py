"""Tests for research/stage5_heldout.py -- phase-separation invariants only.
No held-out QA content is read by any test; synthetic data throughout.
Real /tmp/socialmembench_stage5_* paths are never touched -- every test
patches the module's path constants to temp files first."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

import research.stage5_heldout as stage5
from research.socialmembench_pilot import Candidate


class PrepareMemoryNeverTouchesQATest(unittest.TestCase):
    def test_prepare_memory_has_no_qa_parameter(self):
        sig = inspect.signature(stage5.prepare_memory)
        self.assertEqual(list(sig.parameters), [])

    def test_prepare_memory_source_never_calls_load_data_or_reads_qa_columns(self):
        # Executable code only, not docstrings/comments -- a comment
        # explaining "this deliberately does NOT read qa.parquet" is fine
        # and expected; an actual pd.read_parquet(..."qa.parquet") call or a
        # reference to QA/label columns is not.
        for fn in (stage5.prepare_memory, stage5._read_conversations_only, stage5.build_memory_candidates):
            lines = inspect.getsource(fn).splitlines()
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("--"):
                    continue
                self.assertNotIn("load_data(", line)
                self.assertNotIn("evidence_anchors_json", line)
                self.assertNotIn("correct_option", line)
                self.assertNotIn('"qa.parquet"', line)

    def test_read_conversations_only_reads_only_conversations_file(self):
        with patch("research.stage5_heldout.pd.read_parquet") as mock_read:
            mock_read.return_value = pd.DataFrame()
            stage5._read_conversations_only(Path("/tmp/fake_dir"))
            mock_read.assert_called_once_with(Path("/tmp/fake_dir/conversations.parquet"))


class SnapshotChecksumTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.snapshot_path = Path(self.tmpdir.name) / "snapshot.jsonl"
        self.meta_path = Path(self.tmpdir.name) / "meta.json"
        self.patches = [
            patch.object(stage5, "SNAPSHOT_JSONL", self.snapshot_path),
            patch.object(stage5, "SNAPSHOT_META", self.meta_path),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmpdir.cleanup()

    def test_load_snapshot_raises_if_missing(self):
        with self.assertRaises(RuntimeError):
            stage5._load_snapshot()

    def test_load_snapshot_detects_checksum_mismatch(self):
        self.snapshot_path.write_text('{"a": 1}\n')
        actual = stage5._sha256_file(self.snapshot_path)
        self.meta_path.write_text(json.dumps({"snapshot_sha256": "not_" + actual}))
        with self.assertRaises(RuntimeError):
            stage5._load_snapshot()

    def test_load_snapshot_succeeds_with_matching_checksum(self):
        self.snapshot_path.write_text('{"a": 1}\n')
        actual = stage5._sha256_file(self.snapshot_path)
        self.meta_path.write_text(json.dumps({"snapshot_sha256": actual}))
        rows, meta = stage5._load_snapshot()
        self.assertEqual(rows, [{"a": 1}])


class InferenceColumnRestrictionTest(unittest.TestCase):
    def test_reads_only_allowed_inference_columns(self):
        manifest = {"qa": [{"qa_id": "Q1_x", "network_id": "grp_0d1e2f3a", "query_type": "Q1"}],
                    "networks": ["grp_0d1e2f3a"]}
        fake_df = pd.DataFrame([{
            "qa_id": "Q1_x", "network_id": "grp_0d1e2f3a", "query_type": "Q1",
            "question": "q", "answer_format": "long_form", "options_json": "{}",
        }])
        with patch("research.stage5_heldout.pd.read_parquet", return_value=fake_df) as mock_read:
            with patch.object(stage5, "EXPECTED_QA_COUNT", 1):
                stage5._read_qa_for_inference(manifest)
        _args, kwargs = mock_read.call_args
        self.assertEqual(kwargs.get("columns"), stage5.INFERENCE_COLUMNS)
        for forbidden in ("answer", "correct_option", "evidence_anchors_json", "temporal_anchors_json", "contamination_foil"):
            self.assertNotIn(forbidden, stage5.INFERENCE_COLUMNS)

    def test_score_columns_never_include_question_or_options(self):
        for forbidden in ("question", "options_json"):
            self.assertNotIn(forbidden, stage5.SCORE_COLUMNS)


class SealedPredictionCountTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.pred_path = Path(self.tmpdir.name) / "predictions.jsonl"
        self.pred_meta_path = Path(self.tmpdir.name) / "predictions_meta.json"
        self.snap_path = Path(self.tmpdir.name) / "snapshot.jsonl"
        self.snap_meta_path = Path(self.tmpdir.name) / "snapshot_meta.json"
        self.patches = [
            patch.object(stage5, "PREDICTIONS_JSONL", self.pred_path),
            patch.object(stage5, "PREDICTIONS_META", self.pred_meta_path),
            patch.object(stage5, "SNAPSHOT_JSONL", self.snap_path),
            patch.object(stage5, "SNAPSHOT_META", self.snap_meta_path),
        ]
        for p in self.patches:
            p.start()
        self.snap_path.write_text('{"a": 1}\n')
        snap_checksum = stage5._sha256_file(self.snap_path)
        self.snap_meta_path.write_text(json.dumps({"snapshot_sha256": snap_checksum}))
        self.snap_checksum = snap_checksum

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmpdir.cleanup()

    def _write_predictions(self, n: int):
        rows = [{"qa_id": f"Q{i}"} for i in range(n)]
        self.pred_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        checksum = stage5._sha256_bytes(self.pred_path.read_bytes())
        self.pred_meta_path.write_text(json.dumps({
            "predictions_sha256": checksum, "snapshot_sha256": self.snap_checksum,
        }))

    def test_rejects_wrong_prediction_count(self):
        self._write_predictions(3)
        with patch.object(stage5, "EXPECTED_QA_COUNT", 51):
            with self.assertRaises(RuntimeError):
                stage5._load_sealed_predictions()

    def test_accepts_exact_expected_count(self):
        self._write_predictions(5)
        with patch.object(stage5, "EXPECTED_QA_COUNT", 5):
            predictions, meta = stage5._load_sealed_predictions()
        self.assertEqual(len(predictions), 5)


class PackContextInvariantsTest(unittest.TestCase):
    """Stage 5 reuses Stage 4.9's pack_context verbatim for the atomic-
    provenance / budget / dedup guarantees the checks ask for."""

    def test_derived_candidate_without_all_valid_sources_is_excluded(self):
        from research.stage4_9_hybrid_experiment import pack_context

        raw_by_source = {"t1": Candidate("raw:t1", "raw", "Alice: hi", ("t1",), observed_at="2026-01-01")}
        derived = Candidate("flat:0", "derived:assertion", "claim", ("t1", "t_missing"))
        context, meta = pack_context([derived], raw_by_source)
        self.assertEqual(context, "")
        self.assertEqual(meta["candidate_ids"], [])

    def test_derived_candidate_with_all_valid_sources_is_included(self):
        from research.stage4_9_hybrid_experiment import pack_context

        raw_by_source = {"t1": Candidate("raw:t1", "raw", "Alice: hi", ("t1",), observed_at="2026-01-01")}
        derived = Candidate("flat:0", "derived:assertion", "claim text", ("t1",))
        context, meta = pack_context([derived], raw_by_source)
        self.assertIn("t1", context)
        self.assertEqual(meta["candidate_ids"], ["flat:0"])

    def test_context_words_never_exceed_budget(self):
        from research.stage4_9_hybrid_experiment import pack_context

        raw_by_source = {
            f"t{i}": Candidate(f"raw:t{i}", "raw", " ".join(["word"] * 50), (f"t{i}",), observed_at="2026-01-01")
            for i in range(20)
        }
        candidates = [raw_by_source[f"t{i}"] for i in range(20)]
        _context, meta = pack_context(candidates, raw_by_source, budget=100)
        self.assertLessEqual(meta["context_words"], 100)

    def test_duplicate_raw_turn_id_renders_once(self):
        from research.stage4_9_hybrid_experiment import pack_context

        raw_by_source = {"t1": Candidate("raw:t1", "raw", "Alice: hi", ("t1",), observed_at="2026-01-01")}
        same_candidate_twice = [raw_by_source["t1"], raw_by_source["t1"]]
        context, meta = pack_context(same_candidate_twice, raw_by_source)
        self.assertEqual(context.count("[[t1]]"), 1)
        self.assertEqual(meta["source_ids"], ["t1"])


class FlatAndVersionedSharedCorpusTest(unittest.TestCase):
    def test_flat_and_versioned_derive_from_the_same_validated_assertions(self):
        conversations = pd.DataFrame([
            {"network_id": "grp_test", "session_id": "grp_test_s01", "session_index": 0,
             "turn_id": "grp_test_s01_t000", "timestamp": "2026-01-01T00:00:00",
             "speaker_display_name": "Alice", "message": "I moved to Paris"},
        ])
        cache = {("grp_test", "grp_test_s01"): {"network_id": "grp_test", "session_id": "grp_test_s01", "raw_items": [{
            "viewpoint_owner": "Alice", "subject": "Alice", "facet": "location",
            "assertion_text": "Alice moved to Paris", "modality": "self_report", "confidence": 0.9,
            "normalized_value": "Paris", "source_turn_ids": ["grp_test_s01_t000"],
        }]}}
        slot_resolution = {("grp_test", "grp_test_s01", 0): {
            "topic_key": "alice_location", "temporal_scope": "unspecified", "temporal_precision": "unknown",
        }}
        flat, versioned, stats = stage5.build_memory_candidates(conversations, ["grp_test"], cache, slot_resolution)
        self.assertEqual(len(flat), 1)
        self.assertEqual(len(versioned), 1)
        self.assertEqual(stats["validated_assertion_count"], 1)
        self.assertIn("Alice moved to Paris", flat[0].text)
        self.assertIn("Alice moved to Paris", versioned[0].text)
        self.assertEqual(flat[0].source_ids, versioned[0].source_ids)


class ScoreNeverCallsGeneratorTest(unittest.TestCase):
    def test_score_never_calls_answer_generator(self):
        with patch("research.stage5_heldout._load_sealed_predictions") as mock_load, \
             patch("research.stage5_heldout._read_heldout_manifest") as mock_manifest, \
             patch("research.stage5_heldout.pd.read_parquet") as mock_read_parquet, \
             patch.object(stage5.s48, "cached_answer") as mock_cached_answer, \
             patch("research.socialmembench_pilot._answer") as mock_answer, \
             patch.object(stage5, "_render_score_report", return_value=""), \
             patch.object(stage5, "RESULTS_JSONL", MagicMock()), \
             patch.object(stage5, "REPORT_MD", MagicMock()), \
             patch.object(stage5, "SCORE_META", MagicMock()):
            mock_load.return_value = ([{
                "qa_id": "Q1_x", "network_id": "grp_0d1e2f3a", "query_type": "Q1",
                "answers": {"RAW": "a", "RAW+FLAT": "b", "RAW+VERSIONED": "c"},
                "cited_ids": {"RAW": [], "RAW+FLAT": [], "RAW+VERSIONED": []},
                "exposed_source_ids": {"RAW": [], "RAW+FLAT": [], "RAW+VERSIONED": []},
                "diagnostics": {},
            }], {})
            mock_manifest.return_value = {"qa": [{"qa_id": "Q1_x"}]}
            gold_df = pd.DataFrame([{
                "qa_id": "Q1_x", "answer": "gold", "answer_format": "multiple_choice",
                "correct_option": "A", "evidence_anchors_json": "[]",
            }])
            mock_read_parquet.return_value = gold_df
            stage5.score()
            mock_cached_answer.assert_not_called()
            mock_answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
