import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from research import evermembench_temporal_budget_control as control
from research.paper_benchmark_common import SearchDoc


def raw(doc_id: str, text: str) -> SearchDoc:
    return SearchDoc(f"raw:{doc_id}", text, text, (doc_id,), "raw")


def words(text: str) -> int:
    return len(text.split())


class ContextSelectionTest(unittest.TestCase):
    def test_render_matches_runner_format(self):
        docs = [raw("a", "one"), raw("b", "two")]
        self.assertEqual(control.render_context(docs), "- one\n- two")
        self.assertEqual(control.render_context([]), "(No memories retrieved)")

    def test_unique_sources_remove_duplicates_across_raw_and_episode(self):
        episode = SearchDoc("episode:t:e1", "x", "x", ("a", "b"), "episode")
        self.assertEqual(control.unique_source_ids([raw("a", "one"), episode]), ["a", "b"])

    def test_count_matched_takes_first_distinct_documents(self):
        ranked = [raw("a", "1"), raw("a", "1"), raw("b", "2"), raw("c", "3")]
        self.assertEqual([d.doc_id for d in control.select_count_matched(ranked, 2)], ["raw:a", "raw:b"])
        self.assertEqual(len(control.select_count_matched(ranked, 10)), 3)

    def test_token_matched_stays_under_target_when_overshoot_is_farther(self):
        ranked = [raw("a", "w w w"), raw("b", "w w w"), raw("c", "w w w w w w")]
        # "- w w w" = 4 words, two docs = 8, three docs = 15; target 10 -> 8 is closer than 15
        chosen, tokens = control.select_token_matched(ranked, 10, words)
        self.assertEqual([d.doc_id for d in chosen], ["raw:a", "raw:b"])
        self.assertEqual(tokens, 8)

    def test_token_matched_includes_overshoot_when_strictly_closer(self):
        ranked = [raw("a", "w w w"), raw("b", "w w w")]
        chosen, tokens = control.select_token_matched(ranked, 7, words)
        self.assertEqual(len(chosen), 2)
        self.assertEqual(tokens, 8)

    def test_token_matched_takes_first_document_even_if_it_overshoots(self):
        chosen, _ = control.select_token_matched([raw("a", "w w w w w")], 2, words)
        self.assertEqual(len(chosen), 1)


class SealedSourceGuardTest(unittest.TestCase):
    def test_missing_vector_cache_is_never_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "refusing to recompute"):
                control.question_vectors(Path(directory), ["q"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_question_vectors_use_runner_batch_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            texts = [f"q{i}" for i in range(130)]
            for start in (0, 128):
                batch = texts[start:start + 128]
                key = control.sha256_bytes(json.dumps({"model": control.EMBED_MODEL, "texts": batch},
                                                      ensure_ascii=False).encode())
                np.save(root / f"batch-{key[:24]}.npy", np.full((len(batch), 3), start, dtype=np.float32))
            vectors = control.question_vectors(root, texts)
            self.assertEqual(vectors.shape, (130, 3))
            self.assertEqual(vectors[129, 0], 128)

    def test_run_manifest_refuses_changed_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for name in ("predictions.jsonl", "results.jsonl", "episodes.jsonl"):
                (source / name).write_text("{}\n")
            with mock.patch.object(control, "SOURCE_DIR", source):
                control.ensure_run_manifest(root / "run", ["a"])
                control.ensure_run_manifest(root / "run", ["a"])
                with self.assertRaisesRegex(RuntimeError, "run manifest mismatch"):
                    control.ensure_run_manifest(root / "run", ["a", "b"])


if __name__ == "__main__":
    unittest.main()
