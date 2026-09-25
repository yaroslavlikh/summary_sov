import unittest
import json
import tempfile
from pathlib import Path

import pandas as pd

from research import evermembench_episode_run as ever
from research import groupmembench_episode_run as group
from research.paper_benchmark_common import (
    append_jsonl, build_episode_snapshot, clustered_bootstrap, exact_mcnemar,
    paired_bootstrap, seal_jsonl, usage_summary, verify_sealed_jsonl,
)


class GroupMemBenchAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.raw_by_domain, cls.raw_by_id = group.load_messages(group.DOMAINS)

    def test_complete_official_shape_and_bounded_sessions(self):
        self.assertEqual(len(self.frame), 120_000)
        self.assertEqual(set(self.raw_by_domain), set(group.DOMAINS))
        self.assertTrue(all(len(rows) == 30_000 for rows in self.raw_by_domain.values()))
        self.assertLessEqual(
            self.frame.groupby(["network_id", "session_index"]).size().max(), group.CHUNK_SIZE,
        )
        self.assertEqual(set(self.frame.turn_id.astype(str)), set(self.raw_by_id))

    def test_primary_split_is_filtered_and_fixed(self):
        self.assertEqual(group.PRIMARY_DOMAINS, ("Finance", "Technology"))
        self.assertEqual(group.PRIMARY_QTYPES, ("knowledge_update", "temporal"))
        self.assertEqual(len(group.load_questions(group.DOMAINS, include_gold=False)), 745)
        self.assertTrue(all("gold" not in row for row in group.load_questions(group.DOMAINS, include_gold=False)))


class EverMemBenchAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.raw_by_topic, cls.raw_by_id = ever.load_messages()

    def test_complete_official_shape(self):
        self.assertEqual(len(self.frame), 51_023)
        self.assertEqual(self.frame[["network_id", "session_index"]].drop_duplicates().shape[0], 3_570)
        self.assertEqual(len(ever.load_questions(include_gold=False)), 2_400)

    def test_every_released_reference_resolves_to_a_raw_turn(self):
        references = [source for row in ever.load_questions(include_gold=True) for source in row["gold_source_ids"]]
        self.assertTrue(references)
        self.assertFalse(set(references) - set(self.raw_by_id))

    def test_reference_range_parser(self):
        self.assertEqual(ever._expand_message_indices("2-3, 6-7"), ["2", "3", "6", "7"])


class EverMemBenchOfficialScoringTest(unittest.TestCase):
    def test_fenced_judge_json_is_read_like_official_evaluator(self):
        from research import evermembench_parallel_resume as runner
        self.assertEqual(runner.official_judge_label('```json\n{"label": "CORRECT"}\n```'), (True, "json"))
        self.assertEqual(runner.official_judge_label('Verdict: {"label": "WRONG", "reason": "x"}'), (False, "json"))

    def test_empty_judge_content_is_wrong_under_official_parser(self):
        from research import evermembench_parallel_resume as runner
        self.assertEqual(runner.official_judge_label(""), (False, "substring_fallback"))

    def test_answer_post_processing_matches_official_answerer(self):
        from research import evermembench_parallel_resume as runner
        self.assertEqual(runner.official_answer("B", "multiple_choice"), "B")
        self.assertEqual(runner.official_answer("", "multiple_choice"), "[EMPTY]")
        self.assertEqual(runner.official_answer("I cannot tell", "multiple_choice"), "[INVALID]")
        self.assertEqual(runner.official_answer("", "open_ended"), "[EMPTY]")
        self.assertEqual(runner.official_answer("  Nine days. ", "open_ended"), "Nine days.")

    def test_multiple_choice_scoring_matches_official_evaluator(self):
        from research import evermembench_parallel_resume as runner
        self.assertTrue(runner.official_evaluator_mc("B", "B"))
        self.assertTrue(runner.official_evaluator_mc("C", "C. option text"))
        self.assertFalse(runner.official_evaluator_mc("[EMPTY]", "B"))

    def test_empty_judge_content_is_re_requested_and_never_cached(self):
        import threading
        from types import SimpleNamespace
        from research import evermembench_parallel_resume as runner

        replies = iter(["", '{"label": "CORRECT"}'])

        def create(**kwargs):
            self.assertNotIn("max_tokens", kwargs)
            content = next(replies)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            router = SimpleNamespace(
                label="openrouter", lock=threading.Lock(), values={},
                cache_path=root / "cache.jsonl", usage_path=root / "usage.jsonl",
                client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
            )
            call = runner.judge_chat(router, [{"role": "user", "content": "q"}])
            self.assertEqual(call["attempts"], 2)
            self.assertEqual(len((root / "cache.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((root / "usage.jsonl").read_text().splitlines()), 2)
            self.assertEqual(runner.judge_chat(router, [{"role": "user", "content": "q"}])["attempts"], 0)


class PairedStatisticsTest(unittest.TestCase):
    def test_direction_and_exact_test(self):
        rows = [
            {"qa_key": "a", "condition": "EP", "correct": 1},
            {"qa_key": "a", "condition": "RAW", "correct": 0},
            {"qa_key": "b", "condition": "EP", "correct": 1},
            {"qa_key": "b", "condition": "RAW", "correct": 1},
        ]
        delta, low, high = paired_bootstrap(rows, "EP", "RAW")
        self.assertEqual(delta, 0.5)
        self.assertLessEqual(low, delta)
        self.assertGreaterEqual(high, delta)
        exact = exact_mcnemar(rows, "EP", "RAW")
        self.assertEqual(exact["left_only"], 1)
        self.assertEqual(exact["right_only"], 0)

    def test_clustered_bootstrap_weights_clusters_equally(self):
        rows = []
        for key, cluster, ep, raw in (("a", "x", 1, 0), ("b", "x", 1, 0), ("c", "y", 0, 1)):
            rows += [
                {"qa_key": key, "condition": "EP", "correct": ep, "topic": cluster},
                {"qa_key": key, "condition": "RAW", "correct": raw, "topic": cluster},
            ]
        delta, low, high = clustered_bootstrap(rows, "EP", "RAW", "topic")
        self.assertEqual(delta, 0.0)
        self.assertLessEqual(low, delta)
        self.assertGreaterEqual(high, delta)


class QueryIndependentMemoryTest(unittest.TestCase):
    def test_memory_builder_accepts_conversations_only_and_seals_snapshot(self):
        class EmptyExtractor:
            def chat(self, **_kwargs):
                return '{"items":[]}'

        frame = pd.DataFrame([{
            "network_id": "n", "session_index": 1, "turn_id": "t1",
            "timestamp": "2026-01-01T00:00:00", "speaker_display_name": "Alice",
            "message": "hello",
        }])
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            snapshot = build_episode_snapshot(frame, run_dir, EmptyExtractor(), source_manifest={"test": True})
            self.assertTrue(snapshot.exists())
            metadata = json.loads((run_dir / "episodes_meta.json").read_text())
            self.assertFalse(metadata["qa_read"])
            self.assertEqual(metadata["turn_count"], 1)
            with self.assertRaisesRegex(RuntimeError, "source manifest mismatch"):
                build_episode_snapshot(frame, run_dir, EmptyExtractor(), source_manifest={"test": False})


class ArtifactIntegrityTest(unittest.TestCase):
    def test_complete_jsonl_is_sealed_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, meta = root / "rows.jsonl", root / "rows_meta.json"
            append_jsonl(path, {"qa_key": "q", "condition": "RAW"})
            sealed = seal_jsonl(
                path, meta, expected_rows=1, identity_fields=("qa_key", "condition"),
                metadata={"benchmark": "test"},
            )
            self.assertEqual(sealed["row_count"], 1)
            self.assertEqual(verify_sealed_jsonl(path, meta, expected_rows=1), sealed)

    def test_duplicate_jsonl_identities_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, meta = root / "rows.jsonl", root / "rows_meta.json"
            append_jsonl(path, {"qa_key": "q", "condition": "RAW"})
            append_jsonl(path, {"qa_key": "q", "condition": "RAW"})
            with self.assertRaisesRegex(RuntimeError, "duplicate identities"):
                seal_jsonl(path, meta, expected_rows=2,
                           identity_fields=("qa_key", "condition"), metadata={})

    def test_usage_summary_includes_reproducible_cost_upper_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            append_jsonl(root / "api_usage.jsonl", {
                "model": "gpt-5", "input_tokens": 1_000_000,
                "output_tokens": 1_000_000, "duration_ms": 10,
            })
            self.assertEqual(usage_summary(root)["estimated_usd_upper_bound"], 11.25)


if __name__ == "__main__":
    unittest.main()
