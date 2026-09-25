import itertools
import json
import tempfile
import unittest
from pathlib import Path

from research import evermembench_chronological_order_control as chrono
from research import evermembench_temporal_budget_control as control
from research.paper_benchmark_common import SearchDoc, sha256_bytes


def raw(message_id: str, stamp: str, body: str) -> SearchDoc:
    return SearchDoc(f"raw:{message_id}", body, f"[{stamp}][Group: G][Speaker: Ann]{body}", (message_id,), "raw")


RAW = {
    "m1": raw("m1", "2025-03-01 09:00:00", "kickoff"),
    "m2": raw("m2", "2025-03-04 17:30:00", "done"),
    "m3": raw("m3", "2025-03-04 08:00:00", "review"),
    "m4": raw("m4", "2025-02-27", "draft"),
}
EVENT = SearchDoc("event:01:01:evt1", "idx", "[DERIVED EVENT / owner=Ann / subject=Task]\nstarted\n  [SOURCE m1] x",
                  ("m2", "m1", "missing"), "event")
DOCS = [RAW["m2"], EVENT, RAW["m3"], RAW["m4"]]


class TimestampTest(unittest.TestCase):
    def test_raw_timestamp_reads_rendered_prefix(self):
        self.assertEqual(chrono.raw_timestamp(RAW["m1"]), "2025-03-01 09:00:00")

    def test_date_only_prefix_counts_as_midnight(self):
        self.assertEqual(chrono.raw_timestamp(RAW["m4"]), "2025-02-27 00:00:00")

    def test_missing_prefix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "without a timestamp"):
            chrono.raw_timestamp(SearchDoc("raw:x", "x", "no stamp", ("x",), "raw"))

    def test_event_uses_earliest_known_source(self):
        self.assertEqual(chrono.doc_timestamp(EVENT, RAW), "2025-03-01 09:00:00")

    def test_event_without_known_sources_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "without timestamped sources"):
            chrono.doc_timestamp(SearchDoc("event:01:01:evt2", "idx", "x", ("missing",), "event"), RAW)


class OrderingTest(unittest.TestCase):
    def test_ascending_timestamps(self):
        ordered = [doc.doc_id for _stamp, doc in chrono.chronological(DOCS, RAW)]
        self.assertEqual(ordered, ["raw:m4", "event:01:01:evt1", "raw:m3", "raw:m2"])

    def test_order_does_not_depend_on_input_order(self):
        expected = [doc.doc_id for _s, doc in chrono.chronological(DOCS, RAW)]
        for permutation in itertools.permutations(DOCS):
            self.assertEqual([doc.doc_id for _s, doc in chrono.chronological(list(permutation), RAW)], expected)

    def test_ties_break_by_document_id(self):
        a, b = raw("a", "2025-03-01 09:00:00", "x"), raw("b", "2025-03-01 09:00:00", "y")
        self.assertEqual([d.doc_id for _s, d in chrono.chronological([b, a], {})], ["raw:a", "raw:b"])

    def test_duplicate_document_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-unique"):
            chrono.chronological([RAW["m1"], RAW["m1"]], RAW)


class PurePermutationTest(unittest.TestCase):
    def setUp(self):
        self.ordered = [doc for _s, doc in chrono.chronological(DOCS, RAW)]
        self.source_context = control.render_context(DOCS)
        self.ordered_context = control.render_context(self.ordered)

    def test_valid_permutation_passes_and_adds_nothing(self):
        chrono.verify_pure_permutation(DOCS, self.ordered, self.source_context, self.ordered_context)
        self.assertNotIn("[Day", self.ordered_context)
        self.assertEqual(self.ordered_context, "\n".join(chrono.blocks(self.ordered)))
        self.assertEqual(sorted(chrono.blocks(self.ordered)), sorted(chrono.blocks(DOCS)))
        self.assertEqual(len(self.ordered_context), len(self.source_context))

    def test_changed_document_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ids differ"):
            chrono.verify_pure_permutation(DOCS, self.ordered[:-1] + [RAW["m1"]], self.source_context,
                                           control.render_context(self.ordered[:-1] + [RAW["m1"]]))

    def test_changed_document_text_is_rejected(self):
        tampered = [SearchDoc(d.doc_id, d.index_text, d.rendered + " ", d.source_ids, d.kind) for d in self.ordered]
        with self.assertRaisesRegex(ValueError, "text or sources changed"):
            chrono.verify_pure_permutation(DOCS, tampered, self.source_context, control.render_context(tampered))

    def test_added_prefix_is_rejected(self):
        relative = chrono.render_relative(chrono.chronological(DOCS, RAW))
        with self.assertRaisesRegex(ValueError, "ordered context"):
            chrono.verify_pure_permutation(DOCS, self.ordered, self.source_context, relative)

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            chrono.verify_pure_permutation(DOCS + [RAW["m2"]], self.ordered, self.source_context, self.ordered_context)

    def test_prompt_frame_is_the_text_around_the_context(self):
        template = "Context:\n{context}\n\nQuestion: {question}"
        new = template.format(context=self.ordered_context, question="How long?")
        old = template.format(context=self.source_context, question="How long?")
        self.assertEqual(chrono.prompt_frame(new, self.ordered_context), chrono.prompt_frame(old, self.source_context))


class RelativeRenderingTest(unittest.TestCase):
    def test_day_offsets_follow_the_ordered_documents(self):
        lines = chrono.render_relative(chrono.chronological(DOCS, RAW)).split("\n- ")
        self.assertTrue(lines[0].startswith("- [Day +0] [2025-02-27]"))
        self.assertTrue(lines[1].startswith("[Day +2] [DERIVED EVENT"))
        self.assertTrue(lines[2].startswith("[Day +5] [2025-03-04 08:00:00]"))
        self.assertTrue(lines[3].startswith("[Day +5] [2025-03-04 17:30:00]"))

    def test_empty_context_matches_official_placeholder(self):
        self.assertEqual(chrono.render_relative([]), "(No memories retrieved)")


class ResumeAndFreezeTest(unittest.TestCase):
    def job(self, key: str, condition: str, prompt: str) -> dict:
        return {"question": {"qa_key": key}, "condition": condition, "prompt": prompt}

    def test_cached_predictions_are_skipped(self):
        jobs = [self.job("q1", "A", "p1"), self.job("q1", "B", "p2"), self.job("q2", "A", "p3")]
        pending = chrono.pending_jobs(jobs, {("q1", "A")})
        self.assertEqual([(j["question"]["qa_key"], j["condition"]) for j in pending], [("q1", "B"), ("q2", "A")])

    def test_cached_prediction_with_other_prompt_is_refused(self):
        jobs = [self.job("q1", "A", "p1")]
        chrono.check_resumed_predictions([{"qa_key": "q1", "condition": "A", "prompt_sha256": sha256_bytes(b"p1")}], jobs)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            chrono.check_resumed_predictions([{"qa_key": "q1", "condition": "A", "prompt_sha256": sha256_bytes(b"other")}], jobs)

    def test_inputs_are_frozen_on_first_write(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            records = [{"qa_key": "q1", "condition": "A", "ordered_doc_ids": ["raw:a"]}]
            first = chrono.write_or_verify_inputs(run_dir, records)
            self.assertEqual(chrono.write_or_verify_inputs(run_dir, records), first)
            with self.assertRaisesRegex(RuntimeError, "frozen inputs differ"):
                chrono.write_or_verify_inputs(run_dir, [{**records[0], "ordered_doc_ids": ["raw:b"]}])

    def test_changed_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            chrono.ensure_run_manifest(run_dir, {"runner_sha256": "a"})
            chrono.ensure_run_manifest(run_dir, {"runner_sha256": "a"})
            self.assertEqual(json.loads((run_dir / "run_manifest.json").read_text()), {"runner_sha256": "a"})
            with self.assertRaisesRegex(RuntimeError, "manifest mismatch"):
                chrono.ensure_run_manifest(run_dir, {"runner_sha256": "b"})


class SignFlipTest(unittest.TestCase):
    def test_all_positive_five_clusters_gives_minimum_p(self):
        self.assertAlmostEqual(chrono.sign_flip_p([0.1, 0.2, 0.05, 0.3, 0.1]), 2 / 32)

    def test_symmetric_values_give_p_one(self):
        self.assertEqual(chrono.sign_flip_p([0.1, -0.1]), 1.0)


if __name__ == "__main__":
    unittest.main()
