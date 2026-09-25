import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from research.socialmembench_pilot import (
    Candidate, _cited_ids, _option_lines, _rrf_ids, _set_metrics,
    EXTRACT_WORKERS, _validate_extracted_items, build_cell_assertion_cache,
    materialize_cell_registries, render_context,
)


class SocialMemBenchPilotTest(unittest.TestCase):
    def setUp(self):
        self.raw = Candidate(
            candidate_id="raw:t1", kind="raw", text="Alice: I moved.",
            source_ids=("t1",), asserted_by=("Alice",), observed_at="2026-01-01",
        )
        self.memory = Candidate(
            candidate_id="memory:c1", kind="derived:temporal_shift",
            text="Alice changed where she lives.", source_ids=("t1", "t2"),
            asserted_by=("Alice",), entities=("Alice",),
        )
        self.raw_by_source = {
            "t1": self.raw,
            "t2": Candidate("raw:t2", "raw", "Alice: I returned.", ("t2",)),
        }

    def test_provenance_is_hidden_until_expansion(self):
        plain, plain_sources = render_context([self.memory], self.raw_by_source, False)
        expanded, expanded_sources = render_context([self.memory], self.raw_by_source, True)
        self.assertNotIn("[[t1]]", plain)
        self.assertEqual(plain_sources, set())
        self.assertIn("[[t1]]", expanded)
        self.assertEqual(expanded_sources, {"t1", "t2"})

    def test_evidence_metrics_do_not_treat_empty_as_perfect(self):
        self.assertEqual(_set_metrics(set(), {"t1"}), {"recall": 0.0, "precision": 0.0})
        self.assertEqual(_set_metrics({"t1", "noise"}, {"t1"}), {"recall": 1.0, "precision": 0.5})

    def test_citations_are_stable_raw_ids(self):
        self.assertEqual(_cited_ids("Answer [[t1]] and [[t2]]. Again [[t1]]."), {"t1", "t2"})

    def test_rrf_preserves_both_retrieval_lanes(self):
        ranked = _rrf_ids([["raw:a", "raw:b"], ["memory:c", "raw:a"]], 3)
        self.assertEqual(ranked[0][0], "raw:a")
        self.assertEqual({item[0] for item in ranked}, {"raw:a", "raw:b", "memory:c"})

    def test_options_accept_all_official_shapes(self):
        self.assertEqual(_option_lines({"A": "Alice"}), ["A. Alice"])
        self.assertEqual(_option_lines(["A) Alice"]), ["A) Alice"])
        self.assertEqual(
            _option_lines([{"option": "A", "name": "Alice"}]), ["A. Alice"],
        )

    def test_extracted_memory_requires_real_provenance(self):
        rows = {"t1": SimpleNamespace(speaker_display_name="Alice", timestamp="2026-01-01")}
        candidates = _validate_extracted_items("n1", "s1", [
            {"text": "Alice moved.", "kind": "state", "source_turn_ids": ["t1", "fake"],
             "asserted_by": ["Alice"], "entities": ["Alice"]},
            {"text": "Unsupported.", "kind": "state", "source_turn_ids": ["fake"]},
        ], rows)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source_ids, ("t1",))

    def test_cache_with_extra_networks_does_not_leak_into_materialization_scope(self):
        """A real incident during Stage 4: build_cell_assertion_cache was
        called against the FULL conversations DataFrame before filtering to
        the selected networks, and it started extracting sessions well
        outside the intended scope. materialize_cell_registries itself must
        never let a cache dict holding OTHER networks' sessions leak into
        the registries it returns for a narrower `conversations` slice."""
        conversations = pd.DataFrame([
            {"network_id": "net_a", "session_id": "net_a_s01", "session_index": 0, "turn_id": "net_a_s01_t000",
             "timestamp": "2026-01-01T00:00:00", "speaker_display_name": "Alice", "message": "hi"},
            {"network_id": "net_b", "session_id": "net_b_s01", "session_index": 0, "turn_id": "net_b_s01_t000",
             "timestamp": "2026-01-01T00:00:00", "speaker_display_name": "Bob", "message": "hi"},
        ])
        cache = {
            ("net_a", "net_a_s01"): {"network_id": "net_a", "session_id": "net_a_s01", "raw_items": [{
                "viewpoint_owner": "Alice", "subject": "Alice", "facet": "location",
                "assertion_text": "Alice is in Paris", "modality": "self_report", "confidence": 0.9,
                "normalized_value": "Paris", "source_turn_ids": ["net_a_s01_t000"],
            }]},
            ("net_b", "net_b_s01"): {"network_id": "net_b", "session_id": "net_b_s01", "raw_items": [{
                "viewpoint_owner": "Bob", "subject": "Bob", "facet": "location",
                "assertion_text": "Bob is in Berlin", "modality": "self_report", "confidence": 0.9,
                "normalized_value": "Berlin", "source_turn_ids": ["net_b_s01_t000"],
            }]},
        }
        slot_resolution = {
            ("net_a", "net_a_s01", 0): {"topic_key": "location_slot", "temporal_scope": "unspecified", "temporal_precision": "unknown"},
            ("net_b", "net_b_s01", 0): {"topic_key": "location_slot", "temporal_scope": "unspecified", "temporal_precision": "unknown"},
        }

        # Only net_a is "selected" -- conversations only has net_a rows,
        # even though the cache dict (as returned by build_cell_assertion_cache
        # on a wider call) still has net_b too.
        net_a_only = conversations[conversations.network_id == "net_a"]
        registries, stats, _, validated_assertions = materialize_cell_registries(net_a_only, cache, slot_resolution)
        self.assertEqual(set(registries.keys()), {"net_a"})
        self.assertEqual(stats["raw_items"], 1)  # net_b's item was never touched
        self.assertEqual(set(validated_assertions.keys()), {"net_a"})

    def test_extraction_stops_after_first_failed_worker_batch(self):
        conversations = pd.DataFrame([
            {
                "network_id": "net_a", "session_id": f"s{index}",
                "turn_id": f"t{index}", "timestamp": "2026-01-01T00:00:00",
                "speaker_display_name": "Alice", "message": "hi",
            }
            for index in range(EXTRACT_WORKERS + 3)
        ])
        with TemporaryDirectory() as directory, patch(
            "research.socialmembench_pilot._extract_session_assertions",
            side_effect=RuntimeError("rate limited"),
        ) as extract:
            with self.assertRaisesRegex(RuntimeError, "rate limited"):
                build_cell_assertion_cache(conversations, Path(directory) / "cache.jsonl")
        self.assertEqual(extract.call_count, EXTRACT_WORKERS)


if __name__ == "__main__":
    unittest.main()
