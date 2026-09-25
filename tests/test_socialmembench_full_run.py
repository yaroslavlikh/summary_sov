import unittest

import numpy as np

from research.socialmembench_pilot import Candidate
from research.socialmembench_full_run import (
    CosineIndex, bootstrap_network_ci, pack_context, parse_choice, qa_key,
)


class ChoiceParserTest(unittest.TestCase):
    def test_official_style_variants(self):
        for text in ("A", "A.", "Answer: A", "The answer is A", "Option A"):
            self.assertEqual(parse_choice(text, {"A", "B"}), "A")

    def test_does_not_accept_invalid_letter(self):
        self.assertIsNone(parse_choice("Answer: Z", {"A", "B"}))


class QAKeyTest(unittest.TestCase):
    def test_same_qa_id_in_different_networks_stays_distinct(self):
        self.assertNotEqual(qa_key("net_a", "Q1_same"), qa_key("net_b", "Q1_same"))


class CosineIndexTest(unittest.TestCase):
    def test_top_k_is_shared_cosine_ranking(self):
        candidates = [
            Candidate("raw:1", "raw", "raw", ("t1",)),
            Candidate("flat:net:1", "derived:assertion", "memory", ("t2",)),
        ]
        index = CosineIndex(candidates, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        self.assertEqual(index.search(np.array([0.0, 1.0], dtype=np.float32), 1)[0].candidate_id, "flat:net:1")


class ProvenancePackingTest(unittest.TestCase):
    def test_derived_memory_expands_to_raw_source(self):
        raw = Candidate("raw:t1", "raw", "Alice: moved", ("t1",), observed_at="2026-01-01")
        memory = Candidate(
            "flat:net:1", "derived:assertion", "Alice moved", ("t1",),
            asserted_by=("Alice",), entities=("Alice",),
        )
        context, exposed = pack_context([memory], {"t1": raw}, {"t1": 2})
        self.assertIn("[DERIVED MEMORY]", context)
        self.assertIn("[SOURCE session 2]", context)
        self.assertEqual(exposed, ["t1"])


class BootstrapTest(unittest.TestCase):
    def test_constant_values_have_point_ci(self):
        self.assertEqual(bootstrap_network_ci({"a": 0.5, "b": 0.5}), (0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
