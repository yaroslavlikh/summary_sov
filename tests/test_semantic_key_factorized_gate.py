import unittest

from research.semantic_key_factorized_gate import normalize_factorized_output


class FactorizedGateTest(unittest.TestCase):
    def test_all_identity_axes_are_required(self):
        base = {
            "candidate_index": 0, "same_holder": "YES", "same_viewpoint": "YES",
            "same_topic": "YES", "same_dimension": "YES", "same_slot": "YES",
            "confidence": .9,
        }
        self.assertEqual(normalize_factorized_output({"comparisons": [base]}, 1)[0]["match_score"], .9)
        changed = {**base, "same_dimension": "NO"}
        self.assertEqual(normalize_factorized_output({"comparisons": [changed]}, 1)[0]["match_score"], 0)

    def test_not_applicable_viewpoint_is_allowed(self):
        item = {
            "candidate_index": 0, "same_holder": "YES", "same_viewpoint": "NOT_APPLICABLE",
            "same_topic": "YES", "same_dimension": "YES", "same_slot": "YES",
            "confidence": .8,
        }
        self.assertEqual(normalize_factorized_output({"comparisons": [item]}, 1)[0]["match_score"], .8)

    def test_malformed_or_out_of_range_rows_do_not_link(self):
        output = {"comparisons": [{"candidate_index": 4, "same_holder": "YES"}]}
        self.assertEqual(normalize_factorized_output(output, 1), {})


if __name__ == "__main__":
    unittest.main()
