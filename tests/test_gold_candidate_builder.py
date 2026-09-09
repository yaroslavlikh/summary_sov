"""Regression test for a real cross-subject attribution-collapse bug found
via human review of research/gold_candidate_builder.py's v2 dedup:
near-identical claims from two different people --

  Ярик [71811]:  "Крым России"
  Тигмен [71827]: "Крым абсолютно точно часть России"

-- merged by claim-embedding similarity alone into one candidate under
Ярик's subject, with Тигмен's source message silently reassigned as if it
were Ярик's. Exactly the attribution collapse this whole research effort
exists to prevent, reproduced by the tool built to prevent it.

unittest, not pytest -- project convention. No DB/LLM needed: _dedupe only
calls the local embedding model (embeddings.embed), same as
tests/test_memory_facts.py's setup for other DB-free pieces would if any
existed, except this one genuinely doesn't touch Postgres at all.
"""
import sys
import unittest

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

from research.gold_candidate_builder import _dedupe


class DedupeTestCase(unittest.TestCase):
    def test_similar_claims_different_subjects_stay_separate(self):
        candidates = [
            {
                "claim": "Крым России", "kind": "opinion",
                "subject_key": "yarik", "subject_raw": "Ярик Лихачев",
                "source_message_ids": [71811], "support": "fully_supported", "path": "B-raw-chunk",
            },
            {
                "claim": "Крым абсолютно точно часть России", "kind": "opinion",
                "subject_key": "tigmen", "subject_raw": "Саша Тигмен",
                "source_message_ids": [71827], "support": "fully_supported", "path": "B-raw-chunk",
            },
        ]
        deduped = _dedupe(candidates)
        self.assertEqual(len(deduped), 2)
        by_subject = {c["subject_key"]: c["source_message_ids"] for c in deduped}
        self.assertEqual(by_subject["yarik"], [71811])
        self.assertEqual(by_subject["tigmen"], [71827])

    def test_same_subject_different_kind_stays_separate(self):
        candidates = [
            {
                "claim": "Гордей интересуется стрельбой", "kind": "durable_person_fact",
                "subject_key": "gordey", "subject_raw": "Гордей",
                "source_message_ids": [1], "support": "fully_supported", "path": "B-raw-chunk",
            },
            {
                "claim": "Гордей планирует пойти в тир пострелять", "kind": "commitment",
                "subject_key": "gordey", "subject_raw": "Гордей",
                "source_message_ids": [2], "support": "fully_supported", "path": "B-raw-chunk",
            },
        ]
        deduped = _dedupe(candidates)
        self.assertEqual(len(deduped), 2)

    def test_same_subject_same_kind_near_duplicate_still_merges(self):
        candidates = [
            {
                "claim": "Ярослав любит бильярд", "kind": "durable_person_fact",
                "subject_key": "yaroslavlikh", "subject_raw": "Ярослав",
                "source_message_ids": [1], "support": "partially_supported", "path": "A-moments",
            },
            {
                "claim": "Ярослав обожает бильярд", "kind": "durable_person_fact",
                "subject_key": "yaroslavlikh", "subject_raw": "Ярослав",
                "source_message_ids": [2], "support": "fully_supported", "path": "B-raw-chunk",
            },
        ]
        deduped = _dedupe(candidates)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(sorted(deduped[0]["source_message_ids"]), [1, 2])
        self.assertEqual(deduped[0]["support"], "fully_supported")
        self.assertEqual(set(deduped[0]["discovery_paths"]), {"A-moments", "B-raw-chunk"})

    def test_unresolved_subjects_fall_back_to_normalized_raw_name(self):
        """subject_key can be None for genuinely ambiguous/unmatched
        subjects -- dedup must still not conflate two different unresolved
        people just because both have subject_key=None."""
        candidates = [
            {
                "claim": "Живёт в Казани", "kind": "state",
                "subject_key": None, "subject_raw": "Мигель",
                "source_message_ids": [1], "support": "fully_supported", "path": "B-raw-chunk",
            },
            {
                "claim": "Живёт в Казани сейчас", "kind": "state",
                "subject_key": None, "subject_raw": "Дмитрий",
                "source_message_ids": [2], "support": "fully_supported", "path": "B-raw-chunk",
            },
        ]
        deduped = _dedupe(candidates)
        self.assertEqual(len(deduped), 2)


if __name__ == "__main__":
    unittest.main()
