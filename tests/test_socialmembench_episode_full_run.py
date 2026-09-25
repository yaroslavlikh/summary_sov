import unittest

from research import socialmembench_episode_full_run as run


class EpisodeSerializationTest(unittest.TestCase):
    def test_candidate_keeps_active_history_and_all_provenance(self):
        events = []
        for index in range(7):
            events.append({
                "decision": "create" if index == 0 else "attach_augment",
                "event": {
                    "event_text": f"state {index}", "source_turn_ids": [f"t{index}"],
                    "observed_at": f"2026-01-{index + 1:02d}",
                },
            })
        candidate = run._episode_candidate({
            "episode_id": "ep_net_1", "network_id": "net", "viewpoint_owner": "Alice",
            "subject": "plan", "events": events,
        })
        self.assertNotIn("state 0", candidate.text)
        self.assertIn("state 6", candidate.text)
        self.assertEqual(candidate.source_ids, tuple(f"t{i}" for i in range(7)))
        self.assertEqual(candidate.asserted_by, ("Alice",))


class CostSummaryTest(unittest.TestCase):
    def test_no_usage_is_zero_cost(self):
        original = run.USAGE
        try:
            run.USAGE = original.with_name("definitely_missing_usage_test.jsonl")
            lines, total = run._usage_summary()
            self.assertEqual(total, 0.0)
            self.assertEqual(len(lines), 3)
        finally:
            run.USAGE = original


if __name__ == "__main__":
    unittest.main()
