"""Tests for ask_metrics.py and llm/answer_judges.py -- the metrics recorded on
every production answer and the judges used by tests/evals/compare_episodes.py.
No database, no model calls: the judge model is stubbed.
"""
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import ask_metrics
from llm import answer_judges


def _row(row_id, message_id):
    return (row_id, message_id, None, "Марина", "marina", "ciphertext")


class MechanicalMetricsTests(unittest.TestCase):
    def _state(self, **overrides):
        state = {
            "candidate_ids": [1, 2, 3], "candidate_event_ids": [7], "match_event_ids": [7],
            "window_rows": [_row(1, 101), _row(5, 105), _row(9, 109)],
            "episode_row_ids": [5, 9],
            "context_lines": ["[1] Марина: перенос релиза", "[2] Олег: готово", "[3] Олег: тесты прогнал"],
            "episode_context": "[ЭПИЗОД] ...",
            "answer_plain": "Олег закончил модуль [2][2][7]",
        }
        state.update(overrides)
        return state

    def test_counts_delivery_and_citations(self):
        metrics = ask_metrics.collect(self._state(), "episodes", latency_seconds=1.234)
        self.assertEqual(metrics["memory_condition"], 1.0)
        self.assertEqual(metrics["candidates_memory"], 1.0)
        self.assertEqual(metrics["context_messages"], 3.0)
        self.assertEqual(metrics["context_from_memory"], 2.0)
        self.assertEqual(metrics["citations"], 1.0)           # [2] counted once
        self.assertEqual(metrics["citations_invented"], 1.0)  # [7] is past the end of the context
        self.assertEqual(metrics["citations_from_memory"], 1.0)  # line 2 is row 5, an expanded source
        self.assertEqual(metrics["no_answer"], 0.0)
        self.assertEqual(metrics["latency_seconds"], 1.234)
        self.assertGreater(metrics["context_tokens"], metrics["answer_tokens"])

    def test_raw_condition_has_no_episode_contribution(self):
        metrics = ask_metrics.collect(
            self._state(candidate_event_ids=[], match_event_ids=[], episode_row_ids=[], episode_context=""),
            "raw")
        self.assertEqual(metrics["memory_condition"], 0.0)
        self.assertEqual((metrics["context_from_memory"], metrics["citations_from_memory"]), (0.0, 0.0))
        self.assertNotIn("latency_seconds", metrics)

    def test_refusal_and_style(self):
        metrics = ask_metrics.collect(
            self._state(answer_plain="В истории чата не нашёл ответа на этот вопрос."), "raw")
        self.assertEqual(metrics["no_answer"], 1.0)
        self.assertEqual(metrics["citations"], 0.0)
        markdown = ask_metrics.collect(self._state(answer_plain="**жирный** ответ. И второй."), "raw")
        self.assertEqual(markdown["answer_has_markdown"], 1.0)
        self.assertEqual(markdown["answer_sentences"], 2.0)

    def test_cited_indices(self):
        self.assertEqual(ask_metrics.cited_indices("ответ [1][3][9]", 3), ([1, 3], 1))
        self.assertEqual(ask_metrics.cited_indices(None, 3), ([], 0))

    def test_push_scores_without_langfuse_keys_is_a_noop(self):
        with patch.dict("os.environ", {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""}, clear=False):
            self.assertFalse(ask_metrics.push_scores({"x": 1.0}))


class _Reply:
    def __init__(self, content):
        self.content = content


class JudgeTests(unittest.TestCase):
    def _patch_model(self, *replies):
        model = patch.object(answer_judges, "_judge_model").start()
        model.return_value.invoke.side_effect = [_Reply(r) for r in replies]
        self.addCleanup(patch.stopall)
        return model

    def test_rubric_scores_clamped_and_nulls_skipped(self):
        self._patch_model('{"faithfulness": 0.9, "answer_relevance": 1.4, "attribution": null,'
                          ' "temporal_correctness": "n/a", "context_sufficiency": 0.4, "comment": "ок"}')
        scores = answer_judges.judge_answer("вопрос", "Ярик", "ответ", ["[1] Марина: текст"])
        self.assertEqual(scores["faithfulness"]["value"], 0.9)
        self.assertEqual(scores["answer_relevance"]["value"], 1.0)  # clamped, not dropped
        self.assertNotIn("attribution", scores)       # judge said it does not apply
        self.assertNotIn("temporal_correctness", scores)
        self.assertEqual(scores["context_sufficiency"]["comment"], "ок")

    def test_rubric_without_context_is_not_judged(self):
        self.assertEqual(answer_judges.judge_answer("вопрос", "Ярик", "ответ", []), {})

    def test_unparsable_judge_output_yields_no_scores(self):
        self._patch_model("судья не смог")
        self.assertEqual(answer_judges.judge_answer("вопрос", "Ярик", "ответ", ["[1] текст"]), {})

    def test_pairwise_counts_only_order_consistent_verdicts(self):
        self._patch_model('{"winner": "A"}', '{"winner": "B"}')
        self.assertEqual(answer_judges.judge_pairwise("в", "Ярик", "a", "b", ["[1] текст"])["value"], 1.0)
        patch.stopall()
        self._patch_model('{"winner": "A"}', '{"winner": "A"}')  # same position wins twice -> position bias
        biased = answer_judges.judge_pairwise("в", "Ярик", "a", "b", ["[1] текст"])
        self.assertEqual(biased["value"], 0.0)
        self.assertIn("непоследователен", biased["comment"])

    def test_pairwise_with_a_failed_judge_call_is_dropped(self):
        self._patch_model("нет json", '{"winner": "B"}')
        self.assertEqual(answer_judges.judge_pairwise("в", "Ярик", "a", "b", ["[1] текст"]), {})


if __name__ == "__main__":
    unittest.main()
