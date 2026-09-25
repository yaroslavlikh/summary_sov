import unittest

from research import evermembench_temporal_budget_control as control
from research import final_sprint_evermembench as sprint
from research import final_sprint_groupmembench_repair as repair
from research.paper_benchmark_common import SearchDoc


RAW = {
    "m1": SearchDoc("raw:m1", "a", "[2025-03-01 09:00:00][Group: G][Speaker: Ann]kickoff", ("m1",), "raw"),
    "m2": SearchDoc("raw:m2", "b", "[2025-03-04 17:30:00][Group: G][Speaker: Bob]done", ("m2",), "raw"),
}
EVENT = SearchDoc("event:01:01:evt1", "Viewpoint owner: Ann. Subject: Migration. Migration started",
                  "[DERIVED EVENT / owner=Ann / subject=Migration]\nMigration started\n  [SOURCE m1] " + RAW["m1"].rendered,
                  ("m1", "missing"), "event")
PROMPTS = {"open_ended": "Context:\n{context}\nQ: {question}", "multiple_choice": "Context:\n{context}\nQ: {question}\n{options}"}


class RenderingTest(unittest.TestCase):
    def test_event_documents_become_their_raw_sources_only(self):
        context = sprint.render_without_description([EVENT, RAW["m2"]], RAW)
        self.assertEqual(context.splitlines(), [f"- {RAW['m1'].rendered}", f"- {RAW['m2'].rendered}"])
        for leaked in ("DERIVED", "Migration started", "owner=", "subject="):
            self.assertNotIn(leaked, context)

    def test_empty_context_matches_official_placeholder(self):
        self.assertEqual(sprint.render_without_description([], RAW), "(No memories retrieved)")

    def test_prompt_follows_question_type(self):
        open_row = {"question_type": "open_ended", "question": "How long?"}
        mc_row = {"question_type": "multiple_choice", "question": "Which?", "options": {"B": "two", "A": "one"}}
        self.assertEqual(sprint.build_prompt(PROMPTS, open_row, "ctx"), "Context:\nctx\nQ: How long?")
        self.assertEqual(sprint.build_prompt(PROMPTS, mc_row, "ctx"), "Context:\nctx\nQ: Which?\nA. one\nB. two")

    def test_answer_key_matches_the_frozen_controls(self):
        self.assertEqual(sprint.chat_key("prompt", 1000), control.answer_chat_key("prompt"))


class HydeTest(unittest.TestCase):
    def test_prompt_uses_only_the_question_text(self):
        prompt = sprint.hyde_prompt("How many days did the migration take?")
        self.assertTrue(prompt.endswith("Question: How many days did the migration take?"))
        for forbidden in ("gold", "evidence", "Temporal", "TP", "option"):
            self.assertNotIn(forbidden, prompt)


class TokenMatchedChronoTest(unittest.TestCase):
    """C2 selects by the frozen budget rule, then presents the result in time order."""

    def setUp(self):
        self.count = len  # deterministic stand-in for the tiktoken counter
        self.ranked = [RAW["m2"], RAW["m1"]]  # reverse chronological ranking

    def test_selection_is_reordered_by_timestamp(self):
        docs = sprint.token_matched_chrono(self.ranked, 10_000, RAW, self.count)
        self.assertEqual([d.doc_id for d in docs], ["raw:m1", "raw:m2"])

    def test_budget_cuts_before_ordering(self):
        target = self.count(control.render_context([RAW["m2"]]))
        docs = sprint.token_matched_chrono(self.ranked, target, RAW, self.count)
        self.assertEqual([d.doc_id for d in docs], ["raw:m2"])

    def test_never_returns_an_empty_context(self):
        docs = sprint.token_matched_chrono(self.ranked, 1, RAW, self.count)
        self.assertEqual(len(docs), 1)

    def test_condition_is_distinct_from_the_registered_hyde_arm(self):
        self.assertNotEqual(sprint.CONDITIONS["C2"], sprint.CONDITIONS["C"])
        self.assertNotEqual(sprint.RUN_DIRS["C2"], sprint.RUN_DIRS["C"])


class RepairTest(unittest.TestCase):
    def test_budget_exhaustion_is_diagnosed(self):
        record = repair.classify_original("", {"output_tokens": 2048})
        self.assertTrue(record["answer_empty"])
        self.assertIn("exhausted", record["diagnosis"])
        self.assertIn("below", repair.classify_original("", {"output_tokens": 300})["diagnosis"])

    def test_completion_requires_text_and_no_length_cut(self):
        self.assertTrue(repair.completion_ok("Final: 3 days", "stop"))
        self.assertFalse(repair.completion_ok("", "stop"))
        self.assertFalse(repair.completion_ok("partial", "length"))

    def test_request_key_depends_on_budget(self):
        messages = [{"role": "user", "content": "q"}]
        self.assertNotEqual(repair.request_key(messages, 8192), repair.request_key(messages, 16384))


if __name__ == "__main__":
    unittest.main()
