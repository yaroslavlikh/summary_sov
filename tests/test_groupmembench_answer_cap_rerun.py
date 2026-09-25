import tempfile
from pathlib import Path
from types import SimpleNamespace

from research import groupmembench_answer_cap_rerun as rerun
from research.paper_benchmark_common import CachedAPI, SearchDoc


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Final: yes"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def fake_api(directory: Path) -> tuple[CachedAPI, FakeCompletions]:
    api = CachedAPI(directory, api_key="test")
    completions = FakeCompletions()
    api.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return api, completions


ROW = {"qa_key": "Finance:temporal:1", "question": "When?", "asking_user_id": "U7"}
DOCS = [SearchDoc("raw:a", "x", "[t][Speaker: Ann] started", ("a",), "raw"),
        SearchDoc("episode:F:1", "y", "[DERIVED TEMPORAL EPISODE]\nstory", ("a", "b"), "episode")]


def test_answer_messages_match_the_sealed_prompt_format():
    messages = rerun.answer_messages(ROW, DOCS, "SYSTEM")
    assert messages[0] == {"role": "system", "content": "SYSTEM"}
    assert messages[1]["content"] == (
        "Asking user: U7\n\nQuestion:\nWhen?\n\nRetrieved passages:\n"
        "[1] [t][Speaker: Ann] started\n\n[2] [DERIVED TEMPORAL EPISODE]\nstory\n\n"
        "Answer the question using the retrieved passages."
    )


def test_no_asking_user_line_when_absent():
    content = rerun.answer_messages({**ROW, "asking_user_id": ""}, DOCS, "S")[1]["content"]
    assert content.startswith("Question:\nWhen?")


def test_completion_budgets_sent_to_gpt5():
    with tempfile.TemporaryDirectory() as directory:
        api, completions = fake_api(Path(directory))
        messages = rerun.answer_messages(ROW, DOCS, "S")
        api.chat(model="gpt-5", messages=messages, temperature=rerun.TEMPERATURE,
                 max_tokens=rerun.ORIGINAL_ANSWER_MAX_TOKENS, phase="answer")
        api.chat(model="gpt-5", messages=messages, temperature=rerun.TEMPERATURE,
                 max_tokens=rerun.RERUN_ANSWER_MAX_TOKENS, phase="answer")
        assert [call["max_completion_tokens"] for call in completions.calls] == [2048, 8192]


def test_chat_key_matches_cached_api_key():
    with tempfile.TemporaryDirectory() as directory:
        api, _completions = fake_api(Path(directory))
        messages = rerun.answer_messages(ROW, DOCS, "S")
        api.chat(model="gpt-5", messages=messages, temperature=rerun.TEMPERATURE,
                 max_tokens=rerun.ORIGINAL_ANSWER_MAX_TOKENS, phase="answer")
        key = rerun.chat_key(label="openai", model="gpt-5", messages=messages,
                             temperature=rerun.TEMPERATURE, max_tokens=rerun.ORIGINAL_ANSWER_MAX_TOKENS)
        assert key in api.values
        assert key != rerun.chat_key(label="openai", model="gpt-5", messages=messages,
                                     temperature=rerun.TEMPERATURE, max_tokens=rerun.RERUN_ANSWER_MAX_TOKENS)


def test_empty_answer_detection():
    assert rerun.is_empty("") and rerun.is_empty("  \n") and not rerun.is_empty("Final: 3 days")
