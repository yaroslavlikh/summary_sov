import json
import tempfile
import unittest
from pathlib import Path

from research import evermembench_unlinked_events_control as ablation
from research.paper_benchmark_common import SearchDoc


def event(event_id: str, text: str, sources: list[str], owner="Ann", subject="Task") -> dict:
    return {"event_id": event_id, "network_id": "01", "event_text": text, "viewpoint_owner": owner,
            "subject": subject, "source_turn_ids": sources}


RAW = {
    "m1": SearchDoc("raw:m1", "start", "[t1][Group: G][Speaker: Ann]start", ("m1",), "raw"),
    "m2": SearchDoc("raw:m2", "done", "[t2][Group: G][Speaker: Ann]done", ("m2",), "raw"),
}


class EventDocumentTest(unittest.TestCase):
    def test_event_document_mirrors_episode_format_without_chronology(self):
        doc = ablation.event_doc(event("01:evt3", "Ann started the task", ["m1", "m1", "missing"]), RAW)
        self.assertEqual(doc.doc_id, "event:01:01:evt3")
        self.assertEqual(doc.index_text, "Viewpoint owner: Ann. Subject: Task. Ann started the task")
        self.assertEqual(doc.rendered.splitlines(), [
            "[DERIVED EVENT / owner=Ann / subject=Task]", "Ann started the task",
            "  [SOURCE m1] [t1][Group: G][Speaker: Ann]start",
        ])
        self.assertEqual(doc.source_ids, ("m1", "missing"))
        self.assertEqual(doc.kind, "event")

    def test_missing_owner_and_subject_use_question_mark(self):
        doc = ablation.event_doc(event("01:evt1", "x", ["m2"], owner=None, subject=None), RAW)
        self.assertTrue(doc.index_text.startswith("Viewpoint owner: ?. Subject: ?."))

    def test_every_event_is_used_not_only_the_hot_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            rows = [{"network_id": "01", "events": [
                {"decision": "create", "event": event(f"01:evt{i}", f"step {i}", ["m1"])} for i in range(7)
            ]}]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            docs = ablation.events_by_topic(path, RAW)
            self.assertEqual(len(docs["01"]), 7)
            self.assertEqual([d.index_text.split(". ")[-1] for d in docs["01"]], [f"step {i}" for i in range(7)])

    def test_duplicate_event_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            item = {"decision": "create", "event": event("01:evt1", "same", ["m1"])}
            path.write_text(json.dumps({"network_id": "01", "events": [item, item]}) + "\n")
            with self.assertRaisesRegex(RuntimeError, "duplicate event id"):
                ablation.events_by_topic(path, RAW)


if __name__ == "__main__":
    unittest.main()
