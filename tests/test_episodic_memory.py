"""Tests for episodic_memory.py and its /ask integration in llm/graphs.py.
unittest, project convention.

Pure logic (validation, attach parsing, rendering) runs without a database.
The worker and retrieval tests use the sandbox-schema-on-real-Postgres
pattern of tests/test_memory_facts.py with a scripted LLM: no model calls,
no real chat data touched.
"""
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import episodic_memory as em

CHAT_ID = -100900777
TEST_SCHEMA = "test_episodic_memory_module"
T0 = 1_780_000_000  # well in the past relative to `now` below, so everything is settled


def _anchor(message_id, text, date=T0, author="Марина"):
    return em.SourceMessage(row_id=message_id, message_id=message_id, author=author, text=text, message_date=date)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.anchors = {10: _anchor(10, "Переносим релиз на пятницу — модуль оплаты у Олега не готов", T0),
                        11: _anchor(11, "Готово, можно выкатывать", T0 + 60, "Олег")}

    def _raw(self, **overrides):
        raw = {"event_text": "Марина связала перенос релиза с неготовым модулем оплаты Олега",
               "event_type": "decision", "viewpoint_owner": "Марина", "subject": "Олег",
               "evidence": [{"message_id": "m10", "quote": "модуль оплаты у Олега не готов"}],
               "local_context_message_ids": ["m11", "m999"], "temporal_mode": "current"}
        raw.update(overrides)
        return raw

    def test_valid_event_derives_sources_and_time_from_messages(self):
        event, reason = em.validate_event(self._raw(), self.anchors, {10, 11})
        self.assertEqual(reason, "ok")
        self.assertEqual(event.source_message_ids, (10,))
        self.assertEqual(event.local_context_message_ids, (11,))  # m999 was never shown
        self.assertEqual(event.observed_at, T0)

    def test_paraphrased_quote_is_rejected(self):
        raw = self._raw(evidence=[{"message_id": "m10", "quote": "модуль Олега ещё не доделан"}])
        self.assertEqual(em.validate_event(raw, self.anchors, {10})[1], "no_valid_evidence_quote")

    def test_quote_from_a_non_anchor_message_is_rejected(self):
        raw = self._raw(evidence=[{"message_id": "m12", "quote": "Готово"}])
        self.assertEqual(em.validate_event(raw, self.anchors, {10, 11, 12})[1], "no_valid_evidence_quote")

    def test_bracketed_id_and_whitespace_differences_are_tolerated(self):
        raw = self._raw(evidence=[{"message_id": "[[m11]]", "quote": "Готово,  можно"}])
        event, reason = em.validate_event(raw, self.anchors, {10, 11})
        self.assertEqual(reason, "ok")
        self.assertEqual(event.source_message_ids, (11,))

    def test_filler_event_is_rejected_but_sentence_with_filler_is_kept(self):
        raw = self._raw(event_text="Ок", evidence=[{"message_id": "m11", "quote": "Готово"}])
        self.assertEqual(em.validate_event(raw, self.anchors, {11})[1], "generic_fragment")
        self.assertFalse(em.is_generic_fragment("Ок, Олег закончил модуль оплаты"))

    def test_dangling_event_text_is_rejected_unless_it_names_someone(self):
        for text in ("Я не пропустил ни одной пары", "Но скорее ближе к 8", "У меня одна пара"):
            raw = self._raw(event_text=text)
            self.assertEqual(em.validate_event(raw, self.anchors, {10})[1], "not_self_contained", text)
        for text in ("Это правда, что Олег закончил модуль", "Марина перенесла релиз"):
            self.assertEqual(em.validate_event(self._raw(event_text=text), self.anchors, {10})[1], "ok", text)

    def test_invalid_type_and_null_strings(self):
        self.assertEqual(em.validate_event(self._raw(event_type="fact"), self.anchors, {10})[1], "invalid_event_type")
        event, _ = em.validate_event(self._raw(subject="null", temporal_mode="soon"), self.anchors, {10})
        self.assertIsNone(event.subject)
        self.assertEqual(event.temporal_mode, "unknown")

    def test_cap_per_message_keeps_earlier_events(self):
        events = [em.validate_event(self._raw(event_text=f"Событие номер {i} про релиз"), self.anchors, {10})[0]
                  for i in range(3)]
        kept, dropped = em.enforce_max_events_per_message(events)
        self.assertEqual([e.event_text for e in kept], [events[0].event_text, events[1].event_text])
        self.assertEqual(dropped, 1)


class AttachParsingTests(unittest.TestCase):
    def test_valid_attach(self):
        parsed = em.parse_attach_decision({"decision": "ATTACH_REVISE", "target_episode_index": 1, "confidence": 0.8}, 2)
        self.assertEqual((parsed["decision"], parsed["target"], parsed["defaulted"]), ("ATTACH_REVISE", 1, False))

    def test_low_confidence_out_of_range_and_garbage_become_new(self):
        for output in ({"decision": "ATTACH_AUGMENT", "target_episode_index": 0, "confidence": 0.3},
                       {"decision": "ATTACH_AUGMENT", "target_episode_index": 5, "confidence": 0.9},
                       {"decision": "MERGE", "target_episode_index": 0, "confidence": 0.9}, {}):
            parsed = em.parse_attach_decision(output, 2)
            self.assertEqual((parsed["decision"], parsed["target"]), ("NEW_EPISODE", None), output)

    def test_confident_new_episode_is_not_marked_defaulted(self):
        parsed = em.parse_attach_decision({"decision": "NEW_EPISODE", "confidence": 0.2}, 2)
        self.assertFalse(parsed["defaulted"])


class RenderingTests(unittest.TestCase):
    def test_hot_view_and_sources(self):
        events = [{"event_text": f"e{i}", "viewpoint_owner": "A", "observed_at": T0 + i, "operation": "augment",
                   "source_message_ids": [i, i + 1]} for i in range(3)]
        episode = em.EpisodeView(1, "A", "B", events)
        self.assertEqual(em.hot_source_message_ids(episode), [0, 1, 2, 3])
        self.assertIn("(дополнение): e2", em.render_episode(episode))
        texts = [f"t{i}" for i in range(7)]
        self.assertEqual(em.link_text(texts), "t2 || t3 || t4 || t5 || t6")
        self.assertTrue(em.index_text("A", None, texts).startswith("Viewpoint owner: A. Subject: ?. t2"))

    def test_session_key(self):
        self.assertEqual(em.session_key(77, T0, 5), "conv:77")
        self.assertTrue(em.session_key(None, T0, 5).startswith("day:"))
        self.assertEqual(em.session_key(None, None, 85), "rows:2")

    def test_json_parsing_tolerates_reasoning_and_prose(self):
        self.assertEqual(em.parse_json_object('<think>{"x":1}</think> вот: {"items": []}'), {"items": []})
        self.assertEqual(em.parse_json_object("нет json"), {})


class ScriptedLLM:
    """Returns extraction items per anchor set and attach decisions in order."""

    def __init__(self, extractions, attaches=()):
        self.extractions = extractions
        self.attaches = list(attaches)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith("You will see messages"):
            for marker, items in self.extractions:
                if marker in prompt:
                    return {"items": items}
            return {"items": []}
        return self.attaches.pop(0) if self.attaches else {"decision": "NEW_EPISODE"}


class WorkerAndRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg2
        from psycopg2 import pool as pg_pool
        import database.db as db
        from config import get_database_url
        from database.init_db import init_db

        dsn = get_database_url()
        with psycopg2.connect(dsn) as conn:
            conn.autocommit = True
            with conn.cursor() as c:
                c.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE; CREATE SCHEMA {TEST_SCHEMA};")
        test_pool = pg_pool.ThreadedConnectionPool(1, 5, dsn=dsn, options=f"-c search_path={TEST_SCHEMA},public")
        db._get_pool = lambda: test_pool
        init_db()

    @classmethod
    def tearDownClass(cls):
        import psycopg2
        from config import get_database_url

        with psycopg2.connect(get_database_url()) as conn:
            conn.autocommit = True
            with conn.cursor() as c:
                c.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE;")

    def _insert(self, message_id, text, date, author="Марина", username=None, conversation_id=1, is_bot=False):
        import database.db as db
        from crypto_utils import encrypt
        with db.get_conn() as conn:
            conn.cursor().execute(
                "INSERT INTO messages (user_id, user_name, username, message, message_id, message_date, "
                "conversation_id, is_bot) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (CHAT_ID, author, username, encrypt(text), message_id, date, conversation_id, is_bot),
            )
            conn.commit()

    def test_lease_lets_only_one_worker_in_and_expires_on_its_own(self):
        import database.db as db
        self.assertTrue(em._take_lease(CHAT_ID, "host:1"))
        self.assertFalse(em._take_lease(CHAT_ID, "host:2"))   # someone else is working
        self.assertTrue(em._take_lease(CHAT_ID, "host:1"))    # the holder renews its own
        with db.get_conn() as conn:  # a worker killed mid-pass leaves a lease that simply expires
            conn.cursor().execute("UPDATE memory_worker_lease SET expires_at = now() - interval '1 minute'")
            conn.commit()
        self.assertTrue(em._take_lease(CHAT_ID, "host:2"))
        em._release_lease(CHAT_ID, "host:2")

    def test_worker_end_to_end(self):
        import database.db as db
        self._insert(1, "Переносим релиз на пятницу — модуль оплаты у Олега не готов", T0)
        self._insert(2, "Бот: не нашёл ответа", T0 + 5, author="Бот", is_bot=True)
        self._insert(3, "Готово, модуль оплаты можно выкатывать", T0 + 3600, author="Олег", conversation_id=3)
        self._insert(4, "свежак, ещё не осел", T0 + 10_000, conversation_id=4)
        now = T0 + 10_000 + 60  # message 4 is younger than SETTLE_SECONDS

        llm = ScriptedLLM(
            extractions=[
                ("[[m1]] [ANCHOR]", [
                    {"event_text": "Релиз перенесли на пятницу из-за неготового модуля оплаты Олега",
                     "event_type": "decision", "viewpoint_owner": "Марина", "subject": "модуль оплаты",
                     "evidence": [{"message_id": "m1", "quote": "Переносим релиз на пятницу"}], "temporal_mode": "current"},
                    {"event_text": "Выдуманное событие", "event_type": "state",
                     "evidence": [{"message_id": "m1", "quote": "этого нет в сообщении"}]},
                ]),
                ("[[m3]] [ANCHOR]", [
                    {"event_text": "Олег закончил модуль оплаты, релиз можно выкатывать",
                     "event_type": "state", "viewpoint_owner": "Олег", "subject": "модуль оплаты",
                     "evidence": [{"message_id": "m3", "quote": "модуль оплаты можно выкатывать"}], "temporal_mode": "current"},
                ]),
            ],
            attaches=[{"decision": "ATTACH_REVISE", "target_episode_index": 0, "confidence": 0.9, "rationale": "та же линия"}],
        )
        totals = em.run_for_chat(CHAT_ID, llm=llm, now=now, log=lambda _: None)
        self.assertEqual((totals["blocks"], totals["events"]), (2, 2))
        self.assertFalse(any("Бот: не нашёл" in p or "свежак" in p for p in llm.prompts))

        stats = em.memory_stats(CHAT_ID)
        self.assertEqual(stats["rejected"], {"no_valid_evidence_quote": 1})
        self.assertEqual(stats["operations"], {"create": 1, "revise": 1})
        self.assertEqual((stats["episodes"], stats["longest_episode"]), (1, 2))

        with db.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT last_row_id FROM memory_cursor WHERE chat_id = %s", (CHAT_ID,))
            cursor_row = cursor.fetchone()[0]
            cursor.execute("SELECT id FROM messages WHERE user_id = %s AND message_id = 3", (CHAT_ID,))
            self.assertEqual(cursor_row, cursor.fetchone()[0])  # stops before the unsettled message
            cursor.execute("SELECT event_text FROM memory_events WHERE chat_id = %s", (CHAT_ID,))
            self.assertTrue(all(not text.startswith("Релиз") for (text,) in cursor.fetchall()))  # encrypted at rest

        # A rerun over the same messages makes no calls and no duplicates.
        rerun = ScriptedLLM(extractions=[])
        self.assertEqual(em.run_for_chat(CHAT_ID, llm=rerun, now=now, log=lambda _: None)["blocks"], 0)
        self.assertEqual(rerun.prompts, [])

        # A session that grows after extraction does not re-extract rows already in a block.
        self._insert(5, "Олег, а тесты прогнал?", T0 + 3700, conversation_id=3)
        grown = ScriptedLLM(extractions=[])
        em.run_for_chat(CHAT_ID, llm=grown, now=now + em.SETTLE_SECONDS, log=lambda _: None)
        prompt = next(p for p in grown.prompts if "[[m5]] [ANCHOR]" in p)
        self.assertIn("[[m3]] 20", prompt)             # shown as context
        self.assertNotIn("[[m3]] [ANCHOR]", "".join(grown.prompts))  # never extracted again

        # Retrieval: the event is found in the MiniLM space, carries its episode as
        # chronology, and expands into exactly the message it cites.
        from embeddings import embed, to_vector_literal
        hits = em.search_events(CHAT_ID, to_vector_literal(embed("готов ли модуль оплаты к релизу")))
        self.assertTrue(hits)
        view = em.load_events([hits[0][0]])[0]
        self.assertEqual(view.source_message_ids, [3])
        self.assertEqual([e["operation"] for e in view.episode.hot_events], ["create", "revise"])
        self.assertIn("эта же линия раньше/позже", em.render_event(view))

        # Memory is retrieved as its own ranked list, only when switched on, and only
        # when it is close enough to the question to be worth displacing raw messages.
        import llm.graphs as graphs
        state = {"chat_id": CHAT_ID, "effective_question": view.event_text,
                 "asker_name": "Марина", "use_episodes": False}
        self.assertEqual(graphs._search_memory(state)["memory_ids"], [])
        state["use_episodes"] = True
        memory_ids = graphs._search_memory(state)["memory_ids"]
        self.assertIn(view.event_id, memory_ids)
        off_topic = graphs._search_memory({**state, "effective_question": "какая завтра погода в Осло"})
        self.assertEqual(off_topic["memory_ids"], [])
        fused = graphs._fuse_rrf({**state, "fts_ids": [], "vector_ids": [], "memory_ids": memory_ids})
        self.assertIn(view.event_id, fused["candidate_event_ids"])

        class _Reply:
            content = f"E{view.event_id}, 999999"

        with patch.object(graphs, "get_chat_model") as model:
            model.return_value.invoke.return_value = _Reply()
            reranked = graphs._rerank({**state, **fused}, config={})
        self.assertEqual((reranked["match_ids"], reranked["match_event_ids"]), (set(), [view.event_id]))
        anchor_ids = graphs._episode_source_row_ids(CHAT_ID, [view])
        self.assertTrue(anchor_ids)
        answered = graphs._generate_answer(
            {**state, "match_ids": set(), "match_event_ids": [view.event_id],
             "question": "что с модулем оплаты", "asker_name": "Марина"}, config={})
        # A memory hit anchors a conversation window like any raw match: the cited
        # message is in the context, not stripped down to itself.
        self.assertIn(3, [row[1] for row in answered["window_rows"]])
        self.assertEqual(answered["episode_row_ids"], anchor_ids)

        # Questions about a person query memory by resolved participant, not by cosine.
        key = em.participant_keys(CHAT_ID, em.MemoryEvent(
            "Олег закончил модуль", "state", "Олег", "Олег", "current",
            ({"message_id": 3, "quote": "Готово"},), (3,), (), T0 + 3600))
        self.assertTrue(key[0])
        self.assertTrue(em.search_events_by_participant(CHAT_ID, key[0]))


if __name__ == "__main__":
    unittest.main()
