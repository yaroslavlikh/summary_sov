"""Tests for memory_facts.py + participants.py (research/PROSPECTIVE_MEMORY_RETRIEVAL.md
Stage 1 hardening). unittest, not pytest -- project convention.

Unlike tests/test_handlers.py, these genuinely need Postgres: memory_facts
has no DB-free logic to test in isolation (provenance validation, subject
resolution, and superseding are all queries by design). Uses the same
sandbox-schema-on-real-Postgres pattern used throughout this project's
research/ scripts: an isolated schema per test class, dropped in tearDown,
never touching real chat data.
"""
import sys
import unittest

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

import psycopg2
from psycopg2 import pool as pg_pool

import database.db as db
from config import get_database_url

CHAT_ID = -100900555
TEST_SCHEMA = "test_memory_facts_module"


def _setup_sandbox():
    dsn = get_database_url()
    with psycopg2.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as c:
            c.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE;")
            c.execute(f"CREATE SCHEMA {TEST_SCHEMA};")
    test_pool = pg_pool.ThreadedConnectionPool(1, 5, dsn=dsn, options=f"-c search_path={TEST_SCHEMA},public")
    db._get_pool = lambda: test_pool
    from database.init_db import init_db
    init_db()


def _teardown_sandbox():
    dsn = get_database_url()
    with psycopg2.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as c:
            c.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE;")


def _insert_message(chat_id, message_id, user_name, username, text, message_date):
    from crypto_utils import encrypt
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages (user_id, user_name, username, message, message_id, message_date, search_vector) "
            "VALUES (%s, %s, %s, %s, %s, %s, to_tsvector('russian', %s))",
            (chat_id, user_name, username, encrypt(text), message_id, message_date, text),
        )
        conn.commit()


class MemoryFactsTestCase(unittest.TestCase):
    """One sandbox schema shared across all test methods in this class --
    real Postgres + real local embedding model, so setUp/tearDown are class-
    scoped, not per-test (matches research/ scripts' cost trade-off)."""

    @classmethod
    def setUpClass(cls):
        _setup_sandbox()
        import memory_facts as mf
        import participants as pt
        cls.mf = mf
        cls.pt = pt
        _insert_message(CHAT_ID, 68121, "Misha", "misha_tg", "Я переехал в Казань.", 1_700_000_000)
        _insert_message(CHAT_ID, 68125, "Misha", "misha_tg", "В Москве бываю редко.", 1_700_000_100)
        _insert_message(CHAT_ID, 68200, "Misha", "misha_tg", "Вернулся в Москву.", 1_700_100_000)
        _insert_message(CHAT_ID, 68300, "Yaroslav", "yaroslavlikh", "Го в бильярд.", 1_700_000_050)

    @classmethod
    def tearDownClass(cls):
        _teardown_sandbox()

    # ---- 1. empty provenance rejected ----
    def test_empty_provenance_rejected(self):
        with self.assertRaises(ValueError):
            self.mf.upsert_state(
                CHAT_ID, "misha_tg", "current_location", "location",
                "Миша переехал в Казань", ["где Миша"], source_message_ids=[],
            )

    def test_provenance_outside_chat_rejected(self):
        with self.assertRaises(ValueError):
            self.mf.upsert_state(
                CHAT_ID, "misha_tg", "current_location", "location",
                "Миша переехал в Казань", ["где Миша"], source_message_ids=[9999999],
            )

    # ---- 2. ambiguous / unknown subject rejected ----
    def test_unknown_subject_rejected(self):
        with self.assertRaises(ValueError):
            self.mf.upsert_state(
                CHAT_ID, "СовершенноНеизвестныйЧеловек", "current_location", "location",
                "кто-то переехал", ["где"], source_message_ids=[68121],
            )

    # ---- 3. subject alias resolves consistently ----
    def test_subject_alias_resolves_to_same_key(self):
        by_username = self.pt.resolve_participant_key(CHAT_ID, "misha_tg")
        by_raw_name = self.pt.resolve_participant_key(CHAT_ID, "Misha")
        self.assertIsNotNone(by_username)
        self.assertIsNotNone(by_raw_name)
        self.assertEqual(by_username[0], by_raw_name[0])

    # ---- 4. state_key restricted to fixed enum ----
    def test_invalid_state_key_downgraded_to_none(self):
        fact_id = self.mf.upsert_state(
            CHAT_ID, "misha_tg", "city", "location",  # "city" is not an allowed state_key
            "Миша где-то живёт", ["где Миша"], source_message_ids=[68121],
        )
        facts = self.mf.list_facts(CHAT_ID, active_only=False)
        fact = next(f for f in facts if f["id"] == fact_id)
        self.assertIsNone(fact["state_key"])

    # ---- 5. new state supersedes old (same subject, same state_key) ----
    def test_new_state_supersedes_old(self):
        first_id = self.mf.upsert_state(
            CHAT_ID, "misha_tg", "current_location", "location",
            "Миша переехал в Казань", ["где Миша"], source_message_ids=[68121],
        )
        second_id = self.mf.upsert_state(
            CHAT_ID, "misha_tg", "current_location", "location",
            "Миша вернулся в Москву", ["где Миша"], source_message_ids=[68200],
        )
        active = self.mf.get_active_facts(CHAT_ID, [self.pt.resolve_participant_key(CHAT_ID, "misha_tg")[0]])
        active_ids = [f["id"] for f in active]
        self.assertIn(second_id, active_ids)
        self.assertNotIn(first_id, active_ids)

        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT active, superseded_by FROM memory_facts WHERE id = %s", (first_id,))
            active_flag, superseded_by = cur.fetchone()
        self.assertFalse(active_flag)
        self.assertEqual(superseded_by, second_id)

    # ---- 6. state_key=None never supersedes, accumulates ----
    def test_null_state_key_accumulates(self):
        id_a = self.mf.upsert_state(
            CHAT_ID, "yaroslavlikh", None, "other",
            "Ярослав любит бильярд", ["что любит Ярослав"], source_message_ids=[68300],
        )
        id_b = self.mf.upsert_state(
            CHAT_ID, "yaroslavlikh", None, "other",
            "Ярослав организует встречи", ["кто организует встречи"], source_message_ids=[68300],
        )
        active = self.mf.get_active_facts(CHAT_ID, [self.pt.resolve_participant_key(CHAT_ID, "yaroslavlikh")[0]], limit=50)
        active_ids = {f["id"] for f in active}
        self.assertIn(id_a, active_ids)
        self.assertIn(id_b, active_ids)

    # ---- 7. different people never supersede each other ----
    def test_different_subjects_do_not_supersede(self):
        misha_key = self.pt.resolve_participant_key(CHAT_ID, "misha_tg")[0]
        yaroslav_key = self.pt.resolve_participant_key(CHAT_ID, "yaroslavlikh")[0]
        self.mf.upsert_state(
            CHAT_ID, "misha_tg", "availability", "availability",
            "Миша занят", ["доступен ли Миша"], source_message_ids=[68121],
        )
        self.mf.upsert_state(
            CHAT_ID, "yaroslavlikh", "availability", "availability",
            "Ярослав свободен", ["доступен ли Ярослав"], source_message_ids=[68300],
        )
        misha_active = self.mf.get_active_facts(CHAT_ID, [misha_key])
        yaroslav_active = self.mf.get_active_facts(CHAT_ID, [yaroslav_key])
        self.assertTrue(any("занят" in f["claim"] for f in misha_active))
        self.assertTrue(any("свободен" in f["claim"] for f in yaroslav_active))

    # ---- 8. observed_at computed server-side from source messages ----
    def test_observed_at_from_source_messages(self):
        fact_id = self.mf.upsert_state(
            CHAT_ID, "misha_tg", "plan", "plan",
            "Миша идёт на бильярд", ["план Миши"], source_message_ids=[68121, 68125],
        )
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT observed_at FROM memory_facts WHERE id = %s", (fact_id,))
            observed_at = cur.fetchone()[0]
        # max(message_date) of 68121 (1_700_000_000) and 68125 (1_700_000_100) is the later one
        self.assertEqual(int(observed_at.timestamp()), 1_700_000_100)

    # ---- 9. expired fact not returned ----
    def test_expired_fact_not_returned(self):
        import datetime
        fact_id = self.mf.upsert_state(
            CHAT_ID, "misha_tg", "preference", "preference",
            "Миша временно любит бильярд", ["предпочтения Миши"], source_message_ids=[68121],
            expires_at=datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc),
        )
        misha_key = self.pt.resolve_participant_key(CHAT_ID, "misha_tg")[0]
        active = self.mf.get_active_facts(CHAT_ID, [misha_key], limit=50)
        self.assertNotIn(fact_id, [f["id"] for f in active])

    # ---- 10. source outside current summary batch rejected (graphs.py layer) ----
    def test_source_outside_batch_rejected(self):
        from llm.graphs import _filter_to_batch
        batch = {68121, 68125}
        self.assertEqual(_filter_to_batch([68121, 99999], batch), [68121])
        self.assertEqual(_filter_to_batch([99999], batch), [])
        self.assertEqual(_filter_to_batch([68121, 68125], batch), [68121, 68125])


if __name__ == "__main__":
    unittest.main()
