"""Deterministic tests for research/conversation_disentanglement.py.

unittest, not pytest (project convention -- see tests/test_handlers.py).
No Postgres, no network calls for tests 1-7 (synthetic embeddings). Test 8
(exact "Наполовину" incident reproduction) is the one exception: it loads
the real local embedding model on hardcoded real message text, because the
whole point of that scenario is honesty about real embedding behavior on
the actual incident, not a synthetic stand-in -- it still touches no
database and no network (weights are loaded locally).
"""
import random
import sys
import unittest

sys.path.insert(0, "/Users/yaroslavlikh/summary_sov")

from research.conversation_disentanglement import (
    ConversationTracker,
    Message,
    is_substantive,
    search_similar_episodes,
    semantic_pilot,
    time_gap_baseline,
)

CHAT_ID = -1

_rng = random.Random(1234)


def _topic_vector(topic: str, dim: int = 6) -> list[float]:
    """Fixed base direction per topic name + small deterministic per-call
    noise, so messages of the same topic are close but not identical, and
    different topics are far apart (near-orthogonal one-hot bases)."""
    bases = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    base = [0.0] * dim
    base[bases[topic] % dim] = 1.0
    noisy = [b + _rng.uniform(-0.05, 0.05) for b in base]
    return noisy


def _msg(mid, author, text, ts, topic="A", reply=None, thread=None):
    return Message(
        message_id=mid, chat_id=CHAT_ID, thread_id=thread, reply_to_message_id=reply,
        author=author, timestamp=ts, text=text, embedding=_topic_vector(topic),
    )


class TestIsSubstantive(unittest.TestCase):
    def test_short_replies_are_not_substantive(self):
        for text in ["да", "нет", "ага", "😈😈😈", ".", "", "+1"]:
            self.assertFalse(is_substantive(text), f"{text!r} should not be substantive")

    def test_real_sentences_are_substantive(self):
        self.assertTrue(is_substantive("кто такой Игорь на самом деле"))


class TestScenario1_InterleavedTopics(unittest.TestCase):
    """Two topically distinct conversations interleaved within one minute,
    no reply chains -- must end up as separate conversation_ids."""

    def test_interleaved_topics_are_separated(self):
        t0 = 1_000_000
        msgs = [
            _msg(1, "A1", "обсуждаем архитектуру ретривала для бота", t0 + 0, topic="A"),
            _msg(2, "B1", "го сегодня в бильярд вечером", t0 + 5, topic="B"),
            _msg(3, "A1", "лучше гибридный поиск с реранком", t0 + 12, topic="A"),
            _msg(4, "B1", "во сколько встречаемся на бильярде", t0 + 18, topic="B"),
            _msg(5, "A1", "реранк на LLM а не только эмбеддинги", t0 + 25, topic="A"),
            _msg(6, "B1", "го в семь у входа на бильярд", t0 + 31, topic="B"),
        ]
        predicted, results, _ = semantic_pilot(msgs)
        self.assertEqual(predicted[1], predicted[3])
        self.assertEqual(predicted[3], predicted[5])
        self.assertEqual(predicted[2], predicted[4])
        self.assertEqual(predicted[4], predicted[6])
        self.assertNotEqual(predicted[1], predicted[2])

        # baseline (current production heuristic) is expected to FAIL this --
        # everything is within the 300s gap, so it all collapses into one.
        baseline = time_gap_baseline(msgs)
        self.assertEqual(len(set(baseline.values())), 1, "baseline should NOT separate these (that's the bug)")


class TestScenario2_WeakReplyInheritsTarget(unittest.TestCase):
    def test_semantically_weak_reply_still_inherits_conversation(self):
        t0 = 2_000_000
        msgs = [
            _msg(10, "A1", "какой у нас план миграции на LangGraph", t0, topic="A"),
            _msg(11, "A2", "начнём с /ask, потом /summary", t0 + 4, topic="A"),
        ]
        weak_reply = _msg(12, "A1", "ага", t0 + 8, topic="A", reply=11)
        msgs.append(weak_reply)
        predicted, results, _ = semantic_pilot(msgs)
        self.assertEqual(predicted[12], predicted[11])
        by_id = {r.message_id: r for r in results}
        self.assertEqual(by_id[12].method, "reply")


class TestScenario3And4_OldConversationNewIdButRetrievable(unittest.TestCase):
    def test_similar_old_conversation_gets_a_new_id(self):
        t0 = 3_000_000
        old = [
            _msg(20, "A1", "ссылки на сообщения совсем не открываются", t0, topic="C"),
            _msg(21, "A1", "надо чинить генерацию t.me линков", t0 + 10, topic="C"),
        ]
        much_later = t0 + 3 * 24 * 3600  # 3 days later, way past the active window
        new_but_similar = [
            _msg(22, "A2", "О, ссылки на сообщения снова работают отлично", much_later, topic="C"),
        ]
        predicted, results, tracker = semantic_pilot(old + new_but_similar)
        self.assertNotEqual(predicted[20], predicted[22], "a 3-day-old similar topic must NOT silently continue")

    def test_old_similar_conversation_is_found_via_retrieval(self):
        t0 = 3_100_000
        old = [
            _msg(30, "A1", "ссылки на сообщения совсем не открываются", t0, topic="C"),
            _msg(31, "A1", "надо чинить генерацию t.me линков", t0 + 10, topic="C"),
        ]
        much_later = t0 + 3 * 24 * 3600
        unrelated_recent = [
            _msg(32, "B1", "го в бильярд на выходных", much_later, topic="B"),
        ]
        predicted, results, tracker = semantic_pilot(old + unrelated_recent)
        old_conv_id = predicted[30]

        query_embedding = _topic_vector("C")
        hits = search_similar_episodes(query_embedding, tracker.all_episodes(), top_k=2)
        hit_ids = [h["conversation_id"] for h in hits]
        self.assertIn(old_conv_id, hit_ids, "old topically-similar episode must be found via retrieval")
        # retrieval must not have touched conversation_id / episode membership
        self.assertEqual(predicted[30], predicted[31])


class TestScenario5_ThreadsNeverMerge(unittest.TestCase):
    def test_different_message_thread_id_never_merge(self):
        t0 = 4_000_000
        msgs = [
            _msg(40, "A1", "обсуждаем архитектуру ретривала для бота", t0, topic="A", thread=111),
            _msg(41, "A1", "продолжаем ту же тему тут", t0 + 5, topic="A", thread=222),
        ]
        predicted, results, _ = semantic_pilot(msgs)
        self.assertNotEqual(predicted[40], predicted[41])

    def test_reply_across_threads_is_not_trusted(self):
        # A cross-thread "reply" is a data anomaly, not a real signal -- must
        # not force a merge across the thread boundary.
        t0 = 4_100_000
        msgs = [
            _msg(42, "A1", "тема номер один в топике 111", t0, topic="A", thread=111),
            _msg(43, "A1", "якобы ответ, но топик другой", t0 + 5, topic="A", thread=222, reply=42),
        ]
        predicted, results, _ = semantic_pilot(msgs)
        self.assertNotEqual(predicted[42], predicted[43])
        by_id = {r.message_id: r for r in results}
        self.assertNotEqual(by_id[43].method, "reply")


class TestScenario6_ShortMessageDoesNotContaminate(unittest.TestCase):
    def test_short_irrelevant_message_does_not_pollute_representation(self):
        t0 = 5_000_000
        msgs = [
            _msg(50, "A1", "какой у нас план миграции на LangGraph", t0, topic="A"),
            _msg(51, "A2", "начнём с /ask, потом /summary", t0 + 5, topic="A"),
        ]
        predicted, results, tracker = semantic_pilot(msgs)
        conv_id = predicted[51]
        rep_before = tracker._episodes[conv_id].representation()

        # a short, topically unrelated aside from a THIRD person, no reply,
        # arriving right after (within the active window) -- proximity alone
        # must not let it dilute the conversation's semantic signature.
        noise = _msg(52, "C1", "лол", t0 + 8, topic="D")
        tracker.assign(noise)
        rep_after = tracker._episodes[conv_id].representation()
        self.assertEqual(rep_before, rep_after, "short message must not change the substantive representation")


class TestScenario7_BoundedRetrievalContext(unittest.TestCase):
    def test_retrieval_respects_all_size_bounds(self):
        t0 = 6_000_000
        msgs = []
        mid = 60
        for i in range(20):
            msgs.append(_msg(mid, "A1", f"сообщение номер {i} про архитектуру ретривала", t0 + i * 2, topic="A"))
            mid += 1
        for i in range(20):
            msgs.append(_msg(mid, "B1", f"сообщение номер {i} про бильярд на выходных", t0 + 1000 + i * 2, topic="B"))
            mid += 1

        predicted, results, tracker = semantic_pilot(msgs)
        hits = search_similar_episodes(
            _topic_vector("A"), tracker.all_episodes(),
            top_k=2, messages_per_episode=3, max_total_messages=5,
        )
        total = sum(len(h["message_ids"]) for h in hits)
        self.assertLessEqual(total, 5)
        for h in hits:
            self.assertLessEqual(len(h["message_ids"]), 3)
        self.assertLessEqual(len(hits), 2)


class TestScenario8_NapolovinuReproduction(unittest.TestCase):
    """Exact reproduction of the real production incident (docs/eval_incidents.md
    #1): real text, real reply chain, real ~seconds-apart timestamps. Uses the
    REAL local embedding model (no network, no DB) because faking embeddings
    for this specific scenario would defeat the point of testing it."""

    @classmethod
    def setUpClass(cls):
        from embeddings import embed
        cls.embed = staticmethod(embed)

    def test_dead_links_message_excluded_from_tigmen_hyperbole_episode(self):
        # Real data, decrypted from production messages 72736-72750 (see
        # docs/eval_incidents.md #1 and the conversation confirming exact
        # conversation_id=72717 collapse under the current heuristic).
        base_ts = 1788782939
        rows = [
            (72736, "Игорь", None, "@sov_summary_bot кто такой Игорь", 0),
            (72737, "tigmen", None, "мб это ссылка на запись в бд твоей он попытался", 0),
            (72738, "Бот", None, "Игорь — студент-хаотик, эмоционально-нестабильный, часто переключается от иронии к агрессивным провокациям", 3),
            (72739, "tigmen", None, "@sov_summary_bot кто я", 16),
            (72740, "Бот", None, "Ты — Саша Тигмен, IT-специалист-геймер, известный своей грубостью, частым матом и провокационным стилем", 18),
            (72741, "presccode80", None, "@sov_summary_bot кто я", 27),
            (72742, "Бот", None, "Ты — высокоэнергичный, агрессивный, часто использующий мат и драматизацию, берёшь лидерскую позицию в обсуждениях игр, спорта и сплетен", 30),
            (72743, "Ярослав", None, "Ссылки dead", 58),
            (72744, "Ярослав", None, "Наполовину", 72),
            (72745, "tigmen", 72740, "это неверно, на самом деле я Аурафарм император, убийца пабликов в доте, хорош во всем к чему не прикасается, тру адам, гуль sss ранг, канеки кен, эрен йегер, наруто, саске, итачи, бог, читер, папочка, Киллер сталкер асасин, Паркурист", 81),
            (72746, "Ярослав", 72745, "@sov_summary_bot это правда?", 90),
            (72747, "Бот", None, "Это правда лишь наполовину", 91),
        ]
        msgs = [
            _msg_real(mid, author, text, base_ts + dt, reply)
            for mid, author, reply, text, dt in rows
        ]
        for m in msgs:
            m.embedding = self.embed(m.text)

        predicted, results, tracker = semantic_pilot(msgs)

        # the real production bug: current heuristic collapses ALL of these
        # into one conversation_id (verified against live data == 72717).
        baseline = time_gap_baseline(msgs)
        self.assertEqual(baseline[72744], baseline[72745], "baseline is expected to reproduce the real contamination")

        # the pilot's actual claim: "Наполовину" (72744, no reply, off-topic)
        # must land in a DIFFERENT episode than Тигмен's hyperbolic
        # self-description (72745, which explicitly replies to 72740 and
        # anchors the identity thread).
        self.assertNotEqual(
            predicted[72744], predicted[72745],
            "Наполовину must be excluded from Тигмен's hyperbole episode",
        )
        # and 72745/72746/72747 (its own reply chain) must stay together.
        self.assertEqual(predicted[72745], predicted[72746])


def _msg_real(mid, author, text, ts, reply):
    return Message(
        message_id=mid, chat_id=CHAT_ID, thread_id=None, reply_to_message_id=reply,
        author=author, timestamp=ts, text=text, embedding=[0.0],  # placeholder, replaced before use
    )


if __name__ == "__main__":
    unittest.main()
