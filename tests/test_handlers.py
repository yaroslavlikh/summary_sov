import unittest
from unittest.mock import Mock

from handlers.handlers import build_message_link, format_summary_html, strip_citations
from llm.graphs import ASK_GRAPH, build_summary_graph
from webhook_server import create_app


class MessageLinkTests(unittest.TestCase):
    def test_private_supergroup_link(self):
        self.assertEqual(
            build_message_link(-1002335227490, 42),
            "https://t.me/c/2335227490/42",
        )

    def test_forum_topic_link(self):
        self.assertEqual(
            build_message_link(-1002335227490, 42, 2),
            "https://t.me/c/2335227490/2/42",
        )

    def test_regular_group_has_no_fake_link(self):
        self.assertIsNone(build_message_link(-12345, 42))


class CitationTests(unittest.TestCase):
    def test_only_known_citations_become_links_and_are_renumbered(self):
        rendered = format_summary_html(
            "- факт <важный> [9][4]\n- выдуманный источник [77]",
            {4: "https://example.test/4", 9: "https://example.test/9"},
        )
        self.assertEqual(
            rendered,
            '- факт &lt;важный&gt; <a href="https://example.test/9">[1]</a> '
            '<a href="https://example.test/4">[2]</a>\n- выдуманный источник',
        )

    def test_strip_citations(self):
        self.assertEqual(strip_citations("Факт [1][22]"), "Факт")


class WebhookTests(unittest.TestCase):
    def setUp(self):
        self.bot = Mock()
        self.client = create_app(self.bot, "secret").test_client()

    def test_rejects_missing_secret(self):
        response = self.client.post('/webhook', json={"update_id": 1})
        self.assertEqual(response.status_code, 403)
        self.bot.process_new_updates.assert_not_called()

    def test_accepts_telegram_secret(self):
        response = self.client.post(
            '/webhook',
            json={"update_id": 1},
            headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
        )
        self.assertEqual(response.status_code, 200)
        self.bot.process_new_updates.assert_called_once()


class GraphStructureTests(unittest.TestCase):
    def test_ask_graph_has_explicit_intent_and_parallel_retrieval_nodes(self):
        nodes = set(ASK_GRAPH.get_graph().nodes)
        self.assertTrue({"classify_intent", "rewrite_query", "search_fts", "search_vector", "fuse_rrf", "rerank"} <= nodes)

    def test_summary_graph_uses_tool_node(self):
        nodes = set(build_summary_graph(1).get_graph().nodes)
        self.assertTrue({"generate_summary", "send_summary", "extract_context", "tools"} <= nodes)


if __name__ == "__main__":
    unittest.main()
