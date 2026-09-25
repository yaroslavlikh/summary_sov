import unittest
from types import SimpleNamespace

from openai import OpenAI

from research.ksweep_accuracy import BudgetGuard, BudgetStop, allocation


def response(prompt_tokens, completion_tokens):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))


def cost(model, tokens_in, tokens_out):
    return tokens_in * 0.001 + tokens_out * 0.01


class BudgetGuardTest(unittest.TestCase):
    def test_forwards_kwargs_and_books_actual_usage(self):
        seen = []
        guard = BudgetGuard(lambda **kw: seen.append(kw) or response(100, 10), cost=cost,
                            estimate=lambda kw: 0.5, ceiling=1.0)
        guard(model="m", messages=[{"role": "user", "content": "x"}], temperature=0)
        self.assertEqual(seen, [{"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0}])
        self.assertAlmostEqual(guard.spent, 0.2)
        self.assertAlmostEqual(guard.reserved, 0.0)

    def test_stops_before_calling_when_spent_plus_estimate_exceeds_ceiling(self):
        calls = []
        guard = BudgetGuard(lambda **kw: calls.append(kw) or response(0, 0), cost=cost,
                            estimate=lambda kw: 0.15, ceiling=0.9, spent=0.8)
        with self.assertRaises(BudgetStop):
            guard(model="m", messages=[])
        self.assertEqual(calls, [])

    def test_in_flight_reservations_count_against_the_ceiling(self):
        guard = BudgetGuard(lambda **kw: response(0, 0), cost=cost, estimate=lambda kw: 0.3, ceiling=0.9)
        guard.reserved = 0.7
        with self.assertRaises(BudgetStop):
            guard(model="m", messages=[])

    def test_reservation_released_when_the_call_fails(self):
        def boom(**kw):
            raise ConnectionError
        guard = BudgetGuard(boom, cost=cost, estimate=lambda kw: 0.4, ceiling=0.9)
        with self.assertRaises(ConnectionError):
            guard(model="m", messages=[])
        self.assertAlmostEqual(guard.reserved, 0.0)
        self.assertAlmostEqual(guard.spent, 0.0)

    def test_guard_can_be_installed_on_a_real_client_without_network(self):
        client = OpenAI(api_key="test-not-used", base_url="http://127.0.0.1:9")
        guard = BudgetGuard(client.chat.completions.create, cost=cost, estimate=lambda kw: 0.0, ceiling=1.0)
        client.chat.completions.create = guard
        self.assertIs(client.chat.completions.create, guard)


class AllocationTest(unittest.TestCase):
    def test_proportional_largest_remainder_sums_to_n(self):
        sizes = {"Constraint": 402, "Multi-hop": 249, "Proactivity": 427, "Single-hop": 213, "Skill": 169,
                 "Style": 176, "Temporal Duration": 300, "Title": 196, "Update": 268}
        counts = allocation(304, sizes)
        self.assertEqual(sum(counts.values()), 304)
        self.assertEqual(counts["Proactivity"], 54)


if __name__ == "__main__":
    unittest.main()
