"""Tests for research/temporal_episode_prototype.py -- pure validation,
retrieval, attach-decision parsing, and episode-mutation logic. No LLM
calls, no held-out or production data touched."""
import unittest
from unittest.mock import patch

import numpy as np

from research.temporal_episode_prototype import (
    EpisodeEvent, MemoryEvent, TemporalEpisode, _cache_key, _enforce_max_events_per_turn,
    _is_cross_speaker_attach, _is_generic_fragment, _parse_attach_decision, _quote_is_valid_substring,
    _safe_cosine_batch, materialize_episodes, retrieve_top_candidates, validate_memory_event,
)


def _event(**overrides):
    base = dict(
        event_id="net1:evt0", network_id="net1", event_text="Tigmen thinks Gordey is an asshole.",
        event_type="opinion", viewpoint_owner="Tigmen", subject="Gordey",
        evidence=({"turn_id": "t1", "quote": "asshole"},), source_turn_ids=("t1",),
        local_context_turn_ids=("t1",), temporal_mode="current", observed_at="2026-05-12T00:00:00",
    )
    base.update(overrides)
    return MemoryEvent(**base)


class ValidateMemoryEventTest(unittest.TestCase):
    def setUp(self):
        self.window = {"t1", "t2", "t3"}
        self.timestamps = {"t1": "2026-05-12T10:00:00", "t2": "2026-05-12T10:01:00", "t3": "2026-05-12T10:02:00"}
        self.messages = {
            "t1": "Tigmen says Gordey is a total asshole honestly.",
            "t2": "Priya: sounds good, see you then.",
            "t3": "Marcus commits to the camping trip in March.",
        }

    def test_valid_event_is_accepted(self):
        raw = {
            "event_text": "Tigmen thinks Gordey is an asshole.", "event_type": "opinion",
            "viewpoint_owner": "Tigmen", "subject": "Gordey",
            "evidence": [{"turn_id": "t1", "quote": "Gordey is a total asshole"}],
            "local_context_turn_ids": ["t1", "t2"], "temporal_mode": "current",
        }
        record, reason, attempted, valid = validate_memory_event(raw, self.window, self.timestamps, self.messages, "net1", 0)
        self.assertEqual(reason, "ok")
        self.assertEqual(record.observed_at, "2026-05-12T10:00:00")
        self.assertEqual(attempted, 1)
        self.assertEqual(valid, 1)

    def test_empty_event_text_is_rejected(self):
        raw = {"event_text": "", "event_type": "opinion", "evidence": [{"turn_id": "t1", "quote": "asshole"}]}
        record, reason, _a, _v = validate_memory_event(raw, self.window, self.timestamps, self.messages, "net1", 0)
        self.assertIsNone(record)
        self.assertEqual(reason, "empty_event_text")

    def test_invalid_event_type_is_rejected(self):
        raw = {"event_text": "x", "event_type": "gossip", "evidence": [{"turn_id": "t1", "quote": "asshole"}]}
        record, reason, _a, _v = validate_memory_event(raw, self.window, self.timestamps, self.messages, "net1", 0)
        self.assertIsNone(record)
        self.assertEqual(reason, "invalid_event_type")

    def test_evidence_with_fabricated_turn_id_is_dropped_not_trusted(self):
        raw = {
            "event_text": "Marcus commits to camping in March.", "event_type": "commitment",
            "evidence": [{"turn_id": "t3", "quote": "camping trip in March"}, {"turn_id": "t_fake", "quote": "x"}],
        }
        record, reason, attempted, valid = validate_memory_event(raw, self.window, self.timestamps, self.messages, "net1", 0)
        self.assertEqual(reason, "ok")
        self.assertEqual(record.source_turn_ids, ("t3",))
        self.assertEqual(attempted, 2)
        self.assertEqual(valid, 1)

    def test_all_evidence_invalid_rejects_the_whole_event(self):
        raw = {"event_text": "x", "event_type": "state", "evidence": [{"turn_id": "t_fake", "quote": "x"}]}
        record, reason, attempted, valid = validate_memory_event(raw, self.window, self.timestamps, self.messages, "net1", 0)
        self.assertIsNone(record)
        self.assertEqual(reason, "no_valid_evidence_quote")
        self.assertEqual(valid, 0)

    def test_quote_not_a_real_substring_of_the_turn_is_rejected(self):
        raw = {
            "event_text": "Marcus commits to camping in March.", "event_type": "commitment",
            "evidence": [{"turn_id": "t3", "quote": "totally invented quote that is not in the turn"}],
        }
        record, reason, attempted, valid = validate_memory_event(raw, self.window, self.timestamps, self.messages, "net1", 0)
        self.assertIsNone(record)
        self.assertEqual(reason, "no_valid_evidence_quote")
        self.assertEqual(attempted, 1)
        self.assertEqual(valid, 0)

    def test_quote_matches_after_whitespace_normalization_only(self):
        raw = {
            "event_text": "Marcus commits to camping in March.", "event_type": "commitment",
            "evidence": [{"turn_id": "t3", "quote": "camping   trip\nin March"}],
        }
        record, reason, _a, _v = validate_memory_event(raw, self.window, self.timestamps, self.messages, "net1", 0)
        self.assertEqual(reason, "ok")

    def test_bracket_wrapped_turn_id_is_normalized_not_rejected(self):
        # Real bug observed on an earlier smoke run: the model echoed a
        # turn_id wrapped in literal "[...]" citation-bracket characters
        # (mimicking this project's [[turn_id]] citation convention),
        # causing otherwise-valid events to be wrongly rejected.
        raw = {"event_text": "Marcus commits to camping in March.", "event_type": "commitment",
               "evidence": [{"turn_id": "[t3]", "quote": "camping trip in March"}]}
        record, reason, _a, _v = validate_memory_event(raw, self.window, self.timestamps, self.messages, "net1", 0)
        self.assertEqual(reason, "ok")
        self.assertEqual(record.source_turn_ids, ("t3",))

    def test_uniquely_dropped_namespace_is_restored_without_guessing(self):
        raw = {
            "event_text": "The compliance review gate started.", "event_type": "state",
            "evidence": [{"turn_id": "Msg_86", "quote": "officially kicked off"}],
        }
        record, reason, _a, _v = validate_memory_event(
            raw, {"Finance|Msg_86"}, {"Finance|Msg_86": "2026-09-12"},
            {"Finance|Msg_86": "We’ve officially kicked off the compliance review gate."},
            "Finance", 0,
        )
        self.assertEqual(reason, "ok")
        self.assertEqual(record.source_turn_ids, ("Finance|Msg_86",))

    def test_ambiguous_dropped_namespace_is_still_rejected(self):
        raw = {
            "event_text": "The compliance review gate started.", "event_type": "state",
            "evidence": [{"turn_id": "Msg_86", "quote": "officially kicked off"}],
        }
        window = {"Finance|Msg_86", "Technology|Msg_86"}
        record, reason, _a, _v = validate_memory_event(
            raw, window, {turn: "2026-09-12" for turn in window},
            {turn: "We’ve officially kicked off the compliance review gate." for turn in window},
            "mixed", 0,
        )
        self.assertIsNone(record)
        self.assertEqual(reason, "no_valid_evidence_quote")

    def test_local_context_turn_ids_also_filtered_against_window(self):
        raw = {
            "event_text": "Marcus commits to camping in March.", "event_type": "commitment",
            "evidence": [{"turn_id": "t3", "quote": "camping trip in March"}],
            "local_context_turn_ids": ["t1", "t2", "t_fake"],
        }
        record, reason, _a, _v = validate_memory_event(raw, self.window, self.timestamps, self.messages, "net1", 0)
        self.assertEqual(reason, "ok")
        self.assertEqual(set(record.local_context_turn_ids), {"t1", "t2"})

    def test_observed_at_is_server_derived_never_trusted_from_model(self):
        raw = {
            "event_text": "Marcus commits to camping in March.", "event_type": "commitment",
            "evidence": [{"turn_id": "t3", "quote": "camping trip in March"}],
            "observed_at": "2099-01-01T00:00:00",  # model-supplied, must be ignored (not even read)
        }
        record, reason, _a, _v = validate_memory_event(raw, self.window, self.timestamps, self.messages, "net1", 0)
        self.assertEqual(reason, "ok")
        self.assertEqual(record.observed_at, self.timestamps["t3"])

    def test_missing_resolvable_timestamp_is_rejected(self):
        raw = {"event_text": "Marcus commits to camping in March.", "event_type": "commitment",
               "evidence": [{"turn_id": "t1", "quote": "asshole"}]}
        record, reason, _a, _v = validate_memory_event(raw, {"t1"}, {}, self.messages, "net1", 0)  # no known timestamp for t1
        self.assertIsNone(record)
        self.assertEqual(reason, "no_resolvable_timestamp")

    def test_generic_fragment_event_text_is_rejected_even_with_valid_evidence(self):
        raw = {"event_text": "Sounds good", "event_type": "other",
               "evidence": [{"turn_id": "t2", "quote": "sounds good"}]}
        record, reason, _a, _v = validate_memory_event(raw, self.window, self.timestamps, self.messages, "net1", 0)
        self.assertIsNone(record)
        self.assertEqual(reason, "generic_fragment_or_agreement_marker")

    def test_concrete_content_is_not_rejected_as_a_fragment(self):
        raw = {"event_text": "Marcus commits to the camping trip in March.", "event_type": "commitment",
               "evidence": [{"turn_id": "t3", "quote": "camping trip in March"}]}
        record, reason, _a, _v = validate_memory_event(raw, self.window, self.timestamps, self.messages, "net1", 0)
        self.assertEqual(reason, "ok")


class IsGenericFragmentTest(unittest.TestCase):
    def test_bare_examples_from_the_spec_are_fragments(self):
        for text in ("X is right", "This matters", "I am thinking", "Leave it to me", "Okay", "Sounds good"):
            self.assertTrue(_is_generic_fragment(text), text)

    def test_concrete_sentence_containing_similar_words_is_not_a_fragment(self):
        self.assertFalse(_is_generic_fragment(
            "Marcus decides the camping trip is right for March, not April, after checking with Priya.",
        ))
        self.assertFalse(_is_generic_fragment("Okay, Marcus will bring the tent and confirms he is driving."))


class QuoteSubstringTest(unittest.TestCase):
    def test_exact_substring_matches(self):
        self.assertTrue(_quote_is_valid_substring("locked up", "knee completely went, like proper locked up after coaching"))

    def test_invented_quote_does_not_match(self):
        self.assertFalse(_quote_is_valid_substring("totally invented text", "knee completely went, locked up after coaching"))

    def test_empty_quote_never_matches(self):
        self.assertFalse(_quote_is_valid_substring("", "any message"))


class MaxEventsPerTurnTest(unittest.TestCase):
    def test_third_event_citing_the_same_turn_is_dropped(self):
        events = [_event(event_id=f"net1:evt{i}", source_turn_ids=("t1",)) for i in range(3)]
        kept, dropped = _enforce_max_events_per_turn(events)
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["reason"], "exceeds_max_events_per_turn")

    def test_earlier_events_always_win_ties(self):
        events = [_event(event_id=f"net1:evt{i}", event_text=f"v{i}", source_turn_ids=("t1",)) for i in range(3)]
        kept, _dropped = _enforce_max_events_per_turn(events)
        self.assertEqual([e.event_text for e in kept], ["v0", "v1"])

    def test_different_turns_are_not_capped_together(self):
        events = [_event(event_id="e0", source_turn_ids=("t1",)), _event(event_id="e1", source_turn_ids=("t2",)),
                  _event(event_id="e2", source_turn_ids=("t1",))]
        kept, dropped = _enforce_max_events_per_turn(events)
        self.assertEqual(len(kept), 3)
        self.assertEqual(len(dropped), 0)


class CacheKeyTest(unittest.TestCase):
    def test_different_prompt_hash_changes_key(self):
        self.assertNotEqual(_cache_key("net1", {"t1"}, "hash_a"), _cache_key("net1", {"t1"}, "hash_b"))

    def test_different_network_changes_key(self):
        self.assertNotEqual(_cache_key("net1", {"t1"}, "h"), _cache_key("net2", {"t1"}, "h"))


class RetrieveTopCandidatesTest(unittest.TestCase):
    def test_empty_episode_list_returns_empty(self):
        self.assertEqual(retrieve_top_candidates(_event(), []), [])

    def test_never_returns_more_than_limit(self):
        episodes = [
            TemporalEpisode(episode_id=f"ep{i}", network_id="net1", viewpoint_owner="Tigmen", subject="Gordey",
                             events=[EpisodeEvent(event=_event(event_id=f"e{i}"), decision="create")])
            for i in range(5)
        ]
        result = retrieve_top_candidates(_event(event_text="something about Gordey"), episodes, limit=3)
        self.assertLessEqual(len(result), 3)

    def test_cosine_uses_float64_and_rejects_invalid_rows(self):
        query = np.array([1e30, 1e30], dtype=np.float32)
        matrix = np.array([[1e30, 1e30], [0.0, 0.0], [np.nan, 1.0]], dtype=np.float32)
        with np.errstate(all="raise"):
            scores = _safe_cosine_batch(query, matrix)
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertEqual(scores[1], -np.inf)
        self.assertEqual(scores[2], -np.inf)


class ParseAttachDecisionTest(unittest.TestCase):
    def test_low_confidence_defaults_to_new_episode(self):
        parsed = _parse_attach_decision({"decision": "ATTACH_REVISE", "target_episode_index": 0, "confidence": 0.2}, 1)
        self.assertEqual(parsed["decision"], "NEW_EPISODE")
        self.assertTrue(parsed["defaulted"])
        self.assertIsNone(parsed["target_episode_index"])

    def test_unknown_decision_defaults_to_new_episode(self):
        parsed = _parse_attach_decision({"decision": "MERGE", "confidence": 0.9}, 1)
        self.assertEqual(parsed["decision"], "NEW_EPISODE")
        self.assertTrue(parsed["defaulted"])

    def test_attach_without_valid_target_defaults_to_new_episode(self):
        parsed = _parse_attach_decision({"decision": "ATTACH_AUGMENT", "target_episode_index": 5, "confidence": 0.9}, 2)
        self.assertEqual(parsed["decision"], "NEW_EPISODE")
        self.assertTrue(parsed["defaulted"])

    def test_valid_high_confidence_attach_is_accepted(self):
        parsed = _parse_attach_decision({"decision": "attach_reaffirm", "target_episode_index": 1, "confidence": 0.9}, 2)
        self.assertEqual(parsed["decision"], "ATTACH_REAFFIRM")
        self.assertEqual(parsed["target_episode_index"], 1)
        self.assertFalse(parsed["defaulted"])


class IsCrossSpeakerAttachTest(unittest.TestCase):
    def test_different_resolved_identities_is_cross_speaker(self):
        event = _event(viewpoint_owner="Priya", subject=None)
        episode = TemporalEpisode(episode_id="ep1", network_id="net1", viewpoint_owner="Marcus", subject=None)
        self.assertTrue(_is_cross_speaker_attach(event, episode))

    def test_same_identity_is_not_cross_speaker(self):
        event = _event(viewpoint_owner="Marcus", subject=None)
        episode = TemporalEpisode(episode_id="ep1", network_id="net1", viewpoint_owner="Marcus", subject=None)
        self.assertFalse(_is_cross_speaker_attach(event, episode))

    def test_unresolved_identity_is_not_flagged(self):
        event = _event(viewpoint_owner=None, subject=None)
        episode = TemporalEpisode(episode_id="ep1", network_id="net1", viewpoint_owner="Marcus", subject=None)
        self.assertFalse(_is_cross_speaker_attach(event, episode))


class MaterializeEpisodesTest(unittest.TestCase):
    def test_new_episode_created_when_no_candidates(self):
        events_by_network = {"net1": [_event()]}
        with patch("research.temporal_episode_prototype.run_attach_decision") as mock_attach:
            mock_attach.return_value = {"decision": "NEW_EPISODE", "target_episode_index": None,
                                         "confidence": 1.0, "rationale": "no candidates", "defaulted": False}
            episodes, log, calls = materialize_episodes(events_by_network)
        self.assertEqual(len(episodes["net1"]), 1)
        self.assertEqual(len(episodes["net1"][0].events), 1)
        self.assertEqual(episodes["net1"][0].events[0].decision, "create")
        self.assertFalse(log[0]["cross_speaker_attachment"])

    def test_attach_appends_without_losing_prior_provenance(self):
        old_event = _event(event_id="net1:evt0", event_text="old value", observed_at="2026-05-12T00:00:00", source_turn_ids=("t1",))
        new_event = _event(event_id="net1:evt1", event_text="new value", observed_at="2026-05-18T00:00:00", source_turn_ids=("t2",))
        events_by_network = {"net1": [old_event, new_event]}

        call_sequence = [
            {"decision": "NEW_EPISODE", "target_episode_index": None, "confidence": 1.0, "rationale": "", "defaulted": False},
            {"decision": "ATTACH_REVISE", "target_episode_index": 0, "confidence": 0.9, "rationale": "", "defaulted": False},
        ]
        with patch("research.temporal_episode_prototype.run_attach_decision", side_effect=call_sequence):
            episodes, log, calls = materialize_episodes(events_by_network)

        self.assertEqual(len(episodes["net1"]), 1)
        episode = episodes["net1"][0]
        self.assertEqual(len(episode.events), 2)  # both events retained, nothing dropped
        self.assertEqual(episode.events[0].event.event_text, "old value")
        self.assertEqual(episode.events[1].event.event_text, "new value")
        self.assertEqual(episode.events[1].decision, "attach_revise")
        # old provenance (t1) still present in the episode even though it's
        # no longer the "current" event
        self.assertIn("t1", episode.all_source_turn_ids())
        self.assertIn("t2", episode.all_source_turn_ids())
        # same viewpoint_owner ("Tigmen") on both events -- not cross-speaker
        self.assertFalse(log[1]["cross_speaker_attachment"])

    def test_cross_speaker_attach_is_flagged_but_still_allowed_to_happen(self):
        old_event = _event(event_id="net1:evt0", viewpoint_owner="Marcus", source_turn_ids=("t1",))
        new_event = _event(event_id="net1:evt1", viewpoint_owner="Priya", source_turn_ids=("t2",))
        events_by_network = {"net1": [old_event, new_event]}
        call_sequence = [
            {"decision": "NEW_EPISODE", "target_episode_index": None, "confidence": 1.0, "rationale": "", "defaulted": False},
            {"decision": "ATTACH_AUGMENT", "target_episode_index": 0, "confidence": 0.9, "rationale": "shared plan", "defaulted": False},
        ]
        with patch("research.temporal_episode_prototype.run_attach_decision", side_effect=call_sequence):
            episodes, log, calls = materialize_episodes(events_by_network)
        self.assertTrue(log[1]["cross_speaker_attachment"])

    def test_active_events_capped_but_full_history_retained(self):
        from research.temporal_episode_prototype import MAX_HOT_EVENTS

        episode = TemporalEpisode(episode_id="ep1", network_id="net1", viewpoint_owner="Tigmen", subject="Gordey")
        for i in range(MAX_HOT_EVENTS + 3):
            episode.events.append(EpisodeEvent(event=_event(event_id=f"e{i}", event_text=f"v{i}"), decision="create"))
        self.assertEqual(len(episode.events), MAX_HOT_EVENTS + 3)  # nothing deleted
        self.assertEqual(len(episode.active_events), MAX_HOT_EVENTS)  # only the cap is "hot"
        self.assertEqual(episode.active_events[-1].event.event_text, f"v{MAX_HOT_EVENTS + 2}")


if __name__ == "__main__":
    unittest.main()
