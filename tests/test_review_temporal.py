"""Temporal cutoffs, cited facts, privacy processing and request recovery.

These are local regression checks. Model/network results are replaced here;
the separate development question banks are used for real Gemini trials.
"""

import io
import json
import os
import tempfile
import urllib.error
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from tools import temporal_trial_adapters as temporal
from tools import trial_gemini as client


def observation(frame, track=7, **extra):
    return {"recording_id": "a", "frame_id": frame, "track_id": track,
            "class_name": "Bus", "bbox_xyxy": [4, 5, 20, 18], "source": "annotated", **extra}


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "scene.png"
        Image.new("RGB", (32, 24), (200, 150, 100)).save(self.image)
        self.question = {"question_id": "q", "recording_id": "a", "query_frame": 5, "question": "Which vehicle appears later?"}
        self.recording = SimpleNamespace(
            recording_id="a", iter_frames=lambda: iter([{"frame_id": 1}]),
            get_frame=lambda frame: {"frame_id": frame, "width": 32, "height": 24, "image_path": str(self.image)},
        )
        env = patch.dict(os.environ, {"TRACKRAG_TRIAL_CACHE": str(self.root / "cache")})
        env.start()
        self.addCleanup(env.stop)

    def test_future_tracks_boxes_and_gaps_cannot_change_candidate_context(self):
        old = [observation(2), observation(4)]
        index = {"recording": self.recording, "tracks": {7: old}}
        before = temporal.candidates_at_query(self.question, index)
        index["tracks"][7].extend([observation(6, bbox_xyxy=[90, 90, 99, 99]), observation(90)])
        index["tracks"][999] = [observation(50, 999, class_name="FutureSecret")]
        self.assertEqual(before, temporal.candidates_at_query(self.question, index))
        tracks = [c for c in before if c["track_id"] is not None]
        self.assertEqual(tracks[0]["history"]["annotation_gaps"], [[3, 3]])
        self.assertEqual(tracks[0]["history"]["first_visibility_or_entry"], "not established by annotation timing")
        self.assertEqual(tracks[0]["history"]["first_size_wh_pixels"], [16, 13])
        self.assertEqual([c["start_frame"] for c in before if c["track_id"] is None], [1, 5])

    def test_semantic_selection_accepts_scene_and_full_history_without_keyword_rules(self):
        index = {"recording": self.recording, "tracks": {7: [observation(2), observation(4)]}}
        candidates = temporal.candidates_at_query(self.question, index)
        target = next(c for c in candidates if c["track_id"] == 7 and c["start_frame"] == 2 and c["end_frame"] == 4)
        with patch.object(temporal, "request_json", return_value={"selected": ["scene-F5", target["candidate_id"]], "requested_frame_ids": []}):
            ranked = temporal.rank_evidence(self.question, index)
        self.assertEqual([(r["track_id"], r["start_frame"], r["end_frame"]) for r in ranked], [(None, 5, 5), (7, 2, 4)])

    def test_requested_comparison_frames_replace_a_coarse_candidate_window(self):
        index = {"recording": self.recording, "tracks": {7: [observation(2), observation(4), observation(5)]}}
        candidates = temporal.candidates_at_query(self.question, index)
        target = next(c for c in candidates if c["track_id"] == 7)
        with patch.object(temporal, "request_json", return_value={
            "selected": [target["candidate_id"]], "requested_frame_ids": [2, 4],
        }):
            ranked = temporal.rank_evidence(self.question, index)
        self.assertEqual((ranked[0]["start_frame"], ranked[0]["end_frame"]), (2, 4))

    def test_requested_frames_cannot_use_future_or_invent_a_missing_observation(self):
        index = {"recording": self.recording, "tracks": {7: [observation(2), observation(4)]}}
        target = next(c for c in temporal.candidates_at_query(self.question, index) if c["track_id"] == 7)
        with patch.object(temporal, "request_json", return_value={"selected": [target["candidate_id"]], "requested_frame_ids": [6]}):
            with self.assertRaisesRegex(ValueError, "future"):
                temporal.rank_evidence(self.question, index)
        with patch.object(temporal, "request_json", return_value={"selected": [target["candidate_id"]], "requested_frame_ids": [3, 4]}):
            ranked = temporal.rank_evidence(self.question, index)
        self.assertEqual((ranked[0]["start_frame"], ranked[0]["end_frame"]), (2, 4))

    def test_refined_duplicate_windows_are_removed_and_ranks_remain_consecutive(self):
        question = dict(self.question, query_frame=60)
        index = {"recording": self.recording, "tracks": {7: [observation(f) for f in range(1, 61)]}}
        tracks = [c for c in temporal.candidates_at_query(question, index) if c["track_id"] == 7]
        with patch.object(temporal, "request_json", return_value={
            "selected": [tracks[0]["candidate_id"], tracks[1]["candidate_id"], "scene-F60"],
            "requested_frame_ids": [20, 40],
        }):
            ranked = temporal.rank_evidence(question, index)
        self.assertEqual([(x["track_id"], x["start_frame"], x["end_frame"], x["rank"]) for x in ranked],
                         [(7, 20, 40, 1), (None, 60, 60, 2)])

    def test_identity_selection_is_stable_when_an_earlier_unrelated_track_is_inserted(self):
        target = [observation(1, 42), observation(5, 42)]
        for extra in ({}, {3: [observation(1, 3), observation(5, 3)]}):
            with self.subTest(extra_tracks=list(extra)):
                index = {"recording": self.recording, "tracks": {42: target, **extra}}
                with patch.object(temporal, "request_json", return_value={
                    "selected": ["T42-F1-F5", "scene-F5"], "requested_frame_ids": [1, 5],
                }):
                    ranked = temporal.rank_evidence(self.question, index)
                self.assertEqual([(r["track_id"], r["start_frame"], r["end_frame"]) for r in ranked],
                                 [(42, 1, 5), (None, 5, 5)])

    def test_past_motion_changes_selector_context_even_when_full_history_endpoints_agree(self):
        question = dict(self.question, query_frame=60)
        constant = [observation(f) for f in range(1, 61)]
        changed = [observation(f, bbox_xyxy=[4, 5, 28, 22] if 16 <= f <= 45 else [4, 5, 20, 18]) for f in range(1, 61)]
        contexts = []
        for rows in (constant, changed):
            candidates = temporal.candidates_at_query(question, {"recording": self.recording, "tracks": {7: rows}})
            contexts.append(temporal.selection_context(candidates))
        self.assertEqual(contexts[0]["tracks"][0]["history"], contexts[1]["tracks"][0]["history"])
        self.assertNotEqual(contexts[0], contexts[1])

    def test_selector_rejects_invented_duplicate_or_nonstring_candidates(self):
        index = {"recording": self.recording, "tracks": {}}
        for selected in (["C999"], ["C1", "C1"], [None], {"C1": True}):
            with self.subTest(selected=selected):
                with patch.object(temporal, "request_json", return_value={"selected": selected}):
                    with self.assertRaisesRegex(ValueError, "invalid or duplicate"):
                        temporal.rank_evidence(self.question, index)

    def evidence(self):
        return [{"evidence_id": "E1", "interval": {"track_id": 7, "start_frame": 1, "end_frame": 3},
                 "images": [{"frame_id": 3, "bbox_xyxy": [4, 5, 20, 18], "image_path": "scene.png"}]}]

    def write_run(self):
        rows = [observation(f) for f in (1, 3, 5)] + [observation(6, 999, class_name="FutureSecret")]
        rows[0]["bbox_xyxy"] = [8, 6, 24, 19]
        (self.root / "tracks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (self.root / "run.json").write_text(json.dumps({"first_frame_id": 1}), encoding="utf-8")

    def test_answer_facts_stay_inside_the_cited_interval_and_keep_qualifications(self):
        self.write_run()
        response = {"answer_status": "answered", "answer": "The later bus.",
                    "unverified_parts": ["Its exact first entry is not established."], "cited_evidence_ids": ["E1"]}
        with patch.object(temporal, "request_json", return_value=response) as request:
            actual = temporal.answer_question(self.question, self.evidence(), self.root)
        self.assertEqual(request.call_count, 2)
        prompt = request.call_args_list[0].args[0][0]["text"]
        facts = json.loads(prompt.split("Observed facts: ", 1)[1])
        self.assertEqual(facts[0]["history"]["last_observed_frame"], 3)
        self.assertEqual(facts[0]["history"]["observed_frames"], 2)
        self.assertEqual(facts[0]["history"]["net_image_motion"], {
            "horizontal": "left", "vertical": "up", "dominant_axis": "horizontal",
        })
        self.assertNotIn("FutureSecret", prompt)
        self.assertIn("exact first entry is not established", actual["answer"])
        self.assertEqual(actual["cited_evidence_ids"], ["E1"])

    def test_invalid_answer_citation_is_rejected_before_success(self):
        self.write_run()
        response = {"answer_status": "answered", "answer": "A bus.", "unverified_parts": [], "cited_evidence_ids": ["E9"]}
        with patch.object(temporal, "request_json", return_value=response):
            with self.assertRaisesRegex(ValueError, "citations"):
                temporal.answer_question(self.question, self.evidence(), self.root)

    def test_attention_panel_uses_only_privacy_masked_pixels(self):
        buffer = io.BytesIO()
        Image.new("RGB", (32, 24), "black").save(buffer, format="JPEG")
        with patch.object(temporal, "_hosted_image", return_value=buffer.getvalue()):
            payload = temporal._attention_image(self.image, [], [4, 5, 20, 18])
        with Image.open(io.BytesIO(payload)) as image:
            self.assertLess(max(image.getpixel((10, 24 + 35))), 10)
            self.assertLess(max(image.getpixel((10, 10))), 10)

    def test_selector_sends_only_initial_and_query_frames(self):
        index = {"recording": self.recording, "tracks": {7: [observation(1), observation(5), observation(6)]}}
        parts, sent = temporal.selector_views(self.question, index)
        self.assertEqual([item["frame_id"] for item in sent], [1, 5])
        self.assertEqual(sum("inlineData" in part for part in parts), 2)

    def test_identity_gallery_uses_largest_observation_without_future_pixels_or_tracks(self):
        requested = []
        original = self.recording.get_frame
        self.recording.get_frame = lambda frame: (requested.append(frame), original(frame))[1]
        index = {"recording": self.recording, "tracks": {
            11: [observation(1, 11, bbox_xyxy=[4, 5, 9, 10]), observation(3, 11),
                 observation(6, 11, bbox_xyxy=[1, 1, 31, 23])],
            99: [observation(6, 99, class_name="FutureSecret")],
        }}
        parts, sent = temporal.identity_views(self.question, index)
        self.assertEqual(requested, [3])
        self.assertEqual(sent[0]["identities"], [{"track_id": 11, "frame_id": 3}])
        self.assertNotIn("FutureSecret", json.dumps(parts))
        self.assertIn("not necessarily the query frame", parts[0]["text"])

    def test_identity_gallery_masks_all_observed_people_before_cropping(self):
        index = {"recording": self.recording, "tracks": {
            11: [observation(3, 11)], 12: [observation(3, 12, class_name="Person")],
        }}
        with patch.object(temporal, "_hosted_image", wraps=temporal._hosted_image) as processor:
            _, sent = temporal.identity_views(self.question, index)
        self.assertEqual(processor.call_count, 1)
        self.assertEqual({r["track_id"] for r in processor.call_args.args[1]}, {11, 12})
        self.assertIsNone(processor.call_args.kwargs["max_size"])
        self.assertEqual(len(sent[0]["identities"]), 2)

    def test_identity_gallery_does_not_truncate_tracks_beyond_scene_label_limit(self):
        index = {"recording": self.recording, "tracks": {t: [observation(2, t)] for t in range(41, 76)}}
        parts, sent = temporal.identity_views(self.question, index)
        identities = [r for sheet in sent for r in sheet["identities"]]
        self.assertEqual([r["track_id"] for r in identities], list(range(41, 76)))
        self.assertEqual(sum("inlineData" in part for part in parts), 3)


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        env = patch.dict(os.environ, {"GEMINI_API_KEY": "unused-test-key", "TRACKRAG_TRIAL_MODEL": "test-model",
                                     "TRACKRAG_TRIAL_CACHE": str(self.root / "cache")})
        env.start()
        self.addCleanup(env.stop)

    def wire(self, value):
        return io.BytesIO(json.dumps({"candidates": [{"content": {"parts": [{"text": json.dumps(value)}]}}],
                                      "usageMetadata": {"promptTokenCount": 12}}).encode())

    def call(self, **kwargs):
        return client.request_json([{"text": "Grounded question"}], {"type": "OBJECT"}, "test", **kwargs)

    @staticmethod
    def validate(value):
        if value.get("selected") != ["C1"]:
            raise ValueError("Invalid candidate")

    def test_cache_hit_keeps_current_question_metadata_and_avoids_network(self):
        with patch.object(client.urllib.request, "urlopen", return_value=self.wire({"selected": ["C1"]})) as network:
            self.call(receipt_path=self.root / "a.json", metadata={"question_id": "old"}, validate=self.validate)
            self.call(receipt_path=self.root / "b.json", metadata={"question_id": "new"}, validate=self.validate)
        self.assertEqual(network.call_count, 1)
        receipt = json.loads((self.root / "b.json").read_text())
        self.assertTrue(receipt["cached"])
        self.assertEqual(receipt["question_id"], "new")
        self.assertNotIn("unused-test-key", (self.root / "b.json").read_text())

    def test_sampling_uses_documented_provider_default_and_records_it(self):
        with patch.object(client.urllib.request, "urlopen", return_value=self.wire({"selected": ["C1"]})) as network:
            self.call(receipt_path=self.root / "receipt.json", validate=self.validate)
        sent = json.loads(network.call_args.args[0].data)
        self.assertNotIn("temperature", sent["generationConfig"])
        self.assertEqual(json.loads((self.root / "receipt.json").read_text())["sampling"], "provider_default")

    def test_invalid_model_output_is_never_cached_as_success(self):
        with patch.object(client.urllib.request, "urlopen", side_effect=lambda *a, **k: self.wire({"selected": ["C999"]})) as network:
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "Invalid candidate"):
                    self.call(receipt_path=self.root / "error.json", validate=self.validate)
        self.assertEqual(network.call_count, 2)
        self.assertEqual(list((self.root / "cache").glob("*.json")), [])
        self.assertEqual(json.loads((self.root / "error.json").read_text())["request_status"], "error")

    def test_invalid_cached_result_is_revalidated_and_replaced(self):
        with patch.object(client.urllib.request, "urlopen", side_effect=lambda *a, **k: self.wire({"selected": ["C1"]})) as network:
            self.call(validate=self.validate)
            cached = next((self.root / "cache").glob("*.json"))
            content = json.loads(cached.read_text())
            content["response"] = {"selected": ["C999"]}
            cached.write_text(json.dumps(content))
            self.call(receipt_path=self.root / "retry.json", validate=self.validate)
        self.assertEqual(network.call_count, 2)
        self.assertTrue(json.loads((self.root / "retry.json").read_text())["cache_rejected"])

    def test_network_error_remains_an_error_and_is_not_cached(self):
        with patch.object(client.urllib.request, "urlopen", side_effect=TimeoutError("test timeout")):
            with self.assertRaises(TimeoutError):
                self.call(receipt_path=self.root / "error.json")
        self.assertEqual(list((self.root / "cache").glob("*.json")), [])
        self.assertEqual(json.loads((self.root / "error.json").read_text())["error_type"], "TimeoutError")

    def test_failed_atomic_write_leaves_no_partial_receipt_or_temporary_file(self):
        with self.assertRaises(ValueError):
            client._save_json(self.root / "receipt.json", {"invalid": float("nan")})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_transient_provider_limit_retries_and_records_the_attempt(self):
        provider = {"error": {"message": "Temporary token limit", "details": [
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "2s"}]}}
        limited = urllib.error.HTTPError("https://example.invalid", 429, "rate limit", {}, io.BytesIO(json.dumps(provider).encode()))
        with patch.object(client.urllib.request, "urlopen", side_effect=[limited, self.wire({"selected": ["C1"]})]) as network:
            with patch.object(client.time, "sleep") as pause:
                self.call(receipt_path=self.root / "receipt.json", validate=self.validate)
        self.assertEqual(network.call_count, 2)
        pause.assert_called_once_with(2.0)
        receipt = json.loads((self.root / "receipt.json").read_text())
        self.assertEqual(receipt["request_status"], "completed")
        self.assertEqual(receipt["failed_attempts"][0]["http_status"], 429)
        self.assertNotIn("failure_class", receipt)
        self.assertNotIn("http_status", receipt)

    def test_daily_quota_is_explicit_never_retried_or_cached_as_an_answer(self):
        provider = {"error": {"message": "Requests per day limit exceeded", "details": []}}
        limited = urllib.error.HTTPError("https://example.invalid", 429, "quota", {}, io.BytesIO(json.dumps(provider).encode()))
        with patch.object(client.urllib.request, "urlopen", side_effect=limited) as network:
            with patch.object(client.time, "sleep") as pause:
                with self.assertRaisesRegex(client.ProviderQuotaError, "daily quota exhausted"):
                    self.call(receipt_path=self.root / "quota.json", repair_once=True)
        self.assertEqual(network.call_count, 1)
        pause.assert_not_called()
        receipt = json.loads((self.root / "quota.json").read_text())
        self.assertEqual(receipt["request_status"], "error")
        self.assertEqual(receipt["failure_class"], "daily_quota")
        self.assertEqual(receipt["http_status"], 429)
        self.assertEqual(list((self.root / "cache").glob("*.json")), [])

    def test_retry_after_header_and_exponential_delay_are_respected(self):
        first = urllib.error.HTTPError("https://example.invalid", 429, "temporary", {"Retry-After": "7"}, io.BytesIO(b'{}'))
        second = urllib.error.HTTPError("https://example.invalid", 503, "temporary", {}, io.BytesIO(b'{}'))
        with patch.object(client.urllib.request, "urlopen", side_effect=[first, second, self.wire({"selected": ["C1"]})]):
            with patch.object(client.time, "sleep") as pause:
                self.call(validate=self.validate)
        self.assertEqual([call.args[0] for call in pause.call_args_list], [7.0, 4])

    def test_format_repair_is_bounded_and_preserves_original_failure(self):
        with patch.object(client.urllib.request, "urlopen", side_effect=[
            self.wire({"selected": ["C999"]}), self.wire({"selected": ["C1"]}),
        ]) as network:
            actual = self.call(receipt_path=self.root / "receipt.json", validate=self.validate, repair_once=True)
        self.assertEqual(actual, {"selected": ["C1"]})
        self.assertEqual(network.call_count, 2)
        receipt = json.loads((self.root / "receipt.json").read_text())
        original = self.root / "cache/errors" / (receipt["repair_of_request_sha256"] + ".json")
        self.assertEqual(json.loads(original.read_text())["response"], {"selected": ["C999"]})
        self.assertEqual(receipt["request_status"], "completed")


if __name__ == "__main__":
    unittest.main()
