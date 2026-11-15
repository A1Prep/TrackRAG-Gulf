"""Local checks for real-data trial adapters. Network operations are replaced."""

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tools.emt_trial_adapters import (
    _hosted_image, answer_question, build_index, load_annotated_tracks, open_recording, rank_evidence,
)
from tools.prepare_emt_clip import _BoundedStream, prepare_clip
from tools.review_report import write_report
from trackrag.__main__ import check_adapters


class TrialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "video_000001"
        self.directory.mkdir()
        self.image = self.directory / "one.png"
        Image.new("RGB", (32, 24), (200, 150, 100)).save(self.image)
        self.frames = [{
            "recording_id": "video_000001", "frame_id": frame, "timestamp_s": (frame-1)/10,
            "width": 32, "height": 24, "image_path": "one.png",
        } for frame in range(1, 7)]
        (self.directory / "manifest.json").write_text(json.dumps({
            "recording_id": "video_000001", "frames": self.frames,
        }))
        self.recording = open_recording(self.root, "video_000001")
        self.tracks = [{
            "recording_id": "video_000001", "frame_id": frame, "source": "annotated",
            "track_id": 7, "class_name": "Bus", "bbox_xyxy": [4, 4, 20, 20],
        } for frame in range(1, 7)]
        self.question = {"question_id": "q1", "recording_id": "video_000001", "query_frame": 3, "question": "Which bus on the left?"}

    def test_prepared_recording_resolves_relative_paths_and_real_frame_ids(self):
        self.assertEqual(self.recording.get_frame(3)["image_path"], str(self.image.absolute()))
        self.assertEqual([row["frame_id"] for row in self.recording.iter_frames()], list(range(1, 7)))
        with self.assertRaises(KeyError):
            self.recording.get_frame(99)

    def test_future_changes_cannot_alter_the_trial_rankings_or_scores(self):
        original = rank_evidence(self.question, build_index(self.recording, self.tracks))
        changed = [dict(row) for row in self.tracks]
        for row in changed:
            if row["frame_id"] > 3:
                row.update(class_name="Car", bbox_xyxy=[1, 1, 2, 2])
        changed.append(dict(self.tracks[-1], track_id=99, class_name="Bus"))
        self.assertEqual(original, rank_evidence(self.question, build_index(self.recording, changed)))
        self.assertTrue(all(item["end_frame"] <= 3 for item in original))

    def test_search_changes_with_the_question_and_never_returns_other_recordings(self):
        car = [dict(row, track_id=8, class_name="Car") for row in self.tracks]
        index = build_index(self.recording, self.tracks + car)
        self.assertEqual(rank_evidence(self.question, index)[0]["track_id"], 7)
        question = dict(self.question, question="Which car?")
        self.assertEqual(rank_evidence(question, index)[0]["track_id"], 8)
        with self.assertRaises(ValueError):
            rank_evidence(dict(question, recording_id="elsewhere"), index)

    def test_hosted_processing_masks_lower_vehicle_region(self):
        payload = _hosted_image(self.image, [self.tracks[0]])
        with Image.open(io.BytesIO(payload)) as image:
            # JPEG rounding is allowed, but the lower box must be visibly masked.
            self.assertLess(max(image.getpixel((10, 18))), 60)
            self.assertGreater(image.getpixel((10, 7))[0], 150)

    def test_empty_evidence_makes_no_network_call(self):
        with patch("tools.emt_trial_adapters.urllib.request.urlopen") as request:
            answer = answer_question(self.question, [], self.root)
        self.assertEqual(answer["answer_status"], "insufficient_evidence")
        request.assert_not_called()

    def test_trial_adapter_is_explicit_and_supplies_no_detector(self):
        report = check_adapters("annotated", "none", "tools.emt_trial_adapters")
        self.assertTrue(report["ready"])
        self.assertTrue(all(row["function"].startswith("tools.emt_trial_adapters.") for row in report["adapters"]))
        report = check_adapters("predicted", "none", "tools.emt_trial_adapters")
        self.assertFalse(report["ready"])

    def test_trial_search_connects_to_the_actual_task2_reader(self):
        images = self.root / "images" / "train" / "images" / "video_000002"
        images.mkdir(parents=True)
        (images / "00001.png").write_bytes(self.image.read_bytes())
        labels = self.root / "annotations" / "train" / "video_000002.txt"
        labels.parent.mkdir(parents=True)
        labels.write_text("00001 7 Bus 0 0 0 4 4 20 20\n")
        recording = open_recording(self.root, "video_000002")
        self.assertEqual(type(recording).__module__, "trackrag.emt")
        tracks = load_annotated_tracks(recording)
        query = dict(self.question, recording_id="video_000002", query_frame=1)
        result = rank_evidence(query, build_index(recording, tracks))
        self.assertEqual(result[0]["track_id"], 7)

    def test_vehicle_query_does_not_select_pedestrian_vocabulary(self):
        pedestrian = [dict(row, track_id=8, class_name="Pedestrian") for row in self.tracks]
        query = dict(self.question, question="Which vehicle?")
        result = rank_evidence(query, build_index(self.recording, self.tracks + pedestrian))
        self.assertTrue(result)
        self.assertTrue(all(row["track_id"] == 7 for row in result))

    def evidence(self):
        return [{
            "evidence_id": "E1", "interval": {"track_id": 7},
            "images": [{"frame_id": 3, "image_path": "image.png", "bbox_xyxy": [4, 4, 20, 20]}],
        }]

    def test_live_adapter_payload_and_receipt_without_actual_upload(self):
        (self.root / "image.png").write_bytes(self.image.read_bytes())
        (self.root / "tracks.jsonl").write_text(json.dumps(self.tracks[2]) + "\n")
        response = {"candidates": [{"content": {"parts": [{"text": json.dumps({
            "answer_status": "answered", "answer": "Bus E1", "cited_evidence_ids": ["E1"],
        })}]}}], "usageMetadata": {"totalTokenCount": 1}}
        with patch.dict("os.environ", {"GEMINI_API_KEY": "unit-test-key"}):
            with patch("tools.emt_trial_adapters.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as network:
                result = answer_question(self.question, self.evidence(), self.root)
        self.assertEqual(result["cited_evidence_ids"], ["E1"])
        request = network.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["contents"][0]["parts"][-2]["text"], "Evidence E1, track 7, frame 3.")
        receipts = list((self.root / "hosted").glob("*/receipt.json"))
        self.assertEqual(len(receipts), 1)
        self.assertNotIn("unit-test-key", receipts[0].read_text())

    def test_http_failure_is_distinguishable_from_insufficient_evidence(self):
        (self.root / "image.png").write_bytes(self.image.read_bytes())
        (self.root / "tracks.jsonl").write_text(json.dumps(self.tracks[2]) + "\n")
        failure = urllib.error.HTTPError("https://example.test", 503, "busy", {}, None)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "unit-test-key"}):
            with patch("tools.emt_trial_adapters.urllib.request.urlopen", side_effect=failure):
                with self.assertRaisesRegex(RuntimeError, "503"):
                    answer_question(self.question, self.evidence(), self.root)
        receipt = next((self.root / "hosted").glob("*/receipt.json"))
        self.assertEqual(json.loads(receipt.read_text())["http_status"], 503)

    def test_local_report_escapes_question_text(self):
        result = {
            "question": dict(self.question, question="<script>bad</script>"), "source": "annotated",
            "evidence": [], "answer": {"answer_status": "not_requested", "answer": None, "cited_evidence_ids": []},
        }
        (self.root / "results.jsonl").write_text(json.dumps(result) + "\n")
        report = write_report(self.root).read_text()
        self.assertIn("&lt;script&gt;", report)
        self.assertNotIn("<script>bad", report)

    def test_report_distinguishes_insufficient_evidence_and_displays_citations(self):
        result = {
            "question": self.question, "source": "annotated", "evidence": [],
            "answer": {"answer_status": "insufficient_evidence", "answer": None, "cited_evidence_ids": []},
        }
        path = self.root / "results.jsonl"
        path.write_text(json.dumps(result) + "\n")
        report = write_report(self.root).read_text()
        self.assertIn("do not establish", report)
        self.assertNotIn("generation was disabled", report)
        result["evidence"] = [{
            "evidence_id": "E2", "interval": {"track_id": 7, "start_frame": 1, "end_frame": 3},
            "images": [],
        }]
        result["answer"] = {"answer_status": "answered", "answer": "Bus.", "cited_evidence_ids": ["E2"]}
        path.write_text(json.dumps(result) + "\n")
        report = write_report(self.root).read_text()
        self.assertIn('Cited evidence <a href="#question-1-evidence-1">E2</a>', report)

    def test_download_bound_is_enforced_even_if_server_ignores_range(self):
        stream = _BoundedStream(io.BytesIO(b"0123456789"), 4)
        self.assertEqual(stream.read(2), b"01")
        self.assertEqual(stream.read(50), b"23")
        self.assertEqual(stream.read(), b"")

    def test_recording_id_cannot_escape_the_data_directory(self):
        with self.assertRaises(ValueError):
            prepare_clip(self.root, "../elsewhere", 1, 10, 1)


if __name__ == "__main__":
    unittest.main()
