from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from PIL import Image

from tools.dropoff_passenger_references import _row, _index
from tools.dropoff_passenger_references_pack import build_pack
from tools.dropoff_passenger_references_public import score_outcome
from trackrag.dropoff.data import inference_input_sha256


class PassengerReferenceScoringTests(TestCase):
    def setUp(self):
        self.digest = "a" * 64
        self.request = {
            "request_id": "p", "text": "By the white car", "input_mode": "typed", "dialogue": [],
            "scene": {"frame": {"recording_id": "r", "frame_id": 1, "timestamp_s": 0,
                                "width": 100, "height": 100, "image_path": "example.jpg"},
                      "source": "predicted", "tracks": []}}
        self.target = {"status": "proposed", "destination_point_xy": [20, 20],
                       "destination_box_xyxy": [10, 10, 30, 30], "clarification_question": None}
        self.ref = {"request_id": "p", "recording_id": "r", "frame_id": 1,
                    "image_sha256": self.digest, "target": self.target}
        self.output = {"ok": True, "request_id": "p", "result": deepcopy(self.target),
                       "provenance": {"method": "qwen_landmark_region", "image": {"sha256": self.digest},
                                      "inference_input_sha256": inference_input_sha256(self.request, self.digest)}}

    def score(self, outputs):
        return _row(self.ref, self.request, outputs)

    def test_real_input_and_identical_reference_are_a_match(self):
        result = self.score([self.output])
        self.assertTrue(result["point_in_authored_box"])
        self.assertEqual(result["box_iou"], 1)
        self.assertEqual(result["normalized_point_error"], 0)

    def test_wrong_frame_image_or_utterance_fails(self):
        bound = self.output["provenance"]
        bound["image"]["sha256"] = "b" * 64
        self.assertEqual(self.score([self.output])["failure"], "invalid_or_mismatched_output")
        bound["image"]["sha256"] = self.digest
        self.request["text"] = "By the tree"
        self.assertFalse(self.score([self.output])["status_matches"])

    def test_duplicate_and_absent_predictions_are_failures(self):
        self.assertEqual(self.score([self.output, self.output])["failure"], "duplicate_output")
        self.assertEqual(self.score([])["failure"], "missing_output")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _index([self.request, self.request])

    def test_error_abstention_and_passenger_assistance_are_distinct(self):
        failure = self.score([{"ok": False, "request_id": "p", "error": {"code": "timeout"}}])
        self.assertEqual(failure["failure"], "timeout")
        self.output["result"] = {"status": "insufficient_visual_evidence", "destination_point_xy": None,
                                 "destination_box_xyxy": None, "clarification_question": None}
        self.assertIsNone(self.score([self.output])["failure"])
        self.output["provenance"]["method"] = "explicit_passenger_region"
        self.assertEqual(self.score([self.output])["failure"], "invalid_or_mismatched_output")

    def test_half_open_comparison_and_partial_targets(self):
        self.output["result"]["destination_point_xy"] = [30, 20]
        self.assertFalse(self.score([self.output])["point_in_authored_box"])
        self.output["result"].pop("status")
        self.assertEqual(self.score([self.output])["failure"], "invalid_or_mismatched_output")

    def test_pack_cannot_build_without_checked_blind_source(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaises(FileNotFoundError):
                build_pack(directory, directory / "requests", directory / "pack.html")
            self.assertFalse((directory / "pack.html").exists())

    def test_public_geometry_uses_the_original_mask_and_has_no_semantic_gold(self):
        with TemporaryDirectory() as temp:
            mask_path = Path(temp) / "human-region.png"
            mask = Image.new("L", (10, 10))
            mask.putpixel((2, 2), 255)
            mask.save(mask_path)
            case = {"case_id": "p", "scene_group": "new-scene",
                    "source": {"mask_path": str(mask_path), "image_sha256": self.digest},
                    "target": {"destination_point_xy": [20, 20], "destination_box_xyxy": [20, 20, 30, 30]}}
            result = score_outcome(case, self.request, [self.output])
            self.assertTrue(result["point_in_public_mask"])
            self.assertNotIn("status_matches", result)
            self.output["result"]["destination_point_xy"] = [10, 20]
            self.assertFalse(score_outcome(case, self.request, [self.output])["point_in_public_mask"])
            self.output["provenance"]["injected"] = True
            self.assertEqual(score_outcome(case, self.request, [self.output])["failure"], "invalid_or_unbound_output")
