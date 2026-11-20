"""Preparation tests use generated EMT-layout images/annotations, not real labels."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from trackrag.emt import load_annotated_tracks, open_recording
from trackrag.retrieval import RetrievalConfig
from trackrag.retrieval_dataset import _load_pack
from trackrag.retrieval_realdata import main, prepare_realdata
from trackrag.retrieval_training_data import prepare_reviewed, write_json


class RealDataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        annotation_dir = self.data / "annotations/train"
        annotation_dir.mkdir(parents=True)
        for identity in ("recording-a", "recording-b"):
            images = self.data / "images/train/images" / identity
            images.mkdir(parents=True)
            lines = []
            for frame in range(1, 5):
                Image.new("RGB", (100, 80), (frame * 40, 10, 10)).save(images / f"{frame:05d}.png")
                for track, label, left in ((7, "Bus", 10 + frame), (42, "Car", 60 - frame)):
                    lines.append(f"{frame} {track} {label} 0 0 0 {left} 20 {left + 15} 40\n")
            (annotation_dir / f"{identity}.txt").write_text("".join(lines))
        self.questions = [{"question_id": f"q-{identity}", "recording_id": identity, "query_frame": 3,
                           "question": "Which bus is observed in frame 3?"} for identity in ("recording-a", "recording-b")]
        self.questions_path = self.root / "questions.jsonl"
        self.write_questions(self.questions)
        self.splits = self.root / "splits.json"
        write_json(self.splits, {"train": ["recording-a"], "dev": ["recording-b"], "test": ["recording-c"]})
        self.config = RetrievalConfig(window_frames=2, stride_frames=2)

    def write_questions(self, questions):
        self.questions_path.write_text("".join(json.dumps(row) + "\n" for row in questions), encoding="utf-8")

    def prepare(self, output="prepared", **kwargs):
        with redirect_stderr(io.StringIO()):
            return prepare_realdata(self.data, self.questions_path, self.splits, self.root / output,
                                    config=self.config, **kwargs)

    def test_two_recordings_one_command_pending_pack_and_short_windows(self):
        result = self.prepare()
        self.assertEqual(result["recordings"], ["recording-a", "recording-b"])
        self.assertEqual(result["status"], "prepared_unreviewed")
        self.assertEqual(result["human_labels"], 0)
        self.assertFalse(result["training_completed"])
        self.assertIsNone(result["accuracy_metrics"])
        self.assertEqual(result["pending_labels"], result["candidate_count"])
        self.assertTrue(all(case["candidate_kinds"].get("track_window", 0) > 0 for case in result["cases"]))
        pack, lookup = _load_pack(self.root / "prepared/review")
        self.assertEqual(pack["pack_sha256"], result["pack_sha256"])
        self.assertEqual({case["split"] for case in pack["cases"]}, {"train", "dev"})
        for identity, relative in result["question_files_by_recording"].items():
            rows = [json.loads(line) for line in (self.root / "prepared" / relative).read_text().splitlines()]
            self.assertTrue(rows)
            self.assertTrue(all(row["recording_id"] == identity for row in rows))
        for _, batch, candidate in lookup.values():
            self.assertLessEqual(candidate["end_frame"], 3)
            self.assertTrue(all(row["frame_id"] <= 3 for history in batch["histories"] for row in history["observations"]))
        labels = [json.loads(row) for row in (self.root / "prepared/review/labels.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["judgment"] == "pending" for row in labels))
        with self.assertRaises(ValueError):
            prepare_reviewed(self.root / "prepared/review", self.root / "prepared/review/labels.jsonl",
                             self.root / "cannot-train.json", tracked_only=True)

    def test_future_image_and_observation_changes_preserve_review_pack_hash(self):
        first = self.prepare("first")
        for identity in ("recording-a", "recording-b"):
            path = self.data / "annotations/train" / f"{identity}.txt"
            with path.open("a") as stream:
                stream.write("4 999 Bus 0 0 0 1 1 10 10\n")
            Image.new("RGB", (100, 80), "blue").save(self.data / "images/train/images" / identity / "00004.png")
        second = self.prepare("second")
        self.assertEqual(first["pack_sha256"], second["pack_sha256"])
        self.assertEqual(first["candidate_count"], second["candidate_count"])
        self.assertEqual([row["metadata_sha256"] for row in first["cases"]],
                         [row["metadata_sha256"] for row in second["cases"]])

    def test_test_recordings_gold_fields_empty_questions_and_missing_frames_rejected(self):
        variants = [[], [dict(self.questions[0], recording_id="recording-c")],
                    [dict(self.questions[0], gold_answer="bus")], [dict(self.questions[0], query_frame=99)]]
        for number, questions in enumerate(variants):
            self.write_questions(questions)
            with self.subTest(number=number), self.assertRaises((ValueError, KeyError)):
                self.prepare("invalid-" + str(number))
            self.assertFalse((self.root / f"invalid-{number}").exists())

    def test_saved_predictions_keep_their_own_source_and_track_ids(self):
        rows = []
        for identity in ("recording-a", "recording-b"):
            rows += [dict(row, source="predicted", track_id=row["track_id"] + 100)
                     for row in load_annotated_tracks(open_recording(self.data, identity))]
        path = self.root / "predictions.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        result = self.prepare(source="predicted", predicted_tracks=path)
        _, lookup = _load_pack(self.root / "prepared/review")
        self.assertEqual(result["source"], "predicted")
        self.assertTrue(all(candidate["source"] == "predicted" for _, _, candidate in lookup.values()))
        self.assertTrue(all(candidate["track_id"] in (None, 107, 142) for _, _, candidate in lookup.values()))

    def test_source_arguments_foreign_predictions_and_existing_output_rejected(self):
        with self.assertRaises(ValueError):
            self.prepare(source="predicted")
        path = self.root / "foreign.jsonl"
        path.write_text('{"recording_id":"other","source":"predicted"}\n')
        with self.assertRaises(ValueError):
            self.prepare(source="predicted", predicted_tracks=path)
        self.prepare()
        before = (self.root / "prepared/preparation.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual((self.root / "prepared/preparation.json").read_bytes(), before)

    def test_cli_help_explains_no_model_or_training_call(self):
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["--help"])
        self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
