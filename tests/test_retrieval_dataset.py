"""Task 4 relevance-review checks using generated PNGs, never real labels/models."""

from copy import deepcopy
import hashlib
from html import escape
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from trackrag.retrieval import RetrievalConfig, build_index
from trackrag.retrieval_candidates import build_candidate_batch
from trackrag.retrieval_dataset import export_reviewed, prepare_review
from trackrag.retrieval_review_html import render_review


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def refresh_metadata_hash(batch):
    portable = deepcopy(batch)
    portable.pop("metadata_sha256")
    for candidate in portable["candidates"]:
        for image in candidate["images"]:
            image.pop("image_path")
    batch["metadata_sha256"] = canonical_hash(portable)


class ImageRecording:
    def __init__(self, directory, recording_id):
        self.recording_id = recording_id
        directory.mkdir(parents=True)
        self.frames = {}
        for frame_id in range(1, 4):
            image_path = directory / f"{frame_id}.png"
            Image.new("RGB", (64, 48), (frame_id * 40, 20, 70)).save(image_path)
            self.frames[frame_id] = {
                "recording_id": recording_id, "frame_id": frame_id,
                "timestamp_s": frame_id / 10, "width": 64, "height": 48,
                "image_path": str(image_path),
            }

    def get_frame(self, frame_id):
        return self.frames[frame_id]

    def iter_frames(self):
        raise AssertionError("Dataset preparation must use explicit candidate images")


class RetrievalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest_path = self.root / "splits.json"
        self.write_json(self.manifest_path, {
            "train": ["recording-a"], "dev": ["recording-b"], "test": ["recording-c"],
        })
        self.packet_path, self.batch = self.make_packet("recording-a")
        self.review_dir = self.root / "review"
        self.export_dir = self.root / "export"

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def read_jsonl(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    @staticmethod
    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")

    def make_packet(self, recording_id, source="annotated", question_id="q-bus"):
        recording = ImageRecording(self.root / f"input-{recording_id}-{source}", recording_id)
        tracks = [
            {"recording_id": recording_id, "frame_id": frame_id,
             "track_id": track_id, "class_name": class_name,
             "bbox_xyxy": (5, 8, 20, 25), "source": source}
            for track_id, class_name in ((7, "Bus"), (8, "Car"))
            for frame_id in range(1, 4)
        ]
        index = build_index(recording, tracks, config=RetrievalConfig(window_frames=2, stride_frames=1))
        question = {"question_id": question_id, "recording_id": recording_id,
                    "query_frame": 3, "question": "Which bus is visible?"}
        batch = build_candidate_batch(question, index, source=source)
        packet_path = self.root / f"{recording_id}-{source}.json"
        self.write_json(packet_path, batch)
        return packet_path, batch

    def prepare(self, paths=None):
        prepare_review(paths or [self.packet_path], self.manifest_path, self.review_dir)
        return self.read_json(self.review_dir / "pack.json")

    def labels(self):
        return self.read_jsonl(self.review_dir / "labels.jsonl")

    def candidate_for(self, row):
        pack = self.read_json(self.review_dir / "pack.json")
        case = next(case for case in pack["cases"] if case["case_id"] == row["case_id"])
        batch = self.read_json(self.review_dir / case["batch_file"])
        return next(candidate for candidate in batch["candidates"]
                    if candidate["candidate_id"] == row["candidate_id"])

    def reviewed(self, row, judgment="relevant", reviewer_type="human"):
        row.update({
            "judgment": judgment, "reviewer": "test-reviewer", "reviewer_type": reviewer_type,
            "reason": {"relevant": "supports_question", "irrelevant": "wrong_vehicle",
                       "uncertain": "ambiguous"}[judgment],
            "note": "PRIVATE REVIEW NOTE: checked all representative images.",
            "checked_frame_ids": [image["frame_id"] for image in self.candidate_for(row)["images"]],
            "reviewed_at": "2026-11-18T12:00:00+04:00",
        })
        return row

    def export(self, rows=None, split="train"):
        if rows is not None:
            self.write_jsonl(self.review_dir / "labels.jsonl", rows)
        return export_reviewed(self.review_dir, self.review_dir / "labels.jsonl",
                               self.export_dir, split=split)

    def replace_packet(self, batch, *, refresh=True):
        if refresh:
            refresh_metadata_hash(batch)
        self.write_json(self.packet_path, batch)

    def test_prepare_binds_metadata_split_images_and_all_pending_labels(self):
        pack = self.prepare()
        self.assertEqual(pack["schema_version"], "trackrag.relevance-review.v1")
        self.assertEqual(pack["split_manifest"], self.read_json(self.manifest_path))
        portable = deepcopy(pack)
        portable.pop("pack_sha256")
        self.assertEqual(pack["pack_sha256"], canonical_hash(portable))
        self.assertEqual(len(pack["cases"]), 1)
        case = pack["cases"][0]
        self.assertEqual(case["split"], "train")
        self.assertEqual(case["recording_id"], "recording-a")
        self.assertEqual(case["source"], "annotated")
        self.assertEqual(case["metadata_sha256"], self.batch["metadata_sha256"])
        self.assertEqual(case["batch_sha256"], hashlib.sha256(
            (self.review_dir / case["batch_file"]).read_bytes()).hexdigest())
        for asset in pack["assets"]:
            self.assertEqual(asset["sha256"], hashlib.sha256(
                (self.review_dir / asset["path"]).read_bytes()).hexdigest())
        self.assertTrue((self.review_dir / "review.html").is_file())
        rows = self.labels()
        self.assertEqual({row["candidate_id"] for row in rows},
                         {candidate["candidate_id"] for candidate in self.batch["candidates"]})
        for row in rows:
            self.assertEqual(row["pack_sha256"], pack["pack_sha256"])
            self.assertEqual(row["judgment"], "pending")
            self.assertEqual(row["reviewer"], "")
            self.assertIsNone(row["reviewer_type"])
            self.assertIsNone(row["reason"])
            self.assertEqual(row["note"], "")
            self.assertEqual(row["checked_frame_ids"], [])
            self.assertIsNone(row["reviewed_at"])

    def test_prepared_pack_survives_changes_to_original_inputs(self):
        pack = self.prepare()
        self.packet_path.write_text("not the old packet", encoding="utf-8")
        self.write_json(self.manifest_path, {"train": [], "dev": [], "test": ["recording-a"]})
        for image in self.batch["candidates"][0]["images"]:
            Path(image["image_path"]).write_bytes(b"changed source image")
        rows = self.labels()
        self.reviewed(rows[0])
        result = self.export(rows)
        self.assertEqual(result["exported_examples"], 1)
        self.assertEqual(self.read_json(self.review_dir / "pack.json"), pack)

    def test_prepare_rejects_stale_metadata_hash(self):
        changed = deepcopy(self.batch)
        changed["question"]["question"] = "Which car is visible?"
        self.replace_packet(changed, refresh=False)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_recomputed_hash_does_not_allow_a_future_candidate_image(self):
        changed = deepcopy(self.batch)
        changed["candidates"][0]["images"][-1]["frame_id"] = 4
        self.replace_packet(changed)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_recomputed_hash_does_not_allow_mismatched_recording_or_source(self):
        for field, bad_value in (("recording_id", "recording-b"), ("source", "predicted")):
            with self.subTest(field=field):
                self.review_dir = self.root / ("geometry-" + field)
                changed = deepcopy(self.batch)
                changed["candidates"][0][field] = bad_value
                self.replace_packet(changed)
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_extra_gold_question_field_is_not_allowed_into_model_inputs(self):
        changed = deepcopy(self.batch)
        changed["question"]["gold_answer"] = "The bus."
        self.replace_packet(changed)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_prepare_checks_image_decoding_and_original_dimensions(self):
        image_path = Path(self.batch["candidates"][0]["images"][0]["image_path"])
        for payload in (b"not an image", None):
            with self.subTest(payload=payload):
                if payload is None:
                    Image.new("RGB", (32, 24)).save(image_path)
                else:
                    image_path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_manifest_requires_disjoint_named_partitions_and_known_recordings(self):
        invalid = [
            {"train": ["recording-a"], "dev": ["recording-a"], "test": []},
            {"train": [], "dev": [], "test": []},
            {"train": ["recording-a", "recording-a"], "dev": [], "test": []},
            {"train": ["recording-a"], "dev": []},
            {"train": ["recording-a"], "dev": [], "test": [], "extra": []},
        ]
        for manifest in invalid:
            with self.subTest(manifest=manifest):
                self.write_json(self.manifest_path, manifest)
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_prepare_never_overwrites_an_existing_directory(self):
        self.review_dir.mkdir()
        sentinel = self.review_dir / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises((ValueError, FileExistsError)):
            self.prepare()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_pending_pack_cannot_be_exported_as_training_labels(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "reviewed"):
            self.export()
        self.assertFalse(self.export_dir.exists())

    def test_partial_review_exports_only_explicit_human_labels(self):
        self.prepare()
        rows = self.labels()
        self.reviewed(rows[0])
        self.reviewed(rows[1], "irrelevant")
        result = self.export(rows)
        self.assertEqual(result["exported_examples"], 2)
        examples = self.read_jsonl(self.export_dir / "examples.jsonl")
        self.assertEqual(sorted(example["relevance"] for example in examples), [0, 1])
        self.assertEqual(len(examples), 2)
        self.assertEqual(self.read_json(self.export_dir / "summary.json")["exported_examples"], 2)
        for example in examples:
            features = example["model_input"]
            self.assertEqual(features["question"], self.batch["question"])
            self.assertEqual(features["source"], "annotated")
            self.assertIn("candidate", features)
            self.assertTrue(set(features) <= {"question", "source", "candidate", "history"})
            encoded = json.dumps(features)
            self.assertNotIn("PRIVATE REVIEW NOTE", encoded)
            self.assertNotIn("test-reviewer", encoded)
            self.assertNotIn("judgment", encoded)
            self.assertNotIn("relevance", encoded)
            self.assertIn("provenance", example)
            for image in features["candidate"]["images"]:
                copied = self.export_dir / image["image_path"]
                self.assertTrue(copied.is_file())
                self.assertTrue(copied.resolve().is_relative_to(self.export_dir.resolve()))

    def test_uncertain_review_is_excluded_and_never_becomes_a_negative(self):
        self.prepare()
        rows = self.labels()
        self.reviewed(rows[0])
        self.reviewed(rows[1], "uncertain")
        result = self.export(rows)
        self.assertEqual(result["exported_examples"], 1)
        examples = self.read_jsonl(self.export_dir / "examples.jsonl")
        self.assertEqual([example["relevance"] for example in examples], [1])

    def test_ai_assisted_labels_are_not_promoted_to_human_training_examples(self):
        self.prepare()
        rows = self.labels()
        self.reviewed(rows[0])
        self.reviewed(rows[1], "irrelevant", reviewer_type="ai_assisted")
        result = self.export(rows)
        self.assertEqual(result["exported_examples"], 1)
        examples = self.read_jsonl(self.export_dir / "examples.jsonl")
        self.assertEqual([example["relevance"] for example in examples], [1])

    def test_summary_counts_entire_cohort_including_pending_candidates(self):
        self.prepare()
        rows = self.labels()
        self.reviewed(rows[0])
        self.reviewed(rows[1], "uncertain")
        result = self.export(rows)
        self.assertEqual(result["judgment_counts"]["pending"], len(rows) - 2)
        self.assertEqual(result["judgment_counts"]["relevant"], 1)
        self.assertEqual(result["judgment_counts"]["uncertain"], 1)
        self.assertEqual(result["selected_split_judgment_counts"], result["judgment_counts"])
        self.assertEqual(len(result["questions"]), 1)

    def test_train_export_cannot_mix_development_or_test_recordings(self):
        dev_path, _ = self.make_packet("recording-b", question_id="q-dev")
        test_path, _ = self.make_packet("recording-c", question_id="q-test")
        self.prepare([self.packet_path, dev_path, test_path])
        rows = self.labels()
        for row in rows:
            self.reviewed(row)
        result = self.export(rows, split="train")
        examples = self.read_jsonl(self.export_dir / "examples.jsonl")
        self.assertEqual(result["exported_examples"], len(self.batch["candidates"]))
        self.assertEqual({row["model_input"]["question"]["recording_id"] for row in examples},
                         {"recording-a"})

    def test_sources_with_identical_track_ids_have_distinct_review_cases(self):
        predicted_path, _ = self.make_packet("recording-a", "predicted")
        pack = self.prepare([self.packet_path, predicted_path])
        self.assertEqual(len({case["case_id"] for case in pack["cases"]}), 2)
        self.assertEqual({case["source"] for case in pack["cases"]}, {"annotated", "predicted"})

    def test_export_rejects_unknown_duplicate_or_missing_label_rows(self):
        self.prepare()
        original = self.labels()
        self.reviewed(original[0])
        unknown = deepcopy(original)
        unknown[0]["candidate_id"] = "C-not-in-pack"
        for rows in (unknown, original + [deepcopy(original[0])], original[:-1]):
            with self.subTest(row_count=len(rows)):
                with self.assertRaises(ValueError):
                    self.export(rows)

    def test_export_rejects_label_for_another_pack(self):
        self.prepare()
        rows = self.labels()
        self.reviewed(rows[0])
        rows[0]["pack_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.export(rows)

    def test_reviewed_label_requires_identity_note_time_and_checked_frames(self):
        self.prepare()
        original = self.labels()
        self.reviewed(original[0])
        changes = [
            ("reviewer", " "), ("note", ""), ("reviewed_at", None),
            ("reviewed_at", "2026-11-18T12:00:00"), ("checked_frame_ids", []),
            ("checked_frame_ids", [999]), ("reason", "wrong_vehicle"),
        ]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                rows = deepcopy(original)
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    self.export(rows)

    def test_export_rejects_changed_snapshot_image(self):
        pack = self.prepare()
        rows = self.labels()
        self.reviewed(rows[0])
        image = self.review_dir / pack["assets"][0]["path"]
        image.write_bytes(b"tampered pixels")
        with self.assertRaises(ValueError):
            self.export(rows)

    def test_export_rejects_changed_batch_snapshot(self):
        pack = self.prepare()
        rows = self.labels()
        self.reviewed(rows[0])
        path = self.review_dir / pack["cases"][0]["batch_file"]
        batch = self.read_json(path)
        batch["question"]["question"] = "A different question"
        self.write_json(path, batch)
        with self.assertRaises(ValueError):
            self.export(rows)

    def test_export_rejects_changed_split_snapshot(self):
        pack = self.prepare()
        rows = self.labels()
        self.reviewed(rows[0])
        pack["split_manifest"] = {"train": [], "dev": ["recording-a"], "test": []}
        self.write_json(self.review_dir / "pack.json", pack)
        with self.assertRaises(ValueError):
            self.export(rows)

    def test_export_never_overwrites_existing_output(self):
        self.prepare()
        rows = self.labels()
        self.reviewed(rows[0])
        self.export_dir.mkdir()
        sentinel = self.export_dir / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises((ValueError, FileExistsError)):
            self.export(rows)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_export_rejects_nonpartition_selection(self):
        self.prepare()
        rows = self.labels()
        self.reviewed(rows[0])
        with self.assertRaises(ValueError):
            self.export(rows, split="all")

    def test_recomputed_hash_cannot_hide_contradictory_endpoint_geometry(self):
        for field, value in (("first_center_normalized", [1, 1]),
                             ("center_displacement_normalized", [1, 1]),
                             ("center_range_normalized", [0.5, 0.5])):
            with self.subTest(field=field):
                self.review_dir = self.root / ("geometry-check-" + field)
                changed = deepcopy(self.batch)
                changed["candidates"][0]["features"][field] = value
                self.replace_packet(changed)
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_empty_candidate_questions_remain_visible_in_export_accounting(self):
        recording = ImageRecording(self.root / "empty-recording", "recording-b")
        empty = build_candidate_batch(
            {"question_id": "q-empty", "recording_id": "recording-b", "query_frame": 4,
             "question": "What is visible at this unavailable frame?"},
            build_index(recording, [], config=RetrievalConfig(window_frames=2, stride_frames=1)),
            source="annotated",
        )
        path = self.root / "empty.json"
        self.write_json(path, empty)
        self.prepare([self.packet_path, path])
        rows = self.labels()
        self.reviewed(rows[0])
        report = self.export(rows)
        self.assertEqual(len(report["questions"]), 2)
        case = next(row for row in report["questions"] if row["question_id"] == "q-empty")
        self.assertEqual(case["candidate_count"], 0)
        self.assertTrue(case["empty_pool"])
        self.assertFalse(case["fully_human_labeled"])
        self.assertFalse(case["fully_reviewed_without_positive"])

    def test_question_text_cannot_inject_active_html_or_script(self):
        attack = '</script><script>window.injected=true</script><img src=x onerror="alert(1)">'
        changed = deepcopy(self.batch)
        changed["question"]["question"] = attack
        self.replace_packet(changed)
        self.prepare()
        page = (self.review_dir / "review.html").read_text(encoding="utf-8")
        self.assertNotIn(attack, page)
        self.assertIn(escape(attack, quote=True), page)

    def test_renderer_rejects_remote_image_references(self):
        pack = self.prepare()
        case = pack["cases"][0]
        batch = self.read_json(self.review_dir / case["batch_file"])
        batch["candidates"][0]["images"][0]["image_path"] = "https://example.invalid/private.jpg"
        with self.assertRaises(ValueError):
            render_review(pack, {case["case_id"]: batch}, self.labels())


if __name__ == "__main__":
    unittest.main()
