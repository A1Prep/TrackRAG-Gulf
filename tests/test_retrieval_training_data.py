"""Generated packets/images and synthetic review assertions, not actual human labels."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from trackrag.retrieval_dataset import prepare_review
from trackrag.retrieval_qwen import text_documents
from trackrag.retrieval_training_data import (
    _seal, load_dataset, prepare_reviewed, synthetic_dataset, validate_dataset, write_json,
)


class TrainingDataTests(unittest.TestCase):
    def test_synthetic_data_has_disjoint_recordings_both_labels_and_inference_documents(self):
        bundle = synthetic_dataset()
        samples = validate_dataset(bundle)
        self.assertEqual(bundle["kind"], "synthetic_smoke")
        self.assertEqual(len(samples), 4)
        self.assertEqual({sample["relevance"] for sample in samples if sample["split"] == "train"}, {0, 1})
        for batch in bundle["batches"]:
            expected = set(text_documents(batch))
            selected = [sample for sample in samples if sample["question"] == batch["question"]]
            self.assertTrue(all(sample["document"] in expected for sample in selected))
            self.assertTrue(all("label_provenance" not in sample["document"] for sample in selected))

    def test_hash_unknown_labels_duplicates_and_single_class_are_rejected(self):
        for change in ("hash", "label", "bool", "duplicate", "candidate", "one_class"):
            bundle = synthetic_dataset()
            if change == "hash":
                bundle["provenance"]["changed"] = True
            elif change == "label":
                bundle["examples"][0]["relevance"] = "pending"
            elif change == "bool":
                bundle["examples"][0]["relevance"] = True
            elif change == "duplicate":
                bundle["examples"].append(deepcopy(bundle["examples"][0]))
            elif change == "candidate":
                bundle["examples"][0]["candidate_id"] = "foreign"
            else:
                for example in bundle["examples"]:
                    if example["split"] == "train":
                        example["relevance"] = 1
            if change != "hash":
                _seal(bundle)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_dataset(bundle)

    def test_test_recordings_overlap_and_wrong_partition_rejected(self):
        for change in ("test", "overlap", "wrong_example_split", "no_dev"):
            bundle = synthetic_dataset()
            if change == "test":
                bundle["split_manifest"]["test"] = bundle["split_manifest"].pop("dev")
                bundle["split_manifest"]["dev"] = []
            elif change == "overlap":
                bundle["split_manifest"]["test"] = ["synthetic-train"]
            elif change == "wrong_example_split":
                bundle["examples"][0]["split"] = "dev"
            else:
                bundle["examples"] = [row for row in bundle["examples"] if row["split"] == "train"]
                bundle["batches"] = bundle["batches"][:1]
            _seal(bundle)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_dataset(bundle)

    def test_future_observations_gold_fields_and_synthetic_misrepresentation_rejected(self):
        for change in ("future", "gold", "pretend_human"):
            bundle = synthetic_dataset()
            if change == "future":
                bundle["batches"][0]["histories"][0]["observations"][0]["frame_id"] = 2
            elif change == "gold":
                bundle["batches"][0]["question"]["gold_answer"] = "bus"
            else:
                bundle["kind"] = "human_reviewed"
            _seal(bundle)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_dataset(bundle)

    def make_review(self, root):
        bundle = synthetic_dataset()
        paths = []
        image_path = root / "frame.png"
        Image.new("RGB", (100, 80)).save(image_path)
        for number, batch in enumerate(bundle["batches"]):
            for candidate in batch["candidates"]:
                for image in candidate["images"]:
                    image["image_path"] = str(image_path)
            path = root / f"packet-{number}.json"
            write_json(path, batch)
            paths.append(path)
        manifest = root / "splits.json"
        write_json(manifest, bundle["split_manifest"])
        review = root / "review"
        pack = prepare_review(paths, manifest, review)
        return bundle, pack, review

    def test_pending_pack_cannot_become_a_training_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, review = self.make_review(root)
            output = root / "dataset.json"
            with self.assertRaises(ValueError):
                prepare_reviewed(review, review / "labels.jsonl", output)
            self.assertFalse(output.exists())

    def test_reviewed_preparation_uses_existing_checks_and_excludes_pending_scenes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, _, review = self.make_review(root)
            labels_path = review / "labels.jsonl"
            rows = [json.loads(line) for line in labels_path.read_text().splitlines()]
            judgments = {example["candidate_id"]: example["relevance"] for example in bundle["examples"]}
            for row in rows:
                if row["candidate_id"] not in judgments:
                    continue
                positive = judgments[row["candidate_id"]]
                row.update(judgment="relevant" if positive else "irrelevant", reviewer="Synthetic test assertion",
                           reviewer_type="human", reason="supports_question" if positive else "wrong_vehicle",
                           note="Generated fixture assertion, not a real human review", checked_frame_ids=[1],
                           reviewed_at="2026-11-20T12:00:00+04:00")
            labels_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "dataset.json"
            prepared = prepare_reviewed(review, labels_path, output)
            self.assertEqual(prepared["kind"], "human_reviewed")
            self.assertEqual(len(validate_dataset(load_dataset(output))), 4)
            self.assertEqual(prepared["provenance"]["counts"]["excluded"], 2)
            self.assertTrue(all("reviewer" not in sample["document"] for sample in validate_dataset(prepared)))
            tracked = prepare_reviewed(review, labels_path, root / "tracked.json", tracked_only=True)
            self.assertTrue(tracked["provenance"]["tracked_only"])
            self.assertEqual(tracked["provenance"]["counts"]["excluded_scene"], 2)
            self.assertTrue(all(json.loads(sample["document"])["evidence_kind"] != "scene"
                                for sample in validate_dataset(tracked)))
            with self.assertRaises(FileExistsError):
                prepare_reviewed(review, labels_path, output)
            rows[0]["reviewer_type"] = "ai_assisted"
            labels_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_reviewed(review, labels_path, root / "not-human.json")

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text('{"kind":"synthetic_smoke","kind":"human_reviewed"}')
            with self.assertRaises(ValueError):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
