from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from trackrag.dropoff import (
    assert_local_split_guards, mask_target_and_box, model_input,
    parse_regseg_annotations, prepare_public_cases, read_local_cases,
    reserved_scene_guard, write_release,
)

PROJECT = Path(__file__).resolve().parents[1]


class MaskTests(unittest.TestCase):
    def test_nonconvex_mask_target_is_inside_and_deterministic(self):
        mask = np.zeros((31, 31), dtype=np.uint8)
        mask[3:28, 3:8] = 1
        mask[3:28, 23:28] = 1
        mask[3:8, 3:28] = 1  # U shape; its naive centroid is outside.
        target, box, clearance = mask_target_and_box(mask)
        self.assertEqual(mask[target[1], target[0]], 1)
        self.assertNotEqual(target, (15, 15))
        self.assertEqual(box, (3, 3, 28, 28))
        self.assertGreater(clearance, 0)
        self.assertEqual(target, mask_target_and_box(mask)[0])

    def test_single_pixel_mask(self):
        mask = np.zeros((5, 7), dtype=np.uint8)
        mask[2, 4] = 255
        self.assertEqual(mask_target_and_box(mask)[:2], ((4, 2), (4, 2, 5, 3)))

    def test_empty_mask_rejected(self):
        with self.assertRaisesRegex(ValueError, "no foreground"):
            mask_target_and_box(np.zeros((4, 4), dtype=np.uint8))


class PublicTests(unittest.TestCase):
    def test_actual_pinned_annotation_counts_and_train_only(self):
        train = parse_regseg_annotations(
            PROJECT / "annotations/dropoff/public_source/annotation_new_train.txt", "train"
        )
        val = parse_regseg_annotations(
            PROJECT / "annotations/dropoff/public_source/annotation_new_val.txt", "val"
        )
        self.assertEqual((len(train), len(val)), (170, 338))
        self.assertTrue(all(x["split"] == "train" for x in train + val))
        self.assertTrue(all(x["difficulty_tags"] == ["rnr_region"] for x in train + val))
        self.assertEqual(reserved_scene_guard(train)["status"], "not_enforceable")

    def test_pixel_join_exclusion_receipts_and_gold_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "imgs").mkdir()
            (root / "val_masks_new").mkdir()
            annotations = root / "annotations"
            annotations.mkdir()
            (annotations / "annotation_new_train.txt").write_text(
                "1 park near the car\n2 slow down behind truck\n3 turn left\n"
            )
            (annotations / "annotation_new_val.txt").write_text("")
            for image_id in (1, 2, 3):
                Image.new("RGB", (12, 10), "white").save(root / f"imgs/img_train_{image_id}.jpg")
            mask = np.zeros((10, 12), dtype=np.uint8)
            mask[2:8, 3:9] = 255
            Image.fromarray(mask).save(root / "val_masks_new/gt_img_ann_train_1.png")
            Image.fromarray(np.zeros((9, 12), dtype=np.uint8)).save(
                root / "val_masks_new/gt_img_ann_train_2.png"
            )
            Image.fromarray(mask).save(root / "val_masks_new/gt_img_ann_train_3.png")
            valid, excluded, summary = prepare_public_cases(root, annotations)
            self.assertEqual(len(valid), 2)
            self.assertEqual(len(excluded), 1)
            self.assertEqual(excluded[0]["case_id"], "regseg_train_2")
            self.assertIn("dimensions differ", excluded[0]["error"])
            self.assertEqual(valid[0]["target_xy"], [5, 4])
            self.assertEqual(valid[0]["coarse_box_xyxy"], [3, 2, 9, 8])
            self.assertFalse(summary["audit_matches_source"])
            self.assertEqual(set(model_input(valid[0])), {"case_id", "image_path", "instruction"})


class LocalReleaseTests(unittest.TestCase):
    def test_local_counts_release_and_split_guards(self):
        cases, excluded = read_local_cases(PROJECT / "annotations/dropoff/local")
        self.assertEqual(excluded, [])
        self.assertEqual(len(cases), 115)
        self.assertEqual(
            {s: sum(c["split"] == s for c in cases) for s in ("train", "dev", "eval")},
            {"train": 45, "dev": 20, "eval": 50},
        )
        assert_local_split_guards(cases)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_release(tmp, cases, [])
            self.assertEqual(manifest["counts"], {"train": 45, "dev": 20, "eval": 50})
            inputs = json.loads((Path(tmp) / "annotations/dropoff/model_inputs/eval.json").read_text())
            targets = json.loads((Path(tmp) / "annotations/dropoff/targets/eval.json").read_text())
            self.assertEqual(len(inputs), 50)
            self.assertTrue(all(set(x) == {"case_id", "image_path", "instruction"} for x in inputs))
            self.assertTrue(all("destination_track_id" in x for x in targets))
            for reviewer in (1, 2):
                with (Path(tmp) / f"reviews/reviewer_{reviewer}.csv").open() as handle:
                    self.assertEqual(sum(1 for _ in handle) - 1, 56)
            self.assertEqual(
                json.loads((Path(tmp) / "reviews/review_status.json").read_text())["status"],
                "pending",
            )

    def test_recording_overlap_rejected(self):
        cases, _ = read_local_cases(PROJECT / "annotations/dropoff/local")
        duplicate = dict(cases[0])
        duplicate["case_id"] = "different_case"
        duplicate["split"] = "eval"
        with self.assertRaisesRegex(ValueError, "occurs in train and eval"):
            assert_local_split_guards(cases + [duplicate])


if __name__ == "__main__":
    unittest.main()
