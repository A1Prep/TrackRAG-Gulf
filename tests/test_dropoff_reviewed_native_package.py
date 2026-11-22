import json

import pytest

from tools.dropoff_import_task2_review import RECORDS
from tools.dropoff_package_reviewed_native import check_training


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_native_package_verifies_each_original_case_in_each_epoch(tmp_path):
    train = sorted(RECORDS["train"])
    development = sorted(RECORDS["development"])
    write_json(tmp_path / "dataset_manifest.json", {
        "train_case_ids": train, "development_case_ids": development,
        "train_scene_groups": ["emt-recording:" + k.split("-")[-2] for k in train],
        "development_scene_groups": ["emt-recording:" + k.split("-")[-2] for k in development],
    })
    write_json(tmp_path / "training_result.json", {
        "coverage": "clarification-only", "actual_training_completed": True, "injected_runtime": False,
        "training_examples_unique": 4, "examples_consumed": 8, "epochs_completed": 2,
        "optimizer_steps": 4, "actual_training_case_ids": train * 2,
    })
    log = tmp_path / "training-log.jsonl"
    rows = [{"epoch": e, "case_id": k} for e in (1, 2) for k in train]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert check_training(tmp_path)["optimizer_steps"] == 4

    # The total count per ID is unchanged, but two cases are in the wrong epoch.
    rows[0]["case_id"], rows[5]["case_id"] = rows[5]["case_id"], rows[0]["case_id"]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="training epoch differs"):
        check_training(tmp_path)
