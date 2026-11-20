"""Generated pixels and a temporary pinned source guard; no model-quality claims."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from tools.dropoff_split_public import split_public
from trackrag.dropoff.data import file_sha256, load_cases, validate_disjoint_groups


@pytest.fixture
def public_data(tmp_path, monkeypatch):
    import trackrag.dropoff.data as data
    project = tmp_path / "fixture-project"
    guards = project / "configs" / "dropoff" / "regseg-scene-guards.json"
    guards.parent.mkdir(parents=True)
    scenes = [letter * 32 for letter in "abcdef"]
    scene_indices = [0, 0, 1, 2, 3, 4, 5]
    rules = {
        "regseg_revision": "a" * 40, "regseg_metadata_sha256": "b" * 64,
        "scene_tokens": scenes, "train_scene_indices": scene_indices,
        "reserved_scene_tokens": [scenes[4]], "concise_test_train_ids": [5],
        "train_command_sha256": [hashlib.sha256(f"fixture instruction {i}".encode()).hexdigest()
                                  for i in range(len(scene_indices))],
    }
    guards.write_text(json.dumps(rules), encoding="utf-8")
    monkeypatch.setattr(data, "__file__", str(project / "trackrag" / "dropoff" / "data.py"))
    monkeypatch.setattr(data, "REGSEG_GUARDS_SHA256", file_sha256(guards))
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "masks").mkdir()
    rows = []
    for i in range(len(scene_indices)):
        image, mask = raw / "images" / f"img_{i}.png", raw / "masks" / f"mask_{i}.png"
        Image.new("RGB", (16, 8), (i, 40, 90)).save(image)
        Image.new("L", (16, 8), 255).save(mask)
        rows.append({
            "schema_version": 1, "case_id": f"fixture-{i}",
            "scene_group": "talk2car-scene:" + scenes[scene_indices[i]], "partition": "train",
            "source": {
                "dataset": "Talk2Car-RegSeg", "source_split": "train", "source_index": i,
                "repository_revision": rules["regseg_revision"],
                "metadata_sha256": rules["regseg_metadata_sha256"],
                "scene_token": scenes[scene_indices[i]], "scene_group_verified": True,
                "image_sha256": file_sha256(image), "mask_sha256": file_sha256(mask),
                "mask_path": f"masks/mask_{i}.png",
            },
            "request": {
                "request_id": f"fixture-{i}", "text": f"fixture instruction {i}",
                "input_mode": "typed", "dialogue": [],
                "scene": {"source": "predicted", "tracks": [], "frame": {
                    "recording_id": f"fixture-{i}", "frame_id": 0, "timestamp_s": 0,
                    "width": 16, "height": 8, "image_path": f"images/img_{i}.png"}},
            },
            "target": {"destination_point_xy": [8, 4], "destination_box_xyxy": [0, 0, 16, 8]},
            "supervised_fields": ["destination_point_xy", "destination_box_xyxy"],
            "review": {"status": "public_annotation", "new_human_review": False},
        })
    def write(name, selected):
        path = raw / name
        path.write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
        return path
    return raw, rows, write


def test_grouped_split_keeps_seen_scenes_and_preserves_portable_supervision(public_data, tmp_path):
    raw, rows, write = public_data
    source = write("cases.jsonl", rows[:5])
    keep = write("keep.jsonl", [rows[0]])
    out = raw / "splits"
    manifest = split_public(source, out, development_cases=2, keep_training_cases=keep)
    train = load_cases(out / "train.jsonl")
    development = load_cases(out / "development.jsonl", allowed_partitions={"development"})
    validate_disjoint_groups(train, development)
    assert {"fixture-0", "fixture-1"} <= {row["case_id"] for row in train}
    assert manifest["train_cases"] == 3 and manifest["development_cases"] == 2
    assert manifest["final_evaluation_set"] is False and manifest["new_human_review"] is False
    assert manifest["source_cases_sha256"] == file_sha256(source)
    assert manifest["files"]["train.jsonl"]["sha256"] == file_sha256(out / "train.jsonl")
    original = {row["case_id"]: row for row in rows}
    for row in train + development:
        assert row["target"] == original[row["case_id"]]["target"]
        assert set(row["supervised_fields"]) == {"destination_point_xy", "destination_box_xyxy"}
        assert "status" not in row["target"]
    serialized = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert all(not Path(row["request"]["scene"]["frame"]["image_path"]).is_absolute() for row in serialized)
    moved = tmp_path / "moved"
    shutil.copytree(raw, moved)
    assert len(load_cases(moved / "splits" / "train.jsonl")) == 3


def test_same_seed_is_deterministic_and_existing_output_is_never_overwritten(public_data):
    raw, rows, write = public_data
    source = write("cases.jsonl", rows[:5])
    a = split_public(source, raw / "a", development_cases=2, seed=8)
    b = split_public(source, raw / "b", development_cases=2, seed=8)
    assert a == b
    assert (raw / "a" / "train.jsonl").read_bytes() == (raw / "b" / "train.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        split_public(source, raw / "a", development_cases=2)


def test_copied_image_under_another_scene_is_rejected_before_writing(public_data):
    raw, rows, write = public_data
    changed = deepcopy(rows[:5])
    changed[2]["request"]["scene"]["frame"]["image_path"] = changed[0]["request"]["scene"]["frame"]["image_path"]
    changed[2]["source"]["image_sha256"] = changed[0]["source"]["image_sha256"]
    source = write("copied.jsonl", changed)
    with pytest.raises(ValueError, match="Identical image"):
        split_public(source, raw / "copied-split", development_cases=1)
    assert not (raw / "copied-split").exists()


@pytest.mark.parametrize("kind", ["reserved", "development", "human_claim", "bad_hash"])
def test_complete_input_is_checked_before_split_selection(public_data, kind):
    raw, rows, write = public_data
    changed = deepcopy(rows[:5])
    if kind == "reserved":
        changed.append(rows[5])
    elif kind == "development":
        changed[-1]["partition"] = "development"
    elif kind == "human_claim":
        changed[-1]["review"]["new_human_review"] = True
    else:
        changed[-1]["source"]["mask_sha256"] = "0" * 64
    source = write("invalid.jsonl", changed)
    with pytest.raises(ValueError):
        split_public(source, raw / "invalid-split", development_cases=1)
    assert not (raw / "invalid-split").exists()


def test_prior_case_identity_cannot_change_supervision(public_data):
    raw, rows, write = public_data
    source = write("cases.jsonl", rows[:5])
    changed = deepcopy(rows[0])
    changed["target"]["destination_point_xy"] = [7, 4]
    keep = write("changed-keep.jsonl", [changed])
    with pytest.raises(ValueError, match="identity has changed"):
        split_public(source, raw / "changed-split", development_cases=1, keep_training_cases=keep)


def test_whole_groups_report_a_nonexact_target_without_breaking_scene(public_data):
    raw, rows, write = public_data
    source = write("groups.jsonl", rows[:3])
    keep = write("keep.jsonl", [rows[2]])
    manifest = split_public(source, raw / "nonexact", development_cases=1, keep_training_cases=keep)
    assert manifest["requested_development_cases"] == 1
    assert manifest["development_cases"] == 2 and manifest["train_cases"] == 1
    assert set(manifest["development_case_ids"]) == {"fixture-0", "fixture-1"}


def test_all_previously_seen_scenes_cannot_become_development(public_data):
    raw, rows, write = public_data
    source = write("cases.jsonl", rows[:3])
    with pytest.raises(ValueError, match="No source scene"):
        split_public(source, raw / "locked", development_cases=1, keep_training_cases=source)
