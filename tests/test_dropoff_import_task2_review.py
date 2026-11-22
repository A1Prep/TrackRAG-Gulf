import hashlib
from io import BytesIO
import json

from PIL import Image
import pytest

from tools import dropoff_import_task2_review as intake
from trackrag.dropoff.contracts import FULL_FIELDS
from trackrag.dropoff.data import inference_input_sha256, supervision_sha256


def fixture_source():
    files, lines = {}, []
    for partition, ids in intake.RECORDS.items():
        for identity in sorted(ids):
            _, _, rec, number = identity.split("-")
            path = f"images/{partition}/{rec}/{number}.jpg"
            raw = BytesIO()
            Image.new("RGB", (64, 32), (len(lines) * 30, 1, 17)).save(raw, format="JPEG")
            files[path] = raw.getvalue()
            image_sha = hashlib.sha256(files[path]).hexdigest()
            frame = {"frame_id": int(number), "recording_id": rec, "image_path": path,
                     "timestamp_s": 4, "width": 64, "height": 32}
            request = {"request_id": identity, "input_mode": "typed", "text": "Beside the car", "dialogue": [],
                       "scene": {"source": "predicted", "frame": frame, "tracks": []}}
            target = {"status": "needs_clarification", "clarification_question": "Which car do you mean?",
                      "destination_point_xy": None, "destination_box_xyxy": None}
            fields = sorted(FULL_FIELDS)
            row = {"case_id": identity, "schema_version": 1, "scene_group": rec, "partition": partition,
                   "request": request, "source": {"dataset": "test_fixture", "image_sha256": image_sha},
                   "target": target, "supervised_fields": fields,
                   "review": {"status": "human_reviewed", "software_fixture": True,
                              "approval_source": "two_independent_human_confirmations", "new_human_review": True,
                              "image_sha256": image_sha, "input_sha256": inference_input_sha256(request, image_sha),
                              "supervision_sha256": supervision_sha256(target, fields, frame),
                              "reviewed_at": "2026-11-20T10:00:00+00:00", "reviewers": ["fixture-a", "fixture-b"]}}
            lines.append(json.dumps(row).encode())
    # Invalid deliberately: nonfinal intake must never parse or request a reserved target.
    lines.append(b'{"partition":"test","target":THIS MUST NOT BE DECODED}')
    files["review_cases.jsonl"] = b"\n".join(lines)
    return files


def test_export_preserves_nonfinal_pixels_bindings_and_excludes_test(tmp_path, monkeypatch):
    files = fixture_source()
    paths = []
    def fetch(repo, name):
        paths.append(name)
        return files[name]
    monkeypatch.setattr(intake, "git_blob", fetch)
    output = tmp_path / "reviewed"
    result = intake.materialize(".", output)
    assert result["reserved_rows_materialized"] == 0
    assert len(set(paths)) == 7 and all("test" not in p for p in paths)
    assert [result["splits"][p]["examples"] for p in intake.PARTITIONS] == [4, 2]
    assert all((output / p).read_bytes() == content for p, content in files.items() if p.endswith(".jpg"))
    with pytest.raises(FileExistsError):
        intake.materialize(".", output)


@pytest.mark.parametrize("mutation", ["image", "review", "missing"])
def test_incomplete_or_modified_reviews_leave_no_partial_output(tmp_path, monkeypatch, mutation):
    files = fixture_source()
    rows = files["review_cases.jsonl"].splitlines()
    row = json.loads(rows[0])
    if mutation == "image":
        files[row["request"]["scene"]["frame"]["image_path"]] = b"changed"
    elif mutation == "review":
        row["request"]["text"] = "Instruction changed after review"
        rows[0] = json.dumps(row).encode()
    else:
        rows.pop(0)
    files["review_cases.jsonl"] = b"\n".join(rows)
    monkeypatch.setattr(intake, "git_blob", lambda repo, path: files[path])
    with pytest.raises(ValueError):
        intake.materialize(".", tmp_path / "reviewed")
    assert list(tmp_path.iterdir()) == []
