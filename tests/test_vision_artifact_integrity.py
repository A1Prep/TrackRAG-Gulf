"""Saved-artifact boundaries; no model, GPU, or optional metric packages required."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from tools import vision_evaluate as ve
from tools import vision_saved_adapters as saved


IDENTITY = "a" * 64


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def artifact(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    labels = tmp_path / "labels"
    label_dir = labels / "annotations" / "test"
    label_dir.mkdir(parents=True)
    tracks = [dict(recording_id="rec", frame_id=f, track_id=7, source="predicted",
                   class_name="Car", bbox_xyxy=[10, 10, 30, 30]) for f in (1, 4)]
    write_jsonl(run / "rec.tracks.jsonl", tracks)
    provenance = dict(version=1, recording_id="rec", identity_hash=IDENTITY,
                      frame_ids=[1, 2, 4], frames=3, first_frame=1, last_frame=4,
                      first_frame_size=[100, 60], observations=2,
                      tracks_sha256=hashlib.sha256((run / "rec.tracks.jsonl").read_bytes()).hexdigest())
    (run / "rec.provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    detections = [dict(recording_id="rec", frame_id=f, width=100, height=60,
                       mapped=([dict(class_name="Car", bbox_xyxy=[10, 10, 30, 30], score=0.9)]
                               if f != 2 else [])) for f in (1, 2, 4)]
    write_jsonl(run / "rec.detections.jsonl", detections)
    (label_dir / "rec.txt").write_text(
        "1 1 Car 0 0 0 10 10 30 30 0 0 0 0 0 0 0\n"
        "4 1 Car 0 0 0 10 10 30 30 0 0 0 0 0 0 0\n", encoding="utf-8")
    return SimpleNamespace(run=run, labels=labels, provenance=provenance, tracks=tracks,
                           detections=detections,
                           config=SimpleNamespace(name="fixture", identity_hash=lambda: IDENTITY))


def save_provenance(a):
    (a.run / "rec.provenance.json").write_text(json.dumps(a.provenance), encoding="utf-8")


def resave_tracks(a):
    write_jsonl(a.run / "rec.tracks.jsonl", a.tracks)
    a.provenance["tracks_sha256"] = hashlib.sha256((a.run / "rec.tracks.jsonl").read_bytes()).hexdigest()
    a.provenance["observations"] = len(a.tracks)
    save_provenance(a)


def recording(frames=(1, 2, 4), **metadata):
    def get_frame(frame):
        return dict(recording_id="rec", frame_id=frame, width=100, height=60) | metadata
    return SimpleNamespace(recording_id="rec", frame_ids=list(frames), get_frame=get_frame)


def serve(a, rec=None):
    return saved.predict_tracks(rec or recording(), directory=a.run, config=a.config)


def load(a, tracking_only=False):
    return ve.load_recording(a.run, a.labels, "test", "rec", tracking_only=tracking_only)


@pytest.mark.parametrize("field", ["tracks_sha256", "identity_hash", "recording_id"])
def test_missing_integrity_fields_cannot_be_bypassed_by_stripping_version(artifact, field):
    a = artifact
    a.provenance.pop("version")
    a.provenance.pop(field)
    save_provenance(a)
    for action in (lambda: serve(a), lambda: load(a, True)):
        with pytest.raises(ValueError):
            action()


@pytest.mark.parametrize("value", [None, "", "x" * 64, "a" * 63])
def test_tracks_hash_must_be_valid_sha256(artifact, value):
    artifact.provenance["tracks_sha256"] = value
    save_provenance(artifact)
    with pytest.raises(ValueError, match="tracks_sha256"):
        serve(artifact)


@pytest.mark.parametrize("truncate", [False, True])
def test_tampered_tracks_rejected_by_hash_before_parsing_or_count(artifact, truncate):
    path = artifact.run / "rec.tracks.jsonl"
    path.write_bytes(b"not json" if truncate else path.read_bytes().replace(b'"Car"', b'"Bus"'))
    for action in (lambda: serve(artifact), lambda: load(artifact, True)):
        with pytest.raises(ValueError, match="sha256 differs"):
            action()


def test_row_count_independently_validated_with_matching_hash(artifact):
    artifact.provenance["observations"] += 1
    save_provenance(artifact)
    with pytest.raises(ValueError, match="row count"):
        load(artifact, True)


def test_available_prefix_preserves_original_frame_ids_and_empty_frame(artifact):
    rows = serve(artifact, recording((1, 2)))
    assert [row["frame_id"] for row in rows] == [1]
    data = load(artifact)
    assert data["frames"] == [1, 2, 4]
    assert data["detections"].get(2, []) == []
    assert "detections.jsonl" in data["files"]


@pytest.mark.parametrize("frames", [(2, 4), (1, 4), (1, 2, 3), (1, 2, 4, 5), (1, 1), ()])
def test_nonprefix_or_invalid_available_frames_rejected(artifact, frames):
    with pytest.raises(ValueError, match="prefix|Frame IDs"):
        serve(artifact, recording(frames))


@pytest.mark.parametrize("metadata", [{"recording_id": "other"}, {"frame_id": 99}, {"width": 99}])
def test_available_frame_metadata_rejected(artifact, metadata):
    with pytest.raises(ValueError, match="metadata|size"):
        serve(artifact, recording(**metadata))


def test_all_available_frames_checked_not_only_first(artifact):
    rec = recording()
    get_frame = rec.get_frame
    rec.get_frame = lambda f: get_frame(f) | ({"height": 59} if f == 2 else {})
    with pytest.raises(ValueError, match="size"):
        serve(artifact, rec)


@pytest.mark.parametrize("update", [{"source": "ground_truth"}, {"recording_id": "other"},
                                   {"frame_id": 5}, {"frame_id": 4.0}, {"track_id": -1},
                                   {"class_name": "person"}, {"bbox_xyxy": [0, 0, 101, 20]},
                                   {"bbox_xyxy": [0, 0, float("nan"), 20]},
                                   {"bbox_xyxy": [20, 10, 10, 30]}])
def test_rejects_invalid_tracks_even_outside_requested_prefix(artifact, update):
    artifact.tracks[-1].update(update)
    resave_tracks(artifact)
    for action in (lambda: serve(artifact, recording((1,))), lambda: load(artifact, True)):
        with pytest.raises(ValueError):
            action()


def test_duplicate_track_observations_rejected(artifact):
    artifact.tracks.append(artifact.tracks[0].copy())
    resave_tracks(artifact)
    with pytest.raises(ValueError, match="Duplicate track"):
        load(artifact, True)


@pytest.mark.parametrize("frames", [[1, 1, 4], [1, 4, 2], [1, 2, 4.0], []])
def test_invalid_provenance_frame_ids_rejected(artifact, frames):
    artifact.provenance["frame_ids"] = frames
    save_provenance(artifact)
    with pytest.raises(ValueError, match="Frame IDs"):
        load(artifact, True)


@pytest.mark.parametrize("duplicate", [False, True])
def test_detection_coverage_requires_unique_complete_frames_including_empty(artifact, duplicate):
    rows = artifact.detections + [artifact.detections[1]] if duplicate else [
        row for row in artifact.detections if row["frame_id"] != 2]
    write_jsonl(artifact.run / "rec.detections.jsonl", rows)
    with pytest.raises(ValueError, match="Duplicate detection|coverage differs"):
        load(artifact)


@pytest.mark.parametrize("update", [{"recording_id": "other"}, {"source": "ground_truth"},
                                   {"frame_id": 3}, {"frame_id": 1.0}, {"width": 99}, {"mapped": None}])
def test_detection_frame_identity_and_geometry_rejected(artifact, update):
    artifact.detections[0].update(update)
    write_jsonl(artifact.run / "rec.detections.jsonl", artifact.detections)
    with pytest.raises(ValueError):
        load(artifact)


@pytest.mark.parametrize("update", [{"bbox_xyxy": [0, 0, 101, 20]}, {"bbox_xyxy": [0, 10, 0, 20]},
                                   {"bbox_xyxy": [0, 0, float("inf"), 20]}, {"score": float("nan")},
                                   {"score": 1.1}, {"class_name": "person"}])
def test_invalid_detection_boxes_scores_and_classes_rejected(artifact, update):
    artifact.detections[0]["mapped"][0].update(update)
    write_jsonl(artifact.run / "rec.detections.jsonl", artifact.detections)
    with pytest.raises(ValueError):
        load(artifact)


@pytest.mark.parametrize("exists", [False, True])
def test_tracking_only_cli_never_reads_detections_or_claims_ap(artifact, monkeypatch, exists):
    a = artifact
    path = a.run / "rec.detections.jsonl"
    if exists:
        path.write_bytes(b"invalid JSON must not be read")
    else:
        path.unlink()
    seen = []

    def tracking(data):
        seen.extend(data)
        return {}

    def forbidden(*args, **kwargs):
        raise AssertionError("Detection metric called in tracking-only mode")

    monkeypatch.setattr(ve, "tracking", tracking)
    for name in ("coco_detection", "operating_point", "track_coverage"):
        monkeypatch.setattr(ve, name, forbidden)
    out = a.run / "tracking-metrics.json"
    assert ve.main(["--pred-dir", str(a.run), "--labels-root", str(a.labels), "--split", "test",
                    "--recordings", "rec", "--out", str(out), "--tracking-only"]) == 0
    result = json.loads(out.read_text())
    assert result["tracking_only"] is True and result["frames"] == {"rec": 3}
    assert len(seen) == 1 and len(seen[0]["tracks"]) == 2
    assert not ({"detection", "detection_per_recording", "operating_point", "track_coverage"} & result.keys())
    assert set(result["inputs_sha256"]["rec"]) == {"tracks.jsonl", "provenance.json", "labels"}
    assert "coverage" not in ve.brief(result)
    # The default remains strict even for archives that legitimately contain only tracks.
    with pytest.raises((FileNotFoundError, json.JSONDecodeError)):
        load(a)
