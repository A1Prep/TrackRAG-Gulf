"""Metric sanity for tools/vision_evaluate.py (runs where motmetrics and pycocotools are installed)."""

import json

import pytest

pytest.importorskip("motmetrics")
pytest.importorskip("pycocotools")

from tools import vision_evaluate as ve  # noqa: E402


def _write_run(tmp_path, rows_by_frame, size=(2560, 1440), shift=0.0, drop_track_frames=()):
    """Labels and predictions for one recording; predictions = labels (optionally shifted/dropped)."""
    labels = tmp_path / "labels" / "annotations" / "test"
    labels.mkdir(parents=True)
    run = tmp_path / "run"
    run.mkdir()
    label_lines, tracks, detections = [], [], []
    frames = sorted(rows_by_frame)
    for frame in frames:
        mapped = []
        for track, name, (l, t, r, b) in rows_by_frame[frame]:
            label_lines.append(f"{frame:05d} {track} {name} 0.0 0 0.0 {l} {t} {r} {b} 0 0 0 0 0 0 0")
            box = [l + shift, t, r + shift, b]
            mapped.append({"bbox_xyxy": box, "score": 0.9, "class_name": name, "native": name})
            if (frame, track) not in drop_track_frames:
                tracks.append({"recording_id": "rec", "frame_id": frame, "track_id": track + 100,
                               "class_name": name, "bbox_xyxy": box, "source": "predicted"})
        detections.append({"recording_id": "rec", "frame_id": frame, "width": size[0], "height": size[1],
                           "native": mapped, "mapped": mapped})
    (labels / "rec.txt").write_text("\n".join(label_lines) + "\n")
    (run / "rec.tracks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tracks))
    (run / "rec.detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in detections))
    (run / "rec.provenance.json").write_text(json.dumps({
        "frame_ids": frames, "first_frame_size": list(size), "observations": len(tracks), "identity_hash": "x"}))
    return run, tmp_path / "labels"


SCENE = {f: [(1, "Car", (100 + 10 * f, 500, 300 + 10 * f, 600)), (2, "Pedestrian", (1500, 400, 1560, 600))]
         for f in range(1, 11)}


def test_perfect_predictions_score_one(tmp_path):
    run, labels = _write_run(tmp_path, SCENE)
    result = ve.evaluate(run, labels, "test", ["rec"])
    tracking = result["tracking"]["road_user"]["OVERALL"]
    assert tracking["idf1"] == pytest.approx(1.0) and tracking["mota"] == pytest.approx(1.0)
    assert tracking["num_switches"] == 0
    assert result["detection"]["agnostic"]["summary"]["AP50"] == pytest.approx(1.0)
    assert result["detection"]["group"]["per_class"]["Pedestrian"]["AP50"] == pytest.approx(1.0)
    assert result["operating_point"]["groups"]["road_user"]["f1"] == pytest.approx(1.0)
    assert result["track_coverage"]["coverage"] == pytest.approx(1.0)


def test_misses_and_low_overlap_are_counted(tmp_path):
    # Shift every box by 150 px on a 200 px wide car -> IoU < 0.5 for the car; pedestrian also shifted away.
    run, labels = _write_run(tmp_path, SCENE, shift=150.0)
    result = ve.evaluate(run, labels, "test", ["rec"])
    tracking = result["tracking"]["road_user"]["OVERALL"]
    assert tracking["num_matches"] == 0 and tracking["recall"] == pytest.approx(0.0)
    assert result["detection"]["agnostic"]["summary"]["AP50"] == pytest.approx(0.0)


def test_track_rows_must_match_provenance_count(tmp_path):
    run, labels = _write_run(tmp_path, SCENE)
    lines = (run / "rec.tracks.jsonl").read_text().splitlines()
    (run / "rec.tracks.jsonl").write_text("\n".join(lines[:5]) + "\n")
    with pytest.raises(ValueError, match="row count"):
        ve.evaluate(run, labels, "test", ["rec"])
