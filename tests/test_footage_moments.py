"""Check corpus browsing separately from question answering at a paused frame."""
from threading import Lock
from types import SimpleNamespace

import pytest

from trackrag.footage_app import FootageReview
from trackrag.footage_moments import BUSY, PAGE_SIZE, recording_moments, search_type
from trackrag.passenger_app import PassengerSite


@pytest.mark.parametrize(("words", "expected"), [
    ("Find buses", "Bus"), ("Show me a motorcycle.", "Motorbike"), ("people on foot", "Pedestrian"),
    ("Look for an emergency vehicle", "Emergency_vehicle"), ("Busy traffic", BUSY),
    ("Which bus ran a red light?", None), ("find a red bus", None), ("cars without people", None),
])
def test_moment_query_recognizes_only_the_supported_claims(words, expected):
    assert search_type(words) == expected


def test_moment_uses_original_observed_box_with_repeated_label_and_only_this_recording():
    def row(track, frame, kind, right=20, *, recording="drive-a", source="predicted"):
        return {"recording_id": recording, "frame_id": frame, "track_id": track, "class_name": kind,
                "bbox_xyxy": [10, 5, right, 25], "source": source}

    clock = [{"id": 1, "time": 0.0}, {"id": 3, "time": 0.2}, {"id": 5, "time": 0.4}]
    rows = [row(7, 1, "Large_vehicle"), row(7, 3, "Bus"), row(7, 5, "Bus", 35), row(7, 5, "Bus", 35),
            row(4, 1, "Motorbike"), row(4, 3, "Car"), row(9, 5, "Car"), row(7, 6, "Bus", 90),
            row(7, 5, "Bus", 90, recording="drive-b"), row(8, 5, "Bus", source="annotated")]
    result = recording_moments("drive-a", clock, reversed(rows))
    assert list(result) == ["Bus", BUSY]
    sighting, = result["Bus"]
    assert sighting == {"track_id": 7, "class_name": "Bus", "frame_id": 5, "timestamp_s": 0.4,
                        "bbox_xyxy": [10, 5, 35, 25], "first_frame": 1, "first_box": [10, 5, 20, 25],
                        "observations": 2}
    assert result[BUSY] == [{"track_id": None, "frame_id": 1, "timestamp_s": 0.0, "observations": 2}]


def test_only_playable_recordings_enter_ranked_pages_and_failures_are_reported():
    site = object.__new__(PassengerSite)
    site.scene_lock = Lock()
    site.scenes = {key: {"frame": {"recording_id": rec, "frame_id": frame}}
                   for key, rec, frame in [("b", "drive-b", 1), ("a", "drive-a", 1), ("a-later", "drive-a", 12),
                                           ("c", "drive-c", 1), ("photo", "unverified", 1)]}
    site.footage_scenes = {"a", "a-later", "b", "c"}
    visited = []

    def moments(scene):
        frame = scene["frame"]
        assert frame["frame_id"] == 1
        recording = frame["recording_id"]
        visited.append(recording)
        if recording == "drive-c":
            raise FileNotFoundError(recording)
        entries = PAGE_SIZE if recording == "drive-a" else 1
        score = 3 if recording == "drive-b" else 2
        return {"Bus": [{"track_id": i, "frame_id": i + 1, "observations": score} for i in range(entries)]}

    site.footage = SimpleNamespace(moments=moments)
    first = site.footage_moments("buses", 0)
    assert sorted(visited) == ["drive-a", "drive-b", "drive-c"]
    assert first["recordings_searched"] == 2 and first["recordings_unavailable"] == 1
    assert first["total"] == PAGE_SIZE + 1 and first["next_offset"] == PAGE_SIZE
    assert first["matches"][0]["scene"] == "b"
    second = site.footage_moments("buses", first["next_offset"])
    assert second["next_offset"] is None
    combined = first["matches"] + second["matches"]
    assert len({(row["recording_id"], row["track_id"]) for row in combined}) == PAGE_SIZE + 1


def test_removing_an_original_frame_cuts_the_searchable_prefix_even_with_a_warm_cache(tmp_path, monkeypatch):
    review = FootageReview(tmp_path, tmp_path, tmp_path / "vision", tmp_path / "out")
    (tmp_path / "drive-a.tracks.jsonl").write_bytes(b"fixture tracks")
    paths = {i: tmp_path / f"{i}.jpg" for i in (1, 2, 3, 4)}
    for path in paths.values():
        path.write_bytes(b"image fixture")
    clock = [{"id": i, "time": float(i)} for i in paths]
    def source(scene, last):
        assert paths[last].is_file()
        return None, [{"recording_id": "drive-a", "frame_id": i, "track_id": 17, "class_name": "Bus",
                       "bbox_xyxy": [5, 5, 10 + i, 20], "source": "predicted"} for i in range(1, last + 1)], None

    monkeypatch.setattr(review, "_clip", lambda scene: (clock, paths))
    monkeypatch.setattr(review, "_source", source)
    scene = {"frame": {"recording_id": "drive-a", "frame_id": 1},
             "provenance": {"tracks_sha256": "fixture", "prediction_provenance_sha256": "fixture"}}
    assert review.moments(scene)["Bus"][0]["frame_id"] == 4
    assert review.timeline(scene)["frames"] == clock
    paths[3].unlink()
    assert review.moments(scene)["Bus"][0]["frame_id"] == 2
    assert review.timeline(scene)["frames"] == clock[:2]
    assert set(review.timeline(scene)["frame_context"]) == {1, 2}
