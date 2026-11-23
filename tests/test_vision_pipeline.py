"""predict_tracks pipeline tests. CI-safe: a scripted detector and a simple IoU tracker stand in for
RF-DETR and BoT-SORT; tests using the real ultralytics BOTSORT are skipped where it is not installed."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from trackrag.vision import predict_tracks
from trackrag.vision.config import PLACEHOLDER, config_from_dict, load_config
from trackrag.vision.detections import Detection
from trackrag.vision.detector import verify_checkpoint
from trackrag.vision.geometry import iou
from trackrag.vision.pipeline import postprocess

PROJECT = Path(__file__).resolve().parents[1]
SIZE = (64, 36)  # long side 32 in the test config -> resized 32x18, scale 2 in both axes


def make_config(label_space="emt", long_side=32, **tracker):
    detector = {"label_space": label_space, "checkpoint": "model.pth", "checkpoint_sha256": "0" * 64,
                "inference_long_side": long_side}
    return config_from_dict({"name": "test", "detector": detector, "tracker": tracker})


class FakeRecording:
    """Frames are solid images whose red value encodes the frame id (lossless PNG)."""

    def __init__(self, root, recording_id, frame_ids, size=SIZE, textured=False):
        import random

        self.recording_id = recording_id
        self.frames = {}
        root.mkdir(parents=True, exist_ok=True)
        for frame_id in frame_ids:
            path = root / f"{recording_id}_{frame_id:05d}.png"
            if not path.exists():
                image = Image.new("RGB", size, (frame_id, 0, 0))
                if textured:  # features for the real tracker's optical-flow motion compensation
                    rng = random.Random(7)
                    for _ in range(400):
                        x, y = rng.randrange(1, size[0] - 1), rng.randrange(1, size[1] - 1)
                        image.putpixel((x, y), (frame_id, 255, 255))
                    image.putpixel((0, 0), (frame_id, 0, 0))
                image.save(path)
            self.frames[frame_id] = {"recording_id": recording_id, "frame_id": frame_id, "timestamp_s": frame_id / 10,
                                     "width": size[0], "height": size[1], "image_path": str(path)}
        self.frame_ids = sorted(self.frames)

    def iter_frames(self):
        return (self.frames[frame_id] for frame_id in self.frame_ids)

    def get_frame(self, frame_id):
        return self.frames[frame_id]


class ScriptedDetector:
    """Returns scripted detections (given in ORIGINAL pixels) scaled to the image it receives."""

    kind = "stub"

    def __init__(self, script, original_size=SIZE):
        self.script, self.original_size, self.calls = script, original_size, []

    def detect(self, image):
        frame_id = image.getpixel((0, 0))[0]
        self.calls.append(frame_id)
        sx, sy = self.original_size[0] / image.size[0], self.original_size[1] / image.size[1]
        return [Detection((l / sx, t / sy, r / sx, b / sy), score, name)
                for (l, t, r, b), score, name in self.script.get(frame_id, [])]


class FakeIoUTracker:
    def __init__(self, settings):
        self.tracks, self.next_id, self.updates = {}, 1, 0

    def update(self, detections, image_bgr):
        self.updates += 1
        pairs, used = [], set()
        for index, detection in enumerate(detections):
            best = max(((iou(detection.box, box), track) for track, box in self.tracks.items() if track not in used),
                       default=(0.0, None))
            track = best[1] if best[0] >= 0.3 else None
            if track is None:
                track, self.next_id = self.next_id, self.next_id + 1
            used.add(track)
            self.tracks[track] = detection.box
            pairs.append((track, index))
        return pairs


def moving_script(frames):
    return {f: [((2 + 2 * f, 10, 14 + 2 * f, 20), 0.9, "Car"), ((40, 4, 60, 30), 0.8, "Bus")] for f in frames}


def run(recording, script, factory=FakeIoUTracker, **options):
    return predict_tracks(recording, config=make_config(**options), detector=ScriptedDetector(script),
                          tracker_factory=factory)


def test_frozen_continuity_matches_saved_replay_and_online_prefix(tmp_path):
    from tools import vision_relink_tracks, vision_saved_adapters
    from trackrag.vision.__main__ import JsonlSink
    from trackrag.vision.config import CONTINUITY_KIND

    class SplitTracker:
        def __init__(self, settings):
            self.frame = 0

        def update(self, detections, image):
            self.frame += 1
            return [(index + (10 if self.frame == 1 else 20), index) for index in range(len(detections))]

    original = make_config()
    active = original.with_overrides(tracker={"kind": CONTINUITY_KIND})
    config_file = tmp_path / "target.json"
    config_file.write_text(json.dumps({"name": active.name, "detector": active.identity()["detector"],
                                      "tracker": active.identity()["tracker"]}))
    script = moving_script([1, 2, 3])
    recording = FakeRecording(tmp_path / "images", "drive", [1, 2, 3])
    base = tmp_path / "base"
    base.mkdir()
    predict_tracks(recording, config=original, detector=ScriptedDetector(script), tracker_factory=SplitTracker,
                   sink=JsonlSink(base, "drive"))
    live = predict_tracks(recording, config=active, detector=ScriptedDetector(script), tracker_factory=SplitTracker)
    prefix = FakeRecording(tmp_path / "images", "drive", [1, 2])
    assert predict_tracks(prefix, config=active, detector=ScriptedDetector(script), tracker_factory=SplitTracker
                          ) == [row for row in live if row["frame_id"] <= 2]
    assert len({row["track_id"] for row in live}) == 2
    output = tmp_path / "relinked"
    args = ["--pred-dir", str(base), "--out", str(output), "--recordings", "drive", "--config", str(config_file)]
    assert vision_relink_tracks.main(args) == 0
    assert vision_saved_adapters.predict_tracks(recording, directory=output, config=active) == live
    assert json.loads((output / "drive.provenance.json").read_text())["track_handoffs"] == 2
    with pytest.raises(ValueError, match="identity"):
        vision_saved_adapters.predict_tracks(recording, directory=output, config=original)
    with pytest.raises(SystemExit):
        vision_relink_tracks.main(args)
    # A failed multi-recording replay must not publish partial predictions.
    failed_output = tmp_path / "failed"
    with pytest.raises(FileNotFoundError):
        vision_relink_tracks.main(["--pred-dir", str(base), "--out", str(failed_output),
                                  "--recordings", "drive", "missing", "--config", str(config_file)])
    assert not failed_output.exists()


def test_contract_types_and_downstream_acceptance(tmp_path):
    from trackrag.retrieval import build_index

    recording = FakeRecording(tmp_path, "rec_a", range(1, 6))
    rows = run(recording, moving_script(range(1, 6)))
    assert len(rows) == 10
    for row in rows:
        assert set(row) == {"recording_id", "frame_id", "track_id", "class_name", "bbox_xyxy", "source"}
        assert row["source"] == "predicted" and row["recording_id"] == "rec_a"
        assert type(row["frame_id"]) is int and type(row["track_id"]) is int
        assert isinstance(row["bbox_xyxy"], tuple) and all(type(v) is float for v in row["bbox_xyxy"])
    assert rows == sorted(rows, key=lambda r: (r["frame_id"], r["track_id"]))
    car = [r for r in rows if r["class_name"] == "Car"]
    assert car[0]["bbox_xyxy"] == (4.0, 10.0, 16.0, 20.0)  # mapped back from the 32x18 image exactly
    assert len({r["track_id"] for r in car}) == 1
    json.dumps(rows, allow_nan=False)
    build_index(recording, rows)  # Task 4's real validation accepts the rows


def test_new_recording_starts_with_fresh_tracker(tmp_path):
    first = FakeRecording(tmp_path / "a", "rec_a", range(1, 5))
    second = FakeRecording(tmp_path / "b", "rec_b", range(1, 5))
    detector = ScriptedDetector(moving_script(range(1, 5)))
    config = make_config()
    predict_tracks(first, config=config, detector=detector, tracker_factory=FakeIoUTracker)
    after = predict_tracks(second, config=config, detector=detector, tracker_factory=FakeIoUTracker)
    alone = run(FakeRecording(tmp_path / "b", "rec_b", range(1, 5)), moving_script(range(1, 5)))
    assert after == alone and min(r["track_id"] for r in after) == 1


def test_output_is_causal_prefix_stable(tmp_path):
    script = moving_script(range(1, 9))
    script[6] = []  # a disappearance, then a re-appearance
    full = run(FakeRecording(tmp_path, "rec", range(1, 9)), script)
    for cut in (1, 3, 6, 7):
        truncated = run(FakeRecording(tmp_path, "rec", range(1, cut + 1)), script)
        assert truncated == [r for r in full if r["frame_id"] <= cut]


def test_empty_and_missing_frames_keep_original_numbers(tmp_path):
    trackers = []

    def factory(settings):
        trackers.append(FakeIoUTracker(settings))
        return trackers[-1]

    script = moving_script([1, 2, 5, 9])
    recording = FakeRecording(tmp_path, "rec", [1, 2, 3, 5, 9])  # 3 has no detections; 4, 6-8 missing
    rows = run(recording, script, factory=factory)
    assert sorted({r["frame_id"] for r in rows}) == [1, 2, 5, 9]
    assert trackers[0].updates == 9  # frames 1..9: every present frame plus one update per missing id


def test_coco_path_suppresses_duplicates_before_rider_merge():
    person = Detection((100, 60, 140, 150), 0.90, "person")
    motorcycle = Detection((95, 100, 145, 160), 0.85, "motorcycle")
    duplicate_bicycle = Detection((95, 100, 145, 160), 0.07, "bicycle")
    mapped, dropped = postprocess([person, motorcycle, duplicate_bicycle], "coco", 0.8, 0.1)
    assert [(d.name, d.box) for d in mapped] == [("Motorbike", (95.0, 60.0, 145.0, 160.0))]
    assert abs(mapped[0].score - 0.875) < 1e-12 and dropped == {}


def test_emt_path_drops_background_and_unknown_names():
    mapped, dropped = postprocess([Detection((0, 0, 10, 10), 0.5, "Car"), Detection((0, 0, 10, 10), 0.3, "__background__"),
                                   Detection((20, 0, 30, 10), 0.2, "")], "emt", 0.8, 0.1)
    assert [d.name for d in mapped] == ["Car"] and dropped == {"__background__": 1, "<empty>": 1}


def test_coco_pipeline_exports_emt_names(tmp_path):
    script = {f: [((10, 4, 20, 30), 0.9, "person"), ((30, 10, 50, 20), 0.8, "truck")] for f in (1, 2)}
    rows = run(FakeRecording(tmp_path, "rec", [1, 2]), script, label_space="coco")
    assert {r["class_name"] for r in rows} == {"Pedestrian", "Large_vehicle"}


def test_config_identity_ignores_machine_paths_and_refuses_placeholders(tmp_path, monkeypatch):
    base = load_config(PROJECT / "configs" / "vision" / "pretrained.json")
    assert base.detector.label_space == "coco" and base.detector.checkpoint_md5 and base.detector.checkpoint_sha256
    elsewhere = base.with_overrides(weights_dir=tmp_path, device="cpu")
    assert elsewhere.identity_hash() == base.identity_hash()
    assert base.with_overrides(fuse_score=not base.tracker.fuse_score).identity_hash() != base.identity_hash()
    with pytest.raises(ValueError, match="placeholder"):
        config_from_dict({"name": "x", "detector": {"label_space": "emt", "checkpoint": "m.pth",
                                                    "checkpoint_sha256": PLACEHOLDER}})
    tuned = load_config(PROJECT / "configs" / "vision" / "finetuned.json")
    assert tuned.detector.label_space == "emt" and len(tuned.detector.checkpoint_sha256) == 64
    int(tuned.detector.checkpoint_sha256, 16)
    with pytest.raises(ValueError, match="Unknown"):
        config_from_dict({"name": "x", "detector": {"label_space": "emt", "checkpoint": "m.pth",
                                                    "checkpoint_sha256": "0" * 64, "typo": 1}})
    monkeypatch.delenv("TRACKRAG_VISION_WEIGHTS_DIR", raising=False)
    with pytest.raises(ValueError, match="weights"):
        base.checkpoint_path()


def test_checkpoint_hash_is_verified_before_any_model_library(tmp_path):
    config = make_config().with_overrides(weights_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_checkpoint(config)
    (tmp_path / "model.pth").write_bytes(b"not the pinned checkpoint")
    with pytest.raises(ValueError, match="sha256"):
        verify_checkpoint(config)


def _save_predictions(folder, recording, rows, config):
    import hashlib

    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{recording.recording_id}.tracks.jsonl").open("w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    (folder / f"{recording.recording_id}.provenance.json").write_text(json.dumps({
        "version": 1, "recording_id": recording.recording_id, "identity_hash": config.identity_hash(),
        "tracks_sha256": hashlib.sha256((folder / f"{recording.recording_id}.tracks.jsonl").read_bytes()).hexdigest(),
        "frame_ids": recording.frame_ids, "first_frame_size": list(SIZE), "observations": len(rows)}))


def test_saved_prediction_adapter_prefix_and_identity(tmp_path):
    from tools import vision_saved_adapters as saved

    config = make_config()
    full = FakeRecording(tmp_path / "img", "rec", range(1, 9))
    rows = run(full, moving_script(range(1, 9)))
    _save_predictions(tmp_path / "pred", full, rows, config)
    prefix = FakeRecording(tmp_path / "img", "rec", range(1, 5))
    loaded = saved.predict_tracks(prefix, directory=tmp_path / "pred", config=config)
    assert loaded == [r for r in rows if r["frame_id"] <= 4]
    with pytest.raises(ValueError, match="prefix"):
        saved.predict_tracks(FakeRecording(tmp_path / "img", "rec", [2, 3]), directory=tmp_path / "pred", config=config)
    with pytest.raises(ValueError, match="identity"):
        saved.predict_tracks(prefix, directory=tmp_path / "pred", config=make_config(fuse_score=False))
    # A truncated tracks file is refused.
    tracks = tmp_path / "pred" / "rec.tracks.jsonl"
    tracks.write_text("\n".join(tracks.read_text().splitlines()[:3]) + "\n")
    with pytest.raises(ValueError, match="sha256"):
        saved.predict_tracks(prefix, directory=tmp_path / "pred", config=config)


def test_frozen_replay_reproduces_a_live_run(tmp_path, monkeypatch):
    import trackrag.emt
    from tools import vision_saved_adapters as saved
    from tools.vision_tune_tracker import frozen
    from trackrag.vision.__main__ import JsonlSink

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"name": "test", "detector": {
        "label_space": "emt", "checkpoint": "model.pth", "checkpoint_sha256": "0" * 64, "inference_long_side": 32}}))
    config = load_config(config_path)
    recording = FakeRecording(tmp_path / "img", "rec", range(1, 9))
    script = moving_script(range(1, 9))
    script[4] = []
    live = tmp_path / "live"
    live.mkdir()
    rows = predict_tracks(recording, config=config, detector=ScriptedDetector(script),
                          tracker_factory=FakeIoUTracker, sink=JsonlSink(live, "rec"))
    monkeypatch.setattr(trackrag.emt, "open_recording", lambda root, recording_id: recording)

    frozen(tmp_path, ["rec"], live, config_path, tmp_path / "final", tracker_factory=FakeIoUTracker)
    final = tmp_path / "final"
    for name in ("detections.jsonl", "tracks.jsonl"):
        assert (final / f"rec.{name}").read_bytes() == (live / f"rec.{name}").read_bytes()
    provenance = json.loads((final / "rec.provenance.json").read_text())
    assert provenance["identity_hash"] == config.identity_hash() and provenance["detector_kind"] == "replay"
    assert saved.predict_tracks(recording, directory=final, config=config) == rows

    frozen(tmp_path, ["rec"], live, config_path, tmp_path / "prefix", max_frames=5, tracker_factory=FakeIoUTracker)
    assert [json.loads(line) for line in (tmp_path / "prefix" / "rec.tracks.jsonl").read_text().splitlines()] == \
        [json.loads(json.dumps(row)) for row in rows if row["frame_id"] <= 5]

    lines = (live / "rec.detections.jsonl").read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["mapped"] = []
    (live / "rec.detections.jsonl").write_bytes(("\n".join([json.dumps(tampered)] + lines[1:]) + "\n").encode())
    with pytest.raises(ValueError, match="differ"):
        frozen(tmp_path, ["rec"], live, config_path, tmp_path / "bad", tracker_factory=FakeIoUTracker)
    assert not (tmp_path / "bad").exists() and not (tmp_path / "bad.partial").exists()
    with pytest.raises(FileExistsError):
        frozen(tmp_path, ["rec"], live, config_path, final, tracker_factory=FakeIoUTracker)


def test_config_rejects_wrong_json_types_and_normalises_numbers():
    import json as _json

    from trackrag.vision.config import config_from_dict

    base = _json.loads((PROJECT / "configs" / "vision" / "pretrained.json").read_text())
    for section, key, value in (("tracker", "fuse_score", "false"), ("detector", "antialias", "no"),
                                ("tracker", "new_track_thresh", 7.0), ("tracker", "track_buffer", 30.5)):
        data = _json.loads(_json.dumps(base))
        data[section][key] = value
        with pytest.raises(ValueError):
            config_from_dict(data)
    as_int, as_float = _json.loads(_json.dumps(base)), _json.loads(_json.dumps(base))
    as_int["detector"]["nms_iou"], as_float["detector"]["nms_iou"] = 1, 1.0
    assert config_from_dict(as_int).identity_hash() == config_from_dict(as_float).identity_hash()


def test_adapter_checks_report_task1_ready():
    from trackrag.__main__ import check_adapters

    for module in (None, "tools.vision_saved_adapters"):
        report = check_adapters("predicted", "none", module)
        task1 = [check for check in report["adapters"] if check["task"] == 1]
        assert task1 and task1[0]["status"] == "ready", report


def test_importing_inference_modules_loads_no_model_libraries():
    blocker = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'torch','rfdetr','ultralytics','supervision','cv2'}:\n"
        "            raise ImportError('blocked model library: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import trackrag.vision.config, trackrag.vision.pipeline, trackrag.vision.detector\n"
        "import trackrag.vision.tracker, trackrag.vision.__main__\n"
        "print('ok')\n"
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("TRACKRAG_")}
    env["PYTHONUTF8"] = "1"
    done = subprocess.run([sys.executable, "-c", blocker], cwd=PROJECT, env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr


# --- real ultralytics BOTSORT (model environment only) ---------------------------------------

def _real_tracker():
    pytest.importorskip("ultralytics")
    from trackrag.vision.tracker import BotSortTracker

    return BotSortTracker


def test_real_botsort_is_causal_and_fresh_per_recording(tmp_path):
    tracker = _real_tracker()
    size = (320, 180)
    script = {f: [((20 + 4 * f, 60, 80 + 4 * f, 110), 0.9, "Car"), ((200, 40, 260, 140), 0.7, "Bus")]
              for f in range(1, 13)}
    script[7] = []
    config = make_config(long_side=160)

    def go(root, rec, frames):
        recording = FakeRecording(root, rec, frames, size=size, textured=True)
        return predict_tracks(recording, config=config, detector=ScriptedDetector(script, size), tracker_factory=tracker)

    full = go(tmp_path / "a", "rec", range(1, 13))
    assert full and min(r["track_id"] for r in full) == 1
    for cut in (2, 6, 9):
        assert go(tmp_path / "a", "rec", range(1, cut + 1)) == [r for r in full if r["frame_id"] <= cut]
    other = go(tmp_path / "b", "rec_b", range(1, 13))
    assert [(r["frame_id"], r["track_id"], r["class_name"]) for r in other] == \
        [(r["frame_id"], r["track_id"], r["class_name"]) for r in full]
    car_boxes = {r["frame_id"]: r["bbox_xyxy"] for r in full if r["class_name"] == "Car"}
    assert car_boxes[3] == (32.0, 60.0, 92.0, 110.0)  # the detector's box, not a Kalman estimate


def test_real_botsort_handles_empty_and_missing_frames(tmp_path):
    tracker = _real_tracker()
    size = (320, 180)
    script = {f: [((20 + 4 * f, 60, 80 + 4 * f, 110), 0.9, "Car")] for f in (1, 2, 5, 6)}
    recording = FakeRecording(tmp_path, "rec", [1, 2, 3, 5, 6], size=size, textured=True)
    rows = predict_tracks(recording, config=make_config(long_side=160), detector=ScriptedDetector(script, size),
                          tracker_factory=tracker)
    assert sorted({r["frame_id"] for r in rows}) == [1, 2, 5, 6]
    assert len({r["track_id"] for r in rows}) == 1  # identity kept across the empty and missing frames
