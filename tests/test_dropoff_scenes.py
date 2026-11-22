"""Scene adapter tests. CI-safe: a scripted detector and a simple IoU tracker stand in for RF-DETR and BoT-SORT,
and frames are small synthetic non-square images in the EMT folder layout (no label files anywhere). The stand-in
detector writes ``detector_kind: "stub"``, so these tests pass ``allow_stub=True``; real use refuses stub output."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from trackrag.dropoff import scenes
from trackrag.dropoff.scenes import SceneError
from trackrag.emt import open_recording
from trackrag.vision import predict_tracks
from trackrag.vision.__main__ import JsonlSink, PrefixRecording
from trackrag.vision.config import config_from_dict
from trackrag.vision.detections import Detection
from trackrag.vision.geometry import iou

PROJECT = Path(__file__).resolve().parents[1]
SIZE = (64, 36)
REC = "video_000001"
DETECTOR = {"label_space": "emt", "checkpoint": "model.pth", "checkpoint_sha256": "0" * 64, "inference_long_side": 32}


def make_config(**tracker):
    return config_from_dict({"name": "scene-test", "detector": DETECTOR, "tracker": tracker})


def load(*args, **kwargs):
    return scenes.load_scene(*args, allow_stub=True, **kwargs)


def validate(*args, **kwargs):
    return scenes.validate_scene(*args, allow_stub=True, **kwargs)


class ScriptedDetector:
    """Scripted detections in original pixels; the frame number is the red value of the image."""

    kind = "stub"

    def __init__(self, script, fail_at=None):
        self.script, self.calls, self.fail_at = script, [], fail_at

    def detect(self, image):
        frame_id = image.getpixel((0, 0))[0]
        self.calls.append(frame_id)
        if frame_id == self.fail_at:
            raise RuntimeError("detector failed")
        sx, sy = SIZE[0] / image.size[0], SIZE[1] / image.size[1]
        return [Detection((l / sx, t / sy, r / sx, b / sy), score, name)
                for (l, t, r, b), score, name in self.script.get(frame_id, [])]


class FakeIoUTracker:
    def __init__(self, settings):
        self.tracks, self.next_id = {}, 1

    def update(self, detections, image_bgr):
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


def script_for(frames, empty=()):
    return {f: [] if f in empty else [((2 + 2 * f, 10, 14 + 2 * f, 20), 0.9, "Car"), ((40, 4, 60, 30), 0.8, "Bus")]
            for f in frames}


def frame_folder(root, recording_id, nested=True):
    return root / "images" / "test" / ("images" if nested else "") / recording_id


def write_frame(root, recording_id, frame_id, size=SIZE, colour=None, nested=True):
    folder = frame_folder(root, recording_id, nested)
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour or (frame_id, 0, 0)).save(folder / f"{frame_id:05d}.png")


def make_root(tmp_path, frames, recording_id=REC, name="emt", nested=True):
    root = tmp_path / name
    for frame_id in frames:
        write_frame(root, recording_id, frame_id, nested=nested)
    return root


def predict(root, out, config, script, recording_id=REC, limit=None):
    recording = open_recording(root, recording_id)
    if limit:
        recording = PrefixRecording(recording, limit)
    out.mkdir(parents=True)
    predict_tracks(recording, config=config, detector=ScriptedDetector(script), tracker_factory=FakeIoUTracker,
                   sink=JsonlSink(out, recording_id))
    return out


def write_profile(path):
    path.write_text(json.dumps({"name": "scene-test", "detector": DETECTOR}), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.delenv("TRACKRAG_VISION_CONFIG", raising=False)
    monkeypatch.delenv("TRACKRAG_SAVED_PREDICTIONS", raising=False)
    scenes._artifact_cache.clear()
    scenes._manifest_cache.clear()


@pytest.fixture
def setup(tmp_path):
    frames = [1, 2, 3, 4, 5, 6]
    root = make_root(tmp_path, frames)
    config = make_config()
    preds = predict(root, tmp_path / "preds", config, script_for(frames, empty=(4,)))
    return root, config, preds


def rewrite_tracks(folder, recording_id, change):
    """Change track rows and update the provenance hash and count to match."""
    tracks = folder / f"{recording_id}.tracks.jsonl"
    rows = [json.loads(line) for line in tracks.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = change(rows)
    tracks.write_bytes("".join(json.dumps(r) + "\n" for r in rows).encode("utf-8"))
    provenance_path = folder / f"{recording_id}.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.update(tracks_sha256=hashlib.sha256(tracks.read_bytes()).hexdigest(), observations=len(rows))
    provenance_path.write_bytes(json.dumps(provenance).encode("utf-8"))


def resealed(scene, change):
    scene = json.loads(json.dumps(scene))
    change(scene)
    scene["scene_sha256"] = scenes.scene_sha256(scene)
    return scene


# --- the normal saved route --------------------------------------------------------------------------------------

def test_saved_scene_content_and_identity(setup):
    root, config, preds = setup
    scene = load(root, REC, 3, profile=config, predictions_dir=preds)
    model_input, provenance, context, receipt = (scene[k] for k in ("model_input", "provenance", "context", "receipt"))
    assert [model_input[k] for k in ("recording_id", "frame_id", "width", "height")] == [REC, 3, 64, 36]
    assert model_input["timestamp_s"] == pytest.approx(0.2) and type(model_input["timestamp_s"]) is float
    assert context["image_relpath"] == f"images/test/images/{REC}/00003.png"
    assert model_input["image_sha256"] == hashlib.sha256((root / context["image_relpath"]).read_bytes()).hexdigest()
    assert [o["frame_id"] for o in model_input["observations"]] == [3, 3]
    assert [o["class_name"] for o in model_input["observations"]] == ["Car", "Bus"]
    assert all(tuple(o) == scenes.OBSERVATION_KEYS and o["source"] == "predicted"
               and all(type(v) is float for v in o["bbox_xyxy"]) for o in model_input["observations"])
    assert model_input["observations"][0]["bbox_xyxy"] == pytest.approx([8.0, 10.0, 20.0, 20.0])
    assert model_input["history"] == []
    assert provenance["run_kind"] == "saved_prediction_run" and provenance["detector_kind"] == "stub"
    assert provenance["identity_hash"] == config.identity_hash()
    assert receipt["origin"] == "saved_predictions"
    assert context["annotation_access"] == "none" and context["frame_rate_source"] == "default"
    assert context["frame_rate"] == 10.0 and context["image_binding"] == "frame_id_and_size_only"
    assert context["run_binding"] == "not_pinned"
    assert scene["scene_sha256"] == scenes.scene_sha256(scene)
    validate(scene, root, profile=config, predictions_dir=preds)


def test_stub_outputs_are_refused_unless_allowed(setup):
    root, config, preds = setup
    with pytest.raises(SceneError, match="stub"):
        scenes.load_scene(root, REC, 3, profile=config, predictions_dir=preds)
    scene = load(root, REC, 3, profile=config, predictions_dir=preds)
    with pytest.raises(SceneError, match="stub"):
        scenes.validate_scene(scene)
    relabelled = resealed(scene, lambda s: s["provenance"].update(detector_kind="real"))
    validate(relabelled)  # internally consistent, but not what the run produced
    with pytest.raises(SceneError, match=r"provenance\.detector_kind"):
        validate(relabelled, root, profile=config, predictions_dir=preds)


def test_json_round_trip_validates_and_holds_no_machine_paths(setup):
    root, config, preds = setup
    scene = load(root, REC, 5, profile=config, predictions_dir=preds, history_frames=2)
    text = json.dumps(scene)
    assert "\\" not in text and "Users" not in text and not re.search(r"[A-Za-z]:/", text)
    assert str(root) not in text and str(preds) not in text
    again = json.loads(text)
    validate(again, root, profile=config, predictions_dir=preds)
    assert again["scene_sha256"] == scene["scene_sha256"]


def test_hash_depends_on_content_and_run_only(setup, tmp_path):
    root, config, preds = setup
    plain = load(root, REC, 3, profile=config, predictions_dir=preds)
    manifest = scenes.frame_hash_manifest(root, [REC])
    bound = load(root, REC, 3, profile=config, predictions_dir=preds, frame_manifest=manifest)
    renamed = config_from_dict({"name": "other-name", "detector": DETECTOR})
    named = load(root, REC, 3, profile=renamed, predictions_dir=preds)
    flat = make_root(tmp_path, [1, 2, 3, 4, 5, 6], name="flat", nested=False)  # the reader's other folder layout
    other_layout = load(flat, REC, 3, profile=config, predictions_dir=preds)
    assert plain["scene_sha256"] == bound["scene_sha256"] == named["scene_sha256"] == other_layout["scene_sha256"]
    assert other_layout["context"]["image_relpath"] == f"images/test/{REC}/00003.png"
    validate(plain, flat, profile=config, predictions_dir=preds)  # the flat layout validates too
    changed = json.loads(json.dumps(plain))
    changed["receipt"] = {"origin": "saved_predictions", "created_utc": "2000-01-01T00:00:00Z", "seconds": 99.0}
    changed["context"]["vision_profile"] = "renamed"
    assert scenes.scene_sha256(changed) == plain["scene_sha256"]
    changed["model_input"]["observations"][0]["bbox_xyxy"][0] = 9.0
    assert scenes.scene_sha256(changed) != plain["scene_sha256"]


def test_integer_coordinates_in_saved_rows_become_floats(setup):
    root, config, preds = setup
    rewrite_tracks(preds, REC, lambda rows: [dict(r, bbox_xyxy=[int(v) for v in r["bbox_xyxy"]]) for r in rows])
    scene = load(root, REC, 3, profile=config, predictions_dir=preds)
    assert all(type(v) is float for o in scene["model_input"]["observations"] for v in o["bbox_xyxy"])
    validate(scene)


def test_negative_zero_is_normalised_so_equal_boxes_hash_equally(setup):
    import math

    root, config, preds = setup

    def set_left(value):
        def change(rows):
            rows[0]["bbox_xyxy"] = [value, 10.0, 20.0, 20.0]  # first row: frame 1, track 1
            return rows
        return change

    rewrite_tracks(preds, REC, set_left(-0.0))
    negative = load(root, REC, 1, profile=config, predictions_dir=preds)["model_input"]
    rewrite_tracks(preds, REC, set_left(0.0))
    positive = load(root, REC, 1, profile=config, predictions_dir=preds)["model_input"]
    assert math.copysign(1.0, negative["observations"][0]["bbox_xyxy"][0]) == 1.0
    assert scenes._canonical(negative) == scenes._canonical(positive)


def test_empty_predicted_frame_is_an_empty_list_not_an_error(setup):
    root, config, preds = setup
    assert load(root, REC, 4, profile=config, predictions_dir=preds)["model_input"]["observations"] == []


def test_history_takes_the_last_predicted_frames_across_a_gap(tmp_path):
    frames = [1, 2, 4, 5]
    root = make_root(tmp_path, frames)
    config = make_config()
    preds = predict(root, tmp_path / "preds", config, script_for(frames))
    scene = load(root, REC, 5, profile=config, predictions_dir=preds, history_frames=2)
    assert sorted({o["frame_id"] for o in scene["model_input"]["history"]}) == [2, 4]
    assert all(o["frame_id"] == 5 for o in scene["model_input"]["observations"])
    validate(scene, root)


def test_frame_manifest_binds_the_image(setup):
    root, config, preds = setup
    manifest = scenes.frame_hash_manifest(root, [REC])
    scene = load(root, REC, 2, profile=config, predictions_dir=preds, frame_manifest=manifest)
    assert scene["context"]["frame_manifest_sha256"] == scenes.load_frame_manifest(manifest)[1]
    validate(scene, frame_manifest=manifest)  # no data root needed for this check
    wrong = json.loads(json.dumps(manifest))
    wrong["recordings"][REC]["2"] = "0" * 64
    with pytest.raises(SceneError, match="frame manifest"):
        validate(scene, frame_manifest=wrong)
    with pytest.raises(SceneError, match="frame manifest"):
        load(root, REC, 2, profile=config, predictions_dir=preds, frame_manifest=wrong)
    other = json.loads(json.dumps(manifest))
    other["recordings"]["video_000009"] = {"1": "1" * 64}
    with pytest.raises(SceneError, match="another frame manifest"):
        validate(scene, frame_manifest=other)
    with pytest.raises(SceneError, match="version 1"):
        scenes.load_frame_manifest({"recordings": {}})


def test_run_manifest_pins_the_prediction_files(setup):
    root, config, preds = setup
    unpinned = load(root, REC, 3, profile=config, predictions_dir=preds)
    pin = {"version": 1, "runs": {config.identity_hash(): {REC: {
        "tracks_sha256": unpinned["provenance"]["tracks_sha256"],
        "provenance_sha256": unpinned["provenance"]["provenance_sha256"]}}}}
    pinned = load(root, REC, 3, profile=config, predictions_dir=preds, run_manifest=pin)
    assert pinned["context"]["run_binding"] == "shared_run_manifest"
    validate(pinned, run_manifest=pin)
    rewrite_tracks(preds, REC, lambda rows: [dict(r, bbox_xyxy=[v + 0.5 for v in r["bbox_xyxy"]]) if r["frame_id"] == 3
                                             else r for r in rows])  # tampered, with hashes updated to match
    load(root, REC, 3, profile=config, predictions_dir=preds)  # consistent files load ...
    with pytest.raises(SceneError, match="differ from the shared run"):
        load(root, REC, 3, profile=config, predictions_dir=preds, run_manifest=pin)  # ... but are not the shared run


def test_metadata_frame_rate_is_reported_as_annotation_access(setup):
    root, config, preds = setup
    (root / "annotations").mkdir()
    (root / "annotations" / "metadata.txt").write_text("fps: 5\n", encoding="utf-8")
    scene = load(root, REC, 3, profile=config, predictions_dir=preds)
    assert scene["context"]["annotation_access"] == "annotations/metadata.txt (frame rate only)"
    assert scene["context"]["frame_rate_source"] == "annotations/metadata.txt" and scene["context"]["frame_rate"] == 5.0
    assert scene["model_input"]["timestamp_s"] == pytest.approx(0.4)


def test_environment_variables_select_profile_and_predictions(setup, tmp_path, monkeypatch):
    root, config, preds = setup
    monkeypatch.setenv("TRACKRAG_VISION_CONFIG", str(write_profile(tmp_path / "profile.json")))
    monkeypatch.setenv("TRACKRAG_SAVED_PREDICTIONS", str(preds))
    scene = load(root, REC, 3)
    assert scene["model_input"] == load(root, REC, 3, profile=config, predictions_dir=preds)["model_input"]


# --- explicit refusals (all SceneError) ---------------------------------------------------------------------------

def test_profile_must_be_explicit_and_readable(setup, tmp_path):
    root, _, preds = setup
    with pytest.raises(SceneError, match="explicitly"):
        load(root, REC, 3, predictions_dir=preds)
    bad = tmp_path / "bad_profile.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(SceneError, match="Cannot load vision profile"):
        load(root, REC, 3, profile=str(bad), predictions_dir=preds)


def test_predictions_folder_is_required(setup):
    root, config, _ = setup
    with pytest.raises(SceneError, match="prediction folder"):
        load(root, REC, 3, profile=config)


def test_other_profile_identity_is_refused(setup):
    root, _, preds = setup
    with pytest.raises(SceneError, match="identity"):
        load(root, REC, 3, profile=make_config(fuse_score=False), predictions_dir=preds)


@pytest.mark.parametrize("frame_id", [3.0, True, -1, "3"])
def test_frame_id_must_be_an_int(setup, frame_id):
    root, config, preds = setup
    with pytest.raises(SceneError, match="Frame ID"):
        load(root, REC, frame_id, profile=config, predictions_dir=preds)


@pytest.mark.parametrize("recording_id", ["../video_000001", "video 1", ""])
def test_recording_id_is_checked(setup, recording_id):
    root, config, preds = setup
    with pytest.raises(SceneError, match="recording ID"):
        load(root, recording_id, 3, profile=config, predictions_dir=preds)


def test_wrong_recording_artifact_is_refused(setup):
    root, config, preds = setup
    for frame_id in range(1, 7):
        write_frame(root, "video_000002", frame_id)
    for suffix in ("tracks.jsonl", "provenance.json"):
        shutil.copy(preds / f"{REC}.{suffix}", preds / f"video_000002.{suffix}")
    with pytest.raises(SceneError, match="another recording"):
        load(root, "video_000002", 3, profile=config, predictions_dir=preds)


def test_foreign_track_source_is_refused(setup):
    root, config, preds = setup
    rewrite_tracks(preds, REC, lambda rows: [dict(r, source="annotated") for r in rows])
    with pytest.raises(SceneError, match="foreign source"):
        load(root, REC, 3, profile=config, predictions_dir=preds)


def test_tampered_tracks_are_refused_by_hash_even_after_caching(setup):
    root, config, preds = setup
    load(root, REC, 3, profile=config, predictions_dir=preds)  # cache filled
    tracks = preds / f"{REC}.tracks.jsonl"
    stat = tracks.stat()
    tracks.write_bytes(tracks.read_bytes().replace(b"Bus", b"Car"))
    os.utime(tracks, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same size and modification time
    assert tracks.stat().st_size == stat.st_size
    with pytest.raises(SceneError, match="sha256"):
        load(root, REC, 3, profile=config, predictions_dir=preds)


def test_unreadable_prediction_files_are_refused(setup):
    root, config, preds = setup
    provenance_path = preds / f"{REC}.provenance.json"
    provenance_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(SceneError, match="JSON object"):
        load(root, REC, 3, profile=config, predictions_dir=preds)
    provenance_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SceneError, match="not valid JSON"):
        load(root, REC, 3, profile=config, predictions_dir=preds)
    (preds / f"{REC}.tracks.jsonl").unlink()
    with pytest.raises(SceneError, match="No saved predictions"):
        load(root, REC, 3, profile=config, predictions_dir=preds)


def test_extra_keys_in_saved_rows_never_reach_the_scene(setup):
    root, config, preds = setup
    rewrite_tracks(preds, REC, lambda rows: [dict(r, colour="red", score=0.9) for r in rows])
    scene = load(root, REC, 3, profile=config, predictions_dir=preds)
    assert all(tuple(o) == scenes.OBSERVATION_KEYS for o in scene["model_input"]["observations"])
    text = json.dumps(scene)
    assert '"colour"' not in text and '"score"' not in text


def test_missing_frame_in_images_is_explicit(setup):
    root, config, preds = setup
    with pytest.raises(SceneError, match="Frame 9 is absent"):
        load(root, REC, 9, profile=config, predictions_dir=preds)


def test_frame_in_images_but_not_predicted_is_explicit(tmp_path):
    frames = [1, 2, 3, 4]
    root = make_root(tmp_path, frames)
    config = make_config()
    preds = predict(root, tmp_path / "preds", config, script_for(frames), limit=2)
    load(root, REC, 2, profile=config, predictions_dir=preds)
    with pytest.raises(SceneError, match="No saved predictions"):
        load(root, REC, 3, profile=config, predictions_dir=preds)


def test_frame_predicted_but_missing_from_images_is_explicit(setup):
    root, config, preds = setup
    (frame_folder(root, REC) / "00003.png").unlink()
    with pytest.raises(SceneError, match="Frame 3 is absent"):
        load(root, REC, 3, profile=config, predictions_dir=preds)


def test_image_frames_must_equal_predicted_frames_up_to_t(setup):
    root, config, preds = setup
    write_frame(root, REC, 0)  # an extra earlier image the predictions never saw
    with pytest.raises(SceneError, match="differ from the predicted frames"):
        load(root, REC, 3, profile=config, predictions_dir=preds)


def test_image_size_must_equal_prediction_size(setup):
    root, config, preds = setup
    write_frame(root, REC, 3, size=(36, 64))
    with pytest.raises(SceneError, match="differs from the prediction size"):
        load(root, REC, 3, profile=config, predictions_dir=preds)


# --- validate_scene at a module boundary -------------------------------------------------------------------------

@pytest.mark.parametrize("change, message", [
    (lambda s: s["model_input"]["observations"][0].update(frame_id=4), "Observation from frame 4"),
    (lambda s: s["model_input"]["history"].append(dict(s["model_input"]["observations"][0], frame_id=3)),
     "not earlier"),
    (lambda s: s["model_input"]["history"].append(dict(s["model_input"]["observations"][0], frame_id=-7)),
     "nonnegative"),
    (lambda s: s["model_input"]["observations"][0].update(recording_id="video_000009"), "another recording"),
    (lambda s: s["model_input"]["observations"][0].update(source="annotated"), "another track source"),
    (lambda s: s["model_input"]["observations"][0].update(bbox_xyxy=[10.0, 10.0, 5.0, 20.0]), "outside"),
    (lambda s: s["model_input"]["observations"][0].update(bbox_xyxy=[0.0, 0.0, 65.0, 20.0]), "outside"),
    (lambda s: s["model_input"]["observations"][0].update(bbox_xyxy=[0, 0, 5, 20]), "float"),
    (lambda s: s["model_input"]["observations"][0].update(track_id=True), "ints"),
    (lambda s: s["model_input"]["observations"][0].update(class_name="truck"), "Unknown class"),
    (lambda s: s["model_input"]["observations"][0].update(colour="red"), "six TrackObservation"),
    (lambda s: s["model_input"]["observations"].append(dict(s["model_input"]["observations"][0])), "Duplicate"),
    (lambda s: s["model_input"].update(timestamp_s="noon"), "timestamp_s"),
    (lambda s: s["model_input"].update(timestamp_s=-5.0), "timestamp_s"),
    (lambda s: s["context"].update(image_relpath="C:/data/x.png"), "relative POSIX"),
    (lambda s: s["context"].update(image_relpath=f"images/test/images/{REC}/00006.png"), "does not name frame 3"),
    (lambda s: s["context"].update(image_relpath="images/test/images/video_000009/00003.png"), "does not name frame"),
    (lambda s: s["provenance"].update(track_source="annotated"), "predicted observations only"),
    (lambda s: s["provenance"].update(run_kind="guessed"), "Unknown run kind"),
    (lambda s: s["provenance"].update(detector_kind="magic"), "Unknown detector kind"),
    (lambda s: s["provenance"].update(identity_hash="junk"), "identity_hash"),
    (lambda s: s["provenance"].update(tracks_sha256=None), "tracks_sha256"),
    (lambda s: s["receipt"].update(origin="fresh_inference"), "fresh-scene run"),
    (lambda s: s["receipt"].update(origin="guessed"), "Unknown origin"),
    (lambda s: s["context"].update(config_source="C:/Users/someone/cfg.json"), "relative POSIX"),
    (lambda s: s["context"].update(annotation_access="whatever"), "annotation_access"),
    (lambda s: s["receipt"].update(seconds=-1.0), "Invalid receipt"),
    (lambda s: s["receipt"].update(seconds=float("inf")), "Invalid receipt"),
    (lambda s: s["context"].update(image_binding="frame_hash_manifest"), "disagree"),
    (lambda s: s["context"].update(run_manifest_sha256="1" * 64), "disagree"),
    (lambda s: s["context"].update(frame_rate=0.0), "frame_rate"),
    (lambda s: s.update(receipt={}), "receipt"),
])
def test_validate_scene_rejects_bad_scenes(setup, change, message):
    root, config, preds = setup
    scene = load(root, REC, 3, profile=config, predictions_dir=preds)
    with pytest.raises(SceneError, match=message):
        validate(resealed(scene, change))


def test_a_huge_integer_in_the_receipt_does_not_crash_the_check(setup):
    root, config, preds = setup
    scene = load(root, REC, 3, profile=config, predictions_dir=preds)
    scene["receipt"]["seconds"] = 10 ** 400  # the receipt is not hashed, and an int is always finite
    validate(scene)


def test_a_malformed_replay_record_is_refused(setup, tmp_path):
    root, config, preds = setup
    copy = tmp_path / "copy"
    shutil.copytree(preds, copy)
    provenance_path = copy / f"{REC}.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["replayed_from"] = "earlier run"
    provenance_path.write_bytes(json.dumps(provenance).encode("utf-8"))
    with pytest.raises(SceneError, match="replayed_from in the provenance file must be an object"):
        load(root, REC, 3, profile=config, predictions_dir=copy)


def test_non_finite_values_are_refused_before_hashing(setup):
    root, config, preds = setup
    scene = load(root, REC, 3, profile=config, predictions_dir=preds)
    scene["model_input"]["timestamp_s"] = float("nan")
    with pytest.raises(SceneError, match="timestamp_s"):
        validate(scene)
    with pytest.raises(SceneError, match="cannot be hashed"):
        scenes.scene_sha256(scene)


def test_validate_scene_rejects_a_stale_hash_and_a_changed_image(setup):
    root, config, preds = setup
    scene = load(root, REC, 3, profile=config, predictions_dir=preds)
    tampered = json.loads(json.dumps(scene))
    tampered["model_input"]["observations"].pop()
    with pytest.raises(SceneError, match="scene_sha256"):
        validate(tampered)
    write_frame(root, REC, 3, colour=(3, 1, 0))
    with pytest.raises(SceneError, match="Image bytes differ"):
        validate(scene, root)
    with pytest.raises(SceneError, match="image hash"):
        scenes.scene_image_path(scene, root)


def test_validate_scene_compares_field_by_field_with_the_predictions(setup):
    root, config, preds = setup
    scene = load(root, REC, 3, profile=config, predictions_dir=preds)
    moved = resealed(scene, lambda s: s["model_input"]["observations"][0].update(bbox_xyxy=[9.0, 10.0, 20.0, 20.0]))
    validate(moved, root)  # internally consistent ...
    with pytest.raises(SceneError, match=r"model_input\.observations"):
        validate(moved, root, profile=config, predictions_dir=preds)  # ... but not what the run produced
    with pytest.raises(SceneError, match="needs data_root"):
        validate(scene, profile=config, predictions_dir=preds)


def test_scenes_share_no_state(setup):
    root, config, preds = setup
    first = load(root, REC, 3, profile=config, predictions_dir=preds)
    first["model_input"]["observations"][0]["bbox_xyxy"][0] = 0.0
    second = load(root, REC, 3, profile=config, predictions_dir=preds)
    assert second["model_input"]["observations"][0]["bbox_xyxy"][0] == 8.0


# --- fresh route -------------------------------------------------------------------------------------------------

def fresh(root, config, out, detector=None, **kwargs):
    return scenes.fresh_scene(root, REC, 3, profile=config, out_dir=out, allow_stub=True,
                              detector=detector or ScriptedDetector(script_for(range(1, 7), empty=(4,))),
                              tracker_factory=FakeIoUTracker, **kwargs)


def test_fresh_scene_equals_saved_scene_and_reloads_as_saved(setup, tmp_path):
    root, config, preds = setup
    saved = load(root, REC, 3, profile=config, predictions_dir=preds)
    detector = ScriptedDetector(script_for(range(1, 7), empty=(4,)))
    made = fresh(root, config, tmp_path / "fresh", detector)
    assert made["model_input"] == saved["model_input"]
    assert made["receipt"]["origin"] == "fresh_inference" and made["provenance"]["run_kind"] == "fresh_scene_run"
    assert detector.calls == [1, 2, 3]  # online: nothing after frame 3 was processed
    reloaded = load(root, REC, 3, profile=config, predictions_dir=tmp_path / "fresh")
    assert reloaded["receipt"]["origin"] == "saved_predictions"  # reuse is labelled as saved, not fresh
    assert reloaded["provenance"]["run_kind"] == "fresh_scene_run" and reloaded["scene_sha256"] == made["scene_sha256"]
    validate(made, root, profile=config, predictions_dir=tmp_path / "fresh")
    with pytest.raises(SceneError, match="already exists"):
        fresh(root, config, tmp_path / "fresh")


def test_failed_fresh_run_leaves_nothing_behind(setup, tmp_path):
    root, config, _ = setup
    with pytest.raises(RuntimeError, match="detector failed"):
        fresh(root, config, tmp_path / "fresh", ScriptedDetector(script_for(range(1, 7)), fail_at=2))
    assert not (tmp_path / "fresh").exists() and not (tmp_path / "fresh.partial").exists()
    assert fresh(root, config, tmp_path / "fresh")["receipt"]["origin"] == "fresh_inference"


def test_fresh_run_checks_the_image_before_running_the_model(setup, tmp_path):
    root, config, _ = setup
    manifest = scenes.frame_hash_manifest(root, [REC])
    manifest["recordings"][REC]["3"] = "0" * 64
    detector = ScriptedDetector(script_for(range(1, 7)))
    with pytest.raises(SceneError, match="frame manifest"):
        fresh(root, config, tmp_path / "fresh", detector, frame_manifest=manifest)
    assert detector.calls == [] and not (tmp_path / "fresh").exists()


# --- export, paths and the command line --------------------------------------------------------------------------

def test_export_then_validate_the_folder(setup, tmp_path):
    root, config, preds = setup
    frames = [{"recording_id": REC, "frame_id": 2}, {"recording_id": REC, "frame_id": 5}]
    frames_file = tmp_path / "frames.json"
    frames_file.write_text(json.dumps(frames), encoding="utf-8")
    profile = write_profile(tmp_path / "profile.json")
    out = tmp_path / "scenes"
    common = ["--profile", str(profile), "--predictions", str(preds), "--allow-stub"]
    assert scenes.main(["export", "--data-root", str(root), "--frames", str(frames_file), "--out-dir", str(out),
                        *common]) == 0
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert [item["file"] for item in index] == [f"{REC}_f2.json", f"{REC}_f5.json"]
    assert scenes.main(["validate", "--data-root", str(root), *common, str(out)]) == 0
    assert scenes.main(["validate", "--allow-stub", str(out / "*.json")]) == 2  # the pattern also picks index.json
    assert scenes.main(["validate", "--allow-stub", str(out / "video_*.json")]) == 0  # wildcard expanded here
    (out / f"{REC}_f5.json").unlink()
    assert scenes.main(["validate", "--allow-stub", str(out)]) == 2  # folder no longer matches its index
    with pytest.raises(SceneError, match="repeats"):
        scenes.export_scenes(root, frames * 2, tmp_path / "again", profile=config, predictions_dir=preds,
                             allow_stub=True)
    with pytest.raises(SceneError, match="exactly"):
        scenes.export_scenes(root, [{"recording": REC, "frame": 2}], tmp_path / "bad", profile=config,
                             predictions_dir=preds, allow_stub=True)


def test_manifest_show_and_validate_commands(setup, tmp_path):
    root, _, preds = setup
    manifest_file = tmp_path / "frames.sha256.json"
    assert scenes.main(["manifest", "--data-root", str(root), "--recordings", REC, "--out", str(manifest_file)]) == 0
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert sorted(manifest["recordings"][REC], key=int) == ["1", "2", "3", "4", "5", "6"]
    out = tmp_path / "scene.json"
    profile = write_profile(tmp_path / "profile.json")
    common = ["--profile", str(profile), "--predictions", str(preds), "--allow-stub"]
    assert scenes.main(["show", "--data-root", str(root), "--recording", REC, "--frame", "3", "--frame-manifest",
                        str(manifest_file), "--out", str(out), *common]) == 0
    assert scenes.main(["validate", "--allow-stub", "--frame-manifest", str(manifest_file), str(out)]) == 0
    assert scenes.main(["show", "--data-root", str(root), "--recording", REC, "--frame", "9", *common]) == 2
    assert scenes.main(["show", "--data-root", str(root), "--recording", REC, "--frame", "3", "--out", str(out),
                        *common]) == 2  # refuses to overwrite before doing any work
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert scenes.main(["validate", "--allow-stub", "--frame-manifest", str(broken), str(out)]) == 2
    help_run = subprocess.run([sys.executable, "-m", "trackrag.dropoff.scenes", "--help"], cwd=PROJECT,
                              capture_output=True, text=True)
    assert help_run.returncode == 0 and "validate" in help_run.stdout


def test_scene_image_path_finds_the_frame_through_the_reader(setup):
    root, config, preds = setup
    scene = load(root, REC, 3, profile=config, predictions_dir=preds)
    assert scenes.scene_image_path(scene, root) == frame_folder(root, REC) / "00003.png"
    future = resealed(scene, lambda s: s["context"].update(image_relpath=f"images/test/images/{REC}/00006.png"))
    with pytest.raises(SceneError, match="does not name frame 3"):
        scenes.scene_image_path(future, root)
    with pytest.raises(SceneError, match="A scene has exactly"):
        scenes.scene_image_path({"model_input": {}}, root)


def test_loading_a_scene_needs_no_model_libraries(setup, tmp_path):
    root, _, preds = setup
    profile = write_profile(tmp_path / "profile.json")
    blocked = ("torch", "rfdetr", "ultralytics", "supervision")
    code = ("import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            f"        if name.split('.')[0] in {blocked!r}:\n"
            "            raise ImportError('blocked ' + name)\n"
            "sys.meta_path.insert(0, Block())\n"
            "from trackrag.dropoff.scenes import load_scene, validate_scene\n"
            "scene = load_scene(sys.argv[1], 'video_000001', 3, profile=sys.argv[2], predictions_dir=sys.argv[3],"
            " allow_stub=True)\n"
            "validate_scene(scene, sys.argv[1], allow_stub=True)\n")
    result = subprocess.run([sys.executable, "-c", code, str(root), str(profile), str(preds)], cwd=PROJECT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
