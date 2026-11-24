"""Generated-image scene handoffs using real saved-artifact validation, no model."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from tools import dropoff_prepare_scene as prepare
from tools import vision_saved_adapters as saved
from tools.dropoff_predict import read_requests
from trackrag.emt import open_recording
from trackrag.vision.config import config_from_dict


def encoded(rows):
    return ("".join(json.dumps(row) + "\n" for row in rows)).encode("utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def scene_fixture(tmp_path):
    data = tmp_path / "generated-images"
    predictions = tmp_path / "saved-predictions"
    predictions.mkdir()
    config_data = {
        "name": "generated-fixture-not-a-trained-model",
        "detector": {"label_space": "emt", "checkpoint": "fixture.pth", "checkpoint_sha256": "a" * 64},
    }
    config_path = tmp_path / "vision.json"
    config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    config = config_from_dict(config_data)

    def artifacts(recording_id="rec_a", image_frames=(1, 2, 3, 4), predicted_frames=(1, 2, 3),
                  track_frames=(1, 2, 3)):
        folder = data / "images" / "train" / recording_id
        folder.mkdir(parents=True, exist_ok=True)
        colour_offset = 70 if recording_id == "rec_b" else 10
        for frame_id in image_frames:
            Image.new("RGB", (40, 30), (colour_offset + frame_id, 80, 140)).save(folder / f"{frame_id:05d}.png")
        rows = [
            {"recording_id": recording_id, "frame_id": frame_id, "track_id": 7,
             "source": "predicted", "class_name": "Car", "bbox_xyxy": [2, 3, 20, 22]}
            for frame_id in track_frames
        ]
        tracks_path = predictions / f"{recording_id}.tracks.jsonl"
        provenance_path = predictions / f"{recording_id}.provenance.json"
        provenance = {
            "recording_id": recording_id, "identity_hash": config.identity_hash(),
            "frame_ids": list(predicted_frames), "frames": len(predicted_frames),
            "first_frame": predicted_frames[0], "last_frame": predicted_frames[-1],
            "first_frame_size": [40, 30], "observations": len(rows),
        }

        def write():
            tracks_path.write_bytes(encoded(rows))
            provenance["observations"] = len(rows)
            provenance["tracks_sha256"] = hashlib.sha256(tracks_path.read_bytes()).hexdigest()
            provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

        write()
        return SimpleNamespace(rows=rows, provenance=provenance, tracks=tracks_path,
                               provenance_path=provenance_path, images=folder, write=write)

    def run(rows, output=None, **overrides):
        arguments = dict(data_root=data, predictions_dir=predictions, vision_config=config_path,
                         output_dir=output or tmp_path / "prepared", frame_rate=20.0)
        arguments.update(overrides)
        return prepare.prepare_scene_requests(encoded(rows), **arguments)

    return SimpleNamespace(root=tmp_path, data=data, predictions=predictions,
                           config_path=config_path, config=config, config_data=config_data,
                           artifacts=artifacts, run=run)


def instruction(identity="a", recording_id="rec_a", frame_id=2, **extra):
    return {"request_id": identity, "recording_id": recording_id, "frame_id": frame_id,
            "text": "Please identify the destination next to that car", **extra}


def test_paused_local_footage_asks_which_truck_only_when_the_current_original_has_two(scene_fixture):
    from trackrag.footage_app import FootageReview

    f = scene_fixture
    source = f.artifacts()
    for row in source.rows:
        row["class_name"] = "Medium_vehicle"
    second = {**source.rows[1], "track_id":9, "class_name":"Large_vehicle", "bbox_xyxy":[22,7,30,19]}
    source.rows.append(second)
    source.rows.sort(key=lambda row:(row["frame_id"], row["track_id"]))
    source.write()
    f.run([instruction("first", frame_id=1), instruction("second", frame_id=2)])
    one, two = (entry["scene"] for entry in read_requests(f.root/"prepared"/"requests.jsonl"))
    review = FootageReview(f.data, f.predictions, f.config_path, f.root/"review", frame_rate=20)
    words = "is the truck near the building"
    clarified = review.ask(two, words)
    assert clarified["answer_source"] == "local_tracks"
    assert clarified["answer"]["answer_status"] == "insufficient_evidence"
    assert "on the left and near the centre" in clarified["answer"]["answer"]
    assert not clarified["answer"]["cited_evidence_ids"] and not clarified["hosted_requested"]
    assert any(r["interval"]["track_id"] is None and [im["frame_id"] for im in r["images"]] == [2]
               for r in clarified["evidence"])
    assert not (f.root/"review"/clarified["question"]["question_id"]/"hosted").exists()
    for scene, question in [(one, words), (two, "Is the white truck on the left near the building?")]:
        result = review.ask(scene, question)
        assert result["answer_source"] == "local_tracks" and result["answer"]["answer_status"] == "insufficient_evidence"
        assert "do not measure the physical distance" in result["answer"]["answer"]
    distance = review.ask(one, "How far is the truck from the building?")
    assert not distance["evidence"] and "do not measure the physical distance" in distance["answer"]["answer"]


def test_footage_vehicle_choice_binds_question_source_and_frame_before_visual_review(scene_fixture, monkeypatch):
    from trackrag import answering
    from trackrag.footage_app import FootageReview

    f = scene_fixture
    for recording in ("rec_a", "rec_b"):
        source = f.artifacts(recording_id=recording)
        second = [{**row, "track_id":9, "bbox_xyxy":[27,7,35,19]} for row in source.rows]
        source.rows.extend(second)
        source.rows.sort(key=lambda row: (row["frame_id"], row["track_id"]))
        source.write()
    f.run([instruction("one", frame_id=1), instruction("two", frame_id=2), instruction("other", recording_id="rec_b")])
    first, current, other = (entry["scene"] for entry in read_requests(f.root/"prepared"/"requests.jsonl"))
    review = FootageReview(f.data, f.predictions, f.config_path, f.root/"review", frame_rate=20)

    calls = []
    def capture(question, evidence, folder):
        calls.append((question, evidence, json.loads((folder/"run.json").read_bytes())["configuration"]))
        return {"answer_status":"insufficient_evidence","answer":"The relationship is unclear.","cited_evidence_ids":[]}
    monkeypatch.setattr(answering, "answer_question", capture)
    words = "Is the car near the building?"
    initial = review.ask(current, words, hosted=True)
    assert initial["answer_source"] == "local_tracks" and calls == []
    choices = initial["vehicle_choices"]
    assert [(o["track_id"], o["position"]) for o in choices["options"]] == [(7, "left"), (9, "right")]
    selection = {"track_id":9, "binding":choices["binding"]}
    for scene, text, supplied in [(first, words, selection), (other, words, selection),
                                  (current, "Is the car near the shop?", selection),
                                  (current, words, {**selection, "track_id":777}),
                                  (current, words, {**selection, "bbox_xyxy":[0,0,1,1]})]:
        with pytest.raises(ValueError, match="choice"):
            review.ask(scene, text, hosted=True, selection=supplied)
    local = review.ask(current, words, selection=selection)
    assert local["answer_source"] == "local_tracks" and local["answer"]["answer_status"] == "insufficient_evidence"
    assert local["selected_vehicle"] == choices["options"][1] and not local["vehicle_choices"] and not calls
    distance = review.ask(current, words, hosted=True, selection=selection)
    assert distance["answer_source"] == "local_tracks" and not calls
    assert "Track 9 is selected in frame 2" in distance["answer"]["answer"]
    for other_words in ("Is the car close to the building?", "How far is the car from the building?"):
        clarification = review.ask(current,other_words,hosted=True)
        assert [o["track_id"] for o in clarification["vehicle_choices"]["options"]] == [7,9]
        assert "Which car" in clarification["answer"]["answer"]
        with pytest.raises(ValueError,match="Choose a current vehicle"):
            review.ask(current,other_words,hosted=True,image_relation=True)
        choice = {"binding":clarification["vehicle_choices"]["binding"],"track_id":7}
        chosen = review.ask(current,other_words,hosted=True,selection=choice)
        assert not calls and {e["interval"]["track_id"] for e in chosen["evidence"]} == {7,None}
    timed = review.ask(current,"At frame 1, was the car close to the building?",hosted=True)
    assert timed["answer_source"] == "local_tracks" and not timed["vehicle_choices"] and not timed["image_relation_available"] and not calls
    response = review.ask(current, words, hosted=True, selection=selection, image_relation=True)
    q, evidence, saved = calls.pop()
    assert response["selection"] == selection and response["answer_source"] == "gemini"
    assert saved["original_question"] == response["original_question"] == words
    assert saved["selected_vehicle"] == choices["options"][1]
    assert "image-plane positions of the selected vehicle (track 9) and the building" in q["question"]
    assert saved["image_plane_review"] and response["image_plane_review"]
    assert {(i["interval"]["track_id"], im["frame_id"]) for i in evidence for im in i["images"]} == {(9,2),(None,2)}
    assert list(next(i["images"][0]["bbox_xyxy"] for i in evidence if i["interval"]["track_id"] == 9)) == [27,7,35,19]


def test_footage_search_reads_prefix_dimensions_and_decodes_only_selected_original_pixels(scene_fixture, monkeypatch):
    from trackrag import emt
    from trackrag.footage_app import FootageReview

    f = scene_fixture
    source = f.artifacts()
    f.run([instruction()])
    scene = read_requests(f.root/"prepared"/"requests.jsonl")[0]["scene"]
    review = FootageReview(f.data, f.predictions, f.config_path, f.root/"review", frame_rate=20)
    monkeypatch.setattr(emt.cv2, "imread", lambda *_: pytest.fail("Footage metadata decoded an entire frame."))
    original_convert = Image.Image.convert
    converted = set()
    def capture(image, *args, **kwargs):
        filename = getattr(image, "filename", None)
        if filename and Path(filename).parent == source.images:
            converted.add(Path(filename).name)
        return original_convert(image, *args, **kwargs)
    monkeypatch.setattr(Image.Image, "convert", capture)
    answer = review.ask(scene, "How many cars are detected in this frame?")
    assert answer["answer"]["answer_status"] == "answered" and converted == {"00002.png"}
    original = next(e for e in answer["evidence"] if e["interval"]["track_id"] is None)
    assert [image["frame_id"] for image in original["images"]] == [2]
    archived = f.root/"review"/answer["question"]["question_id"]/"evidence"/"scene"/"2.png"
    with Image.open(archived) as image:
        assert image.getpixel((0,0)) == (12,80,140)
    # Earlier originals still have to match the saved prediction dimensions.
    Image.new("RGB", (41,30)).save(source.images/"00001.png")
    with pytest.raises(ValueError, match="frame size differs"):
        review.ask(scene, "How many cars are detected in this frame?")


def test_hosted_failure_offers_separate_saved_track_recovery_only_for_an_established_fact(scene_fixture, monkeypatch):
    from trackrag import answering
    from trackrag.footage_app import FootageReview

    f = scene_fixture
    f.artifacts()
    f.run([instruction()])
    scene = read_requests(f.root/"prepared"/"requests.jsonl")[0]["scene"]
    review = FootageReview(f.data, f.predictions, f.config_path, f.root/"review", frame_rate=20)
    failure = {"answer_status":"error","answer":"Provider returned HTTP 503","cited_evidence_ids":[]}
    monkeypatch.setattr(answering, "answer_question", lambda *_: failure)
    words = "How many cars are detected in this frame?"
    failed = review.ask(scene, words, hosted=True)
    assert failed["can_answer_from_tracks"] and failed["answer"] == failure and failed["answer_source"] == "gemini"
    recovered = review.ask(scene, words)
    assert recovered["answer_source"] == "local_tracks" and recovered["answer"]["answer_status"] == "answered"
    assert recovered["answer"]["cited_evidence_ids"] and not recovered["can_answer_from_tracks"]
    assert recovered["question"]["question_id"] != failed["question"]["question_id"]
    prior = f.root/"review"/failed["question"]["question_id"]/"results.jsonl"
    assert json.loads(prior.read_bytes())["answer"] == failure
    for visual in ("What colour is the car?", "Which side of the frame is track 555 on?"):
        result = review.ask(scene, visual, hosted=True)
        assert result["answer"] == failure and not result["can_answer_from_tracks"]


def test_footage_player_uses_original_sparse_clock_and_validated_coverage(scene_fixture):
    from trackrag.footage_app import FootageReview

    f = scene_fixture
    source = f.artifacts(image_frames=(3, 6, 10, 20), predicted_frames=(3, 6, 10), track_frames=(3, 6, 10))
    f.run([instruction(frame_id=6)])
    scene = read_requests(f.root/"prepared"/"requests.jsonl")[0]["scene"]
    review = FootageReview(f.data, f.predictions, f.config_path, f.root/"results", frame_rate=20.0)
    assert review.timeline(scene)["frames"] == [{"id":3,"time":0.0},{"id":6,"time":0.15},{"id":10,"time":0.35}]
    mime, data, etag = review.preview_frame(scene, 6)
    assert mime == "image/png" and data == (source.images/"00006.png").read_bytes()
    assert review.preview_frame(scene, 6, etag) == (mime, None, etag)
    for unavailable in (4, 20):
        with pytest.raises(ValueError):
            review.preview_frame(scene, unavailable)
    # A cached manifest must not be served after its provenance changes.
    source.provenance["observations"] += 1
    source.provenance_path.write_text(json.dumps(source.provenance) + "\n")
    with pytest.raises(ValueError, match="provenance"):
        review.timeline(scene)


def test_footage_preview_http_returns_only_attached_originals_and_revalidates_cache(scene_fixture):
    import threading
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    from trackrag.footage_app import FootageReview
    from trackrag.passenger_app import PassengerSite, ThreadingHTTPServer, handler, load_scenes

    f = scene_fixture
    source = f.artifacts()
    f.run([instruction()])
    scenes = load_scenes([f.root/"prepared"/"requests.jsonl"])
    site = PassengerSite(scenes, None, None, f.root/"app", None,
                         FootageReview(f.data, f.predictions, f.config_path, f.root/"out", frame_rate=20))
    with ThreadingHTTPServer(("127.0.0.1",0), handler(site)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/api/footage-preview/"
        prefix = url + next(iter(scenes)) + "/"
        try:
            with urlopen(prefix+"timeline") as src:
                assert [f["id"] for f in json.load(src)["frames"]] == [1,2,3]
            with urlopen(prefix+"2") as src:
                assert src.read() == (source.images/"00002.png").read_bytes()
                etag = src.headers["ETag"]
            with pytest.raises(HTTPError) as same:
                urlopen(Request(prefix+"2",headers={"If-None-Match":etag}))
            assert same.value.code == 304
            for invalid in (prefix+"4", url+"a"*24+"/timeline"):
                with pytest.raises(HTTPError) as missing:
                    urlopen(invalid)
                assert missing.value.code == 404
        finally:
            server.shutdown()
            thread.join()


def test_footage_player_never_advances_past_a_pause_or_replays_an_old_recording():
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("Browser playback timing regression needs Node.js")
    subprocess.run([node,str(Path(__file__).with_name("footage_player_controls.mjs"))],check=True)


def test_browser_footage_search_opens_only_the_saved_prefix_and_selected_sources(scene_fixture):
    from trackrag.footage_app import FootageReview
    f = scene_fixture
    f.artifacts()
    f.run([instruction()])
    scene = read_requests(f.root/"prepared"/"requests.jsonl")[0]["scene"]
    review = FootageReview(f.data,f.predictions,f.config_path,f.root/"footage",frame_rate=20.)
    actual = review.ask(scene,"Which car is visible?")
    assert actual["answer"]["answer_status"] == "not_requested"
    assert actual["vision"] == "validated_saved_predictions" and actual["evidence"]
    assert all(im["frame_id"] <= 2 for item in actual["evidence"] for im in item["images"])
    link = actual["evidence"][0]["images"][0]["url"].split("/")
    mime,payload = review.evidence_image(link[3],int(link[4]),int(link[5]))
    assert mime == "image/png" and payload.startswith(b"\x89PNG")
    with pytest.raises(ValueError):
        review.evidence_image(link[3],0,31)
    for key in ("vision_identity_hash","tracks_sha256","prediction_provenance_sha256","image_sha256"):
        wrong = deepcopy(scene)
        wrong["provenance"][key] = "0"*64
        with pytest.raises(ValueError):
            review.ask(wrong,"Which car is visible?")
    wrong = deepcopy(scene)
    wrong["frame"]["timestamp_s"] += 1
    with pytest.raises(ValueError):
        review.ask(wrong,"Which car is visible?")

    changed = deepcopy(scene)
    changed["tracks"][0]["class_name"] = "different road user"
    assert not review.covers(changed)
    with pytest.raises(ValueError, match="observations differ"):
        review.ask(changed, "Which car is visible?")
    changed["tracks"] = []
    assert not review.covers(changed)


def test_open_unbookmarked_frame_uses_the_same_provenance_and_current_predictions(scene_fixture):
    from trackrag.footage_app import FootageReview
    from trackrag.passenger_app import PassengerSite, load_scenes
    f = scene_fixture
    artifact = f.artifacts()
    f.run([instruction()])
    scenes = load_scenes([f.root/"prepared"/"requests.jsonl"])
    site = PassengerSite(scenes,None,None,f.root/"app",None,
                         FootageReview(f.data,f.predictions,f.config_path,f.root/"footage",frame_rate=20.))
    origin = next(iter(scenes))
    opened = site.open_recording_frame(origin,3)
    assert opened["frame_id"] == 3 and opened["timestamp_s"] == 0.1
    assert site.open_recording_frame(origin,3) == opened
    actual = site.scenes[opened["key"]]
    assert {row["frame_id"] for row in actual["tracks"]} == {3}
    assert actual["provenance"]["image_sha256"] != site.scenes[origin]["provenance"]["image_sha256"]
    assert actual["provenance"]["tracks_sha256"] == site.scenes[origin]["provenance"]["tracks_sha256"]
    with pytest.raises((KeyError,ValueError)):
        site.open_recording_frame(origin,4)
    artifact.rows[0]["bbox_xyxy"]=[1,3,20,22]
    artifact.write()
    with pytest.raises(ValueError,match="different saved predictions"):
        site.open_recording_frame(origin,1)
    assert len(site.catalog()["scenes"])==2


def test_photo_bookmark_does_not_become_a_searchable_recording(scene_fixture):
    from trackrag.footage_app import FootageReview
    from trackrag.passenger_app import PassengerRequestError, PassengerSite, load_scenes, scene_key
    f = scene_fixture
    f.artifacts()
    f.run([instruction()])
    scenes = load_scenes([f.root/"prepared"/"requests.jsonl"])
    main_key, original = next(iter(scenes.items()))
    still = deepcopy(original)
    still["frame"].update(frame_id=781, timestamp_s=39.0)
    still["provenance"]["tracks_sha256"] = "1" * 64
    still_key = scene_key(still["frame"]["recording_id"], 781)
    scenes[still_key] = still
    site = PassengerSite(scenes, None, None, f.root/"app", None,
                         FootageReview(f.data, f.predictions, f.config_path, f.root/"footage", frame_rate=20.))
    assert len(site.catalog()["scenes"]) == 2
    assert site.catalog()["footage_scenes"] == [main_key]
    assert site.open_recording_frame(main_key, 781)["key"] == still_key
    with pytest.raises(PassengerRequestError, match="passenger photo"):
        site.open_recording_frame(main_key, 781, for_footage=True)
    # Opening a real recorded frame from the passenger's extra bookmark still uses the correct detector run.
    new = site.open_recording_frame(still_key, 3)
    assert new["key"] in site.catalog()["footage_scenes"]
    assert site.scenes[new["key"]]["provenance"]["tracks_sha256"] == original["provenance"]["tracks_sha256"]


def test_two_recordings_export_portable_annotation_free_inputs(scene_fixture):
    f = scene_fixture
    first, second = f.artifacts(), f.artifacts("rec_b")
    rows = [instruction(), instruction("b", "rec_b", 3)]
    snapshot = encoded(rows)
    out = f.root / "portable"
    summary = prepare.prepare_scene_requests(snapshot, data_root=f.data, predictions_dir=f.predictions,
                                            vision_config=f.config_path, output_dir=out, frame_rate=20.0)
    assert not (f.data / "annotations").exists()
    assert (summary["requested_rows"], summary["prepared_rows"], summary["preparation_errors"]) == (2, 2, 0)
    assert summary["status"] == "complete"
    assert summary["annotation_labels_used"] is False
    assert summary["fresh_detector_inference"] is False
    assert summary["language_model_called"] is False and summary["grounding_quality_measured"] is False
    assert (out / "instructions.snapshot.jsonl").read_bytes() == snapshot
    assert (out / "vision-config.snapshot.json").read_bytes() == f.config_path.read_bytes()
    raw = read_jsonl(out / "requests.jsonl")
    resolved = read_requests(out / "requests.jsonl")
    for row, loaded, artifact in zip(raw, resolved, (first, second)):
        frame = row["scene"]["frame"]
        assert not Path(frame["image_path"]).is_absolute()
        original = artifact.images / f"{frame['frame_id']:05d}.png"
        staged = Path(loaded["scene"]["frame"]["image_path"])
        assert staged.is_relative_to(out)
        assert staged.read_bytes() == original.read_bytes()
        assert row["scene"]["provenance"]["image_sha256"] == hashlib.sha256(staged.read_bytes()).hexdigest()
        assert (frame["width"], frame["height"]) == (40, 30)
        assert all(track["frame_id"] == frame["frame_id"] for track in row["scene"]["tracks"])
        assert all(track["source"] == "predicted" for track in row["scene"]["tracks"])
        assert not ({"target", "review", "supervised_fields", "destination_point_xy"} & row.keys())
    assert {item["recording_id"] for item in read_jsonl(out / "prediction-receipts.jsonl")} == {"rec_a", "rec_b"}
    for filename in ("requests.jsonl", "preparation-outcomes.jsonl", "prediction-receipts.jsonl"):
        key = filename.removesuffix(".jsonl").replace("-", "_") + "_sha256"
        assert summary[key] == hashlib.sha256((out / filename).read_bytes()).hexdigest()


def test_full_image_recording_can_use_an_earlier_saved_prediction_prefix(scene_fixture):
    f = scene_fixture
    f.artifacts(image_frames=(1, 2, 3, 4), predicted_frames=(1, 2, 3))
    summary = f.run([instruction(frame_id=2)])
    assert summary["prepared_rows"] == 1
    receipt = read_jsonl(f.root / "prepared" / "prediction-receipts.jsonl")[0]
    assert (receipt["prefix_first_frame"], receipt["prefix_last_frame"], receipt["prefix_frames"]) == (1, 2, 2)
    request = read_jsonl(f.root / "prepared" / "requests.jsonl")[0]
    assert [track["frame_id"] for track in request["scene"]["tracks"]] == [2]
    assert request["scene"]["frame"]["timestamp_s"] == pytest.approx(0.05)


def test_empty_current_frame_never_uses_past_or_future_tracks(scene_fixture):
    f = scene_fixture
    f.artifacts(track_frames=(1, 3))
    summary = f.run([instruction()])
    assert summary["prepared_rows"] == 1
    request = read_jsonl(f.root / "prepared" / "requests.jsonl")[0]
    assert request["scene"]["tracks"] == []
    assert read_jsonl(f.root / "prepared" / "preparation-outcomes.jsonl")[0]["current_frame_observations"] == 0


def test_identifiable_bad_rows_remain_in_cohort_with_no_semantic_result(scene_fixture):
    f = scene_fixture
    f.artifacts()
    stale = [{"role": "user", "text": "The earlier car", "recording_id": "rec_a", "frame_id": 1}]
    rows = [instruction("valid"), instruction("absent", frame_id=9),
            instruction("stale", dialogue=stale), instruction("gold", target={"destination_point_xy": [10, 10]}),
            instruction("foreign", recording_id="absent-recording")]
    summary = f.run(rows)
    assert summary["requested_rows"] == 5
    assert (summary["prepared_rows"], summary["preparation_errors"]) == (1, 4)
    assert summary["status"] == "completed_with_errors"
    outcomes = read_jsonl(f.root / "prepared" / "preparation-outcomes.jsonl")
    assert [item["request_id"] for item in outcomes] == [item["request_id"] for item in rows]
    assert outcomes[0]["ok"] is True
    for item in outcomes[1:]:
        assert item["ok"] is False and item["error"]["code"] == "scene_preparation_failed"
        assert not ({"result", "status", "destination_point_xy", "destination_box_xyxy"} & item.keys())
    assert [item["request_id"] for item in read_jsonl(f.root / "prepared" / "requests.jsonl")] == ["valid"]


@pytest.mark.parametrize("mutation", [
    {"source": "annotated"}, {"recording_id": "other"}, {"bbox_xyxy": [0, 0, 41, 12]},
    {"class_name": "unsupported-class"}, {"track_id": -1},
])
def test_corrupt_future_saved_observation_is_rejected_before_frame_filtering(scene_fixture, mutation):
    f = scene_fixture
    artifact = f.artifacts()
    artifact.rows[-1].update(mutation)
    artifact.write()  # Correct hash: validation must inspect the corrupted row itself.
    result = f.run([instruction(frame_id=1)])
    assert result["prepared_rows"] == 0 and result["preparation_errors"] == 1
    assert not (f.root / "prepared" / "requests.jsonl").read_text()


def test_wrong_vision_configuration_cannot_relabel_saved_predictions(scene_fixture):
    f = scene_fixture
    f.artifacts()
    other = deepcopy(f.config_data)
    other["detector"]["threshold"] = 0.2
    f.config_path.write_text(json.dumps(other), encoding="utf-8")
    summary = f.run([instruction()])
    assert summary["prepared_rows"] == 0 and summary["preparation_errors"] == 1
    outcome = read_jsonl(f.root / "prepared" / "preparation-outcomes.jsonl")[0]
    assert "identity" in outcome["error"]["message"]


def test_tampered_track_file_cannot_supply_a_scene(scene_fixture):
    f = scene_fixture
    artifact = f.artifacts()
    artifact.tracks.write_bytes(artifact.tracks.read_bytes().replace(b'"Car"', b'"Bus"'))
    summary = f.run([instruction()])
    assert summary["prepared_rows"] == 0 and summary["preparation_errors"] == 1
    assert "sha256" in read_jsonl(f.root / "prepared" / "preparation-outcomes.jsonl")[0]["error"]["message"]


def test_saved_receipt_binds_consumed_bytes_even_if_files_change_after_loading(scene_fixture, monkeypatch):
    f = scene_fixture
    artifact = f.artifacts(image_frames=(1, 2, 3))
    old_provenance = artifact.provenance_path.read_bytes()
    old_tracks = artifact.tracks.read_bytes()
    original = saved.load_track_artifact

    def mutate_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        artifact.provenance_path.write_bytes(old_provenance + b"\n")
        artifact.tracks.write_bytes(b"changed after validated read\n")
        return result

    monkeypatch.setattr(saved, "load_track_artifact", mutate_after_read)
    rows, receipt = saved.predict_tracks_with_receipt(open_recording(f.data, "rec_a", frame_rate=20),
                                                     directory=f.predictions, config=f.config)
    assert len(rows) == 3
    assert receipt["tracks_sha256"] == hashlib.sha256(old_tracks).hexdigest()
    assert receipt["provenance_sha256"] == hashlib.sha256(old_provenance).hexdigest()
    assert receipt["tracks_sha256"] != hashlib.sha256(artifact.tracks.read_bytes()).hexdigest()
    assert receipt["provenance_sha256"] != hashlib.sha256(artifact.provenance_path.read_bytes()).hexdigest()
    assert receipt["fresh_detector_inference"] is False


def test_source_image_change_between_validation_and_copy_is_retained_as_error(scene_fixture, monkeypatch):
    f = scene_fixture
    artifact = f.artifacts()
    original = prepare.build_instruction_request

    def mutate_after_validation(*args, **kwargs):
        request = original(*args, **kwargs)
        Image.new("RGB", (40, 30), (240, 10, 20)).save(artifact.images / "00002.png")
        return request

    monkeypatch.setattr(prepare, "build_instruction_request", mutate_after_validation)
    summary = f.run([instruction()])
    assert summary["prepared_rows"] == 0 and summary["preparation_errors"] == 1
    outcome = read_jsonl(f.root / "prepared" / "preparation-outcomes.jsonl")[0]
    assert "changed after scene validation" in outcome["error"]["message"]
    assert not (f.root / "prepared" / "requests.jsonl").read_text()


def test_voice_text_correction_preserves_original_transcript(scene_fixture):
    f = scene_fixture
    f.artifacts()
    row = instruction(input_mode="voice", text="Drop me after the parked car",
                      original_transcript="Drop me at the parked car")
    assert f.run([row])["prepared_rows"] == 1
    request = read_requests(f.root / "prepared" / "requests.jsonl")[0]
    assert request["input_mode"] == "voice"
    assert request["text"] == row["text"]
    assert request["original_transcript"] == row["original_transcript"]


@pytest.mark.parametrize("payload", [
    b"", b"[]\n", b"not JSON\n", b'{"text":"missing id"}\n',
    b'{"request_id":"x","request_id":"y"}\n', b'{"request_id":"x","frame_id":NaN}\n',
    b'{"request_id":"duplicate"}\n{"request_id":"duplicate"}\n',
])
def test_ambiguous_or_invalid_bank_rejected_before_creating_output(scene_fixture, payload):
    f = scene_fixture
    out = f.root / "must-not-exist"
    with pytest.raises(ValueError):
        prepare.prepare_scene_requests(payload, data_root=f.data, predictions_dir=f.predictions,
                                       vision_config=f.config_path, output_dir=out)
    assert not out.exists()


@pytest.mark.parametrize("update", [{"frame_id": True}, {"frame_id": -1}, {"recording_id": "../outside"}])
def test_bad_frame_and_path_specifications_remain_row_errors(scene_fixture, update):
    f = scene_fixture
    f.artifacts()
    summary = f.run([instruction(**update)])
    assert (summary["requested_rows"], summary["prepared_rows"], summary["preparation_errors"]) == (1, 0, 1)


def test_existing_output_is_not_overwritten(scene_fixture):
    f = scene_fixture
    f.artifacts()
    out = f.root / "existing"
    out.mkdir()
    marker = out / "preserve.txt"
    marker.write_text("existing result", encoding="utf-8")
    with pytest.raises(FileExistsError):
        f.run([instruction()], output=out)
    assert marker.read_text() == "existing result"
    assert sorted(path.name for path in out.iterdir()) == ["preserve.txt"]


@pytest.mark.parametrize("rate", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_explicit_frame_rate_rejected_before_output(scene_fixture, rate):
    f = scene_fixture
    with pytest.raises(ValueError):
        f.run([instruction()], frame_rate=rate)
    assert not (f.root / "prepared").exists()


def test_single_request_cli_reports_success_and_retained_failure_exit_codes(scene_fixture, capsys):
    f = scene_fixture
    f.artifacts()
    common = ["--data-root", str(f.data), "--predictions", str(f.predictions),
              "--vision-config", str(f.config_path), "--recording", "rec_a",
              "--text", "Drop me next to that car", "--request-id", "cli-request", "--frame-rate", "20"]
    assert prepare.main(common + ["--frame", "2", "--out", str(f.root / "cli-good")]) == 0
    assert json.loads(capsys.readouterr().out)["prepared_rows"] == 1
    assert prepare.main(common + ["--frame", "99", "--out", str(f.root / "cli-bad")]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["requested_rows"] == 1 and summary["preparation_errors"] == 1


def test_model_and_annotation_loaders_are_not_called_for_input_preparation(scene_fixture, monkeypatch):
    f = scene_fixture
    f.artifacts()

    def forbidden(*args, **kwargs):
        raise AssertionError("scene preparation must not load models or human labels")

    monkeypatch.setattr("trackrag.emt.load_annotated_tracks", forbidden)
    monkeypatch.setattr("trackrag.dropoff.model.GroundingModel", forbidden)
    assert f.run([instruction()])["prepared_rows"] == 1
