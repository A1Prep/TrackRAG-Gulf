"""Local application integration over generated EMT images, not real model quality."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from threading import Event, Thread
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import wave

from PIL import Image
import pytest

from trackrag.dropoff.app import AppConfig, LocalApplication, make_server
from trackrag.vision.config import config_from_dict


def idle(app):
    deadline = time.monotonic() + 8
    while any(job["status"] == "pending" for job in app.view()["jobs"]):
        if time.monotonic() > deadline:
            pytest.fail("application job did not finish")
        time.sleep(.01)
    return app.view()


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "images-only"
    for rid in ("recording_a", "recording_b"):
        folder = root / "images" / "train" / "images" / rid
        folder.mkdir(parents=True)
        for frame in (1, 4, 10):
            Image.new("RGB", (320, 180), (30 + frame, 90, 140)).save(folder / f"{frame:05d}.png")
    return root


@pytest.fixture
def app(corpus, tmp_path):
    application = LocalApplication(AppConfig(data_root=corpus, output_dir=tmp_path / "app",
                                            adapters="tools.dropoff_app_demo"))
    yield application
    application.close()


def select(app, rid="recording_a", frame=4):
    app.select({"recording_id": rid, "frame_id": frame})
    return idle(app)


def propose(app):
    selected = select(app)["selected"]
    app.interpret({"scene_token": selected["token"], "text": "Drop me off beside the white car."})
    first = idle(app)
    assert first["state"]["status"] == "needs_clarification"
    app.interpret({"scene_token": selected["token"], "request_id": first["state"]["request"]["request_id"],
                   "text": "The one on the right."}, "reply")
    result = idle(app)
    assert result["state"]["status"] == "proposed"
    return result


def wav_bytes():
    buffer = BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 1600)
    return buffer.getvalue()


def test_catalog_preserves_sparse_original_ids_without_annotations(app, corpus):
    assert not (corpus / "annotations").exists()
    assert [row["recording_id"] for row in app.view()["recordings"]] == ["recording_a", "recording_b"]
    assert app.frames("recording_a")["frame_ids"] == [1, 4, 10]
    selected = select(app)["selected"]
    assert selected["frame"]["frame_id"] == 4
    assert [row["frame_id"] for row in selected["tracks"]] == [4]
    assert selected["provenance"]["application_condition"]["development_adapters"]


def test_both_routes_run_on_two_recordings_through_the_same_application(app):
    for rid in ("recording_a", "recording_b"):
        current = select(app, rid)
        token = current["selected"]["token"]
        app.interpret({"scene_token": token, "text": "Beside the white car."})
        assert idle(app)["state"]["status"] == "needs_clarification"
        app.qa({"scene_token": token, "text": "Which car is visible on the right?"})
        qa = idle(app)["qa"][0]
        assert qa["status"] == "completed"
        assert qa["result"]["answer"]["answer_status"] == "answered"
        assert qa["question"]["recording_id"] == rid
        assert qa["result"]["answer"]["cited_evidence_ids"]
        assert all(image["frame_id"] <= 4 for evidence in qa["result"]["evidence"] for image in evidence["images"])
        assert all(evidence["interval"]["source"] == "predicted" for evidence in qa["result"]["evidence"])
    saved = json.loads((app.output_dir / "session.json").read_text(encoding="utf-8"))
    assert len(saved["qa_calls"]) == 2
    assert saved["condition"]["grounding"] == "development"


def test_proposal_confirmation_and_scene_change_use_the_displayed_binding(app):
    shown = propose(app)
    binding = {"scene_token": shown["selected"]["token"], "confirmation": shown["proposal_binding"]}
    app.confirm(binding)
    assert app.view()["state"]["status"] == "confirmed"
    select(app, "recording_b", 10)
    assert app.view()["state"] is None
    with pytest.raises(ValueError, match="scene changed"):
        app.confirm(binding)
    assert app.audit()["grounding"]["events"]


def test_bad_confirmation_cannot_mutate_a_valid_proposal(app):
    shown = propose(app)
    bad = deepcopy(shown["proposal_binding"])
    bad["scene"]["frame_id"] = 10
    with pytest.raises(ValueError):
        app.confirm({"scene_token": shown["selected"]["token"], "confirmation": bad})
    assert app.view()["state"]["status"] == "proposed"


def test_recording_or_frame_selector_reset_invalidates_even_before_new_scene_loads(app):
    propose(app)
    app.reset_selection({})
    assert app.view()["selected"] is None and app.view()["state"] is None


def test_missing_exact_frame_retains_scene_failure_instead_of_substituting_neighbor(app):
    propose(app)
    app.select({"recording_id": "recording_a", "frame_id": 2})
    result = idle(app)
    assert result["selected"] is None and result["state"] is None
    receipt = app.audit()["scene_calls"][-1]
    assert receipt["status"] == "error" and receipt["frame_id"] == 2


def test_source_image_is_snapshotted_and_qa_rejects_changes_since_selection(app, corpus):
    current = select(app)
    path = app.media(current["selected"]["image_url"].rsplit("/", 1)[1])
    original = path.read_bytes()
    source = corpus / "images/train/images/recording_a/00004.png"
    Image.new("RGB", (320, 180), "red").save(source)
    assert path.read_bytes() == original
    app.qa({"scene_token": current["selected"]["token"], "text": "Which car?"})
    assert idle(app)["qa"][0]["error"]["code"] == "qa_pipeline_error"


def test_changed_cached_scene_snapshot_is_not_reused_as_the_original_image(app):
    current = select(app)
    path = app.media(current["selected"]["image_url"].rsplit("/", 1)[1])
    path.write_bytes(b"changed saved image bytes")
    result = select(app)
    assert result["selected"] is None
    assert "snapshot changed" in app.audit()["scene_calls"][-1]["error"]["message"]


def test_audio_upload_correction_and_source_receipts_share_the_typed_interface(app):
    token = select(app)["selected"]["token"]
    payload = wav_bytes()
    app.audio(token, "fixture.wav", payload)
    recognized = idle(app)["audio"]
    assert recognized["text"] == "Drop me off beside the white car."
    assert recognized["provenance"]["genuine_asr"] is False
    app.interpret({"scene_token": token, "audio_id": recognized["audio_id"],
                   "text": "Drop me off beside the black car."})
    state = idle(app)["state"]
    assert state["request"]["input_mode"] == "voice"
    assert state["request"]["original_transcript"] == recognized["text"]
    assert state["request"]["text"] == "Drop me off beside the black car."
    assert state["transcript_corrections"][0]["corrected_text"] == state["request"]["text"]
    receipt = app.audit()["audio_calls"][0]
    assert (app.output_dir / receipt["audio_path"]).read_bytes() == payload
    assert receipt["sha256"] == sha256(payload).hexdigest()
    assert state["confirmation"] is None


def test_old_scene_audio_cannot_be_reused_on_a_new_scene(app):
    token = select(app)["selected"]["token"]
    app.audio(token, "fixture.wav", wav_bytes())
    audio_id = idle(app)["audio"]["audio_id"]
    token = select(app, "recording_b")["selected"]["token"]
    with pytest.raises(ValueError, match="another scene"):
        app.interpret({"scene_token": token, "audio_id": audio_id, "text": "Beside that car."})
    assert app.view()["state"] is None


@pytest.mark.parametrize("filename,payload", [("empty.wav", b""), ("wrong.txt", b"not audio")])
def test_rejected_audio_replacement_clears_previous_confirmation(app, filename, payload):
    shown = propose(app)
    token = shown["selected"]["token"]
    app.confirm({"scene_token": token, "confirmation": shown["proposal_binding"]})
    with pytest.raises(ValueError, match="supported nonempty"):
        app.audio(token, filename, payload)
    assert app.view()["state"] is None
    saved = json.loads((app.output_dir / "session.json").read_text(encoding="utf-8"))
    assert saved["grounding"]["state"] is None


def test_empty_transcript_is_an_explicit_error_without_grounding(app):
    token = select(app)["selected"]["token"]
    app._transcribe = lambda path: {"text": "", "provenance": {"fixture": True}}
    app.audio(token, "fixture.wav", wav_bytes())
    result = idle(app)
    assert result["audio"] is None and result["state"] is None
    assert app.audit()["audio_calls"][0]["error"]["code"] == "transcription_error"


def test_late_scene_preparation_cannot_replace_a_newer_selection(app, monkeypatch):
    entered, release = Event(), Event()
    original = app._adapters.predict_tracks_with_receipt

    def delayed(recording, **kwargs):
        if recording.recording_id == "recording_a":
            entered.set()
            if not release.wait(3):
                raise TimeoutError("fixture release missing")
        return original(recording, **kwargs)

    monkeypatch.setattr(app._adapters, "predict_tracks_with_receipt", delayed)
    app.select({"recording_id": "recording_a", "frame_id": 10})
    try:
        assert entered.wait(2)
        app.select({"recording_id": "recording_b", "frame_id": 4})
    finally:
        release.set()
    result = idle(app)
    assert result["selected"]["frame"]["recording_id"] == "recording_b"
    assert result["selected"]["frame"]["frame_id"] == 4
    assert app.audit()["scene_calls"][0]["status"] == "discarded"


def test_late_asr_cannot_overwrite_a_new_typed_instruction(app):
    token = select(app)["selected"]["token"]
    entered, release = Event(), Event()

    def delayed(path):
        entered.set()
        if not release.wait(3):
            raise TimeoutError("fixture release missing")
        return {"text": "Stale speech", "provenance": {"fixture": True}}

    app._transcribe = delayed
    app.audio(token, "fixture.wav", wav_bytes())
    try:
        assert entered.wait(2)
        app.interpret({"scene_token": token, "text": "Use this typed instruction."})
    finally:
        release.set()
    result = idle(app)
    assert result["audio"] is None
    assert result["state"]["request"]["text"] == "Use this typed instruction."
    assert app.audit()["audio_calls"][0]["status"] == "discarded"


def test_completed_asr_is_not_redelivered_after_a_new_typed_submission(app):
    token = select(app)["selected"]["token"]
    app.audio(token, "fixture.wav", wav_bytes())
    assert idle(app)["audio"] is not None
    app.interpret({"scene_token": token, "text": "This newer typed instruction wins."})
    result = idle(app)
    assert result["audio"] is None
    assert result["state"]["request"]["text"] == "This newer typed instruction wins."


def test_cancellation_is_available_while_the_application_model_worker_is_running(app):
    token = select(app)["selected"]["token"]
    entered, release = Event(), Event()
    original = app.controller._adapter.interpret

    def delayed(request):
        entered.set()
        if not release.wait(3):
            raise TimeoutError("fixture release missing")
        return original(request)

    app.controller._adapter.interpret = delayed
    app.interpret({"scene_token": token, "text": "Beside the car."})
    try:
        assert entered.wait(2)
        app.cancel({"scene_token": token})
        assert app.view()["state"]["status"] == "cancelled"
    finally:
        release.set()
    assert idle(app)["state"]["status"] == "cancelled"
    assert app.audit()["grounding"]["calls"][0]["disposition"] == "discarded"


def saved_config(corpus, tmp_path):
    data = {"name": "generated-test-fixture", "detector": {
        "label_space": "emt", "checkpoint": "fixture.pth", "checkpoint_sha256": "a" * 64}}
    config = config_from_dict(data)
    config_path = tmp_path / "vision.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for rid in ("recording_a", "recording_b"):
        rows = [{"recording_id": rid, "frame_id": frame, "track_id": 7, "class_name": "Car",
                 "source": "predicted", "bbox_xyxy": [100, 60, 160, 120]} for frame in (1, 4, 10)]
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (predictions / f"{rid}.tracks.jsonl").write_bytes(payload)
        provenance = {"recording_id": rid, "identity_hash": config.identity_hash(), "frame_ids": [1, 4, 10],
                      "frames": 3, "first_frame": 1, "last_frame": 10, "first_frame_size": [320, 180],
                      "observations": len(rows), "tracks_sha256": sha256(payload).hexdigest()}
        (predictions / f"{rid}.provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return AppConfig(data_root=corpus, output_dir=tmp_path / "normal-app", vision_config=config_path,
                     predictions=predictions, qa_answerer="none")


def test_normal_saved_route_uses_strict_existing_loader_without_annotations(corpus, tmp_path):
    normal = LocalApplication(saved_config(corpus, tmp_path))
    try:
        for rid in ("recording_a", "recording_b"):
            result = select(normal, rid)
            assert result["selected"]["provenance"]["kind"] == "validated_saved_vision_predictions"
            assert result["condition"]["development_adapters"] is None
            token = result["selected"]["token"]
            normal.qa({"scene_token": token, "text": "Which car?"})
            assert idle(normal)["qa"][0]["result"]["answer"]["answer_status"] == "not_requested"
        normal.interpret({"scene_token": token, "text": "Beside that car."})
        result = idle(normal)
        assert result["state"]["status"] == "error"
        assert result["last_outcome"]["error"]["code"] == "configuration_unavailable"
        normal.audio(token, "fixture.wav", wav_bytes())
        idle(normal)
        assert normal.audit()["audio_calls"][-1]["status"] == "error"
    finally:
        normal.close()


def test_saved_predictions_from_foreign_config_fail_visibly(corpus, tmp_path):
    config = saved_config(corpus, tmp_path)
    path = config.predictions / "recording_a.provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["identity_hash"] = "foreign"
    path.write_text(json.dumps(provenance), encoding="utf-8")
    normal = LocalApplication(config)
    try:
        assert select(normal)["selected"] is None
        assert normal.audit()["scene_calls"][-1]["status"] == "error"
    finally:
        normal.close()


def test_shipped_fake_speech_cannot_bypass_development_condition_label(corpus, tmp_path):
    config = replace(saved_config(corpus, tmp_path), speech_adapter="tools.dropoff_app_demo")
    with pytest.raises(ValueError, match="explicit --adapters"):
        LocalApplication(config)


@pytest.fixture
def http_app(app):
    server = make_server(app, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield app, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def http(url, body=None, headers=None):
    payload = json.dumps(body).encode() if body is not None else None
    request = Request(url, data=payload, headers=headers or ({"Content-Type": "application/json"} if payload else {}))
    with urlopen(request, timeout=5) as response:
        return response.status, response.headers, response.read()


def test_http_serves_the_app_and_registered_images_only(http_app):
    app, base = http_app
    assert b"Drop-off instruction" in http(base + "/")[2]
    assert b"refresh" in http(base + "/app.js")[2]
    assert b"registerProcessor" in http(base + "/capture.js")[2]
    http(base + "/api/select", {"recording_id": "recording_a", "frame_id": 4})
    current = idle(app)
    status, headers, payload = http(base + current["selected"]["image_url"])
    assert status == 200 and headers["Content-Type"].startswith("image/")
    assert Image.open(BytesIO(payload)).size == (320, 180)
    for path in ("/AGENTS.md", "/media/../../AGENTS.md", "/media/unknown"):
        with pytest.raises(HTTPError):
            http(base + path)


def test_http_rejects_cross_origin_and_reference_fields_without_model_calls(http_app):
    app, base = http_app
    token = select(app)["selected"]["token"]
    for headers, payload in [
        ({"Content-Type": "application/json", "Origin": "https://example.org"},
         {"scene_token": token, "text": "Beside the car."}),
        ({"Content-Type": "application/json"},
         {"scene_token": token, "text": "Beside the car.", "target": [70, 70]}),
    ]:
        with pytest.raises(HTTPError):
            http(base + "/api/instruction", payload, headers)
    assert app.controller.snapshot()["counts"]["requested_rows"] == 0


def test_http_keeps_provider_errors_distinct_and_exposes_downloadable_audit(http_app):
    app, base = http_app
    shown = propose(app)
    original = app._adapters.answer_question
    try:
        app._adapters.answer_question = lambda *args: {
            "answer_status": "error", "answer": "Fixture provider unavailable", "cited_evidence_ids": []}
        http(base + "/api/qa", {"scene_token": shown["selected"]["token"], "text": "Which car?"})
        result = idle(app)
        assert result["qa"][0]["result"]["answer"]["answer_status"] == "error"
        exported = json.loads(http(base + "/api/audit")[2])
        assert exported["qa_calls"][0]["result"]["answer"]["answer_status"] == "error"
    finally:
        app._adapters.answer_question = original
