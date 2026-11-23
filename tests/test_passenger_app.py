"""Browser service tests use the real dialogue with an explicitly substituted grounder."""

import hashlib
import json
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image
import pytest

from trackrag.dropoff.interaction import PassengerSession
from trackrag.dropoff.passenger import digest, proposed, unresolved
from trackrag.passenger_app import PassengerRequestError, PassengerSite, ThreadingHTTPServer, handler, load_scenes
from trackrag.speech import SpeechInputError


class StubGrounder:
    def __init__(self): self.calls = []

    def result(self, req, result):
        return {"ok": True, "request_id": req["request_id"], "result": result, "provenance": {
            "request_sha256": digest(req), "image": {"sha256": req["scene"]["provenance"]["image_sha256"]}}}

    def interpret(self, req):
        self.calls.append(req)
        if req["text"] == "here":
            return self.result(req, unresolved("Which landmark do you mean?"))
        return self.result(req, proposed([32, 30], [16, 20, 48, 40]))

    def passenger_area(self, req, box):
        return self.result(req, proposed([(box[0]+box[2])/2, (box[1]+box[3])/2], box))


@pytest.mark.skipif(not shutil.which("node"), reason="Browser control regression requires Node.js")
def test_passenger_photo_taps_at_different_screen_sizes_and_zooms():
    subprocess.run(["node", str(Path(__file__).with_name("passenger_photo_controls.mjs"))], check=True)


@pytest.mark.skipif(not shutil.which("node"), reason="Microphone regression requires Node.js")
def test_passenger_microphone_capture_and_fallback():
    subprocess.run(["node", str(Path(__file__).with_name("passenger_microphone.mjs"))], check=True)


@dataclass
class StubTranscript:
    transcript: str
    audio_sha256: str


class StubASR:
    def transcribe_wav(self, audio):
        if audio == b"silence":
            raise SpeechInputError("no_speech", "No words were recognized.")
        return StubTranscript("near the read bus", hashlib.sha256(audio).hexdigest())


@pytest.fixture
def site(tmp_path):
    bank = tmp_path / "bank.jsonl"
    records = []
    for index in [1, 2]:
        image = tmp_path / f"{index}.png"
        Image.new("RGB", (100, 60), (index*30, 90, 80)).save(image)
        scene = {"frame": {"recording_id": f"clip{index}", "frame_id": index, "timestamp_s": .1,
                           "width": 100, "height": 60, "image_path": image.name}, "tracks": [], "source": "predicted",
                 "provenance": {"image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}}
        records.append({"scene":scene, "text":"private example never supplied", "target":{"gold":"scoring only"}})
    bank.write_text("\n".join(json.dumps(r) for r in (records[0], *records)))
    return PassengerSite(load_scenes([bank]), StubGrounder(), StubASR(), tmp_path / "sessions", PassengerSession)


def test_preflight_counts_distinct_recordings_with_verified_footage(site):
    first, second = site.scenes
    assert site.check_footage_coverage(0) == []
    with pytest.raises(ValueError, match="0 distinct recording"):
        site.check_footage_coverage(1)
    site.scenes["another-bookmark"] = site.scenes[first]
    site.footage_scenes.update([first, "another-bookmark"])
    with pytest.raises(ValueError, match="1 distinct recording"):
        site.check_footage_coverage(2)
    site.footage_scenes.add(second)
    assert site.check_footage_coverage(2) == ["clip1", "clip2"]


def test_small_laptop_releases_worker_and_preserves_dialogue_on_each_turn(site):
    released = []
    site.release_after_request = lambda: released.append(True)
    first = site.open(next(iter(site.scenes)))
    second = site.change(first["session"], 0, "submit", {"text": "here", "mode": "typed"})
    assert second["view"]["state"] == "needs_clarification"
    with pytest.raises(RuntimeError):
        with site.model_request():
            raise RuntimeError("worker failed")
    assert released == [True, True]
    assert site.lookup(first["session"]).snapshot() == second


def test_catalog_only_supplies_frames_and_no_sample_text_or_reference(site):
    public = site.catalog()
    assert len(public["scenes"]) == 2
    assert "private example" not in json.dumps(public) and "gold" not in json.dumps(public)
    assert all("image_path" not in row for row in public["scenes"])
    keys = list(site.scenes)
    first = site.open(keys[0])
    assert site.image(first["session"])[0] == "image/png"
    opened = site.open(keys[1], first["session"])
    assert opened["session"] != first["session"]
    with pytest.raises(PassengerRequestError, match="no longer open"): site.lookup(first["session"])


def test_clarification_correction_and_exact_confirm_preserve_source(site):
    reply = site.open(next(iter(site.scenes)))
    sid = reply["session"]
    reply = site.change(sid, 0, "submit", {"text":"here", "mode":"typed"})
    assert reply["view"]["state"] == "needs_clarification"
    site.change(sid, 1, "begin-audio", {})
    speech = b"test-only-speech"
    reply = site.change(sid, 2, "audio", speech)
    assert reply["transcript"] == "near the read bus"
    assert reply["audio_sha256"] == hashlib.sha256(speech).hexdigest()
    assert len(site.model.calls) == 1
    reply = site.change(sid, 3, "submit", {"text":"near the red bus", "mode":"voice"})
    req = site.model.calls[-1]; view = reply["view"]
    assert req["text"] == "near the red bus" and req["original_transcript"] == "near the read bus"
    assert req["input_mode"] == "voice" and req["dialogue"][-1]["text"] == "Which landmark do you mean?"
    assert "gold" not in json.dumps(req) and "private example" not in json.dumps(req)
    directory = site.lookup(sid).directory
    assert (directory / "audio" / (hashlib.sha256(speech).hexdigest()+".wav")).read_bytes() == speech
    assert view["state"] == "proposed"
    assert view["selected_by"] == "model"
    with pytest.raises(PassengerRequestError): site.change(sid, 4, "confirm", {"proposal_id":"stale"})
    marked = site.change(sid, 4, "area", {"box":[10, 10, 30, 30]})["view"]
    assert marked["selected_by"] == "passenger" and marked["proposal_id"] != view["proposal_id"]
    with pytest.raises(PassengerRequestError): site.change(sid, 4, "confirm", {"proposal_id":marked["proposal_id"]})
    final = site.change(sid, 5, "confirm", {"proposal_id":marked["proposal_id"]})
    assert final["view"]["state"] == "confirmed"


def test_first_mark_preserves_reviewed_voice_and_never_runs_automatic_model(site, monkeypatch):
    sid = site.open(next(iter(site.scenes)))["session"]
    text = "Near the red bus on the right"
    typed = {"mode":"typed", "text":text, "mark_on_photo":True}
    spoken = {**typed, "mode":"voice"}
    with pytest.raises(PassengerRequestError):
        site.change(sid, 0, "submit", spoken)
    with pytest.raises(PassengerRequestError):
        site.change(sid, 0, "submit", {**typed, "mark_on_photo":"true"})
    with pytest.raises(PassengerRequestError, match="one way"):
        site.change(sid, 0, "submit", {**typed, "point_to_vehicles":True})
    site.change(sid, 0, "begin-audio", {})
    site.change(sid, 1, "audio", b"recorded words")
    with pytest.raises(PassengerRequestError):
        site.change(sid, 2, "submit", typed)
    assert not site.model.calls
    with monkeypatch.context() as patch:
        patch.setattr(site,"model_request",lambda: pytest.fail("Opening the mark entered the inference queue"))
        opened = site.change(sid, 2, "submit", spoken)
    assert opened["voice_stage"] is None and opened["view"]["selected_by"] == "passenger"
    assert opened["view"]["turns"][0]["text"] == text and opened["view"]["proposal_id"] is None
    assert not site.model.calls
    with pytest.raises(PassengerRequestError):
        site.change(sid, 2, "area", {"box":[12, 20, 22, 30]})
    marked = site.change(sid, 3, "area", {"box":[12, 20, 22, 30]})["view"]
    assert marked["state"] == "proposed" and marked["selected_by"] == "passenger"
    assert site.change(sid, 4, "confirm", {"proposal_id":marked["proposal_id"]})["view"]["state"] == "confirmed"
    journal, = site.lookup(sid).directory.glob("*.jsonl")
    submitted = next(row for row in map(json.loads,journal.read_text().splitlines()) if row["action"] == "submitted")
    assert submitted["detail"]["request"]["original_transcript"] == "near the read bus"
    assert submitted["detail"]["request"]["input_mode"] == "voice" and submitted["detail"]["passenger_requested_area"]
    assert not site.model.calls
    assert site.change(sid, 5, "submit", {"text":"New request", "mode":"typed"})["view"]["selected_by"] == "model"


def test_new_failed_audio_invalidates_confirmation_and_previous_transcript(site):
    sid = site.open(next(iter(site.scenes)))["session"]
    token = site.change(sid, 0, "submit", {"mode":"typed","text":"next to the truck"})["view"]["proposal_id"]
    site.change(sid, 1, "begin-audio", {})
    assert site.change(sid, 2, "audio", b"test-speech")["transcript"]
    assert site.change(sid, 3, "begin-audio", {})["transcript"] is None
    failed = site.change(sid, 4, "audio", b"silence")
    assert failed["voice_stage"] == "failed" and failed["speech_error"]["code"] == "no_speech"
    assert failed["view"]["proposal_id"] is None and failed["transcript"] is None
    with pytest.raises(PassengerRequestError): site.change(sid,5,"confirm",{"proposal_id": token})
    with pytest.raises(PassengerRequestError): site.change(sid,5,"submit",{"mode":"voice","text":"old words"})
    attempts = sorted((site.lookup(sid).directory / "audio").glob("transcription-*.json"))
    assert json.loads(attempts[-1].read_text())["error"]["code"] == "no_speech"
    assert site.change(sid,5,"cancel",{})["view"]["state"] == "cancelled"


def test_explicit_spoken_replacement_clears_old_context_preserves_receipt_and_retires_id(site):
    sid = site.open(next(iter(site.scenes)))["session"]
    first = site.change(sid, 0, "submit", {"mode":"typed", "text":"between the two cars"})["view"]
    site.change(sid, 1, "begin-audio", {})
    spoken = site.change(sid, 2, "audio", b"passenger recording")
    payload = {"mode":"voice", "text":"by the shop entrance", "replaces_request_id":"obsolete"}
    with pytest.raises(PassengerRequestError) as exc:
        site.change(sid, 3, "submit", payload)
    assert exc.value.status == 409 and len(site.model.calls) == 1
    payload["replaces_request_id"] = first["request_id"]
    updated = site.change(sid, 3, "submit", payload)["view"]
    request = site.model.calls[-1]
    assert request["dialogue"] == [] and request["original_transcript"] == spoken["transcript"]
    assert request["input_mode"] == "voice" and request["text"] == "by the shop entrance"
    journal = next(site.lookup(sid).directory.glob("*.jsonl"))
    detail = json.loads(journal.read_text().splitlines()[-1])["detail"]
    assert detail["replaces_request_id"] == first["request_id"]
    assert detail["speech"]["audio_sha256"] == spoken["audio_sha256"]
    with pytest.raises(PassengerRequestError):
        site.change(sid, 4, "confirm", {"proposal_id":first["proposal_id"]})
    assert site.change(sid, 4, "confirm", {"proposal_id":updated["proposal_id"]})["view"]["state"] == "confirmed"


def test_discarding_voice_preserves_pending_request_and_retires_audio(site):
    sid = site.open(next(iter(site.scenes)))["session"]
    proposal = site.change(sid, 0, "submit", {"text":"near the bus", "mode":"typed"})["view"]
    site.change(sid, 1, "begin-audio", {})
    reviewed = site.change(sid, 2, "audio", b"new recording")
    assert reviewed["voice_stage"] == "review" and reviewed["view"]["proposal_id"] is None
    restored = site.change(sid, 3, "discard-audio", {})
    assert restored["view"] == proposal and restored["audio_sha256"] is None and restored["transcript"] is None
    for revision, action, payload in ((1, "confirm", {"proposal_id":proposal["proposal_id"]}),
                                      (4, "discard-audio", {}), (4, "submit", {"mode":"voice","text":"old audio"})):
        with pytest.raises(PassengerRequestError) as exc:
            site.change(sid, revision, action, payload)
        assert exc.value.status == 409
    assert site.change(sid, 4, "confirm", {"proposal_id":proposal["proposal_id"]})["view"]["state"] == "confirmed"
    assert len(site.model.calls) == 1


def test_rejects_file_replacement_and_mismatched_hash(site, tmp_path):
    scene = next(iter(site.scenes.values()))
    first = site.open(next(iter(site.scenes)))
    Path(scene["frame"]["image_path"]).write_bytes(b"changed")
    with pytest.raises(PassengerRequestError, match="original image changed"): site.image(first["session"])
    scene["frame"]["image_path"] = str(tmp_path / "2.png")
    bank = tmp_path / "wrong.jsonl"; bank.write_text(json.dumps({"scene":scene}))
    with pytest.raises(ValueError, match="hash"): load_scenes([bank])


def test_reinstalled_scene_accepts_identical_photos_but_verifies_each_copy(site, tmp_path):
    first_bank = tmp_path / "bank.jsonl"
    request = json.loads(first_bank.read_bytes().splitlines()[0])
    installed = tmp_path / "second-install"
    installed.mkdir()
    image = installed / "1.png"
    image.write_bytes((tmp_path / "1.png").read_bytes())
    second_bank = installed / "bank.jsonl"
    second_bank.write_text(json.dumps(request))
    assert load_scenes([first_bank, second_bank]) == site.scenes
    image.write_bytes((tmp_path / "2.png").read_bytes())
    with pytest.raises(ValueError, match="hash"):
        load_scenes([first_bank, second_bank])
    image.write_bytes((tmp_path / "1.png").read_bytes())
    request["scene"]["frame"]["timestamp_s"] += 1
    second_bank.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="Conflicting"):
        load_scenes([first_bank, second_bank])


def test_navigation_and_abandoned_voice_draft_preserve_confirmed_receipt(site):
    first, second = site.scenes
    sid = site.open(first)["session"]
    state = site.change(sid, 0, "submit", {"text": "behind the bus", "mode": "typed"})
    token = state["view"]["proposal_id"]
    site.change(sid, 1, "confirm", {"proposal_id": token})
    for action, payload in (("confirm", {"proposal_id": token}), ("cancel", {})):
        with pytest.raises(PassengerRequestError) as error:
            site.change(sid, 2, action, payload)
        assert error.value.status == 409
    site.change(sid, 2, "begin-audio", {})
    assert site.change(sid, 3, "cancel", {})["view"]["state"] == "confirmed"
    journal = next(site.lookup(sid).directory.glob("*.jsonl"))
    next_session = site.open(second, sid)
    assert next_session["view"]["state"] == "idle" and next_session["view"]["turns"] == []
    assert next_session["voice_stage"] is None and next_session["transcript"] is None
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [event["action"] for event in events] == ["opened", "submitted", "confirmed"]
    assert events[-1]["detail"]["proposal_id"] == token


def test_failed_scene_open_preserves_existing_scene_and_proposal(site):
    first, second = site.scenes
    sid = site.open(first)["session"]
    state = site.change(sid, 0, "submit", {"text": "near the bus", "mode": "typed"})
    token = state["view"]["proposal_id"]
    Path(site.scenes[second]["frame"]["image_path"]).write_bytes(b"changed original")
    with pytest.raises(ValueError):
        site.open(second, sid)
    assert site.lookup(sid).scene_key == first
    assert site.change(sid, 1, "confirm", {"proposal_id": token})["view"]["state"] == "confirmed"


def test_http_allows_only_local_origin_and_known_scene_assets(site):
    with ThreadingHTTPServer(("127.0.0.1",0), handler(site)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(origin+"/", timeout=5) as res:
                assert b"Passenger drop-off" in res.read()
            for path in ["/app.js", "/capture.js", "/photo-tools.js", "/app.css", "/api/scenes"]:
                with urlopen(origin+path,timeout=5) as res: assert res.status == 200
            with pytest.raises(HTTPError) as exc:
                urlopen(Request(origin+"/api/scenes", headers={"Origin":"https://external.invalid"}),timeout=5)
            assert exc.value.code == 403
            with pytest.raises(HTTPError) as exc: urlopen(origin+"/files/../../bank.jsonl",timeout=5)
            assert exc.value.code == 404
        finally:
            server.shutdown(); thread.join(timeout=5)
