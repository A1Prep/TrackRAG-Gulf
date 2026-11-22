"""HTTP state and permissions use synthetic photographs and a visibly labelled software grounder."""

import hashlib
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image
import pytest

from trackrag.dropoff.passenger import digest, proposed, unresolved
from trackrag.dropoff import passenger_pair_choice
from trackrag.dropoff.passenger_pair_choice import METHOD, PassengerSession
from trackrag.passenger_app import PassengerSite, ThreadingHTTPServer, handler, load_scenes


class SoftwareGrounder:
    def __init__(self):
        self.pair_calls = []

    @staticmethod
    def outcome(req, result, method):
        return {"request_id": req["request_id"], "ok": True, "result": result, "provenance": {
            "request_sha256": digest(req), "image": {"sha256": req["scene"]["provenance"]["image_sha256"]},
            "method": method, "software_fixture": True,
        }}

    def interpret(self, req):
        return self.outcome(req, unresolved("Which two vehicles?"), "software_test_question")

    def passenger_pair(self, req, track_ids, placement):
        self.pair_calls.append((track_ids, placement))
        boxes = [t["bbox_xyxy"] for t in req["scene"]["tracks"] if t["track_id"] in track_ids]
        return self.outcome(req, proposed([48, 65], [42, 60, 56, 72], boxes), METHOD)


@pytest.fixture
def http_site(tmp_path, monkeypatch):
    # This file checks the HTTP route's revision, voice and confirmation behavior.
    # The grounding tests revalidate true geometry separately. No synthetic road
    # observation in this fixture is represented as a real learned prediction.
    def check_fixture(req, output, ids, placement):
        assert output["provenance"]["software_fixture"] is True
        assert output["request_id"] == req["request_id"] and len(set(ids)) == 2
        assert placement in {"between","behind"}

    monkeypatch.setattr(passenger_pair_choice, "validate_choice_evidence", check_fixture)
    photo = tmp_path / "painted.png"
    Image.new("RGB", (100, 80), (41, 64, 80)).save(photo)
    scene = {
        "frame": {"recording_id": "fixture", "timestamp_s": 0.1, "frame_id": 2,
                  "width": 100, "height": 80, "image_path": str(photo)},
        "source": "predicted", "provenance": {"image_sha256": hashlib.sha256(photo.read_bytes()).hexdigest()},
        "tracks": [
            {"recording_id": "fixture", "frame_id": frame, "track_id": track, "class_name": kind,
             "bbox_xyxy": box, "source": "predicted"}
            for frame,track,kind,box in [(1,4,"Car",[5,12,10,23]), (2,6,"Car",[14,30,34,58]),
                                         (2,11,"Large_vehicle",[64,32,91,62]), (2,12,"Pedestrian",[1,2,7,11])]
        ],
    }
    bank = tmp_path / "scene.jsonl"
    bank.write_text(json.dumps({"scene": scene, "target": {"private": "must never reach the model"}}), encoding="utf8")
    grounder = SoftwareGrounder()
    service = PassengerSite(load_scenes([bank]), grounder, None, tmp_path / "sessions", PassengerSession)
    with ThreadingHTTPServer(("127.0.0.1", 0), handler(service)) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", next(iter(service.scenes)), grounder
        finally:
            server.shutdown()
            worker.join()


def post(base, path, data, revision=None):
    headers = {"Content-Type": "application/json"}
    if revision is not None:
        headers["X-Session-Revision"] = str(revision)
    request = Request(base + path, data=json.dumps(data).encode(), headers=headers)
    try:
        with urlopen(request, timeout=4) as response:
            return response.status, json.loads(response.read())
    except HTTPError as response:
        return response.code, json.loads(response.read())


def test_current_pair_is_explicit_then_independently_confirmed(http_site):
    base, key, grounder = http_site
    _, opened = post(base, "/api/sessions", {"scene": key})
    url = "/api/sessions/" + opened["session"]
    assert opened["view"]["pair_choices"] == []
    assert post(base, url + "/select-pair", {"track_ids": [6,11], "placement": "behind"}, 0)[0] == 400
    _, submitted = post(base, url + "/submit", {"text": "Drop me between these two vehicles.", "mode":"typed"}, 0)
    assert [t["track_id"] for t in submitted["view"]["pair_choices"]] == [6,11]
    choice = {"track_ids": [6,11], "placement": "behind"}
    assert post(base, url + "/select-pair", choice, 0)[0] == 409 and not grounder.pair_calls
    _, picked = post(base, url + "/select-pair", choice, 1)
    assert picked["view"]["state"] == "proposed" and picked["view"]["selected_by"] == "passenger_landmarks"
    assert grounder.pair_calls == [([6,11],"behind")]
    stale = picked["view"]["proposal_id"]
    _, replaced = post(base, url + "/select-pair", {"track_ids":[6,11], "placement":"between"}, 2)
    token = replaced["view"]["proposal_id"]
    assert token and token != stale
    assert post(base, url + "/confirm", {"proposal_id": stale}, 3)[0] == 409
    status, final = post(base, url + "/confirm", {"proposal_id": token}, 3)
    assert status == 200 and final["view"]["state"] == "confirmed" and final["view"]["pair_choices"] == []


def test_replacing_audio_hides_and_blocks_previous_pair(http_site):
    base, key, grounder = http_site
    _, opened = post(base, "/api/sessions", {"scene": key})
    url = "/api/sessions/" + opened["session"]
    _, prior = post(base, url + "/submit", {"mode":"typed", "text":"Between the two cars please."}, 0)
    assert len(prior["view"]["pair_choices"]) == 2
    _, audio = post(base, url + "/begin-audio", {}, 1)
    assert audio["view"]["pair_choices"] == [] and audio["view"]["proposal_id"] is None
    status, _ = post(base, url + "/select-pair", {"track_ids":[6,11],"placement":"behind"}, 2)
    assert status == 409 and not grounder.pair_calls
    _, cancelled = post(base, url + "/cancel", {}, 2)
    assert cancelled["view"]["state"] == "cancelled" and cancelled["view"]["pair_choices"] == []
