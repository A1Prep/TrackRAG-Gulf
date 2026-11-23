"""Replay archived visual observations against the original photo, tracks and highlighted area."""
from copy import deepcopy
from hashlib import sha256

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_gravel_pullin import frontage
from test_passenger_manual_offlane import provide_map
from trackrag.dropoff import curb_points, parking_evidence, roadside_board, roadside_context, whole_bay
from trackrag.dropoff.passenger import digest


def seal(body, key):
    return {"body":body, "signature":key.sign(parking_evidence.signed_bytes(body)).hex()}


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    original, classes, ctx = frontage()
    path = tmp_path / "gravel.png"
    original.save(path)
    photo = path.read_bytes()
    image_sha = sha256(photo).hexdigest()
    frame = {"recording_id":"test-unpaved", "frame_id":1, "timestamp_s":0, "width":original.width,
             "height":original.height, "image_path":str(path)}
    scene = {"frame":frame, "source":"predicted", "provenance":{"image_sha256":image_sha},
             "tracks":[{"recording_id":frame["recording_id"], "frame_id":1, "track_id":track, "class_name":name,
                        "bbox_xyxy":box, "source":"predicted"} for track,name,box in ctx["current_vehicles"]]}
    request = {"request_id":"roadside", "text":"Drop me by the blue bin.", "input_mode":"typed", "dialogue":[], "scene":scene}
    surface = provide_map(monkeypatch, scene, classes)
    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    monkeypatch.setattr(parking_evidence, "PUBLIC_KEY", public.hex())
    pairs = roadside_context.candidates(request, photo, surface, None, gravel_only=True)
    [(spec,_)] = pairs
    row = {"spec":digest(spec), "view":sha256(roadside_board.render(original,spec)).hexdigest(),
           "description":{"status":"OPEN", "nearer_vehicle":"black hatchback", "farther_vehicle":"red minivan",
                          "frontage":"corner shop", "roadside":"blue bin beside gravel"}}
    body = {"recipe":roadside_board.RECIPE, "model":roadside_board.REPO, "revision":roadside_board.REVISION,
            "prompt":digest(roadside_board.GUIDE), "image_sha256":image_sha, "context":digest(curb_points.source_context(scene)),
            "rows":[row]}
    return request, photo, surface, pairs, seal(body,key), key


def test_verified_gravel_location_accepts_only_a_tap_in_witnessed_ground(reviewed, monkeypatch):
    request, photo, surface, pairs, packet, _ = reviewed
    monkeypatch.setattr(roadside_context,"get_packet",lambda *_,gravel_only=False: packet if gravel_only else None)
    out = whole_bay.explicit_request(request)
    expected = pairs[0][0]["point"]
    assert out["result"]["destination_point_xy"] == expected
    assert out["provenance"]["bay_outline"]["extent"] == "visible_parking_surface"
    assert whole_bay.verify_current(request,out)["archived_gravel_context_records"] == 1
    x, y = expected
    box = [x-1,y-1,x+1,y+1]
    mark = whole_bay.select_mark(request, box)
    assert mark["result"]["destination_point_xy"] == expected
    assert whole_bay.verify_current(request,mark,requested_box=box)["independent_model_proposal"] is False
    # Further along the same gravel strip (no visual review), the travel lane and an observed car.
    for x, y in ((20,210),(190,330),(110,170)):
        other = whole_bay.select_mark(request, [x-1,y-1,x+1,y+1])
        assert other["result"]["destination_point_xy"] is None
    tampered = deepcopy(mark)
    tampered["provenance"]["bay_outline"]["center_xy"][0] += 5
    with pytest.raises(ValueError):
        whole_bay.verify_current(request,tampered,requested_box=box)


@pytest.mark.parametrize("change", ["signature","prompt","revision","context","photo","rows","spec","view","track_boxes","kind"])
def test_current_original_tracks_and_render_are_checked_even_with_resigned_metadata(reviewed,change):
    request, photo, surface, pairs, packet, key = reviewed
    body = deepcopy(packet["body"])
    if change == "signature":
        packet["signature"] = "00"*64
    elif change in ("prompt","revision","context"):
        body[change] = "changed"
    elif change == "photo":
        photo += b"changed"
    elif change == "track_boxes":
        request["scene"]["tracks"][0]["bbox_xyxy"][0] += 1
    elif change == "rows":
        body["rows"] = []
    elif change == "kind":
        pairs[0][0]["kind"] = "stall"
    else:
        body["rows"][0][change] = "changed"
    if change != "signature":
        packet = seal(body,key)
    with pytest.raises(ValueError):
        roadside_context.observations(request["scene"],photo,pairs,packet)


def test_visual_rejection_vetoes_gravel_even_if_the_geometry_passed(reviewed):
    request, photo, surface, pairs, packet, key = reviewed
    packet["body"]["rows"][0]["description"]["status"] = "BLOCKED"
    rejected = seal(packet["body"],key)
    assert roadside_context.verified(request,photo,surface,None,rejected,gravel_only=True) == []
    assert whole_bay.roadside_outcome(request,photo,surface,None,None,gravel_packet=rejected) is None


def test_explicit_through_lane_request_is_clarified_before_other_grounders(reviewed):
    request, photo, *_ = reviewed
    request["text"] = "Drop me in the middle of the street by the blue bin."
    out = whole_bay.explicit_request(request)
    assert out["result"]["status"] == "needs_clarification"
    assert out["provenance"]["reason"] == "destination_needs_clarification"
    receipt = whole_bay.verify_current(request,out)
    assert receipt["decision_reproduced"] and not receipt["offlane_mask_replayed"]
