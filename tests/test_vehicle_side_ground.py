from hashlib import sha256
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest

from trackrag.dropoff import vehicle_side_ground as side, vehicle_side_view as view


def trial_rows():
    return [{"role": role} for role in ("candidate", "candidate", "wrong_side_control", "occupied_control", "lane_control")]


def test_wide_and_crop_must_both_support_the_candidate():
    grades = [("OPEN", "OPEN"), ("OPEN", "UNCLEAR"), ("WRONG_SIDE", "OPEN"),
              ("BLOCKED", "BLOCKED"), ("LANE", "LANE")]
    assert side.qualified_indices(trial_rows(), grades) == [0]
    assert side.qualified_indices(trial_rows(), grades[:-1]) == []
    assert side.qualified_indices(trial_rows()[0:2], grades[0:2]) == []
    assert side.qualified_indices(trial_rows()[:-1], grades[:-1]) == []
    assert side.qualified_indices(trial_rows(), [("maybe", "OPEN"), *grades[1:]]) == []


@pytest.mark.parametrize("failed_control", [2, 3, 4])
def test_false_open_on_any_control_disqualifies_every_offer_from_that_source(failed_control):
    grades = [("OPEN", "OPEN"), ("OPEN", "OPEN"), ("WRONG_SIDE", "WRONG_SIDE"),
              ("BLOCKED", "BLOCKED"), ("LANE", "LANE")]
    grades[failed_control] = ("OPEN", "OPEN")
    assert side.qualified_indices(trial_rows(), grades) == []


def test_archive_requires_an_external_source_bound_release(tmp_path, monkeypatch):
    bank = tmp_path/"signed.zip"
    bank.write_bytes(b"test observation archive")
    review = tmp_path/"independent.json"
    review.write_bytes(b"test review")
    release = tmp_path/"release.json"
    monkeypatch.setattr(side, "BANK", bank)
    monkeypatch.setattr(side, "RELEASE", release)
    side._release_state.cache_clear()
    assert not side.released()
    counts = {"independently_supported_offers": 1, "accepted_negative_controls": 0, "accepted_wrong_places": 0}
    verdict = {"recipe": view.RECIPE, "enabled": True, "review_artifact": review.name,
               "review_sha256": sha256(review.read_bytes()).hexdigest(), "bank_sha256": sha256(bank.read_bytes()).hexdigest(),
               "release_counts": counts}
    release.write_text(json.dumps(verdict))
    assert side.released()
    assert not side.release_passes({**counts, "accepted_wrong_places": 1})
    assert not side.release_passes({**counts, "accepted_negative_controls": 1})
    assert not side.release_passes({**counts, "independently_supported_offers": 0})
    review.write_bytes(b"altered")
    assert not side.released()


def test_native_selection_is_adjacent_connected_and_empty_and_has_hidden_controls():
    value = np.full((150, 200), 13, np.uint8)
    value[60:102, 66:132] = 10
    scene = {"frame": {"width": 200, "height": 150, "frame_id": 1},
             "tracks": [{"frame_id": 1, "class_name": "car", "bbox_xyxy": [91, 42, 119, 84], "track_id": 7}]}
    spec = {"track_id": 7, "box": [91, 42, 119, 84]}
    reading = ["red", "sedan", "REAR", "NONE"]
    native_rows = side.geometry_rows(scene, value, [spec], [reading])
    candidate = next(row for row in native_rows if row["role"] == "candidate")
    assert candidate["side"] == "left" and candidate["point"][0] < spec["box"][0]
    assert {row["role"] for row in native_rows} == {"candidate","wrong_side_control","occupied_control","lane_control"}
    assert not side.geometry_rows(scene, value, [spec, {**spec, "track_id": 8}], [reading,reading])
    road_only = np.full((150, 200), 13, np.uint8)
    assert not side.geometry_rows(scene, road_only, [spec], [reading])


def test_render_and_signed_source_verifier_detect_a_switched_side_or_point(monkeypatch):
    image = Image.new("RGB", (320, 240), "tan")
    raw = BytesIO()
    image.save(raw, "PNG")
    photo = raw.getvalue()
    key = sha256(photo).hexdigest()
    row = {"role": "candidate", "track_id": 1, "anchor_box": [100, 65, 190, 164],
           "side": "left", "point": [66, 163], "box": [61, 158, 71, 168]}
    calls = [{"image":sha256(view.render(image,row,close=flag)).hexdigest(),"verdict":"OPEN"}
             for flag in (False, True)]
    assert calls[0]["image"] != calls[1]["image"]
    body = {"recipe":view.RECIPE, "source":key, "context":"bound", "model":side.identity.MODEL,
            "revision":side.identity.REVISION, "guide":view.digest(view.GUIDE), "rows":[{"spec":view.digest(row),"calls":calls}]}
    packet = {"body":body,"signature":"test"}
    with pytest.raises(ValueError, match="signature"):
        side.verify_rows(photo,key,[row],packet,"bound")
    monkeypatch.setattr(side, "_authenticate", lambda p: None)
    assert side.verify_rows(photo,key,[row],packet,"bound") == [("OPEN","OPEN")]
    for changed in ({**row,"side":"right"}, {**row,"point":[74,160]}):
        with pytest.raises(ValueError, match="original side views"):
            side.verify_rows(photo,key,[changed],packet,"bound")
    with pytest.raises(ValueError, match="current|belong"):
        side.verify_rows(photo,key,[row],packet,"another source")


def test_choice_honors_ground_side_and_anchor_from_actual_question(monkeypatch):
    req = {"text": "Pull up to the left of the red sedan.", "scene": {"frame": {"width": 900, "height": 480}, "provenance": {}}}
    specific = {"relation": "beside", "ground_side": "left", "side": None, "kind": "sedan", "color": "red"}
    specs = [{"track_id": 7,"box": [200, 60, 290, 145]}]
    answers = [["red", "sedan", "REAR", "NONE"]]
    rows = [{"box":[108,104,133,129],"point":[120,116],"anchor_box":specs[0]["box"],"side":"left","track_id":7},
            {"box":[310,109,325,124],"point":[317,116],"anchor_box":specs[0]["box"],"side":"right","track_id":7}]
    monkeypatch.setattr(side, "released", lambda: True)
    monkeypatch.setattr(side, "read_photo_packet", lambda *args: {"signed":"side"})
    monkeypatch.setattr(side, "source_input", lambda req,photo: (req,"identity",None))
    monkeypatch.setattr(side.identity, "direct_reference", lambda req: specific)
    monkeypatch.setattr(side, "_confirmed_geometry", lambda *args: rows)
    options, packet, words = side.side_ground_choices(req,b"",{},specific,specs,answers,packet={"signed":"side"})
    assert [r["point"] for r in options] == [[120,116]]
    assert options[0]["evidence"]["coverage"] == "interior_point_only" and "image left" in words
    assert packet == {"signed":"side"}
    with pytest.raises(ValueError, match="instruction"):
        side.side_ground_choices(req,b"",{},{**specific,"ground_side":"right"},specs,answers,packet={"signed":"side"})
    with pytest.raises(ValueError, match="approved observation archive"):
        side.side_ground_choices(req,b"",{},specific,specs,answers,packet={"signed":"other"})
    assert side.side_ground_choices({**req,"text":"Only if parking is free"},b"",{},specific,specs,answers) is None
    local = {**req,"scene":{**req["scene"],"provenance":{"network_policy":"local_only"}}}
    assert side.side_ground_choices(local,b"",{},specific,specs,answers) is None


def test_malformed_model_text_is_uncertain():
    assert view.verdict("OPEN.") == "OPEN"
    assert view.verdict("Maybe open but I cannot see.") == "UNCLEAR"


def test_front_and_rear_reverse_only_an_explicit_vehicle_side():
    box, size = [120, 57, 204, 162], (450, 300)
    for facing, expected in (("REAR", "right"), ("FRONT", "left")):
        assert side.requested_image_side("beside it on its right", "right", facing, box, size) == (True, expected)
        assert side.requested_image_side("on the right of the car", "right", facing, box, size) == (True, "right")
    assert side.requested_image_side("on the SUV’s left", "left", "SIDE", box, size) == (False, None)
    assert side.requested_image_side("on its own right", "right", "UNCLEAR", box, size) == (False, None)
    assert side.requested_image_side("on its right", "right", "FRONT", [0,57,204,162], size) == (False, None)
