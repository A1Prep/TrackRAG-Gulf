"""Offline evidence and attribution tests. Separate fixture keys sign the generated observations."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from types import SimpleNamespace
from zipfile import ZipFile

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PIL import Image, ImageDraw
import pytest

from test_painted_bays import stall_row
from test_passenger_manual_offlane import provide_map
from tools.dropoff_passenger_handoff import bound_outcome, session as export_journal
from tools.dropoff_passenger_references import _row
from tools.dropoff_passenger_references_public import score_outcome
from trackrag.dropoff import bay_anchors, bay_landmarks, parking_evidence, place_evidence, surface_stalls, whole_bay
from trackrag.dropoff.data import inference_input_sha256
from trackrag.dropoff.passenger import digest


def pixel_sha(data):
    return sha256(data).hexdigest()


def seal(body, key):
    return {"body":body, "signature":key.sign(parking_evidence.signed_bytes(body)).hex()}


def surface_review(image, classes, context, surface, key):
    options = surface_stalls.proposals(image, classes, context)
    assert len(options) >= 2
    rows = []
    for index, option in enumerate(options):
        answers = ["BAY", "BAY"] if index == 0 else ["FRAGMENT"] if index == 1 else ["BAY", "OCCUPIED"]
        reviews = [{"view_sha256":pixel_sha(surface_stalls.review_view(image, option, close=bool(i))), "reply":answer}
                   for i, answer in enumerate(answers)]
        rows.append({"shape_sha256":digest(option), "reviews":reviews})
    body = {"recipe":surface_stalls.RECIPE, "models":bay_landmarks.MODELS, "image_sha256":context["image_sha256"],
            "size":list(image.size), "context_sha256":digest(context), "surface_signature":surface["signature"],
            "prompt_sha256":pixel_sha((surface_stalls.CHECK+"\n"+surface_stalls.WORDS).encode()), "options":rows}
    return seal(body, key)


def anchor_review(image, options, scene, key, recipe):
    vehicles = bay_anchors.vehicles(scene)
    rows, appearance = [], {}
    for index, option in enumerate(options):
        local = bay_anchors.closest_tracks(option, vehicles)
        assert len(local) == 2
        rows.append({"shape":digest(option["polygon"]), "view":pixel_sha(bay_anchors.view(image, option, local)),
                     "frontage":"grocery" if index == 0 else "library", "pair_reply":"1, 2" if index == 0 else "NONE"})
        if index == 0:
            for track, _, box in local:
                appearance[str(track)] = {"reply":"black car" if track == 4 else "white van",
                                          "view":pixel_sha(bay_anchors.crop_car(image, box))}
    body = {"recipe":recipe, "model":bay_landmarks.MODELS, "image_sha256":scene["provenance"]["image_sha256"],
            "size":list(image.size), "current_vehicles_sha256":digest(vehicles), "prompt":bay_anchors.prompt_digest(recipe),
            "rows":rows, "appearance":appearance}
    return seal(body, key)


@pytest.fixture
def bank(tmp_path, monkeypatch):
    key = Ed25519PrivateKey.generate()
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    monkeypatch.setattr(parking_evidence, "PUBLIC_KEY", pub.hex())
    image, classes, context = stall_row()
    draw = ImageDraw.Draw(image)
    draw.rectangle((42,100,60,126), fill="#31323a")
    draw.rectangle((266,93,295,128), fill="#d7d9df")
    photo = tmp_path/"two-bays.png"
    image.save(photo)
    sha = pixel_sha(photo.read_bytes())
    frame = {"width":image.width, "height":image.height, "image_path":str(photo), "frame_id":12,
             "recording_id":"synthetic-bay-row", "timestamp_s":1.2}
    def track(identifier, frame_id, name, box):
        return {"recording_id":frame["recording_id"], "frame_id":frame_id, "track_id":identifier,
                "class_name":name, "bbox_xyxy":box, "source":"predicted"}
    tracks = [track(9,11,"car",[115,90,144,125]), track(4,12,"car",[42,100,60,126]), track(7,12,"van",[266,93,295,128])]
    scene = {"frame":frame, "tracks":tracks, "source":"predicted", "provenance":{"image_sha256":sha}}
    context.update(image_sha256=sha, current_boxes=[t["bbox_xyxy"] for t in tracks if t["frame_id"]==frame["frame_id"]])
    req = {"request_id":"source-bay", "text":"Use the marked parking bay.", "input_mode":"typed", "dialogue":[], "scene":scene}
    surface = provide_map(monkeypatch, scene, classes)
    stalls = surface_review(image, classes, context, surface, key)
    options = whole_bay.visible_stalls(image, classes, context, surface, stalls)
    assert len(options) == 2
    anchors = anchor_review(image, options, scene, key, bay_anchors.RECIPE)
    return SimpleNamespace(image=image, classes=classes, context=context, scene=scene, req=req, key=key,
                           options=options, surface=surface, stalls=stalls, anchors=anchors, photo=photo)


def observe(f, packet):
    return bay_anchors.observations(f.image, f.options, f.scene, packet)


def reviewed(f, packet):
    return surface_stalls.approved(f.image, f.classes, f.context, f.surface, packet)


@pytest.mark.parametrize("version", [bay_anchors.LEGACY_RECIPE, bay_anchors.RECIPE])
def test_complete_signed_anchor_bank_reproduces_both_schemes(bank, version):
    f = bank
    packet = anchor_review(f.image, f.options, f.scene, f.key, version)
    result = observe(f, packet)
    assert {row["frontage"] for row in result.values()} == {"grocery", "library"}
    assert {v["track_id"] for row in result.values() for v in row["pair"]} == {4, 7}
    assert [row[0] for row in bay_anchors.vehicles(f.scene)] == [4, 7]
    packet["body"]["rows"].reverse()
    assert observe(f, seal(packet["body"], f.key)) == result
    assert bay_anchors.prompt_digest(bay_anchors.LEGACY_RECIPE) == digest([bay_anchors.PAIR, bay_anchors.FRONT, bay_anchors.APPEARANCE])
    assert bay_anchors.prompt_digest() != bay_anchors.prompt_digest(bay_anchors.LEGACY_RECIPE)


@pytest.mark.parametrize("changed", ["unsigned", "missing", "duplicate", "shape", "view", "crop", "extra crop",
                                     "missing crop", "tracks", "photo", "size", "model", "version", "prompt"])
def test_signed_anchor_packets_cannot_hide_other_bays_or_substitute_another_crop(bank, changed):
    f = bank
    packet = deepcopy(f.anchors)
    body = packet["body"]
    rows = body["rows"]
    if changed == "unsigned": rows[0]["frontage"] = "new label"
    if changed == "missing": rows.pop()
    if changed == "duplicate": rows[1] = rows[0]
    if changed == "shape": rows[0]["shape"] = "0"*64
    if changed == "view": rows[0]["view"] = rows[1]["view"]
    if changed == "crop": body["appearance"]["4"]["view"] = body["appearance"]["7"]["view"]
    if changed == "extra crop": body["appearance"]["9"] = body["appearance"]["7"]
    if changed == "missing crop": body["appearance"].pop("4")
    if changed == "tracks": body["current_vehicles_sha256"] = "0"*64
    if changed == "photo": body["image_sha256"] = "0"*64
    if changed == "size": body["size"].reverse()
    if changed == "model": body["model"] = []
    if changed == "version": body["recipe"] = "unknown"
    if changed == "prompt": body["prompt"] = bay_anchors.prompt_digest(bay_anchors.LEGACY_RECIPE)
    with pytest.raises(ValueError):
        observe(f, packet if changed == "unsigned" else seal(body, f.key))


@pytest.mark.parametrize("changed", ["unsigned", "missing", "duplicate", "unknown shape", "negative view", "positive crop",
                                     "missing crop", "unexpected crop", "wrong map", "wrong detections", "wrong prompt"])
def test_stall_reviews_bind_every_positive_and_negative_model_view(bank, changed):
    f = bank
    assert len(reviewed(f, f.stalls)) == 1
    packet = deepcopy(f.stalls)
    body = packet["body"]
    rows = body["options"]
    if changed == "unsigned": rows[0]["reviews"][0]["reply"] = "LANE"
    if changed == "missing": rows.pop(1)
    if changed == "duplicate": rows[1] = rows[0]
    if changed == "unknown shape": rows[1]["shape_sha256"] = "0"*64
    if changed == "negative view": rows[1]["reviews"][0]["view_sha256"] = rows[0]["reviews"][0]["view_sha256"]
    if changed == "positive crop": rows[0]["reviews"][1]["view_sha256"] = rows[0]["reviews"][0]["view_sha256"]
    if changed == "missing crop": rows[0]["reviews"].pop()
    if changed == "unexpected crop":
        opt = surface_stalls.proposals(f.image, f.classes, f.context)[1]
        rows[1]["reviews"].append({"reply":"LANE", "view_sha256":pixel_sha(surface_stalls.review_view(f.image,opt,True))})
    if changed == "wrong map": body["surface_signature"] = "0"*128
    if changed == "wrong detections": body["context_sha256"] = "0"*64
    if changed == "wrong prompt": body["prompt_sha256"] = "0"*64
    with pytest.raises(ValueError):
        reviewed(f, packet if changed == "unsigned" else seal(body, f.key))


@pytest.mark.parametrize("module,field", [(bay_anchors,"anchors"), (surface_stalls,"stalls"), (bay_landmarks,"stalls")])
def test_current_archive_returns_isolated_evidence_and_rejects_duplicate_members(bank, tmp_path, monkeypatch, module, field):
    sha = bank.context["image_sha256"]
    packet = getattr(bank, field)
    archive = tmp_path/"release.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr(sha+".json", json.dumps(packet))
    monkeypatch.setattr(module, "BANK", archive)
    original_stat = archive.stat()
    module.get_packet(sha)["body"].clear()
    assert module.get_packet(sha) == packet
    assert module.get_packet("../"+sha) is None
    with ZipFile(archive, "a") as zipped, pytest.warns(UserWarning, match="Duplicate"):
        zipped.writestr(sha+".json", json.dumps(packet))
    os.utime(archive, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert module.get_packet(sha) is None


def resolve(f, request):
    return whole_bay.explicit_outcome(request, f.photo.read_bytes(), f.surface, f.stalls, f.anchors)


def score_both(f, request, output, expected):
    frame = request["scene"]["frame"]
    sha = f.context["image_sha256"]
    ref = {"request_id":request["request_id"], "recording_id":frame["recording_id"], "frame_id":frame["frame_id"],
           "target":expected, "image_sha256":sha}
    mask = f.photo.with_name("fixture-mask.png")
    Image.new("L", f.image.size, 255).save(mask)
    case = {"case_id":request["request_id"], "scene_group":"fixture", "source":{"image_sha256":sha, "mask_path":str(mask)},
            "target":expected}
    return _row(ref, request, [output]), score_outcome(case, request, [output])


def test_semantic_clarification_and_selection_have_different_actors_and_scores(bank):
    f = bank
    menu = resolve(f, f.req)
    assert len(menu["provenance"]["bay_choices"]) == 2 and not menu["provenance"]["independent_model_proposal"]
    checks = whole_bay.verify(f.req, menu)
    assert checks["archived_stall_language_calls"] == sum(len(r["reviews"]) for r in f.stalls["body"]["options"])
    assert checks["archived_bay_context_records"] == len(f.options) and checks["current_language_calls"] == 0
    authored, public = score_both(f, f.req, menu, menu["result"])
    assert authored["status_matches"] and public["status"] == "needs_clarification"
    assert not public["destination_provided"]
    for choice in menu["provenance"]["bay_choices"]:
        selected = whole_bay.choice_outcome(f.req, menu, choice["id"])
        verified = bound_outcome(selected,f.req,passenger_selected=True,selected_bay_id=choice["id"],bay_source_sha=digest(menu))
        assert verified["passenger_mark_verification"]["decision_reproduced"] and not verified["independent_model_proposal"]
        with pytest.raises(ValueError):
            bound_outcome(selected, f.req, dialogue_policy=True)
        selected["provenance"]["independent_model_proposal"] = True
        for row in score_both(f, f.req, selected, selected["result"]):
            assert row["failure"] and row["status"] is None


def test_a_valid_bound_input_does_not_authenticate_an_altered_bay_point(bank):
    f = bank
    request = {**f.req, "text":"Use the leftmost parking bay."}
    output = resolve(f, request)
    assert output["result"]["status"] == "proposed"
    authored, public = score_both(f, request, output, output["result"])
    assert authored["status_matches"] and public["point_in_public_mask"]
    altered = deepcopy(output)
    altered["result"]["destination_point_xy"][0] += 1
    assert altered["provenance"]["inference_input_sha256"] == inference_input_sha256(request, f.context["image_sha256"])
    for row in score_both(f, request, altered, output["result"]):
        assert row["failure"] and row["status"] is None


@pytest.mark.parametrize("archive_policy", [None,whole_bay.VEHICLE_POLICY,whole_bay.CONNECTED_POLICY])
def test_old_storefront_choices_replay_but_cannot_be_selected_or_saved_as_current(bank, archive_policy):
    f = bank
    req = {**f.req,"text":"Use the bay on the left beside the blue SUV, outside Juniper Books shop."}
    old = whole_bay.explicit_outcome(req,f.photo.read_bytes(),f.surface,f.stalls,None,policy=archive_policy)
    assert old["provenance"]["reason"] == "unverified_bay_reference"
    assert len(old["provenance"]["bay_choices"]) == 1
    assert whole_bay.verify(req,old)["decision_reproduced"]
    chosen = whole_bay.choice_outcome(req,old,"bay-1")
    assert whole_bay.verify(req,chosen)["decision_reproduced"]
    for receipt in (old,chosen):
        with pytest.raises(ValueError,match="new check"):
            whole_bay.verify_current(req,receipt)
    current = whole_bay.explicit_outcome(req,f.photo.read_bytes(),f.surface,f.stalls,None)
    assert current["provenance"]["clearance_policy"] == whole_bay.CLEARANCE_POLICY
    assert current["provenance"]["reason"] == "no_supported_bay_for_instruction"
    assert current["provenance"]["bay_choices"] == [] and current["provenance"]["bay_outline"] is None
    assert current["result"]["destination_point_xy"] is None and current["result"]["status"] == "needs_clarification"
    assert whole_bay.verify_current(req,current)["decision_reproduced"]
    with pytest.raises(ValueError):
        whole_bay.choice_outcome(req,current,"bay-1")


@pytest.mark.parametrize("state", ["pending", "busy"])
def test_operational_wait_has_no_model_or_semantic_credit(bank, state):
    f = bank
    req = {**f.req, "text":"Drop me beside the library."}
    clean, sha, binding = parking_evidence.source_input(req, f.photo.read_bytes())
    waiting = place_evidence.waiting_outcome(clean, sha, binding, state)
    for row in score_both(f, req, waiting, waiting["result"]):
        assert row["failure"] == "lookup_"+state and row["status"] is None


@pytest.mark.parametrize("changed", ["time", "future", "recording", "original", "mode", "words"])
def test_bay_replay_keeps_request_time_scene_and_original_source(bank, changed):
    f = bank
    output = resolve(f, f.req)
    req = deepcopy(f.req)
    if changed == "time": req["scene"]["frame"]["timestamp_s"] += 2
    if changed == "future": req["scene"]["tracks"][0]["frame_id"] = req["scene"]["frame"]["frame_id"]+1
    if changed == "recording": req["scene"]["frame"]["recording_id"] = "other-recording"
    if changed == "original": f.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(f.photo)
    if changed == "mode": req.update(input_mode="voice", original_transcript=req["text"])
    if changed == "words": req["text"] = "Use the left parking bay."
    with pytest.raises(ValueError):
        whole_bay.verify(req, output)


def test_submitted_error_cannot_export_a_stale_pictured_destination(bank, tmp_path):
    from trackrag.dropoff.interaction import PassengerSession
    class FailedModel:
        def interpret(self, req):
            raise RuntimeError("fixture model failed")
    dialogue = PassengerSession(bank.scene, FailedModel(), audit_dir=tmp_path/"journal")
    assert dialogue.submit("Find a bay.")["state"] == "error"
    events = [json.loads(line) for line in dialogue._journal.read_bytes().splitlines()]
    assert export_journal(dialogue._journal)["events"][-1]["state"] == "error"
    for field, value in (("bay_choices",[{"id":"bay-1"}]), ("bay_outline", {"id":"bay-1"})):
        edited = deepcopy(events)
        edited[-1]["view"][field] = value
        temp = tmp_path/(field+".jsonl")
        temp.write_text("\n".join(json.dumps(event) for event in edited), encoding="utf-8")
        with pytest.raises(ValueError):
            export_journal(temp)
