from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace
import threading

import numpy as np
from PIL import Image
import pytest

from tools.dropoff_passenger_handoff import bound_outcome, session as export
from trackrag.dropoff import passenger_pair_choice as pc
from trackrag.dropoff.contracts import error_outcome
from trackrag.dropoff.passenger import PassengerGrounder, digest
from trackrag.dropoff.road_cache import mask_shape
from tests.test_passenger_projected_pair import fitted_head, obs


COMMAND = "Leave me between the two black cars."


@pytest.fixture
def setup(tmp_path, monkeypatch):
    picture = tmp_path / "road-test.png"
    Image.new("RGB", (360, 180), "navy").save(picture)
    fingerprint = digest({"software_fixture": "trained-on-synthetic-pairs"})
    monkeypatch.setattr(pc, "GEOMETRY", fingerprint)
    raw = object.__new__(PassengerGrounder)
    raw.config = SimpleNamespace(max_image_bytes=500000, max_original_pixels=360 * 180)
    raw.lock = threading.RLock()
    raw.geometry = SimpleNamespace(sha256=fingerprint, pair_regression=fitted_head())
    raw.road = SimpleNamespace(mask=lambda im, sha: np.ones(mask_shape(im.size), dtype=bool))
    frame = {
        "width": 360,
        "height": 180,
        "recording_id": "fixture-clip",
        "frame_id": 8,
        "timestamp_s": 0.7,
        "image_path": str(picture),
    }
    tracks = [
        obs(51, [130, 68, 171, 100]),
        obs(92, [166, 68, 207, 100], "Bus"),
        obs(70, [250, 110, 258, 132], "Pedestrian"),
    ]
    past = obs(5, [49, 36, 83, 60])
    past["frame_id"] = 7
    scene = {
        "frame": frame,
        "source": "predicted",
        "tracks": tracks + [past],
        "provenance": {"image_sha256": sha256(picture.read_bytes()).hexdigest()},
    }
    req = {
        "request_id": "software-choice",
        "scene": scene,
        "text": COMMAND,
        "input_mode": "typed",
        "dialogue": [],
    }
    raw.interpret = lambda r: error_outcome(
        "timeout",
        "Synthetic original solver failure retained",
        r["request_id"],
        {"method": "qwen_landmark_region"},
    )
    return req, raw


@pytest.mark.parametrize(
    "words,required",
    [
        ("Behind the gap between the car and the bus.", True),
        ("Between the bus stop and a car.", False),
        ("Between the bus by the bus stop and the red car.", True),
        ("Between the two black cars parked by the curb.", True),
        ("Go between the two black cars near the left median.", True),
        ("Between the gate and the buses.", False),
        ("Between the two cars or the two buses.", False),
        ("Between the two cars and the door.", False),
        ("Rather than between those cars, drop me by the gate.", False),
        ("Near the truck instead of between the cars.", False),
        ("Between two vehicles; actually stop by the shop.", False),
        ("Do not drop me between the two cars.", False),
        ("Go between a van and the taxi stand.", False),
        ("Between the car park and the bus stop.", False),
    ],
)
def test_current_positive_road_user_pair_only(words, required):
    assert pc.pair_intent({"text": words, "dialogue": []}) is required


@pytest.mark.parametrize(
    "follow,expected",
    [
        ("The car nearer the camera.", True),
        ("The pair by the median.", True),
        ("Between them, please.", True),
        ("Next to the bus stop, please.", False),
        ("Over by the gate, please.", False),
        ("No, that gate on the right.", False),
        ("I want to get out next to the pharmacy.", False),
        ("Actually by the green shop.", False),
        ("I don’t want that anymore.", False),
        ("Do not stop between them.", False),
        ("Never mind.", False),
    ],
)
def test_replacement_retires_prior_choices_but_true_clarification_preserves_them(follow, expected):
    turns = [{"role": "user", "text": COMMAND}, {"role": "assistant", "text": "Did you mean the bus stop?"}]
    assert pc.pair_intent({"text": follow, "dialogue": turns}) is expected


def test_overlap_needs_explicit_behind_and_actual_current_observations(setup):
    req, model = setup
    assert [r["track_id"] for r in pc.options(req)] == [51, 92]
    between = pc.choose(model, req, [92, 51], "between")
    assert (
        between["result"]["status"] == "needs_clarification"
        and "overlap" in between["result"]["clarification_question"]
    )
    assert pc.validate_choice_evidence(req, between, [92, 51], "between")["frame_id"] == 8
    behind = pc.choose(model, req, [92, 51], "behind")
    assert (
        behind["result"]["status"] == "proposed" and behind["provenance"]["landmark_selection"] == "passenger"
    )
    assert behind["result"]["anchor_boxes_xyxy"] == [req["scene"]["tracks"][i]["bbox_xyxy"] for i in (1, 0)]
    assert (
        pc.validate_choice_evidence(req, behind, [92, 51], "behind")["image_sha256"]
        == req["scene"]["provenance"]["image_sha256"]
    )
    assert behind["result"]["destination_point_xy"][1] > 100
    assert behind["provenance"]["independent_model_proposal"] is False


@pytest.mark.parametrize(
    "ids,placement",
    [
        ([51, 51], "behind"),
        ([True, 92], "between"),
        ([5, 92], "behind"),
        ([70, 92], "behind"),
        ([0, 92], "behind"),
        ([51], "behind"),
        ([51, 92], "near"),
        ([51, 92], ["behind"]),
    ],
)
def test_invalid_or_stale_choices_do_not_run_geometry(setup, ids, placement):
    req, model = setup
    with pytest.raises(pc.InvalidSelection):
        pc.choose(model, req, ids, placement)


def test_real_gap_entire_proposal_remains_inside_corridor_and_outside_all_road_users(setup):
    req, model = setup
    req["scene"]["tracks"][0]["bbox_xyxy"] = [112, 68, 145, 100]
    req["scene"]["tracks"][1]["bbox_xyxy"] = [188, 66, 224, 100]
    for placement in ("between", "behind"):
        out = pc.choose(model, req, [51, 92], placement)
        assert out["result"]["status"] == "proposed"
        pc.validate_choice_evidence(req, out, [51, 92], placement)
        x, y, r, b = out["result"]["destination_box_xyxy"]
        if placement == "between":
            assert 145 <= x < r <= 188
        else:
            assert y >= 100
    model.road.mask = lambda im, sha: np.zeros(mask_shape(im.size), dtype=bool)
    empty = pc.choose(model, req, [51, 92], "between")
    assert (
        empty["result"]["status"] == "needs_clarification"
        and empty["provenance"]["withheld_reason"] == "no_supported_neighborhood"
    )


def test_corrupt_pixels_and_missing_road_computation_are_operational(setup):
    req, model = setup
    model.road.mask = lambda im, sha: None
    with pytest.raises(RuntimeError, match="invalid result"):
        pc.choose(model, req, [51, 92], "behind")

    def missing(im, sha):
        raise pc.Ineligible("no_cached_current_road_mask")

    model.road.mask = missing
    with pytest.raises(RuntimeError, match="could not be loaded"):
        pc.choose(model, req, [51, 92], "behind")
    Image.new("RGB", (360, 180), "red").save(req["scene"]["frame"]["image_path"])
    with pytest.raises(ValueError, match="SHA256"):
        pc.choose(model, req, [51, 92], "behind")


def test_passenger_can_point_without_rewriting_words_or_running_native_model(setup, tmp_path):
    req, model = setup
    words = "This is the place I was talking about."
    def unexpected(_):
        pytest.fail("An explicit request to point made a native model call.")
    model.interpret = unexpected
    directory = tmp_path / "pointing"
    sess = pc.PassengerSession(req["scene"], model, audit_dir=directory)
    asked = sess.submit(words, point_to_vehicles=True)
    assert asked["state"] == "needs_clarification" and asked["selected_by"] == "passenger"
    assert asked["turns"][0]["text"] == words and not asked["proposal_id"]
    assert {r["track_id"] for r in asked["pair_choices"]} >= {51,92}
    proposed = sess.select_pair([51,92], "behind")
    assert proposed["state"] == "proposed"
    sess.confirm(proposed["proposal_id"])
    journal, = directory.glob("*.jsonl")
    events = export(journal)["events"]
    assert [e["phase"] for e in events] == ["opened", "explicit_passenger_pointing_request",
        "passenger_selected_landmarks", "confirmed_previous_selection"]
    provenance = events[2]["outcome"]["saved_provenance"]
    assert "automatic_submission" not in provenance
    assert provenance["pointing_submission"]["request_id"] == asked["request_id"]
    assert provenance["passenger_selection"]["passenger_requested_pointing"] is True
    lines = [json.loads(line) for line in journal.read_bytes().splitlines()]
    del lines[1]["detail"]["passenger_requested_pointing"]
    journal.write_text("".join(json.dumps(line)+"\n" for line in lines), encoding="utf-8")
    with pytest.raises(ValueError):
        export(journal)


def test_explicit_pointing_is_one_turn_only_and_handles_missing_tracks(setup, tmp_path):
    req, model = setup
    req["scene"]["tracks"] = []
    sess = pc.PassengerSession(req["scene"], model, audit_dir=tmp_path / "empty")
    first = sess.submit("I will show you.", point_to_vehicles=True)
    assert not first["pair_choices"] and not first["proposal_id"]
    assert "another landmark" in first["result"]["clarification_question"]
    with pytest.raises(ValueError):
        sess.select_pair([51,92], "behind")
    ordinary = sess.submit("Now take me to the corner.")
    assert ordinary["state"] == "error" and ordinary["selected_by"] is None


def test_prior_failed_original_separate_from_assisted_choice_and_explicit_current_confirmation(setup, tmp_path):
    req, raw = setup
    sess = pc.PassengerSession(req["scene"], raw, audit_dir=tmp_path / "journal")
    assert sess.view()["pair_choices"] == []
    first = sess.submit(COMMAND)
    assert first["state"] == "error" and len(first["pair_choices"]) == 2
    offered = sess.select_pair([51, 92], "behind")
    assert offered["selected_by"] == "passenger_landmarks" and offered["state"] == "proposed"
    invalid = sess.select_pair([51, 51], "between")
    assert invalid["state"] == "error" and invalid["error"]["code"] == "invalid_pair_selection"
    with pytest.raises(ValueError):
        sess.confirm(offered["proposal_id"])
    second = sess.select_pair([51, 92], "behind")
    assert second["proposal_id"] != offered["proposal_id"]
    confirmed = sess.confirm(second["proposal_id"])
    assert confirmed["state"] == "confirmed" and confirmed["pair_choices"] == []
    with pytest.raises(ValueError):
        sess.select_pair([51, 92], "behind")
    (journal,) = (tmp_path / "journal").glob("*.jsonl")
    events = export(journal)["events"]
    assert [e["phase"] for e in events] == [
        "opened",
        "initial_model",
        "passenger_selected_landmarks",
        "passenger_selected_landmarks",
        "passenger_selected_landmarks",
        "confirmed_previous_selection",
    ]
    assert events[1]["outcome"]["ok"] is False and events[3]["outcome"]["ok"] is False
    assert events[4]["selected_by"] == events[-1]["selected_by"] == "passenger_landmarks"
    rows = [json.loads(line) for line in journal.read_bytes().splitlines()]
    for mutation in (
        "ids",
        "prior",
        "mismatch",
        "road",
        "geometry",
        "after-confirm",
        "model-mislabel",
        "null-token",
        "reused-token",
    ):
        broken = deepcopy(rows)
        turn = broken[4]
        if mutation == "ids":
            turn["detail"]["track_ids"].reverse()
        if mutation == "prior":
            turn["detail"]["outcome"]["provenance"]["automatic_submission"]["original_outcome_sha256"] = (
                "0" * 64
            )
        if mutation == "mismatch":
            turn["detail"]["outcome"]["provenance"]["passenger_selection"]["selected_observations"][0][
                "bbox_xyxy"
            ][0] += 1
        if mutation == "road":
            turn["detail"]["outcome"]["provenance"]["road_geometry"]["road_recipe_sha256"] = "0" * 64
        if mutation == "geometry":
            turn["detail"]["outcome"]["provenance"]["road_geometry"]["center_corridor_xyxy"][0] += 1
        if mutation == "after-confirm":
            broken.append(deepcopy(turn))
        if mutation == "model-mislabel":
            turn["action"] = "submitted"
            turn["detail"]["request"] = broken[1]["detail"]["request"]
        if mutation in ("null-token", "reused-token"):
            token = None if mutation == "null-token" else broken[2]["view"]["proposal_id"]
            turn["view"]["proposal_id"] = broken[5]["view"]["proposal_id"] = broken[5]["detail"][
                "proposal_id"
            ] = token
        replay = tmp_path / (mutation + ".jsonl")
        replay.write_text("".join(json.dumps(row) + "\n" for row in broken))
        with pytest.raises((ValueError, KeyError)):
            export(replay)


def test_changed_passenger_request_or_cancellation_cannot_reuse_the_selection(setup, tmp_path):
    req, raw = setup
    sess = pc.PassengerSession(req["scene"], raw, audit_dir=tmp_path / "journal")
    sess.submit(COMMAND)
    old = sess.select_pair([51, 92], "behind")["proposal_id"]
    assert sess.submit("Next to the gate, please.")["pair_choices"] == []
    with pytest.raises(ValueError):
        sess.confirm(old)
    with pytest.raises(ValueError):
        sess.select_pair([51, 92], "behind")
    assert sess.cancel()["pair_choices"] == []


def test_new_photo_invalidates_a_real_offer(setup, tmp_path):
    req, raw = setup
    sess = pc.PassengerSession(req["scene"], raw, audit_dir=tmp_path / "journal")
    sess.submit(COMMAND)
    token = sess.select_pair([51, 92], "behind")["proposal_id"]
    Image.new("RGB", (360, 180), "red").save(req["scene"]["frame"]["image_path"])
    with pytest.raises(ValueError, match="changed"):
        sess.confirm(token)
    assert sess.view()["state"] == "error" and sess.view()["pair_choices"] == []


def test_automatic_which_pair_receipt_recomputed_on_actual_original_without_a_qwen_call(setup, tmp_path):
    from tests.test_dropoff_referent import checkpoint
    from trackrag.dropoff.pair_dialogue import unpointed_pair
    from trackrag.dropoff.referent import ReferentRanker

    req, raw = setup
    weights, sha = checkpoint(tmp_path)
    raw.referent_ranker = ReferentRanker.load(weights, expected_sha256=sha)
    raw.interpret = lambda r: unpointed_pair(raw, r)
    sess = pc.PassengerSession(req["scene"], raw, audit_dir=tmp_path / "dialogue")
    answer = sess.submit("Drop me between those two red cars.")
    assert answer["state"] == "needs_clarification" and len(answer["pair_choices"]) == 2
    proposal = sess.select_pair([92, 51], "behind")
    assert sess.confirm(proposal["proposal_id"])["state"] == "confirmed"
    (journal,) = (tmp_path / "dialogue").glob("*.jsonl")
    events = export(journal, referent_path=weights)["events"]
    assert [e["phase"] for e in events] == [
        "opened",
        "automatic_pair_identification_question",
        "passenger_selected_landmarks",
        "confirmed_previous_selection",
    ]
    assert (
        events[1]["outcome"]["policy_recomputed_from_original"]
        and events[1]["outcome"]["native_model_calls"] == 0
    )
    original = json.loads(journal.read_bytes().splitlines()[1])["detail"]
    for field in (
        "eligible_tracks",
        "ranker_sha256",
        "ranking_sha256",
        "candidate_count",
        "native_model_calls",
        "passenger_phrase",
    ):
        changed = deepcopy(original["outcome"])
        changed["provenance"][field] = "changed"
        with pytest.raises(ValueError):
            bound_outcome(changed, original["request"], dialogue_policy=True, referent_path=weights)
    with pytest.raises(ValueError):
        bound_outcome(original["outcome"], original["request"])


def test_asset_replay_honors_mask_contents_checkpoint_and_empty_gap(setup, tmp_path, monkeypatch):
    from trackrag.dropoff.road_cache import save_mask

    req, raw = setup
    payload = {
        "format": "anchor-relative-regression-v1",
        "vocabulary": [],
        "coefficients": [[0.0] * 6 for _ in range(5)],
        "pair_regression": raw.geometry.pair_regression,
    }
    weights = tmp_path / "geometry.json"
    weights.write_text(json.dumps(payload))
    sha = sha256(weights.read_bytes()).hexdigest()
    raw.geometry.sha256 = sha
    monkeypatch.setattr(pc, "GEOMETRY", sha)
    size = (360, 180)
    cached = tmp_path / "cached"
    mask = raw.road.mask(SimpleNamespace(size=size), None)
    save_mask(cached, req["scene"]["provenance"]["image_sha256"], size, mask)
    out = pc.choose(raw, req, [51, 92], "behind")
    kwargs = dict(passenger_landmarks=True, pair_geometry=weights, road_cache=cached)
    assert bound_outcome(out, req, **kwargs)["passenger_geometry_verification"][
        "learned_geometry_and_mask_replayed"
    ]
    journal_only = bound_outcome(out, req, passenger_landmarks=True)["passenger_geometry_verification"]
    assert (
        journal_only["current_image_and_tracks_checked"]
        and not journal_only["learned_geometry_and_mask_replayed"]
    )
    changed = deepcopy(out)
    changed["provenance"]["road_geometry"]["road_mask_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="replay"):
        bound_outcome(changed, req, **kwargs)
    with pytest.raises(FileNotFoundError):
        bound_outcome(out, req, **dict(kwargs, road_cache=tmp_path / "empty"))
    not_between = pc.choose(raw, req, [51, 92], "between")
    verified = bound_outcome(not_between, req, **dict(kwargs, road_cache=tmp_path / "empty"))[
        "passenger_geometry_verification"
    ]
    assert verified["decision_reproduced"] and not verified["road_mask_needed"]
    weights.write_text(weights.read_text() + " ")
    with pytest.raises(ValueError):
        bound_outcome(out, req, **kwargs)
