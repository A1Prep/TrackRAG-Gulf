"""Controlled images and native-response doubles exercise the optional projected-pair route."""

from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
import threading

import numpy as np
from PIL import Image
import pytest

from trackrag.dropoff import projected_pair as method
from trackrag.dropoff.landmarks import LandmarkResolver, resolve_image
from trackrag.dropoff.pair_geometry import fit
from trackrag.dropoff.pair_resolution import collect_pair_evidence
from trackrag.dropoff.passenger import PassengerGrounder, digest
from trackrag.dropoff.road_cache import mask_shape


TEXT = "The road just behind the gap between the small car on the left and the bus immediately to its right."


@pytest.mark.parametrize(
    "text",
    [
        TEXT,
        "Between the red van and the waiting bus, just behind them.",
        "Mark the road behind this opening between a gray sedan and the blue truck.",
    ],
)
def test_requires_a_current_scoped_behind_destination(text):
    quoted = method.scoped_quotes(text)
    assert all(text[q["start"] : q["end"]] == q["text"] for q in quoted)


@pytest.mark.parametrize(
    "text",
    [
        "Stop between the car and the bus.",
        "Behind those two cars.",
        "Between the sedan behind the bus and the white car.",
        "Between the tree near the car and the bus, just behind them.",
        "Behind the gap between the gray car and the sign on the bus.",
        "Behind the opening between the red car and the van, in front of them.",
        "Do not park behind the gap between the car and the bus.",
        "Behind the gap between a black car and the black car.",
        "Behind the gap between the car and the van; actually by the gate.",
        "Between the sedan and the bus, behind the bus.",
        "Behind the barrier between the car and the bus.",
    ],
)
def test_abstains_on_single_or_uncertain_relations(text):
    with pytest.raises(method.Ineligible):
        method.scoped_quotes(text)


def fitted_head():
    examples = []
    for i in range(35):
        a = [112 + i % 5, 68, 145 + i % 5, 100]
        b = [188 + i % 9, 66, 224 + i % 9, 100]
        point = [a[2] + 0.70 * (b[0] - a[2]), 109]
        examples.append(
            {
                "anchors": [a, b],
                "size": [360, 180],
                "point": point,
                "box": [point[0] - 18, 101, point[0] + 18, 123],
            }
        )
    return fit(examples)


def obs(key, box, name="Car"):
    return {
        "track_id": key,
        "bbox_xyxy": box,
        "class_name": name,
        "recording_id": "fixture-clip",
        "frame_id": 8,
        "source": "predicted",
    }


@pytest.fixture
def make(tmp_path, monkeypatch):
    picture = tmp_path / "software.png"
    Image.new("RGB", (360, 180), "navy").save(picture)
    cfg = SimpleNamespace(
        model_id="fixture-qwen",
        revision="fixture",
        adapter_dir=None,
        max_image_bytes=500000,
        max_original_pixels=360 * 180,
    )
    backend = {"model_id": cfg.model_id, "revision": cfg.revision, "runtime": "streamed", "fixture": True}

    # Only this temporary software test can stand in for a spawned GPU worker.
    # The shipping callable always refuses this marker, exercised separately below.
    def check_software_worker(record, config):
        assert config is cfg
        method.need(record.get("backend_details") == backend, "raw_native_backend_unverified")

    monkeypatch.setattr(method, "native_record", check_software_worker)
    shape = mask_shape((360, 180))
    model = object.__new__(PassengerGrounder)
    model.config, model.lock = cfg, threading.RLock()
    model.geometry = SimpleNamespace(sha256=digest({"software_weights": True}), pair_regression=fitted_head())
    monkeypatch.setattr(method, "GEOMETRY", model.geometry.sha256)
    model.road = SimpleNamespace(mask=lambda image, photo: np.ones(shape, dtype=bool))
    model.verify_landmarks = False
    model.referent_ranker = None
    a, b = obs(51, [130, 68, 171, 100]), obs(92, [166, 68, 207, 100], "Bus")
    frame = {
        "width": 360,
        "height": 180,
        "frame_id": 8,
        "recording_id": "fixture-clip",
        "timestamp_s": 0.7,
        "image_path": str(picture),
    }
    request = {
        "request_id": "software-pair",
        "text": TEXT,
        "input_mode": "typed",
        "dialogue": [],
        "scene": {
            "frame": frame,
            "source": "predicted",
            "tracks": [a, b],
            "provenance": {"image_sha256": sha256(picture.read_bytes()).hexdigest()},
        },
    }

    def build(*, whole="LANDMARK L1 L2", first="LANDMARK L1", second="LANDMARK L2", update=None):
        r = deepcopy(request)
        if update:
            update(r)
        selected = iter((whole, first, second))
        resolver = object.__new__(LandmarkResolver)
        resolver.config, resolver.adapter = cfg, None
        resolver.backend = SimpleNamespace(
            generate=lambda *a, **kw: SimpleNamespace(text=next(selected), details=deepcopy(backend)),
            close=lambda: None,
        )
        model.resolver = resolver
        image, binding = resolve_image(r, cfg)
        with image:
            record = resolver.resolve(r, image)
            record["pair_resolution"] = collect_pair_evidence(r, image, resolver)
            raw = model.from_decision(r, image, binding, record)
        return r, raw, model, backend

    return build


def test_uses_learned_pair_then_actual_supported_cells_and_requires_verified_new_visual_call(make, monkeypatch):
    r, raw, m, backend = make()
    assert raw["result"]["status"] == "needs_clarification"
    before = digest(raw)
    untouched, report = method.run(r, raw, m)
    assert untouched is raw and report["status"] == "eligible_for_crop_verification"
    assert (
        report["verification_attempted"] is False
        and report["candidate"]["geometry_sha256"] == m.geometry.sha256
    )
    geometry = report["candidate"]
    a, b = geometry["observations"]
    assert [a["track_id"], b["track_id"]] == [51, 92]
    assert geometry["point"][1] > max(a["bbox_xyxy"][3], b["bbox_xyxy"][3])
    assert geometry["point"][0] != pytest.approx(
        (method.center_x(a["bbox_xyxy"]) + method.center_x(b["bbox_xyxy"])) / 2
    )
    assert all(0 <= coord <= axis for coord, axis in zip(geometry["box"], (360, 180, 360, 180)))

    def verified(messages, **options):
        assert messages[1]["content"][0]["image"].size == (672, 384)
        return SimpleNamespace(text="MATCH", details=backend)

    m.resolver.backend.generate = verified
    active, report = method.run(r, raw, m, verify=True)
    assert (
        report["status"] == "new_proposal_requires_passenger_confirmation" and report["verification_attempted"]
    )
    assert active["result"]["anchor_boxes_xyxy"] == [a["bbox_xyxy"], b["bbox_xyxy"]]
    assert active["provenance"]["original_result"] == raw["result"] and digest(raw) == before
    receipt = active["provenance"]["projected_pair"]["candidate"]
    assert receipt["image_sha256"] == r["scene"]["provenance"]["image_sha256"]
    assert (
        receipt["native_pair_receipt_sha256"]
        == raw["provenance"]["resolver"]["pair_resolution"]["receipt_sha256"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "single",
        "different",
        "indeterminate",
        "absent",
        "broken_raw",
        "tampered_evidence",
        "foreign_image",
        "foreign_request",
        "wrong_geometry",
        "unknown_backend",
        "wrong_type",
        "reversed",
        "no_road",
        "later_track",
        "no_clear_human_relation",
        "different_original",
    ],
)
def test_original_stays_on_unclear_contradictory_or_stale_native_observations(make, change, monkeypatch):
    kwargs = {}
    if change == "single":
        kwargs["whole"] = "LANDMARK L1"
    if change == "different":
        kwargs["first"] = "LANDMARK L2"
    if change == "indeterminate":
        kwargs["whole"] = "ASK Which two vehicles?"
    if change == "absent":
        kwargs["first"] = "MISSING"
    if change == "wrong_type":
        kwargs["update"] = lambda r: r["scene"]["tracks"][0].update(class_name="Pedestrian")
    if change == "reversed":
        kwargs.update(
            whole="LANDMARK L1 L2",
            first="LANDMARK L2",
            second="LANDMARK L1",
            update=lambda r: r.update(
                text="Behind the gap between this bus and the car immediately to its right."
            ),
        )
    r, raw, m, _ = make(**kwargs)
    p = raw["provenance"]
    if change == "broken_raw":
        p["resolver"]["raw_text"] = "LANDMARK L9 L2"
    if change == "tampered_evidence":
        p["resolver"]["pair_resolution"]["subqueries"][0]["resolver"]["raw_text"] = "LANDMARK L2"
    if change == "foreign_image":
        p["image"]["sha256"] = "0" * 64
    if change == "foreign_request":
        raw["request_id"] = "somebody-else"
    if change == "wrong_geometry":
        p["geometry_sha256"] = "0" * 64
    if change == "unknown_backend":
        p["resolver"]["backend_details"]["runtime"] = "substituted"
    if change == "no_road":
        m.road.mask = lambda i, h: np.zeros(mask_shape(i.size), dtype=bool)
    if change == "later_track":
        r["scene"]["tracks"][0]["frame_id"] += 1
    if change == "no_clear_human_relation":
        r["text"] = "Tell me about the pair."
    if change == "different_original":
        Image.new("RGB", (360, 180), "red").save(r["scene"]["frame"]["image_path"])
    monkeypatch.setattr(
        method,
        "check_landmark",
        lambda *args: pytest.fail("unverified pair must not reach visual verification"),
    )
    unchanged, report = method.run(r, raw, m, verify=True)
    assert unchanged is raw and report["status"] in ("original_retained", "operational_error_original_retained")
    assert report["verification_attempted"] is False


@pytest.mark.parametrize(
    "response", ["MISSING", "ASK Which pair?", "LANDMARK L1 L2", "timeout", "foreign_model"]
)
def test_crop_can_only_admit_a_fresh_match(make, monkeypatch, response):
    r, raw, m, details = make()

    def generate(*args, **kwargs):
        if response == "timeout":
            raise TimeoutError("fixture")
        if response == "foreign_model":
            return SimpleNamespace(text="MATCH", details={**details, "revision": "unrelated"})
        return SimpleNamespace(text=response, details=details)

    m.resolver.backend.generate = generate
    unchanged, report = method.run(r, raw, m, verify=True)
    assert (
        unchanged is raw
        and report["verification_attempted"]
        and not report["status"].startswith("new_proposal")
    )


@pytest.mark.parametrize("field", ["render_sha256", "render_size", "chosen", "messages"])
def test_genuine_crop_receipt_must_reconstruct_every_binding(make, monkeypatch, field):
    r, raw, m, details = make()
    m.resolver.backend.generate = lambda *args, **kw: SimpleNamespace(text="MATCH", details=details)
    actual = method.check_landmark

    def switched(*args):
        decision, receipt = actual(*args)
        receipt[field] = None
        return decision, receipt

    monkeypatch.setattr(method, "check_landmark", switched)
    unchanged, report = method.run(r, raw, m, verify=True)
    assert unchanged is raw and report["status"] == "original_retained" and "verification_" in report["reason"]


@pytest.mark.parametrize("visual,expected", [("MATCH", "proposed"), ("MISSING", "needs_clarification")])
def test_opt_in_passenger_entry_requires_an_actual_confirmation_action(make, tmp_path, visual, expected):
    import json
    from trackrag.dropoff.interaction import PassengerSession

    req, _, m, details = make()
    answers = iter(["LANDMARK L1 L2", "LANDMARK L1", "LANDMARK L2", visual])
    m.resolver.backend.generate = lambda *args, **kw: SimpleNamespace(text=next(answers), details=details)
    session = PassengerSession(req["scene"], method.ProjectedPairGrounder(m), audit_dir=tmp_path / "journal")
    response = session.submit(TEXT)
    assert response["state"] == expected
    (journal,) = (tmp_path / "journal").glob("*.jsonl")
    native = json.loads(journal.read_bytes().splitlines()[-1])["detail"]["outcome"]
    assert native["provenance"]["pair_decision"]["kind"] == "ask"
    assert native["provenance"]["projected_pair"]["verification"]["raw_text"] == visual
    if visual == "MATCH":
        assert response["proposal_id"] and session.confirm(response["proposal_id"])["state"] == "confirmed"
    else:
        assert response["proposal_id"] is None


@pytest.mark.parametrize(
    "boxes,reason",
    [
        (([30, 20, 60, 60], [68, 22, 93, 60]), "overlap"),
        (([30, 20, 60, 60], [44, 21, 79, 63]), "overlap"),
        (([30, 20, 60, 60], [55, 55, 90, 95]), "depth"),
        (([0, 20, 30, 60], [28, 20, 58, 60]), "leave_the_image"),
    ],
)
def test_rejects_unrelated_depth_near_duplicates_or_nonoverlap(boxes, reason):
    current = [obs(11, boxes[0]), obs(12, boxes[1])]
    with pytest.raises(method.Ineligible, match=reason):
        method.neighbor_geometry(current, current, (100, 100))


def test_geometry_excludes_an_intervening_vehicle_and_road_holes():
    a, b = obs(1, [130, 68, 171, 100]), obs(2, [166, 68, 207, 100])
    middle = obs(3, [164, 71, 175, 94])
    with pytest.raises(method.Ineligible, match="another_current"):
        method.neighbor_geometry([a, b], [a, b, middle], (360, 180))
    geometry = method.neighbor_geometry([a, b], [a, b], (360, 180))
    mask = np.ones(mask_shape((360, 180)), dtype=bool)
    mask[:, 240:245] = False
    avoid = obs(3, [170, 102, 180, 115], "Pedestrian")
    proposal = method.supported_patch(mask, (360, 180), fitted_head(), geometry, [a, b, avoid])
    x, y, right, bottom = proposal["box"]
    h, w = mask.shape
    assert mask[round(y * h / 180) : round(bottom * h / 180), round(x * w / 360) : round(right * w / 360)].all()
    assert right <= 170 or x >= 180 or bottom <= 102 or y >= 115
    assert y >= 100 and method.center_x(a["bbox_xyxy"]) <= x < right <= method.center_x(b["bbox_xyxy"])


@pytest.mark.parametrize(
    "contradiction",
    [
        "fixture",
        "substitute",
        "injected",
        "software_fixture",
        "cpu",
        "inprocess",
        "pid",
        "wrong_weights",
        "missing_metadata",
        "unverified",
        "wrong_revision",
    ],
)
def test_production_gate_refuses_any_non_native_source(contradiction):
    config = SimpleNamespace(model_id=method.STREAMED_MODEL_ID, revision=method.STREAMED_REVISION)
    p = {**method.METADATA_SHA256, **method.WEIGHT_SHA256}
    identity = {
        "model_id": config.model_id,
        "revision": config.revision,
        "verified_weights": True,
        "verified_metadata": True,
        "files": {k: {"expected_sha256": v, "sha256": v, "bytes": 1} for k, v in p.items()},
    }
    details = {
        "runtime": "streamed",
        "native_fault_isolation": "spawn_process",
        "worker_pid": 7,
        "device": "cuda:0",
        "model_id": config.model_id,
        "revision": config.revision,
        "weight_identity": identity,
    }
    record = {"backend_details": details}
    if contradiction in ("fixture", "software_fixture"):
        details[contradiction] = True
    elif contradiction == "substitute":
        record["substitute"] = True
    elif contradiction == "injected":
        identity["injected"] = True
    elif contradiction == "cpu":
        details["device"] = "cpu"
    elif contradiction == "inprocess":
        details["native_fault_isolation"] = "not_isolated"
    elif contradiction == "pid":
        details["worker_pid"] = False
    elif contradiction == "wrong_weights":
        next(iter(identity["files"].values()))["sha256"] = "0" * 64
    elif contradiction == "missing_metadata":
        identity["files"].pop("config.json")
    elif contradiction == "unverified":
        identity["verified_weights"] = False
    elif contradiction == "wrong_revision":
        identity["revision"] = "somewhere-else"
    with pytest.raises(method.Ineligible, match="raw_native"):
        method.native_record(record, config)
