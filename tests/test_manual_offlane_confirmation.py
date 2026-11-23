"""The live dialogue and exported journal must agree on the passenger's exact selected point."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from tools.dropoff_passenger_handoff import bound_outcome, session as export_session
from trackrag.dropoff import manual_offlane
from trackrag.dropoff.interaction import PassengerSession
from trackrag.dropoff.passenger import digest
from trackrag.dropoff import passenger_pair_choice
from trackrag.passenger_app import PassengerSite
from test_passenger_manual_offlane import marked_scene


def open_mark_session(tmp_path, request, grounder=manual_offlane.select):
    model = SimpleNamespace(passenger_area=grounder, requires_verified_manual_surface=True)
    session = PassengerSession(request["scene"], model, audit_dir=tmp_path/"journal")
    assert session.submit(request["text"], mark_on_photo=True)["state"] == "needs_clarification"
    return session


def test_mark_token_replays_the_same_offlane_receipt(marked_scene, tmp_path):
    request, _ = marked_scene
    session = open_mark_session(tmp_path, request)
    box = [178, 116, 183, 120]
    offered = session.select_area(box)
    assert offered["result"]["destination_point_xy"] == [180.5, 118]
    assert offered["selected_by"] == "passenger"
    _, receipt, receipt_hash = session._manual_offer
    assert receipt_hash == digest(receipt)
    with pytest.raises(ValueError):
        session._accept(receipt, "model")
    with pytest.raises(ValueError):
        bound_outcome(receipt, session._request, dialogue_policy=True)
    with pytest.raises(ValueError):
        bound_outcome(receipt, session._request, passenger_selected=True, requested_box=[180, 116, 183, 120])
    assert session.confirm(offered["proposal_id"])["state"] == "confirmed"
    journal, = (tmp_path/"journal").glob("*.jsonl")
    export = export_session(journal)["events"]
    assert export[-2]["outcome"]["passenger_mark_verification"]["offlane_mask_replayed"]
    assert export[-1]["verified_manual_outcome_sha256"] == receipt_hash
    rows = [json.loads(line) for line in journal.read_bytes().splitlines()]
    del rows[-1]["detail"]["verified_manual_outcome_sha256"]
    journal.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="verified parking evidence"):
        export_session(journal)


def test_lane_and_changed_receipt_cannot_be_confirmed(marked_scene, tmp_path):
    request, _ = marked_scene
    session = open_mark_session(tmp_path, request)
    box = [178, 116, 183, 120]
    first = session.select_area(box)
    lane = session.select_area([98, 110, 103, 115])
    assert lane["state"] == "needs_clarification" and lane["proposal_id"] is None
    with pytest.raises(ValueError):
        session.confirm(first["proposal_id"])
    offered = session.select_area(box)
    session._manual_offer[1]["result"]["destination_point_xy"][0] += 1
    with pytest.raises(ValueError, match="evidence"):
        session.confirm(offered["proposal_id"])
    assert session.view()["state"] == "error"
    assert session.view()["proposal_id"] is None


def test_old_road_only_method_cannot_offer_a_live_manual_pin(marked_scene, tmp_path):
    request, _ = marked_scene

    def old_method(req, box):
        outcome = deepcopy(manual_offlane.select(req, box))
        outcome["provenance"]["method"] = "explicit_passenger_region"
        return outcome

    session = open_mark_session(tmp_path, request, old_method)
    refused = session.select_area([178, 116, 183, 120])
    assert refused["state"] == "error"
    assert refused["result"] is None and refused["proposal_id"] is None


def test_public_mark_can_finish_while_the_inference_worker_is_busy(marked_scene, tmp_path, monkeypatch):
    request, _ = marked_scene
    class BusyModelLock:
        def __enter__(self):
            pytest.fail("A precomputed parking check waited for an unrelated inference worker.")

        def __exit__(self, *_):
            pass

    model = SimpleNamespace(passenger_area=manual_offlane.select, requires_verified_manual_surface=True,
                            lock=BusyModelLock())
    site = PassengerSite({"original": request["scene"]}, model, None, tmp_path/"sessions",
                         passenger_pair_choice.PassengerSession)
    monkeypatch.setattr(site, "model_request", lambda: pytest.fail("Manual marks must bypass the model queue."))
    opened = site.open("original")
    sid = opened["session"]
    ready = site.change(sid, opened["revision"], "submit",
                        {"text":"By the shop.", "mode":"typed", "mark_on_photo":True})
    marked = site.change(sid, ready["revision"], "area", {"box":[178, 116, 183, 120]})
    assert marked["view"]["state"] == "proposed"
    assert marked["view"]["result"]["destination_point_xy"] == [180.5, 118]


def test_operational_manual_failure_is_preserved_in_the_export(marked_scene, tmp_path):
    from trackrag.dropoff.passenger import PassengerGrounder
    request, _ = marked_scene
    def production_area(req, box):
        return PassengerGrounder.passenger_area(None, req, box)
    session = open_mark_session(tmp_path, request, production_area)
    rejected = session.select_area(None)
    assert rejected["state"] == "error" and rejected["proposal_id"] is None
    journal, = (tmp_path/"journal").glob("*.jsonl")
    last = export_session(journal)["events"][-1]
    assert last["outcome"]["ok"] is False and last["outcome"]["error"] == rejected["error"]
