"""Exercise role, pin and journal binding at the session boundary with test-only signed packets."""
from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools.dropoff_passenger_handoff import bound_outcome, session as export
from trackrag.dropoff import parking_evidence
from trackrag.dropoff.interaction import signed_place_verifier
from trackrag.dropoff.passenger import digest, proposed
from trackrag.dropoff.passenger_pair_choice import PassengerSession
from tests.test_passenger_place_grounding import case


def test_real_specialist_verifiers_reject_fabricated_pins_and_replay_a_withheld_vehicle(tmp_path,monkeypatch):
    from trackrag.dropoff import frontage_evidence, single_vehicle
    req,photo,_,_ = case(tmp_path,monkeypatch)
    assert signed_place_verifier(frontage_evidence.METHOD) is frontage_evidence.verify
    assert signed_place_verifier(single_vehicle.METHOD) is single_vehicle.verify
    assert signed_place_verifier(single_vehicle.METHOD,live=True) is single_vehicle.verify_current
    req["text"] = "Drop me behind the white sedan."
    missing = single_vehicle.outcome(req,photo)
    checked = bound_outcome(missing,req,dialogue_policy=True)
    assert checked["method"] == single_vehicle.METHOD and checked["native_model_calls"] == 0
    assert checked["result"]["status"] == "needs_clarification" and not checked["independent_model_proposal"]
    for method in (frontage_evidence.METHOD,single_vehicle.METHOD):
        changed=deepcopy(missing)
        changed["provenance"]["method"]=method
        changed["result"]=proposed([70,45],[68,43,72,47])
        with pytest.raises((KeyError,ValueError)):
            bound_outcome(changed,req,dialogue_policy=True)


def test_archived_vehicle_decision_can_be_replayed_but_live_offers_and_confirmation_require_current_policy(tmp_path,monkeypatch):
    from tests.test_passenger_single_vehicle import make_receipt
    from trackrag.dropoff import single_vehicle

    req,photo,packet,_ = make_receipt(tmp_path,monkeypatch)
    def old_policy(request):
        return single_vehicle.outcome(request,photo,packet,policy=single_vehicle.PREVIOUS_POLICY)

    historical = bound_outcome(old_policy(req),req,dialogue_policy=True)
    assert historical["independent_model_proposal"]
    old_session = PassengerSession(req["scene"],SimpleNamespace(interpret=old_policy),audit_dir=tmp_path/"old")
    old_view = old_session.submit(req["text"])
    assert old_view["state"] == "error" and old_view["proposal_id"] is None

    session = PassengerSession(req["scene"],SimpleNamespace(interpret=lambda r:single_vehicle.outcome(r,photo,packet)),
                               audit_dir=tmp_path/"current")
    pending = session.submit(req["text"])
    assert pending["state"] == "proposed"
    monkeypatch.setattr(single_vehicle,"POLICY","newer-policy")
    with pytest.raises(ValueError,match="evidence"):
        session.confirm(pending["proposal_id"])
    assert session.view()["proposal_id"] is None


@pytest.mark.parametrize(("method","module"), [
    ("molmo2_frontage_confidence_v1","frontage_evidence"),
    ("single_vehicle_behind_parking_v1","single_vehicle"),
])
def test_signed_model_dispatch_confirmation_and_role_replay(tmp_path, monkeypatch, method, module):
    sample, photo, _, key = case(tmp_path, monkeypatch)
    called = []

    def issue(request):
        _, image_sha, input_sha = parking_evidence.source_input(request, photo)
        target = proposed([70,45], [68,43,72,47])
        body = {"software_fixture":True, "method":method, "request":digest(request), "result":target}
        return {"ok":True, "request_id":request["request_id"], "result":target, "provenance":{
            "method":method, "image":{"sha256":image_sha}, "request_sha256":digest(request),
            "inference_input_sha256":input_sha, "independent_model_proposal":True,
            "proof":{"body":body,"signature":key.sign(parking_evidence.signed_bytes(body)).hex()}}}

    def verify(request, outcome):
        parking_evidence._authenticate(outcome["provenance"]["proof"])
        if outcome != issue(request):
            raise ValueError("Signed software proposal changed")
        called.append(request["request_id"])
        return {"input_verified":True, "decision_reproduced":True, "language_calls":2, "independent_model_proposal":True}

    # This test replaces the model-specific verifier only. The registry, dialogue, replay
    # and confirmation use production code. Separate feature tests exercise each model's evidence.
    software_module = ModuleType("trackrag.dropoff."+module)
    software_module.verify = verify
    software_module.verify_current = verify
    monkeypatch.setitem(sys.modules,software_module.__name__,software_module)
    assert signed_place_verifier(method) is verify
    grounder = SimpleNamespace(interpret=issue, passenger_area=lambda *_:pytest.fail("Unexpected area inference"))
    session = PassengerSession(sample["scene"],grounder,audit_dir=tmp_path/"journal")
    pending = session.submit(sample["text"])
    assert pending["state"] == "proposed" and pending["selected_by"] == "model"
    original = deepcopy(session._place_offer)
    for actor in ("passenger","passenger_landmarks"):
        with pytest.raises(ValueError):
            session._accept(original,actor)
    with pytest.raises(ValueError):
        bound_outcome(original,session._request,passenger_selected=True,dialogue_policy=True)
    forged = deepcopy(original)
    forged["provenance"]["proof"]["body"]["method"] = "other"
    with pytest.raises(ValueError):
        session._accept(forged,"model")
    assert session.confirm(pending["proposal_id"])["state"] == "confirmed"
    # Explicit area selection is a new human turn and must not inherit an automatic pin.
    prior_count = len(called)
    direct = session.submit("I will mark my spot.",mark_on_photo=True)
    assert direct["state"] == "needs_clarification" and direct["selected_by"] == "passenger"
    assert direct["proposal_id"] is None and len(called) == prior_count
    journal, = (tmp_path/"journal").glob("*.jsonl")
    events = export(journal)["events"]
    assert events[1]["outcome"]["method"] == method and events[2]["verified_place_outcome_sha256"] == digest(original)
    assert events[3]["phase"] == "explicit_passenger_area_request" and events[3]["outcome"]["native_model_calls"] == 0
    rows = list(map(json.loads,journal.read_text(encoding="utf-8").splitlines()))
    rows[2]["detail"]["verified_place_outcome_sha256"] = "0"*64
    journal.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
    with pytest.raises(ValueError,match="signed decision"):
        export(journal)
    replaced = session.submit("Beside the store.",replaces_request_id=direct["request_id"])
    session._place_offer["provenance"]["method"] = "another signed method"
    with pytest.raises(ValueError,match="evidence"):
        session.confirm(replaced["proposal_id"])
    assert session.view()["proposal_id"] is None
