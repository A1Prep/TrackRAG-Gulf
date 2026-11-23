"""Software regressions for a cold public-place service, with locally signed fixture evidence."""
from copy import deepcopy
from hashlib import sha256
import json
from threading import Event
from types import SimpleNamespace

import pytest

from tests.test_passenger_place_grounding import case, packet
from tools.dropoff_passenger_handoff import bound_outcome
from trackrag.dropoff import frontage_evidence, place_evidence
from trackrag.dropoff.passenger_pair_choice import PassengerSession
from trackrag.dropoff.place_lookup import PlaceLookups


def service(tmp_path, monkeypatch, *, disk=True):
    request, photo, classes, private = case(tmp_path,monkeypatch)
    manifest = tmp_path/"public.json"
    manifest.write_text(json.dumps({"photos":[{"photo_url":"https://kartaview.org/details/42/2/",
                                               "sha256":sha256(photo).hexdigest()}]}),encoding="utf8")
    backend = place_evidence.PublicPlaceBackend([manifest],python="unused",cache=tmp_path/"cache" if disk else None)
    monkeypatch.setattr(place_evidence,"INTERACTIVE_WAIT_SECONDS",.04)
    return request,photo,packet(request,photo,classes,private),backend


def running(backend, request, photo):
    with backend.lookups._lock:
        return backend.lookups._running[backend.repeat_key(request,sha256(photo).hexdigest())]


@pytest.mark.parametrize("disk",[True,False])
def test_slow_place_deduplicates_then_rebinds_verified_receipt(tmp_path,monkeypatch,disk):
    req,photo,receipt,backend=service(tmp_path,monkeypatch,disk=disk)
    entered,release=Event(),Event()
    calls=[]
    def bridge(*args,**kwargs):
        calls.append(json.loads(kwargs["input"])["input_sha256"]); entered.set()
        assert release.wait(6)
        return SimpleNamespace(returncode=0,stdout=json.dumps(receipt).encode())
    monkeypatch.setattr(place_evidence.subprocess,"run",bridge)
    later={**req,"request_id":"fresh-turn"}
    try:
        waiting=backend.interpret(req)
        assert entered.wait(2)
        assert waiting["result"]["status"]=="needs_clarification" and waiting["result"]["destination_point_xy"] is None
        checked=bound_outcome(waiting,req,dialogue_policy=True)
        assert checked["native_model_calls"]==0 and checked["place_verification"]["lookup_status"]=="pending"
        assert not checked["independent_model_proposal"]
        with pytest.raises(ValueError):
            bound_outcome(waiting,req,passenger_selected=True)
        duplicate=backend.interpret(later)
        assert duplicate["request_id"]==later["request_id"] and duplicate["provenance"]["operational_status"]=="pending"
        job=running(backend,req,photo)
    finally:
        release.set()
    job.result(3)
    actual=backend.interpret(later)
    assert actual["result"]["status"]=="proposed" and actual["provenance"]["reused_source_request_id"]==req["request_id"]
    assert place_evidence.verify(later,actual)["current_language_calls"]==0 and len(calls)==1
    if disk:
        saved=backend.cache_path(later,sha256(photo).hexdigest())
        assert json.loads(saved.read_bytes())=={"request_id":req["request_id"],"packet":receipt}
    # A caller can mutate its own copy without changing the stored evidence.
    actual["provenance"]["place_evidence"]["signature"]="0"*128
    assert place_evidence.verify(later,backend.interpret(later))["decision_reproduced"]


def test_capacity_is_honest_and_a_late_transport_error_is_reported(tmp_path,monkeypatch):
    req,photo,_,backend=service(tmp_path,monkeypatch)
    backend.lookups=PlaceLookups(capacity=1)
    release=Event();calls=[]
    def unavailable(*args,**kwargs):
        calls.append(1); assert release.wait(6)
        return SimpleNamespace(returncode=1,stdout=b"")
    monkeypatch.setattr(place_evidence.subprocess,"run",unavailable)
    try:
        waiting=backend.interpret(req)
        assert waiting["provenance"]["operational_status"]=="pending"
        different={**req,"request_id":"other","text":"By the bakery on the left."}
        busy=backend.interpret(different)
        assert place_evidence.verify(different,busy)["lookup_status"]=="busy"
        assert len(calls)==1 and not busy["result"]["destination_point_xy"]
        forged=deepcopy(busy); forged["result"]["destination_point_xy"]=[12,13]
        with pytest.raises(ValueError): place_evidence.verify(different,forged)
        with pytest.raises(ValueError): place_evidence.verify(req,busy)
        failed=running(backend,req,photo)
    finally:
        release.set()
    with pytest.raises(place_evidence.PlaceTransportUnavailable): failed.result(2)
    repeat={**req,"request_id":"retry"}
    error=backend.interpret(repeat)
    assert error["ok"] is False and error["error"]["code"]=="place_service_unavailable"
    assert place_evidence.verify_unavailable(repeat,error)["input_verified"] and len(calls)==1
    assert backend.interpret(repeat)["ok"] is False and len(calls)==2


def test_pending_same_session_retry_keeps_its_original_dialogue(tmp_path,monkeypatch):
    req,photo,_,backend=service(tmp_path,monkeypatch)
    release=Event()
    model=SimpleNamespace(interpret=backend.interpret,passenger_area=lambda *_:None)
    current=PassengerSession(req["scene"],model,audit_dir=tmp_path/"history")
    # Use the genuine bound implementation with its generated first-turn ID.
    keys=[]
    def bridge(*args,**kwargs):
        keys.append(json.loads(kwargs["input"])["input_sha256"]); assert release.wait(6)
        return SimpleNamespace(returncode=1,stdout=b"")
    monkeypatch.setattr(place_evidence.subprocess,"run",bridge)
    try:
        initial=current.submit(req["text"])
        assert initial["state"]=="needs_clarification" and initial["can_request_area"] and not initial["proposal_id"]
        request=current.prepare_automatic(req["text"])
        assert request["dialogue"]==[]
        assert backend.repeat_key(req,sha256(photo).hexdigest())==backend.repeat_key(request,sha256(photo).hexdigest())
        changed=current.prepare_automatic("Actually, next to the bakery.")
        assert changed["dialogue"] and changed["dialogue"][-1]["role"]=="assistant"
        check=backend.interpret(request)
        assert check["provenance"]["operational_status"]=="pending" and len(keys)==1
        job=running(backend,req,photo)
    finally:
        release.set()
    with pytest.raises(place_evidence.PlaceTransportUnavailable): job.result(3)


def test_frontage_uses_only_time_left_after_primary(tmp_path,monkeypatch):
    from tests.test_passenger_frontage_evidence import trial

    req,_,prior,combined,supplement=trial(tmp_path,monkeypatch)
    time=[50.0];release=Event();entered=Event()
    def slow_primary(request):
        time[0]+=10
        return prior(request)
    def fetch(*args):
        entered.set();assert release.wait(4); return combined
    supplement.dialogue.interpret=slow_primary
    monkeypatch.setattr(frontage_evidence,"monotonic",lambda:time[0])
    monkeypatch.setattr(supplement,"_fetch",fetch)
    try:
        assert supplement.interpret(req)==prior(req)
        assert entered.wait(2)
        with supplement._pending_lock: job=next(iter(supplement._pending.values()))
    finally:
        release.set()
    assert job.result(2)==combined
