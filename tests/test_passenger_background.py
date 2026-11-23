"""Browser requests stay responsive and cannot display an inference overtaken by a passenger action."""
import json
from threading import Event
from time import monotonic, sleep

import pytest

from tests.test_passenger_app import site
from trackrag import passenger_app
from trackrag.dropoff.passenger import proposed
from trackrag.dropoff.passenger_pair_choice import PassengerSession
from tools.dropoff_passenger_handoff import session as export_session


def start(site, monkeypatch, *, voice=False):
    site.session_factory = PassengerSession
    active, release = Event(), Event()
    def delayed(req):
        active.set()
        assert release.wait(6)
        return site.model.result(req,proposed([50,35],[43,31,55,39]))
    monkeypatch.setattr(site.model,"interpret",delayed)
    monkeypatch.setattr(site.model,"passenger_area",lambda *_:None,raising=False)
    view=site.open(next(iter(site.scenes)))
    sid=view["session"]
    if voice:
        view=site.change(sid,0,"begin-audio",{})
        view=site.change(sid,view["revision"],"audio",b"original test recording")
    text="Behind the black car"
    state=site.change(sid,view["revision"],"submit",{"text":text,"mode":"voice" if voice else "typed","background":True})
    assert state["pending_grounding"]["text"]==text and state["view"]["proposal_id"] is None
    assert state["voice_stage"] is None and active.wait(2)
    return sid, state, release


def wait_for_result(site,sid):
    end=monotonic()+4
    while monotonic()<end:
        session=site.lookup(sid)
        with session.lock:
            view=session.snapshot()
        if view["pending_grounding"] is None:
            return view
        sleep(.01)
    pytest.fail("The browser never received a terminal response")


@pytest.mark.parametrize("finish_first",[False,True])
def test_reviewed_voice_can_take_over_in_either_completion_order(site,monkeypatch,finish_first):
    sid,pending,release = start(site,monkeypatch,voice=True)
    working=site.lookup(sid).pending
    try:
        if finish_first:
            release.set()
            assert wait_for_result(site,sid)["view"]["state"]=="proposed"
        marked=site.change(sid,pending["revision"],"take-over-area",{"request_id":working.request["request_id"]})
        assert marked["view"]["selected_by"]=="passenger" and marked["view"]["state"]=="needs_clarification"
        assert marked["pending_grounding"] is None and marked["view"]["proposal_id"] is None
    finally:
        release.set()
        working.future.result(3)
    assert site.lookup(sid).snapshot()==marked
    journal,=site.lookup(sid).directory.glob("*.jsonl")
    record=json.loads(journal.read_bytes().splitlines()[-1])
    assert record["detail"]["request"]["input_mode"]=="voice"
    assert record["detail"]["request"]["original_transcript"]=="near the read bus"
    assert record["detail"]["request"]["text"]=="Behind the black car" and record["detail"]["speech"]["audio_sha256"]


def test_pending_cancel_and_navigation_discard_late_answers(site,monkeypatch):
    sid,pending,release=start(site,monkeypatch)
    item=site.lookup(sid)
    job=item.pending
    try:
        with pytest.raises(passenger_app.PassengerRequestError,match="current photo check"):
            site.change(sid,pending["revision"],"take-over-area",{"request_id":"wrong"})
        cancelled=site.change(sid,pending["revision"],"cancel",{"request_id":job.request["request_id"]})
        assert cancelled["view"]["state"]=="cancelled"
        other=site.open(list(site.scenes)[1],sid)
        assert other["view"]["state"]=="idle"
    finally:
        release.set()
        job.future.result(3)
    assert site.lookup(other["session"]).snapshot()==other and item.snapshot()==cancelled
    journal,=item.directory.glob("*.jsonl")
    assert export_session(journal)["events"][-1]["phase"]=="cancelled"


def test_ordinary_navigation_during_computation_does_not_confirm_or_cancel_for_passenger(site,monkeypatch):
    sid,_,release=start(site,monkeypatch)
    prev=site.lookup(sid)
    job=prev.pending
    try:
        current=site.open(list(site.scenes)[1],sid)
    finally:
        release.set()
        job.future.result(3)
    assert site.lookup(current["session"]).snapshot()==current
    journal,=prev.directory.glob("*.jsonl")
    assert [line["action"] for line in map(json.loads,journal.read_bytes().splitlines())]==["opened"]


def test_interactive_deadline_preserves_words_and_marking_without_an_automatic_pin(site,monkeypatch):
    monkeypatch.setattr(passenger_app,"INTERACTIVE_DEADLINE_SECONDS",.25)
    sid,ack,release=start(site,monkeypatch)
    job=site.lookup(sid).pending
    try:
        expired=wait_for_result(site,sid)
        assert expired["view"]["error"]["code"]=="timeout"
        assert "mark" in expired["view"]["landmark_error_guidance"] and expired["view"]["proposal_id"] is None
        marked=site.change(sid,ack["revision"],"take-over-area",{"request_id":job.request["request_id"]})
        assert marked["view"]["selected_by"]=="passenger"
    finally:
        release.set()
        job.future.result(3)
    assert site.lookup(sid).snapshot()==marked


def test_cloud_grounding_does_not_hold_the_laptop_model_lock(site,monkeypatch):
    monkeypatch.setattr(site.model,"uses_public_service",lambda req:True,raising=False)
    sid,_,release=start(site,monkeypatch)
    job=site.lookup(sid).pending
    try:
        assert site.model_lock.acquire(timeout=.2)
        site.model_lock.release()
    finally:
        release.set()
        job.future.result(3)
    assert wait_for_result(site,sid)["view"]["state"]=="proposed"
