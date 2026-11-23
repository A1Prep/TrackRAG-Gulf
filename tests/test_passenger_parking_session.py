"""Software tests for dispatch and confirmation. No parking weights or cloud service are exercised here."""
from copy import deepcopy
from functools import partial
import json

import pytest

from tests.test_passenger_app import StubASR
from tests.test_passenger_between_road import roadside
from tools.dropoff_passenger_handoff import bound_outcome, session as export
from trackrag.dropoff import passenger_parking as parking
from trackrag.dropoff.contracts import error_outcome
from trackrag.dropoff.infrastructure import PassengerDialogueGrounder
from trackrag.dropoff.passenger import digest, proposed, unresolved
from trackrag.dropoff.passenger_pair_choice import PassengerSession
from trackrag.dropoff.passenger_parking_session import ParkingPassengerSession
from trackrag.passenger_app import PassengerRequestError, PassengerSite


WORDS = "Drop me off between the nearer silver sedan and the white van."


def later(request, text, *, old=WORDS, assistant="Which two did you mean?"):
    frame = request["scene"]["frame"]
    history = [{"role":role, "text":s, "recording_id":frame["recording_id"], "frame_id":frame["frame_id"]}
               for role,s in (("user",old), ("assistant",assistant))]
    return {**request, "dialogue":history, "text":text}


@pytest.mark.parametrize("words", ["The white one on our right.", "Actually, the dark sedan, the one closest to us.",
                                   "I meant the blue van beside that gate.", "No, the rightmost one."])
def test_passenger_only_current_frame_wording_preserves_vehicle_corrections(roadside, words):
    req, *_ = roadside
    assert parking.parking_instruction(later(req,words)) == WORDS+"\n"+words
    revised = "Instead, drop me off between the grey pickup and the black van."
    assert parking.parking_instruction(later(req,revised)) == revised


@pytest.mark.parametrize("words", ["The gate by the sedan please.", "Actually, let me out near the store.",
                                   "Don't stop between those cars.", "Forget it.", "Thanks.", "Behind the blue van.",
                                   "Is the lane between the cars blocked?", "Drop me between the gate and the bus stop."])
def test_replacements_negations_and_other_tasks_return_to_native(roadside, words):
    req, *_ = roadside
    assert parking.parking_instruction(later(req, words)) is None


def test_no_cross_frame_or_assistant_authored_parking_intent(roadside):
    req, *_ = roadside
    assert parking.parking_instruction(later(req,"The white one on the right",old="Here",assistant=WORDS)) is None
    old = later(req,"The white one on our right.")
    old["dialogue"][0]["frame_id"] -= 1
    assert parking.parking_instruction(old) is None


@pytest.mark.parametrize("words,allowed",[
    ("Drop me off between the silver sedan and the gray GMC on the right.",True),
    ("Drop me in the gap between the white SUV and the white convertible on the right.",True),
    ("Stop between the blue Land Cruiser and the black roadster.",True),
    ("Drop me between the Rivian by the gate and the silver minibus.",True),
    ("Drop me between the new Zontra sedan and the Toyota.",True),
    ("Let me out between the orange and gray SUVs.",True),
    ("Drop us off between the white and maroon convertibles on the right, please.",True),
    ("Please stop between the orange and gray SUV.",False),
    ("Drop me between the red and green shops.",False),
    ("Drop me between the black sedan and the new Zontra.",False),
    ("Stop between a silver van and the Ford showroom.",False),
    ("Stop between the SUV and the GMC dealership on the right.",False),
    ("Stop between the truck and Mr. Ford.",False),
    ("Please don't stop between the gray GMC and the black convertible.",False),
    ("The road just behind the space between the BMW and the Subaru.",False),
])
def test_vehicle_types_and_marques_preserve_original_spans_and_exclude_venues(roadside,words,allowed):
    req, *_ = roadside
    assert parking.parking_instruction({**req, "text":words}) == (words if allowed else None)


@pytest.mark.parametrize("question", [
    "Is there a cyclist between the red van and the blue car?",
    "Can you see a person between the silver sedan and the black pickup?",
    "Are the people between the two red cars crossing?",
    "There is a person between the two cars.",
    "Is the white SUV on the right?",
    "The cyclist is standing between the two vans.",
    "Did the taxi stop between the two cars?",
    "Parking between the two vans is prohibited here.",
    "Stopping between the red and grey SUVs would block the driveway.",
    "Between the two cars there is a cyclist.",
    "Can you show me the cyclist in the gap between the two cars?",
    "Can you show me the store between the red van and the blue car?",
])
def test_scene_questions_and_observations_never_request_a_parking_proposal(roadside, question):
    req, *_ = roadside
    assert parking.parking_instruction({**req, "text": question}) is None
    assert parking.parking_instruction(later(req, question)) is None


@pytest.mark.parametrize("words", [
    "Could you please let us out between the sedan and the red van?",
    "I'd like to get out between the two black cars.",
    "Between the two grey vans, please.",
    "In the empty gap between the two blue cars.",
    "I see a white sedan and a red SUV. Could you drop me between those two?",
    "Is it possible to stop between the two dark cars?",
    "Would you mind stopping in the gap between the orange and grey sedans?",
    "Can you show me the curb gap between the cars?",
    "On the right mark the empty bit between the blue minivan and the gray sedan.",
    "There's a white SUV and a blue van. Could you show me the marked space between those two?",
    "Over on our left, show us the clear strip of road between both dark cars.",
])
def test_explicit_spoken_and_elliptical_destination_requests_still_route(roadside, words):
    req, *_ = roadside
    assert parking.parking_instruction({**req, "text": words}) == words


@pytest.fixture
def configured(roadside, tmp_path, monkeypatch):
    req, base, raw, mask = roadside
    class SoftwareTransport:
        def __init__(self):
            self.calls, self.checks, self.mode, self.revoke = [], [], "proposed", False
        def interpret(self, original):
            self.calls.append(deepcopy(original))
            if self.mode == "exception": raise ConnectionError("dispatch unavailable")
            source = {**parking.input_binding(original), "method":parking.METHOD, "software_fixture":True}
            if self.mode == "model_error":
                out = error_outcome("parking_backend_error", "The model returned an invalid response.", original["request_id"], source)
            else:
                targets = {"ask":unresolved("Which two vehicles bound the area?"),
                           "proposed":proposed([55.5,79.5],[54,78,57,81],[t["bbox_xyxy"] for t in original["scene"]["tracks"][:2]])}
                out = {"ok":True,"request_id":original["request_id"],"result":deepcopy(targets.get(self.mode,targets["proposed"])),"provenance":source}
            source = out["provenance"]
            source["test_fingerprint"] = digest(out.get("result",out.get("error")))
            if self.mode == "stale":
                source["image"]["sha256"] = "a"*64
            if self.mode == "wrong_actor":
                source["method"] = "explicit_passenger_region"
            return out
        def verify(self, request, outcome, *, artifacts=None):
            p = outcome["provenance"]
            if self.revoke or p.get("software_fixture") is not True or p["test_fingerprint"] != digest(outcome.get("result",outcome.get("error"))):
                raise ValueError("Software-only test receipt changed")
            self.checks.append(request["request_id"])
            affirmative = outcome["ok"] and outcome["result"]["status"] == "proposed"
            return {"input_verified":True, "decision_reproduced":outcome["ok"], "independent_model_proposal":affirmative}
    cloud = SoftwareTransport()
    monkeypatch.setattr("trackrag.dropoff.parking_evidence.verify", cloud.verify)
    model = PassengerDialogueGrounder(base)
    app = ParkingPassengerSession(req["scene"],model,audit_dir=tmp_path/"app",parking_backend=cloud)
    return req, model, app, cloud


def test_only_real_accepted_parking_turns_get_a_confirmable_independent_actor(configured):
    req, model, app, cloud = configured
    first = app.submit(WORDS)
    assert first["state"] == "proposed" and first["selected_by"] == parking.ACTOR
    first_token = first["proposal_id"]
    second = app.submit("Actually, the white vehicle on the right.")
    assert len(cloud.calls) == 2 and parking.parking_instruction(cloud.calls[-1]).endswith("on the right.")
    with pytest.raises(ValueError):
        app.confirm(first_token)
    assert app.confirm(second["proposal_id"])["selected_by"] == parking.ACTOR
    assert len(cloud.calls) == 2 and len(cloud.checks) == 3
    signed = export(app._journal)
    assert [r["phase"] for r in signed["events"]] == ["opened","automatic_parking_proposal","automatic_parking_proposal","confirmed_previous_selection"]
    assert signed["events"][1]["outcome"]["independent_model_proposal"] is True
    assert signed["events"][-1]["verified_parking_outcome_sha256"]
    assert signed["native_task1_condition"] is False and signed["actor_identity_verified"] is False


def test_ask_manual_recovery_and_its_confirmation_keep_their_distinct_sources(configured,monkeypatch):
    req, model, app, cloud = configured
    from tests.test_passenger_manual_offlane import provide_bay
    provide_bay(monkeypatch,req["scene"],[42,66,88,84])
    cloud.mode = "ask"
    ask = app.submit(WORDS)
    assert ask["state"] == "needs_clarification" and ask["proposal_id"] is None
    assert ask["selected_by"] == parking.ACTOR
    marked = app.select_area([47,70,81,79])
    assert marked["selected_by"] == "passenger"
    assert app.confirm(marked["proposal_id"])["state"] == "confirmed"
    signed = export(app._journal)
    assert signed["events"][1]["phase"] == "automatic_parking_response"
    assert signed["events"][1]["outcome"]["independent_model_proposal"] is False
    assert signed["events"][2]["phase"] == "passenger_marked"
    assert "verified_parking_outcome_sha256" not in signed["events"][-1]


@pytest.mark.parametrize("mode,code", [("exception","parking_backend_unavailable"),("stale","parking_evidence_invalid"),
                                        ("wrong_actor","parking_evidence_invalid"),("model_error","parking_backend_error")])
def test_parking_errors_allow_separate_assistance_and_keep_original_failure(configured,mode,code):
    req, model, app, cloud = configured
    cloud.mode = mode
    failed = app.submit(WORDS)
    assert failed["state"] == "error" and failed["error"]["code"] == code
    assert failed["result"] is None and failed["proposal_id"] is None and failed["parking_recovery"]
    chosen = app.select_pair([21,46],"between")
    assert chosen["selected_by"] == "passenger_landmarks"
    signed = export(app._journal)
    assert signed["events"][1]["phase"] == "parking_operational_error"
    assert signed["events"][1]["outcome"]["independent_model_proposal"] is False
    assert signed["events"][2]["phase"] == "passenger_selected_landmarks"


def test_confirm_revalidates_receipt_and_never_keeps_a_stale_token(configured):
    req, model, app, cloud = configured
    first = app.submit(WORDS)
    cloud.revoke = True
    with pytest.raises(ValueError, match="changed"):
        app.confirm(first["proposal_id"])
    assert app.view()["state"] == "error" and app.view()["proposal_id"] is None
    persisted = json.loads(app._journal.read_bytes().splitlines()[-1])
    assert persisted["action"] == "confirmation_blocked"


@pytest.mark.parametrize("tamper", ["invent_selector", "mark_after_cancel"])
def test_cold_export_rejects_marks_with_inactive_or_forged_provenance(configured, tmp_path, tamper):
    _, _, app, cloud = configured
    cloud.mode = "exception"
    assert app.submit(WORDS)["state"] == "error"
    assert app.select_area([47,70,81,79])["selected_by"] == "passenger"
    app.cancel()
    assert export(app._journal)["events"][2]["phase"] == "passenger_marked"
    opened, failed, marked, cancelled = [json.loads(s) for s in app._journal.read_bytes().splitlines()]
    rows = [opened, failed, marked, cancelled]
    if tamper == "invent_selector":
        failed["view"].update(selected_by="passenger", result=marked["view"]["result"])
    else:
        rows = [opened, failed, cancelled, marked]
    altered = tmp_path/"changed-interaction.jsonl"
    altered.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf8")
    with pytest.raises(ValueError):
        export(altered)


def test_manual_edits_do_not_route_through_external_service_and_native_elsewhere(configured, tmp_path):
    req, model, app, cloud = configured
    explicit = app.submit(WORDS,point_to_vehicles=True)
    assert explicit["selected_by"] == "passenger" and not cloud.calls
    app.cancel()
    model.interpret = lambda r: {"ok":True,"request_id":r["request_id"],"result":unresolved("Which entrance?"),
                                "provenance":{**parking.input_binding(r),"method":"ordinary_local_model"}}
    other = app.submit("Please drop me next to the entrance.")
    assert other["selected_by"] == "model" and other["state"] == "needs_clarification" and not cloud.calls
    disabled = PassengerSession(req["scene"],model,audit_dir=tmp_path/"off")
    disabled.model.interpret = cloud.interpret
    attempted = disabled.submit(WORDS)
    assert attempted["state"] == "error" and attempted["proposal_id"] is None


def test_wrong_actor_and_changed_confirm_record_are_rejected_at_cold_export(configured, tmp_path):
    req, model, app, cloud = configured
    first = app.submit(WORDS)
    app.confirm(first["proposal_id"])
    rows = [json.loads(line) for line in app._journal.read_bytes().splitlines()]
    with pytest.raises(ValueError):
        bound_outcome(rows[1]["detail"]["outcome"],rows[1]["detail"]["request"],passenger_selected=True)
    rows[-1]["detail"]["verified_parking_outcome_sha256"] = "0"*64
    altered = tmp_path/"modified.jsonl"
    altered.write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
    with pytest.raises(ValueError, match="verified decision"):
        export(altered)


def test_normal_speech_review_binds_actual_audio_and_corrected_passenger_words(configured,tmp_path):
    req, model, original, cloud = configured
    site = PassengerSite({"test":req["scene"]},model,StubASR(),tmp_path/"sessions",
                         partial(ParkingPassengerSession,parking_backend=cloud))
    sid = site.open("test")["session"]
    with pytest.raises(PassengerRequestError):
        site.change(sid,0,"submit",{"mode":"voice","text":WORDS})
    site.change(sid,0,"begin-audio",{})
    voice = site.change(sid,1,"audio",b"fixture audio")
    assert voice["voice_stage"] == "review" and not cloud.calls
    offer = site.change(sid,2,"submit",{"mode":"voice","text":WORDS})
    actual, = cloud.calls
    assert actual["text"] == WORDS and actual["input_mode"] == "voice" and actual["original_transcript"] == voice["transcript"]
    site.change(sid,3,"confirm",{"proposal_id":offer["view"]["proposal_id"]})
    file, = (tmp_path/"sessions"/sid).glob("*.jsonl")
    saved = export(file)["events"][1]
    assert saved["original_wav_verified"] and saved["phase"] == "automatic_parking_proposal"
