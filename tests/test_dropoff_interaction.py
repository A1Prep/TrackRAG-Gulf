"""Dialogue wiring checks with explicit fixtures, not model accuracy evidence."""

from copy import deepcopy
import json
from threading import Event, Thread

import pytest

from trackrag.dropoff.contracts import error_outcome, proposal_fingerprint
from trackrag.dropoff.interaction import InteractionController, InvalidTransition


@pytest.fixture
def instruction():
    return {
        "request_id": "instruction-1", "session_id": "session-1",
        "scene": {
            "frame": {"recording_id": "fixture-recording", "frame_id": 10,
                      "timestamp_s": 0.9, "width": 160, "height": 90,
                      "image_path": "fixture-only.png"},
            "source": "predicted",
            "tracks": [{"recording_id": "fixture-recording", "frame_id": 10,
                        "track_id": 2, "class_name": "Car", "source": "predicted",
                        "bbox_xyxy": [10, 20, 40, 50]}],
            "provenance": {"condition": "generated_test_fixture"},
        },
        "text": "Drop me off beside the white car.", "input_mode": "typed", "dialogue": [],
    }


def semantic(request, status="proposed"):
    return {
        "ok": True, "request_id": request["request_id"],
        "result": {"status": status,
                   "destination_point_xy": [70, 70] if status == "proposed" else None,
                   "destination_box_xyxy": [60, 60, 80, 85] if status == "proposed" else None,
                   "clarification_question": "The white car on the left or the right?"
                   if status == "needs_clarification" else None},
        "provenance": {"backend_kind": "injected_test_adapter", "model_id": "fixture-only"},
    }


class ScriptedAdapter:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def interpret(self, request):
        self.requests.append(deepcopy(request))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(request)
        if isinstance(response, str) and response in (
            "proposed", "needs_clarification", "insufficient_visual_evidence",
        ):
            return semantic(request, response)
        return deepcopy(response)


def controller(*responses):
    return InteractionController(ScriptedAdapter(*responses), session_id="session-1")


def confirmation(shown_state):
    proposal = shown_state["proposal"]
    return {"proposal_id": proposal["proposal_id"], "request_id": proposal["request_id"],
            "scene": deepcopy(proposal["scene"]),
            "proposal_sha256": proposal_fingerprint(proposal, shown_state["request"])}


def test_white_car_clarification_proposal_and_explicit_confirmation(instruction):
    adapter = ScriptedAdapter("needs_clarification", "proposed")
    flow = InteractionController(adapter, session_id="session-1")
    first = flow.submit(instruction)
    assert first["status"] == "needs_clarification"
    assert first["proposal"] is None and first["confirmation"] is None
    shown = flow.reply("The one on the right.")
    assert shown["status"] == "proposed"
    assert shown["confirmation"] is None
    assert shown["request"]["request_id"] != instruction["request_id"]
    assert adapter.requests[1]["text"] == "The one on the right."
    assert adapter.requests[1]["dialogue"] == [
        {"role": "user", "text": instruction["text"], "recording_id": "fixture-recording", "frame_id": 10},
        {"role": "assistant", "text": first["pending_clarification"],
         "recording_id": "fixture-recording", "frame_id": 10},
    ]
    confirmed = flow.confirm(confirmation(shown))
    assert confirmed["status"] == "confirmed"
    assert flow.snapshot()["counts"] == {
        "submission_rows": 2, "rejected_inputs": 0,
        "requested_rows": 2, "completed_rows": 2, "unfinished_rows": 0, "operational_errors": 0,
    }


def test_repeated_clarification_keeps_the_original_instruction_and_all_relevant_turns(instruction):
    adapter = ScriptedAdapter("needs_clarification", "needs_clarification", "proposed")
    flow = InteractionController(adapter, session_id="session-1")
    flow.submit(instruction)
    flow.reply("The nearer one.")
    shown = flow.reply("On the right.")
    assert len(adapter.requests[-1]["dialogue"]) == 4
    assert adapter.requests[-1]["dialogue"][0]["text"] == instruction["text"]
    assert adapter.requests[-1]["dialogue"][2]["text"] == "The nearer one."
    assert shown["status"] == "proposed"


def test_a_yes_text_reply_cannot_confirm_a_model_proposal(instruction):
    flow = controller("needs_clarification", "proposed")
    flow.submit(instruction)
    assert flow.reply("yes")["status"] == "proposed"
    assert flow.state["confirmation"] is None
    assert not any(event["action"] == "proposal_confirmed" for event in flow.snapshot()["events"])


def test_revision_invalidates_confirmation_before_the_next_inference(instruction):
    flow = controller("proposed", "proposed")
    shown = flow.submit(instruction)
    old_confirmation = confirmation(shown)
    flow.confirm(old_confirmation)
    pending = flow.begin_revision("Beside the black car instead.")
    assert pending["status"] == "pending"
    assert pending["proposal"] is None and pending["confirmation"] is None
    with pytest.raises(InvalidTransition):
        flow.confirm(old_confirmation)
    revised = flow.run_pending(pending["request"]["request_id"])
    assert revised["proposal"]["proposal_id"] != shown["proposal"]["proposal_id"]
    with pytest.raises(ValueError):
        flow.confirm(old_confirmation)
    assert flow.state["status"] == "proposed"


@pytest.mark.parametrize("field,value", [
    ("proposal_id", "stale"), ("request_id", "stale"), ("proposal_sha256", "0" * 64),
    ("recording_id", "other-recording"), ("frame_id", 11), ("source", "annotated"),
])
def test_confirmation_must_match_the_exact_rendered_proposal_and_scene(instruction, field, value):
    flow = controller("proposed")
    shown = flow.submit(instruction)
    token = confirmation(shown)
    if field in ("recording_id", "frame_id", "source"):
        token["scene"][field] = value
    else:
        token[field] = value
    with pytest.raises(ValueError):
        flow.confirm(token)
    assert flow.state["status"] == "proposed"


def test_voice_correction_preserves_asr_and_sends_only_the_revised_current_utterance(instruction):
    instruction.update(input_mode="voice", text="Beside the wide car.",
                       original_transcript="Beside the wide car.")
    adapter = ScriptedAdapter("proposed", "proposed", "proposed")
    flow = InteractionController(adapter, session_id="session-1")
    shown = flow.submit(instruction)
    flow.confirm(confirmation(shown))
    corrected = flow.revise_instruction("Beside the white car.")
    assert corrected["request"]["original_transcript"] == "Beside the wide car."
    assert corrected["request"]["text"] == "Beside the white car."
    assert corrected["request"]["dialogue"] == []
    assert corrected["confirmation"] is None
    correction = corrected["transcript_corrections"][0]
    assert correction["request_id"] == corrected["request"]["request_id"]
    assert correction["corrected_text"] == "Beside the white car."
    again = flow.revise_instruction("Beside the white car on the right.")
    assert len(again["transcript_corrections"]) == 2
    assert all(item["original_transcript"] == instruction["original_transcript"]
               for item in again["transcript_corrections"])
    assert adapter.requests[-1]["text"] == again["request"]["text"]


def test_voice_clarification_reply_retains_its_own_original_transcript(instruction):
    flow = controller("needs_clarification", "proposed")
    flow.submit(instruction)
    replied = flow.reply("The one on the right.", input_mode="voice",
                        original_transcript="The one on the right.")
    assert replied["request"]["input_mode"] == "voice"
    assert replied["request"]["original_transcript"] == "The one on the right."


@pytest.mark.parametrize("status", ["insufficient_visual_evidence", "needs_clarification"])
def test_semantic_unresolved_results_have_no_current_proposal(instruction, status):
    flow = controller(status)
    current = flow.submit(instruction)
    assert current["status"] == status
    assert current["proposal"] is None
    assert flow.snapshot()["calls"][0]["outcome"]["ok"] is True
    with pytest.raises(InvalidTransition):
        flow.confirm({})


@pytest.mark.parametrize("code", ["checkpoint_unavailable", "timeout", "resource_unavailable",
                                  "invalid_model_output", "artifact_invalid"])
def test_adapter_error_receipts_are_preserved_as_errors(instruction, code):
    flow = controller(lambda request: error_outcome(code, "Explicit fixture failure", request["request_id"],
                                                  {"model_id": "fixture-only"}))
    current = flow.submit(instruction)
    assert current["status"] == "error" and current["proposal"] is None
    receipt = flow.snapshot()["calls"][0]
    assert receipt["outcome"]["error"]["code"] == code
    assert receipt["outcome"]["provenance"]["model_id"] == "fixture-only"
    assert "result" not in receipt["outcome"]


@pytest.mark.parametrize("error,code", [(TimeoutError("late"), "timeout"),
                                       (MemoryError("full"), "resource_unavailable"),
                                       (RuntimeError("adapter broke"), "adapter_exception")])
def test_adapter_exceptions_do_not_leave_unrecorded_requests(instruction, error, code):
    flow = controller(error)
    assert flow.submit(instruction)["status"] == "error"
    receipt = flow.snapshot()["calls"][0]
    assert receipt["outcome"]["error"]["code"] == code
    assert flow.snapshot()["counts"]["operational_errors"] == 1


@pytest.mark.parametrize("mutation", ["missing", "wrong_request", "nan", "outside", "model_confirmation",
                                      "request_hash", "string", "object"])
def test_malformed_outputs_are_auditable_errors_and_never_drawable(instruction, mutation, tmp_path):
    def malformed(request):
        outcome = semantic(request)
        if mutation == "missing":
            outcome["result"].pop("destination_box_xyxy")
        elif mutation == "wrong_request":
            outcome["request_id"] = "foreign-request"
        elif mutation == "nan":
            outcome["result"]["destination_point_xy"][0] = float("nan")
        elif mutation == "outside":
            outcome["result"]["destination_point_xy"] = [160, 70]
        elif mutation == "model_confirmation":
            outcome["result"]["confirmation"] = True
        elif mutation == "request_hash":
            outcome["provenance"]["request_sha256"] = "0" * 64
        elif mutation == "string":
            return "a malformed model response"
        else:
            return object()
        return outcome

    flow = controller(malformed)
    assert flow.submit(instruction)["status"] == "error"
    receipt = flow.snapshot()["calls"][0]
    assert receipt["outcome"]["error"]["code"] == "invalid_model_output"
    assert receipt["raw_outcome"] is not None
    saved = json.loads(flow.save(tmp_path / "session.json").read_text(encoding="utf-8"))
    assert saved["counts"]["operational_errors"] == 1


@pytest.mark.parametrize("text", ["here", "Here.", " HERE?! "])
def test_a_bare_here_proposal_without_pointing_is_rejected_without_faking_abstention(instruction, text):
    instruction["text"] = text
    flow = controller("proposed")
    assert flow.submit(instruction)["status"] == "error"
    receipt = flow.snapshot()["calls"][0]
    assert receipt["raw_outcome"]["value"]["result"]["status"] == "proposed"
    assert receipt["outcome"]["error"]["code"] == "invalid_model_output"


def test_bare_here_can_receive_a_genuine_model_clarification(instruction):
    instruction["text"] = "here"
    assert controller("needs_clarification").submit(instruction)["status"] == "needs_clarification"


@pytest.mark.parametrize("action", ["cancel", "scene", "new_request"])
def test_late_outcomes_cannot_revive_cancelled_or_superseded_requests(instruction, action):
    flow = controller()
    first = flow.begin(instruction)
    if action == "cancel":
        expected = flow.cancel()
    elif action == "scene":
        flow.reset_scene()
        expected = None
    else:
        next_request = deepcopy(instruction)
        next_request["request_id"] = "instruction-2"
        next_request["scene"]["frame"]["frame_id"] = 11
        next_request["scene"]["tracks"][0]["frame_id"] = 11
        expected = flow.begin(next_request)
    assert flow.complete(instruction["request_id"], semantic(first["request"])) == expected
    receipt = flow.snapshot()["calls"][0]
    assert receipt["outcome"]["ok"] is True and receipt["disposition"] == "discarded"
    assert flow.state is None or flow.state["proposal"] is None


def test_a_late_invalid_response_is_retained_without_turning_the_new_request_into_an_error(instruction):
    flow = controller()
    flow.begin(instruction)
    replacement = {**deepcopy(instruction), "request_id": "instruction-2"}
    pending = flow.begin(replacement)
    assert flow.complete(instruction["request_id"], None) == pending
    assert flow.state["status"] == "pending"
    assert flow.snapshot()["calls"][0]["outcome"]["error"]["code"] == "invalid_model_output"


def test_cancel_is_available_while_real_worker_is_waiting_on_the_adapter(instruction):
    entered, release = Event(), Event()
    errors = []

    def blocking(request):
        entered.set()
        if not release.wait(3):
            raise TimeoutError("test adapter was not released")
        return semantic(request)

    flow = controller(blocking)
    flow.begin(instruction)

    def run():
        try:
            flow.run_pending(instruction["request_id"])
        except Exception as error:
            errors.append(error)

    worker = Thread(target=run, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        assert flow.cancel()["status"] == "cancelled"
    finally:
        release.set()
        worker.join(timeout=3)
    assert not worker.is_alive() and errors == []
    assert flow.state["status"] == "cancelled" and flow.state["proposal"] is None
    assert flow.snapshot()["calls"][0]["disposition"] == "discarded"


def test_cancelled_before_dispatch_is_saved_as_unfinished_without_calling_the_adapter(instruction):
    adapter = ScriptedAdapter()
    flow = InteractionController(adapter, session_id="session-1")
    flow.begin(instruction)
    flow.cancel()
    assert flow.run_pending(instruction["request_id"])["status"] == "cancelled"
    assert adapter.requests == []
    assert flow.snapshot()["counts"]["unfinished_rows"] == 1


@pytest.mark.parametrize("mutation", ["future", "foreign_recording", "foreign_source", "gold",
                                      "empty_text", "foreign_session", "missing_asr", "duplicate"])
def test_rejected_new_inputs_clear_old_confirmation_without_inventing_a_model_attempt(instruction, mutation):
    flow = controller("proposed")
    flow.confirm(confirmation(flow.submit(instruction)))
    bad = {**deepcopy(instruction), "request_id": "instruction-2"}
    if mutation == "future":
        bad["scene"]["tracks"][0]["frame_id"] = 11
    elif mutation == "foreign_recording":
        bad["scene"]["tracks"][0]["recording_id"] = "foreign"
    elif mutation == "foreign_source":
        bad["scene"]["tracks"][0]["source"] = "annotated"
    elif mutation == "gold":
        bad["target"] = {"destination_point_xy": [70, 70]}
    elif mutation == "empty_text":
        bad["text"] = ""
    elif mutation == "foreign_session":
        bad["session_id"] = "another-session"
    elif mutation == "missing_asr":
        bad["input_mode"] = "voice"
    else:
        bad["request_id"] = instruction["request_id"]
    with pytest.raises(ValueError):
        flow.begin(bad)
    assert flow.state is None
    assert flow.snapshot()["counts"]["requested_rows"] == 1
    assert flow.snapshot()["counts"]["submission_rows"] == 2
    assert flow.snapshot()["counts"]["rejected_inputs"] == 1
    assert flow.snapshot()["events"][-1]["action"] == "input_rejected"
    assert flow.snapshot()["events"][-1]["input"]["value"] == bad


def test_dialogue_budget_is_not_silently_truncated(instruction):
    instruction["dialogue"] = [{"role": "user", "text": "Relevant context",
                               "recording_id": "fixture-recording", "frame_id": 10}] * 16
    flow = controller("needs_clarification")
    flow.submit(instruction)
    with pytest.raises(ValueError, match="16 turns"):
        flow.begin_reply("On the right.")
    assert flow.snapshot()["counts"]["requested_rows"] == 1


def test_legacy_request_is_bound_to_the_app_session_before_inference(instruction):
    instruction.pop("session_id")
    adapter = ScriptedAdapter("proposed")
    flow = InteractionController(adapter, session_id="session-1")
    assert flow.submit(instruction)["request"]["session_id"] == "session-1"
    assert adapter.requests[0]["session_id"] == "session-1"
    assert "session_id" not in instruction


def test_invalid_or_duplicate_completions_cannot_overwrite_saved_outcomes(instruction):
    flow = controller("proposed")
    flow.submit(instruction)
    before = flow.snapshot()
    for identity in ("unknown-request", instruction["request_id"]):
        with pytest.raises(InvalidTransition):
            flow.complete(identity, None)
    assert flow.snapshot() == before


def test_external_records_and_an_adapter_cannot_mutate_the_controller(instruction):
    original = deepcopy(instruction)

    def mutate(request):
        request["scene"]["tracks"].clear()
        return semantic(request)

    flow = controller(mutate)
    shown = flow.submit(instruction)
    assert instruction == original
    assert flow.state["request"]["scene"]["tracks"] == original["scene"]["tracks"]
    shown["proposal"]["result"]["destination_point_xy"][0] = 0
    exported = flow.snapshot()
    exported["calls"][0]["request"]["text"] = "changed outside"
    assert flow.state["proposal"]["result"]["destination_point_xy"] == [70, 70]
    assert flow.snapshot()["calls"][0]["request"]["text"] == original["text"]


def test_saved_session_keeps_errors_semantic_unresolved_and_unfinished_rows(instruction, tmp_path):
    flow = controller(TimeoutError("fixture timeout"), "insufficient_visual_evidence")
    flow.submit(instruction)
    second = {**deepcopy(instruction), "request_id": "instruction-2"}
    flow.submit(second)
    third = {**deepcopy(instruction), "request_id": "instruction-3"}
    flow.begin(third)
    flow.cancel()
    path = flow.save(tmp_path / "session.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["counts"] == {
        "submission_rows": 3, "rejected_inputs": 0,
        "requested_rows": 3, "completed_rows": 2, "unfinished_rows": 1, "operational_errors": 1,
    }
    assert saved["calls"][0]["outcome"]["ok"] is False
    assert saved["calls"][1]["outcome"]["result"]["status"] == "insufficient_visual_evidence"
    assert saved["calls"][2]["outcome"] is None
    with pytest.raises(FileExistsError):
        flow.save(path)


def test_scene_reset_keeps_confirmation_history_but_removes_current_authority(instruction):
    flow = controller("proposed")
    binding = confirmation(flow.submit(instruction))
    flow.confirm(binding)
    flow.reset_scene()
    assert flow.state is None
    assert any(event.get("confirmation") == binding for event in flow.snapshot()["events"])
    with pytest.raises(InvalidTransition):
        flow.confirm(binding)


def test_actual_adapter_receipts_are_snapshotted_before_another_model_call(instruction, tmp_path):
    class ReceiptAdapter:
        def interpret(self, request):
            self.last_receipt = {"raw_text": "fixture response for " + request["request_id"],
                                 "request": deepcopy(request), "condition": "injected_test_adapter"}
            return semantic(request)

    adapter = ReceiptAdapter()
    flow = InteractionController(adapter, session_id="session-1")
    flow.submit(instruction)
    flow.revise_instruction("Beside the other car.")
    adapter.last_receipt["raw_text"] = "changed after inference"
    saved = json.loads(flow.save(tmp_path / "session.json").read_text(encoding="utf-8"))
    assert saved["calls"][0]["adapter_receipt"]["value"]["raw_text"] == "fixture response for instruction-1"
    assert saved["calls"][1]["adapter_receipt"]["value"]["request"]["text"] == "Beside the other car."


def test_controller_accepts_the_existing_model_adapter_and_preserves_its_raw_receipt(instruction, tmp_path):
    from PIL import Image
    from trackrag.dropoff.model import GroundingModel

    path = tmp_path / "fixture.png"
    Image.new("RGB", (160, 90), "grey").save(path)
    instruction["scene"]["frame"]["image_path"] = str(path)

    class Backend:
        def generate(self, messages, *, config):
            return json.dumps({"status": "proposed", "destination_point_xy": [500, 800],
                               "destination_box_xyxy": [400, 700, 600, 900],
                               "clarification_question": None})

    flow = InteractionController(GroundingModel(backend=Backend()), session_id="session-1")
    shown = flow.submit(instruction)
    assert shown["status"] == "proposed"
    assert shown["proposal"]["result"]["destination_point_xy"] == [80, 72]
    assert shown["proposal"]["provenance"]["backend_kind"] == "injected_test_backend"
    saved = flow.snapshot()["calls"][0]["adapter_receipt"]["value"]
    assert saved["raw_text"]
    assert saved["request"]["request_id"] == instruction["request_id"]


def test_controller_uses_atomic_outcome_and_receipt_instead_of_shared_latest_receipt(instruction):
    class AtomicAdapter:
        def interpret(self, request):
            raise AssertionError("The atomic model must not use the legacy call")

        def interpret_with_receipt(self, request):
            return semantic(request), {"request_id": request["request_id"],
                                       "raw_text": "matching atomic receipt"}

        @property
        def last_receipt(self):
            raise AssertionError("Another caller may have replaced the shared receipt")

    flow = InteractionController(AtomicAdapter(), session_id="session-1")
    assert flow.submit(instruction)["status"] == "proposed"
    flow.revise_instruction("Beside the other car.")
    calls = flow.snapshot()["calls"]
    for call in calls:
        receipt = call["adapter_receipt"]
        assert receipt["request_binding"] == "matches"
        assert receipt["value"]["request_id"] == call["request"]["request_id"]
        assert receipt["value"]["raw_text"] == "matching atomic receipt"
    assert calls[0]["request"]["request_id"] != calls[1]["request"]["request_id"]


def test_atomic_failure_is_retained_without_retrying_the_legacy_method(instruction):
    class AtomicAdapter:
        def interpret(self, request):
            raise AssertionError("A failed model call must not be silently retried")

        def interpret_with_receipt(self, request):
            raise TimeoutError("atomic inference timed out")

        @property
        def last_receipt(self):
            raise AssertionError("There is no atomic receipt for the failed call")

    flow = InteractionController(AtomicAdapter(), session_id="session-1")
    state = flow.submit(instruction)
    assert state["status"] == "error"
    assert state["proposal"] is None
    call = flow.snapshot()["calls"][0]
    assert call["outcome"]["error"]["code"] == "timeout"
    assert call["adapter_receipt"] is None


def test_old_adapter_receipt_after_an_exception_is_explicitly_marked_as_foreign(instruction):
    class Adapter:
        def interpret(self, request):
            if hasattr(self, "last_receipt"):
                raise RuntimeError("failed before replacing the previous receipt")
            self.last_receipt = {"request_id": request["request_id"], "raw_text": "first fixture response"}
            return semantic(request)

    flow = InteractionController(Adapter(), session_id="session-1")
    flow.submit(instruction)
    flow.revise_instruction("The other car.")
    calls = flow.snapshot()["calls"]
    assert calls[0]["adapter_receipt"]["request_binding"] == "matches"
    assert calls[1]["adapter_receipt"]["request_binding"] == "different_request"
    assert calls[1]["outcome"]["error"]["code"] == "adapter_exception"
    assert calls[1]["adapter_receipt"]["value"]["request_id"] == instruction["request_id"]


def test_voice_correction_events_survive_scene_reset(instruction):
    instruction.update(input_mode="voice", text="Beside the wide car.",
                       original_transcript="Beside the wide car.")
    flow = controller("proposed", "proposed")
    flow.submit(instruction)
    revised = flow.revise_instruction("Beside the white car.")
    flow.reset_scene()
    events = flow.snapshot()["events"]
    revision = next(event for event in events if event["action"] == "request_started" and
                    event["request_id"] == revised["request"]["request_id"])
    assert revision["transcript_corrections"][0]["original_transcript"] == instruction["original_transcript"]
    assert revision["transcript_corrections"][0]["corrected_text"] == "Beside the white car."


@pytest.mark.parametrize("action", ["reply", "revise_instruction", "confirm", "cancel"])
def test_actions_require_an_applicable_current_interaction(action):
    flow = controller()
    with pytest.raises(InvalidTransition):
        if action == "confirm":
            flow.confirm({})
        elif action == "cancel":
            flow.cancel()
        else:
            getattr(flow, action)("Some text")
