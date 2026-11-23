"""Scene-bound dialogue controllers for the Task 4 and passenger applications.

Pass an explicitly configured Task 5 adapter. The controller never supplies a
destination, performs speech recognition or confirms on the passenger's behalf.
All outward records are snapshots; late responses remain auditable without
replacing a newer proposal. Scene selection must call reset_scene immediately.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from threading import Lock, RLock
from uuid import uuid4

from .contracts import (
    GroundingAdapter, InstructionRequest, InteractionState, ProposalConfirmation,
    error_outcome, validate_interaction_state,
    validate_outcome, validate_proposal, validate_request, validate_target,
)
from .inputs import _inspect_image


class InvalidTransition(ValueError):
    """The requested UI action does not apply to the current interaction."""


def _raw_snapshot(value):
    """Keep malformed responses inspectable while the saved session remains JSON."""
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
        if len(encoded.encode("utf-8")) <= 2_000_000:
            return {"captured": True, "value": json.loads(encoded)}
        return {"captured": False, "reason": "response_exceeds_receipt_budget",
                "sha256": sha256(encoded.encode("utf-8")).hexdigest()}
    except (TypeError, ValueError, OverflowError, RecursionError):
        try:
            description = repr(value)[:4096]
        except Exception:
            description = "Representation unavailable"
        return {"captured": False, "reason": "response_is_not_finite_json",
                "python_type": type(value).__name__, "description": description}


class InteractionController:
    """Manage one passenger session using the shared, original-pixel contracts.

    submit/reply/revise_instruction are synchronous conveniences. A UI worker can
    instead call begin/begin_reply/begin_revision, render pending state, and call
    run_pending. State transitions are locked; inference releases that lock so
    cancellation/scene changes can invalidate an in-flight call. Model calls are
    serialized because the supplied adapter can keep mutable inference receipts.
    """

    def __init__(self, adapter: GroundingAdapter, *, session_id: str | None = None,
                 allow_annotated: bool = False):
        identity = session_id if session_id is not None else str(uuid4())
        if not isinstance(identity, str) or not identity.strip() or len(identity) > 200:
            raise ValueError("session_id must be nonempty text of at most 200 characters")
        if type(allow_annotated) is not bool:
            raise ValueError("allow_annotated must be an explicit boolean")
        self._session_id = identity
        self._adapter = adapter
        self._allow_annotated = allow_annotated
        self._state: InteractionState | None = None
        self._calls: dict[str, dict] = {}
        self._events: list[dict] = []
        self._lock = RLock()
        self._model_lock = Lock()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def state(self) -> InteractionState | None:
        with self._lock:
            return deepcopy(self._state)

    def _event(self, action, **details):
        self._events.append({"sequence": len(self._events) + 1, "action": action, **deepcopy(details)})

    def _discard_current(self, reason):
        if self._state is not None:
            call = self._calls[self._state["request"]["request_id"]]
            if call["disposition"] == "pending":
                call["disposition"] = "discarded"
                call["discard_reason"] = reason
        self._state = None

    def _begin(self, request, corrections):
        # Invalidate first: even rejected replacement input must not leave an old
        # destination or confirmation looking current in the UI.
        self._discard_current("replaced")
        try:
            checked = validate_request(request, allow_annotated=self._allow_annotated)
            if checked.get("session_id", self.session_id) != self.session_id:
                raise ValueError("request belongs to a different session")
            checked["session_id"] = self.session_id
            request_id = checked["request_id"]
            if request_id in self._calls:
                raise ValueError("request_id has already been used; submit a fresh identity")
            pending = validate_interaction_state({
                "session_id": self.session_id, "request": checked, "status": "pending",
                "pending_clarification": None, "proposal": None, "confirmation": None,
                "transcript_corrections": corrections,
            }, allow_annotated=self._allow_annotated)
        except ValueError as error:
            self._event("input_rejected", message=str(error)[:4096], input=_raw_snapshot(request))
            raise
        self._calls[request_id] = {
            "request": deepcopy(checked), "outcome": None, "raw_outcome": None,
            "adapter_receipt": None, "disposition": "pending", "discard_reason": None,
        }
        self._state = pending
        self._event("request_started", request_id=request_id, transcript_corrections=corrections)
        return deepcopy(pending)

    def begin(self, request: InstructionRequest) -> InteractionState:
        """Start a fresh instruction and clear the previous interaction context.

        Rejected inputs raise ValueError without calling the model. They clear
        the current state and produce an input_rejected event, not a model result.
        Every accepted request gets a receipt immediately, including pending ones.
        """
        with self._lock:
            corrections = []
            if (isinstance(request, dict) and request.get("input_mode") == "voice" and
                    "original_transcript" in request and request.get("text") != request["original_transcript"]):
                corrections.append({"request_id": request.get("request_id"),
                                    "original_transcript": request["original_transcript"],
                                    "corrected_text": request.get("text")})
            return self._begin(request, corrections)

    def complete(self, request_id: str, raw_outcome) -> InteractionState | None:
        """Retain one outcome for a known request; apply it only if still pending.

        Callers with their own model worker may supply its response here. Unknown
        or duplicate completions raise InvalidTransition rather than overwriting
        a saved result. Malformed outcomes become operational error receipts.
        """
        with self._lock:
            call = self._calls.get(request_id)
            if call is None or call["outcome"] is not None:
                raise InvalidTransition("unknown request or outcome already recorded")
            request = call["request"]
            call["raw_outcome"] = _raw_snapshot(raw_outcome)
            try:
                outcome = validate_outcome(raw_outcome, request,
                                           allow_annotated=self._allow_annotated)
                expected_hash = sha256(json.dumps(
                    request, sort_keys=True, separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")).hexdigest()
                supplied_hash = outcome["provenance"].get("request_sha256", expected_hash)
                if supplied_hash != expected_hash:
                    raise ValueError("model provenance is bound to a different request")
                if (outcome["ok"] and outcome["result"]["status"] == "proposed" and
                        request["text"].strip().casefold().strip(".!? ") == "here"):
                    # This contract has no pointing/selection input. Preserve the
                    # faulty response as an error, not a fabricated clarification.
                    raise ValueError("bare 'here' has no pointing or selection context")
            except (ValueError, TypeError, OverflowError) as error:
                outcome = error_outcome("invalid_model_output", str(error)[:4096], request_id,
                                        {"component": "interaction_validation"})
            call["outcome"] = outcome
            active = (self._state is not None and self._state["status"] == "pending" and
                      self._state["request"]["request_id"] == request_id)
            if active:
                next_state = deepcopy(self._state)
                if not outcome["ok"]:
                    next_state["status"] = "error"
                else:
                    result = outcome["result"]
                    next_state["status"] = result["status"]
                    next_state["pending_clarification"] = result["clarification_question"]
                    if result["status"] == "proposed":
                        frame = request["scene"]["frame"]
                        next_state["proposal"] = validate_proposal({
                            "proposal_id": str(uuid4()), "request_id": request_id,
                            "scene": {"recording_id": frame["recording_id"],
                                      "frame_id": frame["frame_id"], "source": request["scene"]["source"]},
                            "result": result, "provenance": outcome["provenance"],
                        }, request, allow_annotated=self._allow_annotated)
                self._state = validate_interaction_state(next_state,
                                                         allow_annotated=self._allow_annotated)
                call["disposition"] = "applied"
            else:
                call["disposition"] = "discarded"
            self._event("outcome_received", request_id=request_id, applied=active)
            return deepcopy(self._state)

    def run_pending(self, request_id: str) -> InteractionState | None:
        """Call the supplied adapter; backend timeouts/resources belong to its config.

        A request cancelled/superseded before dispatch is retained with no outcome.
        If it becomes stale during inference, complete keeps its eventual outcome
        for evaluation while preventing it from changing the current interaction.
        """
        with self._model_lock:
            with self._lock:
                if request_id not in self._calls:
                    raise InvalidTransition("unknown request")
                if (self._state is None or self._state["status"] != "pending" or
                        self._state["request"]["request_id"] != request_id):
                    return deepcopy(self._state)
                request = deepcopy(self._calls[request_id]["request"])
            adapter_receipt = None
            uses_atomic_receipt = False
            try:
                atomic = getattr(self._adapter, "interpret_with_receipt", None)
                uses_atomic_receipt = callable(atomic)
                if uses_atomic_receipt:
                    raw_outcome, adapter_receipt = atomic(request)
                else:
                    raw_outcome = self._adapter.interpret(request)
            except Exception as error:
                code = ("timeout" if isinstance(error, TimeoutError) else
                        "resource_unavailable" if isinstance(error, MemoryError) else "adapter_exception")
                raw_outcome = error_outcome(code, f"{type(error).__name__}: {error}"[:4096], request_id,
                                            {"component": "interaction_adapter_call"})
            # Prefer the model's atomic receipt, which is protected even when
            # another controller shares the adapter. Legacy adapters remain
            # supported under this controller's model lock; do not retry a failed
            # atomic call or substitute its mutable last_receipt.
            try:
                if not uses_atomic_receipt:
                    adapter_receipt = getattr(self._adapter, "last_receipt", None)
                receipt = _raw_snapshot(adapter_receipt) if adapter_receipt is not None else None
                if receipt is not None:
                    declared_id = None
                    if isinstance(adapter_receipt, dict):
                        declared_id = adapter_receipt.get("request_id")
                        if declared_id is None and isinstance(adapter_receipt.get("request"), dict):
                            declared_id = adapter_receipt["request"].get("request_id")
                    receipt["request_binding"] = (
                        "unverified" if not isinstance(declared_id, str) else
                        "matches" if declared_id == request_id else "different_request"
                    )
            except Exception as error:
                receipt = {"captured": False, "reason": "adapter_receipt_unavailable",
                           "description": f"{type(error).__name__}: {error}"[:4096]}
            with self._lock:
                self._calls[request_id]["adapter_receipt"] = receipt
            return self.complete(request_id, raw_outcome)

    def submit(self, request: InstructionRequest) -> InteractionState | None:
        pending = self.begin(request)
        return self.run_pending(pending["request"]["request_id"])

    def begin_reply(self, text: str, *, input_mode: str = "typed",
                    original_transcript: str | None = None) -> InteractionState:
        """Resubmit a clarification reply with the actual same-scene question/context."""
        with self._lock:
            if self._state is None or self._state["status"] != "needs_clarification":
                raise InvalidTransition("there is no pending clarification")
            current = self._state
            previous = current["request"]
            frame = previous["scene"]["frame"]
            binding = {"recording_id": frame["recording_id"], "frame_id": frame["frame_id"]}
            request = {
                "request_id": str(uuid4()), "session_id": self.session_id,
                "scene": deepcopy(previous["scene"]), "text": text, "input_mode": input_mode,
                "dialogue": deepcopy(previous["dialogue"]) + [
                    {"role": "user", "text": previous["text"], **binding},
                    {"role": "assistant", "text": current["pending_clarification"], **binding},
                ],
            }
            if original_transcript is not None:
                request["original_transcript"] = original_transcript
            return self._begin(request, deepcopy(current["transcript_corrections"]))

    def reply(self, text: str, *, input_mode: str = "typed",
              original_transcript: str | None = None) -> InteractionState | None:
        pending = self.begin_reply(text, input_mode=input_mode, original_transcript=original_transcript)
        return self.run_pending(pending["request"]["request_id"])

    def begin_revision(self, text: str) -> InteractionState:
        """Replace the current utterance; preserve ASR separately and clear confirmation.

        The superseded utterance is not appended to the model's dialogue. Existing
        relevant prior turns are preserved, without silently truncating budgets.
        """
        with self._lock:
            if self._state is None:
                raise InvalidTransition("there is no instruction to revise")
            request = deepcopy(self._state["request"])
            corrections = deepcopy(self._state["transcript_corrections"])
            request["request_id"], request["text"] = str(uuid4()), text
            if request["input_mode"] == "voice":
                corrections.append({"request_id": request["request_id"],
                                    "original_transcript": request["original_transcript"],
                                    "corrected_text": text})
            return self._begin(request, corrections)

    def revise_instruction(self, text: str) -> InteractionState | None:
        pending = self.begin_revision(text)
        return self.run_pending(pending["request"]["request_id"])

    def confirm(self, confirmation: ProposalConfirmation) -> InteractionState:
        """Handle an explicit passenger action using the binding rendered by the UI."""
        with self._lock:
            if self._state is None or self._state["status"] != "proposed":
                raise InvalidTransition("there is no current unconfirmed proposal")
            candidate = deepcopy(self._state)
            candidate["status"], candidate["confirmation"] = "confirmed", deepcopy(confirmation)
            self._state = validate_interaction_state(candidate, allow_annotated=self._allow_annotated)
            self._event("proposal_confirmed", confirmation=confirmation)
            return deepcopy(self._state)

    def cancel(self) -> InteractionState:
        with self._lock:
            if self._state is None:
                raise InvalidTransition("there is no active instruction")
            current = deepcopy(self._state)
            self._discard_current("cancelled")
            current.update(status="cancelled", pending_clarification=None,
                           proposal=None, confirmation=None)
            self._state = validate_interaction_state(current, allow_annotated=self._allow_annotated)
            self._event("cancelled", request_id=current["request"]["request_id"])
            return deepcopy(self._state)

    def reset_scene(self) -> None:
        """Invalidate immediately on recording/frame/image/source selection changes."""
        with self._lock:
            self._discard_current("scene_changed")
            self._event("scene_reset")

    def snapshot(self) -> dict:
        """Return all accepted rows, including errors, discarded and unfinished calls."""
        with self._lock:
            calls = list(self._calls.values())
            completed = [call for call in calls if call["outcome"] is not None]
            rejected = sum(event["action"] == "input_rejected" for event in self._events)
            return deepcopy({
                "schema_version": "trackrag.dropoff.interaction.v1", "session_id": self.session_id,
                "allow_annotated": self._allow_annotated, "state": self._state,
                "calls": calls, "events": self._events,
                "counts": {"submission_rows": len(calls) + rejected, "rejected_inputs": rejected,
                           "requested_rows": len(calls), "completed_rows": len(completed),
                           "unfinished_rows": len(calls) - len(completed),
                           "operational_errors": sum(not call["outcome"]["ok"] for call in completed)},
            })

    def save(self, path: Path | str) -> Path:
        """Write a new JSON audit snapshot; never overwrite an existing receipt."""
        payload = json.dumps(self.snapshot(), indent=2, allow_nan=False) + "\n"
        path = Path(path)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        return path


def signed_place_verifier(method, *, live=False):
    """Replay recorded decisions. A new offer or confirmation must also pass today's model policy."""
    if method == "public_painted_bay_landmark_v1":
        from . import whole_bay
        return whole_bay.verify_current if live else whole_bay.verify
    if method == "molmo2_frontage_confidence_v1":
        from .frontage_evidence import verify
        return verify
    if method == "single_vehicle_visible_space_check_v1":
        from .single_vehicle import verify_empty
        return verify_empty
    if method == "single_vehicle_behind_parking_v1":
        if live:
            from .single_vehicle import verify_current
            return verify_current
        from .single_vehicle import verify
        return verify
    from .place_grounding import EMPTY_SURFACE_METHOD, METHOD
    if method in (METHOD, EMPTY_SURFACE_METHOD, "public_place_lookup_wait_v1"):
        from .place_evidence import verify
        return verify
    return None


class PassengerSession:
    """Keep the passenger application's synchronous scene and journal API.

    Its vehicle-selection subclass overrides _interpret_submission and _save.
    Load passenger grounding helpers only on this API's execution paths so the
    standalone InteractionController does not import the vision/model stack.
    """

    def __init__(self, scene, model, *, audit_dir):
        clean = validate_request({"request_id":"scene-open", "scene":scene, "text":"Open frame",
                                  "input_mode":"typed", "dialogue":[]})["scene"]
        path, actual = _inspect_image(clean["frame"])
        if clean.get("provenance",{}).get("image_sha256") != actual:
            raise ValueError("The original frame does not match its declared SHA256")
        clean["frame"]["image_path"] = str(path)
        self._scene, self.model = clean, model
        self._audit = Path(audit_dir)
        self._audit.mkdir(parents=True, exist_ok=True)
        self._journal = self._audit/(uuid4().hex+".jsonl")
        self._lock = RLock()
        self._turns = []
        self._state, self._proposal, self._result = "idle", None, None
        self._place_offer = None
        self._place_offer_sha256 = None
        self._manual_submission_box = None
        self._manual_offer = None
        self._bay_menu = self._bay_selection = None
        self._waiting_place = False
        self._requested_area = False
        self._error, self._selected_by, self._request = None, None, None
        self._save("opened", {"scene":clean})

    @property
    def scene(self):
        return deepcopy(self._scene)

    def _save(self, action, detail):
        item = {"action":action, "timestamp":datetime.now(timezone.utc).isoformat(),
                "detail":detail, "view":self.view()}
        addition = (json.dumps(item, allow_nan=False,separators=(",",":"))+"\n").encode()
        previous = self._journal.read_bytes() if self._journal.is_file() else b""
        partial = None
        try:
            with tempfile.NamedTemporaryFile(dir=self._audit,delete=False) as file:
                partial = Path(file.name)
                file.write(previous)
                file.write(addition)
                file.flush()
                os.fsync(file.fileno())
            os.replace(partial,self._journal)
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)

    def view(self):
        with self._lock:
            frame = self._scene["frame"]
            receipt = self._manual_offer[1] if self._manual_offer else self._place_offer
            bay = {}
            if receipt is not None and receipt["provenance"].get("bay_outline"):
                bay["bay_outline"] = receipt["provenance"]["bay_outline"]
            if self._bay_menu is not None:
                bay["bay_choices"] = self._bay_menu[0]["provenance"]["bay_choices"]
                if self._bay_menu[0]["provenance"]["reason"]=="unverified_bay_reference":
                    bay["bay_reference_uncertain"] = True
            return deepcopy({"state":self._state, "recording_id":frame["recording_id"], "frame_id":frame["frame_id"],
                             "image_sha256":self._scene["provenance"]["image_sha256"],
                             "request_id":self._request.get("request_id") if self._request else None,
                             "proposal_id":self._proposal, "result":self._result, "error":self._error,
                             "selected_by":self._selected_by, "turns":self._turns,
                             "can_request_area":self._supports_area(), **bay})

    def _supports_area(self):
        return self._scene["source"] == "predicted" and callable(getattr(self.model, "passenger_area", None))

    def _turn(self, role, text):
        frame = self._scene["frame"]
        return {"role":role,"text":text,"recording_id":frame["recording_id"],"frame_id":frame["frame_id"]}

    def _accept(self, outcome, selected_by):
        from .passenger import digest
        from . import manual_offlane, whole_bay

        req = self._request
        if not isinstance(outcome,dict) or outcome.get("request_id") != req["request_id"]:
            raise ValueError("Grounding outcome belongs to a different request")
        source = outcome.get("provenance")
        verifier = signed_place_verifier(source.get("method"),live=True) if isinstance(source,dict) else None
        if verifier is not None and selected_by != "model":
            raise ValueError("Only an automatic passenger instruction can offer a place-model result")
        if outcome.get("ok") is False:
            self._state, self._error = "error",deepcopy(outcome["error"])
            return
        if outcome.get("ok") is not True:
            raise ValueError("Invalid grounding outcome")
        info, result = outcome["provenance"], outcome["result"]
        method = info.get("method")
        self._waiting_place = method == "public_place_lookup_wait_v1"
        if (self._manual_submission_box is not None or method == "explicit_passenger_region") and getattr(
                self.model, "requires_verified_manual_surface", False):
            if method not in {manual_offlane.METHOD, whole_bay.MARK}:
                raise ValueError("The marked location requires parking evidence for this photo.")
        if method in {manual_offlane.METHOD, whole_bay.MARK}:
            if selected_by != "passenger" or self._manual_submission_box is None:
                raise ValueError("This location must come from an explicit mark on the current photo.")
            check = manual_offlane.verify if method == manual_offlane.METHOD else whole_bay.verify_current
            check(req, outcome, requested_box=self._manual_submission_box)
        if method == whole_bay.CHOICE:
            if (selected_by != "passenger" or self._bay_selection is None
                    or self._bay_selection != (digest(info.get("choice_source")), info.get("bay_id"))):
                raise ValueError("Select one of the bays offered for the current passenger instruction.")
            whole_bay.verify_current(req, outcome)
        if (info.get("image",{}).get("sha256") != self.scene["provenance"]["image_sha256"]
                or info.get("request_sha256") != digest(req)):
            raise ValueError("Grounding outcome is not bound to this scene and instruction")
        result = validate_target(result,list(result),self.scene["frame"])
        from .manual_mark import REQUEST_METHOD
        if info.get("method") == REQUEST_METHOD and (not self._requested_area or selected_by != "passenger"):
            raise ValueError("Only the passenger can request to mark the photo.")
        if verifier is not None:
            verifier(req, outcome)
            if result["status"] == "proposed":
                self._place_offer = deepcopy(outcome)
                self._place_offer_sha256 = digest(outcome)
        if method in whole_bay.MANUAL_METHODS and result["status"] == "proposed":
            self._manual_offer = (deepcopy(self._manual_submission_box), deepcopy(outcome), digest(outcome))
        if method in {whole_bay.AUTO, whole_bay.MARK} and result["status"]=="needs_clarification" and info.get("bay_choices"):
            self._bay_menu = (deepcopy(outcome), digest(outcome))
        self._state, self._result, self._selected_by = result["status"],result,selected_by
        if self._state == "proposed":
            self._proposal = uuid4().hex
            outline = info.get("bay_outline") or {}
            prompt = ("Is the highlighted point where you mean to get out?" if outline.get("extent")=="interior_point_only" else
                      "Is this visible parking area where you want to get out?" if outline.get("extent") == "visible_parking_surface" else
                      "Is the outlined bay where you want to get out?" if outline else
                      "Please check the pin on the photo. Is this where you want to get out?")
            self._turns.append(self._turn("assistant",prompt))
        elif self._state == "needs_clarification":
            self._turns.append(self._turn("assistant",result["clarification_question"]))
        else:
            self._turns.append(self._turn("assistant","I cannot see the requested reference in this frame. Can you describe another landmark?"))

    def _clear(self):
        self._proposal, self._result, self._error, self._selected_by = None, None, None, None
        self._place_offer = None
        self._place_offer_sha256 = None
        self._manual_offer = None
        self._manual_submission_box = None
        self._bay_menu = self._bay_selection = None
        self._waiting_place = False

    def _save_turn(self, action, detail):
        try:
            self._save(action,detail)
        except OSError:
            self._clear()
            self._state = "error"
            self._error = {"code":"audit_unavailable","message":"This turn could not be saved."}
            raise

    def _interpret_submission(self, request):
        return self.interpret_automatic(request)

    def interpret_automatic(self, request):
        """Compute a model decision without mutating the session or holding its UI lock."""
        return self.model.interpret(request), "model"

    def _accept_automatic(self, outcome, actor):
        self._accept(outcome, actor)

    def _prepare_vehicle_choice(self, point_to_vehicles):
        if point_to_vehicles is not False:
            raise ValueError("Vehicle selection is unavailable for this scene.")

    def _retries_lookup(self, text, original_transcript, replaces_request_id):
        return bool(self._waiting_place and self._state=="needs_clarification" and self._request
                    and replaces_request_id is None and self._request["text"]==text
                    and self._request.get("original_transcript")==original_transcript)

    def _submission_input(self, text, original_transcript, replaces_request_id, request_id):
        if replaces_request_id is not None and (not self._request or self._state in {"confirmed", "cancelled"}
                or self._request["request_id"] != replaces_request_id):
            raise ValueError("Only the current, unfinished description can be replaced.")
        fresh = replaces_request_id is not None or self._state in {"confirmed", "cancelled"}
        request = {"request_id":request_id, "scene":self.scene, "text":text,
                   "input_mode":"typed" if original_transcript is None else "voice",
                   "dialogue":[] if fresh else deepcopy(self._turns)}
        if self._retries_lookup(text, original_transcript, replaces_request_id):
            request["dialogue"] = deepcopy(self._request["dialogue"])
        if original_transcript is not None:
            request["original_transcript"] = original_transcript
        return request

    def _check_input(self, request, speech):
        clean = validate_request(request)
        if len(clean["text"]) > 4096:
            raise ValueError("Passenger turn exceeds the dialogue text limit")
        transcript = clean.get("original_transcript")
        if transcript is not None:
            if not isinstance(speech,dict) or speech.get("transcript") != transcript or not transcript.strip():
                raise ValueError("Voice requires its actual recognized transcript and speech receipt")
            wave_hash = speech.get("audio_sha256")
            if not isinstance(wave_hash,str) or len(wave_hash) != 64 or any(c not in "0123456789abcdef" for c in wave_hash):
                raise ValueError("Invalid speech audio digest")
            wav = self._audit/"audio"/(wave_hash+".wav")
            if not wav.is_file() or sha256(wav.read_bytes()).hexdigest() != wave_hash:
                raise ValueError("The original audio receipt does not match the saved WAV")
        elif speech is not None:
            raise ValueError("Speech receipt requires a voice transcript")
        return clean

    def prepare_automatic(self, text, *, speech=None, replaces_request_id=None):
        with self._lock:
            transcript = speech["transcript"] if speech else None
            req = self._submission_input(text, transcript, replaces_request_id, uuid4().hex)
            return self._check_input(req, speech)

    def cancel_pending(self, request, speech, replaces_request_id):
        with self._lock:
            current = self._submission_input(request["text"],request.get("original_transcript"),replaces_request_id,request["request_id"])
            if self._check_input(current, speech) != request:
                raise ValueError("The pending description has changed.")
            self._request = deepcopy(request)
            self._turns = deepcopy(request["dialogue"]) + [self._turn("user",request["text"])]
            self._clear()
            self._state = "cancelled"
            self._save_turn("cancelled",{"pending_request":request,"speech":speech})
            return self.view()

    def submit(self, text, *, original_transcript=None, speech=None, point_to_vehicles=False,
               mark_on_photo=False, replaces_request_id=None, _completed=None):
        from .passenger import operational_code

        with self._lock:
            request_id = _completed[0]["request_id"] if _completed is not None else uuid4().hex
            retry_id = (self._request["request_id"] if self._retries_lookup(text, original_transcript, replaces_request_id)
                        else None)
            request = self._submission_input(text, original_transcript, replaces_request_id, request_id)
            if _completed is not None and (point_to_vehicles or mark_on_photo or self._check_input(request,speech) != _completed[0]):
                raise ValueError("The completed check belongs to a different passenger request.")
            if type(mark_on_photo) is not bool or (mark_on_photo and (point_to_vehicles or not self._supports_area())):
                raise ValueError("Choose one available way to identify your spot.")
            self._prepare_vehicle_choice(point_to_vehicles)
            self._requested_area = mark_on_photo
            self._clear()
            self._turns = deepcopy(request["dialogue"])
            self._state = "idle"
            self._request = request
            outcome = None
            try:
                self._request = self._check_input(request,speech)
                self._turns.append(self._turn("user",text))
                if mark_on_photo:
                    from .manual_mark import question
                    outcome, selected_by = question(self._request), "passenger"
                elif _completed is not None:
                    outcome, selected_by = _completed[1:]
                else:
                    outcome, selected_by = self._interpret_submission(deepcopy(self._request))
                if mark_on_photo or point_to_vehicles:
                    self._accept(outcome, selected_by)
                else:
                    self._accept_automatic(outcome, selected_by)
            except Exception as exc:
                self._clear()
                self._state = "error"
                self._error = error_outcome(operational_code(exc),str(exc)[:4000] or type(exc).__name__,request_id)["error"]
            detail = {"request":self._request,"outcome":outcome,"speech":speech}
            if retry_id is not None:
                detail["retries_waiting_request_id"] = retry_id
            if mark_on_photo:
                detail["passenger_requested_area"] = True
            if replaces_request_id is not None:
                detail["replaces_request_id"] = replaces_request_id
            self._save_turn("submitted", detail)
            return self.view()

    def _can_mark_area(self):
        return self._request is not None and self._state in {"proposed", "needs_clarification", "insufficient_visual_evidence"}

    def select_area(self, box):
        from .passenger import operational_code

        with self._lock:
            if not self._can_mark_area():
                raise ValueError("Give a passenger instruction before marking its destination")
            self._clear()
            self._state = "idle"
            outcome = None
            try:
                self._manual_submission_box = deepcopy(box)
                outcome = self.model.passenger_area(deepcopy(self._request),deepcopy(box))
                self._accept(outcome,"passenger")
            except Exception as exc:
                self._clear()
                self._state = "error"
                self._error = {"code":operational_code(exc),"message":str(exc)[:4000]}
            finally:
                self._manual_submission_box = None
            self._save_turn("passenger_marked",{"requested_box":box,"outcome":outcome})
            return self.view()

    def select_bay(self, bay_id):
        from . import whole_bay
        from .passenger import digest
        with self._lock:
            if self._state != "needs_clarification" or self._bay_menu is None:
                raise ValueError("The highlighted alternatives are no longer available.")
            source, expected = self._bay_menu
            if digest(source) != expected:
                raise ValueError("The pictured alternatives changed.")
            outcome = whole_bay.choice_outcome(self._request, source, bay_id)
            self._clear()
            self._bay_selection = (expected, bay_id)
            try:
                self._accept(outcome, "passenger")
            finally:
                self._bay_selection = None
            self._save_turn("passenger_selected_bay", {"bay_id":bay_id,"prior_outcome_sha256":expected,"outcome":outcome})
            return self.view()

    def confirm(self, proposal_id):
        from .passenger import digest

        with self._lock:
            if self._state != "proposed" or not self._proposal or self._proposal != proposal_id:
                raise ValueError("This proposal is no longer available for confirmation")
            try:
                matches = _inspect_image(self._scene["frame"])[1] == self._scene["provenance"]["image_sha256"]
            except (ValueError, OSError):
                matches = False
            if not matches or digest(self._scene) != digest(self._request["scene"]):
                self._clear()
                self._state = "error"
                self._error = {"code":"image_changed","message":"The original frame changed before confirmation"}
                self._save("confirmation_blocked",{})
                raise ValueError(self._error["message"])
            detail = {"proposal_id":proposal_id,"selected_by":self._selected_by}
            if self._manual_offer is not None or (self._selected_by == "passenger" and
                    getattr(self.model, "requires_verified_manual_surface", False)):
                from . import manual_offlane, whole_bay
                try:
                    box, receipt, expected = self._manual_offer
                    if (self._selected_by != "passenger" or receipt["result"] != self._result
                            or expected != digest(receipt)):
                        raise ValueError("The marked location changed.")
                    check = manual_offlane.verify if receipt["provenance"]["method"] == manual_offlane.METHOD else whole_bay.verify_current
                    check(self._request, receipt, requested_box=box)
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    self._clear()
                    self._state = "error"
                    self._error = {"code":"manual_confirmation_blocked",
                                   "message":"The parking evidence for your marked location could not be verified."}
                    self._save("confirmation_blocked", {})
                    raise ValueError(self._error["message"]) from exc
                detail["verified_manual_outcome_sha256"] = expected
            if self._place_offer is not None or self._place_offer_sha256 is not None:
                try:
                    if (self._selected_by != "model" or not self._place_offer or
                            self._place_offer.get("result") != self._result or
                            self._place_offer_sha256 != digest(self._place_offer)):
                        raise ValueError("The displayed place proposal changed")
                    verifier = signed_place_verifier(self._place_offer["provenance"]["method"],live=True)
                    if verifier is None:
                        raise ValueError("Unknown place proposal source")
                    verifier(self._request, self._place_offer)
                except Exception as exc:
                    self._clear()
                    self._state = "error"
                    self._error = {"code":"place_confirmation_blocked",
                                   "message":"The evidence for this pin changed before confirmation."}
                    self._save("confirmation_blocked", {})
                    raise ValueError(self._error["message"]) from exc
                detail["verified_place_outcome_sha256"] = self._place_offer_sha256
            self._state = "confirmed"
            try:
                self._save("confirmed",detail)
            except OSError:
                self._state = "proposed"
                raise
            return self.view()

    def cancel(self):
        with self._lock:
            self._clear()
            self._state = "cancelled"
            self._save("cancelled",{})
            return self.view()
