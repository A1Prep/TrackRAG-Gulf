"""Scene-bound passenger dialogue and auditable explicit confirmation."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import threading
import tempfile
from uuid import uuid4

from .contracts import error_outcome, validate_request, validate_target
from .inputs import _inspect_image
from .passenger import digest, operational_code


class PassengerSession:
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
        self._lock = threading.RLock()
        self._turns = []
        self._state, self._proposal, self._result = "idle", None, None
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
            return deepcopy({"state":self._state, "recording_id":frame["recording_id"], "frame_id":frame["frame_id"],
                             "image_sha256":self._scene["provenance"]["image_sha256"],
                             "request_id":self._request.get("request_id") if self._request else None,
                             "proposal_id":self._proposal, "result":self._result, "error":self._error,
                             "selected_by":self._selected_by, "turns":self._turns})

    def _turn(self, role, text):
        frame = self._scene["frame"]
        return {"role":role,"text":text,"recording_id":frame["recording_id"],"frame_id":frame["frame_id"]}

    def _accept(self, outcome, selected_by):
        req = self._request
        if not isinstance(outcome,dict) or outcome.get("request_id") != req["request_id"]:
            raise ValueError("Grounding outcome belongs to a different request")
        if outcome.get("ok") is False:
            self._state, self._error = "error",deepcopy(outcome["error"])
            return
        if outcome.get("ok") is not True:
            raise ValueError("Invalid grounding outcome")
        info, result = outcome["provenance"], outcome["result"]
        if (info.get("image",{}).get("sha256") != self.scene["provenance"]["image_sha256"]
                or info.get("request_sha256") != digest(req)):
            raise ValueError("Grounding outcome is not bound to this scene and instruction")
        result = validate_target(result,list(result),self.scene["frame"])
        self._state, self._result, self._selected_by = result["status"],result,selected_by
        if self._state == "proposed":
            self._proposal = uuid4().hex
            self._turns.append(self._turn("assistant","Please check the highlighted area. Is this where you want to get out?"))
        elif self._state == "needs_clarification":
            self._turns.append(self._turn("assistant",result["clarification_question"]))
        else:
            self._turns.append(self._turn("assistant","I cannot see the requested reference in this frame. Can you describe another landmark?"))

    def _clear(self):
        self._proposal, self._result, self._error, self._selected_by = None, None, None, None

    def _save_turn(self, action, detail):
        try:
            self._save(action,detail)
        except OSError:
            self._clear()
            self._state = "error"
            self._error = {"code":"audit_unavailable","message":"This turn could not be saved."}
            raise

    def _interpret_submission(self, request):
        return self.model.interpret(request), "model"

    def submit(self, text, *, original_transcript=None, speech=None):
        with self._lock:
            prior = self._state
            self._clear()
            if prior in ("cancelled","confirmed"):
                self._turns = []
            self._state = "idle"
            request_id = uuid4().hex
            self._request = {"request_id":request_id,"scene":self.scene,"text":text,
                             "input_mode":"typed" if original_transcript is None else "voice",
                             "dialogue":deepcopy(self._turns)}
            if original_transcript is not None:
                self._request["original_transcript"] = original_transcript
            outcome = None
            try:
                self._request = validate_request(self._request)
                if len(text) > 4096:
                    raise ValueError("Passenger turn exceeds the dialogue text limit")
                if original_transcript is not None:
                    if (not isinstance(speech,dict) or speech.get("transcript") != original_transcript
                            or not original_transcript.strip()):
                        raise ValueError("Voice requires its actual recognized transcript and speech receipt")
                    wave_hash = speech.get("audio_sha256")
                    if not isinstance(wave_hash,str) or len(wave_hash) != 64 or any(
                            c not in "0123456789abcdef" for c in wave_hash):
                        raise ValueError("Invalid speech audio digest")
                    wav = self._audit/"audio"/(wave_hash+".wav")
                    if not wav.is_file() or sha256(wav.read_bytes()).hexdigest() != wave_hash:
                        raise ValueError("The original audio receipt does not match the saved WAV")
                elif speech is not None:
                    raise ValueError("Speech receipt requires a voice transcript")
                self._turns.append(self._turn("user",text))
                outcome, selected_by = self._interpret_submission(deepcopy(self._request))
                self._accept(outcome, selected_by)
            except Exception as exc:
                self._clear()
                self._state = "error"
                self._error = error_outcome(operational_code(exc),str(exc)[:4000] or type(exc).__name__,request_id)["error"]
            self._save_turn("submitted",{"request":self._request,"outcome":outcome,"speech":speech})
            return self.view()

    def select_area(self, box):
        with self._lock:
            if self._request is None or self._state in ("idle","cancelled","confirmed","error"):
                raise ValueError("Give a passenger instruction before marking its destination")
            self._clear()
            self._state = "idle"
            outcome = None
            try:
                outcome = self.model.passenger_area(deepcopy(self._request),box)
                self._accept(outcome,"passenger")
            except Exception as exc:
                self._state = "error"
                self._error = {"code":operational_code(exc),"message":str(exc)[:4000]}
            self._save_turn("passenger_marked",{"requested_box":box,"outcome":outcome})
            return self.view()

    def confirm(self, proposal_id):
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
            self._state = "confirmed"
            try:
                self._save("confirmed",{"proposal_id":proposal_id,"selected_by":self._selected_by})
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
