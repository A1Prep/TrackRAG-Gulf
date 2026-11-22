"""Use the parking model in the ordinary passenger session when configured by the operator."""
from copy import deepcopy

from . import passenger_parking as parking
from .inputs import _inspect_image
from .passenger import digest
from .passenger_pair_choice import PassengerSession


class ParkingPassengerSession(PassengerSession):
    def __init__(self, scene, model, *, audit_dir, parking_backend, parking_artifacts=None):
        self._parking_backend = parking_backend
        self._parking_artifacts = parking_artifacts
        self._parking_offer = self._parking_check = None
        self._parking_recovery = False
        super().__init__(scene, model, audit_dir=audit_dir)

    def _clear(self):
        super()._clear()
        self._parking_offer = self._parking_check = None
        self._parking_recovery = False

    def _interpret_submission(self, request):
        if self._passenger_requested or not parking.parking_instruction(request):
            return super()._interpret_submission(request)
        try:
            response = self._parking_backend.interpret(deepcopy(request))
        except ValueError:
            response = parking.dispatch_failure(request, "parking_evidence_invalid")
        except Exception:
            response = parking.dispatch_failure(request, "parking_backend_unavailable")
        else:
            try:
                if not isinstance(response, dict) or response.get("provenance", {}).get("method") != parking.METHOD:
                    raise ValueError("The backend did not return a parking-model receipt")
                parking.verify_response(request, response, artifacts=self._parking_artifacts)
            except Exception:
                response = parking.dispatch_failure(request, "parking_evidence_invalid", rejected=response)
        self._parking_check = (digest(request), digest(response))
        return deepcopy(response), parking.ACTOR

    def _accept(self, outcome, selected_by):
        source = outcome.get("provenance") if isinstance(outcome, dict) else None
        method = source.get("method") if isinstance(source, dict) else None
        if (method in {parking.METHOD, parking.FAILURE_METHOD}) != (selected_by == parking.ACTOR):
            raise ValueError("Parking response and its actual selection source disagree")
        if selected_by == parking.ACTOR and self._parking_check != (digest(self._request), digest(outcome)):
            raise ValueError("This parking response was not checked for this active submission")
        super()._accept(outcome, selected_by)
        if selected_by != parking.ACTOR:
            return
        if self._state == "proposed":
            self._parking_offer = deepcopy(outcome)
        elif self._state == "error":
            try:
                self._parking_recovery = _inspect_image(self._scene["frame"])[1] == self._scene["provenance"]["image_sha256"]
            except (OSError, ValueError):
                self._parking_recovery = False

    def _can_mark_area(self):
        return super()._can_mark_area() or (self._state == "error" and self._parking_recovery)

    def _active(self):
        return super()._active() or (self._state == "error" and self._parking_recovery and self._supports_pointing())

    def view(self):
        with self._lock:
            view = super().view()
            if self._state == "error" and self._parking_recovery:
                view["parking_recovery"] = True
            return view

    def confirm(self, proposal_id):
        with self._lock:
            if self._state == "proposed" and self._selected_by == parking.ACTOR and proposal_id == self._proposal:
                try:
                    if not self._parking_offer or self._parking_offer["result"] != self._result:
                        raise ValueError("The displayed parking result changed")
                    parking.verify_response(self._request, self._parking_offer, artifacts=self._parking_artifacts)
                except Exception as error:
                    self._clear()
                    self._state = "error"
                    self._error = {"code":"parking_confirmation_blocked",
                                   "message":"The evidence for this parking proposal changed before confirmation."}
                    self._save("confirmation_blocked", {"method":parking.METHOD, "error":deepcopy(self._error)})
                    raise ValueError(self._error["message"]) from error
            return super().confirm(proposal_id)

    def _save(self, action, detail):
        if action == "confirmed" and self._selected_by == parking.ACTOR and self._parking_offer:
            detail = {**detail, "verified_parking_outcome_sha256": digest(self._parking_offer)}
        return super()._save(action, detail)
