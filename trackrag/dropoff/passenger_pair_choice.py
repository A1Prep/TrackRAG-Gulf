"""Passenger-selected vehicles with a learned road-area proposal on the original image."""

from copy import deepcopy
import re

from .contracts import error_outcome, validate_request, validate_target
from .data import inference_input_sha256
from .interaction import PassengerSession as OriginalPassengerSession
from .landmarks import resolve_image
from .pair_resolution import explicit_pair_quotes
from .passenger import PassengerGrounder, digest, operational_code, proposed, unresolved
from .projected_pair import GEOMETRY, MOTOR, Ineligible, center_x, neighbor_geometry, supported_patch
from .road_cache import ROAD_RECIPE_SHA256


METHOD = "explicit_passenger_landmark_pair"
PLACEMENTS = {"between", "behind"}
VEHICLE = re.compile(
    r"\b(?:cars?|sedans?|suvs?|cabs?|taxis?|vehicles?|hatchbacks?|coupes?|vans?|minivans?|"
    r"trucks?|lorr(?:y|ies)|pickups?|coaches|coach|buses|bus|motorbikes?|motorcycles?|scooters?)\b",
    re.I,
)
VEHICLE_PLURAL = re.compile(
    r"\b(?:cars|sedans|suvs|cabs|taxis|vehicles|hatchbacks|coupes|vans|minivans|"
    r"trucks|lorries|pickups|coaches|buses|motorbikes|motorcycles|scooters)\b",
    re.I,
)
OTHER_LOCATION = re.compile(
    r"\b(?:gate|entrance|building|bridge|bus stop|shop|tree|wall|fence|sign|lamp|"
    r"curb|kerb|median|divider|barrier|pedestrian|cyclist|person|people|door)s?\b",
    re.I,
)
STRUCTURE = re.compile(
    r"\b(?:bus\s+(?:stops?|shelters?|stations?)|car\s+parks?|truck\s+stops?|taxi\s+(?:ranks?|stands?))\b", re.I
)
NEG = re.compile(r"\b(?:not|never|avoid|except|rather than|instead of)\b[^,;.!?]*$|n['’]t\b[^,;.!?]*$", re.I)
RESET = re.compile(r"\b(?:actually|instead(?!\s+of)|forget (?:it|that)|change (?:it|that)|rather)\b", re.I)
NEW_LOCATION = re.compile(
    r"\b(?:drop|let)\s+(?:me|us)\s+(?:off\s+|out\s+)?(?:at|by|beside|near|under|here|there)"
    r"|\b(?:park|stop|pull over|get out|(?:drop|let) (?:me|us)(?: off| out)?)\s+"
    r"(?:at|by|beside|under|near|here|there|next to|in front of|across from)\b",
    re.I,
)
ELLIPTICAL_LOCATION = re.compile(
    r"^\s*(?:(?:no|nah|please|actually|rather)[,\s]+)*(?:(?:over|right)\s+)?"
    r"(?:next to|across from|in front of|at|by|beside|near|under|toward|here|there)\b",
    re.I,
)
REPLACED_NOUN = re.compile(r"^\s*(?:(?:no|nah|actually|rather)[,\s]+)+(?:that|this|the)\s+", re.I)
RECOVERABLE = {
    "invalid_model_output",
    "invalid_verification_output",
    "timeout",
    "resource_unavailable",
    "generation_failed",
    "grounding_failed",
    "dependency_unavailable",
    "invalid_pair_selection",
}


class InvalidSelection(ValueError):
    """The passenger can retry a different current choice."""


def pair_intent(request):
    """Inspect passenger words only. A replacement or a negated pair must retire its choices."""
    active = False
    for turn in [v["text"] for v in request["dialogue"] if v["role"] == "user"] + [request["text"]]:
        matches = list(re.finditer(r"\bbetween\b", turn, re.I))
        if matches:
            use = matches[-1]
            replacement = RESET.search(turn, use.end())
            if NEG.search(turn[: use.start()]) or (replacement and turn[replacement.end() :].strip(" ,.;:!?")):
                active = False
                continue
            tail = turn[use.end() :].strip(" ,.?!")
            quotes = explicit_pair_quotes(turn)
            if quotes:
                # "the gate by the car" still names the gate. An object accompanying
                # a vehicle head noun, such as "the sedan by the median", is valid.
                active = True
                for quote in quotes:
                    vehicle = VEHICLE.search(quote["text"])
                    structure = STRUCTURE.search(quote["text"])
                    if (
                        vehicle is None
                        or OTHER_LOCATION.search(quote["text"][: vehicle.start()])
                        or (structure and structure.start() <= vehicle.start())
                    ):
                        active = False
                        break
            else:
                plural = VEHICLE_PLURAL.search(tail)
                other = OTHER_LOCATION.search(tail)
                active = bool(
                    (
                        plural
                        and (not other or plural.start() < other.start())
                        and not re.search(r"\b(?:and|or)\b", tail[plural.end() :], re.I)
                    )
                    or (
                        active and re.fullmatch(r"(?:them|(?:those|these|the) two)(?:,?\s+please)?", tail, re.I)
                    )
                )
                if re.search(r"\b(?:or|except|not)\b", tail, re.I):
                    active = False
        elif (
            RESET.search(turn)
            or NEW_LOCATION.search(turn)
            or ELLIPTICAL_LOCATION.search(turn)
            or (REPLACED_NOUN.search(turn) and (OTHER_LOCATION.search(turn) or not VEHICLE.search(turn)))
        ):
            active = active and bool(
                re.search(
                    r"\b(?:both(?: (?:cars|vehicles|of them))?|the pair|the two|these two|those two)\b",
                    turn,
                    re.I,
                )
            )
        elif active and re.search(
            r"\b(?:no longer|forget|cancel|never mind|(?:do not|don['’]t) want that)\b", turn, re.I
        ):
            active = False
    return active


def current_tracks(request):
    scene = request["scene"]
    return sorted(
        (t for t in scene["tracks"] if t["frame_id"] == scene["frame"]["frame_id"]),
        key=lambda t: (t["bbox_xyxy"][0], t["bbox_xyxy"][1], t["track_id"]),
    )


def options(request):
    if request["scene"]["source"] != "predicted" or not pair_intent(request):
        return []
    selected = [
        {k: deepcopy(t[k]) for k in ("track_id", "class_name", "bbox_xyxy")}
        for t in current_tracks(request)
        if t["class_name"] in MOTOR
    ]
    return selected if len(selected) >= 2 else []


def geometry_for(observations, tracks, size, placement):
    a, b = sorted((t["bbox_xyxy"] for t in observations), key=center_x)
    gap = b[0] - a[2]
    if gap <= 0:
        if placement != "behind":
            raise Ineligible("no_visible_between_gap")
        return neighbor_geometry(observations, tracks, size), "behind_weakly_overlapping_vehicles"
    heights = (a[3] - a[1], b[3] - b[1])
    if abs(a[3] - b[3]) > max(heights):
        raise Ineligible("objects_at_incompatible_projected_depth")
    if placement == "behind":
        corridor = [center_x(a), max(a[3], b[3]), center_x(b), min(size[1], max(a[3], b[3]) + max(heights))]
    else:
        corridor = [
            a[2],
            max(0, min(a[3], b[3]) - max(heights)),
            b[0],
            min(size[1], max(a[3], b[3]) + max(heights)),
        ]
    return {
        "regression_inputs_xyxy": [a, b],
        "center_corridor_xyxy": corridor,
    }, f"{placement}_separated_vehicles"


def choose(model, request, track_ids, placement):
    """The passenger supplies only two current identities and a relation; the region is estimated."""
    request = validate_request(request)
    if (
        not isinstance(track_ids, (list, tuple))
        or len(track_ids) != 2
        or any(type(i) is not int for i in track_ids)
        or track_ids[0] == track_ids[1]
        or not isinstance(placement, str)
        or placement not in PLACEMENTS
    ):
        raise InvalidSelection("Choose two different current vehicles and choose Between or Just behind.")
    available = {r["track_id"] for r in options(request)}
    if not set(track_ids) <= available:
        raise InvalidSelection("These vehicles are unavailable for this passenger request and original frame.")
    if model.geometry.sha256 != GEOMETRY:
        raise ValueError("The selected vehicles require the registered learned two-vehicle geometry.")
    tracks = current_tracks(request)
    by_id = {t["track_id"]: t for t in tracks}
    selected = [by_id[i] for i in track_ids]
    frame = request["scene"]["frame"]
    image, binding = resolve_image(request, model.config)
    with image:
        choice = {
            "version": 1,
            "request_id": request["request_id"],
            "recording_id": frame["recording_id"],
            "frame_id": frame["frame_id"],
            "image_sha256": binding["sha256"],
            "request_sha256": digest(request),
            "placement": placement,
            "track_ids": list(track_ids),
            "selected_observations": selected,
        }
        evidence = {
            "method": METHOD,
            "independent_model_proposal": False,
            "landmark_selection": "passenger",
            "region_estimation": "learned_pair_regression_with_current_road_support",
            "image": binding,
            "request_sha256": digest(request),
            "inference_input_sha256": inference_input_sha256(request, binding["sha256"]),
            "passenger_selection": choice,
            "selection_sha256": digest(choice),
            "geometry_sha256": model.geometry.sha256,
            "current_observations_sha256": digest(tracks),
        }
        try:
            geometry, mode = geometry_for(selected, tracks, image.size, placement)
            evidence["geometry_mode"] = mode
            try:
                mask = model.road.mask(image, binding["sha256"])
            except Ineligible as error:
                raise RuntimeError("The current original's registered road mask could not be loaded") from error
            patch = supported_patch(mask, image.size, model.geometry.pair_regression, geometry, tracks)
        except Ineligible as error:
            reason = str(error)
            if reason in {"unsupported_road_mask", "no_prediction_from_trained_pair_head"}:
                raise RuntimeError(
                    "The current road model or fitted geometry returned an invalid result: " + reason
                ) from error
            evidence["withheld_reason"] = reason
            if reason == "no_visible_between_gap":
                question = "Their boxes overlap in this view. Do you mean the road just behind both vehicles, or another area?"
            else:
                question = "I couldn't place a visible road patch at that position relative to both vehicles. Could you choose another pair or mark the road area?"
            result = unresolved(question)
        else:
            evidence["road_geometry"] = {**geometry, **patch}
            result = proposed(patch["point"], patch["box"], [t["bbox_xyxy"] for t in selected])
    validate_target(result, list(result), frame)
    return {"ok": True, "request_id": request["request_id"], "result": result, "provenance": evidence}


def validate_choice_evidence(request, outcome, track_ids, placement):
    """Check the exact passenger action and original source before exposing a selectable proposal."""
    source, frame = outcome["provenance"], request["scene"]["frame"]
    photo = request["scene"]["provenance"]["image_sha256"]
    if (
        not isinstance(track_ids, (tuple, list))
        or len(track_ids) != 2
        or any(type(i) is not int for i in track_ids)
        or len(set(track_ids)) != 2
        or not isinstance(placement, str)
        or placement not in PLACEMENTS
    ):
        raise ValueError("Invalid passenger landmark-selection receipt")
    tracks = current_tracks(request)
    by_id = {t["track_id"]: t for t in tracks}
    available = {t["track_id"] for t in options(request)}
    if not set(track_ids) <= available:
        raise ValueError("Passenger landmark choice is absent from the current frame")
    expected = {
        "version": 1,
        "request_id": request["request_id"],
        "recording_id": frame["recording_id"],
        "frame_id": frame["frame_id"],
        "image_sha256": photo,
        "request_sha256": digest(request),
        "placement": placement,
        "track_ids": list(track_ids),
        "selected_observations": [by_id[i] for i in track_ids],
    }
    if (
        outcome.get("request_id") != request["request_id"]
        or source.get("method") != METHOD
        or source.get("geometry_sha256") != GEOMETRY
        or source.get("landmark_selection") != "passenger"
        or source.get("independent_model_proposal") is not False
        or source.get("image", {}).get("sha256") != photo
        or source.get("request_sha256") != digest(request)
        or source.get("inference_input_sha256") != inference_input_sha256(request, photo)
        or source.get("passenger_selection") != expected
        or source.get("selection_sha256") != digest(expected)
        or source.get("current_observations_sha256") != digest(tracks)
    ):
        raise ValueError(
            "Passenger landmark choice differs from its request, current observations, original image or learned model"
        )
    target = validate_target(outcome["result"], list(outcome["result"]), frame)
    if target["status"] == "proposed":
        geometry = source.get("road_geometry") or {}
        size = (frame["width"], frame["height"])
        expected_geometry, mode = geometry_for(expected["selected_observations"], tracks, size, placement)
        x, y, c, d = target["destination_box_xyxy"]
        px, py = target["destination_point_xy"]
        a, b, right, bottom = expected_geometry["center_corridor_xyxy"]
        if (
            source.get("geometry_mode") != mode
            or any(geometry.get(k) != v for k, v in expected_geometry.items())
            or not (a <= x <= px <= c <= right and b <= y <= py <= d <= bottom)
            or any(
                x < t["bbox_xyxy"][2]
                and c > t["bbox_xyxy"][0]
                and y < t["bbox_xyxy"][3]
                and d > t["bbox_xyxy"][1]
                for t in tracks
            )
        ):
            raise ValueError(
                "Passenger-selected estimate is outside its road corridor or intersects a current road user"
            )
        if (
            target.get("anchor_boxes_xyxy") != [by_id[i]["bbox_xyxy"] for i in track_ids]
            or geometry.get("point") != target["destination_point_xy"]
            or geometry.get("box") != target["destination_box_xyxy"]
            or geometry.get("road_recipe_sha256") != ROAD_RECIPE_SHA256
            or not isinstance(geometry.get("road_mask_sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", geometry["road_mask_sha256"])
        ):
            raise ValueError("Passenger-assisted proposal lacks its matching geometry or road evidence")
    elif (
        target["status"] != "needs_clarification"
        or not source.get("withheld_reason")
        or "road_geometry" in source
    ):
        raise ValueError("Incomplete unresolved passenger choice")
    return expected


def replay_choice_evidence(request, outcome, geometry_path, cache):
    """Cold-recompute the learned estimate using the current original's authenticated private road mask."""
    from types import SimpleNamespace
    from .anchor_geometry import AnchorGeometry
    from .model import GroundingConfig
    from .road_cache import read_mask

    mask_read = False

    def verified_mask(image, fingerprint):
        nonlocal mask_read
        mask = read_mask(cache, fingerprint, image.size)
        if mask is None:
            raise FileNotFoundError(
                "A registered road mask for this original is required for full assisted-point verification"
            )
        mask_read = True
        return mask

    model = SimpleNamespace(
        config=GroundingConfig(),
        geometry=AnchorGeometry.load(geometry_path, expected_sha=GEOMETRY),
        road=SimpleNamespace(mask=verified_mask),
    )
    action = outcome["provenance"]["passenger_selection"]
    computed = choose(model, request, action["track_ids"], action["placement"])
    expected = {key: val for key, val in outcome["provenance"].items() if key != "automatic_submission"}
    if computed["provenance"] != expected or computed["result"] != outcome["result"]:
        raise ValueError(
            "Saved selected-pair result differs from its exact fitted model and current road mask replay"
        )
    return {
        "current_image_and_tracks_checked": True,
        "learned_geometry_and_mask_replayed": mask_read,
        "decision_reproduced": True,
        "fitted_geometry_sha256": GEOMETRY,
        "road_mask_needed": mask_read,
    }


class PassengerSession(OriginalPassengerSession):
    def __init__(self, scene, model, *, audit_dir):
        self._original_submission = None
        super().__init__(scene, model, audit_dir=audit_dir)

    def _save(self, action, detail):
        super()._save(action, detail)
        if action == "submitted":
            raw = detail.get("outcome")
            self._original_submission = {
                "request_id": detail["request"].get("request_id"),
                "request_sha256": digest(detail["request"]),
                "original_outcome_sha256": digest(raw) if raw else None,
                "original_automatic_state": self._state,
            }

    def _active(self):
        capable = callable(getattr(self.model, "passenger_pair", None)) or isinstance(
            self.model, PassengerGrounder
        )
        return (
            capable
            and self._request is not None
            and (
                self._state in {"proposed", "needs_clarification", "insufficient_visual_evidence"}
                or (self._state == "error" and self._error and self._error.get("code") in RECOVERABLE)
            )
        )

    def view(self):
        with self._lock:
            state = super().view()
            state["pair_choices"] = options(self._request) if self._active() else []
            return state

    def select_pair(self, track_ids, placement):
        with self._lock:
            if not self._active() or not options(self._request):
                raise ValueError("Give an active passenger instruction between two visible vehicles first.")
            self._clear()
            self._state = "idle"
            outcome = None
            try:
                if callable(getattr(self.model, "passenger_pair", None)):
                    outcome = self.model.passenger_pair(deepcopy(self._request), track_ids, placement)
                else:
                    with self.model.lock:
                        outcome = choose(self.model, deepcopy(self._request), track_ids, placement)
                if not isinstance(outcome, dict) or outcome.get("provenance", {}).get("method") != METHOD:
                    raise ValueError("This choice did not return a passenger-selected landmark result.")
                if outcome.get("ok") is True:
                    validate_choice_evidence(self._request, outcome, track_ids, placement)
                if not self._original_submission or self._original_submission["request_sha256"] != digest(
                    self._request
                ):
                    raise ValueError("The original passenger submission is unavailable for this selection.")
                outcome["provenance"]["automatic_submission"] = deepcopy(self._original_submission)
                self._accept(outcome, "passenger_landmarks")
            except Exception as error:
                self._clear()
                self._state = "error"
                code = (
                    "invalid_pair_selection" if isinstance(error, InvalidSelection) else operational_code(error)
                )
                outcome = error_outcome(
                    code,
                    str(error)[:4000] or type(error).__name__,
                    self._request["request_id"],
                    {"method": METHOD, "independent_model_proposal": False},
                )
                self._error = outcome["error"]
            saved_ids = (
                list(track_ids)
                if isinstance(track_ids, (tuple, list))
                and len(track_ids) <= 2
                and all(type(i) is int for i in track_ids)
                else None
            )
            self._save_turn(
                "passenger_selected_pair",
                {
                    "track_ids": saved_ids,
                    "placement": placement if isinstance(placement, str) and len(placement) <= 24 else None,
                    "outcome": outcome,
                },
            )
            return self.view()
