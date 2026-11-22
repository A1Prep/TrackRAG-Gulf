"""Passenger-selected vehicles with a learned road-area proposal on the original image."""

from copy import deepcopy
from contextlib import nullcontext
import re

from . import between_policy, between_road
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
    r"\b(?:cars?|sedans?|suvs?|crossovers?|cabs?|taxis?|vehicles?|hatchbacks?|coupes?|vans?|minivans?|"
    r"trucks?|lorr(?:y|ies)|pickups?|coaches|coach|buses|bus|motorbikes?|motorcycles?|scooters?)\b",
    re.I,
)
VEHICLE_PLURAL = re.compile(
    r"\b(?:cars|sedans|suvs|crossovers|cabs|taxis|vehicles|hatchbacks|coupes|vans|minivans|"
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
PAIR_PRONOUN = re.compile(r"^\s*(?:(?:those|these|the)\s+(?:two|2)|them(?:\s+both)?|both\s+of\s+them)\b", re.I)
LOCATOR = re.compile(r"\b(?:beside|near|behind|by|next to|in front of|across from)\b", re.I)
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


def vehicle_head(text):
    """Detect what was named, ignoring objects mentioned only in a trailing locator or appositive."""
    vehicle = VEHICLE.search(text)
    if not vehicle:
        return False
    before = text[:vehicle.start()]
    structure = STRUCTURE.search(text)
    return (not OTHER_LOCATION.search(before) and not LOCATOR.search(before)
            and not (structure and structure.start() <= vehicle.start()))


def named_vehicle_pair(text, *, prefatory=False):
    """Recognize two coordinated vehicle noun phrases without interpreting a caption or inventing IDs."""
    for joining in re.finditer(r"\s+and\s+", text, re.I):
        first, second = text[:joining.start()], text[joining.end():]
        if prefatory:
            # A scene preface can introduce the pair before a separate 'between those two' request.
            first = re.split(r"[.!?;:\n]", first)[-1]
            first = re.split(r"\b(?:I see|we see|there (?:is|are)|there['’]s|you see)\s+", first, flags=re.I)[-1]
            second = re.split(r"[.!?;:\n]", second, maxsplit=1)[0]
        if re.search(r"\b(?:or|except)\b", first+" "+second, re.I):
            continue
        # A third named alternative/landmark cannot establish which pair is intended.
        if any(re.search(r"\band\s+(?:the|that|this|a|an|another)\b", part, re.I)
               for part in (first, second)):
            continue
        if vehicle_head(first) and vehicle_head(second):
            return True
    return False


def pair_intent(request):
    """Inspect passenger words only. A replacement or a negated pair must retire its choices."""
    from .pair_dialogue import unpointed_reference
    active = False
    for turn in [v["text"] for v in request["dialogue"] if v["role"] == "user"] + [request["text"]]:
        if unpointed_reference(turn):
            active = True
            continue
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
                active = all(vehicle_head(quote["text"]) for quote in quotes)
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
                    or named_vehicle_pair(tail)
                    or (PAIR_PRONOUN.match(tail) and named_vehicle_pair(turn[:use.start()], prefatory=True))
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


def options(request, *, passenger_requested=False):
    if request["scene"]["source"] != "predicted":
        return []
    if not (passenger_requested or pair_intent(request)):
        from .passenger_parking import parking_instruction
        if not parking_instruction(request):
            return []
    selected = [
        {k: deepcopy(t[k]) for k in ("track_id", "class_name", "bbox_xyxy")}
        for t in current_tracks(request)
        if t["class_name"] in MOTOR
    ]
    return selected if len(selected) >= 2 else []


def landmark_error_guidance(request, outcome):
    """Allow a passenger to locate two vehicles after the model returned unusable detector IDs."""
    from .inputs import _inspect_image
    from .landmarks import InvalidLandmarkIDs

    if not isinstance(outcome, dict) or outcome.get("ok") is not False or outcome.get("request_id") != request["request_id"]:
        return None
    source, error = outcome.get("provenance") or {}, outcome.get("error") or {}
    if (source.get("method") != "qwen_landmark_region" or error.get("message") != InvalidLandmarkIDs.message
            or error.get("code") not in {"invalid_model_output", "invalid_input_or_output"}
            or request["scene"]["source"] != "predicted" or not between_road.requested(request)):
        return None
    original = request["scene"]["provenance"]["image_sha256"]
    if (source.get("request_sha256", digest(request)) != digest(request)
            or source.get("image", {}).get("sha256", original) != original):
        return None
    try:
        if _inspect_image(request["scene"]["frame"])[1] != original:
            return None
    except (ValueError, OSError):
        return None
    if options(request):
        return "I couldn’t match both vehicles. Which two do you mean? Select their outlines or mark the intended road area."
    return "I couldn’t match both vehicles, and this photo has fewer than two vehicle outlines. Could you mark the road area or describe another landmark?"


def pointing_question(request):
    """Record an explicit passenger interaction without attributing it to automatic interpretation."""
    from .inputs import _inspect_image

    request = validate_request(request)
    scene = request["scene"]
    _, photo = _inspect_image(scene["frame"])
    if photo != scene.get("provenance", {}).get("image_sha256"):
        raise ValueError("The photograph changed before the vehicles could be identified.")
    candidates = options(request, passenger_requested=True)
    message = ("Which two vehicles do you mean? Select them on the photo and choose where you want to get out."
               if candidates else "I can’t offer two vehicle outlines in this frame. Could you describe another landmark or mark the area?")
    return {"ok":True, "request_id":request["request_id"], "result":unresolved(message),
            "provenance":{"method":"passenger_requested_pointing", "independent_model_proposal":False,
                          "request_sha256":digest(request), "inference_input_sha256":inference_input_sha256(request,photo),
                          "image":{"sha256":photo}, "available_vehicles_sha256":digest(candidates), "native_model_calls":0}}


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


def choose(model, request, track_ids, placement, *, passenger_requested=False, _legacy_between=False):
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
    available = {r["track_id"] for r in options(request, passenger_requested=passenger_requested)}
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
        if passenger_requested:
            choice["passenger_requested_pointing"] = True
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
        current_between = placement == "between" and not _legacy_between
        if current_between:
            evidence["between_road_version"] = between_road.VERSION
            evidence["region_estimation"] = "learned_region_with_visible_road_edge_and_ground_contacts"
            evidence.update(between_policy.evidence(model))
        try:
            if current_between:
                geometry = between_policy.preflight(model, selected, image.size)
                mode = between_road.VERSION
            else:
                geometry, mode = geometry_for(selected, tracks, image.size, placement)
            evidence["geometry_mode"] = mode
            try:
                mask = model.road.mask(image, binding["sha256"])
            except Ineligible as error:
                raise RuntimeError("The current original's registered road mask could not be loaded") from error
            if current_between:
                patch = between_policy.estimate(model, mask, image, selected, tracks, request["text"],image_sha=binding["sha256"])
                if patch["guard"] == between_policy.BAY_GUARD:
                    geometry = {}
                    evidence["geometry_mode"] = patch["guard"]
                    evidence["region_estimation"] = "learned_region_with_visible_bay_edge_and_ground_contacts"
            else:
                patch = supported_patch(mask, image.size, model.geometry.pair_regression, geometry, tracks)
        except Ineligible as error:
            reason = str(error)
            if reason in {"unsupported_road_mask", "no_prediction_from_trained_pair_head"}:
                raise RuntimeError(
                    "The current road model or fitted geometry returned an invalid result: " + reason
                ) from error
            evidence.update(between_policy.refusal(error))
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


def validate_choice_evidence(request, outcome, track_ids, placement, *, passenger_requested=False, live=False):
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
    available = {t["track_id"] for t in options(request, passenger_requested=passenger_requested)}
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
    if passenger_requested:
        expected["passenger_requested_pointing"] = True
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
    current_between = placement == "between" and (
        live or source.get("between_road_version") is not None
    )
    if current_between and source.get("between_road_version") != between_road.VERSION:
        raise ValueError("The passenger-selected between proposal lacks its current road guard")
    enabled = between_policy.check_mode(source.get("between_road_mode",between_policy.EDGE))
    if target["status"] == "proposed":
        geometry = source.get("road_geometry") or {}
        size = (frame["width"], frame["height"])
        if current_between:
            between_policy.validate_patch(geometry, expected["selected_observations"], tracks, size,
                                          target["destination_point_xy"], target["destination_box_xyxy"],enabled)
            mode = geometry["guard"]
            expected_geometry = (between_road.contact_geometry(expected["selected_observations"], size)
                                 if mode == between_road.VERSION else geometry)
        else:
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


def replay_choice_evidence(request, outcome, geometry_path, cache, *, passenger_requested=False):
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
        road=SimpleNamespace(mask=verified_mask,cache=cache,image_hash=request["scene"]["provenance"]["image_sha256"]),
        between_road_mode=between_policy.check_mode(outcome["provenance"].get("between_road_mode",between_policy.EDGE)),
    )
    action = outcome["provenance"]["passenger_selection"]
    legacy = action["placement"] == "between" and "between_road_version" not in outcome["provenance"]
    computed = choose(model, request, action["track_ids"], action["placement"], passenger_requested=passenger_requested,
                      _legacy_between=legacy)
    submission_key = "pointing_submission" if passenger_requested else "automatic_submission"
    expected = {key: val for key, val in outcome["provenance"].items() if key != submission_key}
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
        self._passenger_requested = False
        self._area_landmarks = None
        self._landmark_error_guidance = None
        super().__init__(scene, model, audit_dir=audit_dir)

    def _clear(self):
        super()._clear()
        self._landmark_error_guidance = None

    def _can_mark_area(self):
        return super()._can_mark_area() or (self._state == "error" and self._landmark_error_guidance is not None)

    def _landmarks_awaiting_area(self, base, outcome, actor):
        """Keep two known outlines for the marking UI, never an inferred destination or a confirmation."""
        source = outcome["provenance"]
        request = self._request
        if actor == "passenger_landmarks":
            choice = source.get("passenger_selection") or {}
            if not choice:
                return None
            try:
                checked = validate_choice_evidence(request, outcome, choice.get("track_ids"), choice.get("placement"),
                                                   passenger_requested=self._passenger_requested, live=True)
                ids = checked["track_ids"]
            except (TypeError,ValueError,KeyError):
                return None
        elif actor == "model" and source.get("method") == "qwen_landmark_region":
            evidence = source.get("between_road") or {}
            if evidence.get("withheld_reason") not in {"no_visible_shared_road_edge", "no_supported_neighborhood",
                                                       "indistinguishable_ground_contact_depth"}:
                return None
            from .between_audit import _native_selection
            from .landmarks import resolve_image
            try:
                image, binding = resolve_image(request,base.config)
                with image:
                    reading, _, _ = _native_selection(base,request,outcome,image,binding)
                ids = reading.get("track_ids",[])
                tracks = {t["track_id"]:t for t in current_tracks(request)}
                if (reading.get("kind") != "landmark" or len(ids) != 2 or len(set(ids)) != 2
                        or evidence.get("resolved_track_ids") != ids or source.get("selected_track_ids") != ids
                        or evidence.get("image_sha256") != binding["sha256"] or evidence.get("request_sha256") != digest(request)
                        or evidence.get("geometry_sha256") != base.geometry.sha256
                        or source.get("inference_input_sha256") != inference_input_sha256(request,binding["sha256"])
                        or evidence.get("current_observations_sha256") != digest(current_tracks(request))
                        or any(i not in tracks or tracks[i]["class_name"] not in MOTOR for i in ids)
                        or reading.get("boxes") != [tracks[i]["bbox_xyxy"] for i in ids]):
                    return None
            except (AttributeError,TypeError,ValueError,KeyError,OSError):
                return None
        else:
            return None
        observations = {t["track_id"]:t for t in current_tracks(request)}
        return {"source":actor,"request_id":request["request_id"],"image_sha256":request["scene"]["provenance"]["image_sha256"],
                "track_ids":list(ids),"boxes_xyxy":[list(observations[i]["bbox_xyxy"]) for i in ids]}

    def _accept(self, outcome, selected_by):
        from . import qualified_pair
        from . import passenger_parking
        from . import parking_clarification
        base = getattr(self.model, "model", self.model)
        if isinstance(outcome, dict) and outcome.get("ok") is True:
            method=outcome.get("provenance",{}).get("method")
            source_actors={METHOD:"passenger_landmarks", "explicit_passenger_region":"passenger",
                           "passenger_requested_pointing":"passenger", qualified_pair.METHOD:qualified_pair.ACTOR,
                           passenger_parking.METHOD:passenger_parking.ACTOR,
                           parking_clarification.METHOD:parking_clarification.ACTOR}
            if (method == parking_clarification.METHOD) != (selected_by == parking_clarification.ACTOR):
                raise ValueError("Reference clarification and its source disagree")
            if method == parking_clarification.METHOD:
                parking_clarification.verify(self._request, outcome)
            if ((method in source_actors and source_actors[method] != selected_by)
                    or (selected_by in {"passenger_landmarks", qualified_pair.ACTOR} and method not in source_actors)
                    or (selected_by == "passenger" and method in {"qwen_landmark_region","deictic_pair_clarification"})):
                raise ValueError("Passenger selection source and actual decision method disagree")
            if selected_by == qualified_pair.ACTOR:
                qualified_pair.validate(base,self._request,outcome)
            else:
                # Two browser sessions can share the segmentation model and its selected original.
                with getattr(base,"lock",None) or nullcontext():
                    between_road.validate_offer(base, self._request, outcome, selected_by)
        state=super()._accept(outcome, selected_by)
        if self._state == "error" and selected_by == "model" and self._supports_pointing():
            self._landmark_error_guidance = landmark_error_guidance(self._request, outcome)
        if self._state=="needs_clarification" and selected_by in {"passenger_landmarks","model"}:
            self._area_landmarks=self._landmarks_awaiting_area(base,outcome,selected_by)
        if selected_by == qualified_pair.ACTOR and self._state == "proposed":
            self._turns[-1]["text"]="I found one visible road gap that fits. Is the highlighted area where you want to get out?"
        return state

    def submit(self, text, *, original_transcript=None, speech=None, point_to_vehicles=False):
        if type(point_to_vehicles) is not bool:
            raise ValueError("The request to point to vehicles must be an explicit choice.")
        with self._lock:
            if point_to_vehicles and not self._supports_pointing():
                raise ValueError("Vehicle selection is unavailable for this scene.")
            self._passenger_requested = point_to_vehicles
            self._area_landmarks = None
            return super().submit(text, original_transcript=original_transcript, speech=speech)

    def _interpret_submission(self, request):
        if self._passenger_requested:
            return pointing_question(request), "passenger"
        from . import parking_clarification
        clarification = parking_clarification.clarify(request)
        if clarification is not None:
            return clarification, parking_clarification.ACTOR
        outcome, actor = super()._interpret_submission(request)
        from . import qualified_pair
        if isinstance(outcome,dict) and outcome.get("provenance",{}).get("method") == qualified_pair.METHOD:
            actor = qualified_pair.ACTOR
        return outcome, actor

    def _save(self, action, detail):
        if action == "submitted" and self._passenger_requested:
            detail = {**detail, "passenger_requested_pointing":True}
        super()._save(action, detail)
        if action == "submitted":
            raw = detail.get("outcome")
            self._original_submission = {
                "request_id": detail["request"].get("request_id"),
                "request_sha256": digest(detail["request"]),
                "original_outcome_sha256": digest(raw) if raw else None,
                ("original_pointing_state" if self._passenger_requested else "original_automatic_state"): self._state,
            }

    def _supports_pointing(self):
        return self._scene["source"] == "predicted" and (
            callable(getattr(self.model, "passenger_pair", None)) or isinstance(self.model, PassengerGrounder))

    def _active(self):
        return (
            self._supports_pointing()
            and self._request is not None
            and (
                self._state in {"proposed", "needs_clarification", "insufficient_visual_evidence"}
                or (self._state == "error" and self._error
                    and (self._error.get("code") in RECOVERABLE or self._landmark_error_guidance is not None))
            )
        )

    def view(self):
        with self._lock:
            state = super().view()
            state["can_request_pointing"] = self._supports_pointing()
            state["pair_choices"] = options(self._request, passenger_requested=self._passenger_requested) if self._active() else []
            if self._state == "error" and self._landmark_error_guidance:
                state["landmark_error_guidance"] = self._landmark_error_guidance
            hint=self._area_landmarks
            if (hint and self._state=="needs_clarification" and state["request_id"]==hint["request_id"]
                    and state["image_sha256"]==hint["image_sha256"]):
                state["area_landmarks"]=deepcopy(hint)
            elif self._state=="needs_clarification" and self._selected_by=="model" and self._request and between_road.requested(self._request):
                words=self._request["text"]
                matches=list(re.finditer(r"\bbetween\b",words,re.I))
                if matches:
                    before, after=words[:matches[-1].start()], words[matches[-1].end():]
                    if named_vehicle_pair(after) or (PAIR_PRONOUN.match(after) and named_vehicle_pair(before,prefatory=True)):
                        state["unmatched_described_pair"]=True
            return state

    def select_pair(self, track_ids, placement):
        with self._lock:
            if not self._active() or not options(self._request, passenger_requested=self._passenger_requested):
                raise ValueError("Give an active passenger instruction between two visible vehicles first.")
            self._clear()
            self._state = "idle"
            outcome = None
            try:
                extra = {"passenger_requested":True} if self._passenger_requested else {}
                if callable(getattr(self.model, "passenger_pair", None)):
                    outcome = self.model.passenger_pair(deepcopy(self._request), track_ids, placement, **extra)
                else:
                    with self.model.lock:
                        outcome = choose(self.model, deepcopy(self._request), track_ids, placement, **extra)
                if not isinstance(outcome, dict) or outcome.get("provenance", {}).get("method") != METHOD:
                    raise ValueError("This choice did not return a passenger-selected landmark result.")
                if outcome.get("ok") is True:
                    if placement == "between":
                        extra["live"] = True
                    validate_choice_evidence(self._request, outcome, track_ids, placement, **extra)
                if not self._original_submission or self._original_submission["request_sha256"] != digest(
                    self._request
                ):
                    raise ValueError("The original passenger submission is unavailable for this selection.")
                key = "pointing_submission" if self._passenger_requested else "automatic_submission"
                outcome["provenance"][key] = deepcopy(self._original_submission)
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
