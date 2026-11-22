"""Estimate a visible roadside gap between two identified vehicles in the current picture."""
import re

import numpy as np

from . import pair_geometry
from .contracts import validate_box
from .projected_pair import MOTOR, Ineligible, need, supported_patch
from .road_cache import ROAD_RECIPE_SHA256, mask_shape


VERSION = "vehicle_contact_polygon_and_visible_road_edge_v1"
QUESTION = "I can't locate a clear roadside gap between these vehicles. Could you mark the intended road area?"
WHICH_PAIR = "Which two vehicles do you mean? Select both on the photo or describe each one."
PARKING_QUESTION = "I can't verify which vehicles are parked from this view. Could you select the two you mean?"


def _between_placement(text):
    from .passenger import pair_required
    other = re.search(
        r"\b(?:behind|ahead of|in front of|beside)\s+(?:(?:the|this|that)\s+)?(?:road|gap|space|opening|area|patch)"
        r"\s+between\b|\bbetween\b.*\b(?:behind|ahead of|in front of)\s+(?:both|them|the pair)\b",
        text, re.I)
    return pair_required(text) and not other


def requested(request):
    from .passenger import instruction_context
    from .passenger_pair_choice import pair_intent
    return _between_placement(instruction_context(request)) and pair_intent(request)


def unbound_vehicle_selection(request, decision):
    """Do not send a novel, unbound two-vehicle reading through ordinary one-anchor geometry."""
    from .pair_resolution import explicit_pair_quotes
    from .passenger_pair_choice import current_tracks, NEG, OTHER_LOCATION, RESET, STRUCTURE, VEHICLE
    text = request["text"]
    match = list(re.finditer(r"\bbetween\b",text,re.I))
    if not match or not _between_placement(text) or NEG.search(text[:match[-1].start()]):
        return False
    correction = RESET.search(text,match[-1].end())
    if correction and text[correction.end():].strip(" ,.;:!?"):
        return False
    ids = decision.get("track_ids", [])
    if decision.get("kind") != "landmark" or not isinstance(ids,list) or len(ids) != 2 or ids[0] == ids[1]:
        return False
    current = {t["track_id"]:t for t in current_tracks(request)}
    if any(i not in current or current[i]["class_name"] not in MOTOR for i in decision["track_ids"]):
        return False
    # Preserve explicitly mixed and nonvehicle descriptions on their existing route.
    quotes = explicit_pair_quotes(text)
    phrases = [q["text"] for q in quotes] if quotes else [text[match[-1].end():]]
    for phrase in phrases:
        vehicle = VEHICLE.search(phrase)
        end = vehicle.start() if vehicle else len(phrase)
        structure = STRUCTURE.search(phrase)
        if OTHER_LOCATION.search(phrase[:end]) or (structure and structure.start() <= end):
            return False
    return True


def native_result(model, request, image, binding, decision, record, index):
    from . import between_policy
    from .landmarks import parse_landmark
    from .passenger import digest, instruction_context, proposed, unresolved
    from .passenger_pair_choice import current_tracks
    ids = decision.get("track_ids", [])
    tracks = current_tracks(request)
    by_id = {t["track_id"]: t for t in tracks}
    audit = {"guard": VERSION, "image_sha256": binding["sha256"], "request_sha256": digest(request),
             "current_observations_sha256": digest(tracks), "resolved_track_ids": ids,
             "geometry_sha256": model.geometry.sha256, **between_policy.evidence(model)}
    if (decision.get("kind") != "landmark" or len(set(ids)) != 2 or len(ids) != 2
            or any(i not in by_id or by_id[i]["class_name"] not in MOTOR for i in ids)
            or decision.get("boxes") != [by_id[i]["bbox_xyxy"] for i in ids]):
        return unresolved(WHICH_PAIR), {**audit, "withheld_reason": "two_distinct_current_vehicles_required"}
    from .pair_dialogue import dark_pair_evidence, unpointed_reference, VEHICLE_CLASSES
    collective = unpointed_reference(request["text"])
    if collective and any(by_id[i]["class_name"] not in VEHICLE_CLASSES[collective["noun"].lower()] for i in ids):
        return unresolved(WHICH_PAIR), {**audit, "withheld_reason": "native_vehicle_class_disagrees_with_request"}
    if collective and (collective["colour"] or "").casefold()=="dark":
        candidates = [t for t in tracks if t["class_name"] in VEHICLE_CLASSES[collective["noun"].lower()]]
        appearance = dark_pair_evidence(image, candidates)
        audit["dark_appearance"] = appearance
        if set(appearance["distinct_supported_pair"]) != set(ids):
            return unresolved(WHICH_PAIR), {**audit, "withheld_reason": "requested_dark_pair_not_visually_supported"}
    if "pair_resolution" in record:
        whole = parse_landmark(record["raw_text"], index, request["scene"]["frame"])
        whole_ids = set(whole.get("track_ids", []))
        if whole_ids and not whole_ids.issubset(ids):
            return unresolved(WHICH_PAIR), {**audit, "withheld_reason": "conflicting_native_landmark_selections"}
    text = instruction_context(request)
    if re.search(r"\b(?:parked|stationary)\b", text, re.I):
        return unresolved(PARKING_QUESTION), {**audit, "withheld_reason": "parking_state_unverified_in_still"}
    selected = [by_id[i] for i in ids]
    try:
        between_policy.preflight(model, selected, image.size)
        mask = model.road.mask(image, binding["sha256"])
        patch = between_policy.estimate(model, mask, image, selected, tracks, text,image_sha=binding["sha256"])
    except Ineligible as error:
        if str(error) in {"unsupported_road_mask", "no_prediction_from_trained_pair_head"}:
            raise ValueError("Invalid learned pair geometry or current road model output") from error
        return unresolved(QUESTION), {**audit, **between_policy.refusal(error)}
    return proposed(patch["point"], patch["box"], decision["boxes"]), {**audit, "road_geometry": patch}


def contact_geometry(observations, size):
    need(len(observations) == 2 and observations[0]["track_id"] != observations[1]["track_id"]
         and all(t["class_name"] in MOTOR for t in observations), "two_distinct_current_vehicles_required")
    boxes = [validate_box(t["bbox_xyxy"], *size) for t in observations]
    far, near = sorted(boxes, key=lambda b: b[3])
    if near[3] == far[3]:
        overlaps = min(far[2], near[2]) > max(far[0], near[0])
        raise Ineligible("no_visible_between_gap" if overlaps else "indistinguishable_ground_contact_depth")
    overlap = (max(0,min(far[2],near[2])-max(far[0],near[0]))
               * max(0,min(far[3],near[3])-max(far[1],near[1])))
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    need(overlap <= 0.5*min(areas), "selected_vehicles_substantially_occlude_each_other")
    # The lower edge of each detection is an image-space ground-contact proxy.
    # The two segments form a trapezoid even when their horizontal ranges overlap.
    vertices = [[far[0], far[3]], [far[2], far[3]], [near[2], near[3]], [near[0], near[3]]]
    return {"contact_polygon_xy": vertices, "regression_inputs_xyxy": boxes,
            "center_corridor_xyxy": [min(far[0], near[0]), far[3], max(far[2], near[2]), near[3]]}


def _edges(polygon, y):
    (a, top), (b, _), (d, bottom), (c, _) = polygon
    t = (y - top) / (bottom - top)
    return a + (c-a)*t, b + (d-b)*t


def inside_contact_polygon(polygon, box):
    left, top, right, bottom = box
    if not polygon[0][1] <= top < bottom <= polygon[2][1]:
        return False
    return all(a <= left and right <= b for a,b in
               (_edges(polygon, top), _edges(polygon, bottom)))


def eligible_cells(mask, size, geometry, tracks):
    """Require complete road cells in the trapezoid near one shared, visible outer road edge."""
    need(isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.shape == mask_shape(size),
         "unsupported_road_mask")
    h, w = mask.shape
    need(min(h, w) >= 3 and bool(mask.any()), "no_supported_neighborhood")
    dx, dy = size[0] / w, size[1] / h
    xs = np.arange(w)*dx
    ys = np.arange(h)*dy
    poly = geometry["contact_polygon_xy"]
    a0, b0 = _edges(poly, ys)
    a1, b1 = _edges(poly, ys + dy)
    hull = ((ys >= poly[0][1]) & (ys+dy <= poly[2][1]))[:, None]
    hull = hull & (xs >= np.maximum(a0, a1)[:, None]) & (xs+dx <= np.minimum(b0, b1)[:, None])

    # An isolated road pixel, photo edge or detector occlusion cannot witness a curbside.
    # Use the two *outer* road extents so an internal vehicle-shaped hole cannot become a curb.
    runs = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]
    has_road = runs.any(axis=1)
    left = runs.argmax(axis=1)
    right = w-1-runs[:, ::-1].argmax(axis=1)
    occupied = np.zeros_like(mask)
    for track in tracks:
        x0,y0,x1,y1 = track["bbox_xyxy"]
        occupied |= ((ys < y1) & (ys+dy > y0))[:, None] & (xs < x1) & (xs+dx > x0)
    outside = ~mask & ~occupied
    rows = np.arange(h)
    left_witness = has_road & (left >= 3)
    right_witness = has_road & (right+3 < w)
    for offset in (1, 2, 3):
        left_witness &= outside[rows, np.maximum(0,left-offset)]
        right_witness &= outside[rows, np.minimum(w-1,right+offset)]
    middle = (left+right+1)*dx/2
    centers = [(b[0]+b[2])/2 for b in geometry["regression_inputs_xyxy"]]
    # A trapezoid may graze the roadside even if its far or near vehicle runs in a
    # neighbouring through lane. Test both vehicle contacts on the same observed edge.
    # Allow one own box width between the outside of each box and the estimated road edge.
    def edge_beside_both(witness, boundary):
        for vehicle in geometry["regression_inputs_xyxy"]:
            a,b,c,d = vehicle
            contact_rows = (ys >= d) & (ys < d + max(3*dy,.15*(d-b)))
            near_box = (boundary >= a-(c-a)) & (boundary <= c+(c-a))
            if (contact_rows & witness & near_box).sum() < 2:
                return False
        return True
    # One local projected vehicle width limits the edge band; it is not a distance in metres.
    local_width = b0-a0
    left_side = left_witness & (max(centers) < middle)
    right_side = right_witness & (min(centers) > middle)
    if not edge_beside_both(left_side, left*dx):
        left_side[:] = False
    if not edge_beside_both(right_side, (right+1)*dx):
        right_side[:] = False
    left_band = left_side[:, None] & (xs+dx <= (left*dx+local_width)[:, None])
    right_band = right_side[:, None] & (xs >= ((right+1)*dx-local_width)[:, None])
    eligible = hull & mask & (left_band | right_band)
    need(eligible.any(), "no_visible_shared_road_edge")
    eligible &= ~occupied
    need(eligible.any(), "no_supported_neighborhood")
    return eligible, (left, right, left_side)


def estimate(fitted_geometry, mask, size, observations, tracks, text):
    geometry = contact_geometry(observations, size)
    eligible, edges = eligible_cells(mask, size, geometry, tracks)
    boxes = geometry["regression_inputs_xyxy"]
    pair = pair_geometry.predict(fitted_geometry.pair_regression, boxes, size)
    if pair is None:
        # The released horizontal-gap head has no representation for overlapped x-ranges.
        # Its released companion estimates a region from the actual envelope and utterance.
        point, region, neighbors = fitted_geometry.predict_with_evidence(boxes, size, text)
        pair = (point, region)
        estimator = "learned_anchor_envelope_and_retrieved_offsets"
    else:
        neighbors = []
        estimator = "learned_horizontal_pair_regression"
    patch = supported_patch(mask, size, fitted_geometry.pair_regression, geometry, tracks,
                            forecast=pair, eligible_cells=eligible)
    px, py = patch["point"]
    h, w = mask.shape
    row = min(h-1, int(py*h/size[1]))
    left, right, is_left = edges
    side = "left" if is_left[row] else "right"
    first_exterior = int(left[row]-1 if is_left[row] else right[row]+1)
    return {**geometry, **patch, "guard": VERSION, "learned_estimator": estimator,
            "retrieved_training_examples": neighbors, "mask_shape": [h,w], "side_of_visible_road": side,
            "edge_witness_mask_cell_xy": [first_exterior, row]}


def validate_patch(patch, observations, tracks, size, point, box):
    geometry = contact_geometry(observations, size)
    if (patch.get("guard") != VERSION or any(patch.get(k) != v for k,v in geometry.items())
            or patch.get("road_recipe_sha256") != ROAD_RECIPE_SHA256
            or not re.fullmatch(r"[0-9a-f]{64}", str(patch.get("road_mask_sha256")))
            or patch.get("point") != point or patch.get("box") != box
            or not (box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3])
            or not inside_contact_polygon(geometry["contact_polygon_xy"], box)
            or any(box[0] < t[2] and box[2] > t[0] and box[1] < t[3] and box[3] > t[1]
                   for t in (entry["bbox_xyxy"] for entry in tracks))):
        raise ValueError("Between-vehicle proposal differs from its ground contacts, current tracks or road evidence")


def validate_offer(model, request, outcome, actor):
    """Before issuing a confirmation token, check the offered area on the authenticated current mask."""
    from . import between_policy
    from .landmarks import resolve_image
    from .passenger import digest
    from .passenger_pair_choice import current_tracks
    from .road_cache import mask_sha256
    info = outcome.get("provenance", {})
    if outcome.get("result", {}).get("status") != "proposed":
        return
    if (actor == "model" and info.get("method") == "qwen_landmark_region" and not requested(request)
            and unbound_vehicle_selection(request, {"kind":"landmark", "track_ids":info.get("selected_track_ids", [])})):
        raise ValueError("Unresolved between-vehicle reference must be clarified before offering an area")
    if actor == "model" and info.get("method") == "qwen_landmark_region" and requested(request):
        audit = info.get("between_road") or {}
        ids = info.get("selected_track_ids")
        if (audit.get("guard") != VERSION or audit.get("request_sha256") != digest(request)
                or audit.get("image_sha256") != request["scene"]["provenance"]["image_sha256"]
                or audit.get("geometry_sha256") != model.geometry.sha256
                or audit.get("resolved_track_ids") != ids):
            raise ValueError("Unverified automatic between-vehicle proposal")
        patch = audit.get("road_geometry", {})
    elif actor == "passenger_landmarks" and info.get("passenger_selection", {}).get("placement") == "between":
        audit = info
        ids = info["passenger_selection"]["track_ids"]
        patch = audit.get("road_geometry", {})
    else:
        return
    enabled = between_policy.mode(model)
    if audit.get("between_road_mode",between_policy.EDGE) != enabled:
        raise ValueError("Between-vehicle proposal was created under a different roadside geometry configuration")
    tracks = current_tracks(request)
    by_id = {t["track_id"]: t for t in tracks}
    if (not isinstance(ids, list) or len(ids) != 2 or any(type(i) is not int or i not in by_id for i in ids)
            or ids[0] == ids[1] or audit.get("current_observations_sha256") != digest(tracks)):
        raise ValueError("Between-vehicle proposal uses different current tracks")
    selected = [by_id[i] for i in ids]
    target = outcome["result"]
    if target.get("anchor_boxes_xyxy") != [t["bbox_xyxy"] for t in selected]:
        raise ValueError("Between-vehicle proposal uses different landmark boxes")
    image, binding = resolve_image(request, model.config)
    with image:
        size = image.size
        if actor == "model":
            from .pair_dialogue import VEHICLE_CLASSES, dark_pair_evidence, unpointed_reference
            group = unpointed_reference(request["text"])
            if group and (group["colour"] or "").casefold() == "dark":
                candidates = [t for t in tracks if t["class_name"] in VEHICLE_CLASSES[group["noun"].lower()]]
                appearance = dark_pair_evidence(image, candidates)
                if appearance != audit.get("dark_appearance") or set(appearance["distinct_supported_pair"]) != set(ids):
                    raise ValueError("Between-vehicle proposal disagrees with the dark vehicles in the original")
        between_policy.validate_patch(patch, selected, tracks, size, target["destination_point_xy"],
                                      target["destination_box_xyxy"], enabled)
        mask = model.road.mask(image, binding["sha256"])
        from .between_audit import reproduce
        reproduce(model, request, outcome, actor, image, binding, mask, selected, tracks, patch)
        if mask_sha256(mask) != patch.get("road_mask_sha256"):
            raise ValueError("Between-vehicle proposal differs from the current road mask")
        if patch["guard"] == between_policy.BAY_GUARD:
            from . import curb_facing
            from .sidewalk import mask_for
            curb_facing.validate_mask(patch, mask, size, selected, tracks, image=image,
                                      sidewalk_mask=mask_for(model.road,image,binding["sha256"]))
            return
    eligible, edges = eligible_cells(mask, size, contact_geometry(selected, size), tracks)
    h, w = mask.shape
    dx, dy = size[0] / w, size[1] / h
    left, top, right, bottom = target["destination_box_xyxy"]
    a, b, c, d = [round(v / s) for v,s in zip((left,top,right,bottom),(dx,dy,dx,dy))]
    offered = np.array([left,top,right,bottom])
    if (not np.allclose(offered, np.array([a*dx,b*dy,c*dx,d*dy]), rtol=0, atol=1.e-8)
            or not (0 <= a < c <= w and 0 <= b < d <= h and c-a >= 3 and d-b >= 3)
            or not eligible[b:d,a:c].all()):
        raise ValueError("Between-vehicle proposal includes unsupported road cells")
    row = int(target["destination_point_xy"][1]/dy)
    left_edge, right_edge, is_left = edges
    expected_edge = int(left_edge[row]-1 if is_left[row] else right_edge[row]+1)
    if (patch.get("edge_witness_mask_cell_xy") != [expected_edge,row]
            or patch.get("side_of_visible_road") != ("left" if is_left[row] else "right")
            or patch.get("mask_shape") != list(mask.shape)):
        raise ValueError("Between-vehicle road-edge witness differs from the current image")
