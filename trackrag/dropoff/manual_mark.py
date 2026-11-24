"""Verify an explicit passenger mark in original pixels, optionally replaying its measured road support."""
import re

from .contracts import FULL_FIELDS, validate_box, validate_request, validate_target
from .data import file_sha256, inference_input_sha256
from .passenger import digest, proposed, supported_region, unresolved
from .road_cache import ROAD_RECIPE_SHA256, mask_sha256, mask_shape, read_mask

REQUEST_METHOD = "passenger_requested_area"
MARK_PROMPT = "Which spot do you mean? Mark it on the photo and I’ll check the visible road."


def question(request):
    """Open marking on the exact requested photo, before any automatic interpretation."""
    from .inputs import _inspect_image

    request = validate_request(request)
    scene = request["scene"]
    _, image_sha = _inspect_image(scene["frame"])
    if image_sha != scene["provenance"].get("image_sha256"):
        raise ValueError("The original photograph changed before marking.")
    return {"ok": True, "request_id": request["request_id"],
            "result": unresolved(MARK_PROMPT),
            "provenance": {"method": REQUEST_METHOD, "independent_model_proposal": False, "native_model_calls": 0,
                           "image": {"sha256": image_sha}, "request_sha256": digest(request),
                           "inference_input_sha256": inference_input_sha256(request, image_sha)}}


def verify(request, outcome, *, requested_box=None, road_cache=None):
    request = validate_request(request)
    frame = request["scene"]["frame"]
    size = frame["width"], frame["height"]
    source = outcome.get("provenance", {})
    photo = request["scene"]["provenance"].get("image_sha256")
    if (outcome.get("ok") is not True or outcome.get("request_id") != request["request_id"]
            or source.get("method") != "explicit_passenger_region" or source.get("independent_model_proposal") is not False
            or source.get("resolver") is not None or source.get("image", {}).get("sha256") != photo
            or file_sha256(frame["image_path"]) != photo or source.get("request_sha256") != digest(request)
            or source.get("inference_input_sha256") != inference_input_sha256(request, photo)):
        raise ValueError("Passenger mark lacks its original frame, actual passenger role or exact request")
    box = validate_box(source.get("selected_box"), *size, name="passenger mark")
    if requested_box is not None and validate_box(requested_box, *size, name="submitted mark") != box:
        raise ValueError("Submitted mark differs from the region recorded by the grounding application")
    target = validate_target(outcome["result"], list(outcome["result"]), frame)
    if set(target) != FULL_FIELDS:
        raise ValueError("A passenger mark cannot supply model-selected landmark boxes")
    exclusions = [t["bbox_xyxy"] for t in request["scene"]["tracks"] if t["frame_id"] == frame["frame_id"]]
    recorded = source.get("road_evidence") or {}
    if (recorded.get("road_recipe_sha256") != ROAD_RECIPE_SHA256 or recorded.get("mask_shape") != list(mask_shape(size))
            or recorded.get("current_exclusion_boxes") != exclusions
            or not re.fullmatch(r"[0-9a-f]{64}", str(recorded.get("road_mask_sha256")))):
        raise ValueError("Passenger mark uses different current observations or segmentation evidence")
    point = target["destination_point_xy"]
    if target["status"] == "proposed":
        if (target["destination_box_xyxy"] != box or not (box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3])
                or any(a <= point[0] < c and b <= point[1] < d for a,b,c,d in exclusions)):
            raise ValueError("Offered point is outside the submitted mark or on a current observed object")
    elif target["status"] != "needs_clarification":
        raise ValueError("An explicit passenger mark must offer a supported point or request a different mark")
    receipt = {"current_image_and_tracks_checked":True, "submitted_mark":box, "road_mask_replayed":False,
               "decision_reproduced":False, "independent_model_proposal":False}
    if road_cache is None:
        return receipt
    mask = read_mask(road_cache, photo, size)
    if mask is None:
        raise FileNotFoundError("A bound road mask for the passenger's original photo is required to replay the mark")
    if mask_sha256(mask) != recorded["road_mask_sha256"]:
        raise ValueError("Stored road mask differs from the mask used for the passenger mark")
    center = [(box[0]+box[2])/2,(box[1]+box[3])/2]
    eligible = supported_region(mask,size,center,box,exclusions)
    expected = proposed(*eligible) if eligible else unresolved("The selected area has no visible road support. Can you outline a different patch?")
    if expected != target:
        raise ValueError("The recorded passenger-mark result differs from the original mask and selected region")
    return {**receipt, "road_mask_replayed":True, "decision_reproduced":True, "road_mask_sha256":recorded["road_mask_sha256"]}
