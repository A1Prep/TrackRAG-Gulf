"""Resolve a model's two full-image vehicle boxes to distinct current vehicle observations."""
from itertools import permutations
from hashlib import sha256
import json
import math

from trackrag.dropoff.contracts import validate_request
from trackrag.dropoff.projected_pair import MOTOR
from tools.dropoff_predict import _reject_constant, _unique_object


ANCHOR_PROMPT = (
    "Find the two physical vehicles the passenger uses as references in the road photo. "
    "This step only identifies vehicles. They can be present even if there is no suitable space. "
    "Use vehicle appearance, wording and location. Return just JSON "
    '{"vehicles":[[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax]]} with one tight whole-vehicle box per '
    "distinct reference, on a 0-1000 scale relative to the whole image (x from left,y from top). "
    'If both cannot be unambiguously identified return {"vehicles":[]}. Never box an empty space.'
)


def _iou(one, two):
    x = max(0, min(one[2], two[2]) - max(one[0], two[0]))
    y = max(0, min(one[3], two[3]) - max(one[1], two[1]))
    overlap = x * y
    area = (one[2] - one[0]) * (one[3] - one[1]) + (two[2] - two[0]) * (two[3] - two[1])
    return overlap / (area - overlap) if area > overlap else 0.


def match_vehicles(boxes, tracks):
    """Find two distinct, unambiguous matches and preserve the prompt's vehicle order."""
    if len(boxes) != 2:
        return [], []
    scores = []
    for first, second in permutations((t for t in tracks if t["class_name"] in MOTOR), 2):
        overlaps = [_iou(boxes[0], first["bbox_xyxy"]), _iou(boxes[1], second["bbox_xyxy"])]
        if min(overlaps) >= .50:
            ids = [first["track_id"], second["track_id"]]
            scores.append((sum(overlaps), ids, overlaps))
    if not scores:
        return [], []
    scores.sort(key=lambda result: -result[0])
    top_score, ids, overlaps = scores[0]
    if any(top_score - other_score < .05 and set(ids) != set(other_ids)
           for other_score, other_ids, _ in scores[1:]):
        return [], []
    return ids, overlaps


def resolve_anchor_reply(request, raw_text):
    """Return a distinct operational error or a clarification before offering any region."""
    clean = validate_request(request)
    scene = clean["scene"]
    frame = scene["frame"]
    if any(t["frame_id"] != frame["frame_id"] for t in scene["tracks"]):
        raise ValueError("Parking identification requires current-frame tracks")
    empty = {"boxes": [], "track_ids": [], "ious": []}
    try:
        # Some native chat templates wrap the same JSON in a Markdown fence.
        text = raw_text.strip()
        fence = chr(96) * 3
        text = text.removeprefix(fence + "json").removeprefix(fence).removesuffix(fence).strip()
        body = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        normalized = body["vehicles"]
        if set(body) != {"vehicles"} or not isinstance(normalized, list):
            raise ValueError
        if not normalized:
            return {**empty, "state": "ask"}
        if len(normalized) != 2:
            raise ValueError
        for box in normalized:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError
            if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1000 for v in box):
                raise ValueError
            if box[2] - box[0] < 3 or box[3] - box[1] < 3:
                raise ValueError
    except (TypeError, AttributeError, KeyError, ValueError):
        return {**empty, "state": "error"}
    axes = (frame["width"], frame["height"]) * 2
    boxes = [[value * axis / 1000 for value, axis in zip(box, axes)] for box in normalized]
    if _iou(*boxes) > .80:
        return {**empty, "boxes": boxes, "state": "ask"}
    ids, ious = match_vehicles(boxes, scene["tracks"])
    return {"state": "matched" if ids else "ask", "track_ids": ids, "ious": ious, "boxes": boxes}


def read_anchor_records(requests, payload):
    """Bind unmodified model replies to the same wording, original and registered prompt."""
    records = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(records, list):
        raise ValueError("Expected a list of vehicle identification receipts")
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != {r["request_id"] for r in requests}:
        raise ValueError("Vehicle identification receipts and request IDs differ")
    expected_prompt = sha256(ANCHOR_PROMPT.encode()).hexdigest()
    resolved = {}
    for req in requests:
        record = by_id[req["request_id"]]
        if (record.get("source_sha") != req["scene"]["provenance"].get("image_sha256")
                or record.get("request") != req["text"] or record.get("prompt_sha256") != expected_prompt
                or record.get("revision") != "c202236235762e1c871ad0ccb60c8ee5ba337b9a"):
            raise ValueError("Vehicle identification changed model, photo, wording or prompt")
        resolved[req["request_id"]] = resolve_anchor_reply(req, record.get("raw"))
    return resolved


def export(requests_path, class_maps, out, anchor_records_path, font_file=None):
    """Build a parking-point board only for the two model-identified current vehicles."""
    from tools.passenger_parking_candidates import export as export_candidates
    return export_candidates(requests_path, class_maps, out, "parking", font_file=font_file,
                             anchor_records_path=anchor_records_path)
