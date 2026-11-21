"""Allowlisted model prompts and explicit partial-label supervision.

Model coordinates use a 0..1000 image-edge scale. Points are strictly below
1000; box right/bottom edges may equal 1000. Module outputs use original pixels.
"""

import json

from .contracts import DESTINATION_FIELDS, FULL_FIELDS, validate_request, validate_target


SYSTEM = (
    "Interpret a passenger instruction using only the supplied current road image, "
    "observed facts and scene-bound dialogue. Identify the intended destination, "
    "which is a location in the scene rather than a box around the referred car. "
    "Do not infer physical clearance, stopping legality, hidden objects or future observations. "
    "Image coordinates are normalized to 0..1000: points use [0,1000), "
    "box edges use [0,1000]. A destination point must lie inside its coarse destination box. "
)
INTERPRET = (
    "Return a single JSON object only, with all four required keys: "
    '"status", "destination_point_xy", "destination_box_xyxy", "clarification_question". '
    "Never omit a required key, including when its value is null. Do not add other keys. "
    "Choose exactly one of these mutually exclusive structures. "
    'For "status": "proposed", "destination_point_xy" must be a non-null array of two numbers '
    'and "destination_box_xyxy" must be a non-null array of four numbers ordered left, top, right, bottom; '
    '"clarification_question": null. Supply both geometry fields together, with the point inside the box. '
    'For "status": "needs_clarification", set "destination_point_xy": null and '
    '"destination_box_xyxy": null; "clarification_question" must be a nonempty question '
    "that distinguishes the unresolved visible references or asks for the missing selection context. "
    'For "status": "insufficient_visual_evidence", set "destination_point_xy": null, '
    '"destination_box_xyxy": null and "clarification_question": null. '
    "Never attach a destination box or point to either unresolved status. "
    "Never combine a clarification question with destination coordinates. "
    "Use the supplied image and instruction to choose the status; do not guess a location "
    "when the reference remains unresolved. An unpointed 'here' is unresolved. "
    "Before returning the JSON, check that all four keys are present and their null/non-null "
    "values match the selected status. Do not create confirmation events or invent a confidence score."
)
DESTINATION_ONLY = (
    "This is a destination-only annotation task. Return only a JSON object with "
    "destination_point_xy and destination_box_xyxy. Do not supply status, "
    "ambiguity, anchor identities or a clarification label: this source does not supervise them."
)


def _scale_box(box, width, height, factor):
    return [round(box[0] * factor / width, 6), round(box[1] * factor / height, 6),
            round(box[2] * factor / width, 6), round(box[3] * factor / height, 6)]


def build_messages(request, task="interpret", *, allow_annotated=False):
    request = validate_request(request, allow_annotated=allow_annotated)
    if task not in ("interpret", "destination_only"):
        raise ValueError("unknown grounding task")
    frame = request["scene"]["frame"]
    width, height = frame["width"], frame["height"]
    metadata = {
        "recording_id": frame["recording_id"], "frame_id": frame["frame_id"],
        "timestamp_s": frame["timestamp_s"], "width": width, "height": height,
        "track_source": request["scene"]["source"], "track_box_coordinates": "normalized_0_1000",
        "observations": [
            {"frame_id": row["frame_id"], "track_id": row["track_id"],
             "class_name": row["class_name"],
             "bbox_xyxy": _scale_box(row["bbox_xyxy"], width, height, 1000)}
            for row in request["scene"]["tracks"]
        ],
    }
    messages = [
        {"role": "system", "content": SYSTEM + (INTERPRET if task == "interpret" else DESTINATION_ONLY)},
        {"role": "user", "content": [
            {"type": "image", "image": frame["image_path"]},
            {"type": "text", "text": "Observed scene context:\n" + json.dumps(metadata, sort_keys=True)},
        ]},
    ]
    messages.extend({"role": turn["role"], "content": turn["text"]} for turn in request["dialogue"])
    messages.append({"role": "user", "content": request["text"]})
    return messages


def target_to_model(target, supervised_fields, frame):
    target = validate_target(target, supervised_fields, frame)
    width, height = frame["width"], frame["height"]
    if target.get("destination_point_xy") is not None:
        point = target["destination_point_xy"]
        target["destination_point_xy"] = [round(point[0] * 1000 / width, 6),
                                        round(point[1] * 1000 / height, 6)]
    if target.get("destination_box_xyxy") is not None:
        target["destination_box_xyxy"] = _scale_box(target["destination_box_xyxy"], width, height, 1000)
    if target.get("anchor_boxes_xyxy") is not None:
        target["anchor_boxes_xyxy"] = [
            _scale_box(box, width, height, 1000) for box in target["anchor_boxes_xyxy"]
        ]
    return target


def build_training_messages(case):
    fields = case["supervised_fields"]
    task = "destination_only" if set(fields) == DESTINATION_FIELDS else "interpret"
    request = validate_request(case["request"])
    messages = build_messages(request, task=task)
    target = target_to_model(case["target"], fields, request["scene"]["frame"])
    messages.append({"role": "assistant", "content": json.dumps(target, sort_keys=True, separators=(",", ":"))})
    return messages


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate model field: {key}")
        result[key] = value
    return result


def parse_model_output(text, width, height):
    """Parse one strict normalized JSON result, never recover a guessed substring."""
    if not isinstance(text, str) or len(text) > 32768:
        raise ValueError("model output must be bounded text")
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines[0] not in ("```", "```json") or lines[-1] != "```":
            raise ValueError("invalid model JSON fence")
        text = "\n".join(lines[1:-1])
    raw = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(raw, dict) or not FULL_FIELDS <= raw.keys():
        raise ValueError("model result lacks required interpretation fields")
    # Validate normalized geometry/status before converting; no clipping.
    normalized = validate_target(raw, list(raw), {"width": 1000, "height": 1000})
    result = dict(normalized)
    if normalized["destination_point_xy"] is not None:
        x, y = normalized["destination_point_xy"]
        result["destination_point_xy"] = [x * width / 1000, y * height / 1000]
        result["destination_box_xyxy"] = [
            normalized["destination_box_xyxy"][0] * width / 1000,
            normalized["destination_box_xyxy"][1] * height / 1000,
            normalized["destination_box_xyxy"][2] * width / 1000,
            normalized["destination_box_xyxy"][3] * height / 1000,
        ]
    if normalized.get("anchor_boxes_xyxy") is not None:
        result["anchor_boxes_xyxy"] = [
            [box[0] * width / 1000, box[1] * height / 1000,
             box[2] * width / 1000, box[3] * height / 1000]
            for box in normalized["anchor_boxes_xyxy"]
        ]
    return validate_target(result, list(result), {"width": width, "height": height})
