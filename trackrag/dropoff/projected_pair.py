"""Optional experiment for a passenger who explicitly asks for road just behind two vehicles."""

from copy import deepcopy
from hashlib import sha256
import json
import re

import numpy as np

from . import pair_geometry
from .contracts import validate_box, validate_request, validate_target
from .landmark_check import GUIDE as CHECK_GUIDE, check_landmark, evidence_board
from .landmarks import landmark_messages, parse_landmark, resolve_image
from .model import verify_adapter
from .pair_resolution import explicit_pair_quotes, replay_pair_decision
from .passenger import digest, proposed
from .road_cache import ROAD_RECIPE_SHA256, mask_sha256, mask_shape
from .streamed import METADATA_SHA256, STREAMED_MODEL_ID, STREAMED_REVISION, WEIGHT_SHA256


METHOD = "explicit_behind_neighboring_vehicle_centers_v1"
GEOMETRY = "db9575ac6b82ff9828f29440bacf2349cd2371f23c2cfb747bd7dc05ee04c7e5"
MOTOR = {
    "Car",
    "Bus",
    "Large_vehicle",
    "Medium_vehicle",
    "Emergency_vehicle",
    "Small_motorised_vehicle",
    "Motorbike",
}
NEGATION = re.compile(
    r"\b(?:not|never|neither|without|except|avoid|no|rather than|instead of|or)\b|n['’]t\b", re.I
)
PLACEMENT = re.compile(r"\bbehind\s+(?:(?:the|that|this)\s+)?(?:gap|space|opening)\s+between\s*$", re.I)
TAIL = re.compile(
    r"^[\s,;:–—-]*(?:and\s+)?(?:(?:just|right|directly|immediately)\s+)?behind\s+"
    r"(?:them(?:\s+both)?|both(?:\s+of\s+them)?|the\s+pair)[\s,.!?]*$",
    re.I,
)
VEHICLE = re.compile(
    r"\b(car|sedan|coupe|hatchback|suv|taxi|cab|van|bus|coach|truck|lorry|motorbike|"
    r"motorcycle|scooter|vehicle)\b",
    re.I,
)
HEAD_MODIFIERS = frozenset(
    """the a an that this parked waiting stopped moving dark light black white gray grey
    silver blue red yellow green orange brown purple bright small little big tall large short long open sided
    darkish near nearest far farthest left leftmost right rightmost front rear first second third""".split()
)


class Ineligible(ValueError):
    """A scope or evidence condition failed. Keep the original passenger response."""


def need(value, reason):
    if not value:
        raise Ineligible(reason)


def scoped_quotes(text):
    quotes = explicit_pair_quotes(text)
    need(quotes and not NEGATION.search(text), "no_unambiguous_current_vehicle_pair")
    prefix, tail = text[: quotes[0]["start"]], text[quotes[1]["end"] :]
    courtesy = re.fullmatch(r"[\s,.:;!?]*(?:please[\s,.!?]*)?", tail, re.I)
    need((PLACEMENT.search(prefix) and courtesy) or TAIL.fullmatch(tail), "behind_does_not_apply_to_both")
    need(not re.search(r"\b(?:ahead of|in front of|in place of)\b", prefix, re.I), "conflicting_placement")
    descriptors = []
    for q in quotes:
        noun = VEHICLE.search(q["text"])
        need(
            noun is not None
            and set(re.findall(r"\w+", q["text"][: noun.start()].casefold())) <= HEAD_MODIFIERS,
            "both_descriptions_must_have_a_vehicle_head_noun",
        )
        descriptors.append(
            " ".join(re.findall(r"\w+", re.sub(r"^(?:the|an?|this|that)\s+", "", q["text"].casefold())))
        )
    need(descriptors[0] != descriptors[1], "indistinguishable_vehicle_descriptions")
    return quotes


def category_matches(quote, observation):
    # The head noun is the first vehicle in the quoted description; later nouns can be relative landmarks.
    noun = VEHICLE.search(quote).group().lower()
    cls = observation["class_name"]
    if noun == "vehicle":
        return cls in MOTOR
    if noun in {"bus", "coach"}:
        return cls == "Bus"
    if noun in {"truck", "lorry"}:
        return cls == "Large_vehicle"
    if noun in {"motorbike", "motorcycle", "scooter"}:
        return cls == "Motorbike"
    if noun == "van":
        return cls in {"Car", "Medium_vehicle"}
    return cls == "Car"


def center_x(box):
    return (box[0] + box[2]) / 2


def neighbor_geometry(observations, current, size):
    need(
        len(observations) == 2
        and observations[0]["track_id"] != observations[1]["track_id"]
        and all(t["class_name"] in MOTOR for t in observations),
        "two_distinct_vehicles_required",
    )
    left, right = sorted((validate_box(t["bbox_xyxy"], *size) for t in observations), key=center_x)
    lx, rx = center_x(left), center_x(right)
    lw, rw = left[2] - left[0], right[2] - right[0]
    lh, rh = left[3] - left[1], right[3] - right[1]
    overlap_width = min(left[2], right[2]) - max(left[0], right[0])
    overlap_height = min(left[3], right[3]) - max(left[1], right[1])
    intersection = overlap_width * overlap_height if min(overlap_width, overlap_height) > 0 else 0
    iou = intersection / (lw * lh + rw * rh - intersection)
    need(0 < overlap_width <= 0.25 * min(lw, rw) and 0 < iou <= 0.15, "overlap_is_absent_or_too_strong")
    need(
        max(lh, rh) <= 2 * min(lh, rh) and abs(left[3] - right[3]) <= 0.5 * min(lh, rh),
        "object_depth_or_size_is_incompatible",
    )
    selected = {t["track_id"] for t in observations}
    base = max(left[3], right[3])
    for t in current:
        b = t["bbox_xyxy"]
        if (
            t["track_id"] not in selected
            and t["class_name"] in MOTOR
            and lx < center_x(b) < rx
            and abs(b[3] - base) <= 0.5 * min(lh, rh)
        ):
            raise Ineligible("another_current_vehicle_is_between_them")
    # Move only each box's horizontal origin for the trained gap representation.
    # The model keeps both observed widths, heights and both object-ground contact heights.
    proxy = [[lx - lw, left[1], lx, left[3]], [rx, right[1], rx + rw, right[3]]]
    need(proxy[0][0] >= 0 and proxy[1][2] <= size[0], "projected_training_coordinates_leave_the_image")
    return {
        "center_corridor_xyxy": [lx, base, rx, min(size[1], base + max(lh, rh))],
        "regression_inputs_xyxy": proxy,
        "original_anchor_iou": iou,
        "overlap_fraction_of_narrower_box": overlap_width / min(lw, rw),
    }


def supported_patch(mask, size, head, geometry, tracks):
    """Rank admissible road cells by the frozen learned pair prediction; trim the reported area to road."""
    need(
        isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.shape == mask_shape(size),
        "unsupported_road_mask",
    )
    forecast = pair_geometry.predict(head, geometry["regression_inputs_xyxy"], size)
    need(forecast is not None, "no_prediction_from_trained_pair_head")
    point, learned = forecast
    fitted = validate_box(learned, *size)
    corridor = geometry["center_corridor_xyxy"]
    bounds = [
        max(fitted[0], corridor[0]),
        max(fitted[1], corridor[1]),
        min(fitted[2], corridor[2]),
        min(fitted[3], corridor[3]),
    ]
    h, w = mask.shape
    dx, dy = size[0] / w, size[1] / h
    x, y = np.arange(w) * dx, np.arange(h) * dy
    valid = (
        mask & ((x >= bounds[0]) & (x + dx <= bounds[2])) & ((y >= bounds[1]) & (y + dy <= bounds[3]))[:, None]
    )
    for track in tracks:
        a, b, c, d = track["bbox_xyxy"]
        valid &= ~(((x < c) & (x + dx > a)) & ((y < d) & (y + dy > b))[:, None])
    need(min(h, w) >= 3, "no_supported_neighborhood")
    interior = np.lib.stride_tricks.sliding_window_view(valid, (3, 3)).all(axis=(2, 3))
    rows, cols = np.nonzero(interior)
    need(len(rows) > 0, "no_supported_neighborhood")
    rows, cols = rows + 1, cols + 1
    distance = (((cols + 0.5) * dx - point[0]) / (fitted[2] - fitted[0])) ** 2 + (
        ((rows + 0.5) * dy - point[1]) / (fitted[3] - fitted[1])
    ) ** 2
    i = int(distance.argmin())
    cy, cx = int(rows[i]), int(cols[i])
    # All cells in the displayed rectangle must be supported. Grow its common strip
    # from the winning three-row neighborhood and then vertically inside that strip.
    strip = valid[cy - 1 : cy + 2].all(axis=0)
    a, c = cx - 1, cx + 2
    while a > 0 and strip[a - 1]:
        a -= 1
    while c < w and strip[c]:
        c += 1
    b, d = cy - 1, cy + 2
    while b > 0 and valid[b - 1, a:c].all():
        b -= 1
    while d < h and valid[d, a:c].all():
        d += 1
    return {
        "point": [(cx + 0.5) * dx, (cy + 0.5) * dy],
        "box": [a * dx, b * dy, c * dx, d * dy],
        "learned_point": point,
        "learned_region": fitted,
        "supported_three_by_three_centers": len(rows),
        "road_mask_sha256": mask_sha256(mask),
        "road_recipe_sha256": ROAD_RECIPE_SHA256,
    }


def native_record(record, config):
    backend = record.get("backend_details", {})
    identity = backend.get("weight_identity") or {}
    flags = ("injected", "fixture", "software_fixture", "substitute", "test_fixture")
    files = identity.get("files") or {}
    pinned = {**WEIGHT_SHA256, **METADATA_SHA256}
    need(
        backend.get("runtime") == "streamed"
        and backend.get("native_fault_isolation") == "spawn_process"
        and type(backend.get("worker_pid")) is int
        and backend["worker_pid"] > 0
        and isinstance(backend.get("device"), str)
        and re.fullmatch(r"cuda(?::[0-9]+)?", backend["device"])
        and backend.get("model_id") == identity.get("model_id") == config.model_id == STREAMED_MODEL_ID
        and backend.get("revision") == identity.get("revision") == config.revision == STREAMED_REVISION
        and identity.get("verified_weights") is True
        and identity.get("verified_metadata") is True
        and all(source.get(key) in (None, False) for source in (record, backend, identity) for key in flags)
        and set(files) == set(pinned)
        and all(
            files[name].get("sha256") == files[name].get("expected_sha256") == checksum
            and type(files[name].get("bytes")) is int
            and files[name]["bytes"] > 0
            for name, checksum in pinned.items()
        ),
        "raw_native_backend_unverified",
    )


def verify_crop_receipt(request, image, choice, evidence):
    _, index = landmark_messages(request)
    board, alternatives = evidence_board(image, index, choice)
    with board:
        need(
            evidence.get("render_sha256") == sha256(board.tobytes()).hexdigest()
            and evidence.get("render_size") == list(board.size),
            "verification_different_original_or_crop",
        )
    need(evidence.get("chosen") == choice, "verification_references_different_vehicles")
    context = {
        "prior_dialogue": [{"role": t["role"], "text": t["text"]} for t in request["dialogue"]],
        "passenger": request["text"],
        "candidate_boxes_original_pixels": choice,
        "image_size": list(image.size),
        "visible_alternative_ids": alternatives,
    }
    expected = [
        {"role": "system", "content": CHECK_GUIDE},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "current-frame-evidence-board"},
                {"type": "text", "text": json.dumps(context, ensure_ascii=False)},
            ],
        },
    ]
    need(evidence.get("messages") == expected, "verification_prompt_changed")


def candidate(request, original, model, image, binding):
    scene = request["scene"]
    need(scene["source"] == "predicted", "requires_detector_predictions")
    quotes = scoped_quotes(request["text"])
    info = original["provenance"]
    record = info["resolver"]
    need(
        info.get("method") == "qwen_landmark_region"
        and all(
            source.get(key) in (None, False)
            for source in (info, original)
            for key in ("injected", "fixture", "software_fixture", "substitute", "test_fixture")
        )
        and info.get("geometry_sha256") == model.geometry.sha256 == GEOMETRY
        and original.get("request_id") == request["request_id"]
        and info.get("image") == binding
        and record.get("image_sha256") == binding["sha256"],
        "wrong_request_image_or_geometry",
    )
    need(
        info.get("pair_decision", {}).get("kind") == "ask" and "pair_resolution" in record,
        "original_question_did_not_come_from_native_pair",
    )
    # Certify the unresolved route before asking the production grounder to reconstruct
    # it. A mismatched saved outcome cannot inadvertently enter its optional GPU verifier.
    checked = replay_pair_decision(request, binding, record["pair_resolution"], model.config)
    need(checked == info["pair_decision"] and checked["kind"] == "ask", "source_pair_question_changed")
    # Production from_decision revalidates raw full-utterance and both quoted-subquery
    # messages, adapter, source pixels, IDs, scope, model and revision; it generates no new text.
    rebuilt = model.from_decision(request, image, binding, record)
    if "resolver_execution" in info:
        need(info["resolver_execution"] == "live", "saved_pair_did_not_come_from_native_generation")
        rebuilt["provenance"]["resolver_execution"] = "live"
    need(rebuilt == original, "native_outcome_or_pair_source_changed")
    _, index = landmark_messages(request)
    frame = scene["frame"]
    whole = parse_landmark(record["raw_text"], index, frame)
    native_record(record, model.config)
    need(
        whole == record.get("decision") and whole.get("kind") == "landmark" and len(whole["track_ids"]) == 2,
        "full_utterance_did_not_identify_two_vehicles",
    )
    subs = record["pair_resolution"]["subqueries"]
    observations = []
    current = list(index.values())
    by_id = {t["track_id"]: t for t in current}
    for q, sub in zip(quotes, subs, strict=True):
        raw = sub.get("resolver", {})
        native_record(raw, model.config)
        parsed = parse_landmark(raw["raw_text"], index, frame)
        need(
            parsed == raw.get("decision")
            and parsed.get("kind") == "landmark"
            and len(parsed["track_ids"]) == 1,
            "quoted_reference_did_not_identify_one_vehicle",
        )
        obs = by_id[parsed["track_ids"][0]]
        need(category_matches(q["text"], obs), "native_selection_contradicts_vehicle_type")
        observations.append(obs)
    ids = [o["track_id"] for o in observations]
    need(len(set(ids)) == 2 and set(ids) == set(whole["track_ids"]), "independent_native_selections_disagree")
    # A literal relation to the *other* car can be checked in image coordinates. A
    # statement about a road/median on the passenger's left is not such a relation.
    x1, x2 = (center_x(o["bbox_xyxy"]) for o in observations)
    if re.search(r"\bto (?:its|the (?:other|first) (?:car|vehicle)['’]s) right\b", quotes[1]["text"], re.I):
        need(x2 > x1, "second_vehicle_disagrees_with_stated_side")
    if re.search(r"\bto (?:its|the (?:other|first) (?:car|vehicle)['’]s) left\b", quotes[1]["text"], re.I):
        need(x2 < x1, "second_vehicle_disagrees_with_stated_side")
    geometry = neighbor_geometry(observations, current, image.size)
    patch = supported_patch(
        model.road.mask(image, binding["sha256"]), image.size, model.geometry.pair_regression, geometry, current
    )
    return {
        "observations": observations,
        "quotes": quotes,
        "raw_outcome_sha256": digest(original),
        "native_pair_receipt_sha256": record["pair_resolution"]["receipt_sha256"],
        "request_sha256": digest(request),
        "image_sha256": binding["sha256"],
        "recording_id": frame["recording_id"],
        "frame_id": frame["frame_id"],
        "geometry_sha256": model.geometry.sha256,
        "excluded_current_observations_sha256": digest(current),
        **geometry,
        **patch,
    }


def run(request, original, model, *, verify=False):
    """Inspect a genuine native outcome. Only an explicit fresh native crop MATCH can create a proposal."""
    report = {"method": METHOD, "status": "original_retained", "verification_attempted": False}
    try:
        need(
            isinstance(original, dict)
            and original.get("ok") is True
            and original.get("result", {}).get("status") == "needs_clarification",
            "original_did_not_ask",
        )
        request = validate_request(request)
        scoped_quotes(request["text"])
        image, binding = resolve_image(request, model.config)
        with image:
            found = candidate(request, original, model, image, binding)
            report.update(candidate=found, status="eligible_for_crop_verification")
            if not verify:
                return original, report
            # The same loaded native resolver may be shared with the normal passenger
            # grounder. Replaying a saved call can start only the requested native verifier.
            with model.lock:
                resolver = model.resolver
                expected = verify_adapter(model.config)
                if resolver.backend is None:
                    from .streamed import StreamedProcessBackend

                    resolver.backend = StreamedProcessBackend(model.config)
                    resolver.adapter = expected
                need(resolver.adapter == expected, "verification_adapter_changed")
                anchors = [t["bbox_xyxy"] for t in found["observations"]]
                ids = [t["track_id"] for t in found["observations"]]
                choice = {"kind": "landmark", "boxes": anchors, "track_ids": ids}
                report["verification_attempted"] = True
                try:
                    decision, evidence = check_landmark(resolver, request, image, choice)
                except Exception:
                    resolver.close()
                    raise
                report["verification"] = evidence
                verify_crop_receipt(request, image, choice, evidence)
                native_record(evidence, model.config)
                need(verify_adapter(model.config) == expected, "verification_adapter_changed")
                need(
                    decision == choice and evidence.get("raw_text", "").strip() == "MATCH",
                    "crop_did_not_confirm_pair",
                )
                result = proposed(found["point"], found["box"], anchors)
                validate_target(result, list(result), request["scene"]["frame"])
                report["status"] = "new_proposal_requires_passenger_confirmation"
                info = {
                    **original["provenance"],
                    "projected_pair": deepcopy(report),
                    "original_result": deepcopy(original["result"]),
                    "selected_track_ids": ids,
                }
                return {**original, "result": result, "provenance": info}, report
    except Ineligible as error:
        report.update(status="original_retained", reason=str(error))
    except Exception as error:
        # Optional processing must never manufacture or erase the original application outcome.
        report.update(status="operational_error_original_retained", reason=f"{type(error).__name__}: {error}")
    return original, report


class ProjectedPairGrounder:
    """Opt-in passenger-session wrapper. The default web and batch grounders remain separate."""

    def __init__(self, model):
        from .infrastructure import PassengerDialogueGrounder

        self.model = model
        self.dialogue = PassengerDialogueGrounder(model)

    def interpret(self, request):
        with self.model.lock:
            original = self.dialogue.interpret(request)
            outcome, report = run(request, original, self.model, verify=True)
            if outcome is original and report["verification_attempted"]:
                # Keep the existing question and the actual failed or negative crop answer.
                return {**original, "provenance": {**original["provenance"], "projected_pair": report}}
            return outcome

    def passenger_area(self, request, box):
        return self.dialogue.passenger_area(request, box)
