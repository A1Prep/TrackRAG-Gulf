"""Ask which pair the passenger means when a gesture is missing or the visual identifications conflict."""
from copy import deepcopy
from hashlib import sha256
import json
import re

from .contracts import validate_request
from .data import inference_input_sha256
from .landmarks import landmark_messages, parse_landmark, resolve_image
from .model import verify_adapter
from .pair_resolution import PairResolutionError, replay_pair_decision
from .passenger import digest, unresolved
from .projected_pair import native_record
from .referent import ReferentRanker


QUESTION = "Which two vehicles do you mean? Describe each one or select both on the photo."
FEW_VEHICLES = "Which two vehicles do you mean? Could you describe another landmark or mark the road area on the photo?"
_DEICTIC_PAIR = re.compile(
    r"(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
    r"(?:(?:drop\s+(?:me|us)(?:\s+off)?|let\s+(?:me|us)\s+out|stop|pull\s+over|park|"
    r"i(?:\s+want|\s+would\s+like|['’]d\s+like)\s+to\s+get\s+out)\s+)?"
    r"(?:right\s+)?(?:in\s+(?:the\s+(?:gap|space)\s+)?)?between\s+(?:these|those|the)\s+(?:two|2)\s+"
    r"(?:(black|white|gr[ae]y|silver|red|blue|green|yellow)\s+)?"
    r"(cars|sedans|suvs|vehicles|trucks|buses)(?:[\s,]+please)?[\s.!?]*", re.I)
_CLASSES = {"cars": {"Car"}, "sedans": {"Car"}, "suvs": {"Car"},
            "trucks": {"Large_vehicle", "Medium_vehicle"}, "buses": {"Bus"},
            "vehicles": {"Car", "Large_vehicle", "Medium_vehicle", "Bus", "Emergency_vehicle",
                         "Motorbike", "Small_motorised_vehicle"}}


def unpointed_pair(model, request):
    """An initial demonstrative with no identifiable pair can be resolved by asking for pointing."""
    if not isinstance(request, dict) or not isinstance(request.get("text"), str) or request.get("dialogue"):
        return None
    match = _DEICTIC_PAIR.fullmatch(request["text"].strip())
    ranker = getattr(model, "referent_ranker", None)
    if not match or not isinstance(ranker, ReferentRanker) or not re.fullmatch(r"[0-9a-f]{64}", ranker.sha256 or ""):
        return None
    try:
        clean = validate_request(request)
        scene = clean["scene"]
        if not any(row["frame_id"] == scene["frame"]["frame_id"] for row in scene["tracks"]):
            return None
        image, binding = resolve_image(clean, model.config)
        with image:
            ranking = ranker.rank(image, clean["text"], scene["tracks"], frame=scene["frame"])
    except (OSError, AttributeError, KeyError, TypeError, ValueError):
        # The normal model path reports invalid input and missing originals to the passenger.
        return None
    colour, noun = match.groups()
    colour = (colour or "").lower()
    colour = "gray" if colour in {"grey", "silver"} else colour
    # Sample both the broad box and its centre. Dark windows, reflections and shade
    # make paint uncertain. These checks can ask a question; they never select a point.
    threshold = .20 if colour in {"white", "gray", "black"} else .16
    candidates = [row for row in ranking["alternatives"] if row["class_name"] in _CLASSES[noun.lower()]]
    plausible = [row["track_id"] for row in candidates if not colour or min(
        row["colour_fractions"][colour], row["central_colour_fractions"][colour]) >= threshold]
    if len(plausible) == 2:
        return None
    visible = sum(row["class_name"] in _CLASSES["vehicles"] and row["frame_id"] == scene["frame"]["frame_id"]
                  for row in scene["tracks"])
    return {"ok": True, "request_id": clean["request_id"], "result": unresolved(QUESTION if visible >= 2 else FEW_VEHICLES),
            "provenance": {"method": "deictic_pair_clarification", "request_sha256": digest(clean), "image": binding,
                           "inference_input_sha256": inference_input_sha256(clean, binding["sha256"]),
                           "reason": "passenger_referenced_an_unpointed_pair", "passenger_phrase": match.group(),
                           "colour": colour or None, "eligible_tracks": plausible, "candidate_count": len(candidates),
                           "ranker_sha256": ranker.sha256, "ranking_sha256": digest(ranking),
                           "native_model_calls": 0}}


def review_pair_disagreement(model, request, outcome):
    """Withhold a proposed point if two native readings identify different sets of vehicles."""
    if (not isinstance(request, dict) or not isinstance(outcome, dict) or outcome.get("ok") is not True
            or outcome.get("request_id") != request.get("request_id")
            or not isinstance(outcome.get("result"), dict) or outcome["result"].get("status") != "proposed"):
        return outcome
    source = outcome.get("provenance", {})
    if not isinstance(source, dict) or not isinstance(source.get("resolver"), dict):
        return outcome
    if (source.get("independent_model_proposal") is False or any(record.get(key) not in (None, False)
            for record in (outcome, source) for key in ("fixture", "software_fixture", "injected", "substitute", "test_fixture"))):
        return outcome
    whole = source.get("resolver", {})
    if (source.get("method") != "qwen_landmark_region" or "pair_resolution" not in whole
            or source.get("visual_verification") is not None):
        return outcome
    try:
        clean = validate_request(request)
        messages, index = landmark_messages(clean)
        frame = clean["scene"]["frame"]
        before = parse_landmark(whole.get("raw_text"), index, frame)
        if before.get("kind") != "landmark" or len(before["track_ids"]) != 2:
            return outcome
        image, binding = resolve_image(clean, model.config)
        image.close()
        qhash = digest(clean)
        if (source.get("image") != binding or source.get("request_sha256") != qhash
                or source.get("inference_input_sha256") != inference_input_sha256(clean, binding["sha256"])
                or whole.get("image_sha256") != binding["sha256"] or whole.get("request_sha256") != qhash
                or whole.get("model_id") != model.config.model_id or whole.get("revision") != model.config.revision
                or whole.get("adapter") != verify_adapter(model.config) or whole.get("messages") != messages
                or whole.get("messages_sha256") != sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
                or whole.get("decision") != before):
            return outcome
        quotes = whole["pair_resolution"]
        second = replay_pair_decision(clean, binding, quotes, model.config)
        if (second.get("kind") != "landmark" or len(second["track_ids"]) != 2
                or second != source.get("pair_decision") or second["track_ids"] != source.get("selected_track_ids")
                or second["boxes"] != outcome["result"].get("anchor_boxes_xyxy")
                or set(before["track_ids"]) == set(second["track_ids"])):
            return outcome
        native_record(whole, model.config)
        for sub in quotes["subqueries"]:
            native_record(sub["resolver"], model.config)
        details = {"reason": "different_current_tracks_from_full_and_separate_descriptions",
                   "whole_sentence_ids": before["track_ids"], "separate_description_ids": second["track_ids"],
                   "original_proposal": deepcopy(outcome["result"]), "pair_receipt_sha256": quotes["receipt_sha256"]}
        return {**outcome, "result": unresolved(QUESTION), "provenance": {**source, "pair_disagreement": details}}
    except (AttributeError, OSError, KeyError, ValueError, TypeError, PairResolutionError):
        return outcome
