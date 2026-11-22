"""Withhold a sole detector road-user anchor when the passenger specifies only fixed infrastructure."""
from copy import deepcopy
import re

from trackrag.vision.classes import EMT_CLASSES

from .contracts import validate_request
from .data import inference_input_sha256
from .landmarks import landmark_messages, parse_landmark, resolve_image
from .passenger import digest, instruction_context, unresolved
from .perspective import PassengerViewpoint
from .referent import ReferentRanker


VERSION = "infrastructure-road-user-clarification-v1"
QUESTION = "Could you mark the road area you mean on the photo, or describe another visible landmark?"

# Match fixed compound nouns before inspecting the surrounding words. A bus stop and
# a pedestrian crossing are structures even when their names contain a road-user noun.
_STRUCTURE = re.compile(
    r"\b(?:bus[- ]+(?:stops?|shelters?|stations?)|truck[- ]+stops?|car[- ]+parks?|parking[- ]+(?:lots?|garages?)|"
    r"(?:pedestrian|zebra)[- ]+crossings?|crosswalks?|(?:cycle|bike|bicycle)[- ]+(?:lanes?|racks?)|"
    r"(?:(?:traffic|signal)[- ]+(?:lights?|signals?|poles?)|(?:street|lamp|light)[- ]*(?:lights?|lamps?|posts?|poles?))|"
    r"lampposts?|streetlights?|streetlamps?|lamps?|poles?|(?:road|exit|traffic|stop|directional|speed[- ]+limit)[- ]+signs?|"
    r"signposts?|signs?|guard[- ]?rails?|guard[- ]?railings?|(?:crash|safety)[- ]+barriers?|barriers?|railings?|"
    r"dividers?|medians?|bollards?|walls?|fences?|gates?|entrances?|doorways?|shelters?|buildings?|"
    r"bridges?|overpasses|overpass|underpasses|underpass|flyovers?|roundabouts?|intersections?|"
    r"sidewalks?|footpaths?|pavements?|(?:curb|kerb)(?:[- ]+(?:cuts?|ramps?))?s?|"
    r"(?:lane|road)[- ]+markings?|painted[- ]+(?:arrows?|crossings?))\b", re.I)
_WORDS = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?|\d+", re.UNICODE)
_ROAD_USER = re.compile(
    r"\b(?:cars?|vehicles?|buses|bus|coaches|coach|trucks?|vans?|lorr(?:y|ies)|suvs?|pick[- ]?ups?|"
    r"taxis?|cabs?|sedans?|hatchbacks?|coupes?|jeeps?|minivans?|motorbikes?|motorcycles?|scooters?|"
    r"cyclists?|bicycles?|bikes?|walkers?|people|persons?|pedestrians?|riders?|m[ae]n|wom[ae]n|"
    r"(?:school|delivery|police|fire|ambulance)[- ]+(?:vans?|cars?|trucks?)|ambulances?|"
    r"mercedes|lexus|toyota|nissan|bmw|ford|honda|audi|tesla|kia|hyundai|volkswagen|porsche)\b", re.I)
_RESET = re.compile(r"\b(?:instead(?!\s+of)|forget (?:it|that)|change (?:it|that))\b", re.I)
_SCOPE_CHANGE = re.compile(r"\b(?:actually|i mean|i meant|forget (?:it|that)|change (?:it|that)|instead(?!\s+of))\b", re.I)
# An unrecognised noun, action or implied alternative could identify a moving object.
# Only accept ordinary destination wording around explicit infrastructure nouns.
_SURROUNDING_WORDS = frozenset("""
a an the this that those these my our your its i we me us you can could would please may
let take leave drop off out get stop pull over mark show find point at on in of beside
by near next to opposite before after ahead behind past beneath below under between and
above around across from right left side front far near nearest closest first second
just directly immediately here there now want need mean meant like road roadway street
asphalt tarmac patch area spot surface shoulder base foot corner top bottom rightmost
leftmost middle center centre tall tallest high low small big large short narrow wide
long concrete metal wooden stone brick steel painted marked striped red green blue white
black grey gray yellow orange silver dark light main railway overhead along with
actually instead i'd drive lane lanes
""".split())


def _without_spans(text, matches):
    chars = list(text)
    for match in matches:
        chars[match.start():match.end()] = " " * (match.end() - match.start())
    return "".join(chars)


def _current_start(text):
    start = 0
    for change in _SCOPE_CHANGE.finditer(text):
        if _STRUCTURE.search(text, change.end()) or _ROAD_USER.search(text, change.end()):
            start = change.end()
    return start


def _could_name_something_else(text):
    remaining = _STRUCTURE.sub(" ", text)
    return any(word.group().lower().replace("’", "'") not in _SURROUNDING_WORDS
               for word in _WORDS.finditer(remaining))


def _active_structures(request):
    """Return exact quotes from the current turn only, when its explicit scope is unambiguous."""
    text = request["text"]
    previous_other = False
    for turn in request["dialogue"]:
        if turn["role"] != "user":
            continue
        spoken = turn["text"]
        if _RESET.search(spoken):
            previous_other = False
        previous_other |= _could_name_something_else(spoken[_current_start(spoken):])
    # A short clarification such as "under the bridge" can narrow an earlier car,
    # an unknown vehicle brand, or a passenger's "the dark one".
    start = _current_start(text)
    if (previous_other or _could_name_something_else(text[:start])) and not _RESET.search(text):
        return []
    scope = text[start:]
    matches = list(_STRUCTURE.finditer(scope))
    if not matches:
        return []
    remaining = _without_spans(scope, matches)
    if any(word.group().lower().replace("’", "'") not in _SURROUNDING_WORDS for word in _WORDS.finditer(remaining)):
        return []
    return [{"text": match.group(), "start": start + match.start(), "end": start + match.end()}
            for match in matches]


def _bound_track(request, outcome):
    info, result = outcome["provenance"], outcome["result"]
    resolver = info.get("resolver")
    selected = info.get("selected_track_ids")
    if (not isinstance(resolver, dict) or not isinstance(selected, list) or len(selected) != 1
            or type(selected[0]) is not int or "pair_decision" in info or "pair_resolution" in resolver):
        return None
    if info.get("visual_verification", {}).get("raw_text", "MATCH").strip() != "MATCH":
        return None
    clean = validate_request(request)
    scene, expected = clean["scene"], digest(request)
    frame = scene["frame"]
    image_hash = scene.get("provenance", {}).get("image_sha256")
    if (not isinstance(image_hash, str) or not re.fullmatch("[0-9a-f]{64}", image_hash)
            or info.get("request_sha256") != expected or resolver.get("request_sha256") != expected
            or info.get("image", {}).get("sha256") != image_hash
            or (resolver.get("image_sha256") or resolver.get("image", {}).get("sha256")) != image_hash):
        return None
    _, index = landmark_messages(clean)
    decision = parse_landmark(resolver.get("raw_text"), index, frame)
    if (decision.get("kind") != "landmark" or decision.get("track_ids") != selected
            or resolver.get("decision") != decision or decision.get("boxes") != result.get("anchor_boxes_xyxy")):
        return None
    return next((track for track in index.values()
                 if track["track_id"] == selected[0] and track["class_name"] in EMT_CLASSES
                 and decision["boxes"] == [track["bbox_xyxy"]]), None)


def review(request, outcome):
    """Return a normal outcome; a confirmed contradiction can only yield a request for clarification.

    The image model's raw response, selected detector observations and area remain in the
    audit. Unknown classes, image-derived OBJECT boxes and explicit passenger drawings are
    outside this policy. There is no predicted structure, substitute area or confidence score.
    """
    if (not isinstance(outcome, dict) or outcome.get("ok") is not True or not isinstance(request, dict)
            or outcome.get("request_id") != request.get("request_id")
            or not isinstance(outcome.get("provenance"), dict) or not isinstance(outcome.get("result"), dict)
            or outcome["result"].get("status") != "proposed"
            or outcome["provenance"].get("method") != "qwen_landmark_region"):
        return outcome
    try:
        quotes = _active_structures(request)
        track = _bound_track(request, outcome) if quotes else None
    except (KeyError, AttributeError, TypeError, ValueError):
        # A missing/malformed source is for the grounder/session contract to validate.
        # Never attribute a prediction to a detector object using incomplete evidence.
        return outcome
    if track is None:
        return outcome
    info = outcome["provenance"]
    review_record = {
        "version": VERSION, "action": "ask_for_location",
        "reason": "explicit_fixed_infrastructure_with_only_a_tracked_road_user",
        "passenger_reference_spans": quotes, "selected_observation": deepcopy(track),
        "request_sha256": info["request_sha256"], "image_sha256": info["image"]["sha256"],
        "original_method": info["method"], "raw_landmark_response": info["resolver"]["raw_text"],
        "original_result": deepcopy(outcome["result"]),
    }
    return {**outcome, "result": unresolved(QUESTION),
            "provenance": {**info, "infrastructure_review": review_record}}


class PassengerDialogueGrounder(PassengerViewpoint):
    def passenger_pair(self, request, track_ids, placement, *, passenger_requested=False):
        from .passenger_pair_choice import choose
        with self.model.lock:
            return choose(self.model, request, track_ids, placement, passenger_requested=passenger_requested)

    def _checked_viewpoint(self, request, outcome):
        """Reopen the bound original and reproduce the ranking before offering a new choice."""
        if (not isinstance(outcome, dict) or outcome.get("ok") is not True
                or outcome.get("request_id") != request.get("request_id")
                or outcome.get("result", {}).get("status") != "proposed"):
            return outcome
        provenance = outcome.get("provenance", {})
        ranker = getattr(self.model, "referent_ranker", None)
        ranking = provenance.get("referent_ranking")
        if (provenance.get("method") != "qwen_landmark_region" or not isinstance(ranking, dict)
                or not isinstance(ranker, ReferentRanker) or not isinstance(ranker.sha256, str)
                or not re.fullmatch("[0-9a-f]{64}", ranker.sha256) or ranking.get("checkpoint_sha256") != ranker.sha256):
            return outcome
        try:
            clean = validate_request(request)
            scene = clean["scene"]
            photo = scene.get("provenance", {}).get("image_sha256")
            request_hash = digest(clean)
            resolver = provenance["resolver"]
            if (not isinstance(photo, str) or not re.fullmatch("[0-9a-f]{64}", photo)
                    or provenance["image"]["sha256"] != photo or resolver.get("image_sha256") != photo
                    or provenance.get("request_sha256") != request_hash or resolver.get("request_sha256") != request_hash
                    or provenance.get("inference_input_sha256") != inference_input_sha256(clean, photo)):
                return outcome
            candidate = PassengerViewpoint.review(clean, outcome)
            if candidate is outcome:
                return outcome
            # The initially proposed point can legitimately have an untracked OBJECT
            # anchor. In either case its recorded boxes and IDs must match the raw Qwen
            # decision we will keep in the audit. A later, different visual choice has
            # separate provenance; do not relabel it as this single-source decision.
            if ("pair_resolution" in resolver or "pair_decision" in provenance or
                    provenance.get("visual_verification", {}).get("raw_text", "MATCH").strip() != "MATCH"):
                return outcome
            _, index = landmark_messages(clean)
            initial = parse_landmark(resolver.get("raw_text"), index, scene["frame"])
            if (initial.get("kind") not in {"object", "landmark"} or len(initial["boxes"]) != 1
                    or initial != resolver.get("decision") or initial["boxes"] != outcome["result"].get("anchor_boxes_xyxy")
                    or initial["track_ids"] != provenance.get("selected_track_ids")):
                return outcome
            image, _ = resolve_image(clean, self.model.config)
            with image:
                actual = ranker.rank(image, instruction_context(clean), scene["tracks"], frame=scene["frame"])
            # rank() reads every alternative's current ID, class, box and colours from
            # this exact frame. Also reject reordered, removed or differently scored alternatives.
            if actual != ranking:
                return outcome
            rationale = candidate["provenance"]["viewpoint_clarification"]
            observed = {t["track_id"]:t for t in scene["tracks"] if t["frame_id"] == scene["frame"]["frame_id"]}
            selected = [deepcopy(observed[key]) for key in rationale["decision"]["track_ids"]]
            rationale["source_binding"] = {
                "request_sha256": request_hash, "inference_input_sha256": provenance["inference_input_sha256"],
                "image_sha256": photo, "checkpoint_sha256": ranker.sha256, "ranked_evidence_sha256": digest(actual),
                "current_observations": selected, "verification": "ranking_recomputed_from_current_original_pixels",
            }
            return candidate
        except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError):
            return outcome

    def interpret(self, request):
        from .pair_dialogue import review_pair_disagreement, unpointed_pair
        question = unpointed_pair(self.model, request)
        if question is not None:
            return question
        native = self.model.interpret(request)
        native = review_pair_disagreement(self.model, request, native)
        return review(request, self._checked_viewpoint(request, native))
