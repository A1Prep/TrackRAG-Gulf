"""Match passenger words and marks to visible parking stalls on the exact original frame."""
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from threading import Lock

from PIL import Image

from . import bay_landmarks, manual_offlane, manual_surfaces, place_grounding
from .parking_evidence import source_input
from .passenger import digest, proposed, unresolved

AUTO = "public_painted_bay_landmark_v1"
MARK = "passenger_marked_painted_bay_v1"
CHOICE = "passenger_selected_painted_bay_v1"
MANUAL_METHODS = {MARK, CHOICE, manual_offlane.METHOD}
CLEARANCE_POLICY = "parking_clearance_with_current_vehicles_v2"
_cache, _lock = OrderedDict(), Lock()
_index_cache = OrderedDict()

def visible_stalls(original, classes, context, surface=None, stall_review=None):
    import cv2
    import numpy as np
    from . import curbside_bays, painted_bays, surface_stalls
    stalls = painted_bays.candidates(original, classes, context)
    additional = curbside_bays.candidates(original, classes, context)
    if stall_review is not None:
        additional.extend(surface_stalls.approved(original, classes, context, surface, stall_review))
    for extra in additional:
        p = np.float32(extra["polygon"])
        # When both detectors describe the same space, keep the stall supported by the
        # established painted-divider and semantic parking checks. Keep true neighbours.
        overlaps = False
        for stall in stalls:
            q = np.float32(stall["polygon"])
            if cv2.intersectConvexConvex(p, q)[0] > .14 * min(cv2.contourArea(p), cv2.contourArea(q)):
                overlaps = True
                break
        if not overlaps:
            stalls.append(extra)
    return sorted(stalls,key=lambda stall:tuple(stall["point"]))


def _layout(request, photo, surface, stall_review=None):
    clean, source, portable = source_input(request, photo)
    scene = clean["scene"]
    frame = scene["frame"]
    context = {"size":[frame["width"], frame["height"]], "image_sha256":source,
               "current_boxes":[t["bbox_xyxy"] for t in scene["tracks"] if t["frame_id"] == frame["frame_id"]]}
    classes = manual_surfaces.decode(surface, frame, source)
    key = digest([context,surface["signature"],stall_review])
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return clean, portable, classes, deepcopy(_cache[key])
    with Image.open(BytesIO(photo)) as original:
        options = visible_stalls(original.convert("RGB"), classes, context, surface, stall_review)
    with _lock:
        _cache[key] = deepcopy(options)
        if len(_cache) > 96:
            _cache.popitem(last=False)
    return clean, portable, classes, options


def index_has_outline(scene, photo, indexed_shapes):
    """Match the gallery shortcut to the same surviving shapes as the normal request."""
    from . import bay_clearance, surface_stalls
    key = scene["provenance"]["image_sha256"]
    surface = manual_surfaces.get_packet(key)
    if surface is None:
        return False
    review = surface_stalls.get_packet(key)
    frame_id = scene["frame"]["frame_id"]
    current = [t for t in scene["tracks"] if t["frame_id"] == frame_id]
    stamp = digest([key, current, sorted(indexed_shapes), surface["signature"], review and review["signature"]])
    with _lock:
        if stamp in _index_cache:
            _index_cache.move_to_end(stamp)
            return _index_cache[stamp]
    request = {"scene":scene, "text":"Find the marked parking bays.", "input_mode":"typed", "dialogue":[], "request_id":"catalog"}
    _, _, _, options = _layout(request, photo, surface, review)
    matches = {digest(o["polygon"]) for o in options} == indexed_shapes and any(
        bay_clearance.vehicle_sized(o, scene) for o in options)
    with _lock:
        _index_cache[stamp] = matches
        if len(_index_cache) > 96:
            _index_cache.popitem(last=False)
    return matches


def _present(options, tap):
    from . import surface_stalls
    choices = []
    for index, option in enumerate(options, 1):
        item = {"id": f"bay-{index}", "polygon_xy": option["polygon"], "center_xy": option["point"],
                "passenger_tap_xy": tap}
        # A visually reviewed pavement component can end at the edge of the segmentation.
        # Only the painted-divider and kerb-end detectors reconstruct the entire measured stall.
        if option["evidence"]["recipe"] == surface_stalls.RECIPE:
            item["extent"] = "visible_parking_surface"
        choices.append(item)
    return choices


def _prefer_measured(options):
    from . import surface_stalls
    # For the same words, prefer complete stall outlines to clipped pieces of the parking
    # surface. A partial outline remains useful when it is the only matching location.
    measured = [o for o in options if o["evidence"]["recipe"] != surface_stalls.RECIPE]
    return measured or options


def _make(clean, portable, options, surface, *, method, reason, question, tap=None, require_choice=False, **evidence):
    choices = _present(options,tap)
    one = len(options) == 1 and not require_choice
    result = proposed(options[0]["point"],options[0]["box"]) if one else unresolved(question)
    return {"ok":True, "request_id":clean["request_id"], "result":result, "provenance":{
        "method":method, "image":{"sha256":clean["scene"]["provenance"]["image_sha256"]},
        "request_sha256":digest(clean), "inference_input_sha256":portable, "reason":reason,
        "independent_model_proposal":method==AUTO and one, "surface_evidence":surface,
        "bay_outline":choices[0] if one else None, "bay_choices":[] if one else choices, **evidence}}


def mark_outcome(request, photo, surface, box, stall_review=None, *, active_policy=True):
    from . import bay_clearance, painted_bays
    clean, portable, classes, options = _layout(request, photo, surface, stall_review)
    clean, _, mark, center = manual_offlane.mark_input(clean,box)
    frame = clean["scene"]["frame"]
    reason, checked = manual_offlane.surface_check(frame,clean["scene"]["tracks"],mark,center,classes)
    if reason in {"observed_object","broad_area"}:
        return None
    evidence = {"stall_review": stall_review} if stall_review else {}
    if active_policy:
        evidence["clearance_policy"] = CLEARANCE_POLICY
        usable = [o for o in options if bay_clearance.vehicle_sized(o, clean["scene"])]
        rejected = [o for o in options if o not in usable]
        if (painted_bays.matching_options(rejected,mark,center,frame,classes,offer_visible_side=True)
                and not painted_bays.matching_options(usable,mark,center,frame,classes,offer_visible_side=True)):
            return _make(clean,portable,[],surface,method=MARK,reason="parking_fragment_at_vehicle",
                         question="Only a narrow strip of pavement is visible beside the parked vehicle there. Could you mark a different open space?",
                         tap=center,selected_box=mark,**evidence)
        options = usable
    matches = painted_bays.matching_options(options,mark,center,frame,classes)
    edge_offer = len(matches) == 1 and checked.get("center_class_id") == painted_bays.LINE
    if not matches:
        matches = painted_bays.matching_options(options,mark,center,frame,classes,offer_visible_side=True)
        edge_offer = len(matches) == 1
    if not matches:
        return None
    reason = "single_visible_bay_at_painted_edge" if edge_offer else "passenger_mark_on_painted_bay"
    question = ("I can outline one bay beside this painted line. Did you mean this side, or would you like to mark another?"
                if edge_offer else "Your mark is on a boundary shared by the highlighted bays. Which bay did you mean?")
    return _make(clean,portable,matches,surface,method=MARK,reason=reason,require_choice=edge_offer,question=question,
                 tap=center,selected_box=mark,**evidence)


def select_mark(request, box):
    from . import surface_stalls
    clean, original_sha, mark, _ = manual_offlane.mark_input(request,box)
    try:
        packet = manual_surfaces.get_packet(original_sha)
        if packet is not None:
            data = Path(clean["scene"]["frame"]["image_path"]).read_bytes()
            stall_review = surface_stalls.get_packet(original_sha)
            try:
                bay = mark_outcome(clean,data,packet,mark,stall_review)
            except (KeyError,TypeError,ValueError):
                if stall_review is None:
                    raise
                bay = mark_outcome(clean,data,packet,mark)
            if bay is not None:
                return bay
    except (OSError, KeyError, TypeError, ValueError):
        pass
    # Preserve the historical exact-point check when no supported stall accounts for the mark.
    return manual_offlane.select(clean,box)


_GENERIC = re.compile(r"\b(?:bays?|parking (?:spots?|stalls?|spaces?)|(?:marked|painted) (?:spots?|stalls?|spaces?))\b",re.I)
_UNSUPPORTED = re.compile(r"\b(?:not|except|other than|instead of|opposite|across from|before|after|past|between|(?:left|right) of)\b|n’t|n't",re.I)
_STOP = set(("a an the at next to beside by near outside in front of alongside adjacent along on with please "
             "could can would will you i we me us my our your like want prefer need take drop off let leave get out "
             "stop put pull over park here there right left hand side photo picture open empty unoccupied vacant "
             "marked painted parking bay bays space spaces spot spots stall stalls closest nearest farthest furthest "
             "one this that those these very just directly visible shown beside end where is are it closest "
             "use choose find select try show free available into enter let lets give").split())
_ALIAS = {"restaurants":"restaurant","eateries":"restaurant","eatery":"restaurant","cafes":"cafe","café":"cafe",
          "shops":"shop","store":"shop","stores":"shop","pharmacies":"pharmacy","garages":"garage","doors":"door",
          "entrances":"entrance","houses":"house","trees":"tree","palms":"palm","buildings":"building",
          "supermarkets":"supermarket","markets":"market","groceries":"grocery","carpark":"parking"}
_UNCERTAIN_CAPTION = re.compile(r"\b(?:none|unknown|unclear|unreadable|illegible|maybe|possibly|uncertain|no|not)\b",re.I)


def _terms(words):
    return [_ALIAS.get(word,word) for word in re.findall(r"[^\W_]+",words.casefold()) if word not in _STOP]


def language_candidates(request, options, labels):
    words = place_grounding.place_reference(request)
    if not words or _UNSUPPORTED.search(words):
        return []
    terms = set(_terms(words))
    if not terms and not _GENERIC.search(words):
        return []
    bands = [i for i,label in enumerate(labels or []) if terms and
             any(not _UNCERTAIN_CAPTION.search(part) and terms <= set(_terms(part)) for part in label.split("|"))]
    if terms and not bands:
        return []
    width = request["scene"]["frame"]["width"]
    left,right = bool(re.search(r"\bleft\b",words,re.I)),bool(re.search(r"\bright\b",words,re.I))
    if left and right:
        return []
    found = []
    for option in options:
        # Associate the kerb end of the stall with the part of the scene showing the named frontage.
        evidence = option["evidence"]
        if "frontage_xy" in evidence:
            x = evidence["frontage_xy"][0]/width
        else:
            back = [boundary[-1] for boundary in evidence["painted_boundaries"]]
            x = sum(point[0] for point in back)/(len(back)*width)
        if (left and x > .53) or (right and x < .47):
            continue
        if bands and not any(bay_landmarks.BANDS[i][0] <= x <= bay_landmarks.BANDS[i][1] for i in bands):
            continue
        found.append(option)
    if not terms and len(found)>1 and re.search(r"\b(?:closest|nearest|furthest|farthest)\b",words,re.I):
        reverse = bool(re.search(r"\b(?:closest|nearest)\b",words,re.I))
        by_depth = sorted(found,key=lambda o:o["point"][1],reverse=reverse)
        height = request["scene"]["frame"]["height"]
        if abs(by_depth[0]["point"][1]-by_depth[1]["point"][1]) > .05*height:
            return by_depth[:1]
    return found


def automatic_outcome(request, photo, surface, landmarks, stall_review=None, *, active_policy=True, prefer_complete=True):
    from . import bay_clearance
    clean, portable, _, options = _layout(request,photo,surface,stall_review)
    if not options:
        return None
    f = clean["scene"]["frame"]
    words = place_grounding.place_reference(clean)
    if not words:
        return None
    labels = bay_landmarks.verify(landmarks,photo,(f["width"],f["height"])) if landmarks is not None else None
    matches = language_candidates(clean,options,labels)
    evidence = {"stall_review":stall_review} if stall_review else {}
    if active_policy:
        evidence["clearance_policy"] = CLEARANCE_POLICY
        clear = [o for o in options if bay_clearance.vehicle_sized(o,clean["scene"])]
        if not clear or (matches and not any(o in clear for o in matches)):
            return _make(clean,portable,[],surface,method=AUTO,reason="parking_fragments_at_vehicles",
                         question="I can see parking pavement near the parked vehicles, but I can’t outline an empty bay there. Can you see another open space?",
                         landmark_evidence=landmarks,**evidence)
        matches = [o for o in matches if o in clear]
    if prefer_complete:
        matches = _prefer_measured(matches)
    if not matches:
        return None
    return _make(clean,portable,matches,surface,method=AUTO,reason="painted_bay_matches_instruction",
                 question="I can see more than one marked bay that fits. Which highlighted bay would you like?",
                 landmark_evidence=landmarks,**evidence)


def automatic(request, photo):
    from . import surface_stalls
    words = place_grounding.place_reference(request)
    if not words or _UNSUPPORTED.search(words):
        return None
    source = sha256(photo).hexdigest()
    try:
        surface = manual_surfaces.get_packet(source)
        if surface is None:
            return None
        # A place name needs evidence for that landmark on this original.
        needs_labels = bool(_terms(words))
        labels = bay_landmarks.get_packet(source) if needs_labels else None
        if needs_labels and labels is None:
            return None
        stall_review = surface_stalls.get_packet(source)
        try:
            return automatic_outcome(request,photo,surface,labels,stall_review)
        except (KeyError,TypeError,ValueError):
            if stall_review is None:
                raise
            return automatic_outcome(request,photo,surface,labels)
    except (OSError, TypeError, ValueError, KeyError):
        # Missing or invalid supplementary evidence leaves the trained place route available.
        return None


def explicit_outcome(request, photo, surface, stall_review, anchors, *, active_policy=True):
    """Check a request whose destination is a bay, retaining its vehicle and frontage qualifiers."""
    from . import bay_anchors, bay_clearance, bay_intent
    if bay_intent.parse(request["text"]) is None:
        return None
    clean,portable,_,options=_layout(request,photo,surface,stall_review)
    with Image.open(BytesIO(photo)) as im:
        observations=bay_anchors.observations(im,options,clean["scene"],anchors)
    original_options = options
    evidence = {"stall_review":stall_review} if stall_review else {}
    if active_policy:
        evidence["clearance_policy"] = CLEARANCE_POLICY
        options = [o for o in options if bay_clearance.vehicle_sized(o, clean["scene"])]
    supported,possible=bay_intent.match(clean["text"],options,clean["scene"],observations)
    unresolved_reference=not supported
    shown=_prefer_measured(supported or possible)
    if supported:
        reason="visible_bay_matches_stated_anchors"
        ask="I found several outlined bays that fit those details. Which bay would you like?"
    elif possible:
        reason="unverified_bay_reference"
        ask=("I can outline this bay, but I couldn’t match every detail you gave. Is this the bay you meant?"
             if len(possible)==1 else
             "I can outline these bays, but I couldn’t match every detail you gave. Which one did you mean?")
    elif original_options and not options:
        reason="parking_fragments_at_vehicles"
        ask="I can see strips of parking pavement beside the parked vehicles, but I can’t outline an empty bay there. Can you see another open space?"
    else:
        reason="no_supported_bay_for_instruction" if options else "no_visible_bay_for_instruction"
        ask=("I couldn’t match an outlined bay to those details. Could you tap inside the bay you mean?"
             if options else
             "I couldn’t trace an empty bay in this view. Could you tap inside the space you mean or describe another landmark?")
    return _make(clean,portable,shown,surface,method=AUTO,reason=reason,question=ask,require_choice=unresolved_reference,
                 bay_instruction_checked=True,bay_anchor_evidence=anchors,
                 **evidence)


def explicit_request(request):
    from . import bay_anchors, bay_intent, surface_stalls
    intent = bay_intent.parse(request["text"])
    if intent is None:
        return None
    key = request["scene"]["provenance"]["image_sha256"]
    surface = manual_surfaces.get_packet(key)
    if surface is None:
        return None
    photo = Path(request["scene"]["frame"]["image_path"]).read_bytes()
    review = surface_stalls.get_packet(key)
    anchors = bay_anchors.get_packet(key)
    # The signed region review and the signed language observations are independent. A stale
    # observation must not remove a geometrically valid option supported by a good region review.
    packets = [(review, anchors)]
    if anchors is not None:
        packets.append((review, None))
    if review is not None:
        packets.append((None, None))
    for region_packet, context_packet in packets:
        try:
            result = explicit_outcome(request, photo, surface, region_packet, context_packet)
        except (ValueError, KeyError, TypeError):
            continue
        if (result["provenance"]["reason"] == "no_visible_bay_for_instruction"
                and not (intent.vehicles or intent.blocked or intent.uncertain or intent.rank or intent.orientation)
                and place_grounding.place_reference(request)):
            # No measured footprint. The ordinary trained place grounder may still produce a
            # supported ground point or its own clarification from this exact original.
            return None
        return result
    return None


def choice_outcome(request, offered, bay_id):
    source = offered["provenance"]
    choices = source.get("bay_choices") or []
    if (source.get("method") not in {AUTO,MARK} or offered["result"]["status"] != "needs_clarification"
            or not isinstance(bay_id,str) or not choices or
            (len(choices)==1 and source.get("reason") not in {"single_visible_bay_at_painted_edge","unverified_bay_reference"})):
        raise ValueError("Choose a bay from the current highlighted alternatives.")
    verify(request,offered,requested_box=source.get("selected_box"))
    selected = next((row for row in choices if row["id"]==bay_id),None)
    if selected is None:
        raise ValueError("This bay is outside the current choices.")
    polygon = selected["polygon_xy"]
    xs, ys = zip(*polygon)
    return {"ok":True,"request_id":request["request_id"],
            "result":proposed(selected["center_xy"],[min(xs),min(ys),max(xs),max(ys)]),
            "provenance":{"method":CHOICE,"request_sha256":digest(request),"image":deepcopy(source["image"]),
                          "inference_input_sha256":source["inference_input_sha256"],"independent_model_proposal":False,
                          "choice_source":offered,"bay_id":bay_id,"bay_outline":selected,"bay_choices":[]}}


def _archive_display_matches(expected, outcome):
    if expected == outcome:
        return True
    if expected is None or outcome["provenance"].get("clearance_policy") is not None:
        return False
    # Early archives displayed the same verified polygons before partial surface labels existed.
    historical = deepcopy(expected)
    p = historical["provenance"]
    for outline in [*p.get("bay_choices",[]),p.get("bay_outline")]:
        if outline:
            outline.pop("extent",None)
    return historical == outcome


def verify(request, outcome, *, requested_box=None):
    source = outcome["provenance"]
    photo = Path(request["scene"]["frame"]["image_path"]).read_bytes()
    method = source["method"]
    policy = source.get("clearance_policy")
    if policy not in (None,CLEARANCE_POLICY):
        raise ValueError("Unknown parking clearance checks.")
    if method==CHOICE:
        original = source["choice_source"]
        if original.get("provenance",{}).get("method") not in {MARK,AUTO}:
            raise ValueError("Only an original set of pictured bays can be selected.")
        expected = choice_outcome(request,original,source["bay_id"])
    elif method==MARK:
        if requested_box is None:
            raise ValueError("The original passenger mark is required.")
        expected = mark_outcome(request,photo,source["surface_evidence"],requested_box,source.get("stall_review"),active_policy=bool(policy))
    elif method==AUTO and source.get("bay_instruction_checked") is True:
        expected=explicit_outcome(request,photo,source["surface_evidence"],source.get("stall_review"),source["bay_anchor_evidence"],active_policy=bool(policy))
    elif method==AUTO:
        expected = automatic_outcome(request,photo,source["surface_evidence"],source["landmark_evidence"],source.get("stall_review"),active_policy=bool(policy))
    else:
        raise ValueError("Unknown painted parking result.")
    matched = _archive_display_matches(expected, outcome)
    if not matched and policy is None and method == AUTO and not source.get("bay_instruction_checked"):
        expected = automatic_outcome(request,photo,source["surface_evidence"],source["landmark_evidence"],
                                     source.get("stall_review"),active_policy=False,prefer_complete=False)
        matched = _archive_display_matches(expected, outcome)
    if not matched:
        raise ValueError("Highlighted bay differs from the original photo, surface, words or passenger choice.")
    result = {"input_verified":True,"decision_reproduced":True,"offlane_mask_replayed":True,
              "current_image_and_tracks_checked":True,"language_calls":0,"current_language_calls":0,
              "independent_model_proposal":source["independent_model_proposal"],"bay_outline":source.get("bay_outline"),
              "bay_choices":source.get("bay_choices",[])}
    if method==AUTO and source.get("landmark_evidence") is not None:
        result["archived_landmark_language_calls"] = len(bay_landmarks.BANDS)
    if source.get("stall_review"):
        result["archived_stall_language_calls"] = sum(len(row["reviews"]) for row in source["stall_review"]["body"]["options"])
    if source.get("bay_anchor_evidence"):
        body=source["bay_anchor_evidence"]["body"]
        result["archived_bay_context_records"]=len(body["rows"])
    return result


def verify_current(request, outcome, *, requested_box=None):
    source = outcome["provenance"]
    if source["method"] == CHOICE:
        source = source["choice_source"]["provenance"]
    if source.get("clearance_policy") != CLEARANCE_POLICY:
        raise ValueError("Request a new check on the current photo before saving this location.")
    return verify(request, outcome, requested_box=requested_box)
