"""Match passenger words and marks to visible parking stalls on the exact original frame."""
from collections import OrderedDict
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
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
VEHICLE_POLICY = "parking_clearance_with_current_vehicles_v2"
CONNECTED_POLICY = "parking_clearance_and_connected_pavement_v3"
LANDMARK_POLICY = "parking_clearance_and_connected_pavement_v4"
ROW_POLICY = "parking_rows_and_observed_landmarks_v5"
CLEARANCE_POLICY = "parking_outlines_and_observed_pullins_v6"
LANGUAGE_POLICY = "scoped_landmark_observations_v1"
INDEXED_DIALOGUE_POLICY = "scoped_conversational_landmarks_v2"
SPATIAL_DIALOGUE_POLICY = "scoped_directional_landmarks_v3"
_cache, _lock = OrderedDict(), Lock()
_layout_build = Lock()
_index_cache = OrderedDict()
_GALLERY_INDEX = Path(__file__).resolve().parents[2] / "experiments/dropoff/passenger-gallery-outlines-v1.json"


def _road_for_policy(classes, policy):
    from . import bay_clearance
    if policy in (CLEARANCE_POLICY, ROW_POLICY, LANDMARK_POLICY, CONNECTED_POLICY):
        return bay_clearance.visible_road_component(classes, rectangular_seed=policy == CONNECTED_POLICY)
    if policy in (None, VEHICLE_POLICY):
        return None
    raise ValueError("Unknown parking clearance checks.")


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


def _layout(request, photo, surface, stall_review=None, *, rows=False):
    clean, source, portable = source_input(request, photo)
    scene = clean["scene"]
    frame = scene["frame"]
    context = {"size":[frame["width"], frame["height"]], "image_sha256":source,
               "current_boxes":[t["bbox_xyxy"] for t in scene["tracks"] if t["frame_id"] == frame["frame_id"]]}
    classes = manual_surfaces.decode(surface, frame, source)
    key = digest([context,surface["signature"],stall_review,rows])
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return clean, portable, classes, deepcopy(_cache[key])
    # A cold image can briefly require several large optical arrays. Parallel
    # sessions can read cached results while a single new layout is being built.
    with _layout_build:
        with _lock:
            if key in _cache:
                _cache.move_to_end(key)
                return clean, portable, classes, deepcopy(_cache[key])
        with Image.open(BytesIO(photo)) as original:
            image = original.convert("RGB")
            if rows:
                from .parking_row_bays import candidates
                options = candidates(image, classes, context)
            else:
                options = visible_stalls(image, classes, context, surface, stall_review)
        with _lock:
            _cache[key] = deepcopy(options)
            if len(_cache) > 96:
                _cache.popitem(last=False)
    return clean, portable, classes, options


def _with_rows(request, photo, surface, options, policy):
    """Keep measured dividers over a clipped pavement component for the same space."""
    if policy not in (ROW_POLICY,CLEARANCE_POLICY):
        return options
    import cv2
    import numpy as np
    from . import surface_stalls
    _, _, _, extra = _layout(request, photo, surface, rows=True)
    if not extra:
        return options
    measured = [o for o in options if o["evidence"]["recipe"] != surface_stalls.RECIPE]
    fragments = [o for o in options if o["evidence"]["recipe"] == surface_stalls.RECIPE]
    accepted = list(measured)
    contours = [np.float32(o["polygon"]) for o in accepted]
    for option in [*extra, *fragments]:
        shape = np.float32(option["polygon"])
        if any(cv2.intersectConvexConvex(shape, other)[0] > .14 * min(cv2.contourArea(shape), cv2.contourArea(other))
               for other in contours):
            continue
        accepted.append(option)
        contours.append(shape)
    return sorted(accepted, key=lambda opt: tuple(opt["point"]))


@lru_cache(maxsize=1)
def _published_gallery():
    try:
        entries = json.loads(_GALLERY_INDEX.read_bytes())
        if entries["policy"] == CLEARANCE_POLICY and isinstance(entries["outlines"], dict):
            return entries["outlines"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {}


def gallery_outline_entry(scene, photo, indexed_shapes, *, measure=False):
    """Look up a navigation hint tied to the exact inputs or measure it on demand."""
    from . import bay_clearance, surface_stalls
    key = scene["provenance"]["image_sha256"]
    surface = manual_surfaces.get_packet(key)
    if surface is None:
        return None, False
    review = surface_stalls.get_packet(key)
    frame_id = scene["frame"]["frame_id"]
    current = [t for t in scene["tracks"] if t["frame_id"] == frame_id]
    stamp = digest([CLEARANCE_POLICY, key, current, sorted(indexed_shapes) if indexed_shapes is not None else None,
                    surface, review])
    if not measure:
        published = _published_gallery().get(stamp)
        if type(published) is bool:
            return stamp, published
    with _lock:
        if stamp in _index_cache:
            _index_cache.move_to_end(stamp)
            return stamp, _index_cache[stamp]
    request = {"scene":scene, "text":"Find the marked parking bays.", "input_mode":"typed", "dialogue":[], "request_id":"catalog"}
    _, _, classes, options = _layout(request, photo, surface, review)
    road = bay_clearance.visible_road_component(classes)
    valid_packet = indexed_shapes is None or {digest(o["polygon"]) for o in options} == indexed_shapes
    options = _with_rows(request, photo, surface, options, CLEARANCE_POLICY)
    matches = valid_packet and any(
        bay_clearance.vehicle_sized(o, scene) and bay_clearance.on_visible_road(o,road,scene["frame"]) for o in options)
    with _lock:
        _index_cache[stamp] = matches
        if len(_index_cache) > 96:
            _index_cache.popitem(last=False)
    return stamp, matches


def index_has_outline(scene, photo, indexed_shapes):
    """Only the gallery uses this flag. Each proposed or marked destination is checked on the original."""
    return gallery_outline_entry(scene, photo, indexed_shapes)[1]


def _present(options, tap):
    from . import surface_stalls
    choices = []
    for index, option in enumerate(options, 1):
        item = {"id": f"bay-{index}", "polygon_xy": option["polygon"], "center_xy": option["point"],
                "passenger_tap_xy": tap}
        # A visually reviewed pavement component can end at the edge of the segmentation.
        # Only the painted-divider and kerb-end detectors reconstruct the entire measured stall.
        if option["evidence"]["recipe"] == surface_stalls.RECIPE or option["evidence"].get("coverage") == "visible_portion":
            item["extent"] = "visible_parking_surface"
        elif option["evidence"].get("coverage") == "interior_point_only":
            item["extent"] = "interior_point_only"
        description=option.get("descriptions") or {}
        observed=[]
        for key in ("nearer_vehicle","farther_vehicle","frontage","roadside"):
            label=str(description.get(key,"")).strip().strip(".")
            if label and len(label)<=100 and not _UNCERTAIN_CAPTION.search(label) and label.casefold() not in {v.casefold() for v in observed}:
                observed.append(label.lower() if label.isupper() else label)
        if observed:
            item["observed_landmarks"]=observed
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


def mark_outcome(request, photo, surface, box, stall_review=None, *, policy=CLEARANCE_POLICY, gravel_packet=None):
    from . import bay_clearance, painted_bays
    clean, portable, classes, options = _layout(request, photo, surface, stall_review)
    options = _with_rows(clean, photo, surface, options, policy)
    clean, _, mark, center = manual_offlane.mark_input(clean,box)
    frame = clean["scene"]["frame"]
    reason, checked = manual_offlane.surface_check(frame,clean["scene"]["tracks"],mark,center,classes)
    if reason in {"observed_object","broad_area"}:
        return None
    evidence = {"stall_review": stall_review} if stall_review else {}
    if policy:
        evidence["clearance_policy"] = policy
        road = _road_for_policy(classes, policy)
        if road is not None and checked.get("center_class_id") == manual_offlane.PARKING:
            ix, iy = checked["center_cell_xy"]
            if not road[iy,ix]:
                return _make(clean,portable,[],surface,method=MARK,reason="separated_parking_surface",
                             question="I can’t see a connection from this road to the parking area you marked. Could you mark a spot on this side?",
                             tap=center,selected_box=mark,**evidence)
        sized = [o for o in options if bay_clearance.vehicle_sized(o, clean["scene"])]
        usable = [o for o in sized if bay_clearance.on_visible_road(o,road,frame)]
        rejected = [o for o in options if o not in usable]
        rejected_at_mark = painted_bays.matching_options(rejected,mark,center,frame,classes,offer_visible_side=True)
        if rejected_at_mark and not painted_bays.matching_options(usable,mark,center,frame,classes,offer_visible_side=True):
            separated = any(o in sized for o in rejected_at_mark)
            why = "separated_parking_surface" if separated else "parking_fragment_at_vehicle"
            message = ("I can’t see a connection from this road to the area you marked. Could you mark a spot on this side?" if separated else
                       "Only a narrow strip of pavement is visible beside the parked vehicle there. Could you mark a different open space?")
            return _make(clean,portable,[],surface,method=MARK,reason=why,question=message,
                         tap=center,selected_box=mark,**evidence)
        options = usable
    matches = painted_bays.matching_options(options,mark,center,frame,classes)
    edge_offer = len(matches) == 1 and checked.get("center_class_id") == painted_bays.LINE
    if not matches:
        matches = painted_bays.matching_options(options,mark,center,frame,classes,offer_visible_side=True)
        edge_offer = len(matches) == 1
    if not matches:
        if gravel_packet is not None and policy == CLEARANCE_POLICY and checked.get("center_class_id") == 26:
            from . import roadside_context
            options = roadside_context.verified(clean, photo, surface, stall_review, gravel_packet, gravel_only=True)
            # The signed interior patch is much smaller than the visible gravel. Every
            # corner of the passenger's own mark must fit within the same supported patch.
            for option in options:
                x0, y0, x1, y1 = option["box"]
                if x0 <= mark[0] and y0 <= mark[1] and mark[2] <= x1 and mark[3] <= y1:
                    return _make(clean,portable,[option],surface,method=MARK,reason="passenger_mark_in_observed_pullin",
                                 question="",tap=center,selected_box=mark,gravel_evidence=gravel_packet,**evidence)
        return None
    reason = "single_visible_bay_at_painted_edge" if edge_offer else "passenger_mark_on_painted_bay"
    question = ("I can outline one bay beside this painted line. Did you mean this side, or would you like to mark another?"
                if edge_offer else "Your mark is on a boundary shared by the highlighted bays. Which bay did you mean?")
    return _make(clean,portable,matches,surface,method=MARK,reason=reason,require_choice=edge_offer,question=question,
                 tap=center,selected_box=mark,**evidence)


def select_mark(request, box):
    from . import roadside_context, surface_stalls
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
            if bay is None or bay["provenance"].get("reason")=="parking_fragment_at_vehicle":
                from .observed_pullin import marked_outcome
                paved = marked_outcome(clean,data,packet,mark)
                if paved is not None:
                    return paved
            if bay is not None:
                return bay
            gravel = roadside_context.get_packet(original_sha, gravel_only=True)
            if gravel is not None:
                bay = mark_outcome(clean,data,packet,mark,stall_review,gravel_packet=gravel)
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


def language_candidates(request, options, labels, *, strict_words=False):
    words = place_grounding.place_reference(request)
    if not words or _UNSUPPORTED.search(words):
        return []
    terms = set(_terms(words))
    if not terms and not _GENERIC.search(words):
        return []
    if strict_words:
        from .curb_point_language import observed_landmark
        bands = [i for i,label in enumerate(labels or []) if terms and
                 observed_landmark(terms, {"frontage":label}, ("frontage",)) is True]
    else:
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


def automatic_outcome(request, photo, surface, landmarks, stall_review=None, *, policy=CLEARANCE_POLICY,
                      prefer_complete=True, strict_words=False):
    from . import bay_clearance
    clean, portable, classes, options = _layout(request,photo,surface,stall_review)
    options = _with_rows(clean, photo, surface, options, policy)
    if not options:
        return None
    f = clean["scene"]["frame"]
    words = place_grounding.place_reference(clean)
    if not words:
        return None
    labels = bay_landmarks.verify(landmarks,photo,(f["width"],f["height"])) if landmarks is not None else None
    matches = language_candidates(clean,options,labels,strict_words=strict_words)
    evidence = {"stall_review":stall_review} if stall_review else {}
    if strict_words:
        evidence["language_policy"] = LANGUAGE_POLICY
    if policy:
        evidence["clearance_policy"] = policy
        road = _road_for_policy(classes, policy)
        clear = [o for o in options if bay_clearance.vehicle_sized(o,clean["scene"])]
        same_side = [o for o in clear if bay_clearance.on_visible_road(o,road,f)]
        if (clear and not same_side) or (any(o in clear for o in matches) and not any(o in same_side for o in matches)):
            return _make(clean,portable,[],surface,method=AUTO,reason="separated_parking_surface",
                         question="I can’t trace a connection from this road to that parking area. Is there another visible spot on this side?",
                         landmark_evidence=landmarks,**evidence)
        clear = same_side
        if not clear or (matches and not any(o in clear for o in matches)):
            return _make(clean,portable,[],surface,method=AUTO,reason="parking_fragments_at_vehicles",
                         question=("I couldn’t verify an open parking area beside the place you described. Could you mark a visible space?"
                                   if strict_words else
                                   "I can see parking pavement near the parked vehicles, but I can’t outline an empty bay there. Can you see another open space?"),
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
    from . import bay_intent, surface_stalls
    words = place_grounding.place_reference(request)
    if not words or _UNSUPPORTED.search(words) or not bay_intent.permitted_destination(request["text"]):
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
            return automatic_outcome(request,photo,surface,labels,stall_review,strict_words=True)
        except (KeyError,TypeError,ValueError):
            if stall_review is None:
                raise
            return automatic_outcome(request,photo,surface,labels,strict_words=True)
    except (OSError, TypeError, ValueError, KeyError):
        # Missing or invalid supplementary evidence leaves the trained place route available.
        return None


def explicit_outcome(request, photo, surface, stall_review, anchors, *, row_anchors=None,
                     policy=CLEARANCE_POLICY, strict_words=False):
    """Check a request whose destination is a bay, retaining its vehicle and frontage qualifiers."""
    from . import bay_anchors, bay_clearance, bay_intent
    words = bay_intent.destination_words(request["text"]) if policy in (ROW_POLICY,CLEARANCE_POLICY) else request["text"]
    if bay_intent.parse(words) is None:
        return None
    clean,portable,classes,options=_layout(request,photo,surface,stall_review)
    with Image.open(BytesIO(photo)) as im:
        observations=bay_anchors.observations(im,options,clean["scene"],anchors)
        if row_anchors is not None:
            if policy not in (ROW_POLICY,CLEARANCE_POLICY):
                raise ValueError("Parking row observations require current parking geometry.")
            _, _, _, rows = _layout(clean,photo,surface,rows=True)
            observations.update(bay_anchors.observations(im,rows,clean["scene"],row_anchors))
    options = _with_rows(clean, photo, surface, options, policy)
    original_options = options
    separated = False
    evidence = {"stall_review":stall_review} if stall_review else {}
    if strict_words:
        evidence["language_policy"] = LANGUAGE_POLICY
    if row_anchors is not None:
        evidence["row_anchor_evidence"] = row_anchors
    if policy:
        evidence["clearance_policy"] = policy
        options = [o for o in options if bay_clearance.vehicle_sized(o, clean["scene"])]
        if options:
            road = _road_for_policy(classes, policy)
            options = [o for o in options if bay_clearance.on_visible_road(o,road,clean["scene"]["frame"])]
            separated = not options
    supported,possible=bay_intent.match(words,options,clean["scene"],observations,
                                        require_observed_landmark=policy in (LANDMARK_POLICY, ROW_POLICY, CLEARANCE_POLICY),
                                        **({"strict_words":clean["text"]} if strict_words else {}))
    if words != clean["text"] and not supported:
        # A vehicle or frontage may still be resolved by the dedicated visual model.
        return None
    unresolved_reference=not supported
    shown=_prefer_measured(supported or possible)
    if supported and policy in (ROW_POLICY,CLEARANCE_POLICY):
        shown = bay_intent.one_of_matching_bays(clean["text"], shown)
    if supported:
        reason="visible_bay_matches_stated_anchors"
        ask="I found several outlined bays that fit those details. Which bay would you like?"
    elif possible:
        reason="unverified_bay_reference"
        ask=("I can outline this bay, but I couldn’t match every detail you gave. Is this the bay you meant?"
             if len(possible)==1 else
             "I can outline these bays, but I couldn’t match every detail you gave. Which one did you mean?")
    elif separated:
        reason="separated_parking_surface"
        ask="I can see a parking area, but I can’t trace a connection to it from this road. Is there a visible spot on this side?"
    elif original_options and not options:
        reason="parking_fragments_at_vehicles"
        ask=("I couldn’t verify an open parking area beside the place you described. Could you mark a visible space?"
             if strict_words else
             "I can see strips of parking pavement beside the parked vehicles, but I can’t outline an empty bay there. Can you see another open space?")
    else:
        reason="no_supported_bay_for_instruction" if options else "no_visible_bay_for_instruction"
        ask=("I couldn’t match an outlined bay to those details. Could you tap inside the bay you mean?"
             if options else
             "I couldn’t trace an empty bay in this view. Could you tap inside the space you mean or describe another landmark?")
    return _make(clean,portable,shown,surface,method=AUTO,reason=reason,question=ask,require_choice=unresolved_reference,
                 bay_instruction_checked=True,bay_anchor_evidence=anchors,
                 **evidence)


def _unmatched_index(clean, portable, surface, options, evidence, *, expanded):
    """The inspected candidates support neither a matching location nor a selectable alternative."""
    reason = "indexed_reference_unmatched" if options else "indexed_no_open_location"
    words = ("I couldn’t match all those details to an open area in this photo. Could you mark the spot or "
             "describe a different landmark?" if options else
             "I couldn’t verify an open pull-in for this request in the photo. Could you mark another area "
             "or describe a different landmark?")
    return _make(clean,portable,[],surface,method=AUTO,reason=reason,question=words,
                 language_policy=SPATIAL_DIALOGUE_POLICY if expanded else INDEXED_DIALOGUE_POLICY,
                 clearance_policy=CLEARANCE_POLICY,**evidence)


def point_outcome(request, photo, surface, packet, *, conversational=False, expanded=False):
    from . import curb_points, curb_point_language
    clean, _, portable = source_input(request,photo)
    if not curb_points.may_describe_point(clean["text"]):
        return None
    options=curb_points.verified_options(clean["scene"],photo,surface,packet)
    supported, possible = curb_point_language.match_points(clean["text"],options,curb_points.source_context(clean["scene"])["size"],
                                                          conversational=conversational,expanded=expanded)
    if not (supported or possible):
        if conversational and curb_point_language.indexed_reference(clean["text"],expanded=expanded):
            return _unmatched_index(clean,portable,surface,options,{"curb_point_evidence":packet},expanded=expanded)
        return None
    shown=supported or possible
    question=("I can see more than one area matching those details. Which one did you mean?" if supported else
              "I can show the visible parking area, but I couldn’t verify every landmark you mentioned. Is this where you meant?")
    return _make(clean,portable,shown,surface,method=AUTO,
                 reason="observed_open_pullin" if supported else "unverified_bay_reference",question=question,
                 require_choice=not supported,curb_point_evidence=packet,clearance_policy=CLEARANCE_POLICY,
                 **({"language_policy":SPATIAL_DIALOGUE_POLICY if expanded else INDEXED_DIALOGUE_POLICY} if conversational else {}))


def roadside_outcome(request, photo, surface, review, packet, *, gravel_packet=None, strict=True, conversational=False,
                    expanded=False):
    from . import curb_point_language,roadside_context
    clean,_,portable=source_input(request,photo)
    options = roadside_context.verified(clean,photo,surface,review,packet) if packet is not None else []
    evidence = {"roadside_evidence":packet} if packet is not None else {}
    if gravel_packet is not None:
        options.extend(roadside_context.verified(clean,photo,surface,review,gravel_packet,gravel_only=True))
        evidence["gravel_evidence"] = gravel_packet
    if review is not None:
        evidence["stall_review"]=review
    shown,supported=roadside_context.match(clean,options,strict=strict,conversational=conversational,expanded=expanded)
    if not shown:
        if conversational and (packet is not None or gravel_packet is not None) and curb_point_language.indexed_reference(clean["text"],expanded=expanded):
            return _unmatched_index(clean,portable,surface,options,evidence,expanded=expanded)
        return None
    if strict:
        evidence["language_policy"] = SPATIAL_DIALOGUE_POLICY if expanded else INDEXED_DIALOGUE_POLICY if conversational else LANGUAGE_POLICY
    question=("Which of these visible areas did you mean?" if supported else
              "I can show an open area near some of your landmarks. Could you check the highlight and confirm which place you meant?")
    return _make(clean,portable,shown,surface,method=AUTO,
                 reason="matched_roadside_observations" if supported else "unverified_bay_reference",
                 require_choice=not supported, question=question,**evidence,clearance_policy=CLEARANCE_POLICY)


def _clarify_destination(request, photo):
    from . import bay_intent
    clean, _, portable = source_input(request, photo)
    if bay_intent.permitted_destination(clean["text"]):
        return None
    return _make(clean,portable,[],None,method=AUTO,reason="destination_needs_clarification",
                 question="Could you describe a visible off-lane spot or tap the space you mean on the photo?",
                 destination_scope_checked=True,language_policy=LANGUAGE_POLICY,clearance_policy=CLEARANCE_POLICY)


def _has_location(outcome):
    if outcome is None:
        return False
    source=outcome["provenance"]
    return bool(source.get("bay_outline") or source.get("bay_choices") or outcome.get("result",{}).get("status")=="proposed")


def _context_outcome(request, photo, surface, review, roadside, gravel, points, *, indexed):
    answer = None
    if roadside is not None or gravel is not None:
        try:
            answer=roadside_outcome(request,photo,surface,review,roadside,gravel_packet=gravel,conversational=indexed,expanded=indexed)
        except (ValueError,KeyError,TypeError):
            pass
    # The full roadside board already reviewed all marker points. A rejected point there cannot
    # be silently reinstated by an older reading of the same location.
    if _has_location(answer) or (indexed and answer is not None and roadside is not None):
        return answer
    if points is not None:
        try:
            marker=point_outcome(request,photo,surface,points,conversational=indexed,expanded=indexed)
            if _has_location(marker) or answer is None:
                return marker
        except (ValueError,KeyError,TypeError):
            pass
    return answer


def _native_pair_fallback(request):
    """Preserve the trained vehicle-pair route when the complete request fits its own evidence."""
    from . import between_road,curb_point_language,curb_utterance
    if not between_road.requested(request):
        return False
    words=curb_point_language._normal(curb_utterance.destination_clause(request["text"]))
    required=curb_point_language._requirements(words)
    if required is None or len(required[1])!=2 or required[2]:
        return False
    frame=request["scene"]["frame"]
    return sum(t["frame_id"]==frame["frame_id"] and t["class_name"] in between_road.MOTOR
               for t in request["scene"]["tracks"])>=2


def _empty_geometry(request, photo, surface, review, *, expanded=False):
    """An authenticated source with no eligible pictured candidate can answer immediately."""
    from . import curb_point_language,roadside_context
    clean,_,portable=source_input(request,photo)
    if not curb_point_language.indexed_reference(clean["text"],expanded=expanded):
        return None
    if any(roadside_context.candidates(clean,photo,surface,review,gravel_only=gravel) for gravel in (False,True)):
        return None
    evidence={"index_geometry_checked":True}
    if review is not None:
        evidence["stall_review"]=review
    return _unmatched_index(clean,portable,surface,[],evidence,expanded=expanded)


def explicit_request(request):
    from . import bay_anchors, bay_clearance, bay_intent, curb_point_language, curb_points, roadside_context, surface_stalls
    if not bay_intent.permitted_destination(request["text"]):
        photo = Path(request["scene"]["frame"]["image_path"]).read_bytes()
        return _clarify_destination(request, photo)
    intent = bay_intent.parse(bay_intent.destination_words(request["text"]))
    key = request["scene"]["provenance"]["image_sha256"]
    points = curb_points.get_packet(key) if curb_points.may_describe_point(request["text"]) else None
    roadside=roadside_context.get_packet(key)
    gravel=roadside_context.get_packet(key, gravel_only=True)
    indexed=curb_point_language.indexed_reference(request["text"],expanded=True)
    if not indexed and intent is None and points is None and roadside is None and gravel is None:
        return None
    surface = manual_surfaces.get_packet(key)
    if surface is None:
        return None
    photo = Path(request["scene"]["frame"]["image_path"]).read_bytes()
    review = surface_stalls.get_packet(key)
    def from_context():
        return _context_outcome(request,photo,surface,review,roadside,gravel,points,indexed=indexed)

    preserve_pair = indexed and _native_pair_fallback(request)
    early = from_context() if indexed else None
    if _has_location(early) and not (preserve_pair and early["provenance"]["reason"]=="unverified_bay_reference"):
        return early
    if indexed and early is None and not preserve_pair:
        try:
            early = _empty_geometry(request,photo,surface,review,expanded=True)
        except (ValueError,KeyError,TypeError):
            pass
    if indexed and curb_point_language.references_individual_sides(request["text"]):
        # A broad stall caption cannot establish the individual flanking crop directions.
        return early
    if early is not None and not preserve_pair:
        # Existing measured bays can have independent signed vehicle crops. Preserve their chance
        # to identify a destination even when the general roadside caption could not match it.
        clean,_,classes,old = _layout(request,photo,surface,review)
        old = _with_rows(clean,photo,surface,old,CLEARANCE_POLICY)
        road = _road_for_policy(classes,CLEARANCE_POLICY)
        if intent is None or not any(bay_clearance.vehicle_sized(o,clean["scene"]) and
                                     bay_clearance.on_visible_road(o,road,clean["scene"]["frame"]) for o in old):
            return early
    if intent is None:
        return (None if preserve_pair else early) if indexed else from_context()
    anchors = bay_anchors.get_packet(key)
    row_anchors = bay_anchors.get_row_packet(key)
    # The signed region review and the signed language observations are independent. A stale
    # observation must not remove a geometrically valid option supported by a good region review.
    packets = [(review, anchors)]
    if anchors is not None:
        packets.append((review, None))
    if review is not None:
        packets.append((None, None))
    attempts = [(r,c,row_anchors) for r,c in packets] if row_anchors else []
    attempts.extend((r,c,None) for r,c in packets)
    for region_packet, context_packet, row_packet in attempts:
        try:
            kwargs = {"row_anchors":row_packet} if row_packet is not None else {}
            result = explicit_outcome(request, photo, surface, region_packet, context_packet, strict_words=True, **kwargs)
        except (ValueError, KeyError, TypeError):
            continue
        if (points or roadside or gravel) and (result is None or not result["provenance"].get("bay_outline")):
            alternate=early if indexed else from_context()
            if _has_location(alternate):
                if (result is not None and result["provenance"].get("bay_choices")
                        and alternate["provenance"]["reason"]=="unverified_bay_reference"):
                    return result
                if preserve_pair and not _has_location(result) and alternate["provenance"]["reason"]=="unverified_bay_reference":
                    return None
                return alternate
        if result is not None:
            if preserve_pair and not _has_location(result):
                return None
            if early is not None and not preserve_pair and not _has_location(result):
                return early
            return result
        return None if preserve_pair else early
    return None if preserve_pair else early


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
    language = source.get("language_policy")
    conversational = language in (INDEXED_DIALOGUE_POLICY,SPATIAL_DIALOGUE_POLICY)
    expanded = language==SPATIAL_DIALOGUE_POLICY
    if language not in (None,LANGUAGE_POLICY,INDEXED_DIALOGUE_POLICY,SPATIAL_DIALOGUE_POLICY):
        raise ValueError("Unknown passenger language checks.")
    if policy not in (None,VEHICLE_POLICY,CONNECTED_POLICY,LANDMARK_POLICY,ROW_POLICY,CLEARANCE_POLICY):
        raise ValueError("Unknown parking clearance checks.")
    if method==CHOICE:
        original = source["choice_source"]
        if original.get("provenance",{}).get("method") not in {MARK,AUTO}:
            raise ValueError("Only an original set of pictured bays can be selected.")
        expected = choice_outcome(request,original,source["bay_id"])
    elif source.get("pullin_policy") is not None:
        from . import observed_pullin
        if source["pullin_policy"]!=observed_pullin.POLICY or policy!=CLEARANCE_POLICY:
            raise ValueError("Unknown paved pull-in evidence.")
        if method==MARK and requested_box is not None:
            expected=observed_pullin.marked_outcome(request,photo,source["surface_evidence"],requested_box)
        elif method==AUTO and source.get("visual_space_evidence") and language==INDEXED_DIALOGUE_POLICY:
            expected=observed_pullin.visual_outcome(request,photo,source["surface_evidence"],source["visual_space_evidence"])
        else:
            raise ValueError("A paved pull-in requires its original mark or signed visual observations.")
    elif method==MARK:
        if requested_box is None:
            raise ValueError("The original passenger mark is required.")
        expected = mark_outcome(request,photo,source["surface_evidence"],requested_box,source.get("stall_review"),
                                policy=policy,gravel_packet=source.get("gravel_evidence"))
    elif method==AUTO and source.get("destination_scope_checked") is True and policy == CLEARANCE_POLICY:
        expected = _clarify_destination(request, photo)
    elif (method==AUTO and source.get("index_geometry_checked") is True
          and conversational and policy==CLEARANCE_POLICY):
        expected = _empty_geometry(request,photo,source["surface_evidence"],source.get("stall_review"),expanded=expanded)
    elif method==AUTO and source.get("curb_point_evidence") is not None and policy==CLEARANCE_POLICY:
        expected=point_outcome(request,photo,source["surface_evidence"],source["curb_point_evidence"],
                               conversational=conversational,expanded=expanded)
    elif method==AUTO and (source.get("roadside_evidence") or source.get("gravel_evidence")) and policy==CLEARANCE_POLICY:
        expected=roadside_outcome(request,photo,source["surface_evidence"],source.get("stall_review"),source.get("roadside_evidence"),
                                  gravel_packet=source.get("gravel_evidence"),strict=language==LANGUAGE_POLICY or conversational,
                                  conversational=conversational,expanded=expanded)
    elif method==AUTO and source.get("bay_instruction_checked") is True:
        expected=explicit_outcome(request,photo,source["surface_evidence"],source.get("stall_review"),source["bay_anchor_evidence"],
                                  row_anchors=source.get("row_anchor_evidence"),policy=policy,
                                  strict_words=source.get("language_policy")==LANGUAGE_POLICY)
    elif method==AUTO:
        expected = automatic_outcome(request,photo,source["surface_evidence"],source["landmark_evidence"],source.get("stall_review"),
                                     policy=policy,strict_words=source.get("language_policy")==LANGUAGE_POLICY)
    else:
        raise ValueError("Unknown painted parking result.")
    matched = _archive_display_matches(expected, outcome)
    if not matched and policy is None and method == AUTO and not source.get("bay_instruction_checked"):
        expected = automatic_outcome(request,photo,source["surface_evidence"],source["landmark_evidence"],
                                     source.get("stall_review"),policy=None,prefer_complete=False)
        matched = _archive_display_matches(expected, outcome)
    if not matched:
        raise ValueError("Highlighted bay differs from the original photo, surface, words or passenger choice.")
    original = source["choice_source"]["provenance"] if method == CHOICE else source
    result = {"input_verified":True,"decision_reproduced":True,"offlane_mask_replayed":original.get("surface_evidence") is not None,
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
    if source.get("row_anchor_evidence"):
        result["archived_row_context_records"] = len(source["row_anchor_evidence"]["body"]["rows"])
    if source.get("curb_point_evidence"):
        result["archived_curb_point_records"] = len(source["curb_point_evidence"]["body"]["rows"])
    if source.get("roadside_evidence"):
        result["archived_roadside_context_records"] = len(source["roadside_evidence"]["body"]["rows"])
    if source.get("gravel_evidence"):
        result["archived_gravel_context_records"] = len(source["gravel_evidence"]["body"]["rows"])
    if original.get("pullin_policy"):
        result["native_paved_entrance_checked"] = True
        if original.get("visual_space_evidence"):
            result["archived_ground_inspections"] = sum(len(side["reviewed"]) for side in original["visual_space_evidence"]["body"]["sides"].values())
    return result


def verify_current(request, outcome, *, requested_box=None):
    from . import curb_point_language
    source = outcome["provenance"]
    if source["method"] == CHOICE:
        source = source["choice_source"]["provenance"]
    if source.get("clearance_policy") != CLEARANCE_POLICY:
        raise ValueError("Request a new check on the current photo before saving this location.")
    if (source["method"]==AUTO and any(source.get(key) for key in ("roadside_evidence","curb_point_evidence","gravel_evidence"))
            and curb_point_language.indexed_reference(request["text"],expanded=True)
            and source.get("language_policy")!=SPATIAL_DIALOGUE_POLICY):
        raise ValueError("Recheck this passenger instruction with the current pictured-location policy.")
    return verify(request, outcome, requested_box=requested_box)
