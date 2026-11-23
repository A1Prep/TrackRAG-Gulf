"""Attach verified high-resolution observations to measured stalls and current roadside gaps."""
from hashlib import sha256
from collections import OrderedDict
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from threading import Lock

from PIL import Image

from . import bay_anchors, bay_clearance, curb_points, curb_recess, gravel_pullin, roadside_board, whole_bay
from .bay_packets import read_photo_packet
from .parking_evidence import _authenticate
from .passenger import digest

BANK = Path(__file__).resolve().parents[2] / "experiments/dropoff/passenger-roadside-observations-v1.zip"
GRAVEL_BANK = BANK.with_name("passenger-gravel-observations-v1.zip")
_cache = OrderedDict()
_lock = Lock()


def candidates(request, photo, surface, review, *, gravel_only=False):
    clean, _, classes, legacy = whole_bay._layout(request, photo, surface, review)
    scene = clean["scene"]
    vehicles = bay_anchors.vehicles(scene)
    ctx = curb_points.source_context(scene)
    options = []
    if not gravel_only:
        road = bay_clearance.visible_road_component(classes)
        stalls = whole_bay._with_rows(clean, photo, surface, legacy, whole_bay.ROW_POLICY)
        options.extend(("stall", stall) for stall in stalls if bay_clearance.vehicle_sized(stall, scene)
                       and bay_clearance.on_visible_road(stall, road, scene["frame"]))
    detector = gravel_pullin if gravel_only else curb_recess
    with Image.open(BytesIO(photo)) as original:
        gaps = detector.candidates(original, classes, {"size":ctx["size"], "current_boxes":ctx["boxes"],
                                                      "current_vehicles":vehicles})
    for gap in gaps:
        option={"point":gap["point"],"box":gap["box"],"polygon":gap["support_patch"],
                "evidence":{**gap["evidence"],"coverage":"visible_portion","painted_boundaries":[],
                            "boundary_widths":[],"boundary_opposite_closed":[]}}
        options.append(("unpaved_pullin" if gravel_only else "recess", option))
    if not gravel_only:
        options.extend(("parking_point",o) for o in curb_points.locations(scene,classes))
    boxes={t:box for t,_,box in vehicles}
    pairs=[]
    for kind,option in options:
        spec={"kind":kind,"point":option["point"],"polygon":option["polygon"],
              "vehicles":[[t,boxes[t]] for t in option["evidence"].get("flanking_track_ids",[])]}
        pairs.append((spec,option))
    return pairs


def get_packet(key, *, gravel_only=False):
    return read_photo_packet(GRAVEL_BANK if gravel_only else BANK, key, 90000)


def observations(scene,photo,pairs,packet):
    _authenticate(packet)
    body=packet["body"]
    expected=(roadside_board.RECIPE,roadside_board.REPO,roadside_board.REVISION,digest(roadside_board.GUIDE),
              scene["provenance"]["image_sha256"],digest(curb_points.source_context(scene)))
    if tuple(body.get(k) for k in ("recipe","model","revision","prompt","image_sha256","context"))!=expected:
        raise ValueError("Roadside observations differ from this original, model or tracked objects.")
    rows=body["rows"]
    if len(rows)!=len(pairs):
        raise ValueError("Roadside observations differ from current measured locations.")
    visible=[]
    with Image.open(BytesIO(photo)) as original:
        if sha256(photo).hexdigest() != body["image_sha256"] or list(original.size) != curb_points.source_context(scene)["size"]:
            raise ValueError("Roadside observations require the original photo and image dimensions.")
        for (spec,opt),row in zip(pairs,rows,strict=True):
            if row["spec"]!=digest(spec) or sha256(roadside_board.render(original,spec)).hexdigest()!=row["view"]:
                raise ValueError("A pictured roadside location or vehicle changed.")
            desc=row.get("description") or {}
            if set(desc)!={"status","nearer_vehicle","farther_vehicle","frontage","roadside"} or any(
                    not isinstance(v,str) or len(v)>240 for v in desc.values()):
                raise ValueError("Invalid model observation.")
            if desc["status"]=="OPEN":
                visible.append({**opt,"kind":spec["kind"],"descriptions":desc,"vehicle_crops":spec["vehicles"]})
    return visible


def verified(request, photo, surface, review, packet, *, gravel_only=False):
    scene=request["scene"]
    key=scene["provenance"]["image_sha256"]
    if sha256(photo).hexdigest()!=key:
        raise ValueError("Roadside photograph changed.")
    stamp=digest([key,curb_points.source_context(scene),bay_anchors.vehicles(scene),surface,review,packet,gravel_only])
    with _lock:
        if stamp in _cache:
            _cache.move_to_end(stamp)
            return deepcopy(_cache[stamp])
    pairs=candidates(request,photo,surface,review,gravel_only=gravel_only)
    result=observations(scene,photo,pairs,packet)
    with _lock:
        _cache[stamp]=deepcopy(result)
        if len(_cache)>96: _cache.popitem(last=False)
    return result


def match(request, options, *, strict=True):
    from . import bay_intent,curb_point_language
    scene=request["scene"]; text=request["text"]; size=curb_points.source_context(scene)["size"]
    if strict and not bay_intent.permitted_destination(text):
        return [],False
    words=bay_intent.destination_words(text)
    stalls=[o for o in options if o["kind"]=="stall"]
    context={}
    for o in stalls:
        d=o["descriptions"]
        context[digest(o["polygon"])]={"frontage":d["frontage"]+" | "+d["roadside"],"descriptions":d,"pair":[]}
    # Ordinary landmark requests need the same noun/colour binding on stalls as on unmarked points.
    # The measured-bay parser additionally handles explicit orientation, markings and relative ranks.
    use_measured = not strict or bay_intent.PARKING.search(text)
    traced,possible_stalls = (bay_intent.match(words,stalls,scene,context,require_observed_landmark=True,
                                               strict_words=text if strict else None)
                              if use_measured else ([],[]))
    if not strict and traced:
        return bay_intent.one_of_matching_bays(text, whole_bay._prefer_measured(traced)),True
    if not curb_points.may_describe_point(text):
        if traced:
            return bay_intent.one_of_matching_bays(text, whole_bay._prefer_measured(traced)),True
        return possible_stalls,False
    full,partial = list(traced),[]
    kinds = ("stall","recess","unpaved_pullin","parking_point") if strict else ("recess","stall","parking_point")
    for kind in kinds:
        supported,possible=curb_point_language.match_points(text,[o for o in options if o["kind"]==kind],size,
                                                           native_scalars=strict)
        if strict:
            full.extend(supported)
            partial.extend(possible)
        else:
            if supported:
                return supported,True
            if not partial:
                partial=possible
    if full:
        return bay_intent.one_of_matching_bays(text,bay_intent.distinct_destinations(full)),True
    if strict:
        return bay_intent.distinct_destinations([*partial,*possible_stalls]),False
    return partial or possible_stalls,False
