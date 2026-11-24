"""Match a nearby drop-off to parking that touches the identified vehicle's own row."""
from io import BytesIO
import re

import cv2
import numpy as np
from PIL import Image

from . import bay_clearance, bay_intent, manual_offlane, painted_bays, parking_row_bays, paver_pullin, single_vehicle, visual_space


ORIGINAL_POLICY = "near_vehicle_connected_parking_v1"
POLICY = "near_vehicle_connected_parking_v2"
REASON = "parking_near_identified_vehicle"
_RELATION = re.compile(r"\b(?:near(?:\s+to)?|close\s+to|by)\b", re.I)
_PASSENGER = re.compile(
    r"(?:(?:(?:do|can|could|would|will) you|(?:can|could|may) (?:i|we)|please|kindly|just|i(?: would|'d) like (?:you )?to)\s+)*"
    r"(?:drop (?:me|us)(?: off)?|drop off (?:me|us)|let (?:me|us) (?:out|off)|"
    r"(?:get|hop|step) (?:out|off)|leave (?:me|us)|stop|pull (?:in|over|up)|park)(?: (?:here|there|please|right|somewhere))?",
    re.I,
)


def reference(words):
    """A complete, affirmative location instruction referring to one pictured vehicle."""
    if not isinstance(words,str) or not bay_intent.permitted_destination(words):
        return None
    words = " ".join(words.casefold().replace("’","'").split())
    relations = list(_RELATION.finditer(words))
    if len(relations) != 1:
        return None
    relation = relations[0]
    head = words[:relation.start()].strip(" ,")
    tail = words[relation.end():].strip(" ,.?!")
    if head and not _PASSENGER.fullmatch(head):
        return None
    if not re.fullmatch(r"[a-z -]+",tail):
        return None
    # Reuse the vehicle colour/body/side parser, but keep the original 'near' relation.
    parsed = single_vehicle.request_reference({"text":"beside "+tail})
    if not parsed or parsed["unsupported"]:
        return None
    return {**parsed,"relation":"near"}


def uncertain_matches(specs, answers, ref, width):
    """Keep a second inspected vehicle in the question when an unreadable feature could match."""
    possible = []
    for spec, reading in zip(specs, answers, strict=True):
        colour, body = reading[:2]
        missing_colour = ref["color"] is not None and colour == "unknown"
        if not missing_colour and body != "unknown":
            continue
        if ref["color"] and colour not in {ref["color"], "unknown"}:
            continue
        body_matches = (single_vehicle._coarse(ref["kind"], spec["class"]) if body == "unknown"
                        else single_vehicle._compatible(ref["kind"], body))
        if body_matches and single_vehicle._side(spec["box"], width, ref["side"]):
            possible.append((spec, reading))
    return possible


def conditional_vehicle(specs, answers, ref, width):
    """One stated difference can be offered for confirmation, never counted as a full match."""
    if not ref["color"]:
        return None
    candidates = []
    for spec, row in zip(specs, answers, strict=True):
        colour, body = row[:2]
        if body == "unknown" or not single_vehicle._compatible(ref["kind"], body):
            continue
        if colour == ref["color"]:
            if not ref["side"] or single_vehicle._side(spec["box"], width, ref["side"]):
                return None
            left, _, right, _ = spec["box"]
            centre = (left+right)/2
            requested_half = centre > width/2 if ref["side"] == "right" else centre < width/2
            if left < width/2 < right and requested_half:
                candidates.append((spec, row, f"The {colour} {body} crosses the image centre. "
                                   f"I couldn’t verify your description of the {ref['side']}-hand side."))
        elif {colour, ref["color"]} == {"silver", "gray"} and single_vehicle._side(spec["box"], width, ref["side"]):
            wanted = "grey" if ref["color"] == "gray" else ref["color"]
            measured = "grey" if colour == "gray" else colour
            candidates.append((spec, row, f"You said {wanted}. The photo check reads this {body.upper() if body=='suv' else body} as {measured}."))
    return candidates[0] if len(candidates)==1 else None


def _open_point(option, scene, classes, road):
    point = option["point"]
    frame = scene["frame"]
    gx,gy = np.array(point)*[classes.shape[1]/frame["width"],classes.shape[0]/frame["height"]]
    x,y = int(gx),int(gy)
    if not (0<=x<classes.shape[1] and 0<=y<classes.shape[0]) or road is None or not road[y,x]:
        return False
    tap = [point[0]-1,point[1]-1,point[0]+1,point[1]+1]
    reason,_ = manual_offlane.surface_check(frame,scene["tracks"],tap,point,classes)
    return reason == "parking"


def _adjacent_to_divider(option, anchor, scene, classes):
    """Require an observed tyre-to-paint contact and unobstructed parking on the other side."""
    evidence = option["evidence"]
    if evidence.get("recipe") == parking_row_bays.RECIPE:
        boundaries = evidence.get("optical_dividers") or evidence.get("stud_boundaries")
    elif evidence.get("recipe") in (painted_bays.RECIPE, painted_bays.ROW_RECIPE):
        boundaries = evidence.get("painted_boundaries")
    else:
        return False
    lines = np.asarray(boundaries, dtype=float)
    if lines.shape != (2,2,2) or not np.isfinite(lines).all():
        return False
    if not bay_clearance.vehicle_sized(option,scene):
        return False
    left,top,right,bottom = anchor
    point = np.asarray(option["point"])
    scale = np.array([classes.shape[1]/scene["frame"]["width"],classes.shape[0]/scene["frame"]["height"]])
    other_vehicles = [t["bbox_xyxy"] for t in scene["tracks"] if t["frame_id"]==scene["frame"]["frame_id"]]
    for a,b in lines:
        axis = b-a
        length = np.linalg.norm(axis)
        if length<1:
            continue
        axis /= length
        normal = np.array([-axis[1],axis[0]])
        depth = float((point-a)@normal)
        for fraction in (.25,.5,.75):
            tyre = np.array([left+fraction*(right-left),bottom])
            along = (tyre-a)@axis
            signed_distance = float((tyre-a)@normal)
            # The identified vehicle must meet the span of this actual divider and sit
            # across it from the candidate. A strip in a more distant stall cannot qualify.
            if not 0<along<length or abs(signed_distance)>min(.2*(bottom-top),.40*abs(depth)):
                continue
            if signed_distance*depth > 0 and abs(signed_distance) > .07*abs(depth):
                continue
            boundary = a+along*axis
            samples = boundary + np.linspace(.14,1,27)[:,None]*(point-boundary)
            if any(np.any((samples[:,0]>=x0)&(samples[:,0]<=x1)&(samples[:,1]>=y0)&(samples[:,1]<=y1))
                   for x0,y0,x1,y1 in other_vehicles):
                continue
            grid = np.floor(samples*scale).astype(int)
            if np.any(grid<0) or np.any(grid[:,0]>=classes.shape[1]) or np.any(grid[:,1]>=classes.shape[0]):
                continue
            ground = classes[grid[:,1],grid[:,0]]
            if np.all(np.isin(ground,(10,24))) and np.mean(ground==10)>=.85:
                return True
    return False


def _continuous_parking_to_row(option, anchor, scene, classes):
    """A nearby measured stall can be offered separately when direct adjacency is uncertain."""
    frame = scene["frame"]
    left, top, right, bottom = anchor
    point = np.asarray(option["point"])
    if abs(point[1]-bottom) > bottom-top:
        return False
    scale = np.array([classes.shape[1]/frame["width"], classes.shape[0]/frame["height"]])
    boxes = [t["bbox_xyxy"] for t in scene["tracks"] if t["frame_id"]==frame["frame_id"]]
    for fraction in (.25,.5,.75):
        tyre = np.array([left+fraction*(right-left), bottom])
        if np.linalg.norm(point-tyre) > 1.5*(right-left):
            continue
        samples = tyre + np.linspace(.10,1,31)[:,None]*(point-tyre)
        if any(np.any((samples[:,0]>=x0)&(samples[:,0]<=x1)&(samples[:,1]>=y0)&(samples[:,1]<=y1))
               for x0,y0,x1,y1 in boxes):
            continue
        grid = np.floor(samples*scale).astype(int)
        if np.any(grid<0) or np.any(grid[:,0]>=classes.shape[1]) or np.any(grid[:,1]>=classes.shape[0]):
            continue
        paving = classes[grid[:,1],grid[:,0]]
        if np.all(np.isin(paving,(10,24))) and np.mean(paving==10)>=.85:
            return True
    return False


def optical_parking(request, photo, surface, classes, spec):
    """Return a measured adjoining row, or distinctly label the nearest connected row as an alternative."""
    from . import whole_bay
    scene = request["scene"]
    component = bay_clearance.visible_road_component(classes)
    if component is None:
        return [], False
    _,_,_,rows = whole_bay._layout(request,photo,surface,rows=True)
    rows = [row for row in rows if row['evidence'].get('recipe') == parking_row_bays.RECIPE
            and _open_point(row,scene,classes,component) and bay_clearance.on_visible_road(row,component,scene['frame'])]
    adjacent = [row for row in rows if _adjacent_to_divider(row,spec["box"],scene,classes)]
    if adjacent:
        return adjacent, False
    nearby = [row for row in rows if _continuous_parking_to_row(row,spec["box"],scene,classes)]
    if not nearby:
        return [], False
    left,_,right,bottom = spec["box"]
    tyre = np.array([(left+right)/2,bottom])
    closest = min(nearby,key=lambda row:np.linalg.norm(np.array(row["point"])-tyre))
    return [closest], True


def parking(request, photo, surface, classes, spec):
    """Find the photographed stall next to the car or a known row opening flanked by its exact track."""
    from . import whole_bay
    scene = request["scene"]
    road = bay_clearance.visible_road_component(classes)
    if road is None:
        return []
    # The painted detector works from the native photograph and signed surface. It does
    # not need a cloud review, unrelated storefront captions, or older fragmented parking masks.
    _,_,_,stalls = whole_bay._layout(request,photo,surface)
    options = [o for o in stalls if _open_point(o,scene,classes,road)
               and bay_clearance.on_visible_road(o,road,scene["frame"])
               and _adjacent_to_divider(o,spec["box"],scene,classes)]
    with Image.open(BytesIO(photo)) as image:
        patches = paver_pullin.candidates(image,classes,visual_space.native_context(scene))
    for patch in patches:
        if spec["track_id"] not in patch["evidence"].get("flanking_track_ids",[]):
            continue
        item = {"point":patch["point"],"box":patch["box"],"polygon":patch["support_patch"],
                "evidence":{**patch["evidence"],"coverage":"interior_point_only"}}
        if not _open_point(item,scene,classes,road):
            continue
        if any(cv2.pointPolygonTest(np.float32(o["polygon"]),tuple(item["point"]),False)>=0 for o in options):
            continue
        options.append(item)
    return sorted(options,key=lambda o:tuple(o["point"]))
