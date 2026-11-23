"""Check the passenger's exact mark against visible, unoccupied parking in the original photo."""
import logging

import numpy as np

from .contracts import validate_box, validate_request
from .data import inference_input_sha256
from .inputs import _inspect_image
from .passenger import digest, proposed, unresolved

LOG = logging.getLogger(__name__)
METHOD = "explicit_passenger_offlane_region_v1"
PARKING = 10  # Mapillary Vistas v1.2 semantic class used by the pinned Mask2Former.
QUESTIONS = {
    "observed_object": "Your mark touches a detected object. Could you mark the open ground itself?",
    "through_lane": "That point appears to be in a traffic lane. Could you mark a visible parking bay instead?",
    "unverified_surface": "I can’t verify an off-lane parking area at that point. Could you mark a different visible bay?",
    "mixed_area": "The marked area includes ground I can’t verify as parking. Could you tap a smaller spot inside the bay?",
    "broad_area": "That box covers a large area. Could you tap the exact spot you mean?",
    "unavailable": "I can’t verify a parking area in this photo. Could you choose another photo or describe a different spot?",
}


def mark_input(request, box):
    clean = validate_request(request)
    scene = clean["scene"]
    if scene["source"] != "predicted":
        raise ValueError("Passenger mark checking requires predicted current-frame observations")
    frame = scene["frame"]
    _, original_sha = _inspect_image(frame)
    if scene["provenance"].get("image_sha256") != original_sha:
        raise ValueError("The original image no longer matches the passenger's mark")
    mark = list(validate_box(box,frame["width"],frame["height"], name="passenger mark"))
    center = [(mark[0]+mark[2])/2, (mark[1]+mark[3])/2]
    return clean, original_sha, mark, center


def surface_check(frame, tracks, mark, center, classes):
    width, height = frame["width"], frame["height"]
    x0,y0,x1,y1 = mark
    cx,cy = center
    intersecting = []
    for row in tracks:
        if row["frame_id"] != frame["frame_id"]:
            continue
        left,top,right,bottom = row["bbox_xyxy"]
        if min(x1,right)>max(x0,left) and min(y1,bottom)>max(y0,top):
            intersecting.append(row["track_id"])
    checked = {"intersecting_current_track_ids":sorted(intersecting)}
    if intersecting:
        return "observed_object", checked
    if (x1-x0)/width > .25 or (y1-y0)/height > .25:
        return "broad_area", checked
    if classes is None:
        return "unavailable", checked
    rows, columns = classes.shape
    sx, sy = columns/width, rows/height
    # A cell contributes if its footprint overlaps the actual mark.
    left, top = int(np.floor(x0*sx)), int(np.floor(y0*sy))
    right, bottom = int(np.ceil(x1*sx)), int(np.ceil(y1*sy))
    ix, iy = min(columns-1,int(cx*sx)), min(rows-1,int(cy*sy))
    selected = classes[top:bottom,left:right]
    core = int(classes[iy,ix])
    # A single class-10 pixel can occur at a kerb or a vehicle boundary.
    # Require a small, contiguous parking neighbourhood around the original point.
    radius = max(2, int(np.ceil(.003*max(columns,rows))))
    nx0,ny0 = max(0,ix-radius),max(0,iy-radius)
    nx1,ny1 = min(columns,ix+radius+1),min(rows,iy+radius+1)
    neighborhood = classes[ny0:ny1,nx0:nx1]
    parking = selected == PARKING
    checked.update(class_map_shape=[rows,columns], center_cell_xy=[ix,iy], center_class_id=core,
                   mark_cells=int(selected.size), mark_parking_cells=int(np.count_nonzero(parking)),
                   neighborhood_radius=radius, nearby_cells=int(neighborhood.size),
                   nearby_parking_cells=int(np.count_nonzero(neighborhood==PARKING)))
    if core == 13:
        return "through_lane", checked
    if core != PARKING:
        return "unverified_surface", checked
    if not np.all(parking):
        return "mixed_area", checked
    # The exact pin and its immediate neighbours must sit within the same visible patch.
    # Additional nearby pixels may include a narrow line beyond the selected rectangle.
    if (ix<radius or iy<radius or ix+radius>=columns or iy+radius>=rows
            or not np.all(classes[iy-1:iy+2,ix-1:ix+2]==PARKING)
            or checked["nearby_parking_cells"] < .85*neighborhood.size):
        return "unverified_surface", checked
    return "parking", checked


def _outcome(clean, original_sha, mark, center, packet):
    from . import manual_surfaces
    scene = clean["scene"]
    classes = manual_surfaces.decode(packet, scene["frame"], original_sha) if packet is not None else None
    reason, checked = surface_check(scene["frame"], scene["tracks"], mark, center, classes)
    result = proposed(center,mark) if reason == "parking" else unresolved(QUESTIONS[reason])
    return {"ok":True, "request_id":clean["request_id"], "result":result, "provenance":{
        "method":METHOD, "image":{"sha256":original_sha}, "request_sha256":digest(clean),
        "inference_input_sha256":inference_input_sha256(clean,original_sha),
        "selected_box":mark, "mark_center_xy":center, "offlane_surface_packet":packet,
        "surface_check":checked, "reason":reason,
        "operational_status":"complete" if packet is not None else "unavailable",
        "independent_model_proposal":False,
    }}


def select(request, box):
    """Look up an authorized source-bound map; a missing or corrupt map cannot yield a pin."""
    from . import manual_surfaces
    clean, original_sha, mark, center = mark_input(request, box)
    try:
        packet = manual_surfaces.get_packet(original_sha)
        return _outcome(clean,original_sha,mark,center,packet)
    except (OSError, KeyError, TypeError, ValueError) as error:
        LOG.warning("Parking evidence unavailable for manual mark (%s)", type(error).__name__)
        return _outcome(clean,original_sha,mark,center,None)


def verify(request, outcome, *, requested_box=None):
    """Reproduce an offered or withheld manual decision using its own signed receipt."""
    source = outcome["provenance"]
    box = source["selected_box"] if requested_box is None else requested_box
    clean, original_sha, mark, center = mark_input(request, box)
    packet = source["offlane_surface_packet"]
    if _outcome(clean,original_sha,mark,center,packet) != outcome:
        raise ValueError("Passenger mark differs from the original photo, submitted box or parking evidence")
    return {"current_image_and_tracks_checked":True, "submitted_mark":mark, "mark_center_xy":center,
            "offlane_mask_replayed":packet is not None, "decision_reproduced":True,
            "operational_status":source["operational_status"], "independent_model_proposal":False}
