"""Derive and replay lateral parking observations on registered public driver-view photos."""
from collections import OrderedDict
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from threading import Lock

import numpy as np
from PIL import Image

from . import bay_clearance, manual_surfaces, single_vehicle, vehicle_curb_ground
from . import vehicle_ground_reference as identity, vehicle_reference_qualifiers as qualifiers
from . import vehicle_side_view as view
from .bay_packets import read_photo_packet
from .parking_evidence import _authenticate, source_input
from .single_vehicle_surface import beside_patches
from .vehicle_relative_bays import restricted_instruction


BANK = Path(__file__).resolve().parents[2] / "experiments/dropoff/passenger-vehicle-side-ground-v1.zip"
RELEASE = BANK.with_name("passenger-vehicle-side-ground-release-v1.json")
_verified = OrderedDict()
_lock = Lock()
_OWN_SIDE = re.compile(
    r"\b(?:its|(?:vehicle|car|truck|pickup|sedan|suv|bus|van)'s)\s+(?:own\s+)?(left|right)\b"
)


def identity_packet(key, specs, answers=None, *, packet=None):
    if packet is None:
        packet = read_photo_packet(identity.BANK, key, 8192)
    if packet is None:
        return None
    _authenticate(packet)
    body = packet["body"]
    if (body["source"] != key or body["model"] != identity.MODEL or body["revision"] != identity.REVISION
            or body["recipe"] != identity.view.RECIPE or body["prompt"] != identity.view.checksum(identity.view.GUIDE)
            or body["spec"] != view.digest(specs) or body["count"] != len(specs)
            or len(body["answers"]) != len(specs) or answers is not None and body["answers"] != answers):
        raise ValueError("Vehicle identity evidence differs from the current source or observations")
    return packet


def geometry_rows(scene, classes, specs, answers):
    """Select uniformly across the bank using a legible, distinct vehicle body and clear native parking."""
    frame = scene["frame"]
    width, height = frame["width"], frame["height"]
    boxes = [t["bbox_xyxy"] for t in scene["tracks"] if t["frame_id"] == frame["frame_id"]]
    road = bay_clearance.visible_road_component(classes)
    if road is None:
        return []
    sy, sx = classes.shape[0]/height, classes.shape[1]/width
    rows = []
    for spec, answer in zip(specs, answers, strict=True):
        colour, kind = answer[:2]
        if colour == "unknown" or kind == "unknown":
            continue
        rivals = [r for r in answers if r[0] in {colour, "unknown"}
                  and (r[1] == "unknown" or single_vehicle._compatible(kind, r[1]))]
        if len(rivals) != 1:
            continue
        for patch in beside_patches(classes, spec["box"], boxes, (width, height)):
            left, top, right, bottom = patch["box"]
            x0, y0, x1, y1 = map(round, (left*sx, top*sy, right*sx, bottom*sy))
            if (not 0 <= x0 < x1 <= classes.shape[1] or not 0 <= y0 < y1 <= classes.shape[0]
                    or not road[y0:y1, x0:x1].all() or not (classes[y0:y1, x0:x1] == 10).all()
                    or any(max(left, a) < min(right, c)-1e-6 and max(top, b) < min(bottom, d)-1e-6
                           for a, b, c, d in boxes)):
                continue
            rows.append({"track_id": spec["track_id"], "anchor_box": spec["box"], "point": patch["point"],
                         "box": patch["box"], "side": patch["side"], "role": "candidate"})
    if not rows:
        return []
    # One wrong-side, occupied-footprint and (when visible) travel lane control per source.
    example = rows[0]
    opposite = "right" if example["side"] == "left" else "left"
    rows.append({**example, "side": opposite, "role": "wrong_side_control"})
    left, top, right, bottom = example["anchor_box"]
    occupied = [(left+right)/2, (top+bottom)/2]
    rows.append({**example, "point": occupied, "box": None, "role": "occupied_control"})
    lane = np.asarray((classes == 13) & road, dtype=np.uint8)
    for a, b, c, d in boxes:
        lane[max(0, int(b*sy)):min(classes.shape[0], int(np.ceil(d*sy))),
             max(0, int(a*sx)):min(classes.shape[1], int(np.ceil(c*sx)))] = 0
    import cv2
    clearance = cv2.distanceTransform(lane, cv2.DIST_L2, 5)
    yy, xx = np.nonzero(clearance >= 9)
    if len(xx):
        # Find verified interior road pavement near the car at a comparable contact depth.
        cx = (left+right)/2
        score = ((xx/sx-cx)/max(1, right-left))**2 + 2*((yy/sy-bottom)/max(1, bottom-top))**2
        index = int(np.argmin(score))
        point = [(float(xx[index])+.5)/sx, (float(yy[index])+.5)/sy]
        side = "left" if point[0] < cx else "right"
        rows.append({**example, "point": point, "box": None, "side": side, "role": "lane_control"})
    return rows


def context(scene, surface, observation_packet):
    frame = scene["frame"]
    tracks = [[t["track_id"], t["class_name"], t["bbox_xyxy"]] for t in scene["tracks"]
              if t["frame_id"] == frame["frame_id"]]
    return view.digest([frame["recording_id"], frame["frame_id"], frame["width"], frame["height"],
                        tracks, surface["signature"], observation_packet["signature"]])


def verify_rows(photo, key, rows, packet, stamp):
    _authenticate(packet)
    body = packet["body"]
    if (set(body) != {"recipe", "source", "context", "model", "revision", "guide", "rows"}
            or body["source"] != key or sha256(photo).hexdigest() != key or body["context"] != stamp
            or body["recipe"] != view.RECIPE or body["model"] != identity.MODEL
            or body["revision"] != identity.REVISION or body["guide"] != view.digest(view.GUIDE)
            or len(rows) != len(body["rows"])):
        raise ValueError("The side observations do not belong to this vehicle, original and surface")
    with Image.open(BytesIO(photo)) as image:
        for row, record in zip(rows, body["rows"], strict=True):
            if (record["spec"] != view.digest(row) or len(record["calls"]) != 2
                    or any(call["image"] != sha256(view.render(image, row, close=close)).hexdigest()
                           or call["verdict"] not in view.VERDICTS
                           for close, call in zip((False, True), record["calls"], strict=True))):
                raise ValueError("The original side views or their decisions differ")
    return [tuple(call["verdict"] for call in record["calls"]) for record in body["rows"]]


def qualified_indices(rows, readings):
    """A mistaken source control or missing calibration withholds all options on that original."""
    if len(rows) != len(readings) or not rows:
        return []
    roles = [row["role"] for row in rows]
    if (roles.count("wrong_side_control") != 1 or roles.count("occupied_control") != 1
            or roles.count("lane_control") != 1 or set(roles) - {
                "candidate", "wrong_side_control", "occupied_control", "lane_control"
            }):
        return []
    for row, pair in zip(rows, readings, strict=True):
        if len(pair) != 2 or any(v not in view.VERDICTS for v in pair):
            return []
        if row["role"] != "candidate" and tuple(pair) == ("OPEN", "OPEN"):
            return []
    return [i for i, (row, pair) in enumerate(zip(rows, readings, strict=True))
            if row["role"] == "candidate" and tuple(pair) == ("OPEN", "OPEN")]


def release_passes(summary):
    """Use the preregistered original-photo review to decide a global release, never individual photos."""
    counts = ("independently_supported_offers", "accepted_negative_controls", "accepted_wrong_places")
    return (all(type(summary.get(k)) is int and summary[k] >= 0 for k in counts)
            and summary["independently_supported_offers"] > 0
            and summary["accepted_negative_controls"] == 0 and summary["accepted_wrong_places"] == 0)


@lru_cache(maxsize=4)
def _release_state(archive_stamp, review_stamp):
    record = json.loads(RELEASE.read_bytes())
    return (record.get("recipe") == view.RECIPE and record.get("bank_sha256") == sha256(BANK.read_bytes()).hexdigest()
            and release_passes(record.get("release_counts", {})) and record.get("enabled") is True)


def released():
    try:
        if not _release_state(BANK.stat().st_mtime_ns, RELEASE.stat().st_mtime_ns):
            return False
        record = json.loads(RELEASE.read_bytes())
        name = record["review_artifact"]
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".json") or name == RELEASE.name:
            return False
        return sha256(RELEASE.with_name(name).read_bytes()).hexdigest() == record["review_sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        return False


def release_id():
    """Journal this ID. Replaying a side offer requires its original installed release and review."""
    return sha256(RELEASE.read_bytes()).hexdigest() if released() else None


def requested_image_side(words, described_side, facing, box, size):
    """Convert vehicle-relative left/right to image-relative left/right for a clear front or rear view."""
    own = _OWN_SIDE.findall(words.casefold().replace("’", "'"))
    if not own:
        return True, described_side
    if (len(set(own)) != 1 or own[0] != described_side or facing not in {"FRONT", "REAR"}
            or not single_vehicle.visible_vehicle_end(box, size)):
        return False, None
    if facing == "FRONT":
        return True, "right" if described_side == "left" else "left"
    return True, described_side


def _confirmed_geometry(scene, photo, surface, specs, answers, packet, identity_evidence):
    key = scene["provenance"]["image_sha256"]
    vehicle = identity_packet(key, specs, answers, packet=identity_evidence)
    if not vehicle:
        return []
    stamp = context(scene, surface, vehicle)
    cache_key = view.digest([key, stamp, surface, packet])
    with _lock:
        if cache_key in _verified:
            _verified.move_to_end(cache_key)
            return _verified[cache_key]
    classes = manual_surfaces.decode(surface, scene["frame"], key)
    if vehicle_curb_ground.describe(scene, classes) != specs:
        raise ValueError("Current source geometry differs from signed vehicle crops")
    rows = geometry_rows(scene, classes, specs, answers)
    verdicts = verify_rows(photo, key, rows, packet, stamp)
    confirmed = [rows[i] for i in qualified_indices(rows, verdicts)]
    with _lock:
        _verified[cache_key] = confirmed
        if len(_verified) > 48:
            _verified.popitem(last=False)
    return confirmed


def side_ground_choices(request, photo, surface, ref, specs, answers, *, packet=None, observations=None,
                        identity_evidence=None, expected_release=None):
    """Rebuild an optional passenger choice on the requested side of the uniquely identified vehicle."""
    if (not released() or not ref or ref.get("relation") != "beside" or ref.get("unsupported")
            or restricted_instruction(request.get("text"))
            or request["scene"].get("provenance", {}).get("network_policy") == "local_only"):
        return None
    if expected_release is not None and expected_release != release_id():
        raise ValueError("Recorded side-ground release differs from the installed release")
    clean, key, _ = source_input(request, photo)
    if identity.direct_reference(clean) != ref:
        raise ValueError("Vehicle side differs from the current instruction")
    published = read_photo_packet(BANK, key, 160000)
    if published is None:
        return None
    if packet is not None and packet != published:
        raise ValueError("Signed side evidence differs from the approved observation archive")
    packet = published
    scene = clean["scene"]
    # Resolve the passenger's vehicle attributes first. The separate side views below establish ground_side.
    if ref.get("qualified"):
        # The containing outcome already authenticates the recorded attribute packets. Direct backend
        # callers can omit observations to recheck the normal current-source packets here instead.
        signed_attributes = observations
        if signed_attributes is None:
            signed_attributes = qualifiers.read_observations(clean, photo, surface) if ref.get("qualifiers") else []
        matches, _, question = qualifiers.resolve(clean, {**ref, "ground_side": None}, specs, answers,
                                                   observations=signed_attributes)
        if question:
            return None
    else:
        matches = identity._matching(specs, answers, ref, scene["frame"]["width"])
    if len(matches) != 1:
        return None
    spec, answer = matches[0]
    frame = scene["frame"]
    consistent, side = requested_image_side(clean["text"], ref.get("ground_side"), answer[2], spec["box"],
                                            (frame["width"],frame["height"]))
    if not consistent:
        return None
    verified = _confirmed_geometry(scene, photo, surface, specs, answers, packet, identity_evidence)
    options = []
    for row in verified:
        if row["track_id"] != spec["track_id"] or row["anchor_box"] != spec["box"] or side and row["side"] != side:
            continue
        l, t, r, b = row["box"]
        options.append({"point": list(row["point"]), "box": list(row["box"]), "polygon": [[l,t],[r,t],[r,b],[l,b]],
                        "evidence": {"recipe": view.RECIPE, "coverage": "interior_point_only", "side": row["side"],
                                     "track_id": row["track_id"]}})
    if not options:
        return None
    label = " ".join("SUV" if word == "suv" else word for word in answer[:2] if word != "unknown")
    if len(options) == 1:
        direction = options[0]["evidence"]["side"]
        question = f"I can see open parking on the image {direction} of the outlined {label}. Is this the point you meant?"
    else:
        question = f"I can see open parking on both sides of the outlined {label}. Which point did you mean?"
    return options, packet, question
