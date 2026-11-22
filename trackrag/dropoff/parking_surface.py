"""Current-frame, pair-local parking patches for visible-location confirmation.

The normal surface is Mapillary's learned Parking class. Optional road extensions
are explicitly marked for a separate visual review: neither road segmentation
nor a general lane marking establishes a parking space by itself. No metric
clearance, stopping legality or vehicle control is inferred here.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np

from .contracts import validate_request
from .projected_pair import MOTOR

MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
REVISION = "4772b6bf101d91f2534c106dc524d906aeb3c68a"
METHOD = "paired_semantic_parking_surface_v1"
FALLBACK_SURFACE = "roadside_road_visual_review"
CONFIG = {
    "along": [.10, .90], "normal_radius": .60, "front_margin": .28,
    "minimum_region_width_cells": 5, "minimum_component_cells": 12,
    "hypothesis_along": [.50], "maximum_patch_radius_cells": 18,
    "patch_radius_in_local_height": .10,
    "curb_neighborhood_in_local_dimensions": [.85, .75],
    "minimum_curb_cells": 6, "minimum_parking_witness_cells": 9,
    "minimum_marking_cells": 4, "marking_elongation": 2.5,
    "maximum_marking_parallel_cosine": .75,
    "boundary_parallel_cosine": .85, "boundary_distance_in_local_height": .08,
    "boundary_marking_offset_from_anchor_row": .20,
    "road_extension_only_if_no_parking_patch": True,
    "new_center_separation_display_pixels": 24,
    "forbidden_ground_classes": [2, 7, 8, 9, 11, 12, 15, 23],
}


def class_map_sha256(classes):
    descriptor = json.dumps({"shape": list(classes.shape), "dtype": str(classes.dtype)}, sort_keys=True).encode()
    return sha256(descriptor + b"\0" + classes.tobytes(order="C")).hexdigest()


def _iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def _contact(box):
    left, top, right, bottom = box
    return ((left + right) / 2, bottom - .08 * (bottom - top), right - left, bottom - top)


def _geometry(classes, request, pair_ids):
    frame = request["scene"]["frame"]
    tracks = request["scene"]["tracks"]
    if any(t["frame_id"] != frame["frame_id"] for t in tracks):
        raise ValueError("Parking patches require current-frame observations")
    if (not isinstance(pair_ids, (list, tuple)) or len(pair_ids) != 2
            or any(type(i) is not int for i in pair_ids) or len(set(pair_ids)) != 2):
        raise ValueError("Parking patches require two distinct current vehicle IDs")
    by_id = {t["track_id"]: t for t in tracks}
    if len(by_id) != len(tracks) or any(i not in by_id or by_id[i]["class_name"] not in MOTOR for i in pair_ids):
        raise ValueError("The identified pair is absent, duplicated or not two vehicles")
    anchors = [by_id[i] for i in pair_ids]
    mh, mw = classes.shape
    sx, sy = frame["width"] / mw, frame["height"] / mh
    axes = np.array([sx, sy, sx, sy])
    boxes = [np.array(a["bbox_xyxy"], dtype=float) / axes for a in anchors]
    first, second = [_contact(box) for box in boxes]
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    mean_w, mean_h = (aw + bw) / 2, (ah + bh) / 2
    vx, vy = (bx - ax) / mean_w, (by - ay) / mean_h
    length2 = vx * vx + vy * vy
    if length2 < .12 or _iou(*boxes) > .65:
        return None
    # A selected pair can have a large scale difference under perspective. Use
    # the interpolated local size along its row instead of a fixed size ratio.
    if abs(ax - bx) > 3.2 * max(aw, bw) or abs(ay - by) > 1.6 * max(ah, bh):
        return None
    for track in tracks:
        if track["track_id"] in pair_ids or track["class_name"] not in MOTOR:
            continue
        other = np.array(track["bbox_xyxy"]) / axes
        if any(_iou(other, anchor) > .65 for anchor in boxes):
            continue
        cx, cy, _, _ = _contact(other)
        ox, oy = (cx - ax) / mean_w, (cy - ay) / mean_h
        t = (ox * vx + oy * vy) / length2
        local_w, local_h = aw * (1 - t) + bw * t, ah * (1 - t) + bh * t
        if .12 < t < .88 and math.hypot((cx - ax - t * (bx - ax)) / local_w,
                                      (cy - ay - t * (by - ay)) / local_h) < .38:
            return None
    y, x = np.mgrid[:mh, :mw]
    u, v = (x + .5 - ax) / mean_w, (y + .5 - ay) / mean_h
    along = (u * vx + v * vy) / length2
    clipped = np.clip(along, 0, 1)
    local_w, local_h = aw * (1 - clipped) + bw * clipped, ah * (1 - clipped) + bh * clipped
    center_x, center_y = ax + along * (bx - ax), ay + along * (by - ay)
    offset = np.hypot((x + .5 - center_x) / local_w, (y + .5 - center_y) / local_h)
    parcel = ((along >= CONFIG["along"][0]) & (along <= CONFIG["along"][1])
              & (offset <= CONFIG["normal_radius"])
              & (y + .5 <= center_y + CONFIG["front_margin"] * local_h))
    free = np.ones_like(parcel)
    # Exclude every observed object's complete cell intersection, not merely
    # vehicle centers or the two anchors. Semantic object classes are also absent
    # from both allowed ground surfaces below.
    for track in tracks:
        left, top, right, bottom = np.array(track["bbox_xyxy"]) / axes
        x0, x1 = max(0, math.floor(left)), min(mw, math.ceil(right))
        y0, y1 = max(0, math.floor(top)), min(mh, math.ceil(bottom))
        free[y0:y1, x0:x1] = False
    return {"anchors": anchors, "boxes": boxes, "scale": (sx, sy), "along": along,
            "offset": offset, "local_w": local_w, "local_h": local_h,
            "parcel": parcel, "free": free, "row_direction": (vx, vy)}


def _components(mask, minimum):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"), connectivity=8)
    for identity in range(1, count):
        if stats[identity, cv2.CC_STAT_AREA] >= minimum:
            yield labels == identity


def _established_centers(classes, geometry):
    """Keep the measured one-center-per-component parking search when supported.

    The complete new object/parcel/patch checks are applied afterwards. A high
    perspective ratio can instead use the new local-scale search below.
    """
    first, second = [_contact(box) for box in geometry["boxes"]]
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    if max(ah, bh) > 3.5 * min(ah, bh):
        return []
    mean_w, mean_h = (aw + bw) / 2, (ah + bh) / 2
    vx, vy = (bx - ax) / mean_w, (by - ay) / mean_h
    length2 = vx * vx + vy * vy
    mh, mw = classes.shape
    x0, x1 = max(0, math.floor(min(ax, bx) - .38 * mean_w)), min(mw, math.ceil(max(ax, bx) + .38 * mean_w))
    y0, y1 = max(0, math.floor(min(ay, by) - .68 * max(ah, bh))), min(mh, math.ceil(max(ay, by) + .28 * max(ah, bh)))
    y, x = np.mgrid[y0:y1, x0:x1]
    u, v = (x + .5 - ax) / mean_w, (y + .5 - ay) / mean_h
    along = (u * vx + v * vy) / length2
    offset = np.hypot(u - along * vx, v - along * vy)
    parcel = ((along >= .10) & (along <= .90) & (offset <= .60)
              & (y + .5 <= ay + along * (by - ay) + .26 * mean_h))
    rows = []
    for component in _components((classes[y0:y1, x0:x1] == 10) & parcel, max(9, .003 * mean_w * mean_h)):
        area = int(component.sum())
        inside = cv2.distanceTransform(np.pad(component.astype("uint8"), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        if float(inside.max()) < 2:
            continue
        centrality = np.exp(-2 * (along - .5) ** 2 - 2 * offset ** 2)
        iy, ix = np.unravel_index((inside * centrality).argmax(), inside.shape)
        rank = math.sqrt(aw * ah * bw * bh) / (1 + math.sqrt(length2))
        rank *= min(1, area / (.08 * mean_w * mean_h)) * float(centrality[iy, ix])
        rows.append((rank, x0 + int(ix), y0 + int(iy)))
    selected = []
    for row in sorted(rows, key=lambda r: -r[0]):
        if any(math.hypot((row[1] - old[1]) / mw, (row[2] - old[2]) / mh) < .018 for old in selected):
            continue
        selected.append(row)
        if len(selected) == 8:
            break
    return selected


def _cell_bounds(mask):
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def _roadside_witness(classes, geometry, point):
    """Return measured class-pixel witnesses, not a claim that a bay is legal."""
    px, py = point
    local_w, local_h = geometry["local_w"][py, px], geometry["local_h"][py, px]
    y, x = np.mgrid[:classes.shape[0], :classes.shape[1]]
    nearby = (((x + .5 - px) / (.85 * local_w)) ** 2
              + ((y + .5 - py) / (.75 * local_h)) ** 2 <= 1)
    # Curb/sidewalk evidence must sit behind or beside this visible patch,
    # rather than licensing the opposite side of an arbitrary foreground island.
    curb = np.isin(classes, (2, 15)) & nearby & geometry["free"] & (y <= py + .08 * local_h)
    curb_regions = list(_components(curb, CONFIG["minimum_curb_cells"]))
    if not curb_regions:
        return None
    # Crosswalks are a distinct semantic class; a white crossing cannot become
    # an apparent bay separator in the optional road extension.
    if np.any(np.isin(classes, (8, 23)) & nearby & geometry["parcel"]):
        return None
    curb_region = min(curb_regions, key=lambda region: float(np.min(
        ((x[region] - px) / local_w) ** 2 + ((y[region] - py) / local_h) ** 2)))
    witnesses = [{"kind": "semantic_curb_or_sidewalk", "classes": [2, 15],
                  "cell_bounds_xyxy": _cell_bounds(curb_region), "cells": int(curb_region.sum())}]
    parking = (classes == 10) & nearby & geometry["parcel"] & geometry["free"]
    parking_regions = list(_components(parking, CONFIG["minimum_parking_witness_cells"]))
    if parking_regions:
        region = max(parking_regions, key=lambda a: int(a.sum()))
        return witnesses + [{"kind": "nearby_semantic_parking", "classes": [10],
                              "cell_bounds_xyxy": _cell_bounds(region), "cells": int(region.sum())}]
    vx, vy = geometry["row_direction"]
    row_direction = np.array([vx, vy]) / math.hypot(vx, vy)
    curb_y, curb_x = np.nonzero(curb_region)
    _, curb_axes = np.linalg.eigh(np.cov(np.column_stack((curb_x / local_w, curb_y / local_h)).T))
    curb_parallel = abs(float(curb_axes[:, -1] @ row_direction))
    curb_distance = cv2.distanceTransform((~curb_region).astype("uint8"), cv2.DIST_L2, 5)
    for region in _components((classes == 24) & nearby & geometry["parcel"] & geometry["free"], CONFIG["minimum_marking_cells"]):
        yy, xx = np.nonzero(region)
        points = np.column_stack((xx / local_w, yy / local_h))
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
        elongation = eigenvalues[-1] / max(eigenvalues[0], 1e-10)
        parallel = abs(float(eigenvectors[:, -1] @ row_direction))
        if elongation >= CONFIG["marking_elongation"] and parallel <= CONFIG["maximum_marking_parallel_cosine"]:
            return witnesses + [{"kind": "nonparallel_general_marking", "classes": [24],
                                  "cell_bounds_xyxy": _cell_bounds(region), "cells": int(region.sum()),
                                  "elongation": float(elongation), "parallel_cosine": parallel}]
        # A painted curbside boundary can legitimately run along an occupied
        # parking row rather than across its bays. It is usable only as a review
        # candidate when it closely follows an independently observed curb and
        # the two vehicles' contact band. This does not classify it as parking.
        if (elongation >= CONFIG["marking_elongation"] and parallel >= CONFIG["boundary_parallel_cosine"]
                and curb_parallel >= CONFIG["boundary_parallel_cosine"]
                and float(curb_distance[region].min()) <= max(2, CONFIG["boundary_distance_in_local_height"] * local_h)
                and float(np.median(geometry["offset"][region])) <= CONFIG["boundary_marking_offset_from_anchor_row"]):
            return witnesses + [{"kind": "parallel_marking_along_curb_contact_band", "classes": [24],
                                  "cell_bounds_xyxy": _cell_bounds(region), "cells": int(region.sum()),
                                  "elongation": float(elongation), "parallel_cosine": parallel,
                                  "curb_parallel_cosine": curb_parallel,
                                  "minimum_curb_distance_cells": float(curb_distance[region].min())}]
    return None


def _patch(component, x, y, radius):
    """Grow an all-supported rectangular patch; its complete interior is replayable."""
    h, w = component.shape
    if x < 2 or y < 2 or x + 3 > w or y + 3 > h or not component[y - 2:y + 3, x - 2:x + 3].all():
        return None
    left, top, right, bottom = x - 2, y - 2, x + 3, y + 3
    for _ in range(radius - 2):
        if left > 0 and component[top:bottom, left - 1].all():
            left -= 1
        if right < w and component[top:bottom, right].all():
            right += 1
        if top > 0 and component[top - 1, left:right].all():
            top -= 1
        if bottom < h and component[bottom, left:right].all():
            bottom += 1
    return tuple(map(int, (left, top, right, bottom)))


def options_for(request, class_map, pair_ids, *, max_options=4, allow_road_extension=False, prefer_established=True):
    """Produce source-bound original-pixel patches for one learned reference pair.

    ``surface == 'roadside_road_visual_review'`` always requires an additional
    independent visual decision before it can be offered to a passenger.
    ``prefer_established=False`` is a separately reviewed, parking-only second
    board. The caller may use it only after the first chooser explicitly returns
    null; its candidates must differ visibly from the normal board's points.
    """
    request = validate_request(request)
    if (not isinstance(class_map, np.ndarray) or class_map.dtype != np.uint8 or class_map.ndim != 2
            or not class_map.size or class_map.size > 2_000_000 or int(class_map.max()) > 64):
        raise ValueError("Expected a bounded pinned 65-class semantic map")
    image_sha = request["scene"].get("provenance", {}).get("image_sha256")
    if not isinstance(image_sha, str) or re.fullmatch(r"[0-9a-f]{64}", image_sha) is None:
        raise ValueError("An original-image SHA256 is required for parking patches")
    frame = request["scene"]["frame"]
    mh, mw = class_map.shape
    if abs(mw * frame["height"] / frame["width"] - mh) > 1.01:
        raise ValueError("Semantic map and original photograph have different aspect ratios")
    if (type(max_options) is not int or not 1 <= max_options <= 8
            or type(allow_road_extension) is not bool or type(prefer_established) is not bool):
        raise ValueError("Invalid bounded parking option configuration")
    if not prefer_established and allow_road_extension:
        raise ValueError("An alternative-center board is parking-only")
    geometry = _geometry(class_map, request, pair_ids)
    if geometry is None or not re.search(r"\bbetween\b", request["text"], re.IGNORECASE):
        return []
    sx, sy = geometry["scale"]
    shared = {"method": METHOD, "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "class_map_sha256": class_map_sha256(class_map), "class_map_shape": list(class_map.shape),
              "image_sha256": image_sha,
              "request_sha256": sha256(json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
              "segmentation_model": MODEL, "segmentation_revision": REVISION,
              "configuration_sha256": sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest(),
              "allow_road_extension": allow_road_extension, "maximum_options": max_options,
              "prefer_established": prefer_established}
    first_board = options_for(request, class_map, pair_ids, max_options=max_options) if not prefer_established else []
    display_scale = min(1, 1280 / frame["width"], 800 / frame["height"])
    yy, xx = np.mgrid[:class_map.shape[0], :class_map.shape[1]]
    result = []
    surfaces = [("parking", class_map == 10)]
    if allow_road_extension:
        surfaces.append((FALLBACK_SURFACE, np.isin(class_map, (10, 13, 24))))
    for surface, semantic in surfaces:
        if surface == FALLBACK_SURFACE and result:
            break
        minimum = max(CONFIG["minimum_component_cells"], 25)
        active = semantic & geometry["parcel"] & geometry["free"]
        candidates = []
        components = list(_components(active, minimum))
        established = _established_centers(class_map, geometry) if surface == "parking" and prefer_established else []
        for rank, x, y in established:
            component = next((comp for comp in components if comp[y, x]), None)
            if component is None:
                continue
            radius = min(CONFIG["maximum_patch_radius_cells"], max(2, round(CONFIG["patch_radius_in_local_height"] * geometry["local_h"][y, x])))
            bounds = _patch(component, x, y, radius)
            if bounds is not None:
                candidates.append((rank, x, y, bounds, component, [], "established_parking_component"))
        for component in ([] if candidates else components):
            inside = cv2.distanceTransform(np.pad(component.astype("uint8"), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            valid_centers = cv2.erode(component.astype("uint8"), np.ones((5, 5), dtype="uint8"), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            if surface == FALLBACK_SURFACE:
                valid_centers &= class_map == 13
            for previous in first_board:
                distances = np.hypot(((xx + .5) * sx - previous["point"][0]) * display_scale,
                                     ((yy + .5) * sy - previous["point"][1]) * display_scale)
                valid_centers &= distances >= CONFIG["new_center_separation_display_pixels"]
            if not np.any(valid_centers):
                continue
            for desired in CONFIG["hypothesis_along"]:
                centrality = np.exp(-3 * (geometry["along"] - desired) ** 2 - 2 * geometry["offset"] ** 2)
                score = np.where(valid_centers, inside * centrality, -1.)
                y, x = np.unravel_index(score.argmax(), score.shape)
                local_h = geometry["local_h"][y, x]
                witnesses = _roadside_witness(class_map, geometry, (x, y)) if surface == FALLBACK_SURFACE else []
                if witnesses is None:
                    continue
                radius = min(CONFIG["maximum_patch_radius_cells"], max(2, round(CONFIG["patch_radius_in_local_height"] * local_h)))
                bounds = _patch(component, x, y, radius)
                if bounds is None:
                    continue
                candidates.append((float(score[y, x]), x, y, bounds, component, witnesses, "local_perspective_component"))
        for _, x, y, bounds, component, witnesses, center_source in sorted(candidates, key=lambda row: -row[0]):
            point = [float((x + .5) * sx), float((y + .5) * sy)]
            if center_source != "established_parking_component" and any(math.hypot(
                    (point[0] - old["point"][0]) * display_scale, (point[1] - old["point"][1]) * display_scale)
                    < CONFIG["new_center_separation_display_pixels"] for old in result):
                continue
            left, top, right, bottom = bounds
            patch_classes = class_map[top:bottom, left:right]
            values, counts = np.unique(patch_classes, return_counts=True)
            result.append({"choice": chr(65 + len(result)), "point": point, "tracks": list(pair_ids),
                           "anchor_boxes": [a["bbox_xyxy"] for a in geometry["anchors"]],
                           "region_area_map_pixels": int(component.sum()),
                           "region_box_xyxy": [left * sx, top * sy, right * sx, bottom * sy],
                           "region_cell_bounds_xyxy": list(bounds), "region_support_cells": int(patch_classes.size),
                           "region_class_counts": {str(int(v)): int(c) for v, c in zip(values, counts)},
                           "surface": surface, "requires_visual_review": surface == FALLBACK_SURFACE or not prefer_established,
                           "center_source": center_source,
                           "surface_witnesses": witnesses, "evidence": shared})
            if len(result) == max_options:
                return result
    return result


def validate_option(request, class_map, pair_ids, option, *, max_options=4, allow_road_extension=False, prefer_established=True):
    """Independently replay before confirmation; no edited point/region is accepted."""
    if option not in options_for(request, class_map, pair_ids, max_options=max_options,
                                 allow_road_extension=allow_road_extension, prefer_established=prefer_established):
        raise ValueError("Parking option differs from its current photograph, tracks or semantic patch")
    return True
