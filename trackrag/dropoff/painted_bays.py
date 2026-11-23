"""Recover the visible part of a marked parking stall from its two painted dividers.

Coordinates returned to the passenger application are pixels of the original photograph.
Generic road paint is never sufficient. The paired markings must terminate toward a curb,
enclose visible parking evidence and have an unoccupied interior in the current observations.
"""
from itertools import combinations
from math import ceil

import cv2
import numpy as np
from PIL import Image


PARKING, ROAD, LINE = 10, 13, 24
CROSSINGS = (8, 23)
GROUND = (PARKING, ROAD, LINE)
RECIPE = "paired_painted_stall_dividers_v1"
ROW_RECIPE = "painted_dividers_in_observed_parking_row_v2"


def _samples(array, points):
    x, y = np.rint(points).astype(int).T
    return array[y.clip(0, array.shape[0] - 1), x.clip(0, array.shape[1] - 1)]


def _normal(direction):
    return np.array([-direction[1], direction[0]])


def _coordinates(values):
    return np.round(values, 3).tolist()


def _painted_line(start, end, thickness, gray, neutral, ground, labelled):
    """Follow locally brighter paint using both flanks; a single contrast edge is insufficient."""
    length = float(np.linalg.norm(end - start))
    if length < max(13, gray.shape[1] * .025):
        return None
    direction = (end - start) / length
    normal = _normal(direction)
    spread = min(ceil(length * .64), ceil(gray.shape[1] * .09))
    positions = np.arange(-spread, ceil(length) + spread + 1)
    centres = start + positions[:, None] * direction
    radius = int(np.clip(round(thickness / 2), 1, 4))
    offsets = np.arange(-radius - 1, radius + 2)
    strips = np.stack([_samples(gray, centres + offset * normal) for offset in offsets])
    peak_indices = strips.argmax(axis=0)
    peak = strips[peak_indices, np.arange(len(positions))]
    left = _samples(gray, centres - (radius + 4) * normal)
    right = _samples(gray, centres + (radius + 4) * normal)
    pavement_both_sides = (_samples(ground, centres - (radius + 3) * normal)
                           & _samples(ground, centres + (radius + 3) * normal))
    in_frame = ((centres[:, 0] >= 1) & (centres[:, 1] >= 1)
                & (centres[:, 0] < gray.shape[1] - 1) & (centres[:, 1] < gray.shape[0] - 1))
    good = (in_frame & _samples(ground, centres) & _samples(neutral, centres + offsets[peak_indices, None] * normal)
            & (peak > 62) & (peak - (left + right) * .5 >= 8) & (peak - np.maximum(left, right) >= 3))
    initial = (positions >= 0) & (positions <= length)
    support = float(good[initial].mean())
    if support < (.44 if labelled else .74) or pavement_both_sides[initial].mean() < .59:
        return None
    # Faint repeating paver joints can continue in the same direction as a white stripe.
    # Beyond the observed seed, demand sustained two-flank paint contrast on pavement.
    edge_contrast = peak - np.maximum(left, right)
    strong = good & pavement_both_sides & (edge_contrast >= max(8., np.median(edge_contrast[initial]) * .44))
    sustained = np.convolve(strong.astype(float), np.ones(7) / 7, mode="same") >= .57
    good &= initial | (strong & sustained)
    # Permit small interruptions in worn paint, never continue through a long unseen divider.
    gaps = int(np.clip(round(length * .055), 3, 9))
    runs = np.split(np.flatnonzero(good), np.flatnonzero(np.diff(np.flatnonzero(good)) > gaps + 1) + 1)
    run = max(runs, key=lambda r: np.count_nonzero(initial[r]))
    if len(run) < 10 or np.count_nonzero(initial[run]) < .65 * initial.sum() * support:
        return None
    offset = float(np.median(offsets[peak_indices[run]]))
    a, b = centres[run[[0, -1]]] + normal * offset
    return {"ends": np.array([a, b]), "width": max(2., float(thickness)), "contrast": support,
            "semantic": labelled}


def _same_line(first, other):
    a, b = first["ends"]
    c, d = other["ends"]
    direction = (b - a) / np.linalg.norm(b - a)
    other_dir = (d - c) / np.linalg.norm(d - c)
    if abs(direction @ other_dir) < .994:
        return False
    normal = _normal(direction)
    if abs(((c + d - a - b) / 2) @ normal) > max(3.5, (first["width"] + other["width"]) * .65):
        return False
    interval = sorted(((c - a) @ direction, (d - a) @ direction))
    return min(np.linalg.norm(b - a), interval[1]) - max(0, interval[0]) > .3 * min(
        np.linalg.norm(b - a), np.linalg.norm(d - c))


def _partial_dividers(gray, neutral, ground, classes, base):
    """Measure short/worn dividers even where the surface model called the asphalt 'road'."""
    width = classes.shape[1]
    lines = [dict(line) for line in base]
    # A local contrast stretch retains actual faint paint; it does not manufacture its geometry.
    contrast = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(8, 8)).apply(gray.astype("uint8")).astype("float32")
    count, ids, stats, _ = cv2.connectedComponentsWithStats((classes == LINE).astype("uint8"))
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if max(w, h) < max(12, width * .015) or area < 15:
            continue
        coords = np.column_stack(np.nonzero(ids[y:y + h, x:x + w] == i)[::-1]) + [x, y]
        centre = coords.mean(axis=0)
        _, _, axes = np.linalg.svd(coords - centre, full_matrices=False)
        along = (coords - centre) @ axes[0]
        length = float(np.ptp(along))
        thickness = area / max(1., length)
        if length < max(12., thickness * 3.0):
            continue
        ends = centre + np.outer([np.quantile(along, .012), np.quantile(along, .988)], axes[0])
        if any(_same_line({"ends":ends, "width":thickness}, line) for line in lines):
            continue
        # Confirm optical paint on the signed stroke with dark pavement on both sides of its core.
        # Partial occlusion at the ends can make the measured stripe relatively wide.
        sample = np.linspace(*ends, max(14, round(length)))
        normal = axes[1]
        half = max(1., thickness * .46)
        brightest = np.max([_samples(contrast, sample + normal * off) for off in np.linspace(-half, half, 5)], axis=0)
        flanks = [_samples(contrast, sample + sign * normal * (half + 3.)) for sign in (-1, 1)]
        valid = (_samples(ground, sample) & _samples(neutral, sample)
                 & (brightest - np.maximum(*flanks) >= 3.2) & (brightest - np.mean(flanks, axis=0) >= 6.))
        central = slice(len(sample) // 10, len(sample) - len(sample) // 10)
        support = float(valid[central].mean())
        if support >= .53:
            lines.append({"ends":ends, "width":float(thickness), "contrast":support, "semantic":True})
    return lines[:48]


def _row_features(lines, classes, curb_distance):
    """Identify repetitions of stall dividers oriented into one independently observed kerb."""
    width = classes.shape[1]
    rows = []
    for line in lines:
        ends = line["ends"]
        distances = _samples(curb_distance, ends)
        if distances[0] < distances[1]:
            ends, distances = ends[::-1], distances[::-1]
        mouth, back = ends
        length = float(np.linalg.norm(back - mouth))
        direction = (back - mouth) / length
        samples = np.linspace(mouth, back, 23)[1:-1]
        across_line = _normal(direction) * max(1., line["width"] * .30)
        seen = np.logical_or.reduce([_samples(classes, samples + k * across_line) == LINE for k in (-1, 0, 1)])
        # A painted short cross stroke at the mouth of a parking divider (T or L).
        # Multiple connected paint cells must extend away from the stem into its end cap.
        radius = min(max(8, int(line["width"] * 3.8)), max(9, round(.17 * length)))
        x0, y0 = np.maximum(0, np.floor(mouth - radius).astype(int))
        x1, y1 = np.ceil(mouth + radius + 1).astype(int)
        near = np.column_stack(np.nonzero(classes[y0:y1, x0:x1] == LINE)[::-1]) + [x0, y0] - mouth
        along = near @ direction
        across = near @ _normal(direction)
        cap = np.abs(across) > max(3.3, line["width"] * .9)
        at_end = (along > -max(4., line["width"])) & (along < max(8., line["width"] * 1.7))
        cap_count = np.count_nonzero(cap & at_end)
        line["row_ends"] = ends
        line["row_dir"] = direction
        line["row_length"] = length
        line["row_cap"] = bool(cap_count >= max(4., line["width"] * .9))
        line["row_paint_class_support"] = float(seen.mean())
        line["row_curb"] = float(distances[1])
        line["row_toward"] = float((distances[0] - distances[1]) / length)
        if distances[1] <= max(6, width * .045) and line["row_toward"] > .16:
            rows.append(line)
    return rows


def _row_support(first, second, anchors, classes):
    """Require an additional observed stem on the same curb-facing row for a road-class interior."""
    a, b = first["row_ends"], second["row_ends"]
    axis = first["row_dir"] + second["row_dir"]
    axis /= np.linalg.norm(axis)
    longest = max(first["row_length"], second["row_length"])
    row_span = np.linalg.norm(a[1] - b[1])
    neighbors = []
    for other in anchors:
        if other is first or other is second or other["row_dir"] @ axis < .79:
            continue
        other_ends = other["row_ends"]
        if min(np.linalg.norm(other_ends[1] - a[1]), np.linalg.norm(other_ends[1] - b[1])) > max(row_span * 2.2, longest * .85):
            continue
        # A segmented fragment of the same divider is not a third repeated stall marking.
        if any(abs((other_ends.mean(axis=0) - v["row_ends"].mean(axis=0)) @ _normal(v["row_dir"]))
               < max(5., v["width"] * 1.7) for v in (first, second)):
            continue
        neighbors.append(other)
    capped = [v for v in (first, second, *neighbors) if v["row_cap"]]
    curb_ends = sum(v["row_curb"] <= max(6., classes.shape[1] * .022) for v in (first, second))
    return neighbors, len(capped), curb_ends


def _dividers(gray, neutral, ground, classes, parking_distance):
    width = classes.shape[1]
    count, labels, stats, _ = cv2.connectedComponentsWithStats((classes == LINE).astype("uint8"))
    seeds = []
    for index in range(1, count):
        left, top, w, h, area = stats[index]
        if area < 12 or max(w, h) < max(13, width * .025):
            continue
        ys, xs = np.where(labels[top:top + h, left:left + w] == index)
        points = np.column_stack((xs + left, ys + top))
        center = points.mean(axis=0)
        _, _, axes = np.linalg.svd(points - center, full_matrices=False)
        along = (points - center) @ axes[0]
        length = float(np.ptp(along))
        thickness = area / max(1, length)
        if length >= max(13, 5.5 * thickness):
            seeds.append((center + along.min() * axes[0], center + along.max() * axes[0], thickness, True))

    # A segmenter occasionally labels white parking paint as ordinary pavement. Optical edges
    # may seed a divider too, but still have to pass the two-flank paint and signed surface checks.
    detected = cv2.createLineSegmentDetector().detect(gray.astype("uint8"))[0]
    if detected is not None:
        optical = []
        for x1, y1, x2, y2 in detected.reshape(-1, 4):
            a, b = np.array([x1, y1]), np.array([x2, y2])
            length = np.linalg.norm(b - a)
            if length >= max(22, width * .045):
                sample = np.linspace(a, b, 9)
                if _samples(ground, sample).mean() > .76 and _samples(parking_distance, sample).min() < width * .05:
                    optical.append((a, b, max(2., width * .003), False))
        seeds.extend(sorted(optical, key=lambda s: -np.linalg.norm(s[1] - s[0]))[:50])

    lines = []
    for a, b, thickness, labelled in seeds:
        sample = np.linspace(a, b, 11)
        if _samples(parking_distance, sample).min() > width * .075:
            continue
        # Optical detection locates a contrast edge. Sample either side before judging
        # the centre of the actual narrow stripe against both of its dark flanks.
        normal = _normal((b - a) / np.linalg.norm(b - a))
        shifts = (0.,) if labelled else (0., thickness, -thickness)
        measured = [_painted_line(a + shift * normal, b + shift * normal, thickness, gray,
                                  neutral, ground, labelled) for shift in shifts]
        valid = [line for line in measured if line is not None]
        if valid:
            lines.append(max(valid, key=lambda line: line["contrast"]))
    # Retain the longest measured extent of each physical stripe, not its two opposite edges.
    distinct = []
    for line in sorted(lines, key=lambda s: -np.linalg.norm(s["ends"][1] - s["ends"][0])):
        if not any(_same_line(s, line) for s in distinct):
            distinct.append(line)
    return distinct[:36]


def _clip_edge(polygon, axis, value, keep_less):
    clipped = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        inside_a, inside_b = (a[axis] <= value, b[axis] <= value) if keep_less else (a[axis] >= value, b[axis] >= value)
        if inside_a:
            clipped.append(a)
        if inside_a != inside_b:
            clipped.append(a + (b - a) * ((value - a[axis]) / (b[axis] - a[axis])))
    return np.array(clipped, dtype="float32")


def _visible_polygon(polygon, boxes):
    """A material overlap invalidates the stall. Trim only a small peripheral occlusion."""
    for x0, y0, x1, y1 in boxes:
        if polygon[:, 0].max() <= x0 or polygon[:, 0].min() >= x1 or polygon[:, 1].max() <= y0 or polygon[:, 1].min() >= y1:
            continue
        rect = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        overlap = cv2.intersectConvexConvex(polygon, rect)[0]
        if overlap <= 0:
            continue
        area = cv2.contourArea(polygon)
        if overlap > .065 * area or overlap > .45 * (x1 - x0) * (y1 - y0):
            return None
        alternatives = [_clip_edge(polygon, axis, value, keep_less) for axis, value, keep_less in
                        [(0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)]]
        polygon = max(alternatives, key=lambda p: cv2.contourArea(p) if len(p) >= 3 else 0)
        if len(polygon) < 3 or cv2.contourArea(polygon) < .89 * area:
            return None
    return polygon


def _observed_back_edge(polygon, boundaries, directions, classes, curb_distance, boxes):
    """Complete a truncated rear edge only with visible parking touching the curb inside both divider rays."""
    a, b = boundaries
    reach = min(classes.shape[1] * .15, max(np.linalg.norm(v[1] - v[0]) for v in boundaries) * .88)
    extent = np.float32([a[0], a[1] + reach * directions[0], b[1] + reach * directions[1], b[0]])
    if not cv2.isContourConvex(extent):
        return polygon, 0
    x, y, w, h = cv2.boundingRect(extent)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(classes.shape[1], x + w), min(classes.shape[0], y + h)
    local = classes[y0:y1, x0:x1]
    region = np.zeros(local.shape, dtype="uint8")
    cv2.fillConvexPoly(region, np.rint(extent - [x0, y0]).astype("int32"), 1)
    open_parking = (local == PARKING) & (region > 0)
    for left, top, right, bottom in boxes:
        open_parking[max(0, int(np.floor(top)) - y0):max(0, int(np.ceil(bottom)) - y0),
                     max(0, int(np.floor(left)) - x0):max(0, int(np.ceil(right)) - x0)] = False
    _, areas = cv2.connectedComponents(open_parking.astype("uint8"))
    past = np.zeros_like(region)
    cv2.fillConvexPoly(past, np.rint(polygon - [x0, y0]).astype("int32"), 1)
    joined = np.unique(areas[(past > 0) & open_parking])
    curb_cells = (open_parking & np.isin(areas, joined) & (past == 0)
                  & (curb_distance[y0:y1, x0:x1] <= max(2.2, classes.shape[1] * .006)))
    witnesses = np.column_stack(np.nonzero(curb_cells)[::-1]) + [x0, y0]
    if len(witnesses) < 7:
        return polygon, 0
    # The convex hull connects photographed curb witnesses; projected rays only bound the search.
    extended = cv2.convexHull(np.vstack((polygon, witnesses)).astype("float32"))[:, 0]
    extended = _visible_polygon(extended, boxes)
    if extended is None:
        return polygon, 0
    envelope = np.zeros_like(region)
    cv2.fillConvexPoly(envelope, np.rint(extended - [x0, y0]).astype("int32"), 1)
    added = (envelope > 0) & (past == 0)
    if (not np.any(added) or open_parking[added].mean() < .91
            or np.any(~np.isin(local[added], (PARKING, ROAD, LINE, 2)))):
        return polygon, 0
    return extended, len(witnesses)


def _pair(first, second, lines, classes, curb_distance, curb, boxes, scale, row_anchors=None):
    measured_row = row_anchors is not None
    boundaries, directions, lengths, distances = [], [], [], []
    for line in (first, second):
        ends = line["ends"]
        toward = _samples(curb_distance, ends)
        if toward[0] < toward[1]:
            ends, toward = ends[::-1], toward[::-1]
        length = np.linalg.norm(ends[1] - ends[0])
        if length < 10:
            return None
        boundaries.append(ends)
        directions.append((ends[1] - ends[0]) / length)
        lengths.append(length)
        distances.append(toward)
    d0, d1 = directions
    width = classes.shape[1]
    if d0 @ d1 < (.71 if measured_row else .86) or min(lengths) < (.14 if measured_row else .25) * max(lengths):
        return None
    avg = (d0 + d1) / np.linalg.norm(d0 + d1)
    delta = (boundaries[1] - boundaries[0]).mean(axis=0)
    separation = abs(delta @ _normal(avg))
    if (separation < max(6, width * (.009 if measured_row else .011),
                         (3.1 if measured_row else 3.7) * max(first["width"], second["width"]))
            or separation > (1.04 if measured_row else .88) * max(lengths)
            or abs(delta @ avg) > (1.26 if measured_row else .92) * max(lengths)):
        return None
    near = int(np.argmin([s[1] for s in distances]))
    if (distances[near][1] > max(5., min(width * (.046 if measured_row else .022), separation * (.75 if measured_row else .47)))
            or max((s[0] - s[1]) / length for s, length in zip(distances, lengths)) < .16):
        return None
    end = boundaries[near][1]
    radius = max(10, ceil(width * .04))
    x0, y0 = np.maximum(0, np.floor(end - radius).astype(int))
    x1, y1 = np.ceil(end + radius + 1).astype(int)
    nearby = np.column_stack(np.nonzero(curb[y0:y1, x0:x1])[::-1]) + [x0, y0]
    close = nearby[np.sum((nearby - end) ** 2, axis=1) < radius ** 2]
    if len(close) >= 8:
        _, values, axes = np.linalg.svd(close - close.mean(axis=0), full_matrices=False)
        if values[0] > 2 * values[1] and abs(axes[0] @ avg) > .978:
            return None  # A stripe alongside the curb can be an ordinary lane edge.
    a, b = boundaries
    original = np.float32([a[0], a[1], b[1], b[0]])
    if not cv2.isContourConvex(original) or cv2.contourArea(original) < max(50., width ** 2 * .00040):
        return None
    # Two nonadjacent dividers describe several stalls, not a single destination.
    for other in lines:
        if other is first or other is second:
            continue
        direction = other["ends"][1] - other["ends"][0]
        if abs(direction @ avg) / np.linalg.norm(direction) < .87:
            continue
        sample = np.linspace(*other["ends"], 17)[1:-1]
        if sum(cv2.pointPolygonTest(original, tuple(p), True) > max(2., separation * .10) for p in sample) >= 3:
            return None
    polygon = _visible_polygon(original, boxes)
    if polygon is None:
        return None
    x, y, w, h = cv2.boundingRect(polygon)
    x, y = max(0, x), max(0, y)
    local = classes[y:y + h, x:x + w]
    mask = np.zeros(local.shape, dtype="uint8")
    cv2.fillConvexPoly(mask, np.rint(polygon - [x, y]).astype("int32"), 1)
    inside = cv2.erode(mask, np.ones((3, 3), dtype="uint8"), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    area = int(inside.sum())
    if area < 28 or np.any(np.isin(local[inside], CROSSINGS)):
        return None
    interior = local[inside]
    # A small person, motorcycle or bollard can occupy very little of a large open rectangle.
    # Tolerate perimeter aliasing, but never absorb a coherent obstruction in the stall core.
    inset = max(2, int(np.ceil(separation * .10)))
    depth = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    obstruction = (depth > inset) & ~np.isin(local, GROUND)
    if np.any(obstruction):
        _, _, obstacles, _ = cv2.connectedComponentsWithStats(obstruction.astype("uint8"))
        if np.max(obstacles[1:, cv2.CC_STAT_AREA]) >= 4:
            return None
    parking = inside & (local == PARKING)
    _, _, patches, _ = cv2.connectedComponentsWithStats(parking.astype("uint8"))
    has_parking = (parking.sum() >= max(10, area * .022) and len(patches) >= 2
                   and patches[1:, cv2.CC_STAT_AREA].max() >= max(8, area * .015))
    if np.isin(interior, GROUND).mean() < .96 or (interior == LINE).mean() > .22:
        return None
    row_evidence = {}
    if measured_row:
        neighbors, caps, curb_ends = _row_support(first, second, row_anchors, classes)
        if not has_parking:
            # Negative road evidence only becomes an outlined bay when the terminal curb and
            # multiple signed paint dividers make an observable parking row, with an aisle end-cap.
            # An ordinary traffic lane, two parallel lines or a lone T junction cannot qualify.
            if not (neighbors and caps and curb_ends == 2 and min(
                    first["row_paint_class_support"], second["row_paint_class_support"]) >= .55):
                return None
        elif not (neighbors or (first["semantic"] and second["semantic"] and parking.sum() > .26 * area)):
            return None
        row_evidence = {"supporting_row_dividers":len(neighbors), "painted_aisle_endcaps":caps,
                        "curb_facing_terminations":int(curb_ends), "interior_has_parking_component":bool(has_parking),
                        "divider_semantic_paint_fraction":[round(v["row_paint_class_support"],3) for v in (first, second)]}
        if has_parking and parking.sum() > .46 * area and min(lengths) < .54 * max(lengths):
            polygon, observed = _observed_back_edge(polygon, boundaries, directions, classes, curb_distance, boxes)
            row_evidence["additional_visible_curb_cells"] = observed
    elif not has_parking:
        return None
    moments = cv2.moments(polygon)
    center = np.array([moments["m10"], moments["m01"]]) / moments["m00"]
    if int(_samples(classes, center[None, :])[0]) not in (PARKING, ROAD):
        return None
    # A white shared stripe is ambiguous even when the adjacent stall on its other side is
    # only partly visible. A single boundary may select a bay if another side is visibly closed.
    opposite_closed = []
    for line, ends in zip((first, second), boundaries):
        middle = ends.mean(axis=0)
        axis = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
        outward = _normal(axis) * np.sign((middle - center) @ _normal(axis))
        sample = np.linspace(*ends, 13)[2:-2] + outward * max(4., separation * .6)
        surface = _samples(classes, sample)
        occupied = np.zeros(len(sample), dtype=bool)
        for x0, y0, x1, y1 in boxes:
            occupied |= (sample[:, 0] >= x0) & (sample[:, 0] <= x1) & (sample[:, 1] >= y0) & (sample[:, 1] <= y1)
        opposite_closed.append(bool((~np.isin(surface, GROUND) | occupied).mean() >= .55))
    vertices = polygon * scale
    return {"polygon": _coordinates(vertices), "point": _coordinates(center * scale),
            "box": _coordinates(np.r_[vertices.min(axis=0), vertices.max(axis=0)]),
            "evidence": {"recipe": ROW_RECIPE if measured_row else RECIPE,
                         "painted_boundaries": [_coordinates(e * scale) for e in boundaries], **row_evidence,
                         "boundary_widths": [round(line["width"] * float(scale.mean()), 3) for line in (first, second)],
                         "boundary_opposite_closed": opposite_closed,
                         "paint_contrast_support": [round(line["contrast"], 3) for line in (first, second)],
                         "visible_parking_cells": int(parking.sum()), "interior_cells": area,
                         "curb_distance_cells": round(float(distances[near][1]), 3)}}


def candidates(original, classes, context):
    """Return visible, unoccupied stall polygons in original-image pixels in deterministic order."""
    if not isinstance(original, Image.Image) or tuple(context["size"]) != original.size:
        raise ValueError("The photograph must match the current original dimensions.")
    classes = np.asarray(classes)
    if classes.ndim != 2 or min(classes.shape) < 8 or classes.dtype != np.uint8:
        raise ValueError("Expected a two-dimensional native semantic class map.")
    if not np.any(np.isin(classes, (PARKING, LINE))) or not np.any(np.isin(classes, (2, 15))):
        return []
    if max(classes.shape) > 900:
        ratio = 900 / max(classes.shape)
        classes = cv2.resize(classes, (round(classes.shape[1] * ratio), round(classes.shape[0] * ratio)), interpolation=cv2.INTER_NEAREST)
    rows, cols = classes.shape
    scale = np.array(original.size, dtype=float) / [cols, rows]
    color = np.asarray(original.convert("RGB").resize((cols, rows), Image.Resampling.BOX)).astype("float32")
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    red, green, blue = color.transpose(2, 0, 1)
    neutral = (np.ptp(color, axis=2) < 55) | ((red > 1.12 * blue) & (green > 1.08 * blue) & (np.abs(red - green) < 45))
    parking_distance = cv2.distanceTransform((classes != PARKING).astype("uint8"), cv2.DIST_L2, 5)
    curb = np.isin(classes, (2, 15))
    curb_distance = cv2.distanceTransform((~curb).astype("uint8"), cv2.DIST_L2, 5)
    boxes = np.asarray(context["current_boxes"], dtype=float).reshape(-1, 4) / np.tile(scale, 2)
    lines = _dividers(gray, neutral, np.isin(classes, GROUND), classes, parking_distance)
    options = []
    for first, second in combinations(lines, 2):
        option = _pair(first, second, lines, classes, curb_distance, curb, boxes, scale)
        if option is not None:
            options.append(option)
    # Once a stall is already supported by the stricter two-divider parking-component route,
    # keep that measured polygon. The row detector may add different adjacent physical stalls.
    measured = _partial_dividers(gray, neutral, np.isin(classes, GROUND), classes, lines)
    anchors = _row_features(measured, classes, curb_distance)
    primary = [np.float32(option["polygon"]) for option in options]
    for first, second in combinations(measured, 2):
        option = _pair(first, second, measured, classes, curb_distance, curb, boxes, scale, anchors)
        if option is None:
            continue
        candidate = np.float32(option["polygon"])
        duplicate = any(cv2.intersectConvexConvex(candidate, existing)[0] > .12 * min(
            cv2.contourArea(candidate), cv2.contourArea(existing)) for existing in primary)
        if not duplicate:
            options.append(option)
    # An uncertain repeated divider or crossing rows can otherwise yield overlapping stalls for
    # a single tap. Distinct side-by-side stalls may only share the width of their paint.
    polygons = [np.float32(option["polygon"]) for option in options]
    conflicts = set()
    for i, j in combinations(range(len(polygons)), 2):
        smaller = min(cv2.contourArea(polygons[i]), cv2.contourArea(polygons[j]))
        if cv2.intersectConvexConvex(polygons[i], polygons[j])[0] > .12 * smaller:
            conflicts.update((i, j))
    return sorted((opt for index, opt in enumerate(options) if index not in conflicts), key=lambda opt: tuple(opt["point"]))


def matching_options(options, mark, center, frame, classes, *, offer_visible_side=False):
    """Bind a precise tap; optionally expose one visible side of genuine paint as a suggestion."""
    width, height = frame["width"], frame["height"]
    x0, y0, x1, y1 = mark
    x, y = center
    if (not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height
            or not np.allclose(center, [(x0 + x1) / 2, (y0 + y1) / 2], atol=1e-5, rtol=0)
            or (x1 - x0) / width > .055 or (y1 - y0) / height > .055 or classes is None):
        return []
    rows, columns = classes.shape
    cell = int(classes[min(rows - 1, int(y * rows / height)), min(columns - 1, int(x * columns / width))])
    if cell not in GROUND:
        return []
    p = np.asarray(center)
    scale = max(width / columns, height / rows)
    matches = []
    for option in options:
        inside = cv2.pointPolygonTest(np.asarray(option["polygon"], dtype="float32"), (float(x), float(y)), True)
        evidence = option["evidence"]
        boundary = None
        on_paint = False
        for index, (ends, stroke_width) in enumerate(zip(evidence["painted_boundaries"], evidence["boundary_widths"])):
            a, b = np.asarray(ends)
            axis = b - a
            t = (p - a) @ axis / (axis @ axis)
            tolerance = stroke_width / 2 + scale * .8
            distance = np.linalg.norm(p - (a + t * axis))
            if 0 <= t <= 1 and distance <= tolerance and inside >= -tolerance:
                boundary = index
                on_paint = cell == LINE or distance <= stroke_width / 2
                break
        if inside >= 0 or boundary is not None:
            matches.append((option, boundary, on_paint))
    if len(matches) == 1:
        selected, boundary, on_paint = matches[0]
        if boundary is not None and not selected["evidence"]["boundary_opposite_closed"][boundary]:
            if not (offer_visible_side and on_paint):
                return []
    return [option for option, _, _ in matches]


def match_mark(options, mark, center, frame, classes):
    matches = matching_options(options, mark, center, frame, classes)
    return matches[0] if len(matches) == 1 else None
