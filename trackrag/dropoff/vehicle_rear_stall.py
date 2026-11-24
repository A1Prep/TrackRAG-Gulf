"""Link two independently measured parking dividers to the rear of their actual vehicle."""

from hashlib import sha256
from itertools import combinations

import cv2
import numpy as np

from . import parking_row_bays, single_vehicle, vehicle_rear_boundaries


RECIPE = "visible_marked_compartment_behind_rear_vehicle_v1"


def _borders(view, scene):
    """Reuse the established Task 5 detector's traced paint, aisle and kerb checks."""
    gray, labels, original_scale = parking_row_bays._work_view(view.original, view.classes)
    context = single_vehicle.context({"scene": scene})
    boxes = np.asarray([box for _, _, box in context["tracks"]], dtype=float).reshape(-1, 4)
    boxes /= np.tile(original_scale, 2)
    to_native = np.array([view.classes.shape[1] / labels.shape[1], view.classes.shape[0] / labels.shape[0]])
    return [
        (stem["front"] * to_native, stem["back"] * to_native)
        for stem in parking_row_bays._semantic_stems(gray, labels, boxes)
        if stem["back_curb"] and not stem["front_hood"]
    ]


def _interior(view, polygon):
    """A point must have clear, parking-class ground; a visible terminal line alone proves no destination."""
    centre = polygon.mean(axis=0)
    region = np.zeros_like(view.classes, np.uint8)
    trimmed = np.rint(centre + 0.83 * (polygon - centre)).astype(np.int32)
    cv2.fillConvexPoly(region, trimmed, 1)
    mask = region.astype(bool)
    if not mask.any() or view.occupied[mask].any() or not view.road[mask].all():
        return None
    fraction = float(np.mean(view.classes[mask] == 10))
    if fraction < 0.76:
        return None
    radius = max(2, round(0.055 * min(view.vehicle_size)))
    samples = []
    for depth in np.linspace(0.22, 0.73, 14):
        aisle = (1 - depth) * polygon[0] + depth * polygon[3]
        kerb = (1 - depth) * polygon[1] + depth * polygon[2]
        for across in (0.45, 0.50, 0.55):
            x, y = np.rint((1 - across) * aisle + across * kerb).astype(int)
            if not vehicle_rear_boundaries._clear_patch(view, x, y, radius):
                continue
            if np.all(view.classes[y - radius : y + radius + 1, x - radius : x + radius + 1] == 10):
                distance = np.linalg.norm([x - centre[0], y - centre[1]])
                samples.append((distance, int(x), int(y)))
    if not samples:
        return None
    _, x, y = min(samples)
    return x, y, radius, fraction


def _behind(view, first, second):
    """The two photographed transverse lines must define the first compartment at this bumper depth."""
    cx, bottom = view.bumper
    width, height = view.vehicle_size
    first, second = sorted(
        (first, second), key=lambda endpoints: np.mean([point[1] for point in endpoints])
    )
    polygon = np.float32([first[0], first[1], second[1], second[0]])
    near, far = polygon[:2].mean(axis=0), polygon[2:].mean(axis=0)
    first_edge, second_edge = polygon[1] - polygon[0], polygon[2] - polygon[3]
    lengths = np.linalg.norm(first_edge), np.linalg.norm(second_edge)
    if not all(lengths) or np.dot(first_edge, second_edge) / np.prod(lengths) < 0.93:
        return None
    if (
        abs(first_edge[1]) / max(1, abs(first_edge[0])) > 0.55
        or not 0.58 * width < lengths[0] < 2.7 * width
    ):
        return None
    separation = far - near
    side = 1 if first_edge[0] > 0 else -1
    if not 0.25 * height <= separation[1] <= 1.45 * height or not 1.02 <= lengths[1] / lengths[0] <= 2.45:
        return None
    if (
        not 0.20 < side * separation[0] / separation[1] < 5.8
        or not -0.14 * height <= near[1] - bottom <= 0.33 * height
    ):
        return None
    fraction = (cx - first[0][0]) / (first[1][0] - first[0][0])
    if not 0.08 <= fraction <= 0.92 or not cv2.isContourConvex(polygon):
        return None
    checked = _interior(view, polygon)
    if not checked:
        return None
    return side, polygon, checked


def _compartment(view, scene):
    classes = view.classes
    cx, bottom = view.bumper
    width, height = view.vehicle_size
    h, w = classes.shape
    near = classes[
        max(0, int(bottom - 0.12 * height)) : min(h, int(bottom + 1.3 * height)),
        max(0, int(cx - 2.3 * width)) : min(w, int(cx + 2.3 * width)),
    ]
    if np.count_nonzero(near == 10) < 0.06 * width * height:
        return []
    borders = _borders(view, scene)
    found = []
    for first, second in combinations(borders, 2):
        choice = _behind(view, first, second)
        if choice is None:
            continue
        polygon = choice[1]
        # A nearer measured crossbar splits the area into different stalls. Never jump over it.
        if any(
            other is not first
            and other is not second
            and all(
                cv2.pointPolygonTest(polygon, tuple(point), False) > 0
                for point in np.linspace(*other, 5)[1:-1].astype(float)
            )
            for other in borders
        ):
            continue
        found.append(choice)
    if len(found) > 1:
        return None
    if not found:
        return []
    side, polygon, (x, y, radius, fraction) = found[0]
    from . import traffic_arrow

    # The two outlines include the entire parking compartment in this check, including space away from the pin.
    sections = []
    for yprobe in range(int(polygon[:, 1].min()) + 1, int(polygon[:, 1].max())):
        intersections = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
            if min(a[1], b[1]) <= yprobe < max(a[1], b[1]):
                intersections.append(float(a[0] + (yprobe - a[1]) * (b[0] - a[0]) / (b[1] - a[1])))
        if len(intersections) == 2 and abs(intersections[1] - intersections[0]) > 0.50 * width:
            lane, kerb = sorted(intersections, reverse=side < 0)
            sections.append(vehicle_rear_boundaries._Section(yprobe, lane, kerb))
    if len(sections) < 3 or traffic_arrow.in_measured_strip(
        view.rgb, classes, sections, view.spec["box"], view.original.size, side
    ):
        return None
    evidence = {
        "physical_boundary": "two_measured_transverse_aisle_to_kerb_dividers",
        "measured_interior_parking_fraction": round(fraction, 4),
        "source_dividers_xy": np.round(polygon[[0, 1, 3, 2]].reshape(2, 2, 2) / view.scale, 3).tolist(),
    }
    return [(side, x, y, radius, evidence)]


def candidates(original, classes, scene, spec, *, facing):
    """Return a clear pictured point in the marked row immediately behind this verified rear."""
    from . import vehicle_rear_end_marks

    view = vehicle_rear_boundaries._view(original, classes, scene, spec, facing)
    if view is None:
        return []
    compartment = _compartment(view, scene)
    if compartment is None:
        return []
    choice = compartment[0] if compartment else vehicle_rear_end_marks.measure(view)
    if choice is None:
        return []
    side, x, y, radius, physical = choice
    h, w = classes.shape
    centre, low, high = np.round(
        np.array(
            [[x + 0.5, y + 0.5], [x + 0.5 - radius, y + 0.5 - radius], [x + 0.5 + radius, y + 0.5 + radius]]
        )
        / view.scale,
        3,
    ).tolist()
    left, top = low
    right, bottom = high
    evidence = {
        "recipe": RECIPE,
        **physical,
        "coverage": "interior_point_only",
        "anchor_id": spec["track_id"],
        "verified_facing": facing,
        "curb_side": "left" if side < 0 else "right",
        "current_tracks_clear": True,
        "travel_arrow_checked": True,
        "point_surface": int(classes[y, x]),
        "surface_map_size": [w, h],
        "source_sha256": scene["provenance"]["image_sha256"],
        "rgb_sha256": sha256(view.rgb.tobytes()).hexdigest(),
        "surface_sha256": sha256(np.ascontiguousarray(classes).tobytes()).hexdigest(),
    }
    return [
        {
            "point": centre,
            "box": [left, top, right, bottom],
            "polygon": [[left, top], [right, top], [right, bottom], [left, bottom]],
            "evidence": evidence,
        }
    ]
