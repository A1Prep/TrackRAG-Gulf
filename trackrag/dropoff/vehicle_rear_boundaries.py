"""Find pictured ground behind a rear-facing vehicle along a measured parking boundary."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from . import bay_clearance, single_vehicle, traffic_arrow, vehicle_rear_apron, vehicle_rear_ground

RECIPE = "rear_ground_between_visible_parking_edge_and_kerb_v1"
GROUND = (10, 13)
KERB = (2, 15)


@dataclass
class _View:
    original: Image.Image
    rgb: np.ndarray
    classes: np.ndarray
    road: np.ndarray
    occupied: np.ndarray
    spec: dict
    scale: np.ndarray
    box: np.ndarray

    @property
    def bumper(self):
        left, _, right, bottom = self.box
        return (left + right) / 2, bottom

    @property
    def vehicle_size(self):
        left, top, right, bottom = self.box
        return right - left, bottom - top


@dataclass(frozen=True)
class _Section:
    y: int
    lane: float
    curb: float


def _view(original, classes, scene, spec, facing, *, profile=False):
    ends = ("REAR","SIDE") if profile else ("REAR",)
    if facing not in ends or scene.get("source") != "predicted" or not isinstance(spec, dict):
        return None
    if not isinstance(classes, np.ndarray) or classes.ndim != 2 or classes.dtype != np.uint8:
        return None
    if isinstance(original, bytes):
        if sha256(original).hexdigest() != scene.get("provenance", {}).get("image_sha256"):
            return None
        with Image.open(BytesIO(original)) as source:
            original = source.convert("RGB")
    frame = scene["frame"]
    if not isinstance(original, Image.Image) or original.size != (frame["width"], frame["height"]):
        return None
    context = single_vehicle.context({"scene": scene})
    if not any(
        track == spec.get("track_id") and kind == spec.get("class") and list(box) == spec.get("box")
        for track, kind, box in single_vehicle.motor_rows(context)
    ) or vehicle_rear_apron._anchor_shifted(scene, spec):
        return None
    h, w = classes.shape
    scale = np.array([w / original.width, h / original.height])
    box = np.asarray(spec["box"]) * np.tile(scale, 2)
    width, height = box[2:] - box[:2]
    low, high = (1.65,4.4) if profile else (.62,2.15)
    if width < max(18, 0.04 * w) or height < max(10, 0.04 * h) or not low < width / height < high:
        return None
    road = bay_clearance.visible_road_component(classes)
    if road is None:
        return None
    occupied = np.zeros_like(classes, bool)
    for _, _, current in context["tracks"]:
        left, top, right, bottom = np.asarray(current) * np.tile(scale, 2)
        occupied[
            max(0, int(top)) : min(h, int(np.ceil(bottom))), max(0, int(left)) : min(w, int(np.ceil(right)))
        ] = True
    rgb = np.array(original.convert("RGB").resize((w, h), Image.Resampling.BOX))
    return _View(original, rgb, classes, road, occupied, spec, scale, box)


def _paint_centres(view, row_range):
    classes = view.classes
    cx, _ = view.bumper
    width, _ = view.vehicle_size
    start, end = max(0, round(cx - 2 * width)), min(classes.shape[1], round(cx + 2 * width))
    result = {}
    for y in row_range:
        runs = vehicle_rear_ground._runs(np.flatnonzero(classes[y, start:end] == 24) + start)
        result[y] = [(0.5 * (a + b), a, b) for a, b in runs if b - a + 1 <= max(5, 0.25 * width)]
    return result


def _paint_lines(view, stripes, side):
    """Fit persistent paint paths nearby and require measurements both at and beyond the bumper."""
    cx, bottom = view.bumper
    width, height = view.vehicle_size
    rows = list(stripes)
    sampled = rows[:: max(2, round(0.045 * height))]
    hypotheses = {}
    error = max(3, 0.045 * width)
    for y in sampled:
        for yy in sampled:
            if not 0.29 * height <= yy - y <= 0.94 * height:
                continue
            for x, _, _ in stripes[y]:
                for xx, _, _ in stripes[yy]:
                    slope = (xx - x) / (yy - y)
                    intercept = x + (bottom - y) * slope
                    if not 0.22 < side * slope < 5.5 or not 0.13 < -side * (intercept - cx) / width < 0.99:
                        continue
                    points = []
                    for sy in rows:
                        prediction = (sy - bottom) * slope + intercept
                        close = [
                            (abs(sx - prediction), sy, sx, a, b)
                            for sx, a, b in stripes[sy]
                            if abs(sx - prediction) < error
                        ]
                        if close:
                            points.append(min(close)[1:])
                    if len(points) < 0.34 * height:
                        continue
                    ys, xs = np.asarray([point[:2] for point in points]).T
                    if min(ys) > bottom + 0.07 * height or max(ys) < bottom + 0.21 * height:
                        continue
                    slope, intercept = np.polyfit(ys - bottom, xs, 1)
                    if np.percentile(abs(xs - ((ys - bottom) * slope + intercept)), 85) > max(
                        2, 0.033 * width
                    ):
                        continue
                    key = round(intercept / max(4, 0.10 * width)), round(slope / 0.2)
                    if key not in hypotheses or len(points) > len(hypotheses[key][0]):
                        hypotheses[key] = (points, slope, intercept)
    return sorted(hypotheses.values(), key=lambda item: -len(item[0]))[:10]


def _outer_kerb(view, paint_x, y, side):
    """Find the separate outer sidewalk/kerb and record whether its edge is visible in the RGB photo."""
    classes = view.classes
    width, _ = view.vehicle_size
    w = classes.shape[1]
    start = round(paint_x)
    if not 0 <= start < w:
        return None
    stop = min(w - 1, max(0, start + side * round(1.90 * width)))
    positions = np.arange(start, stop, side)
    values = classes[y, positions]
    outside = np.flatnonzero(np.isin(values, KERB))
    if not len(outside):
        return None
    span = int(outside[0])
    edge = int(positions[span])
    exterior = values[span : span + max(5, round(0.13 * width))]
    if (
        not 0.58 * width < span < 1.85 * width
        or len(exterior) < 2
        or np.mean(np.isin(exterior, KERB)) < 0.62
    ):
        return None
    if (values[round(0.10 * width) : span] == 24).sum() > 0.12 * width:
        return None
    ground_x = int(np.clip(round(edge - side * 0.10 * width), 0, w - 1))
    reach = round(0.09 * width)
    left = view.rgb[y, max(0, edge - reach) : max(0, edge - 1)]
    right = view.rgb[y, min(w, edge + 2) : min(w, edge + reach + 1)]
    optical = (
        classes[y, ground_x] in GROUND
        and len(left) > 0
        and len(right) > 0
        and np.linalg.norm(np.median(left, axis=0) - np.median(right, axis=0)) >= 11
    )
    return int(y), edge, span, bool(optical)


def _repeated_white_paint(view, paint):
    _, bottom = view.bumper
    width, height = view.vehicle_size
    bright = [
        int(y)
        for y, _, a, b in paint
        if y > bottom - 0.11 * height and vehicle_rear_ground._white_paint(view.rgb, y, a, b, width)
    ]
    if len(bright) < max(4, 0.10 * height) or bright[-1] - bright[0] < 0.19 * height:
        return []
    pieces = [
        group
        for group in np.split(bright, np.flatnonzero(np.diff(bright) > 1) + 1)
        if len(group) >= max(2, 0.025 * height)
    ]
    if not any(right[0] - left[-1] >= max(2, 0.03 * height) for left, right in zip(pieces, pieces[1:])):
        return []
    return [[int(group[0]), int(group[-1])] for group in pieces]


def _parking_role(view, run, side, slope, intercept):
    """A roadside through lane can also have a kerb and stripe. Seek separate parking evidence."""
    classes = view.classes
    h, w = classes.shape
    width, height = view.vehicle_size
    _, bottom = view.bumper
    fractions = []
    for y, x, lane, curb in run:
        positions = np.arange(max(0, round(min(lane, curb))), min(w, round(max(lane, curb)) + 1))
        middle = positions[np.abs(positions - x) < 0.22 * abs(lane - curb)]
        if len(middle):
            fractions.append(np.mean(classes[int(y), middle] == 10))
    if fractions and np.median(fractions) > 0.70:
        return {"native_parking_fraction": round(float(np.median(fractions)), 3)}
    # Paired kerbside parking edges can be white dashes with the middle of the carriageway marked amber.
    # Require both paint features on the photographed road-facing side, beyond the proposed parking strip.
    hsv = cv2.cvtColor(view.rgb, cv2.COLOR_RGB2HSV)
    positions = np.arange(w)
    amber = []
    opposite = []
    for y in range(max(1, round(bottom - 0.40 * height)), min(h, round(bottom + 0.70 * height))):
        lane = intercept + (y - bottom) * slope
        beyond = (positions - lane) * (-side) / width
        paint = (classes[y] == 24) & ~view.occupied[y]
        warm = (hsv[y, :, 0] >= 8) & (hsv[y, :, 0] <= 42) & (hsv[y, :, 1] > 40) & (hsv[y, :, 2] >= 70)
        centre = positions[paint & warm & (beyond > 0.55) & (beyond < 2.9)]
        if not len(centre):
            continue
        centre_x = float(np.median(centre))
        amber.append((y, round(centre_x, 1)))
        far_side = (positions - centre_x) * (-side) / width
        white = (hsv[y, :, 1] < 67) & (hsv[y, :, 2] > 53)
        distant = positions[paint & white & (far_side > 0.48) & (far_side < 2.8)]
        if len(distant):
            opposite.append((y, round(float(np.median(distant)), 1)))
    if (
        len(amber) < max(4, 0.07 * height)
        or len(opposite) < max(4, 0.07 * height)
        or amber[-1][0] - amber[0][0] < 0.17 * height
    ):
        return None
    return {
        "amber_centre_rows": len(amber),
        "opposite_boundary_rows": len(opposite),
        "amber_extent_y": [amber[0][0], amber[-1][0]],
    }


def _clear_rows(view, row_range, kerbs, slope, intercept, side, observed_width):
    classes = view.classes
    _, w = classes.shape
    cx, bottom = view.bumper
    width, height = view.vehicle_size
    if abs(intercept + 0.5 * side * observed_width - cx) > 0.33 * width:
        return []
    rows = []
    for y in range(
        int(np.ceil(bottom + 0.035 * height)), min(row_range.stop, round(bottom + 0.59 * height))
    ):
        lane = intercept + (y - bottom) * slope
        nearby = next((sample for sample in kerbs if abs(sample[0] - y) < 2), None)
        local_width = nearby[2] if nearby else observed_width
        x = int(round(lane + 0.5 * side * local_width))
        half_width = max(2, round(0.16 * local_width))
        low, high = x - half_width, x + half_width
        if low < 1 or high >= w - 1:
            continue
        interior = np.s_[y, low : high + 1]
        if (
            not np.isin(classes[interior], GROUND).all()
            or not view.road[interior].all()
            or view.occupied[interior].any()
        ):
            continue
        road_a, road_b = sorted((round(lane - side * 0.22 * width), round(lane - side * 0.4 * width)))
        if road_a < 1 or road_b >= w - 1 or np.mean(classes[y, road_a : road_b + 1] == 13) < 0.88:
            continue
        rows.append((y, x, lane, lane + side * local_width))
    if len(rows) < max(6, 0.16 * height):
        return []
    cuts = np.flatnonzero(np.diff([row[0] for row in rows]) > 1) + 1
    for run in np.split(np.asarray(rows), cuts):
        if len(run) >= 0.15 * height and run[0, 0] < bottom + 0.12 * height:
            return run.tolist()
    return []


def _clear_patch(view, x, y, r):
    h, w = view.classes.shape
    if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
        return False
    patch = np.s_[y - r : y + r + 1, x - r : x + r + 1]
    return bool(
        np.isin(view.classes[patch], GROUND).all()
        and view.road[patch].all()
        and not view.occupied[patch].any()
    )


def _dashed(view):
    h, _ = view.classes.shape
    _, bottom = view.bumper
    width, height = view.vehicle_size
    rows = range(max(0, int(bottom - 0.42 * height)), min(h, int(bottom + 0.68 * height)))
    stripes = _paint_centres(view, rows)
    options = []
    for side in (-1, 1):
        for paint, slope, intercept in _paint_lines(view, stripes, side):
            dashes = _repeated_white_paint(view, paint)
            if not dashes:
                continue
            kerbs = [
                k
                for y in rows
                if (k := _outer_kerb(view, intercept + (y - bottom) * slope, y, side)) is not None
            ]
            if len(kerbs) < max(4, 0.09 * height) or not any(k[0] > bottom - 0.1 * height for k in kerbs):
                continue
            optical = [(k[0], k[1]) for k in kerbs if k[3]]
            if len(optical) < 3:
                continue
            width_evidence = [k[2] for k in kerbs if k[0] > bottom - 0.14 * height] or [k[2] for k in kerbs]
            run = _clear_rows(view, rows, kerbs, slope, intercept, side, float(np.median(width_evidence)))
            if not run or not (role := _parking_role(view, run, side, slope, intercept)):
                continue
            sections = [_Section(int(y), lane, curb) for y, _, lane, curb in run]
            if traffic_arrow.in_measured_strip(
                view.rgb, view.classes, sections, view.spec["box"], view.original.size, side
            ):
                continue
            y, x, _, _ = min(run, key=lambda row: abs(row[0] - (bottom + 0.26 * height)))
            x, y = int(x), int(y)
            radius = max(2, round(0.043 * min(width, height)))
            if _clear_patch(view, x, y, radius):
                evidence = {
                    "physical_boundary": "repeated_white_paint_and_observed_kerb",
                    "white_dash_runs_y": dashes,
                    "registered_optical_kerb_points_yx": optical,
                    "parking_evidence": role,
                    "ground_run_y": [int(run[0][0]), int(run[-1][0])],
                    "observed_stripe_at_bumper": round(float(intercept), 3),
                    "stripe_dx_per_y": round(float(slope), 4),
                }
                options.append((len(run), side, x, y, radius, evidence))
    return options


def candidates(original, classes, scene, spec, *, facing):
    """Return one source-pixel interior patch on measured ground, independently of the request's wording."""
    view = _view(original, classes, scene, spec, facing)
    if view is None:
        return []
    options = _dashed(view)
    if not options:
        from .vehicle_rear_paving import measure

        options = measure(view)
    if not options or len({option[1] for option in options}) != 1:
        return []
    _, side, x, y, radius, evidence = max(options, key=lambda option: option[0])
    points = (
        np.array(
            [[x + 0.5, y + 0.5], [x + 0.5 - radius, y + 0.5 - radius], [x + 0.5 + radius, y + 0.5 + radius]]
        )
        / view.scale
    )
    centre, low, high = np.round(points, 3).tolist()
    left, top = low
    right, bottom = high
    evidence.update(
        recipe=RECIPE,
        coverage="interior_point_only",
        curb_side="left" if side < 0 else "right",
        anchor_id=spec["track_id"],
        verified_facing=facing,
        current_tracks_clear=True,
        point_surface=int(classes[y, x]),
        travel_arrow_checked=True,
        surface_map_size=[classes.shape[1], classes.shape[0]],
        source_sha256=scene["provenance"]["image_sha256"],
        registered_rgb_sha256=sha256(view.rgb.tobytes()).hexdigest(),
        map_sha256=sha256(np.ascontiguousarray(classes).tobytes()).hexdigest(),
    )
    return [
        {
            "point": centre,
            "box": [left, top, right, bottom],
            "polygon": [[left, top], [right, top], [right, bottom], [left, bottom]],
            "evidence": evidence,
        }
    ]
