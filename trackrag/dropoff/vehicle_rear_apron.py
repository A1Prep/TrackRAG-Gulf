"""Measure a small patch of visibly separated parking directly behind one verified vehicle."""

from hashlib import sha256
from io import BytesIO
from math import ceil, cos, pi, sin
from typing import NamedTuple
import cv2
import numpy as np
from PIL import Image

from . import bay_clearance, single_vehicle, traffic_arrow

RECIPE = "single_rear_bumper_apron_with_visible_boundaries_v2"


class _ArrowSearchRow(NamedTuple):
    y: int
    lane: float
    curb: float


def _travel_arrow(rgb, classes, parking_component, anchor, size, box):
    """Look for a painted arrow around the proposed bumper corridor on the same parking surface."""
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    # This envelope only excludes a candidate. It never supplies a parking boundary.
    rows = [
        _ArrowSearchRow(round(y1 + height * fraction), cx - width, cx + width)
        for fraction in (0.12, 0.50, 0.87)
    ]
    reach = max(5, round(0.25 * width)) | 1
    adjacent = cv2.dilate(parking_component.astype(np.uint8), np.ones((reach, reach), np.uint8)).astype(
        bool
    )
    local_labels = classes.copy()
    local_labels[(classes == 24) & ~adjacent] = 13
    return traffic_arrow.in_measured_strip(rgb, local_labels, rows, anchor["box"], size, 1)


def _anchor_shifted(scene, anchor):
    """Withhold a visibly displaced recent track. A single photograph supplies no motion reading."""
    frame = scene["frame"]
    x0, y0, x1, y1 = anchor["box"]
    width, height = x1 - x0, y1 - y0
    recent = [
        row
        for row in scene["tracks"]
        if row["track_id"] == anchor["track_id"]
        and row.get("recording_id", frame["recording_id"]) == frame["recording_id"]
        and 0 < frame["frame_id"] - row["frame_id"] <= 2
    ]
    for row in recent:
        a, b, c, d = row["bbox_xyxy"]
        if (
            abs(a + c - x0 - x1) > 0.75 * width
            or abs(b + d - y0 - y1) > 0.75 * height
            or c - a < 0.65 * width
            or c - a > 1.45 * width
        ):
            return True
    return False


def _flank_kerb(rgb, classes, component, occupied, box):
    """Look round the car on the *same parking component* for a kerb hidden from the point."""
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0
    cx = (x0 + x1) / 2
    h, w = component.shape
    left = max(0, ceil(cx - 1.45 * bw))
    right = min(w, int(cx + 1.45 * bw) + 1)
    top = max(0, int(y0 + 0.22 * bh) + 1)
    bottom = min(h, ceil(y1 + 0.68 * bh))
    yy, xx = np.nonzero(component[top:bottom, left:right])
    points = (yy + top, xx + left)
    matches = set()
    physical = set()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        yy, xx = points
        reach = max(3, round(0.07 * bw))
        limits = (xx >= reach) & (xx < w - reach) & (yy >= reach) & (yy < h - reach)
        yy, xx = yy[limits], xx[limits]
        if not len(yy):
            continue
        adjacent = np.isin(classes[yy + dy, xx + dx], (2, 15))
        yy, xx = yy[adjacent], xx[adjacent]
        if not len(yy):
            continue
        offsets = np.arange(1, reach + 1)[None, :]
        near_y = yy[:, None] - dy * offsets
        near_x = xx[:, None] - dx * offsets
        far_y = yy[:, None] + dy * offsets
        far_x = xx[:, None] + dx * offsets
        valid = (component[near_y, near_x].mean(axis=1) >= 0.8) & (
            np.isin(classes[far_y, far_x], (2, 15)).mean(axis=1) >= 0.65
        )
        valid &= ~occupied[far_y, far_x].any(axis=1)
        for px, py in zip(xx[valid], yy[valid]):
            physical.add((int(px), int(py)))
        near = np.median(rgb[near_y, near_x], axis=1)
        far = np.median(rgb[far_y, far_x], axis=1)
        visible = valid & (np.linalg.norm(near - far, axis=1) >= 24)
        for px, py in zip(xx[visible], yy[visible]):
            matches.add((int(px), int(py)))
    if len(matches) < max(8, 0.2 * bw):
        return None
    patch = np.zeros((h, w), np.uint8)
    ax, ay = np.array(list(physical)).T
    patch[ay, ax] = 1
    px, py = np.array(list(matches)).T
    components = cv2.connectedComponents(cv2.dilate(patch, np.ones((3, 3), np.uint8)), connectivity=8)[1]
    for group in np.unique(components[py, px]):
        ids = components[py, px] == group
        native_count = np.count_nonzero(components[ay, ax] == group)
        if (
            ids.sum() >= max(7, 0.14 * bw)
            and ids.sum() / native_count >= 0.5
            and np.hypot(np.ptp(px[ids]), np.ptp(py[ids])) >= 0.27 * bw
        ):
            return {
                "sample_count": int(ids.sum()),
                "native_bounds": [
                    int(min(px[ids])),
                    int(min(py[ids])),
                    int(max(px[ids])),
                    int(max(py[ids])),
                ],
            }
    return None


def _ray_boundaries(rgb, classes, occupied, access, point, bw, flank_kerb=None):
    h, w = classes.shape
    x, y = point
    step = np.arange(1, max(8, round(1.9 * bw)), dtype=float)
    contacts = []
    for index in range(40):
        theta = 2 * pi * index / 40
        xx = np.rint(x + step * cos(theta)).astype(int)
        yy = np.rint(y + step * sin(theta)).astype(int)
        clip = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        xx, yy = xx[clip], yy[clip]
        if len(xx) < 7:
            continue
        types = classes[yy, xx]
        cars = occupied[yy, xx]
        flags = np.flatnonzero((types != 10) | cars)
        if not len(flags):
            continue
        border = int(flags[0])
        if border < max(4, 0.15 * bw) or cars[border] or not access[yy[:border], xx[:border]].all():
            continue
        end = border
        while end < len(types) and types[end] == 24 and not cars[end]:
            end += 1
        if end + 4 >= len(types) or cars[border : end + 4].any() or end - border > 0.23 * bw:
            continue
        outer = int(types[end])
        if outer == 13:
            look = types[end : min(len(types), end + max(4, round(0.22 * bw)))]
            if np.mean(look == 13) < 0.87:
                continue
            target = "road"
        elif outer in (2, 15):
            look = types[end : min(len(types), end + max(4, round(0.22 * bw)))]
            if np.mean(np.isin(look, (2, 15))) < 0.70:
                continue
            target = "curb"
        else:
            continue
        colours = rgb[yy, xx].astype(float)
        reach = max(4, round(0.11 * bw))
        gap = max(1, round(0.025 * bw))
        if border - reach < 0 or end + reach >= len(xx):
            continue
        before = np.median(colours[border - reach : border - gap], axis=0)
        after = np.median(colours[end + gap : end + reach], axis=0)
        ridge = bool(
            end > border
            and colours[border:end].mean(axis=1).max() > max(before.mean(), after.mean()) + 10
        )
        if not ridge and np.linalg.norm(before - after) < 25:
            continue
        contacts.append((index, target, ridge, int(xx[border]), int(yy[border])))

    def span(kind):
        seq = {i for i, t, *_ in contacts if t == kind}
        return max(
            (
                sum(1 for length in range(1, 7) if all((i + o) % 40 in seq for o in range(length)))
                for i in seq
            ),
            default=0,
        )

    road = [c for c in contacts if c[1] == "road"]
    # A coherent stretch on each side is necessary, not arbitrary isolated mask pixels.
    painted_indices = {c[0] for c in road if c[2]}
    continuous_paint = any(
        all((i + k) % 40 in painted_indices for k in range(3)) for i in painted_indices
    )
    if span("road") < 3 or (span("curb") < 3 and not continuous_paint and not flank_kerb):
        return None
    angles = [i for i, t, *_ in contacts if t == "road"]
    return {
        "road_rays": len(angles),
        "kerb_rays": sum(c[1] == "curb" for c in contacts),
        "painted_road_rays": sum(c[2] for c in road),
        "road_arc": span("road"),
        "kerb_arc": span("curb"),
        "boundary_samples": [list(c) for c in contacts],
        "flank_kerb": flank_kerb,
    }


def _visible_barrier(rgb, cx, bottom, bw, bh):
    """A dark, thin chain/rail traversing the apparent parking strip invalidates the visual clearance."""
    h, w = rgb.shape[:2]
    left, right = max(0, round(cx - 0.4 * bw)), min(w, round(cx + 0.4 * bw))
    top, near = max(0, round(bottom + 0.09 * bh)), min(h, round(bottom + 0.83 * bh))
    if near - top < 6 or right - left < 8:
        return False
    gray = cv2.cvtColor(rgb[top:near, left:right], cv2.COLOR_RGB2GRAY)
    size = max(5, min(15, round(0.15 * bw)) | 1)
    dark = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats((dark > 28).astype(np.uint8), connectivity=8)
    for a, b, dw, dh, area in stats[1:]:
        slender = dh <= max(4, 0.075 * bh) or area / (dw * dh) < 0.53
        if a < 0.23 * bw and a + dw > 0.58 * bw and dw > 0.49 * bw and slender:
            return True
    return False


def candidates(original, classes, scene, spec, *, facing):
    """Return one supported, passenger-selectable interior point or an empty list.

    The caller supplies source-bound identity and facing. An identity-only reader's
    absent ground output contributes no positive or negative vote here.
    """
    anchor = spec
    if (
        facing != "REAR"
        or scene.get("source") != "predicted"
        or not isinstance(anchor, dict)
        or not isinstance(classes, np.ndarray)
        or classes.ndim != 2
        or classes.dtype != np.uint8
    ):
        return []
    if isinstance(original, bytes):
        if sha256(original).hexdigest() != scene.get("provenance", {}).get("image_sha256"):
            return []
        with Image.open(BytesIO(original)) as im:
            return candidates(im, classes, scene, anchor, facing=facing)
    frame = scene["frame"]
    h, w = classes.shape
    if not isinstance(original, Image.Image) or original.size != (frame["width"], frame["height"]):
        return []
    scale = np.array([w / original.width, h / original.height])
    ctx = single_vehicle.context({"scene": scene})
    if not any(
        t == anchor.get("track_id") and k == anchor.get("class") and list(b) == anchor.get("box")
        for t, k, b in single_vehicle.motor_rows(ctx)
    ):
        return []
    if _anchor_shifted(scene, anchor):
        return []
    a, b, c, d = np.array(anchor["box"]) * np.tile(scale, 2)
    bw, bh = c - a, d - b
    cx = (a + c) / 2
    if bw < max(16, 0.045 * w) or bh < max(14, 0.042 * h) or not 0.7 <= bw / bh <= 1.72:
        return []
    if not single_vehicle.behind_patch(
        classes, anchor["box"], [b for _, _, b in ctx["tracks"]], original.size
    ):
        return []
    access = bay_clearance.visible_road_component(classes)
    if access is None:
        return []
    occupied = np.zeros_like(classes, bool)
    for _, _, box in ctx["tracks"]:
        x0, y0, x1, y1 = np.array(box) * np.tile(scale, 2)
        occupied[int(y0) : int(np.ceil(y1)), int(x0) : int(np.ceil(x1))] = True
    rgb = np.asarray(original.convert("RGB").resize((w, h), Image.Resampling.BOX))
    if _visible_barrier(rgb, cx, d, bw, bh):
        return []
    available = (classes == 10) & access & ~occupied
    labels = cv2.connectedComponents(available.astype(np.uint8), connectivity=8)[1]
    curb_cache = {}
    arrow_cache = {}
    # The chosen ground stays directly beneath the named bumper. A region diagonally under the next car is irrelevant.
    for depth in (0.45, 0.36, 0.55, 0.26, 0.66):
        for lateral in (0, -0.10, 0.10, -0.18, 0.18):
            x = int(round(cx + lateral * bw))
            y = int(round(d + depth * bh))
            r = max(3, round(0.075 * min(bh, bw)))
            if (
                x - r < 0
                or y - r < 0
                or x + r >= w
                or y + r >= h
                or not available[y - r : y + r + 1, x - r : x + r + 1].all()
            ):
                continue
            start = int(ceil(d + 1))
            narrow = max(3, round(0.10 * bw))
            yy = np.arange(start, y + 1)
            xx = np.rint(cx + (x - cx) * (yy - d) / (y - d)).astype(int)
            lanes = xx[:, None] + np.arange(-narrow, narrow + 1, 2)[None, :]
            if (
                len(yy) < 5
                or occupied[yy[:, None], lanes].any()
                or not np.isin(classes[yy[:, None], lanes], (10, 13)).all()
            ):
                continue
            cross = available[yy[:, None], lanes].mean(axis=1)
            if np.mean(cross >= 0.8) < 0.90 or (cross == 0).any():
                continue
            component = int(labels[y, x])
            if component not in arrow_cache:
                arrow_cache[component] = _travel_arrow(
                    rgb, classes, labels == component, anchor, original.size, (a, b, c, d)
                )
            if arrow_cache[component]:
                continue
            if component not in curb_cache:
                curb_cache[component] = _flank_kerb(
                    rgb, classes, labels == component, occupied, (a, b, c, d)
                )
            evidence = _ray_boundaries(rgb, classes, occupied, access, (x, y), bw, curb_cache[component])
            if not evidence:
                continue
            xx, yy = np.array([x + 0.5, y + 0.5]) / scale
            sz = np.array([r, r]) / scale
            p = [round(xx, 3), round(yy, 3)]
            left, top = np.array(p) - sz
            right, bottom = np.array(p) + sz
            evidence.update(
                recipe=RECIPE,
                coverage="interior_point_only",
                anchor_id=anchor["track_id"],
                native_bumper=[round(cx, 1), round(float(d), 1)],
                point_surface=10,
                parking_connection=round(float(cross.mean()), 3),
                current_tracks_clear=True,
                travel_arrow_checked=True,
                verified_facing=facing,
                source_sha256=ctx["image_sha256"],
                registered_rgb_sha256=sha256(rgb.tobytes()).hexdigest(),
                map_sha256=sha256(np.ascontiguousarray(classes).tobytes()).hexdigest(),
            )
            kerbs = [contact[3] for contact in evidence["boundary_samples"] if contact[1] == "curb"]
            road = [contact[3] for contact in evidence["boundary_samples"] if contact[1] == "road"]
            if kerbs and road:
                dx = np.median(kerbs) - np.median(road)
                if abs(dx) > 0.50 * bw:
                    evidence["curb_side"] = "right" if dx > 0 else "left"
            return [
                {
                    "point": p,
                    "box": [left, top, right, bottom],
                    "polygon": [[left, top], [right, top], [right, bottom], [left, bottom]],
                    "evidence": evidence,
                }
            ]
    return []
