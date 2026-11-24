"""Trace a parking row's three photographed T-shaped end marks when the bonnet limits visible ground."""

from itertools import combinations
from typing import NamedTuple

import cv2
import numpy as np

from . import traffic_arrow, vehicle_rear_boundaries


class _Marker(NamedTuple):
    quality: float
    x: int
    y: int
    arm_length: int
    cap_slope: float


def _visible_kerb(view, side):
    classes = view.classes
    h, w = classes.shape
    cx, bottom = view.bumper
    width, height = view.vehicle_size
    near = classes[
        max(0, round(bottom + 0.09 * height)) : min(h, round(bottom + 0.54 * height)),
        max(0, round(cx - 0.45 * width)) : min(w, round(cx + 0.45 * width)),
    ]
    if not near.size or np.mean(near == 64) < 0.12:
        return None
    is_kerb = np.isin(classes, (2, 15))
    samples = []
    for y in range(max(1, round(bottom - 0.04 * height)), min(h, round(bottom + 0.36 * height))):
        positions = np.arange(round(cx), int(np.clip(cx + side * 2 * width, -1, w)), side)
        if not len(positions):
            continue
        flags = is_kerb[y, positions]
        for index in np.flatnonzero(flags & np.r_[False, ~flags[:-1]]):
            edge = int(positions[index])
            ground_x = edge - side * max(3, round(0.04 * width))
            if not 0.16 * width < side * (edge - cx) < 1.65 * width or not 1 <= ground_x < w - 1:
                continue
            if (
                classes[y, ground_x] not in (10, 13)
                or not view.road[y, ground_x]
                or not 10 <= edge < w - 10
            ):
                continue
            left = np.median(view.rgb[y, edge - 8 : edge - 2], axis=0)
            right = np.median(view.rgb[y, edge + 2 : edge + 9], axis=0)
            if np.linalg.norm(left - right) >= 12:
                samples.append((y, edge))
                break
    if len(samples) < max(6, round(0.09 * height)):
        return None
    rows, edges = np.asarray(samples).T
    slope, intercept = np.polyfit(rows - bottom, edges, 1)
    residual = np.percentile(abs(edges - (intercept + (rows - bottom) * slope)), 85)
    if not 0.3 < side * slope < 6 or residual > max(4, 0.085 * width):
        return None
    return samples, float(slope), float(intercept)


def _junction_templates(span, side):
    """Visual T geometry in perspective, with a transverse painted arm directed towards the kerb."""
    for dx_per_y in (0.85, 1.7, 2.8):
        for dy_per_x in (-0.055, 0, 0.067):
            half_cap = 0.25 * span
            cap = np.array([side * dx_per_y, 1])
            arm = np.array([side, dy_per_x])
            cap = cap / np.linalg.norm(cap)
            arm = arm / np.linalg.norm(arm)
            cap_start = np.rint(-half_cap * cap).astype(int)
            cap_end = np.rint(half_cap * cap).astype(int)
            arm_end = np.rint(span * arm).astype(int)
            lower = np.minimum(np.minimum(cap_start, cap_end), arm_end) - 4
            upper = np.maximum(np.maximum(cap_start, cap_end), arm_end) + 4
            for thickness in (1, 2):
                template = np.zeros((upper - lower + 1)[::-1], np.uint8)
                cv2.line(
                    template, tuple(cap_start - lower), tuple(cap_end - lower), 255, thickness, cv2.LINE_AA
                )
                cv2.line(template, tuple(-lower), tuple(arm_end - lower), 210, thickness, cv2.LINE_AA)
                yield template, -lower, cap, arm, half_cap, thickness


def _at(array, coordinates):
    points = np.rint(coordinates).astype(int)
    return array[points[:, 1].clip(0, array.shape[0] - 1), points[:, 0].clip(0, array.shape[1] - 1)]


def _light_ridge(gray, start, end, normal, thickness):
    locations = np.linspace(start, end, max(7, round(np.linalg.norm(start - end) * 1.4)))
    core = np.max([_at(gray, locations + normal * offset) for offset in (-1, 0, 1)], axis=0)
    adjacent = np.maximum(
        _at(gray, locations + normal * (thickness + 2.8)), _at(gray, locations - normal * (thickness + 2.8))
    )
    contrast = core - adjacent
    return float(np.mean(contrast > 4)), float(np.median(contrast))


def _stroke_score(gray, classes, centre, cap, arm, half_cap, span, thickness):
    normal = np.array([-arm[1], arm[0]])
    supported, contrast = _light_ridge(
        gray, centre + 0.14 * span * arm, centre + 0.97 * span * arm, normal, thickness
    )
    if supported < 0.56 or contrast < 5:
        return None
    cap_normal = np.array([-cap[1], cap[0]])
    caps = [
        _light_ridge(
            gray,
            centre + side * 0.24 * half_cap * cap,
            centre + side * 0.99 * half_cap * cap,
            cap_normal,
            thickness,
        )[0]
        for side in (-1, 1)
    ]
    if max(caps) < 0.73 or min(caps) < 0.33:
        return None
    # A stripe extending through both sides of the junction is not a terminal T.
    extension, _ = _light_ridge(
        gray, centre - 0.75 * span * arm, centre - 0.15 * span * arm, normal, thickness
    )
    if (
        extension > 0.53
        or not np.isin(_at(classes, np.linspace(centre, centre + arm * span, 10)), (10, 13, 24)).all()
    ):
        return None
    return supported + 0.5 * sum(caps) - 0.18 * extension


def _visible_markers(view, side, kerb_slope, kerb_intercept):
    h, w = view.classes.shape
    cx, bottom = view.bumper
    width, height = view.vehicle_size
    gray = cv2.cvtColor(view.rgb, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(2.9, (12, 8)).apply(gray)
    signed = enhanced.astype(np.float32)
    left, right = max(0, round(cx - 2.25 * width)), min(w, round(cx + 2.25 * width))
    top, low = max(1, round(bottom - 1.02 * height)), min(h, round(bottom + 0.12 * height))
    crop = enhanced[top:low, left:right]
    ground = np.isin(view.classes, (10, 13, 24)) & ~view.occupied
    measured = []
    for span in sorted({max(8, round(fraction * width)) for fraction in (0.075, 0.13, 0.21)}):
        for template, origin, cap, arm, half_cap, thickness in _junction_templates(span, side):
            if any(crop.shape[i] - template.shape[i] < 2 for i in (0, 1)):
                continue
            scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            maxima = scores >= cv2.dilate(scores, np.ones((7, 7), np.uint8))
            for row, col in zip(*np.where(maxima & (scores > 0.45)), strict=True):
                point = np.array([left + col, top + row]) + origin
                x, y = map(int, point)
                distance = side * (kerb_intercept + kerb_slope * (y - bottom) - x)
                if not ground[y, x] or not 0.19 * width < distance < 2.10 * width:
                    continue
                strokes = _stroke_score(signed, view.classes, point, cap, arm, half_cap, span, thickness)
                if strokes is not None:
                    measured.append(
                        _Marker(float(scores[row, col]) + strokes, x, y, span, float(cap[0] / cap[1]))
                    )
    kept = []
    for marker in sorted(measured, reverse=True):
        if not any(
            abs(marker.x - m.x) < max(4, 0.045 * width) and abs(marker.y - m.y) < max(3, 0.045 * height)
            for m in kept
        ):
            kept.append(marker)
    return sorted(kept, key=lambda mark: mark.y)


def _row_hypotheses(markers, view, side, kerb_slope, kerb_intercept):
    cx, bottom = view.bumper
    width, height = view.vehicle_size
    matches = []
    for triple in combinations(markers, 3):
        a, b, c = triple
        first_step, second_step = b.y - a.y, c.y - b.y
        if min(first_step, second_step) < max(5, 0.055 * height) or c.y - a.y < 0.14 * height:
            continue
        if not 0.60 <= second_step / first_step <= 4.2:
            continue
        rows = np.array([mark.y for mark in triple])
        xs = np.array([mark.x for mark in triple])
        slope, intercept = np.polyfit(rows - bottom, xs, 1)
        residual = np.max(abs(xs - (intercept + (rows - bottom) * slope)))
        if (
            not 0.45 < side * slope < 4.6
            or not 0.22 < slope / kerb_slope < 1.18
            or residual > max(2, 0.025 * width)
        ):
            continue
        if any(abs(mark.cap_slope - slope) / max(1, abs(slope)) > 0.72 for mark in triple):
            continue
        span = side * (kerb_intercept - intercept)
        if (
            not 0.60 * width < span < 1.60 * width
            or not 0.18 < (cx - intercept) / (kerb_intercept - intercept) < 0.82
        ):
            continue
        if c.y < bottom - 0.47 * height or bottom + 0.11 * height - c.y > 2.7 * second_step:
            continue
        if (
            not 0.80 * a.arm_length <= b.arm_length <= 2.8 * a.arm_length
            or not 0.80 * b.arm_length <= c.arm_length <= 2.8 * b.arm_length
        ):
            continue
        matches.append(
            (sum(mark.quality for mark in triple) - residual * 0.08, triple, float(slope), float(intercept))
        )
    return sorted(matches, key=lambda match: -match[0])


def _visible_patch(view, kerbs, side, slope, intercept):
    _, bottom = view.bumper
    width, height = view.vehicle_size
    h, w = view.classes.shape
    rows = []
    for y, kerb in kerbs:
        if not bottom + 0.035 * height <= y <= bottom + 0.38 * height:
            continue
        lane = intercept + slope * (y - bottom)
        middle = (lane + kerb) / 2
        if not 0.59 * width < side * (kerb - lane) < 1.84 * width:
            continue
        left, right = round(middle - 0.16 * width), round(middle + 0.16 * width)
        if left < 0 or right >= w:
            continue
        strip = np.s_[y, left : right + 1]
        if (
            np.isin(view.classes[strip], (10, 13)).all()
            and view.road[strip].all()
            and not view.occupied[strip].any()
        ):
            rows.append(vehicle_rear_boundaries._Section(y, lane, kerb))
    if (
        len(rows) < max(5, 0.065 * height)
        or rows[0].y > bottom + 0.14 * height
        or any(b.y - a.y > 2 for a, b in zip(rows, rows[1:]))
    ):
        return None
    if traffic_arrow.in_measured_strip(
        view.rgb, view.classes, rows, view.spec["box"], view.original.size, side
    ):
        return None
    # Compute the clearance of the indicated pixels from the measured edges, bonnet and obstructions.
    mask = np.zeros((h, w), np.uint8)
    for row in rows:
        left, right = sorted([round(row.lane), row.curb])
        mask[row.y, max(0, left + 2) : min(w, right - 1)] = 1
    mask[~view.road | view.occupied | ~np.isin(view.classes, (10, 13))] = 0
    distances = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    candidates = []
    for row in rows:
        for fraction in (0.43, 0.50, 0.57, 0.64):
            x = round(row.lane + fraction * (row.curb - row.lane))
            if 0 <= x < w:
                candidates.append((float(distances[row.y, x]) - 0.15 * abs(fraction - 0.50), x, row.y))
    if not candidates:
        return None
    clearance, x, y = max(candidates)
    radius = max(2, round(0.038 * min(width, height)))
    if clearance < radius + 1 or not vehicle_rear_boundaries._clear_patch(view, x, y, radius):
        return None
    return x, y, radius, rows


def measure(view):
    """Require three separate observed T markings and one clear, connected strip before the car bonnet."""
    choices = []
    for side in (-1, 1):
        kerb = _visible_kerb(view, side)
        if kerb is None:
            continue
        samples, slope, intercept = kerb
        markers = _visible_markers(view, side, slope, intercept)
        matches = _row_hypotheses(markers, view, side, slope, intercept)
        if not matches:
            continue
        score, triplet, edge_slope, edge = matches[0]
        width, _ = view.vehicle_size
        if any(
            abs(other_edge - edge) > 0.16 * width
            for other_score, _, _, other_edge in matches[1:]
            if other_score > 0.95 * score
        ):
            continue
        point = _visible_patch(view, samples, side, edge_slope, edge)
        if point is not None:
            x, y, radius, rows = point
            measured = [
                np.round((np.array([mark.x + 0.5, mark.y + 0.5]) / view.scale), 3).tolist()
                for mark in triplet
            ]
            evidence = {
                "physical_boundary": "three_optical_parking_end_marks_and_observed_kerb_before_bonnet",
                "source_end_marks_xy": measured,
                "observed_ground_run_y": [rows[0].y, rows[-1].y],
                "ground_rows": len(rows),
                "registered_kerb_contacts": len(samples),
                "aisle_edge_dx_per_y": round(edge_slope, 5),
                "kerb_dx_per_y": round(slope, 5),
                "selection": "maximum_visible_pavement_clearance_before_bonnet",
            }
            choices.append((side, x, y, radius, evidence))
    return choices[0] if len(choices) == 1 else None
