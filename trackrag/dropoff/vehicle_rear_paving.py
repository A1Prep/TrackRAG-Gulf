"""Verify a material seam between a photographed parking apron and the driving lane."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from . import traffic_arrow
from .vehicle_rear_boundaries import GROUND, _Section, _clear_patch


@dataclass(frozen=True)
class _PavingRow:
    y: int
    lane: int
    curb: int
    classified_parking: bool
    visible_kerb: bool
    visible_seam: bool


def _measured_section(view, y, edge, side, observed_segment, gray):
    classes = view.classes
    h, w = classes.shape
    width, _ = view.vehicle_size
    x = round(edge)
    if not 2 <= x < w - 2:
        return None
    end = int(np.clip(x + side * round(2.1 * width), -1, w))
    positions = np.arange(x, end, side)
    values = classes[y, positions]
    kerbs = np.flatnonzero(np.isin(values, (2, 9, 15)))
    if not len(kerbs):
        return None
    span = int(kerbs[0])
    if span < 9 or not 0.54 * width <= span <= 1.86 * width:
        return None
    curb = int(positions[span])
    outside = values[span : span + max(5, round(0.18 * width))]
    if len(outside) < 4 or np.mean(np.isin(outside, (2, 9, 15))) < 0.80:
        return None
    strip = positions[3 : span - 2]
    if (
        not np.isin(classes[y, strip], GROUND).all()
        or not view.road[y, strip].all()
        or view.occupied[y, strip].any()
    ):
        return None
    road_x = x - side * max(5, round(0.18 * width))
    if not 0 <= road_x < w or np.mean(classes[y, min(x, road_x) : max(x, road_x)] == 13) < 0.90:
        return None
    parking = np.mean(values[max(1, int(span * 0.66)) : span] == 10) > 0.11
    left = view.rgb[y, max(0, curb - 5) : max(0, curb - 1)]
    right = view.rgb[y, min(w, curb + 2) : min(w, curb + 7)]
    kerb_contrast = bool(
        len(left)
        and len(right)
        and np.linalg.norm(np.median(left, axis=0) - np.median(right, axis=0)) >= 13
    )
    sample = gray[y, max(0, x - 3) : min(w, x + 4)].astype(int)
    seam_contrast = observed_segment[0] - 1 <= y <= observed_segment[1] + 1 and np.ptp(sample) >= 6
    return _PavingRow(y, x, curb, bool(parking), kerb_contrast, bool(seam_contrast))


def _best_supported_point(view, rows, measured_y):
    """Prefer an interior patch corroborated by the semantic parking evidence at the optical seam."""
    width, height = view.vehicle_size
    _, bottom = view.bumper
    radius = max(2, round(0.055 * min(width, height)))
    results = []
    for row in rows:
        y = row.y
        if not measured_y[0] <= y <= measured_y[1]:
            continue
        margin = max(radius + 2, 0.15 * width)
        start, end = sorted((row.lane, row.curb))
        # Neither the kerb itself nor the road-side seam is meant as a passenger-selected spot.
        for x in np.arange(start + margin, end - margin, max(1, round(0.05 * width))).astype(int):
            if not _clear_patch(view, x, y, radius):
                continue
            local = np.s_[y - radius : y + radius + 1, x - radius : x + radius + 1]
            near = np.s_[
                y - radius : y + radius + 1,
                max(0, x - 3 * radius) : min(view.classes.shape[1], x + 3 * radius + 1),
            ]
            classified = float(np.mean(view.classes[local] == 10))
            nearby = float(np.mean(view.classes[near] == 10))
            if nearby <= 0:
                continue
            clearance = min(abs(x - row.lane), abs(x - row.curb)) / abs(row.curb - row.lane)
            quality = (
                classified + nearby + 0.14 * clearance - 0.025 * abs(y - (bottom + 0.33 * height)) / height
            )
            results.append((quality, int(y), int(x), classified, nearby))
    if not results:
        return None
    _, y, x, classified, nearby = max(results)
    return x, y, radius, round(classified, 4), round(nearby, 4)


def _measure_line(view, start, end, side, slope, intercept, gray, y_limit):
    cx, bottom = view.bumper
    width, height = view.vehicle_size
    low, high = sorted((start[1], end[1]))
    measured = []
    for y in range(max(0, round(bottom + 0.02 * height)), min(y_limit, round(bottom + 0.76 * height))):
        sample = _measured_section(view, y, intercept + (y - bottom) * slope, side, (low, high), gray)
        if sample is not None:
            measured.append(sample)
    if not measured:
        return []
    options = []
    groups = np.split(
        np.arange(len(measured)), np.flatnonzero(np.diff([row.y for row in measured]) > 2) + 1
    )
    for group in groups:
        run = [measured[i] for i in group]
        if (
            len(run) < max(8, 0.25 * height)
            or run[0].y > bottom + 0.14 * height
            or run[-1].y < bottom + 0.30 * height
        ):
            continue
        parking = sum(row.classified_parking for row in run)
        outer = sum(row.visible_kerb for row in run)
        seam = sum(row.visible_seam for row in run)
        if (
            parking < max(4, 0.11 * height)
            or outer < max(4, 0.12 * height)
            or seam < max(3, 0.045 * height)
        ):
            continue
        if abs(np.median([0.5 * (row.lane + row.curb) for row in run[:3]]) - cx) > 0.70 * width:
            continue
        change = np.diff([row.curb for row in run]) / np.maximum(1, np.diff([row.y for row in run]))
        if np.quantile(abs(change - np.median(change)), 0.9) > max(8, 0.12 * width):
            continue
        sections = [_Section(row.y, row.lane, row.curb) for row in run]
        if traffic_arrow.in_measured_strip(
            view.rgb, view.classes, sections, view.spec["box"], view.original.size, side
        ):
            continue
        supported = _best_supported_point(view, run, (low, high))
        if supported:
            x, y, radius, local, nearby = supported
            evidence = {
                "physical_boundary": "optical_paving_seam_and_observed_kerb",
                "original_seam_in_registered_pixels": [round(float(coord), 3) for coord in (*start, *end)],
                "ground_run_y": [run[0].y, run[-1].y],
                "map_rows_with_parking": parking,
                "optically_supported_kerb_rows": outer,
                "optically_supported_seam_rows": seam,
                "point_parking_fraction": local,
                "nearby_parking_fraction": nearby,
            }
            options.append((len(run), side, x, y, radius, evidence))
    return options


def measure(view):
    """Use the photographed seam only when positive native parking occurs in the same narrow apron."""
    h, w = view.classes.shape
    cx, bottom = view.bumper
    width, height = view.vehicle_size
    if not 0.64 < width / height < 1.98:
        return []
    xlow, xhigh = max(0, int(cx - 3 * width)), min(w, int(cx + 3 * width))
    ylow, yhigh = max(0, int(bottom - 0.15 * height)), min(h, int(bottom + 0.95 * height))
    if np.count_nonzero(view.classes[ylow:yhigh, xlow:xhigh] == 10) < max(20, 0.011 * width * height):
        return []
    # Keep the source grid registered to the semantic grid. A separately stretched vehicle crop would move the seam.
    detail = view.original.convert("RGB").resize((2 * w, 2 * h), Image.Resampling.BOX)
    full_gray = cv2.cvtColor(np.asarray(detail), cv2.COLOR_RGB2GRAY)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(full_gray)[0]
    if lines is None:
        return []
    gray = cv2.cvtColor(view.rgb, cv2.COLOR_RGB2GRAY)
    options = []
    # OpenCV Python releases expose line segments as (N,4) or (N,1,4).
    for raw in np.asarray(lines).reshape(-1,4):
        x0, y0, x1, y1 = raw / 2
        if max(y0, y1) < ylow or min(y0, y1) > yhigh or not xlow <= (x0 + x1) / 2 <= xhigh:
            continue
        if abs(y1 - y0) < max(5, 0.15 * height) or np.hypot(x1 - x0, y1 - y0) < 0.38 * width:
            continue
        slope = (x1 - x0) / (y1 - y0)
        intercept = x0 + (bottom - y0) * slope
        if not 0.4 < abs(slope) < 5.6:
            continue
        side = -1 if slope < 0 else 1
        if (
            not 0.13 < -side * (intercept - cx) / width < 1.10
            or min(y1, y0) > bottom + 0.43 * height
            or max(y1, y0) < bottom + 0.18 * height
        ):
            continue
        options.extend(_measure_line(view, (x0, y0), (x1, y1), side, slope, intercept, gray, yhigh))
    return options
