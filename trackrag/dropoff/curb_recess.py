"""Find visible pavement between two current vehicles in a recessed parking row.

The result is an interior point with a small observed patch for highlighting. Neither that
patch nor the flanking vehicles establish the footprint of an individual parking space.
The caller authenticates the original photo and its Mapillary class map and ties the
returned track identities to the passenger's words before offering a point.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import cv2
import numpy as np

from .bay_clearance import visible_road_component


RECIPE = "measured_vehicle_flanked_parking_recess_v1"
_VEHICLES = {"car", "sedan", "suv", "light_vehicle", "medium_vehicle", "heavy_vehicle",
             "large_vehicle", "taxi", "van", "minivan", "pickup", "truck", "bus"}


@dataclass(frozen=True)
class _Vehicle:
    track: int
    box: tuple

    @property
    def width(self):
        return self.box[2] - self.box[0]

    @property
    def height(self):
        return self.box[3] - self.box[1]


@dataclass(frozen=True)
class _Band:
    x: int
    back: int
    front: int
    component: int
    paved_edge: bool
    kerb: bool


def _bands(x, classes, components, connected):
    # A vertical section must cross a signed sidewalk/kerb, parking and a connected
    # traffic aisle in that order. A foreground bonnet can hide the aisle locally.
    col = classes[:, x]
    height = len(col)
    occupied = np.r_[False, col == 10, False]
    edges = np.flatnonzero(occupied[1:] != occupied[:-1]).reshape(-1, 2)
    result = []
    for first, after in edges:
        if (first == 0 or after == height or after - first < max(5, .011 * height)
                or after - first > .31 * height or col[first - 1] not in (2, 15)):
            continue
        preceding = col[max(0, first - max(4, round(.018 * height))):first]
        front = after
        while front < height and col[front] == 24 and front - after < .023 * height:
            front += 1
        road = (front + 1 < height and col[front] == 13 and col[front + 1] == 13
                and connected[front, x])
        # A one-cell road sliver can occur where the bonnet meets the aisle.
        # It preserves strip continuity but cannot supply a measured aisle edge.
        bonnet = front < height and (col[front] == 64 or (
            col[front] == 13 and front + 1 < height and col[front + 1] == 64))
        if not road and not bonnet:
            continue
        result.append(_Band(x, int(first), int(after - 1), int(components[first, x]),
                            bool(road), bool((preceding == 2).any())))
    return result


def _box(raw, scale, size):
    box = np.asarray(raw, dtype=float)
    if (box.shape != (4,) or not np.isfinite(box).all() or (box < 0).any()
            or box[2] <= box[0] or box[3] <= box[1] or (box[2:] > size).any()):
        raise ValueError("Invalid current detection box")
    return tuple(box * np.tile(scale, 2))


def _vehicles(context, scale, size, height, width):
    boxes = [_box(b, scale, size) for b in context["current_boxes"]]
    current = set(boxes)
    result, seen = [], set()
    for identifier, label, raw in context["current_vehicles"]:
        box = _box(raw, scale, size)
        if box not in current or identifier in seen:
            raise ValueError("Vehicle identities must match current detections")
        seen.add(identifier)
        car = _Vehicle(identifier, box)
        if (str(label).casefold() in _VEHICLES and car.width >= .030 * width
                and car.height >= .026 * height):
            result.append(car)
    return sorted(result, key=lambda v: (v.box[0], v.track)), boxes


def _continuous(group, step):
    parts = []
    for band in group:
        if not parts or band.x - parts[-1][-1].x > 2 * step:
            parts.append([])
        parts[-1].append(band)
    return max(parts, key=len, default=[])


def _fit_row(left, right, column, width, step):
    xmin, xmax = left.box[2], right.box[0]
    span = xmax - xmin
    unit = min(left.width, right.width)
    minimum_span = max(.027 * width, .39 * unit, .75 * (left.height + right.height) / 2)
    # Two front/rear-facing cars with contacts at similar depth give a direct scale
    # in the image. A small patch between their bumpers is insufficient for a car.
    level = abs(left.box[3] - right.box[3]) <= .30 * min(left.height, right.height)
    if level and max(left.width / left.height, right.width / right.height) <= 1.65:
        minimum_span = max(minimum_span, (left.width + right.width) / 2)
    if (span < minimum_span or span > min(.55 * width, 3.1 * unit)
            or max(left.height, right.height) > 3.4 * min(left.height, right.height)):
        return None
    xs = np.arange(np.ceil(xmin + .035 * span), np.floor(xmax - .035 * span) + 1, step,
                   dtype=int)
    if len(xs) < 9:
        return None
    groups = {}
    for x in xs:
        at = (x - xmin) / span
        depth = (1 - at) * left.box[3] + at * right.box[3]
        car_height = (1 - at) * left.height + at * right.height
        for band in column(int(x)):
            # Tyre contact of both vehicles must follow this same projected parking row.
            if (band.back - .22 * car_height <= depth <= band.front + .22 * car_height
                    and band.front - band.back >= .12 * car_height):
                columns = groups.setdefault(band.component, {})
                previous = columns.get(band.x)
                if previous is None or abs((band.back + band.front) / 2 - depth) < abs(
                        (previous.back + previous.front) / 2 - depth):
                    columns[band.x] = band
    for group in sorted(groups.values(), key=len, reverse=True):
        run = _continuous(group.values(), step)
        if len(run) < .77 * len(xs):
            continue
        kerbs = [b for b in run if b.kerb]
        fronts = [b for b in run if b.paved_edge]
        if (len(kerbs) < .56 * len(run) or len(fronts) < max(8, .28 * len(xs))
                or not fronts or (fronts[-1].x - fronts[0].x) < .24 * span):
            continue
        # Rear and front edges must form a continuous band, not fragments with
        # abrupt depth jumps where another row or a driveway crosses the gap.
        for attr in ("back", "front"):
            edge = np.asarray([getattr(b, attr) for b in run])
            if np.max(np.abs(np.diff(edge))) > max(5., .14 * np.median(
                    [b.front - b.back for b in run])):
                break
        else:
            return run, fronts
    return None


def _touches_row(car, side, components, classes, component):
    l, _, r, b = car.box
    h, w = car.height, car.width
    height, width = classes.shape

    def region(x0, y0, x1, y1):
        return np.s_[max(0, int(y0)):min(height, int(np.ceil(y1))),
                     max(0, int(x0)):min(width, int(np.ceil(x1)))]

    foot = region(l + .08 * w, b - .05 * h, r - .08 * w, b + .16 * h)
    contacts = components[foot] == component
    if contacts.size and contacts.mean() >= .16 and contacts.any(axis=0).mean() >= .36:
        return True
    # The dashcam bonnet can hide the lower tyre. Only credit a masked foot if
    # the gap-facing lower flank still borders the *same* parking component.
    if not contacts.size or (classes[foot] == 64).mean() < .55:
        return False
    x0, x1 = ((r - .04 * w, r + .08 * w) if side == "left" else
              (l - .08 * w, l + .04 * w))
    flank = components[region(x0, b - .36 * h, x1, b + .15 * h)] == component
    return bool(flank.size and flank.mean() >= .16 and flank.any(axis=0).mean() >= .53)


def _clear_corridor(left, right, run, boxes):
    xmin, xmax = left.box[2], right.box[0]
    span = xmax - xmin
    mouth = [b for b in run if b.paved_edge and xmin + .08 * span <= b.x <= xmax - .08 * span]
    for box in boxes:
        if box == left.box or box == right.box:
            continue
        # Test the remaining detections against the strip at THEIR own x and
        # depth. This also rejects a person's box crossing the gap from outside.
        crossing = [b for b in run if max(box[0], xmin + .08 * span) <= b.x <= min(
            box[2], xmax - .08 * span) and box[3] > b.back + .16 * (b.front - b.back)
            and box[1] < b.front - .08 * (b.front - b.back)]
        if len(crossing) >= 2:
            return False
        blocked = [b for b in mouth if box[0] <= b.x <= box[2]
                   and box[3] >= b.front and box[1] <= b.front + max(3, .23 * (b.front - b.back))]
        if len(blocked) >= max(3, .32 * len(mouth)):
            return False
    return True


def _patch(left, right, fronts, clearance):
    xmin, xmax = left.box[2], right.box[0]
    span = xmax - xmin
    middle = (xmax + xmin) / 2
    best = None
    for b in fronts:
        if abs(b.x - middle) > .32 * span:
            continue
        y = round((b.back + b.front) / 2)
        clear = float(clearance[y, b.x])
        thickness = b.front - b.back + 1
        if clear < max(3., .12 * thickness, .055 * min(left.height, right.height)):
            continue
        score = min(clear / (.48 * thickness), 1.) - abs(b.x - middle) / span
        if best is None or score > best[0]:
            best = score, b, y, clear
    return best


def _output(left, right, run, fronts, chosen, scale):
    _, band, y, clear = chosen

    def pixel(x, yy):
        # Map cell centres to the authenticated original pixel coordinate space.
        return [round((x + .5) / scale[0] - .5, 3), round((yy + .5) / scale[1] - .5, 3)]

    radius = max(1, int(clear * .37))
    x = band.x
    point = pixel(x, y)
    patch = [pixel(x - radius, y - radius), pixel(x + radius, y - radius),
             pixel(x + radius, y + radius), pixel(x - radius, y + radius)]
    sample = np.linspace(0, len(fronts) - 1, min(7, len(fronts)), dtype=int)
    return {"point": point, "box": patch[0] + patch[2], "support_patch": patch,
            "evidence": {"recipe": RECIPE, "extent": "interior_point_only",
                         "flanking_track_ids": [left.track, right.track],
                         "both_vehicle_contacts_verified": True,
                         "same_parking_component": True, "signed_parking_class": 10,
                         "observed_sections": len(run), "observed_kerb_sections": sum(b.kerb for b in run),
                         "observed_road_sections": len(fronts),
                         "kerb_and_aisle_sections": [[pixel(fronts[i].x, fronts[i].back),
                                                      pixel(fronts[i].x, fronts[i].front)]
                                                     for i in sample],
                         "distance_to_nonparking_or_detections_cells": round(clear, 3)}}


def candidates(original, classes, context):
    """Propose point-only parking patches from two identified current vehicles.

    `current_vehicles` has the rows returned by `bay_anchors.vehicles(scene)`.
    `current_boxes` includes people and all other current detections as obstacles.
    Missing identity, authenticated surface, or physical edge evidence gives no patch.
    """
    if (classes is None or not isinstance(classes, np.ndarray) or classes.ndim != 2
            or not isinstance(context, dict) or tuple(context.get("size", ())) != original.size
            or min(classes.shape) < 12 or "current_boxes" not in context
            or "current_vehicles" not in context or np.count_nonzero(classes == 10) < 80):
        return []
    height, width = classes.shape
    size = np.asarray(original.size)
    scale = np.array([width, height]) / size
    try:
        cars, boxes = _vehicles(context, scale, size, height, width)
    except (TypeError, ValueError):
        return []
    if len(cars) < 2:
        return []
    road = visible_road_component(classes)
    if road is None:
        return []
    parking = (classes == 10).astype("uint8")
    _, components = cv2.connectedComponents(parking, connectivity=8)
    for x0, y0, x1, y1 in boxes:
        parking[int(y0):int(np.ceil(y1)) + 1, int(x0):int(np.ceil(x1)) + 1] = 0
    clearance = cv2.distanceTransform(parking, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    column = lru_cache(maxsize=None)(lambda x: _bands(x, classes, components, road))
    step = max(1, round(width / 420))
    found = []
    for left, right in combinations(cars, 2):
        fitted = _fit_row(left, right, column, width, step)
        if fitted is None:
            continue
        run, fronts = fitted
        component = run[0].component
        if (not _touches_row(left, "left", components, classes, component)
                or not _touches_row(right, "right", components, classes, component)
                or not _clear_corridor(left, right, run, boxes)):
            continue
        chosen = _patch(left, right, fronts, clearance)
        if chosen is not None:
            found.append(_output(left, right, run, fronts, chosen, scale))
    return sorted(found, key=lambda o: (o["point"], o["evidence"]["flanking_track_ids"]))
