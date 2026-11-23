"""Find a point on an unpaved, vehicle-lined urban frontage, for subsequent visual review.

Terrain class 26 covers much more than parking. Proposals require an original-photo
surface change at a directly connected traffic aisle, a substantial open patch, and
three independent current vehicles standing on the same terrain strip beside buildings.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import cv2
import numpy as np
from PIL import Image

from .bay_clearance import visible_road_component
from .curb_recess import _vehicles


RECIPE = "observed_unpaved_frontage_parking_row_v1"


@dataclass(frozen=True)
class _Slice:
    x: int
    rear: int
    front: int
    lane: bool
    building: bool
    row_y: float
    vehicle_height: float


def _foot(car, components, classes):
    left, _, right, bottom = car.box
    x0 = int(left + .10 * car.width)
    x1 = int(np.ceil(right - .10 * car.width))
    y0 = int(np.floor(bottom))
    y1 = int(np.ceil(bottom + .16 * car.height)) + 1
    support = components[y0:y1, x0:x1]
    if not support.size:
        return None
    labels, counts = np.unique(support[support > 0], return_counts=True)
    if not len(labels):
        return None
    idx = counts.argmax()
    member = support == labels[idx]
    if member.mean() < .35 or member.any(axis=0).mean() < .55:
        return None
    # The broad foot window must not reach across a curb, sidewalk or another
    # road. Look inward from the actual lower box edge; a few pixels assigned to
    # the vehicle itself can cover a tyre, but a material separator breaks contact.
    near = slice(y0, y0 + max(3, round(.045 * car.height)) + 1)
    candidate = components[near, x0:x1] == labels[idx]
    covered = candidate.any(axis=0)
    first = np.argmax(candidate, axis=0)
    before = np.arange(len(candidate))[:, None] < first
    vehicle_pixels = np.isin(classes[near, x0:x1], (54, 55, 56, 57, 58, 60, 61, 62, 63))
    separators = np.count_nonzero(before & ~vehicle_pixels, axis=0)
    if np.mean(covered & (separators <= 1)) < .36:
        return None
    return int(labels[idx])


def _cross_sections(x, classes, components, road):
    col = classes[:, x]
    h = len(col)
    terrain = np.r_[False, col == 26, False]
    transitions = np.flatnonzero(terrain[1:] != terrain[:-1]).reshape(-1, 2)
    bands = []
    for start, end in transitions:
        if start == 0 or end >= h - 1 or end - start < .036 * h or end - start > .40 * h:
            continue
        lane = col[end] == 13 and col[end + 1] == 13 and road[end, x] and road[end + 1, x]
        bonnet = (col[end] == 64 or (col[end] == 13 and np.any(col[end:min(h, end + 4)] == 64)))
        if not lane and not bonnet:
            continue
        rear = col[max(0, start - round(.16 * h)):start]
        near_rear = rear[-max(5, round(.05 * h)):]
        if np.count_nonzero(near_rear == 13) > .18 * len(near_rear):
            continue
        building = np.count_nonzero(rear == 17) >= max(5, .24 * len(rear))
        bands.append((int(components[start, x]), int(start), int(end - 1), bool(lane), building))
    return bands


def _independent_row_member(left, right, others, component):
    lx = (left.box[0] + left.box[2]) / 2
    rx = (right.box[0] + right.box[2]) / 2
    for car, label in sorted(others, key=lambda c: c[0].width, reverse=True):
        if label != component or car in (left, right):
            continue
        cx = (car.box[0] + car.box[2]) / 2
        if not (cx < lx - .50 * min(left.width, car.width)
                or cx > rx + .50 * min(right.width, car.width)):
            continue
        if min(abs(cx - lx), abs(cx - rx)) > 2.6 * max(left.width, right.width, car.width):
            continue
        predicted = left.box[3] + (cx - lx) * (right.box[3] - left.box[3]) / (rx - lx)
        if abs(predicted - car.box[3]) > max(4, .30 * car.height):
            continue
        # A duplicated detection of either flanking vehicle is not a third parked vehicle.
        if any((max(0, min(car.box[2], c.box[2]) - max(car.box[0], c.box[0]))
                * max(0, min(car.box[3], c.box[3]) - max(car.box[1], c.box[1])))
               > .32 * min(car.width * car.height, c.width * c.height) for c in (left, right)):
            continue
        return car
    return None


def _fit(left, right, component, sections, width):
    lo, hi = left.box[2], right.box[0]
    span = hi - lo
    minimum = max(.04 * width, .70 * (left.height + right.height) / 2)
    if abs(left.box[3] - right.box[3]) < .30 * min(left.height, right.height):
        if max(left.width / left.height, right.width / right.height) <= 1.7:
            minimum = max(minimum, (left.width + right.width) / 2)
    if (span < minimum or span > min(.45 * width, 3.1 * min(left.width, right.width))
            or max(left.height, right.height) > 3.35 * min(left.height, right.height)):
        return None
    step = max(1, round(width / 420))
    xs = range(int(np.ceil(lo + .04 * span)), int(np.floor(hi - .04 * span)) + 1, step)
    slices = []
    for x in xs:
        alpha = (x - lo) / span
        height = (1 - alpha) * left.height + alpha * right.height
        # Tyre height is measured at each vehicle centre and interpolated in perspective.
        lcx, rcx = (left.box[0] + left.box[2]) / 2, (right.box[0] + right.box[2]) / 2
        y = left.box[3] + (x - lcx) * (right.box[3] - left.box[3]) / (rcx - lcx)
        matching = [s for s in sections(x) if s[0] == component
                    and .45 * height <= s[2] - s[1] <= 3.5 * height
                    and s[1] - .12 * height < y < s[2] - .20 * height
                    and s[2] - y < 2.3 * height]
        if len(matching) == 1:
            _, rear, front, lane, building = matching[0]
            slices.append(_Slice(x, rear, front, lane, building, y, height))
    if (len(slices) < max(12, .77 * len(xs)) or np.max(np.diff([s.x for s in slices])) > 4 * step
            or sum(s.building for s in slices) < .68 * len(slices)):
        return None
    contacts = [s for s in slices if s.lane]
    if (len(contacts) < max(8, .33 * len(xs))
            or contacts[-1].x - contacts[0].x < .28 * span
            or max(np.diff([s.x for s in contacts])) > 4 * step):
        return None
    return slices, contacts


def _obstruction(left, right, slices, boxes):
    lo, hi = left.box[2], right.box[0]
    span = hi - lo
    for box in boxes:
        if box in (left.box, right.box):
            continue
        # Include the entrance below the pin; ignore background objects behind the row.
        covered = [s for s in slices if max(box[0], lo + .08 * span) <= s.x <= min(box[2], hi - .08 * span)
                   and box[3] > max(s.rear + .08 * (s.front - s.rear), s.row_y - .12 * s.vehicle_height)
                   and box[1] < s.front + .14 * (s.front - s.rear)]
        if len(covered) >= 2:
            return True
    return False


def _contrast(contacts, rgb, labels):
    separations = []
    for s in contacts:
        depth = s.front - s.rear
        soil = rgb[s.front - max(4, round(.20 * depth)):s.front - max(2, round(.08 * depth)), s.x]
        road_slice = slice(s.front + 3, min(len(labels), s.front + max(8, round(.13 * depth))))
        road = rgb[road_slice, s.x][labels[road_slice, s.x] == 13]
        if len(soil) >= 2 and len(road) >= 3:
            separations.append(float(np.linalg.norm(np.median(soil, axis=0) - np.median(road, axis=0))))
    if len(separations) < max(7, .35 * len(contacts)) or np.median(separations) < 15:
        return None
    return float(np.median(separations))


def _granular(x, y, radius, rgb, texture):
    r = max(4, min(15, int(radius)))
    patch = rgb[y-r:y+r+1, x-r:x+r+1].astype("int16")
    detail = texture[y-r:y+r+1, x-r:x+r+1]
    if patch.shape[:2] != (2*r+1, 2*r+1):
        return False
    red, green, blue = patch[..., 0], patch[..., 1], patch[..., 2]
    green_cover = ((green > red + 6) & (green > blue + 7)).mean()
    med = np.median(patch, axis=(0, 1))
    if green_cover > .07 or med.max() - med.min() > 90 or med.mean() < 30 or med.mean() > 244:
        return False
    # Scattered local variations are needed across the patch. A single shadow edge
    # or isolated paint mark on an otherwise smooth surface cannot establish texture.
    quadrants = (detail[:r, :r], detail[:r, -r:], detail[-r:, :r], detail[-r:, -r:])
    return (np.percentile(detail, 80) >= 1.5 and np.percentile(detail, 96) >= 3.4
            and sum(np.percentile(q, 94) >= 2.7 for q in quadrants) >= 3)


def candidates(original, classes, context):
    """Return native point, local support patch and bound current vehicle IDs for review."""
    if (not isinstance(classes, np.ndarray) or classes.ndim != 2 or min(classes.shape) < 24
            or not isinstance(context, dict) or tuple(context.get("size", ())) != original.size
            or "current_boxes" not in context or "current_vehicles" not in context
            or np.count_nonzero(classes == 26) < .004 * classes.size):
        return []
    height, width = classes.shape
    size = np.asarray(original.size)
    scale = np.array([width, height]) / size
    try:
        cars, boxes = _vehicles(context, scale, size, height, width)
    except (ValueError, TypeError):
        return []
    terrain = (classes == 26).astype("uint8")
    _, components = cv2.connectedComponents(terrain, connectivity=8)
    parked = [(c, component) for c in cars if (component := _foot(c, components, classes)) is not None]
    if len(parked) < 3:
        return []
    road = visible_road_component(classes)
    if road is None:
        return []
    for x0, y0, x1, y1 in boxes:
        terrain[int(y0):int(np.ceil(y1))+1, int(x0):int(np.ceil(x1))+1] = 0
    clearance = cv2.distanceTransform(terrain, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    rgb = np.asarray(original.convert("RGB").resize((width, height), Image.Resampling.BOX))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype("float32")
    texture = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.4))
    sections = lru_cache(maxsize=None)(lambda x: _cross_sections(x, classes, components, road))

    def native(x, y):
        return [round((x+.5)/scale[0]-.5, 3), round((y+.5)/scale[1]-.5, 3)]

    result = []
    for (left, component), (right, other) in combinations(parked, 2):
        if component != other:
            continue
        fitted = _fit(left, right, component, sections, width)
        if fitted is None:
            continue
        slices, contacts = fitted
        extra = _independent_row_member(left, right, parked, component)
        if extra is None or _obstruction(left, right, slices, boxes):
            continue
        contrast = _contrast(contacts, rgb, classes)
        if contrast is None:
            continue
        middle = (left.box[2] + right.box[0]) / 2
        span = right.box[0] - left.box[2]
        for s in sorted(contacts, key=lambda c: abs(c.x-middle)):
            if abs(s.x - middle) > .26 * span:
                break
            x = s.x
            y = round(max(s.row_y + .08 * s.vehicle_height, s.rear + .19 * (s.front - s.rear)))
            if not s.rear < y < s.front:
                continue
            clear = float(clearance[y, x])
            if clear < max(5, .14 * s.vehicle_height) or not _granular(x, y, clear * .40, rgb, texture):
                continue
            radius = max(1, int(clear * .29))
            patch = [native(x-radius, y-radius), native(x+radius, y-radius),
                     native(x+radius, y+radius), native(x-radius, y+radius)]
            samples = np.linspace(0, len(contacts)-1, min(7, len(contacts)), dtype=int)
            result.append({"point": native(x, y), "box": patch[0] + patch[2], "support_patch": patch,
                           "evidence": {"recipe": RECIPE, "extent": "interior_point_only",
                               "flanking_track_ids": [left.track, right.track], "additional_row_track_id": extra.track,
                               "terrain_class": 26, "observed_road_sections": len(contacts),
                               "observed_frontage_sections": sum(int(v.building) for v in slices),
                               "observed_row_sections": len(slices), "visible_nonvegetated_texture": True,
                               "terrain_to_road_color_distance": round(contrast, 3),
                               "road_contacts": [native(contacts[i].x, contacts[i].front+1) for i in samples],
                               "distance_to_nonterrain_or_detections_cells": round(clear, 3)}})
            break
    return sorted(result, key=lambda o: (o["point"], o["evidence"]["flanking_track_ids"]))
