"""Measure a narrow, empty continuation of a pictured curbside parking row.

Two current vehicles must sit in the same perspective strip between a visible
painted edge and an outer raised kerb. The proposal is a point, with a small
inspected patch. It does not establish bumper orientation or a full stall.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import numpy as np
from PIL import Image

from .bay_clearance import visible_road_component
from .curb_recess import _vehicles


RECIPE = "visible_painted_edge_parking_row_extension_v1"
GROUND = (10, 13, 24)
OUTSIDE_KERB = (11, 15, 26, 29, 30)
MOTOR_CLASSES = (54, 55, 56, 57, 60, 61, 62, 63)


def _locally_planted(image, bounds):
    # Windshield glare may tint several nearly neutral channels together. A
    # visibly green plant has a substantially stronger green channel locally.
    crop = np.asarray(image.crop(bounds).convert("RGB"), dtype=np.float32)
    green = crop[:,:,1]
    excess = green - np.maximum(crop[:,:,0], crop[:,:,2])
    return np.mean(excess > np.maximum(22., .17 * green)) >= .07


@dataclass(frozen=True)
class Section:
    y: int
    paint_start: int
    paint_end: int
    kerb: int
    kerb_end: int
    paint_contrast: float
    surface_contrast: float
    kerb_contrast: float

    @property
    def width(self):
        return self.kerb - self.paint_end


def _runs(row, category):
    transitions = np.diff(np.r_[False, row == category, False].astype(np.int8))
    return zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))


def _sections(y, labels, available, connected, gray, planted):
    """Inspect from travel lane outwards. All columns below use this orientation."""
    row = labels[y]
    w = len(row)
    sections = []
    for first, end in _runs(row, 24):
        if first < .06 * w or end - first > .030 * w:
            continue
        kerbs = [(a, b) for a, b in _runs(row, 2) if end + .030 * w < a < end + .32 * w]
        if not kerbs:
            continue
        outer, kerb_end = kerbs[0]
        width = outer - end
        exterior = row[kerb_end:min(w, kerb_end + max(4, round(.13 * width)))]
        if len(exterior) < 4 or np.mean(np.isin(exterior, OUTSIDE_KERB)) < .6:
            continue
        if np.any(row[kerb_end:min(w, kerb_end + round(.65 * width))] == 4):
            continue
        # A second thin painted line may follow the edge of the kerb itself.
        if not np.isin(row[end:outer], GROUND).all() or np.mean(row[end:outer] == 24) > .22:
            continue
        access = first - max(7, round(.38 * width))
        if (access < 1 or not np.all(row[access:first] == 13)
                or not connected[y, access] or not available[y, access:outer].all()):
            continue
        # An unmapped planted divider can run alongside the paint. Inspect the
        # whole path from the carriageway to the stopping strip on this scanline.
        # The caller requires vehicle-length continuity across admissible rows.
        green = planted[y, access:outer]
        if len(green) >= 3 and np.any(green[:-2] & green[1:-1] & green[2:]):
            continue
        if end - first > .22 * width or kerb_end - outer < 2:
            continue
        lane = np.median(gray[y, access:first - 2])
        near_edge = gray[y, end + 2:end + max(4, round(.10 * width))]
        stripe = gray[y, max(0, first - 2):min(w, end + 2)]
        if not len(near_edge) or not len(stripe):
            continue
        paint_contrast = float(np.percentile(stripe, 85) - max(lane, np.median(near_edge)))
        def part(low, high):
            return np.median(gray[y, end + round(low * width):end + round(high * width)])
        contrast_a, contrast_b = part(.17, .34) - lane, part(.69, .83) - lane
        # A concentrated patch of sun or glare on an otherwise identical road
        # cannot, by itself, establish a different surface across the stripe.
        surface_contrast = min(abs(contrast_a), abs(contrast_b)) if contrast_a * contrast_b > 0 else 0.
        kerb_pixels = gray[y, max(0, outer - 2):min(w, kerb_end + 2)]
        kerb_contrast = float(np.ptp(np.percentile(kerb_pixels, [15, 85])))
        sections.append(Section(y, int(first), int(end), int(outer), int(kerb_end),
                                paint_contrast, float(surface_contrast), kerb_contrast))
    return sections


def _overlap(a, b):
    l, t = max(a[0], b[0]), max(a[1], b[1])
    r, d = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, r - l) * max(0, d - t)
    return intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)


def _oriented_car(car, w, flipped):
    l, t, r, b = car.box
    return (w - 1 - r, t, w - 1 - l, b) if flipped else (l, t, r, b)


def _pair(a, b):
    ca, cb = (a[0] + a[2]) / 2, (b[0] + b[2]) / 2
    wa, wb, ha = a[2] - a[0], b[2] - b[0], a[3] - a[1]
    delta = a[3] - b[3]
    if not (.15 * ha <= delta <= 1.35 * ha and .40 * wa <= wb < .93 * wa
            and .12 * wa < ca - cb < 1.20 * wa and .65 <= wa / ha <= 1.85
            and _overlap(a, b) < .38):
        return None
    slope = (ca - cb) / delta
    return (ca, slope) if .15 < slope < 4.7 else None


def _observed_boundaries(labels, box, inner, outer):
    """Project both measured edges to the vehicles and require a visible edge at their wheels."""
    y = box[3]
    width = box[2] - box[0]
    lower_body = labels[round(box[1] + .55 * (box[3] - box[1])):int(np.ceil(y)),
                        int(box[0]):int(np.ceil(box[2]))]
    if not lower_body.size or np.mean(np.isin(lower_body, MOTOR_CLASSES)) < .28:
        return False
    x0, x1 = np.polyval(inner, y), np.polyval(outer, y)
    half = max(4, round(.20 * width))
    if not .72 * width <= x1 - x0 <= 1.48 * width:
        return False
    contact = (box[0] + box[2]) / 2
    if not x0 + .04 * width < contact < x1 - .04 * width:
        return False
    y0, y1 = max(0, round(y) - 2), min(labels.shape[0], round(y) + 3)
    def around(x, category):
        xm = round(x)
        crop = labels[y0:y1, max(0, xm - half):min(labels.shape[1], xm + half + 1)]
        return crop.size and (crop == category).any(axis=1).sum() >= 2
    return around(x0, 24) or around(x1, 2)


def _measure(a, b, get_row, labels, gray_shape):
    trajectory = _pair(a, b)
    if trajectory is None:
        return None
    cx, direction = trajectory
    height = a[3] - a[1]
    width = a[2] - a[0]
    first = max(2, int(np.ceil(a[3])))
    last = min(gray_shape[0] - 2, int(a[3] + 1.43 * height))
    if last - first < .77 * height:
        return None
    rows = []
    for y in range(first, last + 1):
        predicted_x = cx + direction * (y - a[3])
        possibilities = [s for s in get_row(y) if .62 * width < s.width < 3.4 * width
                         and s.paint_end - .23 * width < predicted_x < s.kerb + .23 * width]
        if len(possibilities) != 1:
            if rows:
                break
            if y - first > .24 * height:
                return None
            continue
        current = possibilities[0]
        if rows and (abs(rows[-1].paint_end - current.paint_end) > .15 * width
                     or abs(rows[-1].kerb - current.kerb) > .17 * width):
            break
        rows.append(current)
    if len(rows) < max(13, .75 * height):
        return None
    y = np.array([s.y for s in rows])
    inside = np.array([s.paint_end for s in rows])
    outside = np.array([s.kerb for s in rows])
    inner, outer = np.polyfit(y, inside, 1), np.polyfit(y, outside, 1)
    if (inner[0] <= .12 or outer[0] <= inner[0] + .12 or
            np.max(abs(inside - np.polyval(inner, y))) > .13 * width or
            np.max(abs(outside - np.polyval(outer, y))) > .18 * width):
        return None
    if not (_observed_boundaries(labels, a, inner, outer) and _observed_boundaries(labels, b, inner, outer)):
        return None
    optical = np.asarray([[s.paint_contrast, s.surface_contrast, s.kerb_contrast] for s in rows])
    if (np.median(optical[:,0]) < 10 or np.mean(optical[:,0] >= 7) < .65
            or np.median(optical[:,1]) < 12 or np.mean(optical[:,1] >= 8) < .64
            or np.median(optical[:,2]) < 10):
        return None
    return rows, np.median(optical, axis=0)


def _run(original, classes, context, marked_xy=None):
    if (not isinstance(classes,np.ndarray) or classes.ndim!=2 or min(classes.shape)<64
            or not isinstance(context,dict) or tuple(context.get("size",()))!=original.size
            or "current_vehicles" not in context or "current_boxes" not in context):
        return []
    h,w = classes.shape
    scale = np.array([w,h]) / np.array(original.size)
    try:
        cars, boxes = _vehicles(context, scale, np.asarray(original.size), h, w)
    except (ValueError, TypeError):
        return []
    # Restrict this ambiguous shoulder class to a kerbed, built-up street.
    if len(cars)<2 or any(np.mean(classes==c) < limit for c,limit in ((2,.001),(24,.0008),(17,.012))):
        return []
    road = visible_road_component(classes)
    if road is None:
        return []
    free = np.ones((h,w), np.bool_)
    for l,t,r,b in boxes:
        free[int(t):int(np.ceil(b))+1, int(l):int(np.ceil(r))+1] = False
    gray = np.asarray(original.convert("L").resize((w,h),Image.Resampling.BOX), dtype=np.float32)
    rgb = np.asarray(original.convert("RGB").resize((w,h),Image.Resampling.BOX),dtype=np.float32)
    green = rgb[:,:,1]
    planted = green - np.maximum(rgb[:,:,0],rgb[:,:,2]) > np.maximum(22.,.17*green)
    out = []
    for flipped in (False, True):
        order = slice(None,None,-1) if flipped else slice(None)
        lab = classes[:,order]
        free_side, road_side, gray_side = free[:,order], road[:,order], gray[:,order]
        @lru_cache(maxsize=None)
        def get_row(y):
            return _sections(y, lab, free_side, road_side, gray_side, planted[:,order])
        def native(x,y):
            px = w - 1 - x if flipped else x
            return [round((px+.5)/scale[0]-.5,3),round((y+.5)/scale[1]-.5,3)]
        for near,far in permutations(cars,2):
            a = _oriented_car(near,w,flipped)
            b = _oriented_car(far,w,flipped)
            measured = _measure(a,b,get_row,lab,(h,w))
            if measured is None:
                continue
            rows, optical = measured
            row_map = {row.y:row for row in rows}
            if marked_xy is None:
                y = round(a[3] + .70 * near.height)
                row = row_map.get(y)
                if row is None:
                    continue
                x = round(row.paint_end + .58 * row.width)
                point = native(x,y)
            else:
                x,y = np.rint((np.asarray(marked_xy)+.5)*scale-.5).astype(int)
                if flipped:
                    x = w - 1 - x
                row = row_map.get(y)
                if row is None or y-a[3]<.25*near.height:
                    continue
                point = list(marked_xy)
            radius = max(2, min(6, round(.060 * near.width)))
            if (x - row.paint_end < max(radius+1,.15*row.width)
                    or row.kerb - x < max(radius+1,.15*row.width)):
                continue
            native_edges = [native(x-radius,y-radius),native(x+radius,y+radius)]
            l,r = sorted(p[0] for p in native_edges)
            t,d = (p[1] for p in native_edges)
            if _locally_planted(original,(int(l),int(t),int(np.ceil(r)),int(np.ceil(d)))):
                continue
            evidence = {"recipe":RECIPE,"extent":"interior_point_only","nearest_row_track_id":near.track,
                        "supporting_row_track_ids":[near.track,far.track],"relation":"toward_camera_along_outer_row",
                        "signed_center_class":int(lab[y,x]),"observed_clear_rows":[rows[0].y,rows[-1].y],
                        "boundary_samples_native":[[native(s.paint_end,s.y),native(s.kerb,s.y)]
                                                   for s in (rows[0],row,rows[-1])],
                        "median_paint_surface_kerb_contrast":[round(float(v),2) for v in optical]}
            option = {"point":point,"box":[l,t,r,d],"support_patch":[[l,t],[r,t],[r,d],[l,d]],"evidence":evidence}
            if not any(o["evidence"]["nearest_row_track_id"] == near.track for o in out):
                out.append(option)
    return sorted(out,key=lambda item:item["point"])


def candidates(original, classes, context):
    return _run(original,classes,context)


def marked(original, classes, context, mark):
    if not isinstance(mark,(tuple,list)) or len(mark)!=4:
        return None
    try:
        l,t,r,b = map(float,mark)
    except (ValueError,TypeError):
        return None
    w,h = original.size
    if not (0<=l<r<=w and 0<=t<b<=h and r-l<=.012*w and b-t<=.012*h):
        return None
    target = [(l+r)/2,(t+b)/2]
    matches = [o for o in _run(original,classes,context,target)
               if o["box"][0]<=l<=r<=o["box"][2] and o["box"][1]<=t<=b<=o["box"][3]]
    return matches[0] if len(matches)==1 else None
