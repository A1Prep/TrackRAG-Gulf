"""Recognize visible direction arrows inside a measured curbside strip."""
import cv2
import numpy as np


def _head_and_shaft(mask):
    """Look for a tapering head followed by a sustained, distinctly narrower shaft."""
    widths = np.count_nonzero(mask, axis=1)
    length = len(widths)
    for profile in (widths, widths[::-1]):
        # A transverse stall edge reaches its full width immediately. A road
        # arrow widens from a tip then narrows to the stem on the other end.
        peak = int(np.argmax(profile))
        if not .12 * length <= peak <= .62 * length:
            continue
        head = float(profile[peak])
        tip = profile[:max(1, round(.10 * length))]
        stem = profile[max(peak + 2, round(.70 * length)):max(peak + 3, round(.95 * length))]
        if len(stem) < 3 or head < 5 or np.median(tip) > .47 * head:
            continue
        if not .10 * head <= np.median(stem) <= .62 * head or np.quantile(stem, .85) > .72 * head:
            continue
        head_base = profile[max(1, round(.60 * peak)):peak + 1]
        if np.median(head_base) >= .65 * head:
            return True
    return False


def in_measured_strip(rgb, classes, sections, anchor_box, frame_size, direction):
    """Check for an optically corroborated travel arrow inside this specific parking candidate.

    The colour image is registered to the class map. Sections use map pixels,
    and the anchor box uses the original photograph's pixel coordinates.
    """
    if len(sections) < 3 or direction not in (-1, 1):
        return False
    height, width = classes.shape
    if rgb.shape != (height, width, 3):
        return False
    original_width, original_height = frame_size
    bh = (anchor_box[3] - anchor_box[1]) * height / original_height
    bw = (anchor_box[2] - anchor_box[0]) * width / original_width
    bottom = anchor_box[3] * height / original_height
    y0 = max(0, int(bottom))
    y1 = min(height, int(np.ceil(bottom + 1.70 * bh)) + 1)
    if y1 - y0 < 8:
        return False
    rows = np.asarray([s.y for s in sections], dtype=float)
    edges = np.asarray([(s.lane, s.curb) for s in sections], dtype=float)
    line = np.polyfit(rows - bottom, edges, 1)
    lane, curb = ((np.arange(y0, y1) - bottom)[:, None] * line[0] + line[1]).T
    span = curb - lane
    usable = (direction * span > .42 * bw) & (direction * span < 3.0 * bw)
    if not usable.any():
        return False
    x0 = max(0, int(np.floor(min(lane[usable].min(), curb[usable].min()))) - 2)
    x1 = min(width, int(np.ceil(max(lane[usable].max(), curb[usable].max()))) + 3)
    if x1 - x0 < 8:
        return False
    fraction = (np.arange(x0, x1)[None, :] - lane[:, None]) / np.where(usable, span, 1)[:, None]
    inside = usable[:, None] & (fraction > -.05) & (fraction < 1.05)
    surface = classes[y0:y1, x0:x1]
    paint = np.uint8((surface == 24) & inside)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(paint, connectivity=8)
    for index in range(1, count):
        left, top, cw, ch, area = stats[index]
        if ch < max(8, .18 * bh) or cw < 5 or area < 20 or area > 1.6 * bw * bh:
            continue
        component = labels[top:top + ch, left:left + cw] == index
        occupancy = fraction[top:top + ch, left:left + cw][component]
        if occupancy.min() < .09 or occupancy.max() > .91 or occupancy.max() - occupancy.min() < .13:
            continue
        if not _head_and_shaft(component):
            continue
        # Read contrast against the nearby real ground, not an inferred bright
        # paint label. Suppress lane dividers and markings outside this corridor.
        xmin, xmax = max(0, left - 4), min(paint.shape[1], left + cw + 4)
        ymin, ymax = max(0, top - 4), min(paint.shape[0], top + ch + 4)
        local = labels[ymin:ymax, xmin:xmax] == index
        adjacent = cv2.dilate(local.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        adjacent &= ~local & np.isin(surface[ymin:ymax, xmin:xmax], (10, 13))
        if np.count_nonzero(adjacent) < 8:
            continue
        brightness = rgb[y0 + ymin:y0 + ymax, x0 + xmin:x0 + xmax].mean(axis=2)
        if np.median(brightness[local]) >= np.median(brightness[adjacent]) + 15:
            return True
    return False
