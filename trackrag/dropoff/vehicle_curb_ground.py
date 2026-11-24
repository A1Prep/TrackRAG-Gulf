"""Measure the visible curbside strip behind a detected vehicle at its own image scale."""
from typing import NamedTuple

import numpy as np

from . import bay_clearance, single_vehicle


class _Section(NamedTuple):
    """One horizontal stretch of free ground between the outer curb and the lane paint."""

    row: int
    left: int
    right: int
    curb_direction: int
    parking_fraction: float


def _section_at_row(classes, ground, free, row, center_x, vehicle_width, offsets, direction):
    map_width = classes.shape[1]
    curb_columns = np.rint(center_x + direction * offsets).astype(int)
    lane_columns = np.rint(center_x - direction * offsets).astype(int)
    curb_columns = curb_columns[(curb_columns >= 1) & (curb_columns < map_width - 1)]
    lane_columns = lane_columns[(lane_columns >= 1) & (lane_columns < map_width - 1)]
    if not len(curb_columns) or not len(lane_columns):
        return None

    curb_hits = curb_columns[np.isin(classes[row, curb_columns], (2, 15))]
    paint_hits = lane_columns[classes[row, lane_columns] == 24]
    if not len(curb_hits) or not len(paint_hits):
        return None

    paint_edge = int(paint_hits[0])
    left, right = sorted((int(curb_hits[0]), paint_edge))
    if not 0.68 * vehicle_width <= right - left <= 2.35 * vehicle_width:
        return None

    interior = slice(left + 2, right - 1)
    if ground[row, interior].mean() < 0.97 or free[row, interior].mean() < 0.91:
        return None
    parking_fraction = np.mean(classes[row, interior] == 10)
    if parking_fraction < 0.15:
        return None

    # Visible roadway on the other side distinguishes the lane edge from paint
    # between two stalls and from a crossing stripe.
    lane_offsets = np.arange(3, max(6, int(0.24 * vehicle_width)))
    lane_columns = np.clip(paint_edge - direction * lane_offsets, 0, map_width - 1)
    if np.mean(classes[row, lane_columns] == 13) >= 0.62:
        return _Section(row, left, right, direction, float(parking_fraction))
    return None


def _sections(classes, road, box, occupied, size):
    map_height, map_width = classes.shape
    scale = np.array([map_width / size[0], map_height / size[1]])
    (left, top), (right, bottom) = np.asarray(box).reshape(2, 2) * scale
    vehicle_width, vehicle_height = right - left, bottom - top
    center_x = (left + right) / 2
    if (vehicle_width < 0.05 * map_width or vehicle_height < 0.065 * map_height
            or not 0.15 * map_width < center_x < 0.95 * map_width):
        return []

    ground = np.isin(classes, [10, 13, 24])
    free = road.copy()
    for tracked_box in occupied:
        (x0, y0), (x1, y1) = np.asarray(tracked_box).reshape(2, 2) * scale
        rows = slice(max(0, int(y0)), min(map_height, int(np.ceil(y1))))
        columns = slice(max(0, int(x0)), min(map_width, int(np.ceil(x1))))
        free[rows, columns] = False

    # Try the outer curb on either side of the vehicle, keeping unambiguous rows.
    offsets = np.arange(max(2, int(0.08 * vehicle_width)), max(4, int(1.47 * vehicle_width)))
    start_row = max(1, int(bottom + 0.06 * vehicle_height))
    stop_row = min(map_height - 1, int(bottom + 0.76 * vehicle_height) + 1)
    sections = []
    for row in range(start_row, stop_row):
        matches = []
        for direction in (-1, 1):
            section = _section_at_row(classes, ground, free, row, center_x, vehicle_width, offsets, direction)
            if section is not None:
                matches.append(section)
        if len(matches) == 1:
            sections.append(matches[0])

    # Consecutive rows must describe the same side and follow both visible edges.
    runs = []
    for section in sections:
        if not runs:
            runs.append([section])
            continue
        previous = runs[-1][-1]
        edge_shift = max(abs(section.left - previous.left), abs(section.right - previous.right))
        if (section.curb_direction != previous.curb_direction
                or section.row - previous.row > max(2, 0.035 * vehicle_height)
                or edge_shift > 0.18 * vehicle_width):
            runs.append([])
        runs[-1].append(section)

    minimum_depth = max(5, 0.08 * vehicle_height)
    return [
        run for run in runs
        if len(run) >= minimum_depth
        and run[-1].row - run[0].row >= minimum_depth
        and np.mean([section.parking_fraction for section in run]) >= 0.30
    ]


def _ground_in_run(run, map_shape, size, occupied):
    map_height, map_width = map_shape

    def original_point(x, y):
        return [round((x + 0.5) * size[0] / map_width, 2), round((y + 0.5) * size[1] / map_height, 2)]

    middle = run[len(run) // 2]
    center = original_point((middle.left + middle.right) / 2, middle.row)
    if any(left <= center[0] <= right and top <= center[1] <= bottom for left, top, right, bottom in occupied):
        return None

    # Only the observed ground between measured rows is represented in the polygon.
    far, near = run[0], run[-1]
    polygon = [
        original_point(far.left + 2, far.row),
        original_point(far.right - 2, far.row),
        original_point(near.right - 2, near.row),
        original_point(near.left + 2, near.row),
    ]
    return {"point": center, "polygon": polygon, "basis": "curb_and_lane_divider"}


def describe(scene, classes, *, vehicles=None):
    """Order visible road users independently of whether any ground can be measured."""
    frame = scene["frame"]
    size = (frame["width"], frame["height"])
    context = single_vehicle.context({"scene": scene})
    road = bay_clearance.visible_road_component(classes)
    if vehicles is None:
        vehicles = [
            (track_id, kind, box) for track_id, kind, box in single_vehicle.motor_rows(context)
            if box[2] - box[0] >= 0.047 * size[0] and box[3] - box[1] >= 0.055 * size[1]
        ][:4]
    occupied_boxes = [track[2] for track in context["tracks"]]

    specs = []
    for track_id, kind, box in vehicles:
        ground = None
        runs = _sections(classes, road, box, occupied_boxes, size) if road is not None else []
        if runs:
            best = max(runs, key=lambda run: len(run) * np.mean([section.parking_fraction for section in run]))
            ground = _ground_in_run(best, classes.shape, size, occupied_boxes)
        if ground is None:
            rear = single_vehicle.behind_patch(classes, box, occupied_boxes, size)
            if rear:
                left, top, right, bottom = rear["box"]
                polygon = [[left, top], [right, top], [right, bottom], [left, bottom]]
                ground = {"point": rear["point"], "polygon": polygon, "basis": rear["kind"]}

        specs.append({"track_id": track_id, "class": kind, "box": list(map(float, box)), "ground": ground})
    return specs
