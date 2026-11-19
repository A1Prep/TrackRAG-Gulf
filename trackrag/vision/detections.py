"""Detection records and the post-processing applied before tracking.

Every step here works on one frame at a time, so it cannot use information from later
frames.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .classes import COCO_VEHICLE_TO_EMT, EMT_CLASSES
from .geometry import Box, union_box

RIDER_MERGE_IOU = 0.1  # EMT authors' joint person + bike threshold (tracking/eval_detector.py)
DUPLICATE_NMS_IOU = 0.8


@dataclass(frozen=True)
class Detection:
    """One detector output. ``name`` is in the current label space; ``native`` keeps the
    detector's own class name (for COCO merges, e.g. ``"motorcycle+person"``)."""

    box: Box
    score: float
    name: str
    native: str = ""

    def __post_init__(self) -> None:
        box = tuple(float(value) for value in self.box)
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError("Detection box needs four finite coordinates")
        if box[2] < box[0] or box[3] < box[1]:
            raise ValueError("Detection box must have right >= left and bottom >= top")
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Detection score must be within [0, 1]")
        object.__setattr__(self, "box", box)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "native", str(self.native or self.name))


def _iou(a: Box, b: Box) -> float:
    """IoU of two already-validated boxes (fast path; ``geometry.iou`` validates its inputs)."""
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    if inter_w <= 0:
        return 0.0
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def _score_order(detections: Sequence[Detection]) -> list[int]:
    return sorted(range(len(detections)), key=lambda i: (-detections[i].score, detections[i].box, detections[i].name, i))


def nms(detections: Sequence[Detection], iou_threshold: float = DUPLICATE_NMS_IOU) -> list[Detection]:
    """Class-agnostic greedy suppression: a box is dropped when its IoU with an already kept,
    higher-ranked box is >= ``iou_threshold``. Ranking: score, then box, then name (deterministic).

    RF-DETR selects top-k over (query, class) pairs, so one object can be returned twice
    with different classes. A high threshold removes those duplicates while keeping
    neighbouring vehicles.
    """
    kept: list[Detection] = []
    for index in _score_order(detections):
        candidate = detections[index]
        if all(_iou(candidate.box, other.box) < iou_threshold for other in kept):
            kept.append(candidate)
    return kept


def coco_to_emt(detections: Iterable[Detection], merge_iou: float = RIDER_MERGE_IOU) -> tuple[list[Detection], Counter]:
    """Map COCO-named detections to EMT names, following the EMT authors' published rule.

    bicycle + person (IoU > 0.1) -> Cyclist (union box, mean score); motorcycle + person ->
    Motorbike; an unpaired motorcycle -> Motorbike; an unpaired bicycle is dropped; any
    remaining person -> Pedestrian; car/bus/truck/train via ``COCO_VEHICLE_TO_EMT``.

    Differences from the authors' code (tracking/eval_detector.py): bicycles and
    motorcycles are visited in ONE pass by descending score (the authors pair all bicycles
    first, so a low-score duplicate "bicycle" could steal a motorcycle's rider); each bike
    takes its highest-IoU free person; a person joins at most one bike. Callers should also
    run ``nms`` on the raw detections first to drop same-box duplicates.
    Returns the mapped detections and a count of dropped native names.
    """
    persons: list[Detection] = []
    bikes: list[tuple[Detection, str, bool]] = []
    vehicles: list[Detection] = []
    dropped: Counter = Counter()
    for detection in detections:
        if detection.name == "person":
            persons.append(detection)
        elif detection.name == "bicycle":
            bikes.append((detection, "Cyclist", False))
        elif detection.name == "motorcycle":
            bikes.append((detection, "Motorbike", True))
        elif detection.name in COCO_VEHICLE_TO_EMT:
            vehicles.append(detection)
        else:
            dropped[detection.name or "<empty>"] += 1
    used_people: set[int] = set()
    output: list[Detection] = []
    order = sorted(range(len(bikes)), key=lambda i: (-bikes[i][0].score, bikes[i][0].box, bikes[i][1], i))
    for index in order:
        bike, emt_name, keep_unpaired = bikes[index]
        best: tuple[float, int] | None = None
        for person_index, person in enumerate(persons):
            if person_index in used_people:
                continue
            overlap = _iou(bike.box, person.box)
            if overlap > merge_iou and (best is None or overlap > best[0]):
                best = (overlap, person_index)
        if best is not None:
            person = persons[best[1]]
            used_people.add(best[1])
            output.append(Detection(
                union_box(bike.box, person.box), (bike.score + person.score) / 2, emt_name,
                native=f"{bike.native}+{person.native}",
            ))
        elif keep_unpaired:
            output.append(Detection(bike.box, bike.score, emt_name, native=bike.native))
        else:
            dropped["bicycle (unpaired)"] += 1
    for index, person in enumerate(persons):
        if index not in used_people:
            output.append(Detection(person.box, person.score, "Pedestrian", native=person.native))
    for vehicle in vehicles:
        output.append(Detection(vehicle.box, vehicle.score, COCO_VEHICLE_TO_EMT[vehicle.name], native=vehicle.native))
    return [output[i] for i in _score_order(output)], dropped


def drop_ego_vehicle(detections: Iterable[Detection], width: float, height: float,
                     bottom_fraction: float = 0.97, min_width_fraction: float = 0.6) -> tuple[list[Detection], int]:
    """Remove the camera car's own bonnet: a box reaching the bottom ``1 - bottom_fraction`` of the
    frame AND at least ``min_width_fraction`` of its width. Boxes are in original-image pixels.
    Applied identically to every model; the count is reported in provenance."""
    kept, dropped = [], 0
    for detection in detections:
        left, _, right, bottom = detection.box
        if bottom >= bottom_fraction * height and (right - left) >= min_width_fraction * width:
            dropped += 1
        else:
            kept.append(detection)
    return kept, dropped


def keep_emt(detections: Iterable[Detection]) -> tuple[list[Detection], Counter]:
    """For a fine-tuned EMT checkpoint: drop names outside EMT (e.g. ``__background__``)."""
    kept, dropped = [], Counter()
    for detection in detections:
        if detection.name in EMT_CLASSES:
            kept.append(detection)
        else:
            dropped[detection.name or "<empty>"] += 1
    return kept, dropped
