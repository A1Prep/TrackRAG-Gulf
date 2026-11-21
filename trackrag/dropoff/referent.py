"""Rank predicted objects in a current frame as a second opinion for clarification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import validate_request
from .spatial import instruction_terms


KIND = "public-predicted-referents-listwise-v1"
CLASS_WORDS = {
    "car": "car van minivan cab taxi sedan suv jeep hatchback ambulance pickup",
    "truck": "truck lorry pickup dump semi tanker",
    "bus": "bus coach shuttle",
    "person": "person pedestrian man woman lady gentleman guy boy girl child kid worker officer cop jogger runner people",
    "bicycle": "bike bicycle cyclist cycles bicyclist biker cycling",
    "motorcycle": "motorcycle motorbike scooter motorcyclist moped biker",
    "barrier": "barrier barricade fence barricades",
    "cone": "cone cones pylon",
    "trailer": "trailer trailers",
    "construction": "construction excavator bulldozer crane tractor",
    "other": "",
}
COLORS = ("white", "black", "gray", "red", "blue", "green", "yellow")
CLASSES = tuple(CLASS_WORDS)
MENTIONS = {k: set(v.split()) for k, v in CLASS_WORDS.items()}
REGION_NAMES = (
    "none", "cx", "cy", "width", "height", "area_root", "log_area",
    "left", "right", "top", "bottom", "cx2", "cy2", "center_distance",
    "aspect_log", "position_rank", "size_rank",
    *("class_" + c for c in CLASSES), *("color_" + c for c in COLORS),
    "r", "g", "b", "saturation", "brightness", "contrast",
)
ALIGN_NAMES = ("noun_agreement", "named_class_disagreement", "vehicle_agreement", "colour_agreement",
               "colour_central", "colour_agreement_with_noun", "left_cue", "right_cue",
               "front_cue", "far_cue")
DEFAULT_REG = {"base": 0.7, "words": 3.0, "align": 0.7}


def category(name):
    w = name.lower().replace("-", "_").replace(" ", "_")
    if "pedestrian" in w or w in ("person", "human", "rider"):
        return "person"
    if "bicycle_rack" in w:
        return "other"
    if "motorcycle" in w or "motorbike" in w:
        return "motorcycle"
    if "bicycle" in w or w in ("bike", "cyclist"):
        return "bicycle"
    if "bus" in w:
        return "bus"
    if "construction" in w:
        return "construction"
    if "trailer" in w:
        return "trailer"
    if "truck" in w or w in ("lorry", "pickup"):
        return "truck"
    if "barrier" in w:
        return "barrier"
    if "cone" in w:
        return "cone"
    if w in ("car", "van", "taxi", "suv", "vehicle.car", "vehicle.emergency.police",
             "pickup_truck", "light_vehicle", "motorcar"):
        return "car" if w != "pickup_truck" else "truck"
    return "other"


def overlap(boxes, reference):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    r = np.asarray(reference, dtype=np.float64)
    inter = np.maximum(0, np.minimum(boxes[:, 2:], r[2:]) - np.maximum(boxes[:, :2], r[:2])).prod(axis=1)
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) + np.prod(r[2:] - r[:2])
    return np.divide(inter, np.maximum(areas - inter, 1e-10))


def words(text):
    return set(instruction_terms(text))


def _appearance(image, boxes):
    # Sample both an interior grid and the middle body, avoiding most adjacent road pixels.
    rgb = np.asarray(image.convert("RGB").resize((480, 270), Image.Resampling.BILINEAR))
    xs, ys = np.meshgrid(np.linspace(.12, .88, 8), np.linspace(.22, .83, 8))
    px = np.clip(((boxes[:, 0, None] + xs.ravel() * (boxes[:, 2] - boxes[:, 0])[:, None])
                  * rgb.shape[1]).astype(int), 0, rgb.shape[1] - 1)
    py = np.clip(((boxes[:, 1, None] + ys.ravel() * (boxes[:, 3] - boxes[:, 1])[:, None])
                  * rgb.shape[0]).astype(int), 0, rgb.shape[0] - 1)
    samples = rgb[py, px].astype(np.float64) / 255
    r, g, b = samples.transpose(2, 0, 1)
    hi, lo = samples.max(axis=2), samples.min(axis=2)
    chroma = hi - lo
    sat = chroma / np.maximum(.05, hi)
    neutral = sat <= .23
    masks = np.stack((neutral & (hi >= .70), (hi < .29), neutral & (hi >= .29) & (hi < .70),
                      (r > g * 1.14) & (r > b * 1.14) & (sat > .23),
                      (b > r * 1.12) & (b > g * 1.06) & (sat > .23),
                      (g > r * 1.08) & (g > b * 1.08) & (sat > .23),
                      (r > b * 1.30) & (g > b * 1.23) & (r > .80*g) & (g > .58*r)), axis=2)
    center = (xs.ravel() >= .30) & (xs.ravel() <= .72) & (ys.ravel() >= .40) & (ys.ravel() <= .75)
    return (masks.mean(axis=1), masks[:, center].mean(axis=1),
            np.column_stack((samples.mean(axis=1), sat.mean(axis=1), hi.mean(axis=1), hi.std(axis=1))))


def region_features(image, boxes, categories):
    box = np.asarray(boxes, dtype=np.float64).reshape(-1, 4) / np.array(image.size * 2)
    n = len(box)
    regions = np.zeros((n + 1, len(REGION_NAMES)), dtype=np.float64)
    regions[-1, 0] = 1  # Separate trainable "no matching detector proposal" alternative.
    if not n:
        return regions, np.zeros((0, len(COLORS))), np.zeros((0, len(COLORS)))
    x, y, r, b = box.T
    w, h = r - x, b - y
    cx, cy = (x+r)/2, (y+b)/2
    pos = np.argsort(np.argsort(cx, kind="stable"), kind="stable")/max(1, n-1)
    scale = np.argsort(np.argsort(w*h, kind="stable"), kind="stable")/max(1, n-1)
    geometry = np.column_stack((np.zeros(n), cx, cy, w, h, np.sqrt(w*h),
                                 np.log(np.maximum(1e-7, w*h))/8, x, r, y, b, cx*cx, cy*cy,
                                 abs(cx-.5), np.clip(np.log(np.maximum(w, 1e-7)/np.maximum(h, 1e-7)) / 3,-2,2),
                                 pos, scale))
    classes = np.equal.outer(categories, CLASSES).astype(float)
    hist, center, color = _appearance(image, box)
    regions[:-1] = np.column_stack((geometry, classes, hist, color))
    return regions, hist, center


def alignment(text, categories, regions, hist, centers):
    result = np.zeros((len(regions), len(ALIGN_NAMES)), dtype=float)
    if not categories:
        return result
    present = words(text)
    mentioned = {name for name, aliases in MENTIONS.items() if present & aliases}
    mentioned_colors = set(present) | ({"gray"} if "silver" in present else set())
    color_ids = [i for i, c in enumerate(COLORS) if c in mentioned_colors]
    colors = hist[:, color_ids].max(axis=1) if color_ids else np.zeros(len(categories))
    middle = centers[:, color_ids].max(axis=1) if color_ids else np.zeros(len(categories))
    nouns = np.asarray([c in mentioned for c in categories], dtype=float)
    vehicle = np.asarray([c in {"car", "truck", "bus", "trailer", "construction", "motorcycle"}
                          for c in categories]) * ("vehicle" in present)
    cx, cy = regions[:-1, 1], regions[:-1, 2]
    result[:-1] = np.column_stack((nouns, (1-nouns)*bool(mentioned), vehicle, colors, middle,
                                   colors*nouns, (1-cx)*bool(present & {"left", "leftmost"}),
                                   cx*bool(present & {"right", "rightmost"}),
                                   cy*bool(present & {"closest", "nearest", "near"}),
                                   (1-cy)*bool(present & {"far", "furthest", "farthest"})))
    return result


def encode(text, vocabulary):
    present = words(text)
    return np.asarray([float(w in present) for w in vocabulary], dtype=np.float64)


def ordered_alternatives(boxes, categories, probabilities):
    # Pairwise overlaps are cheap on a bounded current-frame proposal list.
    box = np.asarray(boxes, dtype=float).reshape(-1, 4)
    groups = np.asarray(categories, dtype=str)
    extent = np.maximum(0., np.minimum(box[:, None, 2:], box[None, :, 2:])
                        - np.maximum(box[:, None, :2], box[None, :, :2]))
    intersections = extent.prod(axis=2)
    areas = np.prod(box[:, 2:] - box[:, :2], axis=1)
    same = ((groups[:, None] == groups[None, :])
            & (intersections >= .65 * (areas[:, None] + areas[None, :] - intersections)))
    suppressed = np.zeros(len(box), dtype=bool)
    distinct = []
    for index in np.argsort(-np.asarray(probabilities), kind="stable"):
        if not suppressed[index]:
            distinct.append(int(index))
            suppressed |= same[index]
    return distinct


class ReferentRanker:
    def __init__(self, payload, digest=None):
        if payload.get("kind") != KIND or payload.get("region_names") != list(REGION_NAMES) or payload.get("align_names") != list(ALIGN_NAMES):
            raise ValueError("Unsupported referring-object model or feature schema")
        self.vocabulary = payload["vocabulary"]
        if (not isinstance(self.vocabulary, list) or len(self.vocabulary) > 512
            or not all(isinstance(w, str) for w in self.vocabulary)
            or len(set(self.vocabulary)) != len(self.vocabulary)):
            raise ValueError("Invalid referring-object vocabulary")
        self.base = np.asarray(payload["base"], dtype=float)
        self.interactions = np.asarray(payload["interactions"], dtype=float).reshape(len(self.vocabulary), len(REGION_NAMES))
        self.aligned = np.asarray(payload["aligned"], dtype=float)
        if self.base.shape != (len(REGION_NAMES),) or self.aligned.shape != (len(ALIGN_NAMES),):
            raise ValueError("Invalid referring-object coefficients")
        if not all(np.isfinite(x).all() for x in (self.base, self.interactions, self.aligned)):
            raise ValueError("Nonfinite referring-object coefficient")
        self.sha256 = digest

    @classmethod
    def load(cls, path, *, expected_sha256=None):
        content = Path(path).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("Referring-object checkpoint SHA256 mismatch")
        return cls(json.loads(content), digest)

    def scores(self, text, regions, aligned):
        logits = regions @ (self.base + encode(text, self.vocabulary) @ self.interactions) + aligned @ self.aligned
        positive = np.exp(logits - np.max(logits))
        return positive / positive.sum()

    def rank(self, image, text, tracks, *, frame):
        """Rank this verified original image's predicted observations; ignore earlier frames."""
        request = {"request_id": "rank-landmarks", "scene": {"frame": frame, "source": "predicted", "tracks": tracks},
                   "text": text, "input_mode": "typed", "dialogue": []}
        validated = validate_request(request)
        if image.size != (frame["width"], frame["height"]):
            raise ValueError("Original image size does not match the current frame")
        current = [t for t in validated["scene"]["tracks"] if t["frame_id"] == frame["frame_id"]]
        current.sort(key=lambda t: (t["bbox_xyxy"], t["track_id"]))
        boxes = [t["bbox_xyxy"] for t in current]
        classes = [category(t["class_name"]) for t in current]
        regions, hist, middle = region_features(image, boxes, classes)
        probabilities = self.scores(text, regions, alignment(text, classes, regions, hist, middle))
        order = ordered_alternatives(boxes, classes, probabilities[:-1])
        alternatives = [{"track_id": current[i]["track_id"], "class_name": current[i]["class_name"],
                         "category": classes[i], "bbox_xyxy": boxes[i], "score": float(probabilities[i]),
                         "colour_fractions": {c: float(v) for c, v in zip(COLORS, hist[i])},
                         "central_colour_fractions": {c: float(v) for c, v in zip(COLORS, middle[i])}}
                        for i in order]
        return {"model": KIND, "checkpoint_sha256": self.sha256, "candidate_count": len(current),
                "distinct_candidate_count": len(order), "null_score": float(probabilities[-1]),
                "alternatives": alternatives}
