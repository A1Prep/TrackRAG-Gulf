"""EMT class vocabulary, evaluation groups and the pretrained COCO-to-EMT name table.

The order of ``EMT_CLASSES`` is fixed: it defines the COCO category ids (1..9) written for
fine-tuning, the 0-based labels RF-DETR trains on, and the class integers given to the
tracker. A fine-tuned checkpoint must report exactly these names in exactly this order.
"""

from __future__ import annotations

from collections import Counter

EMT_CLASSES: tuple[str, ...] = (
    "Pedestrian",
    "Cyclist",
    "Motorbike",
    "Car",
    "Small_motorised_vehicle",
    "Medium_vehicle",
    "Large_vehicle",
    "Bus",
    "Emergency_vehicle",
)
EMT_INDEX: dict[str, int] = {name: index for index, name in enumerate(EMT_CLASSES)}

# The four groups used by the EMT authors' detection evaluation (arXiv 2502.19260).
GROUPS: tuple[str, ...] = ("Pedestrian", "Cyclist", "Motorbike", "Vehicle")
GROUP_OF: dict[str, str] = {
    "Pedestrian": "Pedestrian",
    "Cyclist": "Cyclist",
    "Motorbike": "Motorbike",
    **{name: "Vehicle" for name in EMT_CLASSES[3:]},
}

# COCO vehicle names with a direct EMT target. Persons, bicycles and motorcycles are handled
# by the rider merge in ``detections.coco_to_emt``. truck/train -> Large_vehicle is an
# approximation (see the coverage table in docs/task1-vision.md).
COCO_VEHICLE_TO_EMT: dict[str, str] = {
    "car": "Car",
    "bus": "Bus",
    "truck": "Large_vehicle",
    "train": "Large_vehicle",
}
COCO_RIDER_PARTS = ("person", "bicycle", "motorcycle")

# Which EMT classes the COCO-pretrained checkpoint can express (see docs/task1-vision.md).
PRETRAINED_COVERAGE: dict[str, tuple[str, str]] = {
    "Pedestrian": ("person (not merged with a bike)", "yes"),
    "Cyclist": ("bicycle + overlapping person, union box", "approximate"),
    "Motorbike": ("motorcycle (+ overlapping person), union box", "approximate"),
    "Car": ("car", "yes"),
    "Small_motorised_vehicle": ("none", "no"),
    "Medium_vehicle": ("none (vans split between COCO car and truck)", "no"),
    "Large_vehicle": ("truck, train", "approximate"),
    "Bus": ("bus", "yes"),
    "Emergency_vehicle": ("none (defined by function, not shape)", "no"),
}

# Ground-truth rows whose class is not an EMT road-user class (ego vehicle, traffic lights).
NON_EMT_LABELS = ("AV", "Vehicle_traffic_light", "Other_traffic_light")


def require_emt_class_order(names) -> None:
    """Raise unless ``names`` lists exactly the EMT classes in the fixed order."""
    observed = tuple(str(name) for name in names)
    if observed != EMT_CLASSES:
        raise ValueError(f"Expected EMT classes in order {list(EMT_CLASSES)}, got {list(observed)}")


class RunningClassVote:
    """Causal per-track class: the most frequent class among a track's matches so far.

    Only past and current observations are used, so the label exported for a frame never
    depends on later frames. Ties go to the current detection's class, then to the
    alphabetically first name, which keeps the output deterministic.
    """

    def __init__(self) -> None:
        self._counts: dict[int, Counter[str]] = {}

    def update(self, track_id: int, class_name: str) -> str:
        counts = self._counts.setdefault(track_id, Counter())
        counts[class_name] += 1
        top = max(counts.values())
        if counts[class_name] == top:
            return class_name
        return min(name for name, count in counts.items() if count == top)
