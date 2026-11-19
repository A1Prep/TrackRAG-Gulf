"""Task 1 vision tests that run without torch, rfdetr, ultralytics or a GPU."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from trackrag.vision.classes import EMT_CLASSES, GROUP_OF, RunningClassVote, require_emt_class_order
from trackrag.vision.detections import Detection, coco_to_emt, keep_emt, nms
from trackrag.vision.geometry import ResizeTransform, clip_box, iou, resize_image

PROJECT = Path(__file__).resolve().parents[1]


# --- geometry -------------------------------------------------------------------------

@pytest.mark.parametrize("size, resized", [
    ((2560, 1440), (1280, 720)),
    ((3840, 2160), (1280, 720)),
    ((896, 672), (896, 672)),
    ((1281, 721), (1280, 720)),
    ((1000, 3000), (427, 1280)),
])
def test_resize_transform_sizes_and_exact_inverse(size, resized):
    transform = ResizeTransform.for_size(*size)
    assert (transform.resized_width, transform.resized_height) == resized
    box = (10.5, 20.25, size[0] - 3.0, size[1] - 7.75)
    back = transform.to_original(transform.to_resized(box))
    assert all(abs(a - b) < 1e-9 for a, b in zip(back, box))
    assert all(type(value) is float for value in back)


def test_resize_transform_keeps_x_and_y_scales_separate():
    transform = ResizeTransform(width=2560, height=1440, resized_width=1280, resized_height=360)
    assert transform.to_original((10, 10, 20, 20)) == (20.0, 40.0, 40.0, 80.0)
    assert transform.to_resized((20, 40, 40, 80)) == (10.0, 10.0, 20.0, 20.0)


def test_resize_transform_rejects_bad_sizes():
    for bad in ((0, 10), (10, -1), (10.0, 10)):
        with pytest.raises(ValueError):
            ResizeTransform.for_size(*bad)


def test_issue_box_check_non_square_image_round_trip():
    """Issue #1 box check: a known box on a non-square image survives resize + reverse."""
    import numpy as np
    from PIL import Image

    width, height = 2560, 1440
    known = (803.0, 411.0, 1517.0, 979.0)
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[int(known[1]):int(known[3]), int(known[0]):int(known[2])] = (255, 0, 0)
    transform = ResizeTransform.for_size(width, height)
    small = np.asarray(resize_image(Image.fromarray(pixels), transform))
    # A stand-in detector: the red region's bounding box on the resized image.
    rows, columns = np.nonzero(small[:, :, 0] > 127)
    found_small = (float(columns.min()), float(rows.min()), float(columns.max() + 1), float(rows.max() + 1))
    exported = transform.to_original(found_small)
    tolerance = transform.tolerance_px  # one resized pixel in original pixels (2.0 here)
    assert tolerance == 2.0
    assert all(abs(a - b) <= tolerance for a, b in zip(exported, known)), (exported, known)


def test_resize_image_requires_matching_size_and_returns_rgb():
    from PIL import Image

    transform = ResizeTransform.for_size(64, 36, long_side=32)
    image = Image.new("L", (64, 36))
    resized = resize_image(image, transform)
    assert resized.size == (32, 18) and resized.mode == "RGB"
    with pytest.raises(ValueError):
        resize_image(Image.new("RGB", (63, 36)), transform)


def test_clip_box_and_iou():
    assert clip_box((-1.0, 5, 20, 10), 15, 8) == (0.0, 5.0, 15.0, 8.0)
    assert clip_box((16, 0, 20, 5), 15, 8) is None
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-12


# --- class vocabulary -----------------------------------------------------------------

def test_emt_class_order_and_groups():
    assert len(EMT_CLASSES) == 9 and len(set(EMT_CLASSES)) == 9
    require_emt_class_order(list(EMT_CLASSES))
    with pytest.raises(ValueError):
        require_emt_class_order(list(EMT_CLASSES[::-1]))
    assert set(GROUP_OF) == set(EMT_CLASSES)
    assert {GROUP_OF[name] for name in EMT_CLASSES} == {"Pedestrian", "Cyclist", "Motorbike", "Vehicle"}


def test_running_class_vote_is_causal_and_deterministic():
    vote = RunningClassVote()
    sequence = ["Car", "Medium_vehicle", "Car", "Medium_vehicle", "Medium_vehicle"]
    outputs = [vote.update(7, name) for name in sequence]
    assert outputs == ["Car", "Medium_vehicle", "Car", "Medium_vehicle", "Medium_vehicle"]
    prefix = RunningClassVote()
    assert [prefix.update(7, name) for name in sequence[:3]] == outputs[:3]
    other = RunningClassVote()
    assert other.update(1, "Bus") == "Bus" and other.update(2, "Car") == "Car"


# --- detection post-processing --------------------------------------------------------

def det(box, score, name):
    return Detection(box, score, name)


def test_detection_normalises_types_and_validates():
    detection = Detection((1, 2, 3, 4), 0.5, "car")
    assert all(type(value) is float for value in detection.box) and type(detection.score) is float
    assert detection.native == "car"
    with pytest.raises(ValueError):
        Detection((0, 0, 1, float("nan")), 0.5, "car")
    with pytest.raises(ValueError):
        Detection((0, 0, 1, 1), 1.5, "car")
    with pytest.raises(ValueError):
        Detection((10, 10, 0, 0), 0.5, "car")  # inverted box


def test_nms_removes_same_box_class_duplicates_but_keeps_neighbours():
    duplicate = [det((100, 100, 200, 160), 0.9, "Car"), det((100, 100, 200, 161), 0.4, "Medium_vehicle")]
    kept = nms(duplicate)
    assert [(d.name, d.score) for d in kept] == [("Car", 0.9)]
    neighbours = [det((100, 100, 200, 160), 0.9, "Car"), det((190, 100, 290, 160), 0.8, "Car")]
    assert len(nms(neighbours)) == 2


def test_coco_to_emt_rider_merge_rules():
    person_on_bike = det((105, 80, 135, 150), 0.8, "person")
    motorcycle = det((100, 110, 140, 160), 0.6, "motorcycle")
    walker = det((300, 80, 320, 150), 0.7, "person")
    lone_bicycle = det((500, 120, 540, 160), 0.5, "bicycle")
    lone_motorcycle = det((600, 110, 640, 160), 0.4, "motorcycle")
    truck = det((700, 50, 900, 200), 0.9, "truck")
    traffic_light = det((10, 10, 20, 30), 0.9, "traffic light")
    empty_name = det((10, 10, 20, 30), 0.9, "")
    mapped, dropped = coco_to_emt([person_on_bike, motorcycle, walker, lone_bicycle, lone_motorcycle,
                                   truck, traffic_light, empty_name])
    by_name = {}
    for detection in mapped:
        by_name.setdefault(detection.name, []).append(detection)
    assert sorted(by_name) == ["Large_vehicle", "Motorbike", "Pedestrian"]
    merged = [d for d in by_name["Motorbike"] if "+" in d.native]
    assert len(merged) == 1 and merged[0].box == (100.0, 80.0, 140.0, 160.0)
    assert abs(merged[0].score - 0.7) < 1e-12
    assert len(by_name["Motorbike"]) == 2  # merged + unpaired motorcycle
    assert [d.box for d in by_name["Pedestrian"]] == [walker.box]
    assert dropped == {"traffic light": 1, "<empty>": 1, "bicycle (unpaired)": 1}
    assert [d.score for d in mapped] == sorted((d.score for d in mapped), reverse=True)


def test_coco_to_emt_person_is_merged_at_most_once():
    rider = det((100, 80, 130, 150), 0.9, "person")
    bicycle = det((95, 110, 135, 160), 0.8, "bicycle")
    motorcycle = det((96, 112, 136, 160), 0.7, "motorcycle")
    mapped, _ = coco_to_emt([rider, bicycle, motorcycle])
    names = sorted(d.name for d in mapped)
    assert names == ["Cyclist", "Motorbike"]
    assert next(d for d in mapped if d.name == "Motorbike").box == motorcycle.box


def test_low_score_duplicate_bicycle_cannot_steal_a_motorcycle_rider():
    """A low-score duplicate bicycle must not take the rider of a higher-score motorcycle."""
    rider = det((100, 60, 140, 150), 0.90, "person")
    motorcycle = det((95, 100, 145, 160), 0.85, "motorcycle")
    duplicate = det((95, 100, 145, 160), 0.07, "bicycle")
    mapped, dropped = coco_to_emt([rider, motorcycle, duplicate])
    assert [(d.name, d.box) for d in mapped] == [("Motorbike", (95.0, 60.0, 145.0, 160.0))]
    assert abs(mapped[0].score - 0.875) < 1e-12
    assert dropped == {"bicycle (unpaired)": 1}
    # With NMS first (as the pipeline does), the duplicate never reaches the merge at all.
    mapped, dropped = coco_to_emt(nms([rider, motorcycle, duplicate]))
    assert [d.name for d in mapped] == ["Motorbike"] and not dropped


def test_rider_merge_thresholds_and_best_iou_choice():
    bike = det((0, 0, 10, 10), 0.8, "motorcycle")
    exactly_point_one = det((0, 0, 10, 1), 0.9, "person")      # IoU == 0.1: not merged (strict >)
    mapped, _ = coco_to_emt([bike, exactly_point_one])
    assert sorted(d.name for d in mapped) == ["Motorbike", "Pedestrian"]
    weak = det((0, 0, 10, 3), 0.9, "person")                    # IoU 0.3
    strong = det((0, 0, 10, 6), 0.5, "person")                  # IoU 0.6: chosen despite lower score
    mapped, _ = coco_to_emt([bike, weak, strong])
    merged = next(d for d in mapped if d.name == "Motorbike")
    assert merged.native == "motorcycle+person" and abs(merged.score - 0.65) < 1e-12
    assert [d.box for d in mapped if d.name == "Pedestrian"] == [weak.box]


def test_nms_threshold_is_inclusive():
    a = det((0, 0, 10, 10), 0.9, "Car")
    b = det((0, 0, 10, 8), 0.5, "Car")  # IoU exactly 0.8 -> suppressed
    assert nms([a, b]) == [a]
    assert len(nms([a, det((0, 0, 10, 7.9), 0.5, "Car")])) == 2


def test_running_vote_tie_break_is_alphabetical_when_current_class_is_not_top():
    vote = RunningClassVote()
    for name in ["Car", "Bus", "Car", "Bus"]:
        vote.update(1, name)
    assert vote.update(1, "Medium_vehicle") == "Bus"


def test_drop_ego_vehicle_removes_only_the_full_width_bonnet():
    from trackrag.vision.detections import drop_ego_vehicle

    bonnet = det((0, 1298, 2560, 1438), 0.6, "Car")           # full width, touching the bottom
    close_bus = det((800, 900, 1840, 1438), 0.9, "Bus")       # 40.6% wide (widest real EMT box)
    far_car = det((1000, 600, 1100, 650), 0.8, "Car")
    kept, dropped = drop_ego_vehicle([bonnet, close_bus, far_car], 2560, 1440)
    assert dropped == 1 and kept == [close_bus, far_car]


def test_keep_emt_drops_background_and_unknown_names():
    kept, dropped = keep_emt([det((0, 0, 5, 5), 0.5, "Car"), det((0, 0, 5, 5), 0.2, "__background__"),
                              det((0, 0, 5, 5), 0.2, "")])
    assert [d.name for d in kept] == ["Car"]
    assert dropped == {"__background__": 1, "<empty>": 1}


# --- import hygiene -------------------------------------------------------------------

def test_importing_vision_modules_loads_no_model_libraries():
    blocker = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'torch','rfdetr','ultralytics','supervision','cv2'}:\n"
        "            raise ImportError('blocked model library: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import trackrag.vision, trackrag.vision.classes, trackrag.vision.geometry, trackrag.vision.detections\n"
        "print('ok')\n"
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("TRACKRAG_")}
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run([sys.executable, "-c", blocker], cwd=PROJECT, env=env,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
