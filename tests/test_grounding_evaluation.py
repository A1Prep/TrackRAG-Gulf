"""Grounding evaluator and input-converter tests. CI-safe: scenes come from the real scene adapter run on small
synthetic non-square frames with a scripted stand-in detector (``detector_kind: "stub"``, hence
``allow_stub_scenes``), references use synthetic masks, and model outputs are written by hand. The converter tests
also run Task 5's own request builder, prediction command (with a stand-in model) and case loader; they are
skipped where that code is not present."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from evaluation import grounding as g
from evaluation import grounding_inputs as gi
from evaluation.grounding import GroundingError
from trackrag.dropoff.scenes import export_scenes
from trackrag.emt import open_recording
from trackrag.vision import predict_tracks
from trackrag.vision.__main__ import JsonlSink
from trackrag.vision.config import config_from_dict
from trackrag.vision.detections import Detection
from trackrag.vision.geometry import iou

PROJECT = Path(__file__).resolve().parents[1]
W, H = 80, 48
REC = "video_000001"
CAR, BUS = [4.0, 20.0, 24.0, 36.0], [50.0, 6.0, 76.0, 40.0]
DUP, DUP2 = [30.0, 30.0, 44.0, 44.0], [32.0, 30.0, 46.0, 44.0]  # IoU 0.75: both survive NMS at 0.8
DETECTOR = {"label_space": "emt", "checkpoint": "model.pth", "checkpoint_sha256": "0" * 64, "inference_long_side": 32}
REVIEWED = {"independent_reviews": 2, "disagreements_resolved": True}
PROTOCOL = g.load_protocol(None)[0]
INSTRUCTIONS = {"C1": "Stop in the bay left of the bus", "C2": "Drop me at the corner",
                "A1": "Stop by the gap", "M1": "Stop behind the red truck"}


def sha(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


class Detector:
    kind = "stub"

    def detect(self, image):
        sx, sy = W / image.size[0], H / image.size[1]
        return [Detection((l / sx, t / sy, r / sx, b / sy), 0.9, name)
                for (l, t, r, b), name in ((CAR, "Car"), (BUS, "Bus"), (DUP, "Car"), (DUP2, "Car"))]


class Tracker:
    def __init__(self, settings):
        self.tracks, self.next_id = {}, 1

    def update(self, detections, image_bgr):
        pairs, used = [], set()
        for index, detection in enumerate(detections):
            best = max(((iou(detection.box, box), track) for track, box in self.tracks.items() if track not in used),
                       default=(0.0, None))
            track = best[1] if best[0] >= 0.3 else None
            if track is None:
                track, self.next_id = self.next_id, self.next_id + 1
            used.add(track)
            self.tracks[track] = detection.box
            pairs.append((track, index))
        return pairs


def save_mask(path, array, mode="L"):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = array.shape
    if mode == "P":
        image = Image.new("P", (width, height), 0)
        image.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
        image.putdata([int(v) for v in array.astype(np.uint8).flatten()])
    elif mode == "1":
        image = Image.fromarray(array.astype(bool))
    else:
        image = Image.fromarray((array.astype(np.uint8) * 255), mode="L").convert(mode)
    image.save(path)
    return path


def c_mask():
    mask = np.zeros((H, W), bool)
    mask[10:40, 10:14] = True
    mask[10:14, 10:30] = True
    mask[36:40, 10:30] = True
    return mask


def two_region_mask():
    mask = np.zeros((H, W), bool)
    mask[40:48, 0:8] = True
    mask[40:48, 72:80] = True
    return mask


def intended_mask():
    mask = np.zeros((H, W), bool)
    mask[40:48, 50:70] = True
    return mask


MASKS = {"C1": (c_mask, "1"), "C2": (two_region_mask, "L"), "A1": (intended_mask, "P")}


def reference_rows(scenes):
    """C1, C2 clear (group g1), A1 ambiguous (g2), M1 missing reference (g3)."""
    def base(case, group, frame, category, expected):
        scene = scenes[frame]["model_input"]
        return {"case_id": case, "group_id": group, "partition": "final_evaluation",
                "scene_sha256": scenes[frame]["scene_sha256"], "recording_id": REC, "frame_id": frame,
                "image_sha256": scene["image_sha256"], "width": W, "height": H, "category": category,
                "expected_status": expected, "instruction": INSTRUCTIONS[case], "dialogue": [],
                "mask": None, "mask_sha256": None, "target_box": None, "box_rule": None,
                "reference_point": None, "anchors": [], "anchor_roles_ordered": False, "intended_mask": None,
                "intended_mask_sha256": None, "review": dict(REVIEWED)}

    return [
        dict(base("C1", "g1", 3, "clear", "proposed"), mask="masks/C1.png", target_box=[10, 10, 30, 40],
             box_rule="pixel_bbox", reference_point=[11.5, 25.0],
             anchors=[{"role": "left_of", "bbox_xyxy": CAR}, {"role": "right_of", "bbox_xyxy": BUS}]),
        dict(base("C2", "g1", 2, "clear", "proposed"), mask="masks/C2.png", target_box=[0, 40, 80, 48],
             box_rule="pixel_bbox", reference_point=[4, 44], anchor_roles_ordered=True,
             anchors=[{"role": "first", "bbox_xyxy": CAR}, {"role": "second", "bbox_xyxy": BUS}]),
        dict(base("A1", "g2", 3, "ambiguous", "needs_clarification"), intended_mask="masks/A1.png",
             anchors=[{"role": "near", "bbox_xyxy": [60.0, 0.0, 79.0, 5.0]}]),
        base("M1", "g3", 2, "missing_reference", "insufficient_visual_evidence"),
    ]


def write_jsonl(path, rows, encoding="utf-8", newline="\n", ensure_ascii=True):
    path.write_bytes("".join(json.dumps(r, ensure_ascii=ensure_ascii) + newline for r in rows).encode(encoding))
    return path


def write_references(folder, rows, masks=None, encoding="utf-8", newline="\n"):
    """Writes the masks and the references, pinning each mask by the sha256 of its file."""
    folder.mkdir(parents=True, exist_ok=True)
    masks = masks or {case: (make(), mode) for case, (make, mode) in MASKS.items()}
    rows = json.loads(json.dumps(rows))
    for case, (array, mode) in masks.items():
        save_mask(folder / "masks" / f"{case}.png", array, mode)
    for row in rows:
        for field in ("mask", "intended_mask"):
            if row.get(field):
                row[f"{field}_sha256"] = sha((folder / row[field]).read_bytes())
    return write_jsonl(folder / "references.jsonl", rows, encoding, newline)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    folder = tmp_path_factory.mktemp("grounding")
    root = folder / "emt"
    frames = root / "images" / "test" / "images" / REC
    frames.mkdir(parents=True)
    for frame in (1, 2, 3):
        Image.new("RGB", (W, H), (frame, 0, 0)).save(frames / f"{frame:05d}.png")
    config = config_from_dict({"name": "grounding-test", "detector": DETECTOR, "tracker": {}})
    predictions = folder / "predictions"
    predictions.mkdir()
    predict_tracks(open_recording(root, REC), config=config, detector=Detector(), tracker_factory=Tracker,
                   sink=JsonlSink(predictions, REC))
    built = export_scenes(root, [{"recording_id": REC, "frame_id": 2}, {"recording_id": REC, "frame_id": 3}],
                          folder / "scenes", profile=config, predictions_dir=predictions, allow_stub=True)
    scenes = {s["model_input"]["frame_id"]: s for s in built}
    rows = reference_rows(scenes)
    references_path = write_references(folder / "refs", rows)
    loaded_scenes = g.load_scenes(folder / "scenes", allow_stub=True)
    references, _ = g.load_references(references_path, loaded_scenes)
    return SimpleNamespace(folder=folder, root=root, predictions=predictions, scenes_dir=folder / "scenes",
                           scenes=loaded_scenes, rows=rows, references_path=references_path, references=references)


def tracks(world, case, box):
    """Track IDs whose box matches ``box`` in the case's scene (in ID order)."""
    scene = world.scenes[world.references[case]["scene_sha256"]]
    return sorted(o["track_id"] for o in scene["model_input"]["observations"] if iou(o["bbox_xyxy"], box) > 0.99)


def proposal(point, box, landmarks=()):
    return {"status": "proposed", "point": point, "box": box, "landmarks": list(landmarks), "clarification": None}


def status(name, **extra):
    return dict({"status": name, "point": None, "box": None, "landmarks": [], "clarification": "which one?"}, **extra)


def output(world, case, result=None, *, model="pretrained", mode="direct", transcript="typed", turns=None, text=None,
           **changes):
    provenance = {"base_revision": "base-rev", "adapter_sha256": "a" * 64 if model == "trained" else None,
                  "settings_sha256": "s" * 64, "image_budget": 1024, "context_policy": "scene_frame",
                  "coordinate_convention": "original_pixels", "turn_budget": 3 if mode == "clarification" else None,
                  "substitute": False}
    provenance.update(changes.pop("provenance", {}))
    record = {"case_id": case, "condition": {"model": model, "mode": mode, "transcript": transcript},
              "scene_sha256": world.references[case]["scene_sha256"],
              "instruction_sha256": g.instruction_sha256(INSTRUCTIONS[case] if text is None else text),
              "call": {"outcome": "ok", "error_code": None}, "result": result if mode == "direct" else None,
              "raw_response": None, "provenance": provenance}
    if mode == "clarification":
        record["turns"] = turns
    record.update(changes)
    return record


def score(world, case, record, mode="direct", replies=None, protocol=PROTOCOL):
    reference = world.references[case]
    return g.score_output(record, reference, world.scenes[reference["scene_sha256"]], replies or {}, protocol, mode,
                          reference["instruction_sha256"])


def run(world, tmp_path, records, **kwargs):
    outputs = write_jsonl(tmp_path / "outputs.jsonl", records)
    kwargs.setdefault("allow_stub_scenes", True)
    return g.evaluate(kwargs.pop("references", world.references_path), world.scenes_dir, outputs, **kwargs)


GOOD = {"C1": proposal([11.5, 25.0], [10, 10, 30, 40]), "C2": proposal([4.0, 44.0], [0, 40, 8, 48]),
        "A1": status("needs_clarification"), "M1": status("insufficient_visual_evidence", clarification=None)}
INSIDE_A1, OUTSIDE_A1 = proposal([60.0, 44.0], [50, 40, 70, 48]), proposal([10.0, 44.0], [0, 40, 20, 48])


# --- destination scoring -----------------------------------------------------------------------------------------

def test_correct_point_inside_and_wrong_point_outside(world):
    good = score(world, "C1", output(world, "C1", GOOD["C1"]))
    assert good["outcome"] == "proposed" and good["initial_point_in_mask"] and good["initial_distance_px"] == 0.0
    assert good["initial_box_iou"] == 1.0 and good["initial_box_success"] and good["correct_behaviour"]
    wrong = score(world, "C1", output(world, "C1", proposal([20.0, 25.0], [15, 15, 25, 35])))
    assert wrong["outcome"] == "proposed" and not wrong["initial_point_in_mask"]
    assert wrong["initial_incorrect_proposal"] and wrong["initial_distance_px"] == 7.0
    assert 0 < wrong["initial_box_iou"] < 0.5 and not wrong["initial_box_success"] and not wrong["correct_behaviour"]


def test_interior_point_counts_and_the_centroid_of_a_c_shaped_mask_does_not(world, tmp_path):
    mask = c_mask()
    ys, xs = np.nonzero(mask)
    centroid = [float(xs.mean()), float(ys.mean())]
    assert not mask[int(centroid[1]), int(centroid[0])]
    at_centroid = score(world, "C1", output(world, "C1", proposal(centroid, [10, 10, 30, 40])))
    assert not at_centroid["initial_point_in_mask"] and at_centroid["initial_box_iou"] == 1.0
    rows = json.loads(json.dumps(world.rows))
    rows[0]["reference_point"] = centroid
    with pytest.raises(GroundingError, match="reference_point must lie inside the mask"):
        g.load_references(write_references(tmp_path / "refs", rows), world.scenes)


def test_two_region_mask_either_region_counts(world):
    for point in ([4.0, 44.0], [76.0, 44.0]):
        assert score(world, "C2", output(world, "C2", proposal(point, [0, 40, 80, 48])))["initial_point_in_mask"]
    between = score(world, "C2", output(world, "C2", proposal([40.0, 44.0], [0, 40, 80, 48])))
    assert not between["initial_point_in_mask"] and between["initial_box_iou"] == 1.0


def test_points_follow_task5_bounds_x_below_w(world):
    inside = score(world, "C2", output(world, "C2", proposal([79.999, 47.999], [72, 40, 80, 48])))
    assert inside["outcome"] == "proposed" and inside["initial_point_in_mask"]
    for point in ([80.0, 44.0], [40.0, 48.0]):
        edge = score(world, "C2", output(world, "C2", proposal(point, [0, 40, 80, 48])))
        assert edge["outcome"] == "invalid_output" and edge["reasons"] == ["proposed without a valid point inside the image"]


def test_raw_response_conversion_check_in_both_formats(world):
    record = output(world, "C2", proposal([79.92, 44.0], [72, 40, 80, 48]),
                    raw_response='{"point_2d": [999, 917], "bbox_2d": [900, 833, 1000, 1000]}')
    assert score(world, "C2", record)["conversion_mismatch"] is False
    record["raw_response"] = json.dumps({"status": "proposed", "destination_point_xy": [999, 917],
                                         "destination_box_xyxy": [900, 833.34, 1000, 1000],
                                         "clarification_question": None})
    assert score(world, "C2", record)["conversion_mismatch"] is False
    record["raw_response"] = '```json\n{"destination_point_xy": [500, 917], "destination_box_xyxy": [900, 833, 1000, 1000]}\n```'
    assert score(world, "C2", record)["conversion_mismatch"] is True
    for raw in ("not json", "[" * 200000, None, "{}"):
        record["raw_response"] = raw
        assert score(world, "C2", record)["conversion_mismatch"] is None


@pytest.mark.parametrize("box", [[30, 10, 10, 40], [10, 10, 10, 40], [10, 10, float("nan"), 40], [10, 10, 30, 49],
                                 [-1, 10, 30, 40], [10, 10, 30], None, [True, 10, 30, 40], [10, 10, 10 ** 400, 40]])
def test_invalid_boxes(world, box):
    case = score(world, "C1", output(world, "C1", proposal([11.5, 25.0], box)))
    assert case["outcome"] == "invalid_output" and case["reasons"] == ["proposed without a valid box inside the image"]
    assert not case["initial_proposed"] and not case["correct_behaviour"] and case["initial_box_iou"] == 0.0


@pytest.mark.parametrize("point", [[True, 25.0], ["11", "25"], [11.5, 25.0, 1.0], None, [float("inf"), 1.0], [-0.5, 3],
                                   [10 ** 400, 5]])
def test_malformed_points(world, point):
    case = score(world, "C1", output(world, "C1", proposal(point, [10, 10, 30, 40])))
    assert case["outcome"] == "invalid_output" and case["reasons"] == ["proposed without a valid point inside the image"]


def test_empty_proposal_unknown_status_and_point_outside_its_own_box(world):
    for result, reason in ((proposal(None, None), "proposed without a valid point inside the image"),
                           ({}, "unknown status None"), (None, "no parsed result"),
                           ({"status": "maybe"}, "unknown status 'maybe'"),
                           (proposal([11.5, 25.0], [20, 10, 30, 40]), "point outside its own box"),
                           (proposal([11.5, 25.0], [10, 10, 30, 40], [{"role": "x", "track_id": "1"}]),
                            "malformed landmarks"),
                           (status("insufficient_visual_evidence", landmarks=None), "malformed landmarks"),
                           (dict(GOOD["C1"], landmark_boxes=[[1, 1, 1, 1]]), "malformed landmark boxes")):
        case = score(world, "C1", output(world, "C1", result))
        assert case["outcome"] == "invalid_output" and case["reasons"] == [reason]
    unparsed = output(world, "C1", None, invalid_reason="model output could not be parsed (invalid_model_output)")
    assert score(world, "C1", unparsed)["reasons"] == ["model output could not be parsed (invalid_model_output)"]


def test_location_attached_to_a_non_proposal_is_counted_not_scored(world):
    case = score(world, "C1", output(world, "C1", status("needs_clarification", point=[11.5, 25.0])))
    assert case["outcome"] == "needs_clarification" and case["location_on_non_proposal"]
    assert not case["initial_proposed"] and not case["initial_point_in_mask"]


# --- outcome precedence ------------------------------------------------------------------------------------------

def test_request_error_with_a_parsed_result_is_still_an_error(world):
    record = output(world, "C1", GOOD["C1"], call={"outcome": "error", "error_code": "timeout"})
    case = score(world, "C1", record)
    assert case["outcome"] == "request_error" and case["error_code"] == "timeout"
    assert not case["initial_proposed"] and not case["initial_point_in_mask"] and not case["correct_behaviour"]


def test_wrong_scene_then_wrong_instruction_then_malformed_call(world):
    other_scene = world.references["C2"]["scene_sha256"]
    assert score(world, "C1", output(world, "C1", GOOD["C1"], scene_sha256=other_scene))["outcome"] == "wrong_scene"
    both = output(world, "C1", GOOD["C1"], scene_sha256=other_scene, call=None, text="other words")
    assert score(world, "C1", both)["outcome"] == "wrong_scene"
    other_text = output(world, "C1", GOOD["C1"], text="Stop anywhere", call=None)
    assert score(world, "C1", other_text)["outcome"] == "wrong_instruction"
    no_call = score(world, "C1", output(world, "C1", GOOD["C1"], call=None))
    assert no_call["outcome"] == "invalid_output" and no_call["reasons"] == ["missing or unknown call outcome"]
    with_turns = dict(output(world, "C1", GOOD["C1"]), turns=[{"turn": 0}])
    assert score(world, "C1", with_turns)["reasons"] == ["direct mode output carries turns"]


def test_missing_output_and_duplicate_unknown_or_unreadable_records(world, tmp_path):
    summary, rows = run(world, tmp_path, [output(world, c, GOOD[c]) for c in ("C1", "C2", "A1")])
    condition = summary["conditions"]["pretrained/direct/typed"]
    assert condition["initial"]["outcomes"] == {"proposed": 2, "needs_clarification": 1, "missing": 1}
    assert condition["correct_behaviour"]["count"] == 3 and condition["correct_behaviour"]["n"] == 4
    assert [r["outcome"] for r in rows if r["case_id"] == "M1"] == ["missing"]
    with pytest.raises(GroundingError, match="Two outputs for case C1"):
        run(world, tmp_path, [output(world, "C1", GOOD["C1"])] * 2)
    with pytest.raises(GroundingError, match="unknown case 'X9'"):
        run(world, tmp_path, [dict(output(world, "C1", GOOD["C1"]), case_id="X9")])
    (tmp_path / "deep.jsonl").write_text("[" * 100000 + "\n", encoding="utf-8")
    with pytest.raises(GroundingError, match="not JSON"):
        g.evaluate(world.references_path, world.scenes_dir, tmp_path / "deep.jsonl", allow_stub_scenes=True)


def test_outputs_can_come_from_several_files_and_lines_split_only_at_line_feeds(world, tmp_path):
    first = write_jsonl(tmp_path / "a.jsonl", [output(world, "C1", GOOD["C1"], raw_response="odd text")],
                        ensure_ascii=False)
    assert " " in first.read_text(encoding="utf-8")
    second = write_jsonl(tmp_path / "b.jsonl", [output(world, c, GOOD[c]) for c in ("C2", "A1", "M1")],
                         newline="\r\n")
    summary, _ = g.evaluate(world.references_path, world.scenes_dir, [first, second], allow_stub_scenes=True)
    assert summary["conditions"]["pretrained/direct/typed"]["correct_behaviour"]["count"] == 4
    assert [o["file"] for o in summary["inputs_sha256"]["outputs"]] == ["a.jsonl", "b.jsonl"]


def test_abstaining_on_everything_scores_zero_coverage_and_zero_clear_success(world, tmp_path):
    abstain = status("insufficient_visual_evidence", clarification=None)
    summary, _ = run(world, tmp_path, [output(world, c, abstain) for c in ("C1", "C2", "A1", "M1")])
    initial = summary["conditions"]["pretrained/direct/typed"]["initial"]
    assert initial["proposal_coverage"]["count"] == 0 and initial["proposal_coverage"]["n"] == 4
    assert initial["clear"]["point_in_mask"]["count"] == 0 and initial["clear"]["point_in_mask"]["n"] == 2
    assert initial["clear"]["box_iou_mean_non_proposals_zero"]["mean"] == 0.0
    assert initial["clear"]["box_iou_mean_among_proposals"]["mean"] is None
    assert initial["correct"]["count"] == 1  # only the missing-reference case
    assert initial["clarification_precision"] is None
    assert initial["clarification_recall_on_ambiguous"]["count"] == 0
    assert initial["clarification_recall_on_ambiguous"]["n"] == 1


def test_clarification_precision_counts_asks_on_missing_references_as_acceptable(world, tmp_path):
    ask = status("needs_clarification")
    summary, _ = run(world, tmp_path, [output(world, c, ask) for c in GOOD])
    initial = summary["conditions"]["pretrained/direct/typed"]["initial"]
    assert initial["clarification_precision"]["count"] == 2 and initial["clarification_precision"]["n"] == 4
    assert initial["clarification_recall_on_ambiguous"]["rate"] == 1.0 and initial["correct"]["count"] == 2


def test_refused_outputs_are_listed_and_scored_as_missing(world, tmp_path):
    records = [output(world, "C1", GOOD["C1"], provenance={"substitute": True}),
               output(world, "C2", GOOD["C2"], provenance={"substitute": None}),
               output(world, "A1", GOOD["A1"], model="trained", provenance={"adapter_sha256": None}),
               output(world, "M1", GOOD["M1"], provenance={"adapter_sha256": "b" * 64}),
               output(world, "C1", GOOD["C1"], model="trained", provenance={"coordinate_convention": "qwen_0_1000"}),
               output(world, "C2", GOOD["C2"], model="trained", provenance={"image_budget": None})]
    summary, _ = run(world, tmp_path, records)
    assert [r["reason"] for r in summary["refused_outputs"]] == [
        "substitute or stub output", "substitute or stub output",
        "model identity missing (base_revision, settings_sha256, adapter_sha256)",
        "pretrained output names an adapter", "coordinates are not original pixels",
        "setup missing (image_budget, context_policy)"]
    for key in ("pretrained/direct/typed", "trained/direct/typed"):
        assert summary["conditions"][key]["initial"]["outcomes"] == {"missing": 4}


def test_a_required_condition_with_only_refused_outputs_counts_as_missing(world, tmp_path):
    records = [output(world, c, GOOD[c], provenance={"substitute": True}) for c in GOOD] + [
        output(world, c, GOOD[c], model="trained") for c in GOOD]
    summary, _ = run(world, tmp_path, records)
    assert "pretrained/direct/typed" in summary["required_conditions_missing"]
    assert "outputs refused in required conditions" in summary["status"]
    assert "pretrained/direct/typed: 4 output(s) refused and scored as missing" in summary["warnings"]


def test_one_condition_must_be_one_model_setup(world, tmp_path):
    records = [output(world, c, GOOD[c], model="trained", provenance={"adapter_sha256": f"{i}" * 64})
               for i, c in enumerate(GOOD)]
    with pytest.raises(GroundingError, match="trained/direct/typed mixes values of provenance.adapter_sha256"):
        run(world, tmp_path, records)


# --- landmarks ---------------------------------------------------------------------------------------------------

def test_unordered_pair_correct_wrong_pair_same_id_twice_unknown_and_extra_ids(world):
    car, bus, dup = tracks(world, "C1", CAR)[0], tracks(world, "C1", BUS)[0], tracks(world, "C1", DUP)[0]

    def landmarks(*pairs):
        return score(world, "C1", output(world, "C1", proposal([11.5, 25.0], [10, 10, 30, 40],
                                                                 [{"role": r, "track_id": t} for r, t in pairs])))

    right = landmarks(("left_of", car), ("right_of", bus))
    assert right["landmarks_all_correct"] and right["pair_correct"] and right["landmarks"]["correct"] == 2
    assert landmarks(("left_of", bus), ("right_of", car))["pair_correct"]  # roles are not ordered for C1
    wrong = landmarks(("left_of", car), ("right_of", dup))
    assert not wrong["pair_correct"] and wrong["landmarks"]["correct"] == 1
    twice = landmarks(("left_of", car), ("right_of", car))
    assert not twice["pair_correct"] and twice["landmarks"]["correct"] == 1
    unknown = landmarks(("left_of", car), ("right_of", bus), ("near", 999))
    assert unknown["outcome"] == "proposed" and unknown["landmarks"]["invalid_references"] == 1
    assert unknown["landmarks"]["correct"] == 2 and not unknown["pair_correct"]
    scene = world.scenes[world.references["C1"]["scene_sha256"]]
    everything = landmarks(*[("junk", o["track_id"]) for o in scene["model_input"]["observations"]])
    assert everything["landmarks"]["correct"] == 2 and everything["landmarks"]["extra"] == 2
    assert not everything["pair_correct"]
    none = landmarks()
    assert none["landmarks"]["correct"] == 0 and not none["pair_correct"]


def test_ordered_roles_swapped_or_padded_are_wrong(world):
    car, bus, dup = tracks(world, "C2", CAR)[0], tracks(world, "C2", BUS)[0], tracks(world, "C2", DUP)[0]

    def named(*pairs):
        result = proposal([4.0, 44.0], [0, 40, 8, 48], [{"role": r, "track_id": t} for r, t in pairs])
        return score(world, "C2", output(world, "C2", result))

    assert named(("first", car), ("second", bus))["pair_correct"]
    swapped = named(("first", bus), ("second", car))
    assert not swapped["pair_correct"] and swapped["landmarks"]["correct"] == 0
    padded = named(("first", car), ("second", bus), ("third", dup))
    assert padded["landmarks"]["correct"] == 2 and not padded["pair_correct"]


def test_landmarks_given_as_boxes(world):
    def boxes(case, given):
        point_box = ([11.5, 25.0], [10, 10, 30, 40]) if case == "C1" else ([4.0, 44.0], [0, 40, 8, 48])
        return score(world, case, output(world, case, dict(proposal(*point_box), landmark_boxes=given)))

    right = boxes("C1", [[5.0, 21.0, 24.0, 36.0], BUS])
    assert right["landmarks"]["form"] == "boxes" and right["pair_correct"]
    assert not boxes("C1", [CAR, BUS, DUP])["pair_correct"]  # an extra box
    assert not boxes("C1", [CAR, CAR])["pair_correct"]  # one anchor twice
    assert boxes("C1", [CAR, DUP])["landmarks"]["correct"] == 1
    ordered = boxes("C2", [BUS, CAR])
    assert ordered["pair_correct"] and ordered["landmarks"]["ordered_roles_not_checked"]


def test_duplicate_predicted_boxes_cannot_fill_two_anchors(world):
    first, second = tracks(world, "C1", DUP)[0], tracks(world, "C1", DUP2)[0]
    bus = tracks(world, "C1", BUS)[0]
    reference = dict(world.references["C1"], anchors=[{"role": "a", "bbox_xyxy": DUP}, {"role": "b", "bbox_xyxy": BUS}])
    scene = world.scenes[reference["scene_sha256"]]

    def named(*ids):
        return g.landmark_score({"landmarks": [{"role": "x", "track_id": t} for t in ids]}, reference, scene, PROTOCOL)

    assert named(first, second)["correct"] == 1 and not named(first, second)["pair_correct"]
    assert named(second, bus)["pair_correct"] and named(first, bus)["pair_correct"]


def test_anchor_missing_from_the_scene_is_reported_for_every_condition(world, tmp_path):
    records = [output(world, c, GOOD[c], model=m) for c in GOOD for m in ("pretrained", "trained")]
    summary, _ = run(world, tmp_path, records)
    for condition in summary["conditions"].values():
        assert condition["initial"]["landmarks"]["cases_with_an_anchor_vision_did_not_detect"] == 1


# --- clarification episodes --------------------------------------------------------------------------------------

REPLIES = [{"case_id": "A1", "reply_id": "A1-r1", "kind": "answer", "text": "the gap after the bus"},
           {"case_id": "A1", "reply_id": "A1-r2", "kind": "answer", "text": "after the bus, by the kerb"},
           {"case_id": "A1", "reply_id": "A1-r3", "kind": "rejection", "text": "no, further right, after the bus"},
           {"case_id": "C1", "reply_id": "C1-r1", "kind": "rejection", "text": "no, the bay on the left"},
           {"case_id": "M1", "reply_id": "M1-r1", "kind": "answer", "text": "the place I mean is not in view"},
           {"case_id": "M1", "reply_id": "M1-r2", "kind": "rejection", "text": "no, it is not in the picture"}]
REPLY_INDEX = {(r["case_id"], r["reply_id"]): dict(r, sha256=sha(r["text"])) for r in REPLIES}
ASK = {"result": status("needs_clarification")}


def confirmed(proposal_id):
    return {"proposal_id": proposal_id, "confirmed_proposal_id": proposal_id}


def episode(world, *turns, budget=3, case="A1", model="pretrained"):
    built = []
    for index, turn in enumerate(turns):
        reply = REPLY_INDEX.get((case, turn.get("reply_id")))
        base = {"turn": index, "scene_sha256": world.references[case]["scene_sha256"], "raw_response": None,
                "reply_id": None, "reply_sha256": reply["sha256"] if reply else None,
                "proposal_id": None, "confirmed_proposal_id": None}
        if "outcome" not in turn:
            base["call"] = {"outcome": "ok", "error_code": None}
        built.append(dict(base, **turn))
    return output(world, case, model=model, mode="clarification", provenance={"turn_budget": budget}, turns=built)


def clarify(world, case, record):
    return score(world, case, record, mode="clarification", replies=REPLY_INDEX)


def test_clarify_then_propose_is_scored_initial_and_after_intervention(world):
    case = clarify(world, "A1", episode(world, ASK, {"result": INSIDE_A1, "reply_id": "A1-r1", **confirmed("p1")}))
    assert case["initial_outcome"] == "needs_clarification" and case["initial_correct"] and case["asked"]
    assert case["outcome"] == "proposed" and case["completed"] and case["final_point_in_mask"]
    assert case["correct_behaviour"] and case["turns_used"] == 2 and not case["initial_proposed"]


def test_an_ambiguous_case_needs_an_ask_before_the_confirmed_proposal(world):
    guess = clarify(world, "A1", episode(world, {"result": INSIDE_A1, **confirmed("p1")}))
    assert guess["completed"] and guess["final_point_in_mask"] and not guess["correct_behaviour"]
    clear = clarify(world, "C1", episode(world, {"result": GOOD["C1"], **confirmed("p1")}, case="C1"))
    assert clear["completed"] and clear["correct_behaviour"]  # a clear case needs no question


def test_unconfirmed_wrong_or_over_budget_episodes_do_not_complete(world):
    unconfirmed = episode(world, ASK, {"result": INSIDE_A1, "reply_id": "A1-r1", "proposal_id": "p1"})
    assert not clarify(world, "A1", unconfirmed)["completed"]
    other_id = episode(world, ASK, {"result": INSIDE_A1, "reply_id": "A1-r1", "proposal_id": "p1",
                                    "confirmed_proposal_id": "p0"})
    assert not clarify(world, "A1", other_id)["completed"]
    wrong = clarify(world, "A1", episode(world, ASK, {"result": OUTSIDE_A1, "reply_id": "A1-r1", **confirmed("p1")}))
    assert wrong["completed"] and not wrong["final_point_in_mask"] and wrong["final_incorrect_proposal"]
    long = clarify(world, "A1", episode(world, ASK, dict(ASK, reply_id="A1-r1"),
                                        {"result": INSIDE_A1, "reply_id": "A1-r2", **confirmed("p2")}, budget=2))
    assert long["outcome"] == "proposed" and not long["within_budget"] and not long["completed"]
    assert not long["correct_behaviour"]


def test_rejection_then_corrected_proposal_and_the_responder_rule(world):
    record = episode(world, ASK, {"result": OUTSIDE_A1, "reply_id": "A1-r1", "proposal_id": "p1"},
                     {"result": INSIDE_A1, "reply_id": "A1-r3", **confirmed("p2")})
    case = clarify(world, "A1", record)
    assert case["completed"] and case["final_point_in_mask"] and case["incorrect_proposals_after_replies"] == 1
    answer_after_proposal = episode(world, ASK, {"result": OUTSIDE_A1, "reply_id": "A1-r1", "proposal_id": "p1"},
                                    {"result": INSIDE_A1, "reply_id": "A1-r2", **confirmed("p2")})
    case = clarify(world, "A1", answer_after_proposal)
    assert case["outcome"] == "invalid_output" and not case["completed"]
    assert case["reasons"] == ["reply kind does not follow the responder rule"]


def test_replies_must_be_the_registered_ones_and_turn_0_must_come_first(world):
    for turn, reason in (({"reply_id": "made-up"}, "reply is not pre-registered for this case"),
                         ({"reply_id": "A1-r1", "reply_sha256": sha("something else")},
                          "reply_sha256 does not match the registered reply text")):
        case = clarify(world, "A1", episode(world, ASK, dict({"result": INSIDE_A1, **confirmed("p1")}, **turn)))
        assert case["initial_outcome"] == "needs_clarification"  # turn 0 came before any reply and still counts
        assert case["outcome"] == "invalid_output" and not case["completed"] and case["reasons"] == [reason]
    after_reply = clarify(world, "A1", episode(world, {"result": INSIDE_A1, "reply_id": "A1-r1"}))
    assert after_reply["initial_outcome"] == "invalid_output" and not after_reply["initial_correct"]
    assert after_reply["reasons"] == ["turn 0 must come before any passenger reply"]
    shifted = episode(world, {"result": INSIDE_A1})
    shifted["turns"][0]["turn"] = 1
    assert clarify(world, "A1", shifted)["reasons"] == ["turns must be numbered 0, 1, 2, ..."]
    for turns in ([], None):
        broken = output(world, "A1", mode="clarification", turns=turns)
        assert clarify(world, "A1", broken)["reasons"] == ["clarification output without turns"]


def test_reused_proposal_id_and_turns_after_a_confirmation_are_invalid(world):
    reused = episode(world, {"result": proposal([20.0, 25.0], [15, 15, 25, 35]), "proposal_id": "p1"},
                     {"result": GOOD["C1"], "reply_id": "C1-r1", **confirmed("p1")}, case="C1")
    assert clarify(world, "C1", reused)["reasons"] == ["proposal ID malformed or reused"]
    continued = episode(world, {"result": INSIDE_A1, **confirmed("p1")}, dict(ASK, reply_id="A1-r3"))
    assert clarify(world, "A1", continued)["reasons"] == ["the episode continues after a confirmation"]


def test_missing_reference_episode_that_proposes_is_wrong_even_after_reverting(world):
    record = episode(world, ASK, {"result": proposal([40.0, 20.0], [30, 10, 50, 30]), "reply_id": "M1-r1",
                                  "proposal_id": "p1"},
                     {"result": GOOD["M1"], "reply_id": "M1-r2"}, case="M1")
    case = clarify(world, "M1", record)
    assert case["outcome"] == "insufficient_visual_evidence" and case["within_budget"]
    assert case["final_incorrect_proposal"] and not case["correct_behaviour"]
    clean = clarify(world, "M1", episode(world, ASK, {"result": GOOD["M1"], "reply_id": "M1-r1"}, case="M1"))
    assert clean["correct_behaviour"] and not clean["final_incorrect_proposal"]


def test_a_timeout_or_wrong_scene_inside_an_episode(world):
    timeout = clarify(world, "A1", episode(world, ASK, {"result": None, "reply_id": "A1-r1",
                                                        "call": {"outcome": "error", "error_code": "timeout"}}))
    assert timeout["initial_outcome"] == "needs_clarification" and timeout["outcome"] == "request_error"
    assert timeout["error_code"] == "timeout"
    moved = clarify(world, "A1", episode(world, ASK, {"result": INSIDE_A1, "reply_id": "A1-r1", **confirmed("p1"),
                                                      "scene_sha256": world.references["C2"]["scene_sha256"]}))
    assert moved["outcome"] == "wrong_scene" and not moved["completed"]


REAL_TURN = {"backend_kind": "transformers_local", "injected": False, "requested_condition": "pretrained",
             "model_id": "base", "revision": "rev"}
ASKED = {"ok": True, "request_id": "A1", "provenance": REAL_TURN,
         "result": {"status": "needs_clarification", "destination_point_xy": None, "destination_box_xyxy": None,
                    "clarification_question": "Before or after the bus?"}}
PROPOSED = {"ok": True, "request_id": "A1", "provenance": REAL_TURN,
            "result": {"status": "proposed", "destination_point_xy": [60.0, 44.0],
                       "destination_box_xyxy": [50.0, 40.0, 70.0, 48.0], "clarification_question": None}}


def envelope_episode(world, first=ASKED, second=PROPOSED):
    record = episode(world, {"outcome": first}, {"outcome": second, "reply_id": "A1-r1", **confirmed("p1")})
    record["provenance"]["base_revision"] = "base@rev"
    return record


def test_turns_can_carry_task5_outcome_envelopes(world):
    case = clarify(world, "A1", envelope_episode(world))
    assert case["correct_behaviour"] and case["turns_used"] == 2
    failed = {"ok": False, "request_id": "A1", "provenance": REAL_TURN, "error": {"code": "timeout", "message": "slow"}}
    assert clarify(world, "A1", episode(world, {"outcome": failed}))["outcome"] == "request_error"
    zero = dict(failed, ok=0)  # 0 == False in Python, but it is not a Task 5 envelope
    assert clarify(world, "A1", episode(world, {"outcome": zero}))["reasons"] == ["a turn without a call outcome"]


def test_turns_from_another_backend_model_or_request_are_refused(world, tmp_path):
    replies = write_jsonl(tmp_path / "replies.jsonl", REPLIES)
    for change, reason in (({"injected": True, "backend_kind": "injected_test_backend"},
                            "a turn came from a stand-in or another backend"),
                           ({"requested_condition": "separate_pretrained_capacity_baseline"},
                            "a turn came from another model condition"),
                           ({"revision": "other"}, "a turn came from another base model"),
                           ({"adapter": {"manifest_sha256": "m" * 64}}, "a turn came from another adapter")):
        bad = dict(PROPOSED, provenance=dict(REAL_TURN, **change))
        summary, _ = run(world, tmp_path, [envelope_episode(world, second=bad)], replies_path=replies)
        assert [r["reason"] for r in summary["refused_outputs"]] == [reason]
    summary, _ = run(world, tmp_path, [envelope_episode(world, second=dict(PROPOSED, request_id="C1"))],
                     replies_path=replies)
    assert summary["refused_outputs"][0]["reason"] == "a turn's Task 5 outcome is for another request"
    summary, _ = run(world, tmp_path, [envelope_episode(world)], replies_path=replies)
    assert summary["refused_outputs"] == []
    assert summary["conditions"]["pretrained/clarification/typed"]["correct_behaviour"]["count"] == 1


def test_every_turn_must_name_the_case_scene(world):
    record = episode(world, ASK, {"result": INSIDE_A1, "reply_id": "A1-r1", **confirmed("p1")})
    for turn in record["turns"]:
        del turn["scene_sha256"]
    case = clarify(world, "A1", record)
    assert case["initial_outcome"] == "wrong_scene" and not case["completed"] and not case["correct_behaviour"]


def test_clarification_summary_reports_initial_and_after_intervention_separately(world, tmp_path):
    replies = write_jsonl(tmp_path / "replies.jsonl", REPLIES)
    records = [episode(world, {"result": GOOD["C1"], **confirmed("p1")}, case="C1"),
               episode(world, {"result": GOOD["C2"], **confirmed("p2")}, case="C2"),
               episode(world, ASK, {"result": INSIDE_A1, "reply_id": "A1-r1", **confirmed("p3")}),
               episode(world, {"result": GOOD["M1"]}, case="M1")]
    summary, _ = run(world, tmp_path, records, replies_path=replies)
    condition = summary["conditions"]["pretrained/clarification/typed"]
    assert condition["initial"]["correct"]["count"] == 4 and condition["initial"]["proposal_coverage"]["count"] == 2
    after = condition["after_intervention"]
    assert after["completion_coverage"]["count"] == 3 and after["correct"]["count"] == 4
    assert after["ambiguous_confirmed_inside_intended_mask_after_asking"]["count"] == 1 and after["over_budget"] == 0
    assert after["replies"].startswith("pre-registered")
    records[2] = episode(world, ASK, {"result": None, "reply_id": "A1-r1",
                                      "call": {"outcome": "error", "error_code": "timeout"}})
    summary, _ = run(world, tmp_path, records, replies_path=replies)
    condition = summary["conditions"]["pretrained/clarification/typed"]
    assert condition["initial"]["failed_or_invalid"]["count"] == 0 and condition["failed_or_invalid"]["count"] == 1
    assert summary["headline"]["conditions"]["pretrained/clarification/typed"]["failed_or_invalid"] == 0.25


# --- comparisons and the headline --------------------------------------------------------------------------------

def test_pretrained_and_trained_are_compared_case_by_case(world, tmp_path):
    trained = dict(GOOD, C2=proposal([40.0, 44.0], [0, 40, 80, 48]))
    records = [output(world, c, GOOD[c]) for c in GOOD] + [output(world, c, trained[c], model="trained") for c in GOOD]
    timeout = output(world, "M1", None, model="trained", call={"outcome": "error", "error_code": "timeout"})
    del timeout["scene_sha256"], timeout["instruction_sha256"]
    records[-1] = timeout
    summary, _ = run(world, tmp_path, records)
    comparison = summary["comparisons"][0]
    assert comparison["first"] == "pretrained/direct/typed" and "refused" not in comparison
    result = comparison["correct_behaviour"]
    assert (result["first_correct"], result["second_correct"]) == (4, 2)
    assert (result["only_first_correct"], result["only_second_correct"]) == (2, 0)
    assert result["mcnemar_exact_p_assumes_independent_cases"] == 0.5
    assert result["difference_second_minus_first"] == -0.5
    clear = comparison["clear_point_in_mask"]
    assert (clear["cases"], clear["first_correct"], clear["second_correct"]) == (2, 2, 1)
    headline = summary["headline"]
    assert headline["conditions"]["trained/direct/typed"]["clear_point_in_mask"] == 0.5
    assert headline["trained_vs_pretrained"][0]["correct_behaviour"]["difference_second_minus_first"] == -0.5


def test_a_model_that_always_asks_gets_no_clear_credit_and_a_baseline_shows_its_score(world, tmp_path):
    ask = status("needs_clarification")
    records = [output(world, c, GOOD[c]) for c in GOOD] + [output(world, c, ask, model="trained") for c in GOOD]
    summary, _ = run(world, tmp_path, records)
    comparison = summary["comparisons"][0]
    assert comparison["correct_behaviour"]["second_correct"] == 2
    assert comparison["clear_point_in_mask"]["second_correct"] == 0
    constant = summary["fixed_point_baseline"]["constant_answer_correct_behaviour"]
    assert constant["always_needs_clarification"] == {"count": 2, "n": 4, "rate": 0.5}
    assert constant["always_insufficient_visual_evidence"]["count"] == 1


def test_mismatched_conditions_are_refused_named_and_labelled(world, tmp_path):
    records = [output(world, c, GOOD[c]) for c in GOOD] + [
        output(world, c, GOOD[c], model="trained", provenance={"image_budget": 512}) for c in GOOD]
    summary, _ = run(world, tmp_path, records)
    assert summary["comparisons"][0]["refused"] == ["provenance.image_budget differs"]
    assert "correct_behaviour" not in summary["comparisons"][0]
    assert "required comparison pretrained/direct/typed vs trained/direct/typed refused" in summary["status"]
    assert any("refused" in w for w in summary["warnings"])
    assert summary["headline"]["trained_vs_pretrained"][0]["not_available"] == ["provenance.image_budget differs"]


def test_missing_conditions_are_listed(world, tmp_path):
    summary, _ = run(world, tmp_path, [output(world, c, GOOD[c]) for c in GOOD])
    assert summary["required_conditions_missing"] == ["pretrained/clarification/typed", "trained/direct/typed",
                                                      "trained/clarification/typed"]
    assert "required conditions missing: pretrained/clarification/typed, trained/direct/typed" in summary["status"]
    assert {c["second"]: c["not_run"] for c in summary["comparisons"]} == {
        "trained/direct/typed": "no outputs for trained/direct/typed",
        "pretrained/clarification/typed": "no outputs for pretrained/clarification/typed"}


def test_direct_and_clarification_are_compared_on_the_same_cases(world, tmp_path):
    replies = write_jsonl(tmp_path / "replies.jsonl", REPLIES)
    direct = [output(world, c, GOOD[c]) for c in GOOD]
    clarification = [episode(world, {"result": GOOD[c], **(confirmed(f"p-{c}") if c in ("C1", "C2") else {})}, case=c)
                     for c in GOOD]
    summary, _ = run(world, tmp_path, direct + clarification, replies_path=replies)
    entry = [c for c in summary["comparisons"] if c["second"] == "pretrained/clarification/typed"][0]
    assert entry["initial_correct"]["only_first_correct"] == 0 and entry["initial_correct"]["cases"] == 4
    assert entry["clear_destination_reached"]["cases"] == 2
    assert entry["clear_destination_reached"]["second_correct"] == 2


# --- references and masks ----------------------------------------------------------------------------------------

def test_masks_are_read_like_task5_reads_them(world, tmp_path):
    masks = {"C1": (c_mask(), "RGB"), "C2": (two_region_mask(), "RGBA"), "A1": (intended_mask(), "P")}
    references, _ = g.load_references(write_references(tmp_path / "colour", world.rows, masks), world.scenes)
    assert references["C1"]["mask"].sum() == c_mask().sum() and references["C2"]["mask"].sum() == 128
    double = {case: (np.kron(make(), np.ones((2, 2))).astype(bool), mode) for case, (make, mode) in MASKS.items()}
    references, _ = g.load_references(write_references(tmp_path / "double", world.rows, double), world.scenes)
    reference = references["C1"]
    assert reference["mask"].shape == (2 * H, 2 * W)
    assert g.in_mask(reference["mask"], [11.5, 25.0], W, H) and not g.in_mask(reference["mask"], [20.0, 25.0], W, H)
    assert g.mask_box(reference["mask"], W, H) == [10.0, 10.0, 30.0, 40.0]
    assert g.distance_to_mask(reference["mask"], [20.0, 25.0], W, H) == 6.5  # measured at the mask's resolution
    (tmp_path / "jpeg" / "masks").mkdir(parents=True)
    path = write_references(tmp_path / "jpeg", world.rows)
    Image.open(path.parent / "masks" / "C1.png").convert("L").save(path.parent / "masks" / "C1.png", format="JPEG")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["mask_sha256"] = sha((path.parent / "masks" / "C1.png").read_bytes())
    with pytest.raises(GroundingError, match="masks must be PNG files"):
        g.load_references(write_jsonl(path, rows), world.scenes)


def test_empty_mask_wrong_box_redrawn_mask_and_misplaced_masks_are_refused(world, tmp_path):
    masks = {"C1": (np.zeros((H, W), bool), "1"), "C2": (two_region_mask(), "L"), "A1": (intended_mask(), "P")}
    with pytest.raises(GroundingError, match="the mask is empty"):
        g.load_references(write_references(tmp_path / "b", world.rows, masks), world.scenes)
    rows = json.loads(json.dumps(world.rows))
    rows[0]["target_box"] = [10, 10, 29, 40]
    with pytest.raises(GroundingError, match="differs from the mask's pixel_bbox"):
        g.load_references(write_references(tmp_path / "c", rows), world.scenes)
    path = write_references(tmp_path / "d", world.rows)
    redrawn = np.zeros((H, W), bool)
    redrawn[40:48, 0:20] = True
    save_mask(tmp_path / "d" / "masks" / "A1.png", redrawn, "P")
    with pytest.raises(GroundingError, match="intended_mask_sha256 does not match"):
        g.load_references(path, world.scenes)
    rows = json.loads(json.dumps(world.rows))
    rows[2]["mask"] = "masks/C1.png"
    with pytest.raises(GroundingError, match="an ambiguous case has no single destination mask"):
        g.load_references(write_references(tmp_path / "e", rows), world.scenes)


def test_palette_mask_is_accepted(world):
    mask = world.references["A1"]["intended_mask"]
    assert mask.dtype == bool and mask.sum() == intended_mask().sum() and mask[44, 60] and not mask[44, 40]


def test_unreviewed_references_are_refused_unless_allowed_and_then_labelled(world, tmp_path):
    rows = json.loads(json.dumps(world.rows))
    rows[2]["review"] = {"independent_reviews": 1, "disagreements_resolved": True}
    path = write_references(tmp_path / "refs", rows)
    with pytest.raises(GroundingError, match="two independent reviews"):
        g.load_references(path, world.scenes)
    summary, _ = run(world, tmp_path, [output(world, "C1", GOOD["C1"])], references=path, allow_unreviewed=True)
    assert "some references not fully reviewed" in summary["status"]
    assert "A1: reference not fully reviewed" in summary["warnings"]


def test_reference_must_match_its_scene_and_carry_its_instruction(world, tmp_path):
    rows = json.loads(json.dumps(world.rows))
    rows[0]["frame_id"] = 2
    with pytest.raises(GroundingError, match="frame_id differs from its scene"):
        g.load_references(write_references(tmp_path / "refs", rows), world.scenes)
    rows = json.loads(json.dumps(world.rows))
    rows[0]["reference_point"] = [10 ** 400, 25.0]
    with pytest.raises(GroundingError, match="reference_point must lie inside the mask"):
        g.load_references(write_references(tmp_path / "huge", rows), world.scenes)
    rows = json.loads(json.dumps(world.rows))
    del rows[1]["instruction"]
    with pytest.raises(GroundingError, match="instruction is required"):
        g.load_references(write_references(tmp_path / "text", rows), world.scenes)


def test_text_files_hash_the_same_with_windows_line_endings(world, tmp_path):
    lf = write_references(tmp_path / "lf", world.rows)
    crlf = write_references(tmp_path / "crlf", world.rows, encoding="utf-8-sig", newline="\r\n")
    assert lf.read_bytes() != crlf.read_bytes()
    plain = tmp_path / "plain.txt"
    plain.write_bytes(b"a\nb\n")
    windows = tmp_path / "windows.txt"
    windows.write_bytes(b"a\r\nb\r\n")
    assert g.text_sha256(plain) == g.text_sha256(windows) != g.sha256_file(windows)
    assert len(g.load_references(crlf, world.scenes)[0]) == 4


# --- statistics and diagnostics ----------------------------------------------------------------------------------

def test_statistics_edge_cases_and_known_values():
    assert g.wilson(0, 0) is None
    assert g.wilson(5, 10) == pytest.approx([0.2366, 0.7634], abs=1e-4)
    assert g.mcnemar_exact(0, 0) == 1.0 and g.mcnemar_exact(0, 5) == 0.0625 and g.mcnemar_exact(3, 3) == 1.0
    assert g.group_bootstrap({"g1": [1.0, 1.0], "g2": [0.0]}, 2000, 7) == [0.0, 1.0]
    assert g.group_bootstrap({"g1": [1.0, 0.0], "g2": [0.0, 1.0]}, 500, 7) == [0.5, 0.5]
    data = {"g1": [1.0, 0.0], "g2": [1.0], "g3": [0.0, 0.0, 1.0]}
    first = g.group_bootstrap(data, 500, 7)
    assert first == g.group_bootstrap(data, 500, 7) and 0.0 <= first[0] <= first[1] <= 1.0
    assert g.group_bootstrap({}, 500, 7) is None and g.group_bootstrap({"g1": [1.0, 0.0]}, 500, 7) is None
    assert g.rate([], "x", PROTOCOL)["rate"] is None


def test_fixed_point_baseline_and_mask_shares(world, tmp_path):
    summary, _ = run(world, tmp_path, [output(world, "C1", GOOD["C1"])])
    baseline = summary["fixed_point_baseline"]
    assert baseline["clear_cases"] == 2 and baseline["point_fraction"] == [0.5, 0.8]
    assert baseline["point_in_mask"] == 0  # (40, 38) misses both clear masks
    assert baseline["mask_area_share"]["min"] == pytest.approx(128 / (W * H))
    assert baseline["intended_mask_area_share"]["max"] == pytest.approx(160 / (W * H))
    assert sorted(baseline["per_case_mask_area_share"]) == ["A1", "C1", "C2"]


def test_road_user_box_near_constant_and_scene_pinning_diagnostics(world, tmp_path):
    records = [output(world, "C1", proposal([10.0, 25.0], CAR), raw_response='```json\n[{"point_2d": [125, 521]}]\n```'),
               output(world, "C2", GOOD["C2"])]
    summary, _ = run(world, tmp_path, records)
    diagnostics = summary["conditions"]["pretrained/direct/typed"]["diagnostics"]
    assert diagnostics["proposed_boxes_matching_a_road_user_box"]["count"] == 1
    assert diagnostics["coordinate_conversions_checked"] == 1 and diagnostics["coordinate_conversion_mismatches"] == 0
    assert diagnostics["box_size_spread"][0] is not None and diagnostics["box_sizes_near_constant"] is None
    assert "2 scene(s) made without --frame-manifest and --run-manifest" in summary["warnings"]
    assert g._near_constant([0.001, 0.0], 5, PROTOCOL) is True
    assert g._near_constant([0.2, 0.0], 5, PROTOCOL) is False and g._near_constant([0.0, 0.0], 4, PROTOCOL) is None


# --- voice, leakage, protocol and command line -------------------------------------------------------------------

def voice_row(case, human, asr, **changes):
    return dict({"case_id": case, "human_transcript": human, "asr_transcript": asr, "recognizer": "asr-x",
                 "speech": "read", "noise": "none", "consent": True, "source": "team recording"}, **changes)


def test_voice_wer_keeps_failed_recognition_and_compares_human_with_asr(world, tmp_path):
    voice = write_jsonl(tmp_path / "voice.jsonl", [
        voice_row("C1", "Stop near the Car.", "stop near car", critical_errors=["landmark dropped"]),
        voice_row("C2", "Drop me at the corner", "", noise="real")])
    records = [output(world, "C1", GOOD["C1"], transcript="human", text="Stop near the Car."),
               output(world, "C2", GOOD["C2"], transcript="human", text="Drop me at the corner"),
               output(world, "C1", GOOD["C1"], transcript="asr", text="stop near car"),
               output(world, "C2", status("insufficient_visual_evidence"), transcript="asr", text="")]
    summary, _ = run(world, tmp_path, records, voice_path=voice)
    assert summary["voice_cases"] == 2 and summary["conditions"]["pretrained/direct/asr"]["cases"] == 2
    assert summary["conditions"]["pretrained/direct/asr"]["initial"]["outcomes"] == {"proposed": 1,
                                                                                    "wrong_instruction": 1}
    voice_result = summary["voice"]
    assert voice_result["corpus_wer"] == pytest.approx(6 / 9)
    assert voice_result["corpus_wer_recognised_only"] == 0.25 and voice_result["empty_or_failed_recognition"] == 1
    assert voice_result["critical_errors"] == {"landmark dropped": 1} and voice_result["noise"] == {"none": 1, "real": 1}
    downstream = voice_result["downstream"][0]
    assert downstream["human"]["correct_behaviour"]["count"] == 2 and downstream["asr"]["correct_behaviour"]["count"] == 1
    assert downstream["paired_correct_behaviour"]["only_first_correct"] == 1


def test_word_normalisation():
    assert g.normalise_words("Don't stop, HERE!") == g.normalise_words("dont stop here") == ["dont", "stop", "here"]
    assert g.normalise_words("Stop here—by the bus") == ["stop", "here", "by", "the", "bus"]
    assert g.normalise_words("left/right") == ["left", "right"]
    assert g.word_edits(g.normalise_words("Stop here—by the bus"), g.normalise_words("stop here by the bus")) == 0


def test_voice_rows_must_be_reserved_cases_with_consent_and_text(world, tmp_path):
    for row, message in ((voice_row("Z1", "a", "a"), "not a reserved reference"),
                         (voice_row("C1", "a", "a", consent=False), "consent, source and recognizer"),
                         (voice_row("C1", "a", "a", noise="loud"), "noise none, real or added"),
                         (voice_row("C1", 5, "a"), "human_transcript must be text"),
                         (voice_row("C1", "a", ["a"]), "asr_transcript must be text or null")):
        voice = write_jsonl(tmp_path / "voice.jsonl", [row])
        with pytest.raises(GroundingError, match=message):
            run(world, tmp_path, [output(world, "C1", GOOD["C1"])], voice_path=voice)
    with pytest.raises(GroundingError, match="need the voice file"):
        run(world, tmp_path, [output(world, "C1", GOOD["C1"], transcript="asr")])
    voice = write_jsonl(tmp_path / "voice.jsonl", [voice_row("C1", "stop here", "stop here")])
    with pytest.raises(GroundingError, match="Output C2 in pretrained/direct/human is not a voice case"):
        run(world, tmp_path, [output(world, "C2", GOOD["C2"], transcript="human")], voice_path=voice)


def test_leakage_reads_every_string_in_any_manifest_shape(world, tmp_path):
    manifest = write_jsonl(tmp_path / "train.jsonl", [{"case_id": "C1"}, {"case_id": "other"}])
    summary, _ = run(world, tmp_path, [output(world, "C1", GOOD["C1"])], train_manifests=[manifest])
    assert summary["leakage"]["violations"] == [
        {"manifest": "1:train.jsonl", "field": "case_id", "value": "C1", "occurrences": 1}]
    assert "reserved cases appear in training or selection data" in summary["status"]
    assert len(summary["leakage"]["manifests_sha256"]["1:train.jsonl"]) == 64
    shapes = {"strings.json": json.dumps(["C10", "C1"]),
              "nested.json": json.dumps({"train": [{"scene": {"recording_id": REC}}]}, indent=2),
              "paths.jsonl": json.dumps({"image": f"images/train/images/{REC}/00001.jpg"}) + "\n",
              "keyed.json": json.dumps({"g2": ["x"]}),
              "case_file.json": json.dumps(["data/cases/M1.json"]),
              "group_folder.json": json.dumps(["data\\g3\\img.png"]),
              "bom.json": "﻿" + json.dumps([world.references["A1"]["scene_sha256"]])}
    expected = {"strings.json": ("case_id", "C1"), "nested.json": ("recording_id", REC),
                "paths.jsonl": ("recording_id", REC), "keyed.json": ("group_id", "g2"),
                "case_file.json": ("case_id", "M1"), "group_folder.json": ("group_id", "g3"),
                "bom.json": ("scene_sha256", world.references["A1"]["scene_sha256"])}
    for name, text in shapes.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
        hits = g.leakage(world.references, [tmp_path / name])["violations"]
        assert [(h["field"], h["value"]) for h in hits] == [expected[name]], name
    references = {"X": {"recording_id": "video_054907", "group_id": "gx", "scene_sha256": "s" * 64}}
    (tmp_path / "adjacent.json").write_text(json.dumps([{"recording_id": "video_054604"}]), encoding="utf-8")
    assert g.leakage(references, [tmp_path / "adjacent.json"])["violations"][0]["value"] == "video_054604"
    assert "case.001" in g._parts("data/case.001.json") and "case" in g._parts("data/case.001.json")


def all_conditions(world):
    return [output(world, c, GOOD[c], model=m) for c in GOOD for m in g.MODELS] + [
        episode(world, {"result": GOOD[c]}, case=c, model=m) for c in GOOD for m in g.MODELS]


def protocol_file(tmp_path, world, **changes):
    path = tmp_path / "protocol.json"
    body = gi.freeze_protocol(world.references_path, evaluator_commit="abc",
                              responder_rules="after a proposal: rejection with the correction; otherwise: answer")
    body.update(changes)
    path.write_text(json.dumps({k: v for k, v in body.items() if v != "drop"}), encoding="utf-8")
    return path


def test_frozen_protocol_leaves_only_the_stub_label_and_refuses_changes(world, tmp_path):
    clean = write_jsonl(tmp_path / "train.jsonl", [{"case_id": "T1", "recording_id": "video_999999"}])
    protocol = protocol_file(tmp_path, world, bootstrap={"resamples": 500, "seed": 3, "note": float("nan")})
    summary, _ = run(world, tmp_path, all_conditions(world), protocol_path=protocol, train_manifests=[clean])
    assert summary["status"] == "not assessed: " + g.STUB_LABEL
    assert summary["protocol"]["evaluator_commit"] == "abc" and len(summary["protocol_sha256"]) == 64
    assert summary["protocol"]["bootstrap"] == {"resamples": 500, "seed": 3}
    assert summary["evaluator_sha256"] == g.text_sha256(PROJECT / "evaluation" / "grounding.py")
    json.dumps(summary, allow_nan=False)
    for changes, message in (({"references_sha256": "0" * 64}, "references file differs"),
                             ({"responder_rules": "drop"}, "must record responder_rules"),
                             ({"replies_sha256": "drop"}, "must record replies_sha256"),
                             ({"voice_sha256": "drop"}, "must record voice_sha256"),
                             ({"box_iou": 10 ** 400}, r"protocol.box_iou must be in \(0, 1\]")):
        with pytest.raises(GroundingError, match=message):
            run(world, tmp_path, [], protocol_path=protocol_file(tmp_path, world, **changes))
    voice = write_jsonl(tmp_path / "voice.jsonl", [voice_row("C1", "stop here", "stop here")])
    with pytest.raises(GroundingError, match="voice file differs"):
        run(world, tmp_path, [], protocol_path=protocol_file(tmp_path, world), voice_path=voice)


def test_command_line_writes_results_prints_the_headline_and_refuses_bad_input(world, tmp_path, capsys):
    outputs = write_jsonl(tmp_path / "outputs.jsonl", [output(world, c, GOOD[c]) for c in GOOD])
    out, cases = tmp_path / "summary.json", tmp_path / "cases.csv"
    args = ["--references", str(world.references_path), "--scenes", str(world.scenes_dir), "--outputs", str(outputs),
            "--allow-stub-scenes", "--out", str(out), "--cases-out", str(cases)]
    assert g.main(args) == 0
    printed = capsys.readouterr().out
    assert "pretrained/direct/typed" in printed and "trained/direct/typed vs pretrained/direct/typed" in printed
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["status"].startswith("not assessed: protocol not frozen in advance")
    assert summary["conditions"]["pretrained/direct/typed"]["correct_behaviour"]["count"] == 4
    assert cases.read_text(encoding="utf-8").count("\n") == 5
    assert g.main(args) == 2  # the outputs already exist
    bad = write_jsonl(tmp_path / "bad.jsonl", [dict(output(world, "C1", GOOD["C1"]), condition={"model": "x"})])
    assert g.main(args[:4] + ["--outputs", str(bad), "--allow-stub-scenes", "--out", str(tmp_path / "n.json")]) == 2
    assert not (tmp_path / "n.json").exists()
    same = args[:7] + ["--out", str(tmp_path / "s.json"), "--cases-out", str(tmp_path / "s.json")]
    assert g.main(same) == 2 and not (tmp_path / "s.json").exists()
    assert g.main(args[:7] + ["--out", str(tmp_path / "missing" / "s.json")]) == 2


def test_unreadable_masks_and_bad_replies_are_refused_without_a_traceback(world, tmp_path, monkeypatch):
    path = write_references(tmp_path / "refs", world.rows)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["mask"] = "masks/C1\x00.png"
    with pytest.raises(GroundingError, match="Cannot read"):
        g.load_references(write_jsonl(tmp_path / "refs" / "nul.jsonl", rows), world.scenes)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(GroundingError, match="Cannot read mask"):
        g.load_references(path, world.scenes)
    monkeypatch.undo()
    for row, message in (({"case_id": "Z9", "reply_id": "r", "kind": "answer", "text": "x"}, "must be a reserved case"),
                         ({"case_id": "A1", "reply_id": "r", "kind": "correction", "text": "x"},
                          "kind must be one of answer, rejection")):
        replies = write_jsonl(tmp_path / "replies.jsonl", [row])
        with pytest.raises(GroundingError, match=message):
            run(world, tmp_path, [], replies_path=replies)


def test_stub_scenes_are_refused_without_the_test_flag(world):
    with pytest.raises(GroundingError, match="was refused"):
        g.load_scenes(world.scenes_dir)


def test_help_runs_as_a_module():
    for module in ("evaluation.grounding", "evaluation.grounding_inputs"):
        result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=PROJECT, capture_output=True, text=True,
                                check=True)
        assert "usage:" in result.stdout


# --- converters for the team's formats ----------------------------------------------------------------------------

def test_protocol_command_writes_hashes_of_the_inputs(world, tmp_path):
    out = tmp_path / "protocol.json"
    assert gi.main(["protocol", "--references", str(world.references_path), "--evaluator-commit", "abc",
                    "--responder-rules", "rules", "--out", str(out)]) == 0
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["references_sha256"] == g.text_sha256(world.references_path)
    assert body["replies_sha256"] is None and body["voice_sha256"] is None and body["version"] == 1
    assert gi.main(["protocol", "--references", str(world.references_path), "--evaluator-commit", "abc",
                    "--responder-rules", "rules", "--out", str(out)]) == 2


class StandInBackend:
    """Answers each request with the next scripted reply, as Task 5's injected test backend does."""

    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, messages, *, config):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def normalised(point, box):
    return json.dumps({"status": "proposed", "destination_point_xy": [point[0] * 1000 / W, point[1] * 1000 / H],
                       "destination_box_xyxy": [box[0] * 1000 / W, box[1] * 1000 / H, box[2] * 1000 / W,
                                                box[3] * 1000 / H], "clarification_question": None})


def task5_requests(world, texts=None, drop_track=None):
    """Requests built with Task 5's own builder from the same saved predictions the scenes came from."""
    inputs = pytest.importorskip("trackrag.dropoff.inputs")
    rows = [json.loads(line) for line in (world.predictions / f"{REC}.tracks.jsonl").read_text().splitlines()]
    if drop_track is not None:
        rows = [r for r in rows if r["track_id"] != drop_track]
    recording = open_recording(world.root, REC)
    texts = texts or INSTRUCTIONS
    return [inputs.build_instruction_request(recording, rows, frame_id=3 if case in ("C1", "A1") else 2,
                                             request_id=case, text=texts[case])
            for case in ("C1", "C2", "A1", "M1")]


def task5_run(world, tmp_path, replies, requests=None, name="run"):
    model = pytest.importorskip("trackrag.dropoff.model")
    predict = pytest.importorskip("tools.dropoff_predict")
    requests_path = write_jsonl(tmp_path / f"{name}-requests.jsonl", requests or task5_requests(world))
    out = tmp_path / name
    predict.predict(requests_path, model.GroundingConfig(), out,
                    adapter_factory=lambda config: model.GroundingModel(config, backend=StandInBackend(replies)))
    return out


TASK5_REPLIES = [normalised([11.5, 25.0], [10, 10, 30, 40]), normalised([4.0, 44.0], [0, 40, 8, 48]),
                 json.dumps({"status": "needs_clarification", "destination_point_xy": None, "destination_box_xyxy": None,
                             "clarification_question": "Before or after the bus?"}),
                 json.dumps({"status": "insufficient_visual_evidence", "destination_point_xy": None,
                             "destination_box_xyxy": None, "clarification_question": None})]


def test_a_real_task5_prediction_run_converts_and_scores(world, tmp_path):
    out = task5_run(world, tmp_path, TASK5_REPLIES)
    records = gi.convert_run(out, world.references_path, world.scenes_dir, "typed", allow_stub_scenes=True)
    assert [r["case_id"] for r in records] == ["C1", "C2", "A1", "M1"]
    first = records[0]
    assert first["condition"] == {"model": "pretrained", "mode": "direct", "transcript": "typed"}
    assert first["scene_sha256"] == world.references["C1"]["scene_sha256"] and "scene_problems" not in first
    assert first["instruction_sha256"] == world.references["C1"]["instruction_sha256"]
    assert first["result"]["point"] == pytest.approx([11.5, 25.0]) and first["raw_response"]
    assert first["provenance"]["substitute"] is True  # Task 5's stand-in backend is labelled as such
    summary, _ = run(world, tmp_path, records)
    assert summary["refused_outputs"][0]["reason"] == "substitute or stub output"
    for record in records:
        record["provenance"]["substitute"] = False  # score them as if from the real model, to test the mapping
    summary, _ = run(world, tmp_path, records)
    condition = summary["conditions"]["pretrained/direct/typed"]
    assert condition["correct_behaviour"]["count"] == 4
    assert condition["diagnostics"]["coordinate_conversions_checked"] == 2
    assert condition["diagnostics"]["coordinate_conversion_mismatches"] == 0


def test_task5_failures_map_to_request_errors_and_invalid_outputs(world, tmp_path):
    out = task5_run(world, tmp_path, ["not json", TimeoutError("slow"), TASK5_REPLIES[2], TASK5_REPLIES[3]])
    records = gi.convert_run(out, world.references_path, world.scenes_dir, "typed", allow_stub_scenes=True)
    assert records[0]["invalid_reason"] == "model output could not be parsed (invalid_model_output)"
    assert records[1]["call"] == {"outcome": "error", "error_code": "timeout"}
    for record in records:
        record["provenance"]["substitute"] = False
    summary, rows = run(world, tmp_path, records)
    outcomes = {r["case_id"]: r["outcome"] for r in rows}
    assert outcomes == {"C1": "invalid_output", "C2": "request_error", "A1": "needs_clarification",
                        "M1": "insufficient_visual_evidence"}


def test_a_task5_request_from_other_predictions_is_a_wrong_scene(world, tmp_path):
    car = tracks(world, "C1", CAR)[0]
    out = task5_run(world, tmp_path, TASK5_REPLIES, requests=task5_requests(world, drop_track=car))
    records = gi.convert_run(out, world.references_path, world.scenes_dir, "typed", allow_stub_scenes=True)
    assert all(r["scene_sha256"] is None for r in records)
    assert records[0]["scene_problems"] == ["predicted observations differ (another prediction run or profile)"]


def make_run(folder, requests, outcomes, configuration):
    """A run folder in tools/dropoff_predict.py's layout, written by hand with consistent hashes."""
    folder.mkdir()
    files = {"resolved-requests.jsonl": requests, "model-outcomes.jsonl": outcomes,
             "receipts.jsonl": [{"request_id": o["request_id"]} for o in outcomes]}
    for name, rows in files.items():
        (folder / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    run_json = {"configuration": configuration, "requested_rows": len(requests), "requests_sha256": "r" * 64,
                "resolved_requests_sha256": sha((folder / "resolved-requests.jsonl").read_bytes()),
                "outcomes_sha256": sha((folder / "model-outcomes.jsonl").read_bytes()),
                "receipts_sha256": sha((folder / "receipts.jsonl").read_bytes())}
    (folder / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    return folder


def test_trained_ollama_tampered_and_unknown_runs(world, tmp_path):
    requests = task5_requests(world)
    configuration = {"model_id": "Qwen/Qwen3-VL-8B-Instruct", "revision": "a" * 40, "min_pixels": 65536,
                     "max_pixels": 262144}
    real = {"backend_kind": "transformers_local", "injected": False}

    def outcomes(condition, adapter=None, extra=None):
        provenance = dict(real, requested_condition=condition, **({"adapter": adapter} if adapter else {}))
        provenance.update(extra or {})
        return [{"ok": True, "request_id": r["request_id"], "provenance": provenance,
                 "result": {"status": "insufficient_visual_evidence", "destination_point_xy": None,
                            "destination_box_xyxy": None, "clarification_question": None}} for r in requests]

    trained = make_run(tmp_path / "trained", requests, outcomes("fine_tuned", {"manifest_sha256": "m" * 64}),
                       dict(configuration, adapter_dir="somewhere"))
    records = gi.convert_run(trained, world.references_path, world.scenes_dir, "typed", allow_stub_scenes=True)
    assert records[0]["condition"]["model"] == "trained" and records[0]["provenance"]["adapter_sha256"] == "m" * 64
    assert records[0]["provenance"]["substitute"] is False
    assert records[0]["provenance"]["base_revision"] == "Qwen/Qwen3-VL-8B-Instruct@" + "a" * 40
    ollama = make_run(tmp_path / "ollama", requests, outcomes("separate_pretrained_capacity_baseline"), configuration)
    with pytest.raises(GroundingError, match="separate baseline"):
        gi.convert_run(ollama, world.references_path, world.scenes_dir, "typed", allow_stub_scenes=True)
    (trained / "model-outcomes.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(GroundingError, match="does not match run.json"):
        gi.convert_run(trained, world.references_path, world.scenes_dir, "typed", allow_stub_scenes=True)
    stray = [dict(r, request_id="X9") if r["request_id"] == "C1" else r for r in requests]
    unknown = make_run(tmp_path / "unknown", stray, [dict(o, request_id=r["request_id"])
                                                     for o, r in zip(outcomes("pretrained"), stray)], configuration)
    with pytest.raises(GroundingError, match="request_id must equal case_id"):
        gi.convert_run(unknown, world.references_path, world.scenes_dir, "typed", allow_stub_scenes=True)
    oracle = make_run(tmp_path / "oracle", requests, outcomes("pretrained"), dict(configuration, allow_annotated=True))
    with pytest.raises(GroundingError, match="annotated"):
        gi.convert_run(oracle, world.references_path, world.scenes_dir, "typed", allow_stub_scenes=True)
    labelled = [dict(r, scene=dict(r["scene"], source="annotated")) for r in requests]
    annotated = make_run(tmp_path / "annotated", labelled, outcomes("pretrained"), configuration)
    with pytest.raises(GroundingError, match="does not use a predicted scene"):
        gi.convert_run(annotated, world.references_path, world.scenes_dir, "typed", allow_stub_scenes=True)
    for name, rows in (("odd-provenance", [dict(o, provenance=[1]) for o in outcomes("pretrained")]),
                       ("odd-adapter", outcomes("fine_tuned", {"path": "somewhere"})),
                       ("odd-condition", outcomes("pretrained", extra={"requested_condition": ["pretrained"]}))):
        folder = make_run(tmp_path / name, requests, rows, configuration)
        with pytest.raises(GroundingError):
            gi.convert_run(folder, world.references_path, world.scenes_dir, "typed", allow_stub_scenes=True)
    no_scene = make_run(tmp_path / "no-scene", [{k: v for k, v in r.items() if k != "scene"} for r in requests],
                        outcomes("pretrained"), configuration)
    with pytest.raises(GroundingError, match="does not use a predicted scene"):
        gi.convert_run(no_scene, world.references_path, world.scenes_dir, "typed", allow_stub_scenes=True)


def task5_case(world, request, target, fields, folder, mask=None, mask_mode="L"):
    data = pytest.importorskip("trackrag.dropoff.data")
    image = Path(request["scene"]["frame"]["image_path"])
    image_sha = sha(image.read_bytes())
    source = {"dataset": "EMT local", "image_sha256": image_sha}
    if mask is not None:
        path = save_mask(folder / "masks" / f"{request['request_id']}.png", mask, mask_mode)
        source.update(mask_path=f"masks/{path.name}", mask_sha256=sha(path.read_bytes()))
    review = {"status": "human_reviewed", "reviewers": ["reviewer-1", "reviewer-2"],
              "reviewed_at": "2026-11-21T10:00:00+00:00", "image_sha256": image_sha,
              "input_sha256": data.inference_input_sha256(request, image_sha),
              "supervision_sha256": data.supervision_sha256(target, fields, request["scene"]["frame"])}
    return {"schema_version": 1, "case_id": request["request_id"], "scene_group": world.references[
        request["request_id"]]["group_id"], "partition": "test", "source": source, "request": request,
        "target": target, "supervised_fields": fields, "review": review}


FULL = ["status", "destination_point_xy", "destination_box_xyxy", "clarification_question"]


def task5_cases(world, folder, requests=None):
    by_id = {r["request_id"]: r for r in (requests or task5_requests(world))}
    folder.mkdir(parents=True, exist_ok=True)
    clear = {"status": "proposed", "destination_point_xy": [11.5, 25.0], "destination_box_xyxy": [10.0, 10.0, 30.0, 40.0],
             "clarification_question": None, "anchor_boxes_xyxy": [CAR, BUS]}
    ask = {"status": "needs_clarification", "destination_point_xy": None, "destination_box_xyxy": None,
           "clarification_question": "Before or after the bus?"}
    none = {"status": "insufficient_visual_evidence", "destination_point_xy": None, "destination_box_xyxy": None,
            "clarification_question": None}
    return [task5_case(world, by_id["C1"], clear, FULL + ["anchor_boxes_xyxy"], folder, mask=c_mask()),
            task5_case(world, by_id["A1"], ask, FULL, folder),
            task5_case(world, by_id["M1"], none, FULL, folder)]


def test_task2_cases_in_task5_schema_convert_to_references(world, tmp_path):
    cases_dir = tmp_path / "cases"
    cases = task5_cases(world, cases_dir)
    cases_path = write_jsonl(cases_dir / "test-cases.jsonl", cases)
    intended = save_mask(cases_dir / "intended" / "A1.png", intended_mask())
    intended_path = write_jsonl(cases_dir / "intended.jsonl", [
        {"case_id": "A1", "category": "ambiguous", "intended_mask": "intended/A1.png",
         "intended_mask_sha256": sha(intended.read_bytes())}])
    record = gi.convert_cases(cases_path, world.scenes_dir, tmp_path / "refs", intended_path=intended_path,
                              allow_stub_scenes=True)
    assert record["categories"] == {"clear": 1, "ambiguous": 1, "missing_reference": 1}
    references, _ = g.load_references(tmp_path / "refs" / "references.jsonl", world.scenes)
    assert references["C1"]["instruction_sha256"] == world.references["C1"]["instruction_sha256"]
    assert references["C1"]["scene_sha256"] == world.references["C1"]["scene_sha256"]
    assert [a["role"] for a in references["C1"]["anchors"]] == ["anchor1", "anchor2"]
    assert references["A1"]["intended_mask"].sum() == intended_mask().sum()
    assert references["M1"]["category"] == "missing_reference"


def test_task2_case_conversion_refusals_leave_nothing_behind(world, tmp_path):
    cases_dir = tmp_path / "cases"
    cases = task5_cases(world, cases_dir)
    cases_path = write_jsonl(cases_dir / "test-cases.jsonl", cases)
    with pytest.raises(GroundingError, match="must be listed in the intended file"):
        gi.convert_cases(cases_path, world.scenes_dir, tmp_path / "a", allow_stub_scenes=True)
    assert not (tmp_path / "a").exists()
    destination_only = task5_case(world, cases[0]["request"], {"destination_point_xy": [11.5, 25.0],
                                                               "destination_box_xyxy": [10.0, 10.0, 30.0, 40.0]},
                                  ["destination_point_xy", "destination_box_xyxy"], cases_dir, mask=c_mask())
    with pytest.raises(GroundingError, match="full interpretation target"):
        gi.convert_cases(write_jsonl(cases_dir / "dest.jsonl", [destination_only]), world.scenes_dir, tmp_path / "b",
                         allow_stub_scenes=True)
    car = tracks(world, "C1", CAR)[0]
    other = task5_cases(world, tmp_path / "other", requests=task5_requests(world, drop_track=car))
    with pytest.raises(GroundingError, match="no matching Task 1 scene"):
        gi.convert_cases(write_jsonl(tmp_path / "other" / "c.jsonl", other[2:]), world.scenes_dir, tmp_path / "c",
                         allow_stub_scenes=True)
    train = dict(cases[2], partition="train")
    with pytest.raises(GroundingError, match="case loader refused"):
        gi.convert_cases(write_jsonl(cases_dir / "train.jsonl", [train]), world.scenes_dir, tmp_path / "d",
                         allow_stub_scenes=True)
    (cases_dir / "deep.jsonl").write_text("[" * 100000 + "\n", encoding="utf-8")
    with pytest.raises(GroundingError, match="case loader refused"):
        gi.convert_cases(cases_dir / "deep.jsonl", world.scenes_dir, tmp_path / "e", allow_stub_scenes=True)
    assert not any((tmp_path / name).exists() for name in "abcde")


def test_task5_cases_with_a_colour_mask_at_another_size(world, tmp_path):
    cases_dir = tmp_path / "cases"
    request = {r["request_id"]: r for r in task5_requests(world)}["C1"]
    target = {"status": "proposed", "destination_point_xy": [11.5, 25.0],
              "destination_box_xyxy": [10.0, 10.0, 30.0, 40.0], "clarification_question": None}
    double = np.kron(c_mask(), np.ones((2, 2))).astype(bool)
    case = task5_case(world, request, target, FULL, cases_dir, mask=double, mask_mode="RGB")
    gi.convert_cases(write_jsonl(cases_dir / "c.jsonl", [case]), world.scenes_dir, tmp_path / "refs",
                     allow_stub_scenes=True)
    references, _ = g.load_references(tmp_path / "refs" / "references.jsonl", world.scenes)
    reference = references["C1"]
    assert reference["mask"].shape == (2 * H, 2 * W)
    case = g.score_output(output(world, "C1", GOOD["C1"]), reference, world.scenes[reference["scene_sha256"]], {},
                          PROTOCOL, "direct", reference["instruction_sha256"])
    assert case["initial_point_in_mask"] and case["initial_box_iou"] == 1.0
