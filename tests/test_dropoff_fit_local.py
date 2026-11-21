"""CPU-light training/evaluation boundary checks using synthetic fixtures.

These tests never instantiate a real model, initialize CUDA, download data or
claim grounding quality. They protect cohorts, parsing and reference scoring.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import threading

from PIL import Image
import pytest

from tools.dropoff_fit_local import cohort, free_evaluate, parse_destination, spatial_score
from trackrag.dropoff.model import BackendResponse, GroundingConfig


def test_cohort_is_seeded_target_independent_and_input_order_independent():
    cases = [{"case_id": f"case-{index}", "scene_group": f"scene-{index // 2}",
              "target": {"answer": index}} for index in range(20)]
    selected = [row["case_id"] for row in cohort(cases, 8, seed=42)]
    changed = deepcopy(list(reversed(cases)))
    for row in changed:
        row["target"] = {"answer": "changed gold", "unrelated_score": 1e100}
    assert [row["case_id"] for row in cohort(changed, 8, seed=42)] == selected
    assert [row["case_id"] for row in cohort(cases, 8, seed=43)] != selected


def test_cohort_uses_distinct_scenes_before_repeated_views():
    cases = [{"case_id": f"case-{index}", "scene_group": group}
             for index, group in enumerate(["one", "one", "two", "two", "three"])]
    first = cohort(cases, 3)
    assert len(first) == 3 and len({row["scene_group"] for row in first}) == 3
    all_rows = cohort(cases, 100)
    assert len(all_rows) == 5
    assert len({row["scene_group"] for row in all_rows[:3]}) == 3
    assert {row["case_id"] for row in all_rows} == {row["case_id"] for row in cases}


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_cohort_rejects_invalid_counts(count):
    with pytest.raises(ValueError):
        cohort([{"case_id": "one", "scene_group": "scene"}], count)


@pytest.mark.parametrize("seed", [-1, True, 42.0])
def test_cohort_rejects_ambiguous_or_negative_seeds(seed):
    with pytest.raises(ValueError):
        cohort([{"case_id": "one", "scene_group": "scene"}], 1, seed=seed)


@pytest.mark.parametrize("limit,prefix", [(100, 192), (1201, 192), (625, 31), (625, 321)])
def test_expansion_rejects_unbounded_requests_before_network(tmp_path, monkeypatch, limit, prefix):
    from tools import dropoff_expand_regseg as expansion
    def forbidden(*args, **kwargs):
        raise AssertionError("Network source must not be created")
    monkeypatch.setattr(expansion.source, "BoundedSource", forbidden)
    with pytest.raises(ValueError):
        expansion.expand(tmp_path / "new-corpus", limit=limit, prefix_mib=prefix)


def test_expansion_never_overwrites_an_existing_corpus(tmp_path, monkeypatch):
    from tools import dropoff_expand_regseg as expansion
    destination = tmp_path / "existing-corpus"
    destination.mkdir()
    receipt = destination / "keep.json"
    receipt.write_text('{"existing":true}')
    with pytest.raises(FileExistsError):
        expansion.expand(destination)
    assert receipt.read_text() == '{"existing":true}'


@pytest.mark.parametrize("fence", [None, "```", "```json"])
def test_destination_normalization_preserves_original_image_coordinates(fence):
    raw = json.dumps({"destination_point_xy": [250, 750], "destination_box_xyxy": [200, 700, 400, 900]})
    if fence:
        raw = fence + "\n" + raw + "\n```"
    result = parse_destination(raw, 200, 100)
    assert result["destination_point_xy"] == pytest.approx([50, 75])
    assert result["destination_box_xyxy"] == pytest.approx([40, 70, 80, 90])


def test_full_interpretation_can_supply_a_destination_without_changing_geometry():
    raw = json.dumps({"status": "proposed", "clarification_question": None,
                      "destination_point_xy": [250, 750], "destination_box_xyxy": [200, 700, 400, 900]})
    result = parse_destination(raw, 200, 100)
    assert set(result) == {"destination_point_xy", "destination_box_xyxy"}
    assert result["destination_point_xy"] == [50, 75]


@pytest.mark.parametrize("raw", [
    '{"destination_point_xy":[20,20],"destination_point_xy":[30,30],"destination_box_xyxy":[0,0,100,100]}',
    '{"destination_point_xy":[NaN,20],"destination_box_xyxy":[0,0,100,100]}',
    '{"destination_point_xy":[Infinity,20],"destination_box_xyxy":[0,0,100,100]}',
    '{"destination_point_xy":[20,20],"destination_box_xyxy":[0,0,NaN,100]}',
    '{"destination_point_xy":[1000,20],"destination_box_xyxy":[0,0,1000,100]}',
    '{"destination_point_xy":[20,-1],"destination_box_xyxy":[0,0,100,100]}',
    '{"destination_point_xy":[true,20],"destination_box_xyxy":[0,0,100,100]}',
    '{"destination_point_xy":[20,20],"destination_box_xyxy":[0,0,1001,100]}',
    '{"destination_point_xy":[20,20],"destination_box_xyxy":[0,0,0,100]}',
    '{"destination_point_xy":[500,500],"destination_box_xyxy":[0,0,100,100]}',
    '{"destination_point_xy":[20,20]}',
    '[20,20]',
    '```python\n{"destination_point_xy":[20,20],"destination_box_xyxy":[0,0,100,100]}\n```',
])
def test_invalid_destination_json_is_rejected_without_clipping_or_repair(raw):
    with pytest.raises((ValueError, TypeError)):
        parse_destination(raw, 200, 100)


def test_public_region_scoring_does_not_invent_an_unresolved_destination():
    raw = json.dumps({"status": "needs_clarification", "destination_point_xy": None,
        "destination_box_xyxy": None, "clarification_question": "Which entrance?"})
    with pytest.raises(ValueError, match="No destination proposal"):
        parse_destination(raw, 200, 100)


def _case(tmp_path, index=1):
    image_path = tmp_path / f"frame-{index}.png"
    mask_path = tmp_path / f"mask-{index}.png"
    Image.new("RGB", (200, 100), (index, 20, 30)).save(image_path)
    mask = Image.new("L", (4, 2), 0)
    mask.putpixel((2, 1), 255)
    mask.save(mask_path)
    return {
        "case_id": f"fixture-{index}", "scene_group": f"fixture-scene-{index}",
        "partition": "development",
        "request": {"request_id": f"fixture-{index}", "text": "Stop near the visible entrance",
            "input_mode": "typed", "dialogue": [], "scene": {"source": "predicted", "tracks": [],
              "frame": {"recording_id": f"fixture-recording-{index}", "frame_id": 3,
                "timestamp_s": 0.3, "width": 200, "height": 100, "image_path": str(image_path)}}},
        "source": {"image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                   "mask_path": str(mask_path), "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest()},
        "target": {"destination_point_xy": [125, 75], "destination_box_xyxy": [100, 50, 150, 100]},
        "supervised_fields": ["destination_point_xy", "destination_box_xyxy"],
    }


def test_native_mask_scaling_iou_and_normalized_error(tmp_path):
    case = _case(tmp_path)
    perfect = spatial_score({"destination_point_xy": [125, 75], "destination_box_xyxy": [100, 50, 150, 100]}, case)
    assert perfect == {"point_in_mask": True, "coarse_box_iou": 1, "normalized_point_error": 0}
    # A 4x2 mask pixel occupies a 50x50 region in the 200x100 source image.
    boundary = spatial_score({"destination_point_xy": [100, 50], "destination_box_xyxy": [100, 50, 150, 100]}, case)
    assert boundary["point_in_mask"] is True
    assert boundary["normalized_point_error"] == pytest.approx(math.hypot(25 / 200, 25 / 100))
    outside = spatial_score({"destination_point_xy": [80, 75], "destination_box_xyxy": [60, 50, 160, 100]}, case)
    assert outside["point_in_mask"] is False
    assert outside["coarse_box_iou"] == pytest.approx(0.5)
    assert outside["normalized_point_error"] == pytest.approx(45 / 200)


def test_mask_reference_mutation_is_detected_before_scoring(tmp_path):
    case = _case(tmp_path)
    Image.new("L", (4, 2), 255).save(case["source"]["mask_path"])
    with pytest.raises(ValueError, match="mask bytes changed"):
        spatial_score(case["target"], case)


def test_public_reload_probe_is_portable_and_contains_no_supervision(tmp_path):
    from tools.dropoff_fit_local import prepare_public_probe
    from tools.dropoff_predict import read_requests
    case = _case(tmp_path)
    case.update(partition="train", review={"status": "public_annotation"})
    case["source"]["dataset"] = "Talk2Car-RegSeg"
    case["_training_image_sha256"] = case["source"]["image_sha256"]
    output = tmp_path / "probe"
    probe = prepare_public_probe(case, output)
    rows = read_requests(output / "requests.jsonl")
    image = Path(rows[0]["scene"]["frame"]["image_path"])
    assert image.parent == output.resolve()
    assert image.read_bytes() == Path(case["request"]["scene"]["frame"]["image_path"]).read_bytes()
    assert probe["image_sha256"] == case["source"]["image_sha256"]
    assert "target" not in rows[0] and "review" not in rows[0]
    assert probe["partition"] == "train"
    with pytest.raises(FileExistsError):
        prepare_public_probe(case, output)


def test_public_reload_probe_rejects_local_images_before_copying(tmp_path):
    from tools.dropoff_fit_local import prepare_public_probe
    case = _case(tmp_path)
    case.update(partition="train", review={"status": "human_reviewed"})
    case["source"]["dataset"] = "EMT"
    output = tmp_path / "probe"
    with pytest.raises(ValueError, match="public training"):
        prepare_public_probe(case, output)
    assert not output.exists()


class FakeRuntime:
    def __init__(self, responses):
        self.responses = list(responses)
        self.config = GroundingConfig()
        self.execution_lock = threading.RLock()
        self.model = self
        self.disabled = False
        self.seen = []

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False

    def generate(self, messages, *, config):
        assert self.disabled is True
        image = messages[1]["content"][0]["image"]
        assert isinstance(image, Image.Image) and image.size == (200, 100)
        text = " ".join(message["content"] if isinstance(message.get("content"), str) else
                        " ".join(part.get("text", "") for part in message["content"])
                        for message in messages)
        assert "mask_path" not in text and "mask_sha256" not in text
        assert "625.0" not in text and "750.0" not in text
        self.seen.append(messages[-1]["content"])
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return BackendResponse(response, {"backend_kind": "synthetic_test_fixture"})


def test_generation_failures_stay_in_cohort_denominator_and_raw_records(tmp_path):
    cases = [_case(tmp_path, index) for index in range(1, 5)]
    # Fourth input fails byte binding before inference; it still remains a row.
    cases[3]["source"]["image_sha256"] = "0" * 64
    valid = json.dumps({"destination_point_xy": [625, 750], "destination_box_xyxy": [500, 500, 750, 1000]})
    runtime = FakeRuntime([valid, TimeoutError("synthetic timeout"), "{invalid json"])
    output = tmp_path / "results.jsonl"
    summary = free_evaluate(runtime, cases, output, disabled=True)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["requested_cases"] == len(rows) == 4
    assert summary["valid_geometry"] == 1
    assert summary["coverage"] == summary["point_in_mask_all_requests"] == pytest.approx(0.25)
    assert summary["coarse_box_iou_all_requests"] == pytest.approx(0.25)
    assert summary["mean_point_error_valid_only"] == 0
    assert summary["failures_in_denominator"] is True
    assert summary["gold_fields_sent_to_model"] is False
    assert summary["case_ids"] == [case["case_id"] for case in cases]
    assert summary["results_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert [row["ok"] for row in rows] == [True, False, False, False]
    assert rows[1]["error"]["type"] == "TimeoutError"
    assert rows[2]["raw_text"] == "{invalid json"
    assert "image changed" in rows[3]["error"]["message"]
    assert len(runtime.seen) == 3 and runtime.disabled is False


def test_scoring_reference_failure_is_retained_as_a_failed_row(tmp_path):
    case = _case(tmp_path)
    case["source"]["mask_sha256"] = "0" * 64
    valid = json.dumps({"destination_point_xy": [625, 750], "destination_box_xyxy": [500, 500, 750, 1000]})
    output = tmp_path / "results.jsonl"
    summary = free_evaluate(FakeRuntime([valid]), [case], output, disabled=True)
    row = json.loads(output.read_text())
    assert summary["requested_cases"] == 1 and summary["valid_geometry"] == 0
    assert row["ok"] is False and "mask bytes changed" in row["error"]["message"]
    assert row["raw_text"] == valid
