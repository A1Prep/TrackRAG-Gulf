import numpy as np
import pytest

from tools import passenger_public_dialogue_score as study


def proposal(point, box):
    return {"ok": True, "result": {"status": "proposed", "destination_point_xy": point,
                                    "destination_box_xyxy": box, "clarification_question": None}}


def test_mask_hit_and_mask_extent_overlap_measure_different_things():
    # A ring has a hole. Its tight rectangle contains the centre even though its mask does not.
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    centre = study.measure(proposal([15, 15], [0, 0, 30, 30]), mask, 30, 30)
    corner = study.measure(proposal([5, 5], [0, 0, 10, 10]), mask, 30, 30)
    assert centre == {"status": "proposed", "point_in_mask": False, "box_iou_with_mask_extent": 1.0}
    assert corner["point_in_mask"] and corner["box_iou_with_mask_extent"] == pytest.approx(1 / 9)


def test_missing_reference_and_failed_or_out_of_bounds_predictions_stay_in_denominator():
    mask = np.ones((2, 2), dtype=bool)
    timeout = {"ok": False, "error": {"code": "timeout"}}
    malformed = {"ok": False, "error": {"code": "invalid_model_output"}}
    values = [
        (study.measure(proposal([0, 0], [0, 0, 6, 6]), mask, 6, 6), True),
        (study.measure(proposal([0, 0], [0, 0, 6, 6]), None, 6, 6), False),
        (study.measure(proposal([6, 6], [0, 0, 6, 6]), mask, 6, 6), True),
        (study.measure(timeout, mask, 6, 6), True),
        (study.measure(malformed, mask, 6, 6), True),
    ]
    result = study.aggregate([{**row, "reference_available": real} for row, real in values])
    assert result["cases"] == 5 and result["usable_masks"] == 4
    assert result["point_in_mask_rate"] == result["mean_box_iou_with_mask_extent_missing_zero"] == 0.2
    assert result["point_in_mask_rate_available_masks"] == 0.25
    assert result["statuses"] == {"invalid_output": 2, "operational_error": 1, "proposed": 2}


def test_followup_uses_actual_prior_words_and_exact_full_instruction_on_same_frame():
    first = {"request_id": "a", "scene": {"frame": {"recording_id": "scene", "frame_id": 9}},
             "text": "Stop nearby.", "dialogue": []}
    raw = '{"status":"needs_clarification"}'
    second = study.restatement(first, "Stop on the left by the entrance.", raw)
    assert second["text"] == "Stop on the left by the entrance."
    assert [(turn["role"], turn["text"]) for turn in second["dialogue"]] == [
        ("user", first["text"]), ("assistant", raw)]
    assert all(turn["recording_id"] == "scene" and turn["frame_id"] == 9 for turn in second["dialogue"])
    assert first["dialogue"] == [] and second["scene"] == first["scene"]


def test_paired_comparison_preserves_direction_and_denominator():
    make = lambda flags: {key: {"point_in_mask": hit} for key, hit in enumerate(flags)}
    before, after = make([1, 1, 0, 0, 0]), make([1, 0, 1, 1, 0])
    report = study.paired(before, after)
    assert (report["both_hit"], report["both_missed"], report["lost"], report["gained"]) == (1, 1, 1, 2)
    assert report["point_hit_rate_difference"] == 0.2
    with pytest.raises(ValueError, match="same source scenes"):
        study.paired(before, make([1]))


def test_incomplete_model_run_never_opens_reference_masks(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("Read targets before complete sealed predictions")
    monkeypatch.setattr(study, "target_masks", forbidden)
    with pytest.raises(FileNotFoundError, match="sealed-outputs.json"):
        study.score(tmp_path, tmp_path / "masks.tar")


def test_stages_can_pause_between_jobs_but_cannot_overlap_or_change_order():
    def stamp(hour):
        return f"2026-11-22T{hour:02d}:00:00+00:00"
    def stage(start, end):
        return {"started_utc": stamp(start), "finished_utc": stamp(end), "run_sha256": "sample"}
    execution = {"started_utc": stamp(1), "completed_utc": stamp(12),
                 "stages": {"a": stage(1, 2), "b": stage(7, 8), "unused": {"run_sha256": None}}}
    study.check_stage_order(["a", "b", "unused"], execution)
    with pytest.raises(ValueError, match="overlapping|order"):
        study.check_stage_order(["b", "a", "unused"], execution)
    execution["stages"]["b"] = stage(1, 8)
    with pytest.raises(ValueError, match="overlapping|order"):
        study.check_stage_order(["a", "b", "unused"], execution)
