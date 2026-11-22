from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from tools import passenger_reviewed_geometry as diagnostic


def reference(status="proposed", width=100, height=50):
    return SimpleNamespace(
        request={"scene": {"frame": {"width": width, "height": height}}},
        target={"status": status, "destination_point_xy": [30, 15], "destination_box_xyxy": [20, 10, 40, 20]},
    )


def point(x, y):
    return {"ok": True, "result": {"status": "proposed", "destination_point_xy": [x, y],
                                     "destination_box_xyxy": [0, 0, 50, 25]}}


def test_inclusive_axes_and_missing_requests_have_distinct_denominators():
    refs = {key: reference() for key in ("edge", "above", "right_below", "failed", "question")}
    refs["other_reference"] = reference("needs_clarification")
    outcomes = {"edge": point(20, 20), "above": point(30, 5), "right_below": point(50, 25),
                "failed": {"ok": False}, "question": {"ok": True, "result": {"status": "needs_clarification"}},
                "other_reference": point(30, 15)}
    result = diagnostic.describe(refs, outcomes)
    assert (result["reviewed_location_cases"], result["model_returned_points"], result["missing_points"]) == (5, 3, 2)
    assert result["returned_points_horizontal"] == {"left": 0, "within": 2, "right": 1}
    assert result["returned_points_vertical"] == {"above": 1, "within": 1, "below": 1}
    assert result["returned_points_within_both_axes"] == 1
    assert result["median_signed_box_fraction_of_frame_all_reviewed"] == pytest.approx(0.04)
    assert result["median_predicted_box_fraction_of_frame_on_returned_points"] == pytest.approx(0.25)
    assert result["median_signed_horizontal_error_in_frame_widths_on_returned_points"] == 0
    assert result["median_signed_vertical_error_in_frame_heights_on_returned_points"] == pytest.approx(0.1)


def test_missing_points_have_no_conditional_medians_or_invented_locations():
    result = diagnostic.describe({"a": reference()}, {"a": {"ok": False}})
    assert result["reviewed_location_cases"] == result["missing_points"] == 1
    assert result["distinct_returned_normalized_points_six_decimals"] == result["returned_points_within_both_axes"] == 0
    assert result["median_predicted_box_fraction_of_frame_on_returned_points"] is None
    assert result["median_signed_box_fraction_of_frame_on_returned_points"] is None


def test_report_records_the_bytes_verified_before_a_source_is_replaced(tmp_path, monkeypatch):
    metrics = {"points_in_signed_box": 0, "returned_proposals_on_reference_proposals": 0,
               "native_file_sha256": {"model-outcomes.jsonl": sha256(b"").hexdigest()}}
    published = {"preanalysis": {"predeclared_utc": "2026-11-22T08:47:34+00:00"},
                 "models": {role: metrics for role in ("pretrained", "trained")}}
    primary = tmp_path / "primary.json"
    original = json.dumps(published).encode()
    primary.write_bytes(original)
    for role in published["models"]:
        folder = tmp_path / role
        folder.mkdir()
        (folder / "model-outcomes.jsonl").write_bytes(b"")

    def verify(*_args):
        primary.write_bytes(b"{}")
        return published

    monkeypatch.setattr(diagnostic, "score", verify)
    monkeypatch.setattr(diagnostic, "reviewed_bank", lambda *_args: ({}, {}))
    result = diagnostic.run(tmp_path / "requests", tmp_path, primary, tmp_path / "eligibility")
    assert result["primary_sha256"] == sha256(original).hexdigest()
