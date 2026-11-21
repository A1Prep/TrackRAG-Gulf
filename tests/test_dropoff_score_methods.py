"""Software scoring checks only; no model or human judgement is simulated."""
from copy import deepcopy
import json

from PIL import Image
import pytest

from tools import dropoff_score_methods as scoring


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def dump_lines(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if path.name == "outcomes.jsonl":
        dump(path.with_name("run.json"), {"outcomes_sha256": scoring.sha(path.read_bytes()), "complete": True})


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    cases_path = tmp_path / "development.jsonl"
    cases_path.write_text("software fixture source bytes\n", encoding="utf-8")
    cases = []
    options = []
    for i in range(2):
        image = tmp_path / f"image-{i}.png"
        Image.new("RGB", (4, 4)).save(image)
        mask_path = tmp_path / f"mask-{i}.png"
        mask = Image.new("L", (2, 2)); mask.putpixel((i, i), 255); mask.save(mask_path)
        frame = {"width": 4, "height": 4, "image_path": str(image), "recording_id": f"regseg-fixture-{i}", "frame_id": 0, "timestamp_s": 0.0}
        identity = f"public-{i}"
        cases.append({"case_id": identity, "request": {"request_id": identity, "input_mode": "typed", "dialogue": [],
                                                       "scene": {"frame": frame, "source": "predicted", "tracks": []}, "text": "Park beside the car"},
                      "source": {"mask_path": str(mask_path), "mask_sha256": scoring.sha(mask_path.read_bytes()),
                                 "image_sha256": scoring.sha(image.read_bytes())}})
        options.append({"request_id": identity, "image_sha256": scoring.sha(image.read_bytes()),
                        "candidates": [{"candidate_id": "R01", "point_original_px": [1.5, 1.5],
                                        "bbox_original_px": [1, 1, 2, 2], "semantic_class": "road"}]})
    monkeypatch.setattr(scoring, "load_cases", lambda *a, **k: deepcopy(cases))
    inputs = tmp_path / "input-snapshot.jsonl"
    dump_lines(inputs, [c["request"] for c in cases])
    protocol = tmp_path / "protocol.json"
    dump(protocol, {"public_case_ids": [c["case_id"] for c in cases],
                    "requests_sha256": scoring.sha(inputs.read_bytes()),
                    "development_source_sha256": scoring.sha(cases_path.read_bytes())})
    candidates = tmp_path / "candidates.jsonl"; dump_lines(candidates, options)
    (tmp_path / "candidates-snapshot.jsonl").write_bytes(candidates.read_bytes())
    outcomes = tmp_path / "outcomes.jsonl"; dump_lines(outcomes, [])
    return cases_path, protocol, candidates, outcomes, cases, options


def output(case, method="candidates", raw=None):
    if method == "candidates":
        text = json.dumps({"status": "proposed", "candidate_id": "R01", "clarification_question": None})
    else:
        text = json.dumps({"status": "proposed", "destination_point_xy": [375, 375],
                           "destination_box_xyxy": [250, 250, 500, 500], "clarification_question": None})
    messages = [{"role": "user", "content": case["request"]["text"]}]
    return {"request_id": case["case_id"], "method": method, "provider": "local", "ok": True,
            "original_image_sha256": case["source"]["image_sha256"], "raw_text": text if raw is None else raw,
            "request_sha256": scoring.sha(json.dumps(case["request"], sort_keys=True).encode()),
            "messages": messages, "messages_sha256": scoring.sha(json.dumps(messages, sort_keys=True).encode()),
            "result": {"status": "proposed", "destination_point_xy": [1.5, 1.5],
                       "destination_box_xyxy": [1, 1, 2, 2], "clarification_question": None}, "elapsed_s": 2.5}


def test_original_mask_floor_not_rounding():
    mask = Image.new("L", (2, 2)); mask.putpixel((0, 0), 255)
    assert scoring.point_hit([1.999, 1.999], mask, 4, 4)
    assert not scoring.point_hit([2, 2], mask, 4, 4)
    for point in ([4, 1], [-1, 0], [True, 1], [float("nan"), 0]):
        with pytest.raises(ValueError): scoring.point_hit(point, mask, 4, 4)


def test_denominators_ceiling_missing_and_conditional_choice(cohort):
    a, p, c, o, cases, _ = cohort
    rows = [output(cases[0]), output(cases[1]), output(cases[0], "direct")]
    dump_lines(o, rows)
    result = scoring.score(a, p, c, o)
    assert result["candidate_ceiling"]["representative_point_covered"] == 1
    candidate = result["methods"]["candidates"]
    assert candidate["requested"] == 2 and candidate["point_hits"] == 1
    assert candidate["point_agreement_all_requests"] == .5
    assert candidate["point_agreement_among_candidate_coverable"] == 1
    assert candidate["semantic_status_accuracy"] is None
    direct = result["methods"]["direct"]
    assert direct["requested"] == 2 and direct["missing_outputs"] == 1
    assert direct["point_agreement_all_requests"] == .5
    assert direct["latency_observations"] == 1
    assert result["source_bindings"]["outcomes"]["sha256"] == scoring.sha(o.read_bytes())


@pytest.mark.parametrize("raw", [
    '{"status":"proposed","candidate_id":"R99","candidate_id":"R01","clarification_question":null}',
    '{"status":"proposed","candidate_id":"R01","clarification_question":NaN}',
    '{"status":"proposed","candidate_id":"R99","clarification_question":null}',
])
def test_reparse_raw_selection_rejects_previously_accepted_invalid_output(cohort, raw):
    a, p, c, o, cases, _ = cohort
    dump_lines(o, [output(cases[0], raw=raw)])
    row = scoring.score(a, p, c, o)["methods"]["candidates"]["cases"][0]
    assert not row["operational_ok"] and not row["point_in_original_mask"]
    assert row["error_type"] == "scoring_revalidation_error"


def test_saved_result_must_equal_raw_candidate_lookup(cohort):
    a, p, c, o, cases, _ = cohort
    row = output(cases[0]); row["result"]["destination_point_xy"] = [1.6, 1.6]
    dump_lines(o, [row])
    result = scoring.score(a, p, c, o)
    assert result["methods"]["candidates"]["point_hits"] == 0
    assert result["methods"]["candidates"]["operational_errors"] == 2


def test_instruction_drift_is_a_failure_even_with_correct_geometry(cohort):
    a, p, c, o, cases, _ = cohort
    row = output(cases[0]); row["messages"][-1]["content"] = "Different instruction"
    dump_lines(o, [row])
    result = scoring.score(a, p, c, o)["methods"]["candidates"]
    assert result["point_hits"] == 0
    assert "instruction" in result["cases"][0]["revalidation_error"]


def test_unresolved_and_errors_keep_full_denominator(cohort):
    a, p, c, o, cases, _ = cohort
    row = output(cases[0])
    row["raw_text"] = json.dumps({"status": "needs_clarification", "candidate_id": None, "clarification_question": "Which car?"})
    row["result"] = {"status": "needs_clarification", "destination_point_xy": None,
                     "destination_box_xyxy": None, "clarification_question": "Which car?"}
    failed = {"provider": "local", "method": "candidates", "request_id": cases[1]["case_id"], "ok": False,
              "error": {"type": "TimeoutError"}, "elapsed_s": 3}
    dump_lines(o, [row, failed])
    result = scoring.score(a, p, c, o)["methods"]["candidates"]
    assert result["requested"] == 2 and result["point_hits"] == 0
    assert result["operational_errors"] == 1 and result["proposals"] == 0
    assert result["status_counts"] == {"needs_clarification": 1}
    assert result["total_latency_s_observed_only"] == 5.5


def test_hash_drift_and_duplicate_outcomes_refused(cohort):
    a, p, c, o, cases, _ = cohort
    dump_lines(o, [output(cases[0]), output(cases[0])])
    with pytest.raises(ValueError, match="Duplicate"): scoring.score(a, p, c, o)
    dump_lines(o, [])
    a.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Development source"): scoring.score(a, p, c, o)


def test_source_mask_hash_drift_refused(cohort):
    a, p, c, o, cases, _ = cohort
    mask = __import__("pathlib").Path(cases[0]["source"]["mask_path"])
    Image.new("L", (2, 2), 255).save(mask)
    with pytest.raises(ValueError, match="Original mask hash"): scoring.score(a, p, c, o)


def test_run_manifest_must_bind_exact_complete_output_bytes(cohort):
    a, p, c, o, cases, _ = cohort
    dump(o.with_name("run.json"), {"outcomes_sha256": "wrong", "complete": True})
    with pytest.raises(ValueError, match="Run manifest"): scoring.score(a, p, c, o)
    dump(o.with_name("run.json"), {"outcomes_sha256": scoring.sha(o.read_bytes()), "complete": True})
    assert scoring.score(a, p, c, o)["run_manifest"]["complete"] is True


def test_cli_immutable_output(cohort, tmp_path):
    a, p, c, o, _, _ = cohort
    out = tmp_path / "scores.json"
    argv = ["--cases", str(a), "--protocol", str(p), "--candidates", str(c), "--outcomes", str(o), "--out", str(out)]
    assert scoring.main(argv) == 0
    with pytest.raises(FileExistsError): scoring.main(argv)
