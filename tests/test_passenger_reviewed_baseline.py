"""Throwaway signed examples and synthetic native envelopes for the alternate study reader."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json

from PIL import Image
import pytest

from tools import passenger_reviewed_baseline as baseline
from trackrag.dropoff.formatting import build_messages, parse_model_output


def answer(status, point=None, box=None, question=None):
    return {"status": status, "destination_point_xy": point, "destination_box_xyxy": box,
            "clarification_question": question}


def request(path, key="fixture"):
    return {"request_id": key, "text": "Stop beside the test marker.", "input_mode": "typed", "dialogue": [],
            "scene": {"source": "predicted", "tracks": [], "provenance": {"mode": "image_only_local_annotation", "detector_run": False},
                      "frame": {"image_path": str(path), "recording_id": "unit-fixture", "frame_id": 1,
                                "timestamp_s": 0, "width": 80, "height": 60}}}


def signed(tmp_path):
    path = tmp_path / "source/images/test/unit-fixture/00001.jpg"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (80, 60)).save(path)
    q = request(path)
    t = answer("proposed", [40, 30], [30, 20, 50, 40])
    fields = list(t)
    image = baseline.file_sha256(path)
    ground = baseline.supervision_sha256(t, fields, q["scene"]["frame"])
    inp = baseline.inference_input_sha256(q, image)
    dates = ("2026-11-01T00:00:00+00:00", "2026-11-02T00:00:00+00:00")
    reviewers = ["fixture-one", "fixture-two"]
    common = {"case_id": "fixture", "partition": "test"}
    case = {**common, "request": deepcopy(q), "target": t, "supervised_fields": fields,
            "source": {"image_sha256": image, "recording_id": "unit-fixture", "frame_id": 1},
            "review": {"input_sha256": inp, "supervision_sha256": ground, "image_sha256": image,
                       "status": "human_reviewed", "new_human_review": True, "independent_reviewers_required": 2,
                       "first_reviewed_at": dates[0], "second_reviewed_at": dates[1], "reviewed_at": dates[1],
                       "reviewers": reviewers}}
    binding = {"case_id": "fixture", "recording_id": "unit-fixture", "frame_id": 1, "image_sha256": image, "reviewed_input_sha256": inp}
    forms = [{**common, "recording_id": "unit-fixture", "frame_id": "1", "decision": "accept",
              "reviewer_id": actor, "reviewed_at": stamp, "image_sha256": image, "input_sha256": inp,
              "supervision_sha256": ground, "instruction": q["text"], "status": t["status"],
              "destination_point_xy": json.dumps(t["destination_point_xy"]), "destination_box_xyxy": json.dumps(t["destination_box_xyxy"]),
              "clarification_question": ""} for actor, stamp in zip(reviewers, dates)]
    return case, q, binding, forms


def test_full_review_binds_two_people_original_pixels_and_all_fields(tmp_path):
    case, q, binding, forms = signed(tmp_path)
    ref = baseline.confirmed_reference(case, q, binding, forms)
    assert ref.target == case["target"]
    assert ref.image_sha256 == binding["image_sha256"]
    with pytest.raises(ValueError, match="postdates"):
        baseline.confirmed_reference(case, q, binding, forms, as_of=baseline.moment("2026-11-01T20:00:00Z"))
    Image.new("RGB", (80, 60), "red").save(q["scene"]["frame"]["image_path"])
    with pytest.raises(ValueError, match="photo hash"):
        baseline.confirmed_reference(case, q, binding, forms)


@pytest.mark.parametrize("changed", ["same_actor", "unsigned", "wording", "point", "added_provenance", "different_input", "extra_target"])
def test_mismatches_are_rejected(tmp_path, changed):
    case, q, binding, forms = signed(tmp_path)
    if changed == "same_actor":
        forms[1]["reviewer_id"] = forms[0]["reviewer_id"]
    elif changed == "unsigned":
        forms[1]["reviewed_at"] = ""
    elif changed == "wording":
        forms[0]["instruction"] += " Edited."
    elif changed == "point":
        forms[0]["destination_point_xy"] = "[39,30]"
    elif changed == "added_provenance":
        q["scene"]["provenance"]["image_sha256"] = binding["image_sha256"]
    elif changed == "different_input":
        binding["reviewed_input_sha256"] = "f"*64
    else:
        case["supervised_fields"].append("anchor_boxes_xyxy")
    with pytest.raises(ValueError):
        baseline.confirmed_reference(case, q, binding, forms)


def test_all_denominators_and_literal_clarification_are_independent(tmp_path):
    q = request(tmp_path / "unused")
    clear = answer("proposed", [40, 30], [35, 20, 45, 40])
    edge = answer("proposed", [45, 30], [35, 20, 45, 40])
    vague = answer("needs_clarification", question="Which one?")
    missing = answer("insufficient_visual_evidence")
    refs = {str(i): baseline.Reference(q, t, "") for i, t in enumerate([clear, clear, clear, vague, vague, missing, missing])}
    results = [edge, None, missing, vague, answer("needs_clarification", question="which one?"), edge, missing]
    outcomes = {str(i): {"ok": True, "result": t} if t else {"ok": False, "error": {"code": "invalid_model_output"}}
                for i, t in enumerate(results)}
    data, matched = baseline.model_metrics(refs, outcomes)
    assert data["cases"] == 7 and data["status_correct"] == sum(matched.values()) == 4
    assert data["status_confusion_expected_then_returned"]["insufficient_visual_evidence"]["proposed"] == 1
    assert data["failed_requests_by_code"] == {"invalid_model_output": 1}
    assert (data["valid_semantic_outputs"], data["malformed_model_outputs"], data["operational_errors_excluding_malformed"]) == (6, 1, 0)
    assert data["reference_proposals"] == 3 and data["returned_proposals_on_reference_proposals"] == 1
    assert data["missing_proposals_or_failures_on_reference_proposals"] == 2
    assert data["points_in_signed_box"] == 1 and data["point_in_signed_box_rate_all_eligible"] == pytest.approx(1/3)
    assert data["signed_box_iou_at_least_half"] == 1 and data["signed_box_iou_mean_all_missing_zero"] == pytest.approx(1/3)
    assert data["point_error_image_diagonals_mean_returned"] == .05
    assert data["point_error_image_diagonals_mean_all_missing_cost_one"] == pytest.approx(2.05/3)
    assert data["clarification_question_literal_matches"] == 1 and data["reviewed_clarifications"] == 2
    del outcomes["0"]
    with pytest.raises(ValueError, match="including failures"):
        baseline.model_metrics(refs, outcomes)


def test_metrics_return_null_with_no_eligible_reference_or_point(tmp_path):
    ref = baseline.Reference(request(tmp_path / "unused"), answer("needs_clarification", question="Where?"), "")
    results = {"fixture": {"ok": False, "error": {"code": "private detail never published"}}}
    data, _ = baseline.model_metrics({"fixture": ref}, results)
    assert data["failed_requests_by_code"] == {"other": 1}
    assert data["point_error_image_diagonals_mean_returned"] is None
    assert data["point_error_image_diagonals_mean_all_missing_cost_one"] is None
    assert data["signed_box_iou_mean_all_missing_zero"] is None
    assert data["clarification_question_literal_match_rate"] == 0


def native_error(q, config, image, *, code="timeout"):
    p = {"model_id": config.model_id, "revision": config.revision, "injected": False,
         "backend_kind": "transformers_local_streamed", "requested_condition": "pretrained",
         "request_sha256": baseline.digest(q), "image": {"sha256": image}, "adapter": None,
         "generation": {"do_sample": False, **{k: getattr(config, k) for k in
                        ("max_new_tokens", "max_input_tokens", "min_pixels", "max_pixels", "max_time_s")}}}
    err = {"code": code, "message": "synthetic unit failure; no model was called"}
    out = {"request_id": q["request_id"], "ok": False, "error": err, "provenance": p}
    receipt = {"request_id": q["request_id"], "request": q, "status": "error", "error": err, "provenance": p}
    return out, receipt


def test_raw_failures_remain_and_injected_or_wrong_context_is_refused(tmp_path):
    case, q, bind, _ = signed(tmp_path)
    cfg = baseline.GroundingConfig(runtime="streamed")
    outcome, rec = native_error(q, cfg, bind["image_sha256"])
    baseline.verify_outcome(q, rec, outcome, "pretrained", cfg, {}, bind["image_sha256"])
    with pytest.raises(ValueError, match="reviewed original"):
        baseline.verify_outcome(q, rec, outcome, "pretrained", cfg, {}, "a"*64)
    rec["provenance"]["injected"] = True
    with pytest.raises(ValueError, match="model source"):
        baseline.verify_outcome(q, rec, outcome, "pretrained", cfg, {}, bind["image_sha256"])


def test_raw_success_must_reparse_and_have_real_worker_identity(tmp_path):
    _, q, b, _ = signed(tmp_path)
    cfg = baseline.GroundingConfig(runtime="streamed")
    o, r = native_error(q, cfg, b["image_sha256"])
    raw = json.dumps(answer("proposed", [500,500], [400,400,600,600]))
    messages = build_messages(q, observation_format=cfg.observation_format)
    p = o["provenance"]
    p["messages_sha256"] = baseline.digest(messages)
    p["backend_details"] = {"software_fixture": True}
    result = parse_model_output(raw, 80, 60)
    r.update(status="completed", raw_text=raw, messages=messages, result=result)
    o.update(ok=True, result=deepcopy(result))
    del r["error"], o["error"]
    with pytest.raises(ValueError, match="raw_native_backend_unverified"):
        baseline.verify_outcome(q, r, o, "pretrained", cfg, {}, b["image_sha256"])
    r["raw_text"] = json.dumps(answer("insufficient_visual_evidence"))
    with pytest.raises(ValueError, match="does not reproduce"):
        baseline.verify_outcome(q, r, o, "pretrained", cfg, {}, b["image_sha256"])


def test_declared_malformed_output_cannot_hide_a_valid_response(tmp_path):
    _, q, b, _ = signed(tmp_path)
    cfg = baseline.GroundingConfig(runtime="streamed")
    o, r = native_error(q, cfg, b["image_sha256"], code="invalid_model_output")
    r["raw_text"] = "{"
    baseline.verify_outcome(q, r, o, "pretrained", cfg, {}, b["image_sha256"])
    r["raw_text"] = json.dumps(answer("insufficient_visual_evidence"))
    with pytest.raises(ValueError, match="parsed as a valid"):
        baseline.verify_outcome(q, r, o, "pretrained", cfg, {}, b["image_sha256"])


def test_native_run_checks_snapshot_all_rows_and_native_file_hashes(tmp_path, monkeypatch):
    case, q, binding, forms = signed(tmp_path)
    ref = baseline.confirmed_reference(case, q, binding, forms)
    snapshots = (json.dumps(q) + "\n").encode()
    requested = sha256(snapshots).hexdigest()
    monkeypatch.setattr(baseline, "REQUEST_HASH", requested)
    cfg = baseline.GroundingConfig(runtime="streamed")
    o, r = native_error(q, cfg, ref.image_sha256)
    folder = tmp_path / "throwaway-run"; folder.mkdir()
    (folder / "input-snapshot.jsonl").write_bytes(snapshots)
    meta = {"configuration": asdict(cfg), "created_at": "2026-11-03T00:00:00+00:00",
            "requests_sha256": requested, "requested_rows": 1, "semantic_results": 0, "operational_errors": 1}
    artifacts = {"resolved-requests.jsonl": q, "receipts.jsonl": r, "model-outcomes.jsonl": o}
    for name, field in baseline.TASK5_RUN_FILES:
        content = (json.dumps(artifacts[name])+"\n").encode()
        (folder / name).write_bytes(content)
        meta[field] = sha256(content).hexdigest()
    (folder / "run.json").write_text(json.dumps(meta))
    launch = {"started_utc": "2026-11-02T00:00:00Z", "end_utc": "2026-11-03T00:10:00Z",
              "run_sha256": baseline.file_sha256(folder/"run.json")}
    records, proof = baseline.checked_run(folder, {"fixture": ref}, "pretrained", cfg, {}, "2026-11-01T00:00:00Z", launch)
    assert proof["native_run_sha256"] == baseline.file_sha256(folder/"run.json")
    assert records["fixture"]["error"]["code"] == "timeout"
    with pytest.raises(ValueError, match="timing"):
        baseline.checked_run(folder, {"fixture": ref}, "pretrained", cfg, {}, "2026-11-02T00:01:00Z", launch)
    with pytest.raises(ValueError, match="seal"):
        baseline.checked_run(folder, {"fixture": ref}, "pretrained", cfg, {}, "2026-11-01T00:00:00Z",
                             {**launch, "run_sha256": "0"*64})
    (folder / "receipts.jsonl").write_text("")
    with pytest.raises(ValueError, match="sealed digest"):
        baseline.checked_run(folder, {"fixture": ref}, "pretrained", cfg, {}, "2026-11-01T00:00:00Z", launch)
    meta["receipts_sha256"] = sha256(b"").hexdigest()
    (folder/"run.json").write_text(json.dumps(meta))
    launch["run_sha256"] = baseline.file_sha256(folder/"run.json")
    with pytest.raises(ValueError, match="reordered"):
        baseline.checked_run(folder, {"fixture": ref}, "pretrained", cfg, {}, "2026-11-01T00:00:00Z", launch)


def test_paired_report_only_exports_aggregates_and_requires_matched_seals(tmp_path, monkeypatch):
    private = "unit-test-PRIVATE-individual-id"
    q = request(tmp_path / "image", private)
    target = answer("needs_clarification", question="Specific private participant wording?")
    ref = baseline.Reference(q, target, "unit-test-PRIVATE-photo")
    frozen = "2026-11-01T00:00:00Z"
    record = {"kind": "unit-test-only", "predeclared_utc": frozen}
    seal = tmp_path / "eligibility.json"
    seal.write_text(json.dumps(record))
    roles = {m: tmp_path / m for m in ("pretrained", "trained")}
    fake = {m: {"run_sha256": m * 2} for m in roles}
    ext = {"eligibility_sha256": baseline.file_sha256(seal), "conditions": fake}
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps(ext))
    monkeypatch.setattr(baseline, "eligibility", lambda *args: (record, {private: ref}, roles, {}))

    def synthetic_run(folder, refs, role, config, registry, freeze, launch):
        assert folder == roles[role] and launch == fake[role] and freeze == frozen
        result = target if role == "trained" else answer("insufficient_visual_evidence")
        return {private: {"ok": True, "result": result}}, {"native_run_sha256": launch["run_sha256"]}

    monkeypatch.setattr(baseline, "checked_run", synthetic_run)
    data = baseline.score(tmp_path / "input", seal, execution, roles)
    assert data["paired_status_agreement"] == {"both_correct": 0, "pretrained_only": 0, "trained_only": 1, "neither": 0}
    assert data["official_four_condition_score"] is False
    published = json.dumps(data)
    assert private not in published and target["clarification_question"] not in published and ref.image_sha256 not in published
    ext["eligibility_sha256"] = "0"*64
    execution.write_text(json.dumps(ext))
    with pytest.raises(ValueError, match="exact preanalysis"):
        baseline.score(tmp_path / "input", seal, execution, roles)
