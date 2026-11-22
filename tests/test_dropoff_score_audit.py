"""Independent software-only mutation checks for saved trial evidence binding."""
from copy import deepcopy
import json

from PIL import Image
import pytest

from tools import dropoff_score_methods as scoring
from tools.dropoff_method_trial import candidate_messages


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    if path.name == "outcomes.jsonl":
        write_json(path.with_name("run.json"), {"outcomes_sha256": scoring.sha(path.read_bytes()), "complete": True})


@pytest.fixture
def audit_trial(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (40, 40), "gray").save(image)
    mask = tmp_path / "mask.png"
    Image.new("L", (40, 40), 255).save(mask)
    frame = {"recording_id": "software-scene", "frame_id": 0, "timestamp_s": 0.0,
             "width": 40, "height": 40, "image_path": str(image)}
    request = {"request_id": "software-public-1", "scene": {"frame": frame, "source": "predicted", "tracks": []},
               "text": "Drop me beside the white truck.", "input_mode": "typed", "dialogue": []}
    case = {"case_id": request["request_id"], "request": deepcopy(request),
            "source": {"image_sha256": scoring.sha(image.read_bytes()), "mask_path": str(mask),
                       "mask_sha256": scoring.sha(mask.read_bytes())}}
    cases_path = tmp_path / "development.jsonl"
    cases_path.write_bytes(b"software reference source fixture\n")
    monkeypatch.setattr(scoring, "load_cases", lambda *a, **k: [deepcopy(case)])
    candidate = {"candidate_id": "R01", "point_original_px": [10.5, 10.5],
                 "bbox_original_px": [8, 8, 13, 13], "semantic_class": "road"}
    candidate_row = {"request_id": request["request_id"], "image_sha256": case["source"]["image_sha256"],
                     "candidates": [candidate]}
    candidates_path = tmp_path / "candidates.jsonl"
    write_lines(candidates_path, [candidate_row])
    write_lines(tmp_path / "candidates-snapshot.jsonl", [candidate_row])
    input_snapshot = tmp_path / "input-snapshot.jsonl"
    write_lines(input_snapshot, [request])
    protocol_path = tmp_path / "protocol.json"
    protocol = {"public_case_ids": [case["case_id"]], "emt_case_ids": [],
                "development_source_sha256": scoring.sha(cases_path.read_bytes()),
                "requests_sha256": scoring.sha(input_snapshot.read_bytes())}
    write_json(protocol_path, protocol)
    messages = candidate_messages(request, [candidate], image)
    output = {"request_id": case["case_id"], "method": "candidates", "provider": "local", "ok": True,
              "original_image_sha256": case["source"]["image_sha256"],
              "request_sha256": scoring.sha(json.dumps(request, sort_keys=True).encode()),
              "messages": messages, "messages_sha256": scoring.sha(json.dumps(messages, sort_keys=True).encode()),
              "model_image_sha256": scoring.sha(image.read_bytes()),
              "raw_text": json.dumps({"status": "proposed", "candidate_id": "R01", "clarification_question": None}),
              "result": {"status": "proposed", "destination_point_xy": [10.5, 10.5],
                         "destination_box_xyxy": [8, 8, 13, 13], "clarification_question": None}, "elapsed_s": 1.0}
    outcomes_path = tmp_path / "outcomes.jsonl"
    write_lines(outcomes_path, [output])
    write_json(tmp_path / "settings.json", {"requests": str(input_snapshot), "candidates": str(candidates_path),
               "candidate_manifest_sha256": scoring.sha(candidates_path.read_bytes()),
               "source_file_sha256": scoring.sha(scoring.Path(scoring.__file__).with_name("dropoff_method_trial.py").read_bytes())})
    return {"args": (cases_path, protocol_path, candidates_path, outcomes_path), "request": request,
            "output": output, "protocol": protocol, "case": case, "candidate_row": candidate_row, "tmp": tmp_path}


def rejected_or_failed(trial):
    try:
        result = scoring.score(*trial["args"])
    except ValueError:
        return
    row = result["methods"]["candidates"]["cases"][0]
    assert not row["operational_ok"], "tampered evidence was accepted as a valid scored answer"
    assert row["point_in_original_mask"] is False


def test_audit_fixture_reaches_a_real_scored_success(audit_trial):
    result = scoring.score(*audit_trial["args"])
    assert result["methods"]["candidates"]["point_hits"] == 1


def test_wrong_instruction_request_digest_does_not_receive_point_credit(audit_trial):
    changed = deepcopy(audit_trial["request"])
    changed["text"] = "Drop me beside the black bus."
    output = deepcopy(audit_trial["output"])
    output["request_sha256"] = scoring.sha(json.dumps(changed, sort_keys=True).encode())
    write_lines(audit_trial["args"][3], [output])
    rejected_or_failed(audit_trial)


def test_changed_inference_snapshot_is_not_accepted_under_original_protocol(audit_trial):
    changed = deepcopy(audit_trial["request"])
    changed["text"] = "Drop me beside the black bus."
    write_lines(audit_trial["tmp"] / "input-snapshot.jsonl", [changed])
    rejected_or_failed(audit_trial)


def test_rebound_snapshot_still_must_match_source_instruction(audit_trial):
    changed = deepcopy(audit_trial["request"])
    changed["text"] = "Drop me beside the black bus."
    path = audit_trial["tmp"] / "input-snapshot.jsonl"
    write_lines(path, [changed])
    protocol = deepcopy(audit_trial["protocol"])
    protocol["requests_sha256"] = scoring.sha(path.read_bytes())
    write_json(audit_trial["args"][1], protocol)
    output = deepcopy(audit_trial["output"])
    output["request_sha256"] = scoring.sha(json.dumps(changed, sort_keys=True).encode())
    write_lines(audit_trial["args"][3], [output])
    rejected_or_failed(audit_trial)


def test_changed_candidate_snapshot_cannot_be_replaced_by_different_score_options(audit_trial):
    changed = deepcopy(audit_trial["candidate_row"])
    changed["candidates"][0]["point_original_px"] = [11.5, 10.5]
    write_lines(audit_trial["tmp"] / "candidates-snapshot.jsonl", [changed])
    rejected_or_failed(audit_trial)


def test_non_text_model_output_counts_as_failure_without_attribute_error(audit_trial):
    output = deepcopy(audit_trial["output"])
    output["raw_text"] = {"status": "proposed"}
    write_lines(audit_trial["args"][3], [output])
    result = scoring.score(*audit_trial["args"])
    row = result["methods"]["candidates"]["cases"][0]
    assert not row["operational_ok"] and not row["point_in_original_mask"]


def test_malformed_saved_error_stays_in_denominator(audit_trial):
    output = deepcopy(audit_trial["output"])
    output.update(ok=False, error="not a structured error")
    write_lines(audit_trial["args"][3], [output])
    result = scoring.score(*audit_trial["args"])
    summary = result["methods"]["candidates"]
    assert summary["requested"] == 1 and summary["point_hits"] == 0
    assert summary["operational_errors"] == 1
