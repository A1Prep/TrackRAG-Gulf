"""Software-only checks for bounded method trials; no live model is substituted as evidence."""

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import urllib.error

import pytest
from PIL import Image

from tools import dropoff_method_trial as trial


def request(image_path="input.png", identity="trial-1", recording="video_054604"):
    return {
        "request_id": identity,
        "text": "Propose ground beside the white car",
        "input_mode": "typed",
        "dialogue": [],
        "scene": {
            "frame": {"recording_id": recording, "frame_id": 10, "timestamp_s": 1.0,
                      "width": 200, "height": 100, "image_path": str(image_path)},
            "source": "predicted", "tracks": [],
        },
    }


def candidate(identity="R1"):
    return {"candidate_id": identity, "point_original_px": [80, 70],
            "bbox_original_px": [60, 60, 100, 80]}


def unresolved(status="insufficient_visual_evidence", question=None):
    return {"status": status, "destination_point_xy": None,
            "destination_box_xyxy": None, "clarification_question": question}


def args_for(tmp_path, rows, *, method="direct", candidates=None):
    source = tmp_path / "requests.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return Namespace(requests=str(source), out=str(tmp_path / "out"), provider="gemini",
                     method=method, candidates=str(candidates) if candidates else None,
                     model="software-fixture", config="not-used.json", max_calls=16,
                     wall_seconds=100)


def image_file(tmp_path):
    path = tmp_path / "input.png"
    Image.new("RGB", (200, 100), "gray").save(path)
    return path


def fake_privacy(_request, output):
    # This fixture verifies routing only; it is not a privacy or real-image check.
    Image.new("RGB", (100, 50), "black").save(output)
    return {"sha256": trial.digest(Path(output).read_bytes()), "fixture": True}


def test_selected_candidate_keeps_original_pixels_without_rescaling_or_mutation():
    options = [candidate()]
    untouched = deepcopy(options)
    result = trial.decode_selection({"status": "proposed", "candidate_id": "R1",
                                     "clarification_question": None}, options,
                                    request()["scene"]["frame"])
    assert result["destination_point_xy"] == [80.0, 70.0]
    assert result["destination_box_xyxy"] == [60.0, 60.0, 100.0, 80.0]
    result["destination_point_xy"][0] = 1
    assert options == untouched


def test_direct_xyxy_normalization_matches_selected_original_pixel_geometry():
    text = json.dumps({"status": "proposed", "destination_point_xy": [400, 700],
                       "destination_box_xyxy": [300, 600, 500, 800],
                       "clarification_question": None})
    direct = trial.parse_model_output(text, 200, 100)
    selected = trial.decode_selection({"status": "proposed", "candidate_id": "R1",
                                       "clarification_question": None}, [candidate()],
                                      request()["scene"]["frame"])
    assert direct == selected


@pytest.mark.parametrize("fence", [None, "```", "```json"])
def test_selection_parser_accepts_one_complete_json_object_or_fence(fence):
    value = {"status": "insufficient_visual_evidence", "candidate_id": None,
             "clarification_question": None}
    text = json.dumps(value)
    if fence:
        text = fence + "\n" + text + "\n```"
    assert trial.parse_selection("  " + text + "  ") == value


@pytest.mark.parametrize("text", [
    '{"status":"proposed","status":"needs_clarification","candidate_id":null,"clarification_question":"Which car?"}',
    '{"status":"proposed","candidate_id":"R1","candidate_id":"R2","clarification_question":null}',
    '{"status":"proposed","candidate_id":NaN,"clarification_question":null}',
    '{"status":"proposed","candidate_id":Infinity,"clarification_question":null}',
    '{"status":"proposed","candidate_id":-Infinity,"clarification_question":null}',
    'Explanation: {"status":"insufficient_visual_evidence","candidate_id":null,"clarification_question":null}',
    '```json\n{"status":"insufficient_visual_evidence","candidate_id":null,"clarification_question":null}\n```\nextra',
    '```JSON\n{}\n```',
    '```json5\n{}\n```',
    '{} {}',
])
def test_selection_parser_rejects_duplicates_nonfinite_and_prose_without_repair(text):
    with pytest.raises(ValueError):
        trial.parse_selection(text)


@pytest.mark.parametrize("value", [
    {"status": "proposed", "candidate_id": "R99", "clarification_question": None},
    {"status": "proposed", "candidate_id": None, "clarification_question": None},
    {"status": "needs_clarification", "candidate_id": "R1", "clarification_question": "Which car?"},
    {"status": "needs_clarification", "candidate_id": None, "clarification_question": ""},
    {"status": "insufficient_visual_evidence", "candidate_id": None, "clarification_question": "Which car?"},
    {"status": "invented", "candidate_id": None, "clarification_question": None},
    {"status": "proposed", "candidate_id": "R1", "clarification_question": None, "extra": True},
])
def test_selection_does_not_repair_unknown_ids_or_mixed_statuses(value):
    with pytest.raises(ValueError):
        trial.decode_selection(value, [candidate()], request()["scene"]["frame"])


@pytest.mark.parametrize("status,question", [
    ("needs_clarification", "Which of the two cars do you mean?"),
    ("insufficient_visual_evidence", None),
])
def test_semantic_unresolved_remains_a_valid_nonproposal(status, question):
    value = {"status": status, "candidate_id": None, "clarification_question": question}
    assert trial.decode_selection(value, [candidate()], request()["scene"]["frame"]) == unresolved(status, question)


@pytest.mark.parametrize("options", [
    [candidate(), candidate()],
    [{**candidate(), "point_original_px": [200, 70]}],
    [{**candidate(), "point_original_px": [50, 70]}],
    [{**candidate(), "bbox_original_px": [60, 60, 201, 80]}],
])
def test_invalid_selected_geometry_is_rejected_instead_of_clipped(options):
    with pytest.raises(ValueError):
        trial.decode_selection({"status": "proposed", "candidate_id": "R1",
                                 "clarification_question": None}, options,
                                request()["scene"]["frame"])


def test_candidate_prompt_retains_instruction_and_context_without_gold_fields():
    row = request()
    row["scene"]["tracks"] = [{"recording_id": "video_054604", "frame_id": 9,
                                "track_id": 3, "class_name": "car",
                                "bbox_xyxy": [1, 2, 30, 40], "source": "predicted"}]
    messages = trial.candidate_messages(row, [candidate()], "marked.png")
    encoded = json.dumps(messages)
    assert messages[1]["content"][0]["image"] == "marked.png"
    assert messages[-1]["content"] == row["text"]
    assert "Observed scene context" in encoded and '"point_xy_0_1000"' in encoded.replace('\\"', '"')
    assert "400" in encoded and "700" in encoded
    assert row["scene"]["frame"]["image_path"] == "input.png"
    assert "candidate_id" in messages[0]["content"]


@pytest.mark.parametrize("corruption", ["target", "future", "annotated", "stale_dialogue"])
def test_gold_future_and_stale_context_never_reach_candidate_prompt(corruption):
    row = request()
    if corruption == "target":
        row["target"] = {"destination_point_xy": [80, 70]}
    elif corruption == "future":
        row["scene"]["tracks"] = [{"recording_id": "video_054604", "frame_id": 11,
                                    "track_id": 3, "class_name": "car",
                                    "bbox_xyxy": [1, 2, 30, 40], "source": "predicted"}]
    elif corruption == "annotated":
        row["scene"]["source"] = "annotated"
    else:
        row["dialogue"] = [{"role": "user", "text": "The left one", "recording_id": "video_054604", "frame_id": 9}]
    with pytest.raises(ValueError):
        trial.candidate_messages(row, [candidate()], "marked.png")


def test_hosted_parts_contains_only_selected_processed_image(tmp_path):
    image = image_file(tmp_path)
    messages = trial.candidate_messages(request("never-upload-original.jpg"), [candidate()], "marked.png")
    parts = trial.hosted_parts(messages, image)
    assert len(parts) == 2
    assert parts[1]["inlineData"]["mimeType"] == "image/png"
    assert "never-upload-original.jpg" not in json.dumps(parts)
    import base64
    assert base64.b64decode(parts[1]["inlineData"]["data"]) == image.read_bytes()


def test_hosted_trial_filters_public_images_and_keeps_unresolved_distinct_from_error(tmp_path, monkeypatch):
    image = image_file(tmp_path)
    rows = [request(image, "emt"), request(image, "public", "regseg-train-fixture")]
    args = args_for(tmp_path, rows)
    calls = []
    monkeypatch.setattr(trial, "privacy_image", fake_privacy)
    def call(messages, processed, schema, model):
        calls.append((messages, Path(processed).read_bytes(), schema, model))
        return json.dumps(unresolved()), {"fixture": True}
    monkeypatch.setattr(trial, "call_gemini", call)
    trial.run(args)
    outcomes = [json.loads(line) for line in (Path(args.out) / "outcomes.jsonl").read_text().splitlines()]
    assert len(calls) == len(outcomes) == 1
    assert outcomes[0]["request_id"] == "emt" and outcomes[0]["ok"] is True
    assert outcomes[0]["result"] == unresolved()
    assert calls[0][1] != image.read_bytes()
    run = json.loads((Path(args.out) / "run.json").read_text())
    assert run["quality_measured"] is False and run["errors"] == 0


def test_request_image_hash_mismatch_prevents_any_model_call(tmp_path, monkeypatch):
    image = image_file(tmp_path)
    row = request(image)
    row["scene"]["provenance"] = {"image_sha256": "0" * 64}
    args = args_for(tmp_path, [row])
    def never(*args, **kwargs):
        raise AssertionError("hash-mismatched image reached model")
    monkeypatch.setattr(trial, "call_gemini", never)
    trial.run(args)
    result = json.loads((Path(args.out) / "outcomes.jsonl").read_text())
    assert result["ok"] is False
    assert result["error"]["type"] == "ValueError"
    assert "result" not in result


def test_candidate_image_hash_mismatch_never_reaches_model(tmp_path, monkeypatch):
    image = image_file(tmp_path)
    manifest = tmp_path / "candidates.jsonl"
    manifest.write_text(json.dumps({"request_id": "trial-1", "image_sha256": "0" * 64,
                                    "candidates": [candidate()]}) + "\n", encoding="utf-8")
    args = args_for(tmp_path, [request(image)], method="candidates", candidates=manifest)
    monkeypatch.setattr(trial, "privacy_image", fake_privacy)
    def never(*args, **kwargs):
        raise AssertionError("mismatched candidate reached model")
    monkeypatch.setattr(trial, "call_gemini", never)
    trial.run(args)
    result = json.loads((Path(args.out) / "outcomes.jsonl").read_text())
    assert result["ok"] is False and "another image" in result["error"]["message"]


def test_candidate_preflight_binds_actual_original_image_dimensions(tmp_path):
    image = image_file(tmp_path)
    frame = request(image)["scene"]["frame"]
    valid = {**candidate("R01"), "semantic_class": "road"}
    manifest = {"image_sha256": trial.digest(image.read_bytes()), "candidates": [valid]}
    assert trial.validate_candidates(manifest, frame, manifest["image_sha256"]) == [valid]
    with pytest.raises(ValueError, match="dimensions"):
        trial.validate_candidates(manifest, {**frame, "width": 201}, manifest["image_sha256"])


@pytest.mark.parametrize("corruption", ["duplicate", "unknown_id", "nonroad", "outside", "empty", "too_many"])
def test_candidate_preflight_rejects_invalid_manifests_before_inference(tmp_path, corruption):
    image = image_file(tmp_path)
    frame = request(image)["scene"]["frame"]
    valid = {**candidate("R01"), "semantic_class": "road"}
    options = [valid]
    if corruption == "duplicate":
        options.append(deepcopy(valid))
    elif corruption == "unknown_id":
        options[0]["candidate_id"] = "private-labelled-answer"
    elif corruption == "nonroad":
        options[0]["semantic_class"] = "car"
    elif corruption == "outside":
        options[0]["point_original_px"] = [201, 70]
    elif corruption == "empty":
        options = []
    else:
        options = [valid] * 37
    image_hash = trial.digest(image.read_bytes())
    with pytest.raises(ValueError):
        trial.validate_candidates({"image_sha256": image_hash, "candidates": options}, frame, image_hash)


@pytest.mark.parametrize("recording", ["video_122233", "video_125233", "video_155425", "video_160325", "video_204647"])
def test_reserved_final_recordings_rejected_before_output_or_model(tmp_path, recording):
    args = args_for(tmp_path, [request("missing-final-image.png", recording=recording)])
    with pytest.raises(ValueError, match="Reserved final"):
        trial.run(args)
    assert not Path(args.out).exists()


def test_two_hosted_failures_stop_without_retry_or_false_semantic_output(tmp_path, monkeypatch):
    image = image_file(tmp_path)
    args = args_for(tmp_path, [request(image, str(i)) for i in range(3)])
    monkeypatch.setattr(trial, "privacy_image", fake_privacy)
    calls = []
    def fail(*args, **kwargs):
        calls.append(True)
        raise urllib.error.HTTPError("https://provider.invalid", 429, "quota", {}, None)
    monkeypatch.setattr(trial, "call_gemini", fail)
    with pytest.raises(RuntimeError, match="Two hosted failures"):
        trial.run(args)
    outcomes = [json.loads(line) for line in (Path(args.out) / "outcomes.jsonl").read_text().splitlines()]
    assert len(calls) == len(outcomes) == 2
    assert all(not row["ok"] and "result" not in row for row in outcomes)
    run = json.loads((Path(args.out) / "run.json").read_text())
    assert run["complete"] is False and run["errors"] == 2
