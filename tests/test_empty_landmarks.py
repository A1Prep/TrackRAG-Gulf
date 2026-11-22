"""Prove what the empty-detector grammar can express without loading any model weights."""

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from trackrag.dropoff import landmarks, structured


def request(tracks):
    frame = {"recording_id": "test", "frame_id": 3, "timestamp_s": .2, "width": 200,
             "height": 100, "image_path": str(Path("original.png").resolve())}
    return {"request_id": "test-request", "scene": {"frame": frame, "source": "predicted", "tracks": tracks,
            "provenance": {"image_sha256": "a" * 64}}, "text": "By the entrance.",
            "dialogue": [{"role": "user", "text": "Near the station.", "recording_id": "test", "frame_id": 3}],
            "input_mode": "typed"}


def track(frame_id):
    return {"recording_id": "test", "frame_id": frame_id, "class_name": "Car", "track_id": 87,
            "source": "predicted", "bbox_xyxy": [10, 20, 60, 80]}


@pytest.mark.parametrize("tracks,expected", [([], structured.EMPTY_LANDMARK_CONTRACT),
    ([track(2)], structured.EMPTY_LANDMARK_CONTRACT), ([track(3)], structured.LANDMARK_CONTRACT)])
def test_choose_contract_only_from_current_candidates(monkeypatch, tracks, expected):
    row = request(tracks)
    messages, index = landmarks.landmark_messages(row)
    if expected == structured.EMPTY_LANDMARK_CONTRACT:
        assert index == {} and messages[0]["content"] == landmarks.EMPTY_GUIDE
        assert "LANDMARK" not in messages[0]["content"] and "L3" not in str(messages)
        text = "MISSING"
    else:
        assert set(index) == {"L1"}
        assert sha256(messages[0]["content"].encode()).hexdigest() == (
            "808aa9020d25c30c2a4683869d0b47ed021bde2580c4846be1347fd646ef6d52")
        assert messages[1]["content"][1]["text"] == "Current detector candidates:\nL1 Car 50 200 300 800"
        text = "LANDMARK L1"
    assert messages[-2:] == [{"role": "user", "content": "Near the station."},
                             {"role": "user", "content": "By the entrance."}]
    calls = []
    def generate(messages, *, config, contract):
        calls.append(contract)
        return SimpleNamespace(text=text, details={})
    resolver = object.__new__(landmarks.LandmarkResolver)
    resolver.config = SimpleNamespace(model_id="stub", revision="stub")
    resolver.backend = SimpleNamespace(generate=generate)
    resolver.adapter = None
    monkeypatch.setattr(landmarks, "verify_adapter", lambda config: None)
    record = resolver.resolve(row, object())
    assert record["contract"] == expected and calls == [expected]
    assert record["raw_text"] == text and "parse_error" not in record


@pytest.mark.parametrize("text,valid", [
    ("MISSING", True), ("ASK Which of the two entrances?", True), ("OBJECT 0 10 999 1000", True),
    ("LANDMARK L3", False), ("L3", False), ("OBJECT 10 20 30", False), ("OBJECT 0 10 30 1001", False),
    ("ASK", False), ("ASK Which entrance", False), ("ASK Which?\nMISSING", False), ("MISSING now", False)])
def test_actual_empty_grammar(text, valid):
    pytest.importorskip("lmformatenforcer")
    parser = structured.new_empty_landmark_parser()
    for character in text:
        if character not in parser.get_allowed_characters():
            accepted = False
            break
        parser = parser.add_character(character)
    else:
        accepted = parser.can_end()
    assert accepted is valid


def test_unlisted_geometry_semantics_and_malformed_outputs_are_independent_of_decoding():
    frame = request([])["scene"]["frame"]
    assert landmarks.parse_landmark("OBJECT 0 250 750 1000", {}, frame) == {
        "kind": "object", "boxes": [[0, 25, 150, 100]], "track_ids": []}
    assert landmarks.parse_landmark("ASK Which entrance?", {}, frame) == {"kind": "ask", "question": "Which entrance?"}
    assert landmarks.parse_landmark("MISSING", {}, frame) == {"kind": "missing"}
    for invalid in ("LANDMARK L3", "ASK Which entrance", "OBJECT 0 250 0 1000", "OBJECT 999 250 100 1000",
                    "OBJECT 0 250 750 1001"):
        with pytest.raises(ValueError):
            landmarks.parse_landmark(invalid, {}, frame)
