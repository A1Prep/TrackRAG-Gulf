from hashlib import sha256

import pytest

from tools import dropoff_egocentric_language_study as study
from tools.dropoff_egocentric_language_study import canonical, evaluate_case, registered_python_bytes


def test_registered_python_allows_only_checkout_line_endings(tmp_path):
    path = tmp_path / "policy.py"
    source = b"def decision():\n    return True\n"
    original_sha = sha256(source).hexdigest()
    for checkout in (source, source.replace(b"\n", b"\r\n")):
        path.write_bytes(checkout)
        assert registered_python_bytes(path, original_sha) == source
        assert registered_python_bytes(path, original_sha, git_blob=checkout) == source
    for edit in (source.replace(b"True", b"False"), source.replace(b"\n", b"\r"),
                 source[:-1], b"\xef\xbb\xbf" + source):
        path.write_bytes(edit)
        with pytest.raises(ValueError, match="source changed"):
            registered_python_bytes(path, original_sha)


def test_registered_evidence_files_still_require_exact_bytes(tmp_path, monkeypatch):
    raw = b'{\n  "frozen": true\n}\n'
    target = tmp_path / "protocol.json"
    target.write_bytes(raw)
    expected = sha256(raw).hexdigest()
    monkeypatch.setattr(study, "PROTOCOL", expected)
    monkeypatch.setattr(study, "PROTOCOL_FILE", target)
    assert study.load_protocol() == {"frozen": True}
    target.write_bytes(raw.replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="protocol changed"):
        study.load_protocol()
    for name in ("protocol.json", "predictions.jsonl", "photo.jpg", "model.safetensors"):
        with pytest.raises(ValueError, match="Only Python"):
            registered_python_bytes(tmp_path / name, expected, git_blob=raw)


@pytest.mark.parametrize("ids", ([6, 6], [6, 99], [6], [6, 7]))
def test_question_credit_requires_two_real_buses_on_the_two_bus_control(ids):
    ranking = {"alternatives": []}
    row = {"ranking": ranking, "ranking_sha256": sha256(canonical(ranking)).hexdigest(),
           "original_track_ids": [6, 7, 9]}
    case = {"scene": "two_buses", "utterance": "The bus up ahead", "expected_guard_action": "ask"}
    scene = {"size": [2560, 1440], "observed_bus_ids": [6, 9]}
    outcome = evaluate_case(lambda *_: {"question": "Which bus?", "track_ids": ids}, case, row, scene)
    assert outcome["action"] == "error" and not outcome["matches"]
    good = evaluate_case(lambda *_: {"question": "Which bus?", "track_ids": [6, 9]}, case, row, scene)
    assert good["action"] == "ask" and good["matches"]
