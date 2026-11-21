"""tools.vision_compare_metrics: number-by-number comparison of two metrics files."""

import json

from tools.vision_compare_metrics import compare, main

ARCHIVED = {"recordings": ["a", "b"], "detection": {"AP": 0.3546156291512044, "AP50": 0.61, "n": 10},
            "tracking": {"idf1": 0.44, "nan_value": float("nan")}}


def test_identical_files_pass():
    result = compare(ARCHIVED, json.loads(json.dumps(ARCHIVED)))
    assert result["identical_within_tolerance"] and result["numbers_compared"] == 5


def test_small_difference_is_within_tolerance_and_reported_as_largest():
    reproduced = json.loads(json.dumps(ARCHIVED))
    reproduced["detection"]["AP"] = 0.3546156215792306
    result = compare(ARCHIVED, reproduced, tolerance=1e-7)
    assert result["identical_within_tolerance"]
    assert 7e-9 < result["largest_difference"] < 8e-9
    strict = compare(ARCHIVED, reproduced, tolerance=1e-9)
    assert not strict["identical_within_tolerance"] and strict["differences"][0]["path"] == "/detection/AP"


def test_order_kind_and_missing_keys_are_reported():
    reproduced = json.loads(json.dumps(ARCHIVED))
    reproduced["recordings"] = ["b", "a"]
    reproduced["detection"]["n"] = "10"
    del reproduced["tracking"]["idf1"]
    reproduced["tracking_only"] = False
    result = compare(ARCHIVED, reproduced)
    paths = {d["path"] for d in result["differences"]}
    assert paths == {"/recordings[0]", "/recordings[1]", "/detection/n"}
    assert {k["path"] for k in result["keys_in_one_file_only"]} == {"/tracking/idf1", "/tracking_only"}
    ignored = compare(ARCHIVED, dict(json.loads(json.dumps(ARCHIVED)), tracking_only=False), ignore={"tracking_only"})
    assert ignored["identical_within_tolerance"]


def test_booleans_are_not_numbers_and_equal_infinities_match():
    result = compare({"flag": True, "zero": False, "big": float("inf")}, {"flag": 1, "zero": 0, "big": float("inf")})
    assert {d["path"] for d in result["differences"]} == {"/flag", "/zero"}
    assert result["numbers_compared"] == 1 and result["largest_difference"] == 0.0
    assert not compare({"x": float("inf")}, {"x": 1.0})["identical_within_tolerance"]


def test_command_line_exit_status_and_record(tmp_path):
    archived, reproduced, record = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "record.json"
    archived.write_text(json.dumps(ARCHIVED), encoding="utf-8")
    reproduced.write_text(json.dumps(dict(ARCHIVED, tracking_only=False)), encoding="utf-8")
    assert main([str(archived), str(reproduced)]) == 1
    assert main([str(archived), str(reproduced), "--ignore", "tracking_only", "--out", str(record)]) == 0
    saved = json.loads(record.read_text(encoding="utf-8"))
    assert saved["archived_file"] == "a.json" and len(saved["archived_sha256"]) == 64 and saved["ignored_keys"] == ["tracking_only"]
