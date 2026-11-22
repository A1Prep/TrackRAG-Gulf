"""Generated software fixtures for the research pipeline, not model accuracy."""
from copy import deepcopy
from hashlib import sha1, sha256
import json
from pathlib import Path
from contextlib import nullcontext
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from tools import passenger_parking_study as study
from tools.passenger_parking_candidates import export


def inputs(tmp_path, count=1):
    path = tmp_path / "photo.png"
    Image.new("RGB", (512, 288), (65, 65, 65)).save(path)
    image_sha = study.file_hash(path)
    rows = []
    for index in range(count):
        rows.append({"request_id": f"software-{index}", "text": "Between the two cars please",
                     "input_mode": "typed", "dialogue": [], "scene": {
                         "frame": {"recording_id": "software", "frame_id": 2, "timestamp_s": .2,
                                   "width": 512, "height": 288, "image_path": str(path)},
                         "source": "predicted", "provenance": {"image_sha256": image_sha}, "tracks": [
                             {"recording_id": "software", "frame_id": 2, "track_id": i,
                              "class_name": "Car", "bbox_xyxy": box, "source": "predicted"}
                             for i, box in ((1, [80, 100, 140, 160]), (2, [220, 100, 280, 160]))]}})
    requests = tmp_path / "requests.jsonl"
    requests.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    return requests, rows


class FakeSegmenter:
    """Software fixture only; its class map conveys no real-world correctness."""
    receipt = {"software_fixture": True}
    labels = {str(i): "parking" if i == 10 else "road" if i == 13 else f"class-{i}" for i in range(65)}
    seen = []

    def __init__(self, *args):
        pass

    def predict(self, image, shape):
        self.seen.append((image.size, shape))
        return np.full(shape, 10, dtype=np.uint8), [800, 1344]


def boards(tmp_path, count=1):
    requests, rows = inputs(tmp_path, count)
    maps = tmp_path / "maps"
    study.segment_requests(requests, "not-a-real-model", maps, "cpu", segmenter_factory=FakeSegmenter)
    choices = tmp_path / "choices"
    export(requests, maps, choices, "parking")
    return requests, rows, maps, choices


@pytest.mark.parametrize("text", [
    '{"choice":"Z"}', '{"choice":true}', '{"choice":NaN}',
    '{"choice":"A","choice":null}', '{"choice":"A","explanation":"x"}',
    '[{"choice":"A"}]', 'Answer: {"choice":"A"}',
])
def test_parser_rejects_unoffered_malformed_or_ambiguous_responses(text):
    with pytest.raises(ValueError):
        study.parse_choice(text, ["A", "B"])


def test_parser_and_thought_boundary_preserve_explicit_null_not_hidden_reasoning():
    assert study.parse_choice('```json\n{"choice":"B"}\n```', ["A", "B"]) == {"choice": "B"}
    assert study.parse_choice('{"choice":null}', ["A"]) == {"choice": None}
    assert study.final_answer_tokens([1, 2, 9, 8, 4, 5], [9, 8], True) == [4, 5]
    assert study.final_answer_tokens([1, 2, 3], [9, 8], True) == []
    assert study.final_answer_tokens([1, 2, 3], [9, 8], False) == [1, 2, 3]
    assert study.content_tokens([4, 5, 99, 0, 0], {99, 0}) == [4, 5]


def test_segmenter_receipts_bind_original_pixels_and_use_actual_original_for_processor(tmp_path):
    requests, rows = inputs(tmp_path, 2)
    image = Path(rows[0]["scene"]["frame"]["image_path"])
    Image.new("RGB", (1200, 800), (50, 50, 50)).save(image)
    for row in rows:
        row["scene"]["frame"].update(width=1200, height=800)
        row["scene"]["provenance"]["image_sha256"] = study.file_hash(image)
    requests.write_text("".join(json.dumps(r) + "\n" for r in rows))
    FakeSegmenter.seen = []
    receipt, = study.segment_requests(requests, "fixture", tmp_path / "maps", "cpu", segmenter_factory=FakeSegmenter)
    assert FakeSegmenter.seen == [((1200, 800), (600, 900))]
    assert receipt["mask_shape"] == [600, 900]
    assert receipt["size"] == [1200, 800]
    assert receipt["mask_sha"] == study.file_hash(tmp_path / "maps" / receipt["file"])
    manifest = json.loads((tmp_path / "maps" / "manifest.json").read_text())
    assert manifest["unique_photographs"] == 1
    assert manifest["checkpoint"] == {"software_fixture": True}


@pytest.mark.parametrize("change", ["past", "future", "gold", "photo"])
def test_input_failure_precedes_any_heavy_model_construction(tmp_path, change):
    requests, rows = inputs(tmp_path)
    if change in ("past", "future"):
        rows[0]["scene"]["tracks"][0]["frame_id"] = 1 if change == "past" else 3
    elif change == "gold":
        rows[0]["target"] = {"point": [180, 160]}
    else:
        rows[0]["scene"]["provenance"]["image_sha256"] = "0" * 64
    requests.write_text(json.dumps(rows[0]) + "\n")
    def forbidden(*args):
        raise AssertionError("Must fail before model construction")
    with pytest.raises(ValueError):
        study.segment_requests(requests, "fixture", tmp_path / "out", segmenter_factory=forbidden)
    assert not (tmp_path / "out").exists()


def test_stale_candidate_points_are_recomputed_even_if_manifest_hash_is_updated(tmp_path):
    requests, _, maps, choices = boards(tmp_path)
    saved = json.loads((choices / "choices.json").read_text())
    assert saved[0]["options"]
    saved[0]["options"][0]["point"][0] += 1
    (choices / "choices.json").write_text(json.dumps(saved))
    manifest = json.loads((choices / "manifest.json").read_text())
    manifest["choices_sha256"] = study.file_hash(choices / "choices.json")
    (choices / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="geometry changed"):
        study.validate_choices(requests, choices, maps)


def test_valid_choice_maps_to_exact_original_point_and_requires_confirmation(tmp_path):
    requests, _, maps, choices = boards(tmp_path)
    rows, _ = study.validate_choices(requests, choices, maps)
    class FakeChooser:
        receipt = {"software_fixture": True}
        def __init__(self, *args):
            pass
        def infer_batch(self, group):
            return [{"id": r["id"], "result": {"choice": r["options"][0]["choice"]}} for r in group]
    result, = study.select_choices(requests, choices, "fixture", tmp_path / "selection",
                                   class_maps=maps, device="cpu", chooser_factory=FakeChooser)
    assert result["point_original_xy"] == rows[0]["options"][0]["point"]
    assert result["anchor_track_ids"] == [1, 2]
    assert result["requires_confirmation"] and not result["passenger_confirmed"]
    assert not result["vehicle_control"] and result["research_only"]


def test_failed_batch_does_not_become_abstention_or_remove_later_requests(tmp_path):
    requests, _, maps, choices = boards(tmp_path, 4)
    class FailsFirstBatch:
        receipt = {"software_fixture": True}
        def __init__(self, *args):
            self.calls = []
        def infer_batch(self, group):
            self.calls.append(len(group))
            if len(self.calls) == 1:
                raise RuntimeError("generated fixture failure")
            return [{"id": r["id"], "result": {"choice": None}} for r in group]
    chooser = FailsFirstBatch()
    decisions = study.select_choices(requests, choices, "fixture", tmp_path / "selection", class_maps=maps,
                                      chooser_factory=lambda *_: chooser)
    assert chooser.calls == [3, 1]
    assert [d["status"] for d in decisions] == ["operational_error"] * 3 + ["ask"]
    assert len(json.loads((tmp_path / "selection" / "raw-model-results.json").read_text())) == 4


def test_no_candidates_skips_model_and_never_claims_learned_abstention(tmp_path):
    requests, _ = inputs(tmp_path)
    class EmptySegmenter(FakeSegmenter):
        def predict(self, image, shape):
            return np.zeros(shape, dtype=np.uint8), [800, 1344]
    maps = tmp_path / "maps"
    study.segment_requests(requests, "fixture", maps, segmenter_factory=EmptySegmenter)
    export(requests, maps, tmp_path / "boards", "parking")
    def forbidden(*_):
        raise AssertionError("No candidate must skip the model")
    result, = study.select_choices(requests, tmp_path / "boards", "fixture", tmp_path / "selection",
                                   class_maps=maps, chooser_factory=forbidden)
    assert result["status"] == "unresolved" and result["reason"] == "no_candidates"
    assert not result["model_called"]


def test_offline_pin_validation_checks_real_bytes_and_git_blob_not_folder_name(tmp_path, monkeypatch):
    data = b'{"fixture":true}'
    (tmp_path / "config.json").write_bytes(data)
    (tmp_path / "preprocessor_config.json").write_bytes(data)
    (tmp_path / "model.safetensors").write_bytes(b"fixture weights")
    monkeypatch.setattr(study, "SEG_FILES", {
        "model.safetensors": ("sha256", sha256(b"fixture weights").hexdigest()),
        "config.json": ("git_blob", sha1(f"blob {len(data)}\0".encode() + data).hexdigest()),
    })
    assert study.verify_checkpoint(tmp_path, "segment")["files"]["model.safetensors"]["verified_by"] == "sha256"
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        study.verify_checkpoint(tmp_path, "segment")


def test_verified_qwen_shards_cannot_be_overridden_by_extra_monolithic_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "QWEN_FILES", {})
    (tmp_path / "model.safetensors").write_bytes(b"stale unrelated weights")
    with pytest.raises(ValueError, match="monolithic"):
        study.verify_checkpoint(tmp_path, "qwen")


def test_anchor_first_records_keep_source_prompt_binding_and_upstream_error_distinct(tmp_path):
    from tools.passenger_parking_anchors import ANCHOR_PROMPT, export as anchor_export
    requests, _, maps, _ = boards(tmp_path, 2)
    class FakeAnchors:
        receipt = {"software_fixture": True}
        def __init__(self, *args):
            pass
        def infer_anchors(self, rows):
            return [{"id": r["id"], "source_sha": r["sha256"], "request": r["request"],
                     "raw": '{"vehicles":[]}' if index == 0 else "broken fixture",
                     "prompt_sha256": sha256(ANCHOR_PROMPT.encode()).hexdigest(), "revision": study.QWEN_REVISION}
                    for index, r in enumerate(rows)]
    study.anchor_requests(requests, "fixture", tmp_path / "anchors", chooser_factory=FakeAnchors)
    anchor_export(requests, maps, tmp_path / "restricted", tmp_path / "anchors" / "anchor-records.json")
    def forbidden(*_):
        raise AssertionError("Unresolved/error anchors must not call a letter chooser")
    decisions = study.select_choices(requests, tmp_path / "restricted", "fixture", tmp_path / "selection",
                                      class_maps=maps, chooser_factory=forbidden)
    assert [d["status"] for d in decisions] == ["ask", "operational_error"]
    assert all(d["anchor_model_called"] for d in decisions)


def test_cli_help_does_not_import_optional_gpu_libraries():
    code = "import sys; from tools.passenger_parking_study import main; " \
           "assert 'torch' not in sys.modules and 'transformers' not in sys.modules; main(['--help'])"
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)
    assert "--anchor-first" in result.stdout


@pytest.mark.parametrize("thinking", [False, True])
def test_batch_generation_crops_padded_input_width_and_preserves_request_order(tmp_path, thinking):
    """Stub token streams check plumbing only, never pretrained capability."""
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (64, 32), (35, 35, 35)).save(photo)
    rows = [{"id": str(i), "marked": str(photo), "marked_sha256": study.file_hash(photo),
             "sha256": study.file_hash(photo), "request": "Between the cars", "options": [{"choice": "A"}]}
            for i in range(2)]
    class Batch(dict):
        __getattr__ = dict.__getitem__
        def to(self, device):
            assert device == "software-device"
            return self
    class Tokenizer:
        pad_token_id = 0
        def convert_tokens_to_ids(self, text):
            return 100
        def encode(self, text, **kwargs):
            assert text == "</think>"
            return [99]
        def decode(self, tokens, **kwargs):
            mapping = {21: '{"choice":"A"}', 22: '{"choice":', 23: 'null}', 42: "reasoning", 99: "</think>"}
            return "".join(mapping.get(token, "BAD_PROMPT_TOKEN") for token in tokens)
    class Processor:
        tokenizer = Tokenizer()
        def apply_chat_template(self, chat, **kwargs):
            assert kwargs["enable_thinking"] == thinking
            assert chat[0]["content"].endswith(" Offered letters: A")
            assert chat[1]["content"][0]["image"].mode == "RGB"
            return "software template<think>\n" if thinking else "software template<think>\n\n</think>\n\n"
        def __call__(self, **kwargs):
            assert kwargs["padding"] is True and len(kwargs["images"]) == 2
            return Batch(input_ids=np.array([[0, 0, 5, 6, 7], [1, 2, 3, 4, 5]]),
                         attention_mask=np.array([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]),
                         token_type_ids=np.zeros((2, 5)))
    class Model:
        generation_config = SimpleNamespace(eos_token_id=[100])
        def generate(self, **kwargs):
            assert "token_type_ids" not in kwargs
            assert kwargs["max_new_tokens"] == (1800 if thinking else 85)
            assert kwargs["do_sample"] is False and kwargs["use_cache"] is True
            tails = [[42, 99, 21, 100, 0], [42, 99, 22, 23, 100]] if thinking else [[21, 100, 0], [22, 23, 100]]
            return np.array([[1, 2, 3, 4, 5, *tail] for tail in tails])
    chooser = object.__new__(study.QwenChooser)
    chooser.processor, chooser.model = Processor(), Model()
    chooser.torch = SimpleNamespace(inference_mode=nullcontext)
    chooser.device, chooser.thinking = "software-device", thinking
    records = chooser.infer_batch(rows)
    assert [r["id"] for r in records] == ["0", "1"]
    assert [r["result"] for r in records] == [{"choice": "A"}, {"choice": None}]
    assert [r["input_tokens"] for r in records] == [3, 5]
    assert [r["generated_tokens"] for r in records] == ([3, 4] if thinking else [1, 2])
    assert all(r["stop_token_present"] and not r["reached_generation_limit_without_stop"] for r in records)
    assert all("BAD_PROMPT_TOKEN" not in r["decoded_final_answer"] for r in records)
