"""Software boundary checks using local images and explicitly injected replies.

These tests do not load Qwen weights or establish grounding accuracy.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from contextlib import nullcontext

import pytest
from PIL import Image

from trackrag.dropoff.model import (
    ADAPTER_MANIFEST, BASE_MODEL_ID, BASE_REVISION, ArtifactError, BackendResponse,
    GroundingConfig, GroundingModel, verify_adapter,
)


def proposed():
    return json.dumps({
        "status": "proposed", "destination_point_xy": [500, 600],
        "destination_box_xyxy": [300, 400, 700, 800],
        "clarification_question": None,
    })


class ReplyBackend:
    """A software substitute, never a pretrained/fine-tuned execution."""

    def __init__(self, reply=None, error=None):
        self.reply = proposed() if reply is None else reply
        self.error = error
        self.calls = []

    def generate(self, messages, *, config):
        self.calls.append((messages, config))
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def grounding_request(tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (100, 50), (70, 80, 90)).save(image)
    return {
        "request_id": "grounding_request-1",
        "scene": {
            "frame": {
                "recording_id": "video_054604", "frame_id": 10, "timestamp_s": 0.9,
                "width": 100, "height": 50, "image_path": str(image),
            },
            "source": "predicted",
            "tracks": [{
                "recording_id": "video_054604", "frame_id": 10, "track_id": 1,
                "class_name": "Car", "bbox_xyxy": [10, 20, 30, 40], "source": "predicted",
            }],
        },
        "text": "Drop me off beside the car.",
        "input_mode": "typed", "dialogue": [],
    }


def make_adapter(path, **manifest_changes):
    path.mkdir()
    config = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": BASE_MODEL_ID, "revision": BASE_REVISION,
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    # Hash validation fixture only, not an actual trained safetensors file.
    (path / "adapter_model.safetensors").write_bytes(b"software-hash-fixture")
    manifest = {
        "schema_version": 1, "artifact_kind": "lora_adapter",
        "base_model_id": BASE_MODEL_ID, "base_revision": BASE_REVISION,
        "files": {
            name: sha256((path / name).read_bytes()).hexdigest()
            for name in ("adapter_config.json", "adapter_model.safetensors")
        },
    }
    manifest.update(manifest_changes)
    (path / ADAPTER_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_config_and_import_are_lazy():
    project = Path(__file__).resolve().parents[1]
    script = (
        "import sys; from trackrag.dropoff.model import GroundingModel; "
        "model=GroundingModel(); "
        "assert not any(n in sys.modules for n in ('torch','transformers','peft')); "
        "assert model.config.local_files_only is True; print('lazy-offline')"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=project, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lazy-offline"


@pytest.mark.parametrize("settings", [
    {"revision": "main"}, {"revision": "a" * 39}, {"revision": "G" * 40},
    {"device": None}, {"device": "mps"}, {"max_new_tokens": True},
    {"max_input_tokens": 40000}, {"max_time_s": float("nan")},
    {"min_pixels": 999, "max_pixels": 998}, {"local_files_only": 1},
    {"dtype": "int8"}, {"device": "cpu", "quantization": "nf4"},
    {"unexpected_setting": 1},
])
def test_config_rejects_unpinned_or_unbounded_settings(settings):
    with pytest.raises(ValueError):
        GroundingModel(settings)


def test_empty_nonmapping_configuration_is_not_treated_as_defaults():
    with pytest.raises(ValueError):
        GroundingModel([])


def test_injected_generation_is_labelled_and_scaled_to_original_pixels(grounding_request):
    backend = ReplyBackend()
    model = GroundingModel(backend=backend)
    before = deepcopy(grounding_request)
    outcome = model.interpret(grounding_request)
    assert outcome["ok"] is True
    assert outcome["result"]["status"] == "proposed"
    assert outcome["result"]["destination_point_xy"] == pytest.approx([50, 30])
    assert outcome["result"]["destination_box_xyxy"] == pytest.approx([30, 20, 70, 40])
    assert grounding_request == before
    assert outcome["provenance"]["backend_kind"] == "injected_test_backend"
    assert outcome["provenance"]["injected"] is True
    assert model.last_receipt["raw_text"] == proposed()
    assert len(backend.calls) == 1
    assert backend.calls[0][1].max_new_tokens == 192


def test_snapshot_binds_the_actual_pixels_even_if_path_changes(grounding_request):
    path = Path(grounding_request["scene"]["frame"]["image_path"])
    original_hash = sha256(path.read_bytes()).hexdigest()

    class RewritingBackend:
        def generate(self, messages, *, config):
            Image.new("RGB", (100, 50), "red").save(path)
            images = [
                item["image"] for message in messages if isinstance(message.get("content"), list)
                for item in message["content"] if item.get("type") == "image"
            ]
            assert len(images) == 1
            assert images[0].getpixel((0, 0)) == (70, 80, 90)
            return proposed()

    outcome = GroundingModel(backend=RewritingBackend()).interpret(grounding_request)
    assert outcome["ok"] is True
    assert outcome["provenance"]["image"]["sha256"] == original_hash
    assert sha256(path.read_bytes()).hexdigest() != original_hash


@pytest.mark.parametrize("binding", ["0" * 64, "invalid", None])
def test_changed_or_invalid_declared_image_binding_cannot_reach_model(grounding_request, binding):
    grounding_request["scene"]["provenance"] = {"image_sha256": binding}
    backend = ReplyBackend()
    model = GroundingModel(backend=backend)
    outcome = model.interpret(grounding_request)
    assert outcome["ok"] is False
    assert outcome["error"]["code"] == "invalid_image"
    assert backend.calls == []
    assert outcome["provenance"]["image"]["sha256"] == sha256(
        Path(grounding_request["scene"]["frame"]["image_path"]).read_bytes()
    ).hexdigest()


def test_valid_declared_image_binding_is_checked_against_snapshot(grounding_request):
    digest = sha256(Path(grounding_request["scene"]["frame"]["image_path"]).read_bytes()).hexdigest()
    grounding_request["scene"]["provenance"] = {"image_sha256": digest.upper()}
    model = GroundingModel(backend=ReplyBackend())
    outcome = model.interpret(grounding_request)
    assert outcome["ok"] is True
    assert outcome["provenance"]["declared_image_sha256"] == digest
    assert outcome["provenance"]["image"]["sha256"] == digest


@pytest.mark.parametrize("change", ["future_track", "foreign_track", "gold", "foreign_dialogue"])
def test_invalid_context_never_reaches_backend(grounding_request, change):
    if change == "future_track":
        grounding_request["scene"]["tracks"][0]["frame_id"] = 11
    elif change == "foreign_track":
        grounding_request["scene"]["tracks"][0]["recording_id"] = "different"
    elif change == "gold":
        grounding_request["target"] = {"destination_point_xy": [50, 30]}
    else:
        grounding_request["dialogue"] = [{
            "role": "user", "text": "the other car", "recording_id": "different", "frame_id": 10,
        }]
    backend = ReplyBackend()
    result = GroundingModel(backend=backend).interpret(grounding_request)
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_request"
    assert backend.calls == []


def test_annotated_input_requires_an_explicit_development_override(grounding_request):
    grounding_request["scene"]["source"] = "annotated"
    grounding_request["scene"]["tracks"][0]["source"] = "annotated"
    backend = ReplyBackend()
    result = GroundingModel(backend=backend).interpret(grounding_request)
    assert result["ok"] is False
    assert backend.calls == []
    result = GroundingModel({"allow_annotated": True}, backend=backend).interpret(grounding_request)
    assert result["ok"] is True


@pytest.mark.parametrize("image_change", ["missing", "dimensions", "corrupt", "budget"])
def test_invalid_original_image_cannot_trigger_model_loading(grounding_request, image_change):
    path = Path(grounding_request["scene"]["frame"]["image_path"])
    config = {}
    if image_change == "missing":
        path.unlink()
    elif image_change == "dimensions":
        Image.new("RGB", (99, 50)).save(path)
    elif image_change == "corrupt":
        path.write_bytes(b"not-an-image")
    else:
        config["max_image_bytes"] = 1
    backend = ReplyBackend()
    outcome = GroundingModel(config, backend=backend).interpret(grounding_request)
    assert outcome["ok"] is False
    assert outcome["error"]["code"] == "invalid_image"
    assert backend.calls == []


@pytest.mark.parametrize("error,code", [
    (TimeoutError("cooperative deadline"), "timeout"),
    (RuntimeError("CUDA out of memory"), "resource_unavailable"),
    (RuntimeError("backend stopped"), "generation_failed"),
])
def test_backend_errors_are_operational_not_semantic_abstentions(grounding_request, error, code):
    model = GroundingModel(backend=ReplyBackend(error=error))
    outcome = model.interpret(grounding_request)
    assert outcome["ok"] is False
    assert outcome["error"]["code"] == code
    assert "result" not in outcome
    assert model.last_receipt["status"] == "error"


@pytest.mark.parametrize("reply", [
    "not JSON", "{}", "null", '{"status":"insufficient_visual_evidence"}',
    '{"status":"proposed","destination_point_xy":[NaN,0]}',
])
def test_malformed_output_is_not_a_supported_abstention(grounding_request, reply):
    model = GroundingModel(backend=ReplyBackend(reply=reply))
    outcome = model.interpret(grounding_request)
    assert outcome["ok"] is False
    assert outcome["error"]["code"] == "invalid_model_output"
    assert model.last_receipt["raw_text"] == reply


@pytest.mark.parametrize("status,question", [
    ("insufficient_visual_evidence", None), ("needs_clarification", "Which pair do you mean?"),
])
def test_valid_unresolved_results_remain_semantic_results(grounding_request, status, question):
    text = json.dumps({
        "status": status, "destination_point_xy": None,
        "destination_box_xyxy": None, "clarification_question": question,
    })
    outcome = GroundingModel(backend=ReplyBackend(reply=text)).interpret(grounding_request)
    assert outcome["ok"] is True
    assert outcome["result"]["status"] == status
    assert outcome["result"]["destination_point_xy"] is None


def test_backend_details_are_retained_without_changing_injection_label(grounding_request):
    backend = ReplyBackend(reply=BackendResponse(proposed(), {"input_tokens": 123, "output_tokens": 42}))
    outcome = GroundingModel(backend=backend).interpret(grounding_request)
    assert outcome["ok"] is True
    assert outcome["provenance"]["backend_details"]["input_tokens"] == 123
    assert outcome["provenance"]["backend_kind"] == "injected_test_backend"


def test_callers_cannot_rewrite_the_saved_model_receipt(grounding_request):
    model = GroundingModel(backend=ReplyBackend())
    outcome = model.interpret(grounding_request)
    assert outcome["ok"] is True
    outcome["result"]["destination_point_xy"][0] = 99
    outcome["provenance"]["model_id"] = "different"
    assert model.last_receipt["result"]["destination_point_xy"] == pytest.approx([50, 30])
    assert model.last_receipt["provenance"]["model_id"] == BASE_MODEL_ID


def test_long_backend_errors_still_return_a_valid_error_envelope(grounding_request):
    model = GroundingModel(backend=ReplyBackend(error=RuntimeError("x" * 5000)))
    outcome = model.interpret(grounding_request)
    assert outcome["ok"] is False
    assert outcome["error"]["code"] == "generation_failed"
    assert len(outcome["error"]["message"]) == 4096
    assert len(model.last_receipt["full_error_message"]) > 5000


def test_nonfinite_request_still_has_a_serializable_failure_receipt(grounding_request):
    grounding_request["scene"]["provenance"] = {"invalid": float("nan")}
    backend = ReplyBackend()
    model = GroundingModel(backend=backend)
    outcome = model.interpret(grounding_request)
    assert outcome["ok"] is False
    assert outcome["error"]["code"] == "invalid_request"
    assert backend.calls == []
    assert "request" not in model.last_receipt
    json.dumps(model.last_receipt, allow_nan=False)


@pytest.mark.parametrize("kind", ["object", "bytes", "nonfinite", "circular", "bad_repr"])
def test_nontext_backend_values_cannot_break_receipt_serialization(grounding_request, kind):
    class BadRepr:
        def __repr__(self):
            raise RuntimeError("broken repr")
    raw = {
        "object": object(), "bytes": b"not text", "nonfinite": float("nan"),
        "circular": {}, "bad_repr": BadRepr(),
    }[kind]
    if kind == "circular":
        raw["self"] = raw
    model = GroundingModel(backend=ReplyBackend(reply=BackendResponse(raw)))
    result = model.interpret(grounding_request)
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_model_output"
    assert "raw_text" not in model.last_receipt
    assert len(model.last_receipt["invalid_raw_response"]["repr"]) <= 4096
    json.dumps(model.last_receipt, allow_nan=False)


def test_invalid_backend_details_preserve_raw_text_in_a_serializable_receipt(grounding_request):
    model = GroundingModel(
        backend=ReplyBackend(reply=BackendResponse(proposed(), {"tokens": float("inf")}))
    )
    result = model.interpret(grounding_request)
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_model_output"
    assert model.last_receipt["raw_text"] == proposed()
    assert "backend_details" not in model.last_receipt["provenance"]
    json.dumps(model.last_receipt, allow_nan=False)


def test_adapter_hashes_and_base_revision_are_required(tmp_path):
    directory = tmp_path / "adapter"
    manifest = make_adapter(directory)
    config = GroundingConfig(adapter_dir=str(directory))
    verified = verify_adapter(config)
    assert verified["files"]["adapter_model.safetensors"]["sha256"] == manifest["files"]["adapter_model.safetensors"]
    assert verified["base_revision"] == BASE_REVISION
    (directory / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ArtifactError, match="SHA256"):
        verify_adapter(config)


@pytest.mark.parametrize("changes", [
    {"base_revision": "1" * 40}, {"base_model_id": "different/model"},
    {"schema_version": True}, {"artifact_kind": "pretrained"},
    {"files": {"../../other": "0" * 64}},
])
def test_adapter_manifest_cannot_relabel_or_escape_artifacts(tmp_path, changes):
    directory = tmp_path / "adapter"
    make_adapter(directory, **changes)
    with pytest.raises(ArtifactError):
        verify_adapter(GroundingConfig(adapter_dir=str(directory)))


def test_missing_adapter_never_falls_back_to_pretrained(grounding_request, tmp_path):
    backend = ReplyBackend()
    outcome = GroundingModel({"adapter_dir": str(tmp_path / "missing")}, backend=backend).interpret(grounding_request)
    assert outcome["ok"] is False
    assert outcome["error"]["code"] == "artifact_invalid"
    assert backend.calls == []


@pytest.mark.parametrize("changes", [
    {"coordinate_system": "absolute_pixels"},
    {"image_processing": {"min_pixels": 1, "max_pixels": 262144}},
    {"image_processing": {"min_pixels": True, "max_pixels": 262144}},
])
def test_adapter_preprocessing_cannot_silently_change(tmp_path, changes):
    directory = tmp_path / "adapter"
    make_adapter(directory, **changes)
    with pytest.raises(ArtifactError):
        verify_adapter(GroundingConfig(adapter_dir=str(directory)))


def test_duplicate_manifest_fields_are_rejected(tmp_path):
    directory = tmp_path / "adapter"
    make_adapter(directory)
    path = directory / ADAPTER_MANIFEST
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ArtifactError, match="Duplicate"):
        verify_adapter(GroundingConfig(adapter_dir=str(directory)))


def test_unavailable_local_weights_are_an_error_without_a_provider_fallback(grounding_request, monkeypatch):
    def unavailable(config):
        assert config.local_files_only is True
        raise OSError("No pinned snapshot in the offline cache")
    monkeypatch.setattr("trackrag.dropoff.model._TransformersBackend", unavailable)
    model = GroundingModel()
    outcome = model.interpret(grounding_request)
    assert outcome["ok"] is False
    assert outcome["error"]["code"] == "checkpoint_unavailable"
    assert model._backend is None


def test_changed_adapter_requires_a_fresh_model_instance(grounding_request, tmp_path, monkeypatch):
    directory = tmp_path / "adapter"
    manifest = make_adapter(directory)
    backend = ReplyBackend()
    monkeypatch.setattr("trackrag.dropoff.model._TransformersBackend", lambda config: backend)
    model = GroundingModel({"adapter_dir": str(directory)})
    assert model.interpret(grounding_request)["ok"] is True
    (directory / "adapter_model.safetensors").write_bytes(b"second-software-fixture")
    manifest["files"]["adapter_model.safetensors"] = sha256(
        (directory / "adapter_model.safetensors").read_bytes()
    ).hexdigest()
    (directory / ADAPTER_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    result = model.interpret(grounding_request)
    assert result["ok"] is False
    assert result["error"]["code"] == "artifact_invalid"
    assert len(backend.calls) == 1


def fake_hf_modules(monkeypatch, tokens=5):
    """Exercise real backend orchestration using substitute library objects."""
    calls = {}

    class Batch(dict):
        def to(self, device):
            calls["input_device"] = device
            return self

    tail = SimpleNamespace(shape=(1, 3))

    class Generated:
        def __getitem__(self, selection):
            calls["decode_slice"] = selection
            return tail

    class Processor:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls["processor_load"] = (name, kwargs)
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            calls["template"] = (messages, kwargs)
            # Real Transformers 5.1 cannot tokenize mixed string/content-block
            # messages directly; the renderer supports that valid chat format.
            assert kwargs["tokenize"] is False
            return "rendered chat with one image placeholder"

        def __call__(self, **kwargs):
            calls["processor_encode"] = kwargs
            return Batch({
                "input_ids": SimpleNamespace(shape=(1, tokens)),
                "pixel_values": "image-tensor", "token_type_ids": "not-for-qwen",
            })

        def batch_decode(self, value, **kwargs):
            assert value is tail
            calls["decode"] = kwargs
            return [proposed()]

    class Model:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls["model_load"] = (name, kwargs)
            return cls()

        def eval(self):
            calls["eval"] = True

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return Generated()

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.float32 = "float32-dtype"
    torch.inference_mode = nullcontext
    transformers = ModuleType("transformers")
    transformers.AutoProcessor = Processor
    transformers.Qwen3VLForConditionalGeneration = Model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


@pytest.fixture
def materialized_messages(grounding_request):
    from trackrag.dropoff.formatting import build_messages
    from trackrag.dropoff.model import _materialize_image

    with Image.open(grounding_request["scene"]["frame"]["image_path"]) as original:
        image = original.convert("RGB")
    try:
        yield _materialize_image(build_messages(grounding_request), image)
    finally:
        image.close()


def test_real_backend_uses_pinned_offline_api_and_decodes_only_new_tokens(monkeypatch, materialized_messages):
    from trackrag.dropoff.model import _TransformersBackend

    calls = fake_hf_modules(monkeypatch)
    config = GroundingConfig(device="cpu")
    result = _TransformersBackend(config).generate(materialized_messages, config=config)
    assert result.text == proposed()
    assert result.details["input_tokens"] == 5
    assert result.details["output_tokens"] == 3
    for key in ("processor_load", "model_load"):
        assert calls[key][1]["revision"] == BASE_REVISION
        assert calls[key][1]["local_files_only"] is True
        assert calls[key][1]["trust_remote_code"] is False
    assert calls["processor_load"][1]["max_pixels"] == config.max_pixels
    assert calls["template"][0] is materialized_messages
    assert calls["template"][1] == {"tokenize": False, "add_generation_prompt": True}
    encoding = calls["processor_encode"]
    assert encoding["text"] == ["rendered chat with one image placeholder"]
    assert encoding["images"] == [materialized_messages[1]["content"][0]["image"]]
    assert encoding["images"][0].getpixel((0, 0)) == (70, 80, 90)
    assert encoding["padding"] is True
    assert encoding["truncation"] is False
    assert encoding["return_tensors"] == "pt"
    assert calls["generate"]["do_sample"] is False
    assert calls["generate"]["max_new_tokens"] == 192
    assert calls["generate"]["max_time"] == 120.0
    assert "token_type_ids" not in calls["generate"]
    assert calls["decode_slice"][1].start == 5


def test_real_backend_rejects_long_inputs_without_truncation(monkeypatch, materialized_messages):
    from trackrag.dropoff.model import ResourceError, _TransformersBackend

    calls = fake_hf_modules(monkeypatch, tokens=2049)
    config = GroundingConfig(device="cpu")
    backend = _TransformersBackend(config)
    with pytest.raises(ResourceError, match="not truncated"):
        backend.generate(materialized_messages, config=config)
    assert "generate" not in calls


def test_real_backend_deadline_truncation_cannot_become_a_semantic_result(monkeypatch, materialized_messages):
    from trackrag.dropoff.model import _TransformersBackend

    calls = fake_hf_modules(monkeypatch)
    clock = iter((0.0, 2.0))
    monkeypatch.setattr("trackrag.dropoff.model.time.monotonic", lambda: next(clock))
    config = GroundingConfig(device="cpu", max_time_s=1)
    with pytest.raises(TimeoutError):
        _TransformersBackend(config).generate(materialized_messages, config=config)
    assert "generate" in calls
    assert "decode" not in calls


@pytest.mark.parametrize("invalid", ("missing", "duplicate", "path"))
def test_backend_preparation_never_reopens_an_image_path_or_selects_an_ambiguous_image(
    monkeypatch, materialized_messages, invalid,
):
    from trackrag.dropoff.model import _TransformersBackend

    calls = fake_hf_modules(monkeypatch)
    config = GroundingConfig(device="cpu")
    messages = deepcopy(materialized_messages)
    if invalid == "missing":
        messages[1]["content"] = [messages[1]["content"][1]]
    elif invalid == "duplicate":
        messages[1]["content"].append(messages[1]["content"][0])
    else:
        messages[1]["content"][0]["image"] = "path-that-must-not-be-reopened.jpg"
    with pytest.raises(ValueError, match="materialized current-frame image"):
        _TransformersBackend(config).prepare_inputs(messages, config=config)
    assert "template" not in calls
    assert "processor_encode" not in calls
    assert "generate" not in calls
