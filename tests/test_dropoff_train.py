"""Training-boundary software checks; no real Qwen training is claimed here."""

from dataclasses import asdict, replace
import ast
import hashlib
import inspect
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from tools.dropoff_train import (
    BASE_REVISION, GroundingCollator, INTERPRET_FIELDS, PACKAGES, TrainConfig,
    adapter_parameter_hashes, load_config, load_training_cases, preflight, sha256_file,
    verify_reload,
    checkpoint_intervals,
    load_reload_probe, prepare_reload_probe,
)


@pytest.mark.parametrize("changes", [
    {"base_revision": "main"}, {"base_model_id": "another/model"},
    {"learning_rate": float("nan")}, {"batch_size": True},
    {"max_steps": 0}, {"lora_dropout": 1}, {"seed": -1},
    {"max_train_cases": 0}, {"min_pixels": 200, "max_pixels": 100},
    {"dtype": "float32"}, {"extra": "unrecognized"},
    {"save_steps": 3, "eval_steps": 2},
])
def test_invalid_training_config_is_rejected(tmp_path, changes):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(changes), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        load_config(path)


def test_checked_in_configs_use_same_pinned_model():
    pilot = load_config("configs/dropoff/training-pilot.json")
    adaptation = load_config("configs/dropoff/training-adaptation.json")
    assert pilot.base_revision == adaptation.base_revision == BASE_REVISION
    assert pilot.run_kind == "pilot" and pilot.max_steps == 2
    assert adaptation.run_kind == "adaptation" and adaptation.max_train_cases is None
    assert pilot.min_pixels == adaptation.min_pixels
    assert pilot.max_pixels == adaptation.max_pixels


def test_short_adaptation_still_has_development_checkpoint_selection():
    config = replace(TrainConfig(), run_kind="adaptation", max_steps=-1,
                     num_train_epochs=2, gradient_accumulation_steps=8,
                     eval_steps=25, save_steps=25)
    assert checkpoint_intervals(config, train_count=80) == (10, 10)
    assert checkpoint_intervals(config, train_count=1) == (1, 1)
    assert checkpoint_intervals(config, train_count=1000) == (25, 25)


class FakeCuda:
    def __init__(self, total=24, free=22, available=True, bf16=True):
        self.total, self.free, self.available, self.bf16 = total, free, available, bf16

    def is_available(self):
        return self.available

    def current_device(self):
        return 0

    def get_device_properties(self, index):
        return SimpleNamespace(name="fixture GPU")

    def mem_get_info(self, index):
        return int(self.free * 2**30), int(self.total * 2**30)

    def is_bf16_supported(self):
        return self.bf16


def resource_report(tmp_path, monkeypatch, cuda, config=None, **kwargs):
    monkeypatch.setattr("tools.dropoff_train.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=100 * 2**30))
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="fixture"))
    return preflight(config or TrainConfig(), tmp_path / "cache", tmp_path / "run",
                     torch_module=torch, versions={name: "fixture" for name in PACKAGES}, **kwargs)


def test_four_gib_is_blocked_without_substituting_model(tmp_path, monkeypatch):
    result = resource_report(tmp_path, monkeypatch, FakeCuda(total=4, free=3.5))
    assert result["ready"] is False
    assert any("GPU has 4.00" in reason for reason in result["blockers"])
    assert result["config"]["base_model_id"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert result["fit_proven"] is False
    assert not (tmp_path / "cache").exists()


def test_resource_floor_never_claims_fit(tmp_path, monkeypatch):
    result = resource_report(tmp_path, monkeypatch, FakeCuda())
    assert result["ready"] is True and result["fit_proven"] is False


def test_bf16_on_unsupported_gpu_is_rejected(tmp_path, monkeypatch):
    result = resource_report(tmp_path, monkeypatch, FakeCuda(bf16=False),
                             replace(TrainConfig(), dtype="bfloat16"))
    assert not result["ready"]
    assert any("bfloat16" in value for value in result["blockers"])


def test_no_cuda_does_not_fall_back_to_cpu(tmp_path, monkeypatch):
    result = resource_report(tmp_path, monkeypatch, FakeCuda(available=False))
    assert not result["ready"]
    assert any("CPU training is not substituted" in value for value in result["blockers"])


def test_reload_reuses_existing_cache_disk_floor(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.dropoff_train.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=15 * 2**30))
    torch = SimpleNamespace(cuda=FakeCuda(), version=SimpleNamespace(cuda="fixture"))
    versions = {name: "fixture" for name in PACKAGES}
    snapshot = tmp_path / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots" / BASE_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"fixture.tensor": "model-00001-of-00001.safetensors"}
    }), encoding="utf-8")
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"fixture-never-loaded")
    normal = preflight(TrainConfig(), tmp_path, tmp_path, torch_module=torch, versions=versions)
    reload = preflight(TrainConfig(), tmp_path, tmp_path, torch_module=torch,
                       versions=versions, for_reload=True)
    assert not normal["ready"] and reload["ready"]
    assert reload["base_weights_cached"] is True


def test_reload_with_empty_cache_still_needs_base_download_space(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.dropoff_train.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=15 * 2**30))
    torch = SimpleNamespace(cuda=FakeCuda(), version=SimpleNamespace(cuda="fixture"))
    report = preflight(TrainConfig(), tmp_path, tmp_path, torch_module=torch,
                       versions={name: "fixture" for name in PACKAGES}, for_reload=True)
    assert not report["ready"] and report["base_weights_cached"] is False


def make_case(tmp_path, case_id="train-1", group="train-scene", fields=None):
    """Canonical synthetic fixture. Its reviewer names never represent human evidence."""
    from trackrag.dropoff.data import inference_input_sha256, supervision_sha256
    image = tmp_path / f"{case_id}.png"
    colour = tuple((sum(case_id.encode()) + value * 37) % 256 for value in range(3))
    Image.new("RGB", (20, 10), colour).save(image)
    fields = list(fields if fields is not None else {"destination_point_xy", "destination_box_xyxy"})
    request = {
        "request_id": case_id,
        "scene": {"frame": {
            "recording_id": group, "frame_id": 0, "timestamp_s": 0.0,
            "width": 20, "height": 10, "image_path": str(image),
        }, "source": "predicted", "tracks": []},
        "text": "Synthetic fixture instruction", "input_mode": "typed", "dialogue": [],
    }
    target = {
        "destination_point_xy": [10.0, 5.0], "destination_box_xyxy": [2.0, 2.0, 18.0, 8.0],
    }
    if "status" in fields:
        target["status"] = "proposed"
    if "clarification_question" in fields:
        target["clarification_question"] = None
    if "anchor_boxes_xyxy" in fields:
        target["anchor_boxes_xyxy"] = [[0.0, 0.0, 2.0, 2.0]]
    image_sha = sha256_file(image)
    return {
        "schema_version": 1, "partition": "development" if case_id.startswith("dev") else "train",
        "case_id": case_id, "scene_group": group,
        "source": {"dataset": "synthetic_test_fixture", "source_split": "train",
                   "image_sha256": image_sha},
        "supervised_fields": fields,
        "request": request, "target": target,
        "review": {
            "status": "human_reviewed", "reviewers": ["fixture-reviewer-not-real-human-data"],
            "reviewed_at": "2026-11-20T00:00:00+00:00", "image_sha256": image_sha,
            "supervision_sha256": supervision_sha256(target, fields, request["scene"]["frame"]),
            "input_sha256": inference_input_sha256(request, image_sha), "fixture_only": True,
        },
    }


def case_paths(tmp_path):
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train.write_text("train fixture", encoding="utf-8")
    dev.write_text("dev fixture", encoding="utf-8")
    return train, dev


def test_allowed_partitions_passed_strictly_and_bytes_bound(tmp_path):
    train, dev = case_paths(tmp_path)
    rows = {train: [make_case(tmp_path)], dev: [make_case(tmp_path, "dev-1", "dev-scene")]}
    calls = []

    def loader(path, *, allowed_partitions):
        calls.append((Path(path), allowed_partitions))
        return rows[Path(path)]

    used_train, used_dev, manifest = load_training_cases(train, dev, TrainConfig(), case_loader=loader)
    assert calls == [(train, {"train"}), (dev, {"development"})]
    assert len(manifest["case_artifacts"]) == 2
    assert all(len(row["image_sha256"]) == 64 for row in manifest["case_artifacts"])
    assert used_train[0]["_training_image_sha256"]
    assert used_dev[0]["_training_image_sha256"]


def test_overlap_is_checked_before_pilot_slicing(tmp_path):
    train, dev = case_paths(tmp_path)
    rows = {
        train: [make_case(tmp_path, "train-a", "first"), make_case(tmp_path, "train-b", "shared")],
        dev: [make_case(tmp_path, "dev-a", "shared")],
    }
    with pytest.raises(ValueError, match="Source-scene overlap"):
        load_training_cases(train, dev, replace(TrainConfig(), max_train_cases=1),
                            case_loader=lambda path, **kw: rows[Path(path)])


def test_adaptation_rejects_region_only_training(tmp_path):
    train, dev = case_paths(tmp_path)
    rows = {train: [make_case(tmp_path)], dev: [make_case(tmp_path, "dev-1", "dev-scene")]}
    with pytest.raises(ValueError, match="region-only"):
        load_training_cases(train, dev, replace(TrainConfig(), run_kind="adaptation"),
                            case_loader=lambda path, **kw: rows[Path(path)])


def test_adaptation_requires_development(tmp_path):
    train, _ = case_paths(tmp_path)
    with pytest.raises(ValueError, match="separate development"):
        load_training_cases(train, None, replace(TrainConfig(), run_kind="adaptation"),
                            case_loader=lambda path, **kw: [make_case(tmp_path, fields=INTERPRET_FIELDS)])


def test_adaptation_loads_review_bound_canonical_fixtures(tmp_path):
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    first = make_case(tmp_path, fields=INTERPRET_FIELDS)
    second = make_case(tmp_path, "dev-1", "dev-scene", fields=INTERPRET_FIELDS)
    train.write_text(json.dumps(first) + "\n", encoding="utf-8")
    dev.write_text(json.dumps(second) + "\n", encoding="utf-8")
    _, _, manifest = load_training_cases(train, dev, replace(TrainConfig(), run_kind="adaptation"))
    assert manifest["fully_supervised_interpretation_train_cases"] == 1


def test_manifest_hash_binds_snapshot_consumed_even_if_jsonl_changes(tmp_path, monkeypatch):
    from trackrag.dropoff import data
    path = tmp_path / "train.jsonl"
    row = make_case(tmp_path)
    payload = (json.dumps(row) + "\n").encode("utf-8")
    path.write_bytes(payload)
    real_loader = data.load_cases

    def loader(file, **kwargs):
        assert kwargs["payload"] == payload
        Path(file).write_text("changed source file; not the consumed snapshot", encoding="utf-8")
        return real_loader(file, **kwargs)

    monkeypatch.setattr(data, "load_cases", loader)
    train, _, manifest = load_training_cases(path, None, TrainConfig())
    assert train[0]["case_id"] == row["case_id"]
    assert manifest["train_sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["train_sha256"] != sha256_file(path)
    assert manifest["manifest_binding"] == "consumed_jsonl_byte_snapshots"


def test_changed_image_cannot_be_rebound_to_previously_validated_labels(tmp_path, monkeypatch):
    from trackrag.dropoff import data
    path = tmp_path / "train.jsonl"
    row = make_case(tmp_path)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    real_loader = data.load_cases

    def loader(file, **kwargs):
        validated = real_loader(file, **kwargs)
        Path(row["request"]["scene"]["frame"]["image_path"]).write_bytes(b"changed image after validation")
        return validated

    monkeypatch.setattr(data, "load_cases", loader)
    with pytest.raises(ValueError, match="changed after case validation"):
        load_training_cases(path, None, TrainConfig())


def test_identical_image_bytes_cannot_hide_under_different_groups(tmp_path):
    train, dev = case_paths(tmp_path)
    first = make_case(tmp_path, "train-case", "train-scene")
    second = make_case(tmp_path, "dev-case", "dev-scene")
    second["source"]["image_sha256"] = first["source"]["image_sha256"]
    rows = {train: [first], dev: [second]}
    with pytest.raises(ValueError, match="identical image bytes"):
        load_training_cases(train, dev, TrainConfig(),
                            case_loader=lambda path, **kw: rows[Path(path)])


class FakeProcessor:
    """Only controlled tokens for checking the training loss boundary."""
    def __init__(self, torch, mismatch=False):
        self.torch, self.mismatch = torch, mismatch
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        content = messages[-1]["content"]
        tag = content if isinstance(content, str) else content[-1]["text"]
        return ("prompt:" if add_generation_prompt else "full:") + tag

    def __call__(self, *, text, images, padding, truncation, return_tensors):
        self.calls.append({"images": len(images), "truncation": truncation})
        rows = []
        for string in text:
            if string.startswith("prompt:"):
                row = [10, 999, 11] if "short" in string else [10, 999, 11, 12]
                if self.mismatch:
                    row[-1] = 77
            else:
                row = [10, 999, 11, 20, 9] if "short" in string else [10, 999, 11, 12, 21, 22, 9]
            rows.append(row)
        width = max(map(len, rows))
        return {
            # Fixture pad id deliberately equals the genuine end-of-turn id.
            "input_ids": self.torch.tensor([row + [9] * (width - len(row)) for row in rows]),
            "attention_mask": self.torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
            "pixel_values": self.torch.ones((len(images), 2)),
            "image_grid_thw": self.torch.ones((len(images), 3), dtype=self.torch.long),
        }


def messages(case):
    tag = case["tag"]
    return [
        {"role": "system", "content": "fixture system"},
        {"role": "user", "content": [
            {"type": "image", "image": case["image"]},
            {"type": "text", "text": tag},
        ]},
        {"role": "assistant", "content": tag},
    ]


def collator_case(tmp_path, tag):
    image = tmp_path / f"{tag}.png"
    Image.new("RGB", (20, 10), "green").save(image)
    return {"tag": tag, "image": str(image)}


def test_loss_masks_prompt_image_padding_but_not_real_eos(tmp_path):
    torch = pytest.importorskip("torch")
    processor = FakeProcessor(torch)
    collator = GroundingCollator(processor, 100, message_builder=messages)
    batch = collator([collator_case(tmp_path, "short"), collator_case(tmp_path, "long")])
    assert batch["labels"].tolist() == [
        [-100, -100, -100, 20, 9, -100, -100],
        [-100, -100, -100, -100, 21, 22, 9],
    ]
    assert all(not call["truncation"] and call["images"] == 2 for call in processor.calls)
    assert batch["pixel_values"].shape[0] == 2


def test_token_prefix_mismatch_refuses_unsafe_supervision(tmp_path):
    torch = pytest.importorskip("torch")
    collator = GroundingCollator(FakeProcessor(torch, mismatch=True), 100, message_builder=messages)
    with pytest.raises(ValueError, match="exact token prefix"):
        collator([collator_case(tmp_path, "short")])


def test_sequence_limit_refuses_truncation(tmp_path):
    torch = pytest.importorskip("torch")
    collator = GroundingCollator(FakeProcessor(torch), 4, message_builder=messages)
    with pytest.raises(ValueError, match="instead of truncating"):
        collator([collator_case(tmp_path, "short")])


def test_changed_training_image_is_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    case = collator_case(tmp_path, "short")
    case["_training_image_sha256"] = "0" * 64
    collator = GroundingCollator(FakeProcessor(torch), 100, message_builder=messages)
    with pytest.raises(ValueError, match="bytes changed"):
        collator([case])


def test_collator_decodes_same_hashed_image_snapshot(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    case = collator_case(tmp_path, "short")
    case["_training_image_sha256"] = sha256_file(case["image"])
    real_open = Image.open

    def open_consumed_snapshot(file, *args, **kwargs):
        assert isinstance(file, BytesIO), "Expected the exact hashed image bytes"
        Path(case["image"]).write_bytes(b"file changed after the consumed snapshot was captured")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(Image, "open", open_consumed_snapshot)
    batch = GroundingCollator(FakeProcessor(torch), 100, message_builder=messages)([case])
    assert batch["labels"][0].tolist() == [-100, -100, -100, 20, 9]


def test_adapter_hash_detects_fixture_parameter_updates():
    torch = pytest.importorskip("torch")

    class FixtureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.ones((2, 2)))
            self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)

    model = FixtureModel()
    before = adapter_parameter_hashes(model)
    with torch.no_grad():
        model.lora_A[0, 0] += 1
    after = adapter_parameter_hashes(model)
    assert list(before) == ["lora_A"] and before != after


def test_installed_transformers_accepts_training_constructor_keywords():
    """Check pinned-library API compatibility without loading a model or GPU."""
    pytest.importorskip("transformers")
    from transformers import Trainer, TrainingArguments
    import tools.dropoff_train as module

    tree = ast.parse(inspect.getsource(module.train))
    checked = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Subscript):
            continue
        key = node.func.slice
        if not isinstance(key, ast.Constant) or key.value not in {"Trainer", "TrainingArguments"}:
            continue
        constructor = Trainer if key.value == "Trainer" else TrainingArguments
        inspect.signature(constructor).bind_partial(**{item.arg: None for item in node.keywords})
        checked.add(key.value)
    assert checked == {"Trainer", "TrainingArguments"}


def test_invalid_reload_generation_retained_and_cannot_complete_pilot(tmp_path, monkeypatch):
    """Controlled backend: checks failure accounting, not adapter loading capability."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture-never-loaded")
    config = TrainConfig()
    (tmp_path / "training_config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    expected = {"fixture.lora_A": "fixture-hash"}
    (tmp_path / "training_result.json").write_text(json.dumps({
        "nonzero_updates_verified": True, "optimizer_steps": 1,
        "cache_dir": str(tmp_path / "cache"), "adapter_parameter_hashes_after": expected,
    }), encoding="utf-8")
    case = make_case(tmp_path, "reload-fixture")
    case["_training_image_sha256"] = case["source"]["image_sha256"]
    prepare_reload_probe(case, tmp_path)
    manifest_path = adapter / "grounding_adapter_manifest.json"
    manifest_path.write_text(json.dumps({
        "base_model_id": config.base_model_id, "base_revision": config.base_revision,
        "files": {name: sha256_file(adapter / name)
                  for name in ("adapter_config.json", "adapter_model.safetensors")},
        "verification": {},
    }), encoding="utf-8")
    modules = {
        "PeftModel": SimpleNamespace(from_pretrained=lambda *a, **kw: object()),
        "AutoProcessor": SimpleNamespace(from_pretrained=lambda *a, **kw: object()),
        "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
    }
    monkeypatch.setattr("tools.dropoff_train._training_imports", lambda: modules)
    monkeypatch.setattr("tools.dropoff_train.preflight", lambda *a, **kw: {"ready": True})
    monkeypatch.setattr("tools.dropoff_train._base_model", lambda *a: object())
    monkeypatch.setattr("tools.dropoff_train.adapter_parameter_hashes", lambda model: expected)
    monkeypatch.setattr("tools.dropoff_train._probe_generation", lambda *a: {
        "raw_text": "not valid spatial JSON", "structured_output_valid": False,
        "spatial_proposal_generated": False, "parse_error": "fixture invalid JSON",
    })
    with pytest.raises(RuntimeError, match="no valid spatial proposal"):
        verify_reload(tmp_path)
    receipt = json.loads((tmp_path / "reload_verification.json").read_text(encoding="utf-8"))
    assert receipt["verified"] is False and receipt["status"] == "failed_spatial_probe"
    assert receipt["generation"]["raw_text"] == "not valid spatial JSON"
    assert not (tmp_path / "completion.json").exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["verification"]["fresh_base_reload_verified"] is True
    assert manifest["verification"]["structured_generation_valid"] is False


def test_reload_probe_survives_run_relocation_and_missing_source(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    case = make_case(source_dir)
    case["_training_image_sha256"] = case["source"]["image_sha256"]
    original_run = tmp_path / "original-run"
    original_run.mkdir()
    probe = prepare_reload_probe(case, original_run)
    assert probe["request"]["scene"]["frame"]["image_path"] == "probe-image.png"
    # Copy only run artifacts to a new location, then remove the original source image.
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for name in ("probe-image.png", "reload_probe.json"):
        (relocated / name).write_bytes((original_run / name).read_bytes())
    Path(case["request"]["scene"]["frame"]["image_path"]).unlink()
    loaded = load_reload_probe(relocated)
    assert loaded["request"]["scene"]["frame"]["image_path"] == str(relocated / "probe-image.png")
    assert loaded["image_sha256"] == case["source"]["image_sha256"]


def test_reload_probe_refuses_changed_copied_image(tmp_path):
    case = make_case(tmp_path)
    case["_training_image_sha256"] = case["source"]["image_sha256"]
    prepare_reload_probe(case, tmp_path)
    (tmp_path / "probe-image.png").write_bytes(b"changed fixture bytes")
    with pytest.raises(ValueError, match="changed since training"):
        load_reload_probe(tmp_path)
