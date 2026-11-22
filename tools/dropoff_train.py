"""GPU QLoRA training for passenger-instruction grounding.

This command does not substitute a small model when the configured model cannot
fit. Validate/preflight work without downloading weights; train performs real
parameter updates and fresh-base adapter reload. A pilot is not held-out model
accuracy or completion of the full grounding corpus.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import re
import shutil
import sys
import time
from typing import Any


BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
BASE_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
PACKAGES = ("torch", "torchvision", "transformers", "peft", "accelerate",
            "bitsandbytes", "Pillow", "safetensors")
INTERPRET_FIELDS = {"status", "destination_point_xy", "destination_box_xyxy",
                    "clarification_question"}
MAX_CASE_SNAPSHOT_BYTES = 64 * 1024 * 1024
MAX_IMAGE_SNAPSHOT_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class TrainConfig:
    schema_version: int = 1
    run_kind: str = "pilot"
    base_model_id: str = BASE_MODEL
    base_revision: str = BASE_REVISION
    quantization: str = "nf4"
    dtype: str = "auto"
    min_gpu_memory_gib: float = 12.0
    min_free_gpu_memory_gib: float = 10.0
    min_disk_gib: float = 30.0
    min_pixels: int = 65536
    max_pixels: int = 262144
    max_sequence_length: int = 2048
    max_new_tokens: int = 192
    max_steps: int = 2
    num_train_epochs: float = 1.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 0.0002
    warmup_ratio: float = 0.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
    seed: int = 42
    eval_steps: int = 1
    save_steps: int = 1
    save_total_limit: int = 2
    max_train_cases: int | None = 10
    max_dev_cases: int | None = 4


def load_config(path: str | Path) -> TrainConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Training configuration must be an object")
    unknown = set(raw) - {field.name for field in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    config = TrainConfig(**raw)
    if config.schema_version != 1 or isinstance(config.schema_version, bool):
        raise ValueError("Unsupported training schema_version")
    if config.run_kind not in {"pilot", "adaptation"}:
        raise ValueError("run_kind must be pilot or adaptation")
    if config.base_model_id != BASE_MODEL:
        raise ValueError("This runner supports the explicitly selected Qwen3-VL-8B-Instruct only")
    if not re.fullmatch(r"[0-9a-f]{40}", config.base_revision):
        raise ValueError("base_revision must be a pinned 40-character commit")
    if config.quantization != "nf4" or config.dtype not in {"auto", "float16", "bfloat16"}:
        raise ValueError("Use nf4 with auto, float16 or bfloat16 compute")
    positive_ints = ("min_pixels", "max_pixels", "max_sequence_length", "max_new_tokens",
                     "batch_size", "gradient_accumulation_steps", "lora_rank",
                     "lora_alpha", "eval_steps", "save_steps", "save_total_limit")
    for name in positive_ints:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(config.max_steps, bool) or not isinstance(config.max_steps, int) or config.max_steps == 0 or config.max_steps < -1:
        raise ValueError("max_steps must be positive or -1 for epoch-based training")
    for name in ("min_gpu_memory_gib", "min_free_gpu_memory_gib", "min_disk_gib",
                 "num_train_epochs", "learning_rate"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("lora_dropout", "warmup_ratio"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    for name in ("max_train_cases", "max_dev_cases"):
        value = getattr(config, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be positive or null")
    if config.min_pixels > config.max_pixels:
        raise ValueError("min_pixels exceeds max_pixels")
    if config.save_steps % config.eval_steps:
        raise ValueError("save_steps must be a multiple of eval_steps for best-checkpoint selection")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(config.lora_target_modules, str) or not config.lora_target_modules:
        raise ValueError("lora_target_modules must be a nonempty language-module regex")
    re.compile(config.lora_target_modules)
    return config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_snapshot(path: str | Path, limit: int, label: str) -> bytes:
    with Path(path).open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"{label} exceeds the {limit}-byte snapshot limit")
    return payload


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_versions() -> dict[str, str | None]:
    result = {}
    for package in PACKAGES:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def _existing_parent(path: Path) -> Path:
    path = path.resolve()
    while not path.exists():
        path = path.parent
    return path


def _pinned_base_weights_cached(config: TrainConfig, cache_dir: str | Path) -> bool:
    """Recognize a complete HF snapshot without importing/downloading a model."""
    snapshot = (Path(cache_dir) / ("models--" + config.base_model_id.replace("/", "--"))
                / "snapshots" / config.base_revision)
    if not (snapshot / "config.json").is_file():
        return False
    single_file = snapshot / "model.safetensors"
    if single_file.is_file() and single_file.stat().st_size > 0:
        return True
    try:
        index = json.loads(_read_snapshot(snapshot / "model.safetensors.index.json",
                                          2 * 1024 * 1024, "Model shard index"))
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        names = set(weight_map.values())
        return all(
            isinstance(name, str) and Path(name).name == name and name.endswith(".safetensors")
            and (snapshot / name).is_file() and (snapshot / name).stat().st_size > 0
            for name in names
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def preflight(config: TrainConfig, cache_dir: str | Path, output_dir: str | Path,
              *, torch_module=None, versions: dict | None = None, for_reload=False) -> dict:
    """Report actual resources before any model download. Thresholds do not prove fit."""
    versions = package_versions() if versions is None else versions
    report = {
        "schema_version": 1, "purpose": "grounding_training_preflight",
        "created_at": utc_now(), "python": platform.python_version(),
        "platform": platform.platform(), "packages": versions, "config": asdict(config),
        "cache_dir": str(Path(cache_dir).resolve()), "output_dir": str(Path(output_dir).resolve()),
        "ready": False, "blockers": [], "fit_proven": False, "for_reload": for_reload,
        "note": "Only a real model training pilot can establish fit and numerical compatibility.",
    }
    report["base_weights_cached"] = _pinned_base_weights_cached(config, cache_dir)
    for package in PACKAGES:
        if not versions.get(package):
            report["blockers"].append(f"Missing training dependency: {package}")
    for role, path in (("cache", Path(cache_dir)), ("output", Path(output_dir))):
        free = shutil.disk_usage(_existing_parent(path)).free / 2**30
        report[f"{role}_free_gib"] = free
        minimum = 1.0 if for_reload and report["base_weights_cached"] else config.min_disk_gib
        if free < minimum:
            report["blockers"].append(f"{role} volume has {free:.2f} GiB free; need at least {minimum:g} GiB")
    try:
        if torch_module is None:
            import torch as torch_module
        report["cuda_available"] = bool(torch_module.cuda.is_available())
        if not report["cuda_available"]:
            report["blockers"].append("A CUDA GPU is required for this QLoRA runner; CPU training is not substituted")
        else:
            index = torch_module.cuda.current_device()
            properties = torch_module.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch_module.cuda.mem_get_info(index)
            total_gib, free_gib = total_bytes / 2**30, free_bytes / 2**30
            report["gpu"] = {"index": index, "name": properties.name,
                             "total_gib": total_gib, "free_gib": free_gib,
                             "cuda_build": torch_module.version.cuda,
                             "bf16_supported": bool(torch_module.cuda.is_bf16_supported())}
            if total_gib < config.min_gpu_memory_gib:
                report["blockers"].append(
                    f"GPU has {total_gib:.2f} GiB; selected 8B model pilot requires at least "
                    f"{config.min_gpu_memory_gib:g} GiB as a preflight floor")
            if free_gib < config.min_free_gpu_memory_gib:
                report["blockers"].append(f"Only {free_gib:.2f} GiB GPU memory free")
            if config.dtype == "bfloat16" and not report["gpu"]["bf16_supported"]:
                report["blockers"].append("Requested bfloat16 is unsupported by this GPU")
    except Exception as error:
        report["blockers"].append(f"CUDA probe failed: {type(error).__name__}: {error}")
    report["ready"] = not report["blockers"]
    return report


def load_training_cases(train_path: str | Path, dev_path: str | Path | None, config: TrainConfig,
                        *, case_loader=None) -> tuple[list[dict], list[dict], dict]:
    from trackrag.dropoff.data import validate_disjoint_groups
    # Bind the manifest to the exact bytes parsed, even if the source file changes
    # while image/annotation validation is running.
    train_payload = _read_snapshot(train_path, MAX_CASE_SNAPSHOT_BYTES, "Training JSONL")
    dev_payload = _read_snapshot(dev_path, MAX_CASE_SNAPSHOT_BYTES, "Development JSONL") if dev_path else None
    real_loader = case_loader is None
    if case_loader is None:
        from trackrag.dropoff.data import load_cases
        case_loader = load_cases
    train_kwargs = {"allowed_partitions": {"train"}}
    dev_kwargs = {"allowed_partitions": {"development"}}
    if real_loader:
        train_kwargs["payload"] = train_payload
        dev_kwargs["payload"] = dev_payload
    train = [dict(row) for row in case_loader(train_path, **train_kwargs)]
    dev = [dict(row) for row in case_loader(dev_path, **dev_kwargs)] if dev_path else []
    if not train:
        raise ValueError("Training partition is empty")
    if config.run_kind == "adaptation" and not dev:
        raise ValueError("Adaptation requires a separate development partition")
    train_groups = {row["scene_group"] for row in train}
    dev_groups = {row["scene_group"] for row in dev}
    if train_groups & dev_groups:
        raise ValueError(f"Source-scene overlap between train/development: {sorted(train_groups & dev_groups)}")
    ids = [row["case_id"] for row in train + dev]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case identifiers across training/development")
    validate_disjoint_groups(train, dev)
    # Check the complete supplied files before applying pilot limits.
    train_total, dev_total = len(train), len(dev)
    train = train[:config.max_train_cases] if config.max_train_cases else train
    dev = dev[:config.max_dev_cases] if config.max_dev_cases else dev
    interpreted = sum(INTERPRET_FIELDS <= set(row["supervised_fields"]) for row in train)
    if config.run_kind == "adaptation" and interpreted == 0:
        raise ValueError("Adaptation needs reviewed interpretation/status examples; region-only public data is a pilot, not the complete grounding task")
    artifacts = []
    for row in train + dev:
        image_path = Path(row["request"]["scene"]["frame"]["image_path"])
        image_payload = _read_snapshot(image_path, MAX_IMAGE_SNAPSHOT_BYTES, "Training image")
        image_sha = hashlib.sha256(image_payload).hexdigest()
        if image_sha != row["source"]["image_sha256"]:
            raise ValueError("Training image bytes changed after case validation")
        artifacts.append({
            "case_id": row["case_id"], "image_sha256": image_sha,
            "image_bytes": len(image_payload),
            "canonical_case_sha256": hashlib.sha256(json.dumps(
                row, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest(),
        })
        row["_training_image_sha256"] = image_sha
    manifest = {
        "train_path": str(Path(train_path).resolve()),
        "train_sha256": hashlib.sha256(train_payload).hexdigest(),
        "development_path": str(Path(dev_path).resolve()) if dev_path else None,
        "development_sha256": hashlib.sha256(dev_payload).hexdigest() if dev_payload is not None else None,
        "manifest_binding": "consumed_jsonl_byte_snapshots" if real_loader else "explicit_fixture_loader",
        "available_train_cases": train_total, "available_development_cases": dev_total,
        "used_train_cases": len(train), "used_development_cases": len(dev),
        "train_case_ids": [row["case_id"] for row in train],
        "development_case_ids": [row["case_id"] for row in dev],
        "train_scene_groups": sorted({row["scene_group"] for row in train}),
        "development_scene_groups": sorted({row["scene_group"] for row in dev}),
        "fully_supervised_interpretation_train_cases": interpreted,
        "case_artifacts": artifacts,
        "supervised_field_counts": {
            key: sum(key in row["supervised_fields"] for row in train)
            for key in sorted({key for row in train for key in row["supervised_fields"]})
        },
    }
    return train, dev, manifest


class CaseDataset:
    def __init__(self, cases: list[dict]):
        self.cases = cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        return self.cases[index]


class GroundingCollator:
    """Supervise only the final assistant completion; never truncate image tokens."""

    def __init__(self, processor, max_sequence_length: int, *, message_builder=None):
        self.processor = processor
        self.max_sequence_length = max_sequence_length
        self.message_builder = message_builder

    def __call__(self, cases: list[dict]) -> dict:
        from PIL import Image
        if self.message_builder is None:
            from trackrag.dropoff.formatting import build_training_messages
            builder = build_training_messages
        else:
            builder = self.message_builder
        full_texts, prompt_texts, images = [], [], []
        try:
            for case in cases:
                messages = builder(case)
                if not messages or messages[-1].get("role") != "assistant":
                    raise ValueError("A training example must end with one assistant target")
                content = messages[-1].get("content")
                if not isinstance(content, str):
                    if not isinstance(content, list) or any(part.get("type") != "text" for part in content):
                        raise ValueError("The final target must contain text only")
                image_parts = [
                    part for message in messages[:-1]
                    for part in message.get("content", []) if isinstance(message.get("content"), list)
                    if part.get("type") == "image"
                ]
                if len(image_parts) != 1:
                    raise ValueError("Each training example must contain exactly one scene image")
                part = image_parts[0]
                path = part.get("image") or part.get("path") or part.get("url")
                if not isinstance(path, str) or "://" in path or not Path(path).is_file():
                    raise ValueError("Training images must be validated local files")
                image_payload = _read_snapshot(path, MAX_IMAGE_SNAPSHOT_BYTES, "Training image")
                if (case.get("_training_image_sha256")
                        and hashlib.sha256(image_payload).hexdigest() != case["_training_image_sha256"]):
                    raise ValueError("Training image bytes changed after dataset validation")
                with BytesIO(image_payload) as image_stream, Image.open(image_stream) as original:
                    image = original.convert("RGB")
                    image.load()
                images.append(image)
                full_texts.append(self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False, enable_thinking=False))
                prompt_texts.append(self.processor.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False))
            # Both passes receive identical images and image budgets.
            batch = self.processor(text=full_texts, images=images, padding=True,
                                   truncation=False, return_tensors="pt")
            prompts = self.processor(text=prompt_texts, images=images, padding=True,
                                     truncation=False, return_tensors="pt")
        finally:
            for image in images:
                image.close()
        ids, mask = batch["input_ids"], batch["attention_mask"]
        labels = ids.clone()
        labels[:] = -100
        for index in range(len(cases)):
            positions = (mask[index] != 0).nonzero(as_tuple=False).flatten()
            prompt_positions = (prompts["attention_mask"][index] != 0).nonzero(as_tuple=False).flatten()
            full_ids = ids[index, positions]
            prefix = prompts["input_ids"][index, prompt_positions]
            if full_ids.numel() > self.max_sequence_length:
                raise ValueError("Example exceeds max_sequence_length; reduce image/input budget instead of truncating")
            if prefix.numel() >= full_ids.numel() or not full_ids[:prefix.numel()].equal(prefix):
                raise ValueError("Chat template prompt is not an exact token prefix; refusing unsafe loss masking")
            target_positions = positions[prefix.numel():]
            labels[index, target_positions] = ids[index, target_positions]
            if not (labels[index] != -100).any():
                raise ValueError("No supervised target tokens remain")
        batch["labels"] = labels
        return batch


def adapter_parameter_hashes(model) -> dict[str, str]:
    hashes = {}
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        tensor = parameter.detach().cpu().contiguous()
        raw = tensor.view(__import__("torch").uint8).numpy().tobytes()
        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(raw)
        hashes[name] = digest.hexdigest()
    if not hashes:
        raise ValueError("No LoRA parameters were found")
    return hashes


def _training_imports():
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen3VLForConditionalGeneration, Trainer,
                              TrainingArguments, set_seed)
    return locals()


def _compute_dtype(config: TrainConfig, torch):
    if config.dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return getattr(torch, config.dtype)


def checkpoint_intervals(config: TrainConfig, train_count: int) -> tuple[int, int]:
    """Ensure a short run can actually evaluate/save a development-selected adapter."""
    step_floor = config.max_steps if config.max_steps > 0 else max(
        1, int(train_count * config.num_train_epochs /
               (config.batch_size * config.gradient_accumulation_steps)))
    if step_floor < config.save_steps:
        interval = max(1, min(config.eval_steps, step_floor // 2))
        return interval, interval
    return config.eval_steps, config.save_steps


def _base_model(config: TrainConfig, cache_dir: Path, modules):
    torch = modules["torch"]
    dtype = _compute_dtype(config, torch)
    quantization = modules["BitsAndBytesConfig"](
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
    )
    return modules["Qwen3VLForConditionalGeneration"].from_pretrained(
        config.base_model_id, revision=config.base_revision, cache_dir=str(cache_dir),
        quantization_config=quantization, dtype=dtype,
        device_map={"": torch.cuda.current_device()}, attn_implementation="sdpa",
        trust_remote_code=False,
    )


def _probe_generation(model, processor, request: dict, config: TrainConfig) -> dict:
    from PIL import Image
    from trackrag.dropoff.formatting import build_messages, parse_model_output
    messages = build_messages(request, task="interpret")
    image_path = request["scene"]["frame"]["image_path"]
    with Image.open(image_path) as original:
        image = original.convert("RGB")
        image.load()
    try:
        text = processor.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        torch = __import__("torch")
        model.eval()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=config.max_new_tokens,
                                       do_sample=False, use_cache=True)
        completion = generated[:, inputs["input_ids"].shape[1]:]
        output = processor.batch_decode(completion, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)[0]
    finally:
        image.close()
    receipt = {"request_id": request["request_id"], "raw_text": output,
               "purpose": "fresh_base_adapter_reload_generation_smoke",
               "accuracy_claim": False, "structured_output_valid": False,
               "spatial_proposal_generated": False}
    try:
        frame = request["scene"]["frame"]
        parsed = parse_model_output(output, frame["width"], frame["height"])
        receipt.update({"structured_output_valid": True, "parsed_result": parsed,
                        "spatial_proposal_generated": parsed["status"] == "proposed"})
    except (TypeError, ValueError) as error:
        receipt["parse_error"] = str(error)
    return receipt


def prepare_reload_probe(case: dict, run_dir: str | Path) -> dict:
    """Copy one bound training image so the reload smoke input travels with the run."""
    from PIL import Image
    run_dir = Path(run_dir).resolve()
    source = Path(case["request"]["scene"]["frame"]["image_path"]).resolve()
    expected_sha = case["_training_image_sha256"]
    if sha256_file(source) != expected_sha:
        raise ValueError("Reload probe source image changed after dataset validation")
    with Image.open(source) as image:
        suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format)
    if suffix is None:
        raise ValueError("Reload probe requires a JPEG, PNG or WebP source image")
    destination = run_dir / ("probe-image" + suffix)
    if destination.exists():
        raise FileExistsError("Portable reload probe image already exists")
    shutil.copyfile(source, destination)
    if sha256_file(destination) != expected_sha:
        raise ValueError("Portable reload probe copy has a different image hash")
    request = deepcopy(case["request"])
    request["scene"]["frame"]["image_path"] = destination.name
    probe = {
        "schema_version": 1, "case_id": case["case_id"], "partition": "train",
        "request": request, "image_sha256": expected_sha,
        "note": "Training-case generation is a reload smoke check, not quality evaluation.",
    }
    write_json(run_dir / "reload_probe.json", probe)
    return probe


def load_reload_probe(run_dir: str | Path) -> dict:
    """Resolve only a run-local copied image; do not depend on the training dataset."""
    from trackrag.dropoff.contracts import validate_request
    run_dir = Path(run_dir).resolve()
    probe = json.loads((run_dir / "reload_probe.json").read_text(encoding="utf-8"))
    if probe.get("schema_version") != 1 or probe.get("partition") != "train":
        raise ValueError("Invalid portable training-probe record")
    request = validate_request(probe["request"])
    relative = Path(request["scene"]["frame"]["image_path"])
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name not in {
        "probe-image.jpg", "probe-image.png", "probe-image.webp"
    }:
        raise ValueError("Reload probe image must be the run-local portable basename")
    image_path = run_dir / relative
    if image_path.is_symlink() or image_path.resolve().parent != run_dir:
        raise ValueError("Reload probe image escapes the run directory")
    if sha256_file(image_path) != probe["image_sha256"]:
        raise ValueError("Reload probe image changed since training")
    request["scene"]["frame"]["image_path"] = str(image_path)
    probe["request"] = request
    return probe


def verify_reload(run_dir: str | Path, *, cache_dir: str | Path | None = None) -> dict:
    """Load a new quantized base and the saved adapter; retain failed verification."""
    run_dir = Path(run_dir).resolve()
    config = load_config(run_dir / "training_config.json")
    result = json.loads((run_dir / "training_result.json").read_text(encoding="utf-8"))
    if not result.get("nonzero_updates_verified") or result.get("optimizer_steps", 0) < 1:
        raise ValueError("No completed parameter-update evidence is available for this run")
    adapter = run_dir / "adapter"
    manifest_path = adapter / "grounding_adapter_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("base_model_id") != config.base_model_id or manifest.get("base_revision") != config.base_revision:
        raise ValueError("Adapter manifest disagrees with the training base")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if manifest.get("files", {}).get(name) != sha256_file(adapter / name):
            raise ValueError(f"Adapter hash mismatch: {name}")
    probe = load_reload_probe(run_dir)
    modules = _training_imports()
    cache_dir = Path(cache_dir or result["cache_dir"]).resolve()
    check = preflight(config, cache_dir, run_dir, for_reload=True)
    if not check["ready"]:
        write_json(run_dir / "reload_preflight.json", check)
        raise RuntimeError("Reload preflight blocked; see reload_preflight.json")
    base = _base_model(config, cache_dir, modules)
    model = modules["PeftModel"].from_pretrained(base, str(adapter),
                                               is_trainable=False, autocast_adapter_dtype=True)
    actual = adapter_parameter_hashes(model)
    if actual != result["adapter_parameter_hashes_after"]:
        raise RuntimeError("Reloaded adapter parameters do not exactly match the trained values")
    processor = modules["AutoProcessor"].from_pretrained(
        config.base_model_id, revision=config.base_revision, cache_dir=str(cache_dir),
        min_pixels=config.min_pixels, max_pixels=config.max_pixels,
        trust_remote_code=False,
    )
    generation = _probe_generation(model, processor, probe["request"], config)
    generation.update({"probe_partition": probe["partition"], "probe_case_id": probe["case_id"]})
    valid_spatial_probe = generation["structured_output_valid"] and generation["spatial_proposal_generated"]
    receipt = {"completed_at": utc_now(), "parameter_hashes_match": True,
               "generation_completed": True, "generation": generation,
               "verified": valid_spatial_probe,
               "status": "verified" if valid_spatial_probe else "failed_spatial_probe",
               "model_quality_verified": False}
    write_json(run_dir / "reload_verification.json", receipt)
    manifest["verification"]["fresh_base_reload_verified"] = True
    manifest["verification"]["generation_smoke_completed"] = True
    manifest["verification"]["structured_generation_valid"] = generation["structured_output_valid"]
    manifest["verification"]["spatial_proposal_generated"] = generation["spatial_proposal_generated"]
    write_json(manifest_path, manifest)
    del model, base
    gc.collect()
    modules["torch"].cuda.empty_cache()
    if not valid_spatial_probe:
        raise RuntimeError(
            "Fresh-base adapter parameters reloaded, but no valid spatial proposal was generated; "
            "see reload_verification.json. The full update/save/reload/spatial-prediction pilot is incomplete."
        )
    return receipt


def train(train_path: str | Path, dev_path: str | Path | None, config: TrainConfig,
          output_dir: str | Path, cache_dir: str | Path) -> dict:
    output_dir, cache_dir = Path(output_dir).resolve(), Path(cache_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Choose a new run directory; existing training artifacts will not be overwritten")
    check = preflight(config, cache_dir, output_dir)
    if not check["ready"]:
        raise RuntimeError("Training preflight blocked: " + "; ".join(check["blockers"]))
    train_cases, dev_cases, data_manifest = load_training_cases(train_path, dev_path, config)
    if any(case["review"].get("software_fixture") is True or case["source"].get("target_origin") == "software_fixture"
           for case in train_cases + dev_cases):
        raise ValueError("Software fixtures cannot be used for real model training or checkpoint selection")
    output_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "training_config.json", asdict(config))
    write_json(output_dir / "preflight.json", check)
    write_json(output_dir / "dataset_manifest.json", data_manifest)
    started = time.monotonic()
    try:
        prepare_reload_probe(train_cases[0], output_dir)
        modules = _training_imports()
        torch = modules["torch"]
        modules["set_seed"](config.seed)
        processor = modules["AutoProcessor"].from_pretrained(
            config.base_model_id, revision=config.base_revision, cache_dir=str(cache_dir),
            min_pixels=config.min_pixels, max_pixels=config.max_pixels,
            trust_remote_code=False,
        )
        processor.tokenizer.padding_side = "right"
        model = _base_model(config, cache_dir, modules)
        model.config.use_cache = False
        model = modules["prepare_model_for_kbit_training"](
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        lora = modules["LoraConfig"](
            r=config.lora_rank, lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout, target_modules=config.lora_target_modules,
            task_type="CAUSAL_LM", bias="none",
        )
        model = modules["get_peft_model"](model, lora)
        model.peft_config["default"].base_model_name_or_path = config.base_model_id
        model.peft_config["default"].revision = config.base_revision
        trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable or any("language_model" not in name or "lora_" not in name for name, _ in trainable):
            raise ValueError("Only language-model LoRA parameters may be trainable in this runner")
        before = adapter_parameter_hashes(model)
        dtype = _compute_dtype(config, torch)
        eval_steps, save_steps = checkpoint_intervals(config, len(train_cases))
        arguments = modules["TrainingArguments"](
            output_dir=str(output_dir / "checkpoints"),
            max_steps=config.max_steps, num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate, warmup_ratio=config.warmup_ratio,
            lr_scheduler_type="cosine", optim="adamw_torch",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            fp16=dtype == torch.float16, bf16=dtype == torch.bfloat16,
            eval_strategy="steps" if dev_cases else "no", eval_steps=eval_steps,
            save_strategy="steps", save_steps=save_steps,
            save_total_limit=config.save_total_limit,
            load_best_model_at_end=bool(dev_cases), metric_for_best_model="eval_loss",
            greater_is_better=False, logging_steps=1, report_to="none",
            remove_unused_columns=False, dataloader_num_workers=0,
            seed=config.seed, data_seed=config.seed,
        )
        trainer = modules["Trainer"](
            model=model, args=arguments,
            train_dataset=CaseDataset(train_cases),
            eval_dataset=CaseDataset(dev_cases) if dev_cases else None,
            data_collator=GroundingCollator(processor, config.max_sequence_length),
            processing_class=processor,
        )
        training_output = trainer.train()
        after = adapter_parameter_hashes(model)
        changed = [name for name in before if before[name] != after.get(name)]
        if trainer.state.global_step < 1 or not changed:
            raise RuntimeError("Training did not produce verifiable nonzero adapter updates")
        if not math.isfinite(float(training_output.training_loss)):
            raise RuntimeError("Training loss is nonfinite")
        adapter = output_dir / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        processor.save_pretrained(adapter)
        trainer.save_state()
        result = {
            "schema_version": 1, "run_kind": config.run_kind, "completed_at": utc_now(),
            "cache_dir": str(cache_dir), "optimizer_steps": trainer.state.global_step,
            "trainable_parameters": sum(count for _, count in trainable),
            "trainable_modules": [name for name, _ in trainable],
            "adapter_parameter_hashes_before": before,
            "adapter_parameter_hashes_after": after,
            "changed_adapter_tensors": len(changed), "nonzero_updates_verified": True,
            "training_loss": float(training_output.training_loss),
            "metrics": training_output.metrics, "log_history": trainer.state.log_history,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "effective_checkpoint_intervals": {"eval_steps": eval_steps, "save_steps": save_steps},
            "elapsed_training_seconds": time.monotonic() - started,
            "held_out_quality_claim": False,
        }
        write_json(output_dir / "training_result.json", result)
        manifest = {
            "schema_version": 1, "artifact_kind": "lora_adapter",
            "base_model_id": config.base_model_id, "base_revision": config.base_revision,
            "files": {name: sha256_file(adapter / name)
                      for name in ("adapter_config.json", "adapter_model.safetensors")},
            "training_config_sha256": sha256_file(output_dir / "training_config.json"),
            "dataset_manifest_sha256": sha256_file(output_dir / "dataset_manifest.json"),
            "training_result_sha256": sha256_file(output_dir / "training_result.json"),
            "run_kind": config.run_kind, "packages": package_versions(),
            "coordinate_system": "normalized_0_1000_image_edges",
            "image_processing": {"min_pixels": config.min_pixels, "max_pixels": config.max_pixels},
            "generation": {"max_new_tokens": config.max_new_tokens, "do_sample": False},
            "max_input_tokens": config.max_sequence_length,
            "verification": {"nonzero_updates_verified": True,
                             "fresh_base_reload_verified": False,
                             "generation_smoke_completed": False,
                             "structured_generation_valid": False,
                             "spatial_proposal_generated": False},
        }
        write_json(adapter / "grounding_adapter_manifest.json", manifest)
        write_json(output_dir / "inference_config.json", {
            "model_id": config.base_model_id, "revision": config.base_revision,
            "adapter_dir": str(adapter), "cache_dir": str(cache_dir),
            "device": "cuda", "dtype": config.dtype, "quantization": "nf4",
            "local_files_only": True, "min_pixels": config.min_pixels,
            "max_pixels": config.max_pixels, "max_input_tokens": config.max_sequence_length,
            "max_new_tokens": config.max_new_tokens,
        })
        # A new base must really load; do not merely generate with the trainer's in-memory model.
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        reload_result = verify_reload(output_dir)
        result["fresh_base_reload_verified"] = reload_result["parameter_hashes_match"]
        write_json(output_dir / "completion.json", {
            "status": "pilot_completed" if config.run_kind == "pilot" else "adaptation_completed",
            "completed_at": utc_now(), "training_result": "training_result.json",
            "reload_verification": "reload_verification.json", "model_quality_verified": False,
            "structured_generation_valid": reload_result["generation"]["structured_output_valid"],
            "spatial_proposal_generated": reload_result["generation"]["spatial_proposal_generated"],
            "note": "Training/reload completion alone does not mean a valid spatial proposal or task accuracy.",
        })
        return result
    except Exception as error:
        write_json(output_dir / "failure.json", {
            "failed_at": utc_now(), "error_type": type(error).__name__, "message": str(error),
            "elapsed_seconds": time.monotonic() - started,
            "note": "Retained artifacts are not a successful completion or model-quality result.",
        })
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "validate", "train"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/dropoff/training-pilot.json"))
        if name != "validate":
            command.add_argument("--cache-dir", type=Path, default=Path("models/grounding-cache"))
            command.add_argument("--out", type=Path, required=True)
        if name != "preflight":
            command.add_argument("--train", type=Path, required=True)
            command.add_argument("--dev", type=Path)
    reload_command = commands.add_parser("verify-reload")
    reload_command.add_argument("--run-dir", type=Path, required=True)
    reload_command.add_argument("--cache-dir", type=Path,
                                help="Override the original training-host model cache location")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-reload":
            result = verify_reload(args.run_dir, cache_dir=args.cache_dir)
        else:
            config = load_config(args.config)
            if args.command == "preflight":
                result = preflight(config, args.cache_dir, args.out)
                print(json.dumps(result, indent=2, allow_nan=False))
                return 0 if result["ready"] else 2
            if args.command == "validate":
                _, _, result = load_training_cases(args.train, args.dev, config)
            else:
                result = train(args.train, args.dev, config, args.out, args.cache_dir)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(f"Grounding {args.command} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
