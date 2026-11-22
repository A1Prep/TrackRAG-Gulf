"""Lazy Qwen3-VL grounding inference with explicit artifact and error boundaries.

The default is offline. Constructing an adapter never downloads weights or imports
torch. Injected backends are for software checks and are labelled as such.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from hashlib import sha256
from importlib import metadata
from io import BytesIO
import json
import math
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Mapping, Protocol


BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BASE_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
ADAPTER_MANIFEST = "grounding_adapter_manifest.json"
_SHA = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


class ArtifactError(ValueError):
    """A saved adapter is absent, inconsistent or not bound to this base."""


class ResourceError(RuntimeError):
    """The requested inference resources are unavailable."""


@dataclass(frozen=True)
class GroundingConfig:
    model_id: str = BASE_MODEL_ID
    revision: str = BASE_REVISION
    adapter_dir: str | None = None
    cache_dir: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    quantization: str = "none"
    local_files_only: bool = True
    allow_annotated: bool = False
    max_new_tokens: int = 192
    max_input_tokens: int = 2048
    max_time_s: float = 120.0
    min_pixels: int = 256 * 256
    max_pixels: int = 512 * 512
    max_image_bytes: int = 32 * 1024 * 1024
    max_original_pixels: int = 48_000_000
    download_min_free_gib: float = 24.0
    runtime: str = "resident"
    model_dir: str | None = None
    suffix_layers: int = 8
    inference_resident_layers: int = 20
    observation_format: str = 'json'

    def __post_init__(self):
        if self.observation_format not in ('json', 'compact_rows_v1'):
            raise ValueError('Unsupported observation formatting profile')
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be nonempty")
        if not isinstance(self.revision, str) or not _REVISION.fullmatch(self.revision):
            raise ValueError("revision must be a pinned 40-character lowercase commit SHA")
        for name in ("adapter_dir", "cache_dir", "model_dir"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty path string or null")
        if self.runtime not in ("resident", "streamed"):
            raise ValueError("runtime must be resident or streamed")
        if type(self.suffix_layers) is not int or not 1 <= self.suffix_layers <= 8:
            raise ValueError("suffix_layers must be an integer between 1 and 8")
        if (type(self.inference_resident_layers) is not int
                or not 0 <= self.inference_resident_layers <= 24):
            raise ValueError("inference_resident_layers must be between 0 and 24")
        if not isinstance(self.device, str) or (
            self.device not in ("auto", "cpu", "cuda") and not re.fullmatch(r"cuda:[0-9]+", self.device)
        ):
            raise ValueError("device must be auto, cpu, cuda or cuda:N")
        if self.dtype not in ("auto", "float16", "bfloat16", "float32"):
            raise ValueError("Unsupported dtype")
        if self.quantization not in ("none", "nf4"):
            raise ValueError("quantization must be none or nf4")
        if self.quantization == "nf4" and self.device == "cpu":
            raise ValueError("NF4 inference requires a CUDA device")
        for name in ("local_files_only", "allow_annotated"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("max_new_tokens", "max_input_tokens", "min_pixels", "max_pixels",
                     "max_image_bytes", "max_original_pixels"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_new_tokens > 4096 or self.max_input_tokens > 32768:
            raise ValueError("Token budgets exceed this adapter's supported bounds")
        if self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels exceeds max_pixels")
        for name in ("max_time_s", "download_min_free_gib"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GroundingConfig":
        if not isinstance(value, Mapping):
            raise ValueError("Grounding configuration must be an object")
        extra = set(value) - {field.name for field in fields(cls)}
        if extra:
            raise ValueError(f"Unknown grounding configuration fields: {sorted(extra)}")
        return cls(**value)

    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir).expanduser().resolve() if self.cache_dir else (
            Path(__file__).resolve().parents[2] / "models" / "dropoff-cache"
        )


def _digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _load_json(path: Path) -> dict:
    def no_constant(value):
        raise ArtifactError(f"Nonfinite artifact JSON: {value}")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactError(f"Duplicate artifact field: {key}")
            result[key] = value
        return result
    try:
        if path.stat().st_size > 1024 * 1024:
            raise ArtifactError(f"{path.name} exceeds the artifact metadata byte budget")
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=no_constant, object_pairs_hook=unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Cannot read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"{path.name} must contain an object")
    return value


def verify_adapter(config: GroundingConfig) -> dict | None:
    """Validate the manifest and both files before PEFT sees the directory.

    Matching hashes bind this handoff to its files; they do not independently prove
    who trained it or that its parameters improve task performance.
    """
    if config.adapter_dir is None:
        return None
    directory = Path(config.adapter_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ArtifactError("Adapter directory is unavailable")
    manifest_path = directory / ADAPTER_MANIFEST
    if manifest_path.is_symlink():
        raise ArtifactError("Adapter manifest must be a regular local file")
    manifest = _load_json(manifest_path)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ArtifactError("Unsupported adapter manifest schema")
    if manifest.get("artifact_kind") != "lora_adapter":
        raise ArtifactError("Manifest does not describe a LoRA adapter")
    if manifest.get("base_model_id") != config.model_id or manifest.get("base_revision") != config.revision:
        raise ArtifactError("Adapter is bound to a different base model/revision")
    if "coordinate_system" in manifest and manifest["coordinate_system"] != "normalized_0_1000_image_edges":
        raise ArtifactError("Adapter uses a different coordinate convention")
    if "observation_format" in manifest and manifest["observation_format"] != config.observation_format:
        raise ArtifactError("Adapter observation serialization differs from this inference profile")
    if "image_processing" in manifest:
        processing = manifest["image_processing"]
        if not isinstance(processing, dict) or any(
            type(processing.get(name)) is not int or processing[name] != getattr(config, name)
            for name in ("min_pixels", "max_pixels")
        ):
            raise ArtifactError("Adapter image preprocessing differs from this inference configuration")
    names = {"adapter_config.json", "adapter_model.safetensors"}
    digests = manifest.get("files")
    if not isinstance(digests, dict) or set(digests) != names:
        raise ArtifactError("Manifest must bind exactly adapter_config.json and adapter_model.safetensors")
    verified = {}
    for name in sorted(names):
        wanted = digests[name]
        if not isinstance(wanted, str) or not _SHA.fullmatch(wanted):
            raise ArtifactError(f"Invalid SHA256 for {name}")
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ArtifactError(f"Missing, empty or linked adapter file: {name}")
        actual = _digest(path)
        if actual != wanted:
            raise ArtifactError(f"SHA256 mismatch for {name}")
        verified[name] = {"sha256": actual, "bytes": path.stat().st_size}
    peft = _load_json(directory / "adapter_config.json")
    if peft.get("peft_type") != "LORA" or peft.get("base_model_name_or_path") != config.model_id:
        raise ArtifactError("PEFT config is not a LoRA for the configured base model")
    if peft.get("revision") not in (None, config.revision):
        raise ArtifactError("PEFT config revision disagrees with the base revision")
    return {
        "path": str(directory), "manifest_sha256": _digest(manifest_path),
        "base_model_id": config.model_id, "base_revision": config.revision, "files": verified,
    }


@dataclass
class BackendResponse:
    text: str
    details: dict | None = None


class GroundingBackend(Protocol):
    def generate(self, messages: list[dict], *, config: GroundingConfig) -> str | BackendResponse:
        ...


class _TransformersBackend:
    """Real local HF/PEFT backend. No hosted-provider or smaller-model fallback."""

    def __init__(self, config: GroundingConfig):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if config.adapter_dir:
            from peft import PeftModel

        self.torch = torch
        self.config = config
        if config.device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = "cuda:0" if config.device == "cuda" else config.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ResourceError("CUDA was requested but is unavailable")
        if config.quantization == "nf4" and not device.startswith("cuda"):
            raise ResourceError("NF4 inference requires CUDA")
        self.device = device
        dtype = config.dtype
        if dtype == "auto":
            dtype = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device.startswith("cuda") else "float32"
        if dtype == "bfloat16" and device.startswith("cuda") and not torch.cuda.is_bf16_supported():
            raise ResourceError("This CUDA device does not support bfloat16")
        torch_dtype = getattr(torch, dtype)
        cache = config.resolved_cache_dir()
        if not config.local_files_only:
            ancestor = cache
            while not ancestor.exists():
                ancestor = ancestor.parent
            available = shutil.disk_usage(ancestor).free / 2**30
            if available < config.download_min_free_gib:
                raise ResourceError(
                    f"Only {available:.2f} GiB free; configured download headroom is {config.download_min_free_gib:g} GiB"
                )
            cache.mkdir(parents=True, exist_ok=True)
        shared = {
            "revision": config.revision, "cache_dir": str(cache),
            "local_files_only": config.local_files_only, "trust_remote_code": False,
        }
        self.processor = AutoProcessor.from_pretrained(
            config.model_id, **shared, min_pixels=config.min_pixels, max_pixels=config.max_pixels,
        )
        options = {
            **shared, "dtype": torch_dtype, "device_map": device, "attn_implementation": "sdpa",
        }
        if config.quantization == "nf4":
            from transformers import BitsAndBytesConfig
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch_dtype,
            )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(config.model_id, **options)
        if config.adapter_dir:
            self.model = PeftModel.from_pretrained(
                self.model, str(Path(config.adapter_dir).expanduser().resolve()),
                is_trainable=False, local_files_only=True,
            )
        self.model.eval()
        self.dtype = dtype

    def prepare_inputs(self, messages: list[dict], *, config: GroundingConfig):
        """Encode the rendered chat and the already verified current image.

        Transformers 5.1's multimodal ``tokenize=True`` template path assumes
        every message has content blocks. Our valid chat includes string system,
        dialogue and instruction messages. Render text first, as the training
        collator does, then supply the same in-memory image snapshot explicitly.
        No image path is reopened and over-budget inputs are never truncated.
        """
        from PIL.Image import Image

        images = [
            part.get("image")
            for message in messages if isinstance(message.get("content"), list)
            for part in message["content"] if part.get("type") == "image"
        ]
        if len(images) != 1 or not isinstance(images[0], Image):
            raise ValueError("Processor inputs require exactly one materialized current-frame image")
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise ValueError("Chat template must render text before multimodal encoding")
        inputs = self.processor(
            text=[rendered], images=images, padding=True,
            truncation=False, return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        tokens = int(inputs["input_ids"].shape[-1])
        if tokens > config.max_input_tokens:
            raise ResourceError(
                f"Input has {tokens} tokens, exceeding {config.max_input_tokens}; input was not truncated"
            )
        return inputs

    def generate(self, messages: list[dict], *, config: GroundingConfig) -> BackendResponse:
        inputs = self.prepare_inputs(messages, config=config)
        tokens = int(inputs["input_ids"].shape[-1])
        inputs = inputs.to(self.model.device)
        started = time.monotonic()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=config.max_new_tokens,
                max_time=config.max_time_s, use_cache=True,
            )
        elapsed = time.monotonic() - started
        # HF max_time is cooperative, checked between decoding steps, not a hard
        # interrupt of a native GPU call. Do not accept deadline-truncated JSON.
        if elapsed >= config.max_time_s:
            raise TimeoutError("Generation exceeded its cooperative time budget")
        tail = generated[:, tokens:]
        texts = self.processor.batch_decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if len(texts) != 1:
            raise ValueError("Expected exactly one grounding generation")
        packages = {}
        for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "Pillow"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
        return BackendResponse(texts[0], {
            "device": self.device, "dtype": self.dtype, "packages": packages,
            "input_tokens": tokens, "output_tokens": int(tail.shape[-1]),
            "generation_elapsed_s": elapsed,
        })


def _image_snapshot(frame: dict, config: GroundingConfig):
    from PIL import Image

    path = Path(frame["image_path"])
    if not path.is_file():
        raise FileNotFoundError("Original frame image is unavailable")
    if path.stat().st_size > config.max_image_bytes:
        raise ValueError("Original frame exceeds the configured byte budget")
    content = path.read_bytes()
    if len(content) > config.max_image_bytes:
        raise ValueError("Original frame exceeds the configured byte budget")
    with Image.open(BytesIO(content)) as opened:
        if opened.size != (frame["width"], frame["height"]):
            raise ValueError("Original image dimensions disagree with the frame metadata")
        if opened.width * opened.height > config.max_original_pixels:
            raise ValueError("Original frame exceeds the configured pixel budget")
        opened.load()
        image = opened.convert("RGB")
    return image, {"sha256": sha256(content).hexdigest(), "bytes": len(content),
                   "width": image.width, "height": image.height}


def _materialize_image(messages: list[dict], image):
    """Use the validated pixel snapshot rather than reopening a mutable path."""
    replaced = 0
    normalized = []
    for message in messages:
        item = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "image":
                    parts.append({"type": "image", "image": image})
                    replaced += 1
                else:
                    parts.append(dict(part))
            item["content"] = parts
        normalized.append(item)
    if replaced != 1:
        raise ValueError("The grounding prompt must contain exactly the current frame image")
    return normalized


def _invalid_value_receipt(value) -> dict[str, str]:
    """Describe malformed backend data without putting non-JSON objects in receipts."""
    try:
        representation = repr(value)
    except Exception:
        representation = "<repr unavailable>"
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}"[:256],
        "repr": representation[:4096],
    }


class GroundingModel:
    """Shared application adapter. Heavy backend creation is deferred to interpret."""

    def __init__(self, config: GroundingConfig | Mapping | None = None, *, backend: GroundingBackend | None = None):
        self.config = config if isinstance(config, GroundingConfig) else GroundingConfig.from_mapping(
            {} if config is None else config
        )
        self._backend = backend
        self._injected = backend is not None
        self._loaded_adapter = None
        self.last_receipt: dict = {}
        self._interpret_lock = threading.RLock()

    def interpret(self, request: Mapping) -> dict:
        with self._interpret_lock:
            return self._interpret(request)

    def interpret_with_receipt(self, request: Mapping) -> tuple[dict, dict]:
        """Return a per-call outcome/receipt snapshot atomically.

        last_receipt remains the latest shared record for older callers. Use
        this method when another thread may interpret a request immediately
        after this call returns.
        """
        with self._interpret_lock:
            outcome = self.interpret(request)
            return deepcopy(outcome), deepcopy(self.last_receipt)

    def close(self):
        """Release an owned backend before another model is loaded."""
        with self._interpret_lock:
            if self._backend is not None and not self._injected:
                close = getattr(self._backend, "close", None)
                if close is not None:
                    close()
                self._backend = None
                self._loaded_adapter = None

    def _interpret(self, request: Mapping) -> dict:
        from .contracts import error_outcome, validate_request
        from .formatting import build_messages, parse_model_output

        started = time.monotonic()
        request_id = request.get("request_id", "") if isinstance(request, Mapping) else ""
        if not isinstance(request_id, str):
            request_id = ""
        provenance = {
            "backend_kind": ("injected_test_backend" if self._injected else
                             "transformers_local_streamed" if self.config.runtime == "streamed" else
                             "transformers_local"),
            "model_id": self.config.model_id, "revision": self.config.revision,
            "requested_condition": "fine_tuned" if self.config.adapter_dir else "pretrained",
            "injected": self._injected,
            "generation": {"do_sample": False, "max_new_tokens": self.config.max_new_tokens,
                           "max_input_tokens": self.config.max_input_tokens, "max_time_s": self.config.max_time_s,
                           "min_pixels": self.config.min_pixels, "max_pixels": self.config.max_pixels},
        }
        self.last_receipt = {"request_id": request_id, "status": "pending", "provenance": provenance}
        phase = "request"
        try:
            validated = validate_request(request, allow_annotated=self.config.allow_annotated)
            frame = validated["scene"]["frame"]
            request_id = validated["request_id"]
            provenance["request_sha256"] = sha256(
                json.dumps(validated, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            self.last_receipt["request"] = deepcopy(validated)
            phase = "image"
            image, image_record = _image_snapshot(frame, self.config)
            provenance["image"] = image_record
            scene_provenance = validated["scene"].get("provenance", {})
            if "image_sha256" in scene_provenance:
                declared = scene_provenance["image_sha256"]
                if not isinstance(declared, str) or not _SHA.fullmatch(declared.lower()):
                    raise ValueError("Declared original image SHA256 is invalid")
                provenance["declared_image_sha256"] = declared.lower()
                if image_record["sha256"] != declared.lower():
                    raise ValueError("Original image bytes changed from the declared scene binding")
            phase = "artifact"
            adapter = verify_adapter(self.config)
            provenance["adapter"] = adapter
            if self._backend is not None and not self._injected and adapter != self._loaded_adapter:
                raise ArtifactError("Adapter files changed after model loading; construct a fresh model")
            phase = "request"
            if self.config.observation_format == 'json':
                messages = build_messages(validated, task="interpret", allow_annotated=self.config.allow_annotated)
            else:
                provenance['observation_format'] = self.config.observation_format
                messages = build_messages(validated, task="interpret", allow_annotated=self.config.allow_annotated,
                                          observation_format=self.config.observation_format)
            self.last_receipt["messages"] = deepcopy(messages)
            # Bind prompt text and the exact original image content without
            # serializing a PIL object or rereading the image during generation.
            provenance["messages_sha256"] = sha256(
                json.dumps(messages, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            messages = _materialize_image(messages, image)
            phase = "load"
            if self._backend is None:
                if self.config.runtime == "streamed":
                    from .streamed import StreamedProcessBackend
                    self._backend = StreamedProcessBackend(self.config)
                else:
                    self._backend = _TransformersBackend(self.config)
                self._loaded_adapter = adapter
                if verify_adapter(self.config) != adapter:
                    self._backend = None
                    raise ArtifactError("Adapter changed while the model was loading")
            phase = "generate"
            response = self._backend.generate(messages, config=self.config)
            phase = "parse"
            if isinstance(response, BackendResponse):
                raw = response.text
            else:
                raw = response
            if not isinstance(raw, str):
                self.last_receipt["invalid_raw_response"] = _invalid_value_receipt(raw)
                raise ValueError("Grounding backend must return text")
            self.last_receipt["raw_text"] = raw
            if isinstance(response, BackendResponse) and response.details is not None:
                try:
                    if not isinstance(response.details, dict):
                        raise ValueError("Backend details must be an object")
                    json.dumps(response.details, allow_nan=False)
                except (TypeError, ValueError, OverflowError) as error:
                    self.last_receipt["invalid_backend_details"] = _invalid_value_receipt(response.details)
                    raise ValueError("Grounding backend returned non-JSON provenance") from error
                provenance["backend_details"] = deepcopy(response.details)
            result = parse_model_output(raw, frame["width"], frame["height"])
            provenance["elapsed_s"] = time.monotonic() - started
            self.last_receipt.update(
                status="completed", result=deepcopy(result), provenance=deepcopy(provenance)
            )
            return {"ok": True, "request_id": request_id, "result": result, "provenance": provenance}
        except Exception as error:
            # The isolated worker closes itself on a resource failure or deadline. A later,
            # independent request must be allowed to create a fresh worker with the same config.
            if (phase == "generate" and not self._injected and self.config.runtime == "streamed"
                    and self._backend is not None and isinstance(error, (ResourceError, TimeoutError))):
                failed_backend = self._backend
                self._backend = self._loaded_adapter = None
                try:
                    failed_backend.close()
                except Exception as cleanup_error:
                    self.last_receipt["worker_cleanup_error"] = str(cleanup_error)[:1000]
            if isinstance(error, ArtifactError):
                code = "artifact_invalid"
            elif isinstance(error, (TimeoutError,)):
                code = "timeout"
            elif isinstance(error, (ImportError, ModuleNotFoundError)):
                code = "dependency_unavailable"
            elif isinstance(error, ResourceError) or "out of memory" in str(error).lower():
                code = "resource_unavailable"
            elif phase == "request":
                code = "invalid_request"
            elif phase == "image":
                code = "invalid_image"
            elif phase == "parse":
                code = "invalid_model_output"
            elif phase == "load" and isinstance(error, (OSError, FileNotFoundError)):
                code = "checkpoint_unavailable"
            else:
                code = "generation_failed"
            provenance["elapsed_s"] = time.monotonic() - started
            full_message = f"{type(error).__name__}: {error}"
            message = full_message[:4096]
            self.last_receipt.update(
                status="error", error={"code": code, "message": message},
                full_error_message=full_message, provenance=deepcopy(provenance),
            )
            return error_outcome(code, message, request_id, provenance=provenance)
