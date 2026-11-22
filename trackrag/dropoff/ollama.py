"""Explicit local pretrained Qwen3.5 capacity route, separate from trained Qwen3-VL.

No weights are downloaded, no adapter is loaded, and no HF/Torch import is needed.
Use the same public scene/result contracts while retaining provider failures.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass, fields
from hashlib import sha256
import http.client
from importlib import metadata
from io import BytesIO
import json
import math
from pathlib import Path
import re
import socket
import threading
import time
from typing import Mapping
import urllib.error
import urllib.parse

from .contracts import error_outcome, validate_request
from .formatting import build_messages, parse_model_output


QWEN35_9B_DIGEST = "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
QWEN35_4B_DIGEST = "2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd"
SUPPORTED_MODELS = frozenset(("qwen3.5:9b", "qwen3.5:4b"))
MAX_HTTP_BYTES = 4 * 1024 * 1024


class LocalModelError(ValueError):
    """The installed/loaded provider model does not match the explicit pin."""


@dataclass(frozen=True)
class OllamaGroundingConfig:
    model_id: str = "qwen3.5:9b"
    model_digest: str = QWEN35_9B_DIGEST
    endpoint: str = "http://127.0.0.1:11434"
    allow_annotated: bool = False
    context_length: int = 8192
    max_new_tokens: int = 192
    max_prompt_utf8_bytes: int = 4096
    max_time_s: float = 180.0
    gpu_layers: int = 14
    batch_size: int = 16
    image_max_width: int = 672
    image_max_height: int = 384
    jpeg_quality: int = 92
    max_image_bytes: int = 32 * 1024 * 1024
    max_original_pixels: int = 48_000_000
    keep_alive_s: int = 120
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.model_id, str) or self.model_id not in SUPPORTED_MODELS:
            raise ValueError("Only the explicitly supported local Qwen3.5 tags are accepted")
        if not isinstance(self.model_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", self.model_digest):
            raise ValueError("model_digest must be an explicit lowercase SHA256 pin")
        if not isinstance(self.endpoint, str):
            raise ValueError("endpoint must be a literal loopback HTTP URL")
        url = urllib.parse.urlsplit(self.endpoint)
        try:
            valid_port = url.port is None or 1 <= url.port <= 65535
        except ValueError as error:
            raise ValueError("Invalid local provider port") from error
        if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "::1")
                or url.username is not None or url.password is not None
                or url.path not in ("", "/") or url.query or url.fragment or not valid_port):
            raise ValueError("endpoint must be HTTP on literal 127.0.0.1 or [::1], with no credentials/path/query")
        if type(self.allow_annotated) is not bool:
            raise ValueError("allow_annotated must be boolean")
        for name, lower, upper in (
            ("context_length", 2048, 16384), ("max_new_tokens", 1, 1024),
            ("max_prompt_utf8_bytes", 1, 12288), ("gpu_layers", 0, 64),
            ("batch_size", 1, 64), ("image_max_width", 32, 672),
            ("image_max_height", 32, 384), ("jpeg_quality", 50, 95),
            ("max_image_bytes", 1, 32 * 1024 * 1024),
            ("max_original_pixels", 1, 48_000_000), ("keep_alive_s", 1, 300),
            ("seed", 0, 2**31 - 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        # A conservative UTF8 bound leaves room for provider image/template tokens.
        # It is not an exact tokenizer or a claim of HF-equivalent image budgets.
        if self.max_prompt_utf8_bytes + 4096 > self.context_length:
            raise ValueError("context_length must reserve 4096 tokens beyond the text-byte budget")
        if (isinstance(self.max_time_s, bool) or not isinstance(self.max_time_s, (int, float))
                or not math.isfinite(self.max_time_s) or not 1 <= self.max_time_s <= 300):
            raise ValueError("max_time_s must be finite and between 1 and 300")

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, Mapping):
            raise ValueError("Local grounding configuration must be an object")
        extra = set(value) - {field.name for field in fields(cls)}
        if extra:
            raise ValueError(f"Unknown local pretrained configuration fields: {sorted(extra)}")
        return cls(**value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate provider JSON field: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"Nonfinite provider JSON: {value}")


def _request_json(config, path, payload, deadline):
    """Use one direct socket with an absolute deadline across headers and body.

    A socket read timeout alone restarts whenever the server drips another byte.
    The watchdog shuts down this operation's captured socket at the deadline, so
    slow headers, successful bodies and HTTP-error bodies cannot extend the call.
    http.client neither consults ambient proxies nor follows redirects.
    """
    if path not in ("/api/version", "/api/tags", "/api/show", "/api/chat", "/api/ps"):
        raise ValueError("Unknown local provider operation")
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    if body is not None and len(body) > MAX_HTTP_BYTES:
        raise ValueError("Local provider request exceeds the byte budget")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Local grounding request deadline reached")
    url = urllib.parse.urlsplit(config.endpoint)
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=remaining)
    response, watchdog = None, None
    expired = threading.Event()
    try:
        # Numeric loopback connect has the current remaining socket timeout.
        # Capture its socket before HTTPResponse can detach it on Connection:close.
        connection.connect()
        owned_socket = connection.sock
        if owned_socket is None:
            raise ConnectionError("Local provider connection has no socket")

        def expire():
            expired.set()
            try:
                owned_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            finally:
                owned_socket.close()

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Local grounding request deadline reached")
        owned_socket.settimeout(remaining)
        watchdog = threading.Timer(remaining, expire)
        watchdog.daemon = True
        watchdog.start()
        connection.request("GET" if body is None else "POST", path, body=body,
                           headers={"Content-Type": "application/json", "Connection": "close"})
        response = connection.getresponse()
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_HTTP_BYTES:
            raise ValueError("Local provider response exceeds the byte budget")
        status, reason, headers = response.status, response.reason, response.headers
        # Error bodies are completely bounded/deadlined here. The caller only
        # receives a BytesIO-backed HTTPError, never a still-streaming network body.
        limit = MAX_HTTP_BYTES if 200 <= status < 300 else 16384
        content = response.read(limit + 1)
    except Exception as error:
        if expired.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("Local grounding request deadline reached") from error
        raise
    finally:
        if watchdog is not None:
            watchdog.cancel()
        if response is not None:
            response.close()
        connection.close()
    if expired.is_set() or time.monotonic() >= deadline:
        raise TimeoutError("Local grounding request deadline reached")
    if not 200 <= status < 300:
        raise urllib.error.HTTPError(config.endpoint.rstrip("/") + path, status, reason,
                                     headers, BytesIO(content[:16384]))
    if len(content) > MAX_HTTP_BYTES:
        raise ValueError("Local provider response exceeds the byte budget")
    value = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    if not isinstance(value, dict):
        raise ValueError("Local provider response must be a JSON object")
    if time.monotonic() >= deadline:
        raise TimeoutError("Local grounding request deadline reached")
    return value


def _installed_model(config, tags):
    models = tags.get("models")
    if not isinstance(models, list):
        raise LocalModelError("Provider did not return an installed-model list")
    matches = [model for model in models if isinstance(model, dict) and model.get("name") == config.model_id]
    if len(matches) != 1:
        raise FileNotFoundError("Pinned local model is not installed exactly once; downloads are never automatic")
    digest = matches[0].get("digest")
    if not isinstance(digest, str) or digest.removeprefix("sha256:") != config.model_digest:
        raise LocalModelError("Installed model digest differs from the explicit configuration pin")
    return matches[0]


def _snapshot(frame, config):
    from PIL import Image

    with Path(frame["image_path"]).open("rb") as stream:
        content = stream.read(config.max_image_bytes + 1)
    if len(content) > config.max_image_bytes:
        raise ValueError("Original frame exceeds the image byte budget")
    with Image.open(BytesIO(content)) as opened:
        if opened.size != (frame["width"], frame["height"]):
            raise ValueError("Original image dimensions disagree with the frame metadata")
        if opened.width * opened.height > config.max_original_pixels:
            raise ValueError("Original frame exceeds the pixel budget")
        opened.load()
        image = opened.convert("RGB")
    original = {"sha256": sha256(content).hexdigest(), "bytes": len(content),
                "width": image.width, "height": image.height}
    try:
        scale = min(config.image_max_width / image.width, config.image_max_height / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        with image.resize(size, Image.Resampling.LANCZOS) as resized:
            buffer = BytesIO()
            resized.save(buffer, format="JPEG", quality=config.jpeg_quality)
            sent = buffer.getvalue()
    finally:
        image.close()
    sent_record = {"sha256": sha256(sent).hexdigest(), "bytes": len(sent),
                   "width": size[0], "height": size[1], "format": "JPEG",
                   "aspect_ratio_preserved": True, "resampling": "LANCZOS",
                   "jpeg_quality": config.jpeg_quality, "Pillow": metadata.version("Pillow")}
    return sent, original, sent_record


class OllamaGroundingModel:
    """The same interpret/receipt interface, explicitly a different pretrained model."""

    def __init__(self, config=None):
        self.config = config if isinstance(config, OllamaGroundingConfig) else OllamaGroundingConfig.from_mapping(
            {} if config is None else config)
        self.last_receipt = {}

    def interpret(self, request):
        started = time.monotonic()
        config = self.config
        deadline = started + config.max_time_s
        request_id = request.get("request_id", "") if isinstance(request, Mapping) else ""
        if not isinstance(request_id, str):
            request_id = ""
        provenance = {"backend_kind": "ollama_local_pretrained", "model_id": config.model_id,
                      "model_digest": config.model_digest, "endpoint": config.endpoint,
                      "requested_condition": "separate_pretrained_capacity_baseline",
                      "trained_adapter_used": False, "training_executed": False,
                      "matched_qwen3_vl_8b_control": False, "model_quality_measured": False,
                      "generation": {"context_length": config.context_length,
                                     "max_new_tokens": config.max_new_tokens, "do_sample": False,
                                     "gpu_layers": config.gpu_layers, "batch_size": config.batch_size,
                                     "max_time_s": config.max_time_s, "seed": config.seed},
                      "input_budget_policy": "UTF8_text_bound_with_4096_reserved_visual_template_tokens"}
        self.last_receipt = {"request_id": request_id, "status": "pending", "provenance": provenance}
        phase = "request"
        try:
            validated = validate_request(request, allow_annotated=config.allow_annotated)
            frame = validated["scene"]["frame"]
            request_id = validated["request_id"]
            self.last_receipt["request"] = deepcopy(validated)
            provenance["request_sha256"] = sha256(json.dumps(
                validated, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            phase = "image"
            sent, original, sent_record = _snapshot(frame, config)
            provenance.update(image=original, sent_image=sent_record)
            declared = validated["scene"].get("provenance", {}).get("image_sha256")
            if "image_sha256" in validated["scene"].get("provenance", {}):
                if (not isinstance(declared, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", declared)
                        or declared.lower() != original["sha256"]):
                    raise ValueError("Original image bytes differ from the declared scene binding")
            phase = "request"
            messages = []
            recorded_messages = []
            for message in build_messages(validated, allow_annotated=config.allow_annotated):
                if isinstance(message["content"], str):
                    converted = {"role": message["role"], "content": message["content"]}
                else:
                    text = "\n".join(part["text"] for part in message["content"] if part["type"] == "text")
                    converted = {"role": message["role"], "content": text,
                                 "images": [base64.b64encode(sent).decode("ascii")]}
                messages.append(converted)
                recorded = {key: value for key, value in converted.items() if key != "images"}
                if "images" in converted:
                    recorded["images"] = [sent_record]
                recorded_messages.append(recorded)
            text_bytes = sum(len(message["content"].encode("utf-8")) for message in messages)
            if text_bytes > config.max_prompt_utf8_bytes:
                raise ValueError("Prompt exceeds the conservative local text budget; nothing was truncated")
            provenance["prompt_utf8_bytes"] = text_bytes
            self.last_receipt["messages"] = deepcopy(recorded_messages)
            provenance["messages_sha256"] = sha256(json.dumps(
                recorded_messages, sort_keys=True, allow_nan=False).encode()).hexdigest()
            phase = "provider"
            version = _request_json(config, "/api/version", None, deadline)
            if not isinstance(version.get("version"), str) or len(version["version"]) > 128:
                raise LocalModelError("Local provider did not report its runtime version")
            provenance["ollama_version"] = version["version"]
            _installed_model(config, _request_json(config, "/api/tags", None, deadline))
            details = _request_json(config, "/api/show", {"model": config.model_id}, deadline)
            if not isinstance(details.get("capabilities"), list) or "vision" not in details["capabilities"]:
                raise LocalModelError("Pinned installed model does not declare vision support")
            provenance["model_details"] = details.get("details")
            phase = "generate"
            response = _request_json(config, "/api/chat", {
                "model": config.model_id, "messages": messages, "stream": False,
                "think": False, "format": "json", "keep_alive": config.keep_alive_s,
                "options": {"num_ctx": config.context_length, "num_predict": config.max_new_tokens,
                            "num_batch": config.batch_size, "num_gpu": config.gpu_layers,
                            "temperature": 0, "seed": config.seed, "use_mmap": True},
            }, deadline)
            self.last_receipt["provider_response"] = deepcopy(response)
            phase = "parse"
            if response.get("model") != config.model_id:
                raise LocalModelError("Generated response reports a different model tag")
            if response.get("done") is not True or response.get("done_reason") != "stop":
                raise ValueError("Provider generation was incomplete or truncated")
            count = response.get("prompt_eval_count")
            if type(count) is not int or not 0 < count < config.context_length - config.max_new_tokens:
                raise ValueError("Prompt count is invalid or reaches the context boundary; truncation cannot be excluded")
            output_count = response.get("eval_count")
            if type(output_count) is not int or not 0 < output_count <= config.max_new_tokens:
                raise ValueError("Provider output token count exceeds the declared budget")
            message = response.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant" or not isinstance(message.get("content"), str):
                raise ValueError("Provider returned no assistant text")
            self.last_receipt["raw_text"] = message["content"]
            result = parse_model_output(message["content"], frame["width"], frame["height"])
            phase = "provider"
            _installed_model(config, _request_json(config, "/api/tags", None, deadline))
            running = _request_json(config, "/api/ps", None, deadline).get("models")
            if not isinstance(running, list):
                raise LocalModelError("Provider did not report the loaded model identity")
            matches = [model for model in running if isinstance(model, dict) and model.get("name") == config.model_id]
            loaded_digest = matches[0].get("digest") if len(matches) == 1 else None
            if not isinstance(loaded_digest, str) or loaded_digest.removeprefix("sha256:") != config.model_digest:
                raise LocalModelError("Loaded model identity differs from the explicit pin or cannot be verified")
            actual_context = matches[0].get("context_length")
            if type(actual_context) is not int or actual_context < config.context_length:
                raise LocalModelError("Loaded context length is missing, invalid or below the requested budget")
            provenance["loaded_model"] = deepcopy(matches[0])
            provenance["actual_context_length"] = actual_context
            provenance["input_tokens"] = count
            provenance["output_tokens"] = output_count
            provenance["elapsed_s"] = time.monotonic() - started
            self.last_receipt.update(status="completed", result=deepcopy(result), provenance=deepcopy(provenance))
            return {"ok": True, "request_id": request_id, "result": result, "provenance": provenance}
        except Exception as error:
            provider_body = ""
            if isinstance(error, urllib.error.HTTPError):
                try:
                    provider_body = error.read(16384).decode("utf-8", errors="replace")
                except Exception:
                    provider_body = "Provider error body could not be read"
                self.last_receipt["provider_error"] = {"http_status": error.code, "body": provider_body}
            details = str(error) + " " + provider_body
            if isinstance(error, (TimeoutError, socket.timeout)) or (
                    isinstance(error, urllib.error.URLError) and isinstance(error.reason, (TimeoutError, socket.timeout))):
                code = "timeout"
            elif "out of memory" in details.lower() or "unable to allocate" in details.lower():
                code = "resource_unavailable"
            elif isinstance(error, LocalModelError):
                code = "artifact_invalid"
            elif phase == "request":
                code = "invalid_request"
            elif phase == "image":
                code = "invalid_image"
            elif phase == "parse":
                code = "invalid_model_output"
            elif isinstance(error, FileNotFoundError):
                code = "checkpoint_unavailable"
            elif isinstance(error, urllib.error.HTTPError):
                code = "provider_error"
            elif isinstance(error, (urllib.error.URLError, ConnectionError)):
                code = "provider_unavailable"
            else:
                code = "generation_failed"
            message = f"{type(error).__name__}: {details}".strip()[:4096]
            provenance["elapsed_s"] = time.monotonic() - started
            self.last_receipt.update(status="error", error={"code": code, "message": message},
                                     provenance=deepcopy(provenance))
            return error_outcome(code, message, request_id, provenance=provenance)
