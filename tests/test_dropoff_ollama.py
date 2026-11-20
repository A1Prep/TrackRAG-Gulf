"""Local HTTP/scene boundary checks with generated images and mocked model replies.

These software checks do not execute Qwen or measure grounding quality.
"""

import base64
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error

from PIL import Image
import pytest

from tools import dropoff_predict
from trackrag.dropoff import ollama
from trackrag.dropoff.model import GroundingConfig
from trackrag.dropoff.ollama import (
    OllamaGroundingConfig, OllamaGroundingModel, QWEN35_9B_DIGEST,
)


def proposed():
    return json.dumps({"status": "proposed", "destination_point_xy": [500, 600],
                       "destination_box_xyxy": [300, 400, 700, 800], "clarification_question": None})


@pytest.fixture
def request_row(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (100, 50), (70, 80, 90)).save(path)
    return {"request_id": "local-request", "scene": {
        "frame": {"recording_id": "video-real-id-fixture", "frame_id": 10,
                  "timestamp_s": 0.9, "width": 100, "height": 50, "image_path": str(path)},
        "source": "predicted", "tracks": [{
            "recording_id": "video-real-id-fixture", "frame_id": 10, "track_id": 3,
            "class_name": "Car", "bbox_xyxy": [10, 20, 30, 40], "source": "predicted"}],
        "provenance": {"image_sha256": sha256(path.read_bytes()).hexdigest(),
                       "reference_note_that_must_not_be_prompted": "private-reference-fixture"}},
        "text": "Drop me off beside the car.", "input_mode": "typed", "dialogue": []}


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.digest = QWEN35_9B_DIGEST
        self.reply = proposed()
        self.chat_error = None
        self.chat_changes = {}
        self.loaded_digest = QWEN35_9B_DIGEST
        self.capabilities = ["completion", "vision"]
        self.on_tags = None

    def __call__(self, config, path, payload, deadline):
        self.calls.append((path, deepcopy(payload)))
        assert deadline > time.monotonic()
        if path == "/api/version":
            return {"version": "fixture-runtime"}
        if path == "/api/tags":
            if self.on_tags:
                self.on_tags()
            return {"models": [{"name": config.model_id, "digest": self.digest}]}
        if path == "/api/show":
            return {"capabilities": self.capabilities, "details": {"family": "fixture-qwen35"}}
        if path == "/api/ps":
            return {"models": [{"name": config.model_id, "digest": self.loaded_digest,
                                "size_vram": 12345, "context_length": config.context_length}]}
        if path == "/api/chat":
            if self.chat_error:
                raise self.chat_error
            value = {"model": config.model_id, "done": True, "done_reason": "stop",
                     "prompt_eval_count": 1000, "eval_count": 60,
                     "message": {"role": "assistant", "content": self.reply}}
            value.update(self.chat_changes)
            return value
        raise AssertionError(path)


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr(ollama, "_request_json", fake)
    return fake


def test_import_and_construction_are_torch_free_and_lazy():
    script = ("import sys; from trackrag.dropoff.ollama import OllamaGroundingModel; "
              "m=OllamaGroundingModel(); "
              "assert not any(n in sys.modules for n in ('torch','transformers','peft')); "
              "assert not m.last_receipt; print('lazy-local')")
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lazy-local"


@pytest.mark.parametrize("changes", [
    {"model_id": "unknown-vlm"}, {"model_id": []}, {"model_digest": "main"},
    {"endpoint": "https://127.0.0.1:11434"}, {"endpoint": "http://example.com"},
    {"endpoint": "http://localhost:11434"}, {"endpoint": "http://127.0.0.1:11434/api"},
    {"endpoint": "http://user@127.0.0.1:11434"}, {"endpoint": "http://127.0.0.1:11434?x=1"},
    {"context_length": 2048}, {"gpu_layers": True}, {"batch_size": 1024},
    {"max_time_s": float("nan")}, {"keep_alive_s": 0},
    {"adapter_dir": "must-not-load-an-adapter"}, {"local_files_only": False},
])
def test_config_refuses_unsafe_ambiguous_or_trained_route_settings(changes):
    with pytest.raises(ValueError):
        OllamaGroundingConfig.from_mapping(changes)


def test_real_route_provenance_and_original_coordinate_conversion(request_row, provider):
    original = deepcopy(request_row)
    model = OllamaGroundingModel()
    outcome = model.interpret(request_row)
    assert outcome["ok"] is True
    assert outcome["result"]["destination_point_xy"] == pytest.approx([50, 30])
    assert outcome["result"]["destination_box_xyxy"] == pytest.approx([30, 20, 70, 40])
    assert request_row == original
    provenance = outcome["provenance"]
    assert provenance["backend_kind"] == "ollama_local_pretrained"
    assert provenance["model_digest"] == QWEN35_9B_DIGEST
    assert provenance["trained_adapter_used"] is False
    assert provenance["matched_qwen3_vl_8b_control"] is False
    assert provenance["model_quality_measured"] is False
    assert provenance["loaded_model"]["digest"] == QWEN35_9B_DIGEST
    assert provenance["loaded_model"]["size_vram"] == 12345
    assert model.last_receipt["raw_text"] == proposed()
    payload = next(value for path, value in provider.calls if path == "/api/chat")
    assert payload["options"]["num_gpu"] == 14
    assert payload["options"]["num_batch"] == 16
    assert payload["options"]["num_ctx"] == 8192
    assert payload["stream"] is False and payload["think"] is False
    images = [b64 for message in payload["messages"] for b64 in message.get("images", [])]
    assert len(images) == 1
    sent = base64.b64decode(images[0])
    assert sha256(sent).hexdigest() == provenance["sent_image"]["sha256"]
    with Image.open(BytesIO(sent)) as image:
        assert image.size == (672, 336)
    assert "private-reference-fixture" not in json.dumps(payload["messages"])
    # Receipts identify the image without repeating a large base64 payload.
    assert images[0] not in json.dumps(model.last_receipt)


def test_consumed_snapshot_is_not_reopened_after_provider_calls(request_row, provider):
    path = Path(request_row["scene"]["frame"]["image_path"])
    before = sha256(path.read_bytes()).hexdigest()
    provider.on_tags = lambda: Image.new("RGB", (100, 50), "red").save(path)
    outcome = OllamaGroundingModel().interpret(request_row)
    assert outcome["ok"] is True
    assert outcome["provenance"]["image"]["sha256"] == before
    assert sha256(path.read_bytes()).hexdigest() != before
    payload = next(value for route, value in provider.calls if route == "/api/chat")
    sent = next(base64.b64decode(message["images"][0]) for message in payload["messages"] if "images" in message)
    with Image.open(BytesIO(sent)) as image:
        assert image.getpixel((0, 0)) == pytest.approx((70, 80, 90), abs=3)


@pytest.mark.parametrize("mutation", ("future", "foreign", "gold", "dialogue", "hash", "dimensions", "too_long"))
def test_invalid_inputs_fail_before_provider_access(request_row, provider, mutation):
    row = deepcopy(request_row)
    if mutation == "future":
        row["scene"]["tracks"][0]["frame_id"] = 11
    elif mutation == "foreign":
        row["scene"]["tracks"][0]["recording_id"] = "other"
    elif mutation == "gold":
        row["target"] = {"destination_point_xy": [50, 30]}
    elif mutation == "dialogue":
        row["dialogue"] = [{"role": "user", "text": "Here", "recording_id": "other", "frame_id": 10}]
    elif mutation == "hash":
        row["scene"]["provenance"]["image_sha256"] = "0" * 64
    elif mutation == "dimensions":
        row["scene"]["frame"]["width"] = 101
    else:
        row["text"] = "a" * 7000
    outcome = OllamaGroundingModel().interpret(row)
    assert outcome["ok"] is False and "result" not in outcome
    assert outcome["error"]["code"] in ("invalid_request", "invalid_image")
    assert provider.calls == []


@pytest.mark.parametrize("mutation", ("installed_pin", "loaded_pin", "no_vision", "missing"))
def test_model_identity_and_capability_fail_closed(request_row, provider, monkeypatch, mutation):
    if mutation == "installed_pin":
        provider.digest = "0" * 64
    elif mutation == "loaded_pin":
        provider.loaded_digest = "0" * 64
    elif mutation == "no_vision":
        provider.capabilities = ["completion"]
    else:
        real = provider.__call__
        monkeypatch.setattr(ollama, "_request_json", lambda c, p, v, d:
                            {"models": []} if p == "/api/tags" else real(c, p, v, d))
    model = OllamaGroundingModel()
    outcome = model.interpret(request_row)
    assert outcome["ok"] is False and "result" not in outcome
    assert outcome["error"]["code"] == ("checkpoint_unavailable" if mutation == "missing" else "artifact_invalid")
    if mutation in ("installed_pin", "no_vision", "missing"):
        assert not any(path == "/api/chat" for path, _ in provider.calls)


@pytest.mark.parametrize("changes", [
    {"done": False}, {"done_reason": "length"}, {"model": "different-model"},
    {"prompt_eval_count": 8000}, {"prompt_eval_count": True}, {"eval_count": 193},
    {"message": {"role": "assistant", "content": "not JSON"}},
])
def test_bad_or_truncated_provider_output_is_not_a_semantic_result(request_row, provider, changes):
    provider.chat_changes = changes
    model = OllamaGroundingModel()
    outcome = model.interpret(request_row)
    assert outcome["ok"] is False and "result" not in outcome
    assert outcome["error"]["code"] in ("invalid_model_output", "artifact_invalid")
    assert "provider_response" in model.last_receipt


@pytest.mark.parametrize("actual_context", ("missing", None, True, "8192", 0, 8191))
def test_loaded_context_cannot_silently_reduce_requested_budget(request_row, provider, monkeypatch, actual_context):
    real = provider.__call__
    def changed_context(config, path, payload, deadline):
        value = real(config, path, payload, deadline)
        if path == "/api/ps":
            if actual_context == "missing":
                value["models"][0].pop("context_length")
            else:
                value["models"][0]["context_length"] = actual_context
        return value
    monkeypatch.setattr(ollama, "_request_json", changed_context)
    model = OllamaGroundingModel()
    outcome = model.interpret(request_row)
    assert outcome["ok"] is False and "result" not in outcome
    assert outcome["error"]["code"] == "artifact_invalid"
    assert model.last_receipt["raw_text"] == proposed()


def test_larger_actual_context_is_recorded_separately(request_row, provider, monkeypatch):
    real = provider.__call__
    def changed_context(config, path, payload, deadline):
        value = real(config, path, payload, deadline)
        if path == "/api/ps":
            value["models"][0]["context_length"] = 16384
        return value
    monkeypatch.setattr(ollama, "_request_json", changed_context)
    outcome = OllamaGroundingModel().interpret(request_row)
    assert outcome["ok"] is True
    assert outcome["provenance"]["actual_context_length"] == 16384
    assert outcome["provenance"]["generation"]["context_length"] == 8192


@pytest.mark.parametrize("error,code", [
    (TimeoutError("fixture deadline"), "timeout"),
    (urllib.error.URLError(ConnectionRefusedError("fixture offline")), "provider_unavailable"),
    (urllib.error.HTTPError("http://127.0.0.1:11434/api/chat", 500, "fixture OOM", {},
                            BytesIO(b'{"error":"out of memory during startup"}')), "resource_unavailable"),
    (urllib.error.HTTPError("http://127.0.0.1:11434/api/chat", 429, "fixture busy", {},
                            BytesIO(b'{"error":"busy"}')), "provider_error"),
])
def test_operational_failures_cannot_be_valid_abstentions(request_row, provider, error, code):
    provider.chat_error = error
    model = OllamaGroundingModel()
    outcome = model.interpret(request_row)
    assert outcome["ok"] is False and outcome["error"]["code"] == code
    assert "result" not in outcome
    assert model.last_receipt["status"] == "error"


def test_local_route_uses_normal_prediction_artifacts_without_dropping_errors(tmp_path, request_row, provider):
    other = deepcopy(request_row)
    other["request_id"] = "local-request-2"
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps(request_row) + "\n" + json.dumps(other) + "\n", encoding="utf-8")
    calls = 0
    real = provider.__call__
    def second_request_fails(config, path, payload, deadline):
        nonlocal calls
        if path == "/api/chat":
            calls += 1
            if calls == 2:
                raise TimeoutError("fixture second request failure")
        return real(config, path, payload, deadline)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ollama, "_request_json", second_request_fails)
        out = tmp_path / "prediction"
        summary = dropoff_predict.predict(requests, OllamaGroundingConfig(), out,
                                          adapter_factory=OllamaGroundingModel)
    assert summary["requested_rows"] == 2
    assert summary["semantic_results"] == 1 and summary["operational_errors"] == 1
    outcomes = [json.loads(line) for line in (out / "model-outcomes.jsonl").read_text().splitlines()]
    receipts = [json.loads(line) for line in (out / "receipts.jsonl").read_text().splitlines()]
    assert len(outcomes) == len(receipts) == 2
    assert outcomes[1]["error"]["code"] == "timeout"
    assert all(item["provenance"]["backend_kind"] == "ollama_local_pretrained" for item in outcomes)


def test_cli_requires_explicit_local_backend_and_rejects_adapter_flags(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(asdict(OllamaGroundingConfig())), encoding="utf-8")
    args = ["predict", "--backend", "ollama", "--requests", str(tmp_path / "requests.jsonl"),
            "--config", str(config), "--out", str(tmp_path / "run")]
    observed = {}
    def capture(requests, settings, out, **kwargs):
        observed.update(config=settings, factory=kwargs.get("adapter_factory"))
        return {"operational_errors": 0}
    monkeypatch.setattr(dropoff_predict, "predict", capture)
    assert dropoff_predict.main(args) == 0
    assert isinstance(observed["config"], OllamaGroundingConfig)
    assert observed["factory"] is OllamaGroundingModel
    assert dropoff_predict.main(args + ["--adapter-dir", str(tmp_path)]) == 2
    assert "accepts no adapters" in capsys.readouterr().err
    # Passing this separate configuration to the unchanged default HF route fails.
    assert dropoff_predict.main([value for value in args if value not in ("--backend", "ollama")]) == 2


@pytest.fixture
def local_http_server():
    class Handler(BaseHTTPRequestHandler):
        reply = b'{"version":"local-fixture"}'
        mode = "normal"
        calls = []
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.__class__.calls.append(self.path)
            def drip(content):
                for byte in content:
                    self.wfile.write(bytes((byte,)))
                    self.wfile.flush()
                    time.sleep(0.04)
            try:
                if self.__class__.mode == "slow_headers":
                    drip(b"HTTP/1.0 200 OK\r\nContent-Length: "
                         + str(len(self.__class__.reply)).encode() + b"\r\n\r\n" + self.__class__.reply)
                elif self.__class__.mode == "redirect":
                    self.send_response(302)
                    self.send_header("Location", "/must-not-follow")
                    self.end_headers()
                else:
                    self.send_response(500 if self.__class__.mode in ("error", "slow_error_body") else 200)
                    self.send_header("Content-Length", str(ollama.MAX_HTTP_BYTES + 1)
                                     if self.__class__.mode == "oversized" else str(len(self.__class__.reply)))
                    self.end_headers()
                    if self.__class__.mode in ("slow_body", "slow_error_body"):
                        drip(self.__class__.reply)
                    elif self.__class__.mode != "oversized":
                        self.wfile.write(self.__class__.reply)
            except OSError:
                # A deadline test intentionally closes this socket mid-response.
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, Handler
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_http_transport_ignores_proxies_and_refuses_redirects(local_http_server, monkeypatch):
    server, handler = local_http_server
    config = OllamaGroundingConfig(endpoint=f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    assert ollama._request_json(config, "/api/version", None, time.monotonic() + 5)["version"] == "local-fixture"
    handler.mode = "redirect"
    with pytest.raises(urllib.error.HTTPError):
        ollama._request_json(config, "/api/version", None, time.monotonic() + 5)
    assert handler.calls == ["/api/version", "/api/version"]


def test_http_transport_bounds_bytes_and_rejects_ambiguous_json(local_http_server):
    server, handler = local_http_server
    config = OllamaGroundingConfig(endpoint=f"http://127.0.0.1:{server.server_port}")
    handler.mode = "oversized"
    with pytest.raises(ValueError, match="byte budget"):
        ollama._request_json(config, "/api/version", None, time.monotonic() + 5)
    handler.mode = "normal"
    handler.reply = b'{"version":"one","version":"two"}'
    with pytest.raises(ValueError, match="Duplicate"):
        ollama._request_json(config, "/api/version", None, time.monotonic() + 5)
    with pytest.raises(TimeoutError):
        ollama._request_json(config, "/api/version", None, time.monotonic() - 1)


@pytest.mark.parametrize("mode", ("slow_body", "slow_headers", "slow_error_body"))
def test_http_absolute_deadline_stops_dripping_headers_and_bodies(local_http_server, mode):
    server, handler = local_http_server
    config = OllamaGroundingConfig(endpoint=f"http://127.0.0.1:{server.server_port}")
    handler.mode = mode
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        ollama._request_json(config, "/api/version", None, started + 0.15)
    elapsed = time.monotonic() - started
    # Each response would otherwise take at least one second while each received
    # byte keeps an ordinary 150ms socket timeout alive. Allow scheduling margin.
    assert elapsed < 0.65


def test_http_error_body_is_bounded_and_buffered_before_exception(local_http_server):
    server, handler = local_http_server
    config = OllamaGroundingConfig(endpoint=f"http://127.0.0.1:{server.server_port}")
    handler.mode = "error"
    handler.reply = b'{"error":"fixture model unavailable"}'
    with pytest.raises(urllib.error.HTTPError) as raised:
        ollama._request_json(config, "/api/version", None, time.monotonic() + 2)
    assert raised.value.code == 500
    assert isinstance(raised.value.fp, BytesIO)
    assert raised.value.read() == handler.reply
