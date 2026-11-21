"""Software and CPU math checks; these are not real 8B quality measurements."""

from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from trackrag.dropoff.model import ArtifactError, BackendResponse, GroundingConfig, GroundingModel, ResourceError
from trackrag.dropoff import streamed


def config(**changes):
    return GroundingConfig(model_id=streamed.STREAMED_MODEL_ID, revision=streamed.STREAMED_REVISION,
        runtime="streamed", model_dir="unused", quantization="nf4", dtype="bfloat16",
        min_pixels=65536, max_pixels=262144, **changes)


def test_import_is_lazy():
    result = subprocess.run([sys.executable, "-c",
        "import sys; import trackrag.dropoff.streamed; "
        "assert not any(x in sys.modules for x in ['torch','transformers','peft','bitsandbytes'])"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("changes", [
    {"runtime": "other"}, {"suffix_layers": True}, {"suffix_layers": 9},
    {"inference_resident_layers": -1}, {"inference_resident_layers": 25},
    {"inference_resident_layers": False}, {"model_dir": ""},
])
def test_invalid_streamed_config(changes):
    with pytest.raises(ValueError):
        replace(config(), **changes)


@pytest.mark.parametrize("resident", [0, 2, 7])
def test_frozen_inference_can_stream_adapted_suffix_layers(resident):
    cfg = replace(config(), inference_resident_layers=resident)
    assert cfg.inference_resident_layers == resident and cfg.suffix_layers == 8


def test_target_covers_only_requested_suffix():
    import re
    pattern = re.compile(streamed.lora_target(8))
    assert pattern.fullmatch("base_model.model.model.language_model.layers.28.self_attn.q_proj")
    assert pattern.fullmatch("model.language_model.layers.35.self_attn.o_proj")
    assert not pattern.fullmatch("model.language_model.layers.27.self_attn.q_proj")
    assert not pattern.fullmatch("model.visual.blocks.35.attn.q_proj")
    assert not pattern.fullmatch("model.language_model.layers.35.mlp.gate_proj")


def test_loader_patch_restored_after_failure(monkeypatch):
    original, temporary = object(), object()
    module = SimpleNamespace(safe_open=original)
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "old")
    with pytest.raises(RuntimeError):
        with streamed._readonly_loader(module, temporary):
            assert module.safe_open is temporary
            assert __import__("os").environ["HF_DEACTIVATE_ASYNC_LOAD"] == "1"
            raise RuntimeError("load failed")
    assert module.safe_open is original
    assert __import__("os").environ["HF_DEACTIVATE_ASYNC_LOAD"] == "old"


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    data = {"one.safetensors": b"weights", "config.json": b"{}", "model.safetensors.index.json": b"index"}
    for name, content in data.items():
        (tmp_path / name).write_bytes(content)
    monkeypatch.setattr(streamed, "WEIGHT_SHA256", {"one.safetensors": hashlib.sha256(b"weights").hexdigest()})
    monkeypatch.setattr(streamed, "METADATA_SHA256", {name: hashlib.sha256(data[name]).hexdigest()
                        for name in ("config.json", "model.safetensors.index.json")})
    return tmp_path


def test_checkpoint_binds_metadata_even_without_bulk_hashing(checkpoint):
    result = streamed.checkpoint_identity(checkpoint, verify_weights=False)
    assert result["verified_metadata"] is True
    assert result["files"]["one.safetensors"]["sha256"] is None
    (checkpoint / "config.json").write_text('{"changed":true}')
    with pytest.raises(ArtifactError, match="digest mismatch"):
        streamed.checkpoint_identity(checkpoint, verify_weights=False)


def test_checkpoint_rejects_changed_weights(checkpoint):
    (checkpoint / "one.safetensors").write_bytes(b"changed")
    with pytest.raises(ArtifactError, match="digest mismatch"):
        streamed.checkpoint_identity(checkpoint)


def test_no_smaller_or_other_distribution_fallback():
    with pytest.raises(ArtifactError, match="separately pinned"):
        streamed.StreamedRuntime(replace(config(), model_id="other/8b"))


@pytest.mark.parametrize("changes", [
    {"init_lora_weights": "pissa"}, {"use_dora": True}, {"r": 10000},
    {"rank_pattern": {"q_proj": 200}}, {"fan_in_fan_out": True},
    {"lora_dropout": float("nan")}, {"lora_alpha": 100000},
])
def test_lora_base_mutation_or_unbounded_layout_rejected(changes):
    fields = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "init_lora_weights": True}
    fields.update(changes)
    with pytest.raises(ArtifactError):
        streamed._validate_lora_initialization(SimpleNamespace(**fields))


class FakeConnection:
    def __init__(self, replies=()):
        self.replies, self.sent, self.closed = list(replies), [], False
    def poll(self, timeout): return bool(self.replies)
    def recv(self):
        value = self.replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return value
    def send(self, value): self.sent.append(value)
    def close(self): self.closed = True


class FakeProcess:
    def __init__(self):
        self.alive, self.pid, self.exitcode, self.terminations = False, None, None, 0
    def start(self): self.alive, self.pid = True, 123
    def is_alive(self): return self.alive
    def join(self, timeout): pass
    def terminate(self):
        self.alive, self.exitcode = False, -15
        self.terminations += 1
    def kill(self): self.alive, self.exitcode = False, -9


class FakeContext:
    def __init__(self, replies):
        self.parent, self.child, self.process = FakeConnection(replies), FakeConnection(), FakeProcess()
    def Pipe(self, duplex): return self.parent, self.child
    def Process(self, **kwargs):
        assert kwargs["daemon"] is True
        self.target = kwargs["target"]
        return self.process


def test_worker_returns_raw_text_without_inventing_fields(checkpoint):
    context = FakeContext([{"kind": "ready"}, {"kind": "result", "text": '{"destination_point_xy":[1,2]}',
                                                "details": {"native_fault_isolation": "spawn_process"}}])
    backend = streamed.StreamedProcessBackend(replace(config(), model_dir=str(checkpoint)), context=context)
    try:
        response = backend.generate([{"role": "user", "content": "instruction"}])
        assert response.text == '{"destination_point_xy":[1,2]}'
        assert "status" not in response.text
    finally:
        backend.close()
    assert context.process.terminations == 1


def test_generation_timeout_stops_owned_worker(checkpoint):
    context = FakeContext([{"kind": "ready"}])
    backend = streamed.StreamedProcessBackend(replace(config(), model_dir=str(checkpoint)), context=context)
    with pytest.raises(TimeoutError, match="parent-enforced"):
        backend.generate([])
    assert context.process.terminations == 1
    with pytest.raises(ResourceError, match="closed"):
        backend.generate([])


def test_native_child_exit_becomes_operational_error(checkpoint):
    context = FakeContext([{"kind": "ready"}, EOFError()])
    backend = streamed.StreamedProcessBackend(replace(config(), model_dir=str(checkpoint)), context=context)
    with pytest.raises(ResourceError, match="without a response"):
        backend.generate([])
    assert context.process.terminations == 1


def test_watchdog_also_interrupts_a_blocked_receive(checkpoint):
    import threading
    class BlockedConnection(FakeConnection):
        def __init__(self):
            super().__init__([{"kind": "ready"}])
            self.interrupted = threading.Event()
        def poll(self, timeout): return True
        def recv(self):
            if self.replies:
                return super().recv()
            assert self.interrupted.wait(2), "watchdog failed to interrupt transport"
            raise EOFError()
        def close(self):
            super().close()
            self.interrupted.set()
    context = FakeContext([])
    context.parent = BlockedConnection()
    backend = streamed.StreamedProcessBackend(
        replace(config(), model_dir=str(checkpoint), max_time_s=0.02), context=context)
    with pytest.raises(TimeoutError, match="parent-enforced"):
        backend.generate([])
    assert not context.process.is_alive()
    assert context.process.terminations == 1


def test_deadline_and_close_share_one_termination_path():
    from concurrent.futures import ThreadPoolExecutor
    process, connection = FakeProcess(), FakeConnection()
    process.start()
    cleanup = streamed._WorkerCleanup(process, connection)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: cleanup(), range(16)))
    cleanup()
    assert process.terminations == 1 and connection.closed


def test_child_load_failure_preserves_checkpoint_error(checkpoint):
    context = FakeContext([{"kind": "error", "phase": "load", "error_type": "FileNotFoundError", "message": "missing processor"}])
    with pytest.raises(FileNotFoundError, match="missing processor"):
        streamed.StreamedProcessBackend(replace(config(), model_dir=str(checkpoint)), context=context)
    assert context.process.terminations == 1


def test_real_spawn_reports_child_load_error_without_loading_torch(checkpoint):
    # Parent fixture digests are intentionally not present in a freshly spawned
    # child. The child must fail its pinned-file check before importing Torch.
    with pytest.raises(ArtifactError, match="pinned checkpoint shard"):
        streamed.StreamedProcessBackend(replace(config(), model_dir=str(checkpoint)), startup_timeout_s=15)


def _abrupt_worker_exit(connection, config_dict):
    import os
    os._exit(37)


class AbruptExitContext:
    def __init__(self):
        import multiprocessing
        self.context = multiprocessing.get_context("spawn")
    def Pipe(self, duplex): return self.context.Pipe(duplex=duplex)
    def Process(self, **kwargs):
        kwargs["target"] = _abrupt_worker_exit
        return self.context.Process(**kwargs)


def test_real_spawn_abrupt_exit_does_not_terminate_parent(checkpoint):
    with pytest.raises(ResourceError, match="without a response"):
        streamed.StreamedProcessBackend(replace(config(), model_dir=str(checkpoint)),
            startup_timeout_s=15, context=AbruptExitContext())


def test_paging_limit_is_a_resource_error():
    with pytest.raises(ResourceError, match="paging file"):
        streamed._raise_worker_error({"error_type": "OSError", "phase": "load", "message": "The paging file is too small"})


def _request(tmp_path):
    from PIL import Image
    path = tmp_path / "frame.png"
    Image.new("RGB", (10, 20)).save(path)
    return {"request_id": "request", "text": "Drop me near the entrance", "input_mode": "typed", "dialogue": [],
            "scene": {"source": "predicted", "tracks": [], "frame": {"recording_id": "recording", "frame_id": 3,
              "timestamp_s": 0.3, "width": 10, "height": 20, "image_path": str(path)}}}


def test_grounding_model_selects_real_streamed_condition_and_original_pixels(tmp_path, monkeypatch):
    calls = []
    class Backend:
        def __init__(self, cfg): calls.append(cfg)
        def generate(self, messages, *, config):
            assert messages[1]["content"][0]["image"].size == (10, 20)
            return BackendResponse(json.dumps({"status": "proposed", "destination_point_xy": [250, 500],
                "destination_box_xyxy": [200, 400, 300, 600], "clarification_question": None}))
    monkeypatch.setattr(streamed, "StreamedProcessBackend", Backend)
    outcome = GroundingModel(config()).interpret(_request(tmp_path))
    assert outcome["ok"] is True
    assert outcome["result"]["destination_point_xy"] == [2.5, 10]
    assert outcome["provenance"]["backend_kind"] == "transformers_local_streamed"
    assert outcome["provenance"]["injected"] is False
    assert len(calls) == 1


def test_partial_region_response_is_not_upgraded_to_semantic_success(tmp_path, monkeypatch):
    class Backend:
        def __init__(self, cfg): pass
        def generate(self, messages, *, config):
            return BackendResponse('{"destination_point_xy":[200,400],"destination_box_xyxy":[100,300,300,500]}')
    monkeypatch.setattr(streamed, "StreamedProcessBackend", Backend)
    model = GroundingModel(config())
    outcome = model.interpret(_request(tmp_path))
    assert outcome["ok"] is False and outcome["error"]["code"] == "invalid_model_output"
    assert "destination_point_xy" in model.last_receipt["raw_text"]


def test_future_observation_rejected_before_worker_load(tmp_path, monkeypatch):
    def forbidden(config): raise AssertionError("worker must not start")
    monkeypatch.setattr(streamed, "StreamedProcessBackend", forbidden)
    request = _request(tmp_path)
    request["scene"]["tracks"] = [{"recording_id": "recording", "frame_id": 4, "source": "predicted",
        "track_id": 1, "class_name": "car", "bbox_xyxy": [0, 0, 5, 5]}]
    outcome = GroundingModel(config()).interpret(request)
    assert outcome["ok"] is False and outcome["error"]["code"] == "invalid_request"


def test_concurrent_first_calls_create_one_backend_and_preserve_request_ids(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    count = []
    ready = threading.Barrier(2)
    class Backend:
        def __init__(self, cfg):
            count.append(cfg)
            time.sleep(0.02)
        def generate(self, messages, *, config):
            return BackendResponse(json.dumps({"status": "proposed", "destination_point_xy": [250, 500],
                "destination_box_xyxy": [200, 400, 300, 600], "clarification_question": None}))
        def close(self): count.append("closed")
    monkeypatch.setattr(streamed, "StreamedProcessBackend", Backend)
    model = GroundingModel(config())
    request = _request(tmp_path)
    def call(index):
        value = {**request, "request_id": f"request-{index}"}
        ready.wait()
        return model.interpret_with_receipt(value)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(call, [1, 2]))
    assert len(count) == 1
    assert all(value[0]["ok"] is True for value in outcomes)
    assert {value[0]["request_id"] for value in outcomes} == {"request-1", "request-2"}
    assert all(value[0]["request_id"] == value[1]["request_id"] for value in outcomes)
    before = __import__("copy").deepcopy(model.last_receipt)
    for outcome, receipt in outcomes:
        receipt["request"]["text"] = "caller mutation"
        outcome["result"]["destination_point_xy"][0] = -999
    assert model.last_receipt == before
    model.close()
    assert count[-1] == "closed"


def test_prediction_export_uses_atomic_receipt_when_available(tmp_path):
    from tools.dropoff_predict import predict
    class Adapter:
        def __init__(self, config): self.last_receipt = {"wrong": "shared legacy receipt"}
        def interpret(self, request): raise AssertionError("Atomic interface should be used")
        def interpret_with_receipt(self, request):
            return {"ok": True, "request_id": request["request_id"]}, {"request_id": request["request_id"], "atomic": True}
    request_file = tmp_path / "requests.jsonl"
    request_file.write_text(json.dumps(_request(tmp_path)) + "\n")
    output = tmp_path / "predictions"
    predict(request_file, GroundingConfig(), output, adapter_factory=Adapter)
    receipt = json.loads((output / "receipts.jsonl").read_text())
    assert receipt == {"request_id": "request", "atomic": True}


def test_reader_dtypes_lifetime_and_empty_tensor(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file
    from trackrag.dropoff.streamed_weights import ReadOnlySafeOpen
    expected = {"bf16": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
                "f32": torch.arange(7, dtype=torch.float32), "u8": torch.arange(32, dtype=torch.uint8),
                "empty": torch.empty((0, 3))}
    path = tmp_path / "tiny.safetensors"
    save_file(expected, path, metadata={"format": "pt"})
    with ReadOnlySafeOpen(path) as reader:
        values = {name: reader.get_tensor(name) for name in reader.keys()}
        assert reader.metadata() == {"format": "pt"}
    del reader
    gc.collect()
    assert all(torch.equal(values[name], value) for name, value in expected.items())


@pytest.mark.parametrize("tensor", [
    {"dtype": "F32", "shape": [3], "data_offsets": [0, 4]},
    {"dtype": "F32", "shape": [1], "data_offsets": [-1, 3]},
    {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]},
    {"dtype": [], "shape": [1], "data_offsets": [0, 4]},
])
def test_reader_rejects_bad_bounds_before_exporting_views(tmp_path, tensor):
    pytest.importorskip("torch")
    from trackrag.dropoff.streamed_weights import ReadOnlySafeOpen
    header = json.dumps({"weight": tensor}).encode()
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
    with pytest.raises(ValueError):
        ReadOnlySafeOpen(path)


def test_restore_keeps_original_cpu_storage_without_copy():
    torch = pytest.importorskip("torch")
    from trackrag.dropoff.streamed_weights import preserve_frozen_cpu_views
    module = torch.nn.Linear(4, 3).requires_grad_(False)
    module.weight._is_hf_initialized = True
    old = module.weight.detach()
    hook = SimpleNamespace(offload=None)
    preserve_frozen_cpu_views(module, hook)
    module.weight = torch.nn.Parameter(torch.ones_like(old) * 99, requires_grad=False)
    hook.offload()
    assert module.weight.data_ptr() == old.data_ptr()
    assert module.weight._is_hf_initialized is True
    assert not module.weight.requires_grad


def test_nf4_restore_preserves_quant_state_and_hf_metadata():
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")
    from trackrag.dropoff.streamed_weights import preserve_frozen_cpu_views
    module = bnb.nn.Linear4bit(16, 16, bias=False, compute_dtype=torch.float32,
                               compress_statistics=True, quant_type="nf4").to("cpu")
    module.requires_grad_(False)
    module.weight._is_hf_initialized = True
    packed = module.weight.data.clone()
    code = module.weight.quant_state.code.clone()
    hook = SimpleNamespace(offload=None)
    preserve_frozen_cpu_views(module, hook)
    module.weight.data = torch.zeros_like(module.weight.data)
    module.weight.quant_state.code = torch.ones_like(module.weight.quant_state.code)
    hook.offload()
    assert module.weight._is_hf_initialized is True
    assert torch.equal(module.weight.data, packed)
    assert torch.equal(module.weight.quant_state.code, code)


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_streamed_head_forward_and_gradient_match_dense_cpu(dtype_name):
    torch = pytest.importorskip("torch")
    from trackrag.dropoff.streamed_edges import FrozenHeadFunction
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(42)
    weight = torch.randn(257, 37, dtype=dtype)
    x = torch.randn(2, 3, 37, dtype=dtype, requires_grad=True)
    dense = torch.nn.functional.linear(x, weight)
    grad = torch.randn_like(dense)
    expected = torch.autograd.grad(dense, x, grad)[0]
    y = x.detach().clone().requires_grad_(True)
    streamed_head = FrozenHeadFunction.apply(y, weight, 31)
    actual = torch.autograd.grad(streamed_head, y, grad)[0]
    tolerance = 2e-5 if dtype_name == "float32" else 0.1
    torch.testing.assert_close(streamed_head, dense, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert weight.grad is None
