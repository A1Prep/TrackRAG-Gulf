"""Explicit, offline low-memory runtime for the pinned prequantized 8B VLM.

Importing this module is lightweight. Torch and model dependencies are loaded
only when constructing StreamedRuntime. No remote answerer, coordinate rule or
smaller model is substituted if the local checkpoint is unavailable.
"""

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import threading
import time
import weakref

from .model import (BASE_MODEL_ID, BASE_REVISION, ArtifactError, BackendResponse,
                    GroundingConfig, ResourceError, verify_adapter)


STREAMED_MODEL_ID = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
STREAMED_REVISION = "10d842936325905934ea24e64325d774693f166e"
WEIGHT_SHA256 = {
    "model-00001-of-00002.safetensors": "4b1b6eb8c99b0ae3f6c793f4c8d5113c5246c98ca65f5d54d50dc6465efc3a70",
    "model-00002-of-00002.safetensors": "1ff0a1be4f302b91aec0695fbf225398b14c5b2e9d50e6bedd8c2bf015ec8d00",
}
METADATA_SHA256 = {
    "config.json": "4292e1056469d772b0574e28e0fc5cb648ccb624fdef4c340f88e6588aa0410a",
    "model.safetensors.index.json": "6192698e7d4dab819026cf5aa58f9d3b1c7b5272c6329ebeb0dac0cc9668356e",
}
_LOAD_LOCK = threading.RLock()


def lora_target(suffix_layers):
    if type(suffix_layers) is not int or not 1 <= suffix_layers <= 8:
        raise ValueError("The streamed profile supports 1 to 8 trainable suffix layers")
    indices = "|".join(str(index) for index in range(36 - suffix_layers, 36))
    return rf".*language_model\.layers\.({indices})\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"


def checkpoint_identity(model_dir, *, verify_weights=True):
    """Bind local weight bytes to this distribution's published immutable files."""
    directory = Path(model_dir).expanduser().resolve()
    if not directory.is_dir() or not (directory / "config.json").is_file():
        raise FileNotFoundError("Pinned local 8B checkpoint directory is unavailable")
    files = {}
    for name, expected in {**WEIGHT_SHA256, **METADATA_SHA256}.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ArtifactError(f"Missing or linked pinned checkpoint shard: {name}")
        actual = None
        if verify_weights or name in METADATA_SHA256:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b""):
                    digest.update(block)
            actual = digest.hexdigest()
            if actual != expected:
                raise ArtifactError(f"Pinned checkpoint digest mismatch: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": actual,
                       "expected_sha256": expected}
    return {"model_id": STREAMED_MODEL_ID, "revision": STREAMED_REVISION,
            "verified_weights": verify_weights, "verified_metadata": True,
            "files": files, "path": str(directory)}


@contextmanager
def _readonly_loader(modeling_utils, reader):
    """Keep the HF loader override and async-loading option inside this load."""
    with _LOAD_LOCK:
        previous = modeling_utils.safe_open
        old_env = os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")
        try:
            modeling_utils.safe_open = reader
            os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
            yield
        finally:
            modeling_utils.safe_open = previous
            if old_env is None:
                os.environ.pop("HF_DEACTIVATE_ASYNC_LOAD", None)
            else:
                os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = old_env


def _tensor_leaves(value, torch):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensor_leaves(child, torch)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensor_leaves(child, torch)


class StreamedRuntime:
    """Native image-conditioned 8B model with an immutable streamed prefix.

training_lora=None loads pretrained inference or config.adapter_dir. For new
training, pass {"rank": 8, "alpha": 16, "dropout": 0.0}. Only attention LoRA in
the configured last 1..8 decoder layers receives gradients. The remaining base
    must not be initialized, merged, upcast or optimized after mapping. Manual
    training forwards/backward must be exclusive; use execution_lock if sharing
    this object with another caller. Application inference uses an isolated
    worker so a native driver/loader fault cannot terminate the parent app.
"""

    def __init__(self, config: GroundingConfig, *, training_lora=None, verify_weights=True):
        self.execution_lock = threading.RLock()
        if config.runtime != "streamed":
            raise ValueError("Select the streamed runtime explicitly")
        if (config.model_id, config.revision) != (STREAMED_MODEL_ID, STREAMED_REVISION):
            raise ArtifactError("Streamed runtime requires its separately pinned prequantized 8B distribution")
        if not config.model_dir:
            raise FileNotFoundError("Streamed runtime needs a local model_dir")
        if not config.local_files_only:
            raise ValueError("Streamed runtime is offline; prepare checkpoint files separately")
        if config.quantization != "nf4" or config.dtype not in ("auto", "bfloat16"):
            raise ValueError("Streamed runtime uses the checkpoint's NF4 decoder and BF16 compute")
        if not 65536 <= config.min_pixels <= config.max_pixels <= 262144:
            raise ResourceError("Streamed image budgets must lie between 65536 and 262144 pixels")
        self.identity = checkpoint_identity(config.model_dir, verify_weights=verify_weights)
        self.config = config
        self.lora_target_modules = lora_target(config.suffix_layers)
        self.training_lora = training_lora
        if training_lora is not None and (not isinstance(training_lora, dict)
                or set(training_lora) - {"rank", "alpha", "dropout"}):
            raise ValueError("training_lora supports rank, alpha and dropout only")
        if config.adapter_dir:
            verify_adapter(config)

        import torch
        from accelerate import cpu_offload_with_hook
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
        import transformers.modeling_utils as modeling_utils
        from .streamed_edges import CPUBackedHead, CPUEmbedding
        from .streamed_weights import ReadOnlySafeOpen, preserve_frozen_cpu_views

        self.torch = torch
        if not torch.cuda.is_available() or config.device == "cpu":
            raise ResourceError("The streamed 8B profile requires CUDA")
        self.device = torch.device("cuda:0" if config.device in ("auto", "cuda") else config.device)
        if not torch.cuda.is_bf16_supported():
            raise ResourceError("The streamed 8B profile requires BF16-capable CUDA")
        self.processor = AutoProcessor.from_pretrained(
            BASE_MODEL_ID, revision=BASE_REVISION, cache_dir=str(config.resolved_cache_dir()),
            local_files_only=True, trust_remote_code=False,
            min_pixels=config.min_pixels, max_pixels=config.max_pixels)
        checkpoint_config = AutoConfig.from_pretrained(config.model_dir, local_files_only=True,
                                                       trust_remote_code=False)
        checkpoint_config.quantization_config["llm_int8_skip_modules"].append("model.visual")
        with _readonly_loader(modeling_utils, ReadOnlySafeOpen):
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_dir, config=checkpoint_config, local_files_only=True,
                trust_remote_code=False, dtype=torch.bfloat16,
                device_map={"": "cpu"}, attn_implementation="sdpa")
        if hasattr(model, "hf_device_map"):
            delattr(model, "hf_device_map")
        if (model.config.model_type != "qwen3_vl"
                or model.config.text_config.hidden_size != 4096
                or len(model.model.language_model.layers) != 36):
            raise ArtifactError("Unexpected architecture for the pinned 8B checkpoint")
        model.requires_grad_(False)
        model.config.use_cache = False
        if config.adapter_dir:
            adapter_dir = Path(config.adapter_dir).expanduser().resolve()
            lora = LoraConfig.from_pretrained(adapter_dir, local_files_only=True)
            if (lora.base_model_name_or_path != STREAMED_MODEL_ID
                    or lora.revision != STREAMED_REVISION or lora.target_modules != self.lora_target_modules
                    or lora.bias != "none" or lora.modules_to_save):
                raise ArtifactError("Adapter base or trainable modules differ from the streamed profile")
            _validate_lora_initialization(lora)
            lora.inference_mode = training_lora is None
            model = get_peft_model(model, lora)
            result = set_peft_model_state_dict(model, load_file(adapter_dir / "adapter_model.safetensors"))
            if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
                raise ArtifactError("Missing or unexpected learned adapter tensors")
        elif training_lora is not None:
            rank, alpha, dropout = (training_lora.get("rank", 8), training_lora.get("alpha", 16),
                                    training_lora.get("dropout", 0.0))
            if (type(rank) is not int or not 1 <= rank <= 32
                    or type(alpha) is not int or not 1 <= alpha <= 256
                    or isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1):
                raise ValueError("Invalid streamed LoRA rank, alpha or dropout")
            model = get_peft_model(model, LoraConfig(
                r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=self.lora_target_modules,
                bias="none", task_type="CAUSAL_LM"))

        self.model = model
        base = model.get_base_model() if hasattr(model, "peft_config") else model
        text = base.model.language_model
        first = 36 - config.suffix_layers
        self.effective_resident_layers = (config.suffix_layers if training_lora is not None
                                          else config.inference_resident_layers)
        resident_start = 36 - self.effective_resident_layers
        for layer in text.layers[resident_start:]:
            layer.to(self.device)
        text.norm.to(self.device)
        text.rotary_emb.to(self.device)
        text.embed_tokens = CPUEmbedding(text.embed_tokens, self.device)
        base.lm_head = CPUBackedHead(base.lm_head)
        trainables = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if training_lora is None:
            model.requires_grad_(False)
            trainables = []
        if any("lora_" not in name or parameter.device.type != "cuda" for name, parameter in trainables):
            raise RuntimeError("Only resident CUDA LoRA parameters may be trainable")
        if training_lora is not None and not trainables:
            raise RuntimeError("Training profile created no trainable LoRA parameters")
        if any(parameter.requires_grad for parameter in text.layers[:first].parameters()):
            raise RuntimeError("Frozen prefix acquired trainable parameters")
        if any(parameter.requires_grad for parameter in base.model.visual.parameters()):
            raise RuntimeError("Frozen visual encoder acquired trainable parameters")
        self.trainable_parameter_count = sum(parameter.numel() for _, parameter in trainables)
        self._offload_hooks, self._handles = [], []
        previous = None
        for module in [base.model.visual, *text.layers[:resident_start]]:
            _, previous = cpu_offload_with_hook(module, execution_device=self.device,
                                                prev_module_hook=previous)
            preserve_frozen_cpu_views(module, previous)
            self._offload_hooks.append(previous)
            self._handles.append(module.register_forward_hook(self._post_forward(previous)))

        def enter_suffix(module, args, kwargs):
            previous.offload()
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is not None and hidden.requires_grad:
                raise RuntimeError("Frozen prefix unexpectedly created an autograd graph")
        self._handles.append(text.layers[first].register_forward_pre_hook(enter_suffix, with_kwargs=True))
        self.model.eval()
        self.dtype = "bfloat16"
        self.provenance = {
            "runtime": "streamed", "model_id": STREAMED_MODEL_ID, "revision": STREAMED_REVISION,
            "processor_model_id": BASE_MODEL_ID, "processor_revision": BASE_REVISION,
            "suffix_layers": config.suffix_layers, "lora_target_modules": self.lora_target_modules,
            "effective_resident_layers": self.effective_resident_layers,
            "image_processing": {"min_pixels": config.min_pixels, "max_pixels": config.max_pixels},
            "trainable_parameters": self.trainable_parameter_count,
            "weight_identity": self.identity, "head_chunk_rows": 4096,
            "frozen_prefix_storage": "read_only_file_backed",
            "coordinate_system": "normalized_0_1000_image_edges",
        }

    def _post_forward(self, offload_hook):
        def after_forward(module, inputs, output):
            if any(value.requires_grad for value in _tensor_leaves(output, self.torch)):
                raise RuntimeError("Cannot evict a frozen module with a live autograd output")
            offload_hook.offload()
            self.torch.cuda.empty_cache()
        return after_forward

    def prepare_inputs(self, messages, *, config=None):
        from PIL.Image import Image
        config = self.config if config is None else config
        images = [part.get("image") for message in messages if isinstance(message.get("content"), list)
                  for part in message["content"] if part.get("type") == "image"]
        if len(images) != 1 or not isinstance(images[0], Image):
            raise ValueError("Processor inputs require exactly one materialized current-frame image")
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[rendered], images=images, padding=True,
                                truncation=False, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        if int(inputs["input_ids"].shape[-1]) > config.max_input_tokens:
            raise ResourceError("Grounding input exceeds its token budget; input was not truncated")
        return inputs

    def generate(self, messages, *, config=None, output_contract=None):
        with self.execution_lock:
            return self._generate(messages, config=config, output_contract=output_contract)

    def _generate(self, messages, *, config=None, output_contract=None):
        if self.model is None:
            raise ResourceError("This streamed model has been closed")
        config = self.config if config is None else config
        started = time.monotonic()
        inputs = self.prepare_inputs(messages, config=config)
        tokens = int(inputs["input_ids"].shape[-1])
        inputs = inputs.to(self.device)
        generation_options = {}
        constraint_details = None
        if output_contract is not None:
            from .structured import INTERPRET_CONTRACT, InterpretationConstraint
            if output_contract != INTERPRET_CONTRACT:
                raise ValueError("Unknown streamed output contract")
            constraint = getattr(self, "_interpretation_constraint", None)
            if constraint is None:
                head = self.model.get_output_embeddings()
                constraint = InterpretationConstraint(self.processor.tokenizer, head.out_features)
                self._interpretation_constraint = constraint
            callback, constraint_details = constraint.for_request()
            generation_options["prefix_allowed_tokens_fn"] = callback
        # Trie construction, parser creation and image preparation all consume
        # this request's cooperative deadline, as well as the parent watchdog.
        remaining = config.max_time_s - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Grounding preparation exceeded the generation budget")
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, do_sample=False, use_cache=True,
                max_new_tokens=config.max_new_tokens, max_time=remaining, logits_to_keep=1,
                **generation_options)
        elapsed = time.monotonic() - started
        if elapsed >= config.max_time_s:
            raise TimeoutError("Generation exceeded its cooperative time budget")
        tail = generated[:, tokens:]
        texts = self.processor.batch_decode(tail, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
        if len(texts) != 1:
            raise ValueError("Expected exactly one grounding generation")
        packages = {}
        for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
        details = {
            **self.provenance, "device": str(self.device), "dtype": self.dtype,
            "packages": packages, "input_tokens": tokens, "output_tokens": int(tail.shape[-1]),
            "generation_elapsed_s": elapsed,
        }
        if constraint_details is not None:
            details["structured_decoding"] = constraint_details
        return BackendResponse(texts[0], details)

    def close(self):
        """Release this owned model; callers must first release external tensors."""
        with self.execution_lock:
            self._close()

    def _close(self):
        for handle in getattr(self, "_handles", []):
            handle.remove()
        self._handles = []
        for hook in getattr(self, "_offload_hooks", []):
            hook.offload()
            hook.remove()
        self._offload_hooks = []
        self.model = None
        self._interpretation_constraint = None
        import gc
        gc.collect()
        self.torch.cuda.empty_cache()


def _validate_lora_initialization(lora):
    """Reject adapter modes that can rewrite the read-only frozen base."""
    if (type(lora.r) is not int or not 1 <= lora.r <= 32
            or type(lora.lora_alpha) is not int or not 1 <= lora.lora_alpha <= 256
            or isinstance(lora.lora_dropout, bool)
            or not isinstance(lora.lora_dropout, (int, float)) or not 0 <= lora.lora_dropout < 1
            or type(lora.init_lora_weights) is not bool
            or getattr(lora, "use_dora", False) or getattr(lora, "use_rslora", False)
            or getattr(lora, "rank_pattern", None) or getattr(lora, "alpha_pattern", None)
            or getattr(lora, "layer_replication", None) or getattr(lora, "target_parameters", None)
            or getattr(lora, "lora_bias", False) or getattr(lora, "fan_in_fan_out", False)
            or getattr(lora, "alora_invocation_tokens", None) or getattr(lora, "use_qalora", False)):
        raise ArtifactError("Adapter uses an unsupported initialization or LoRA layout for a read-only base")


def _worker_error(error, phase):
    return {"kind": "error", "phase": phase, "error_type": type(error).__name__, "message": str(error)[:4096]}


def _streamed_worker(connection, config_dict):
    """Spawn target: all native model state remains outside the application."""
    runtime = None
    try:
        try:
            checkpoint_identity(config_dict["model_dir"], verify_weights=False)
            import torch
            torch.set_num_threads(2)
            if torch.cuda.is_available():
                device = "cuda:0" if config_dict["device"] in ("auto", "cuda") else config_dict["device"]
                if device.startswith("cuda"):
                    torch.cuda.set_per_process_memory_fraction(0.90, device)
            runtime = StreamedRuntime(GroundingConfig.from_mapping(config_dict))
            connection.send({"kind": "ready"})
        except Exception as error:
            connection.send(_worker_error(error, "load"))
            return
        while True:
            try:
                command = connection.recv()
            except EOFError:
                break
            if command.get("kind") == "close":
                break
            if command.get("kind") != "generate":
                connection.send(_worker_error(ValueError("Unknown local worker command"), "generate"))
                continue
            try:
                contract = command.get("contract", "interpret_v1")
                if contract not in ("interpret_v1", "landmark_v1"):
                    raise ValueError("Unknown worker output contract")
                response = runtime.generate(command["messages"],
                    output_contract="interpret_v1" if contract == "interpret_v1" else None)
                connection.send({"kind": "result", "text": response.text,
                                 "details": {**(response.details or {}), "worker_pid": os.getpid(),
                                             "native_fault_isolation": "spawn_process"}})
            except Exception as error:
                connection.send(_worker_error(error, "generate"))
    except (EOFError, BrokenPipeError, OSError):
        # The parent may have reached its deadline or already exited.
        pass
    finally:
        if runtime is not None:
            runtime.close()
        connection.close()


def _terminate_worker(process, connection):
    """Bound cleanup without waiting indefinitely for a native GPU call."""
    try:
        connection.close()
    except OSError:
        pass
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=1)
    if not process.is_alive():
        process.join(timeout=0)


class _WorkerCleanup:
    """One serialized termination path shared by deadline, close and finalizer."""
    def __init__(self, process, connection):
        self.process, self.connection = process, connection
        self.lock = threading.Lock()
        self.done = False

    def __call__(self):
        with self.lock:
            if self.done:
                return
            try:
                _terminate_worker(self.process, self.connection)
            finally:
                self.done = True


def _raise_worker_error(response):
    name, message = response.get("error_type"), response.get("message", "Local model worker failed")
    classes = {"ArtifactError": ArtifactError, "ResourceError": ResourceError,
               "TimeoutError": TimeoutError, "FileNotFoundError": FileNotFoundError,
               "OSError": OSError, "ImportError": ImportError, "ModuleNotFoundError": ModuleNotFoundError,
               "ValueError": ValueError}
    exception = classes.get(name, RuntimeError)
    if (name == "MemoryError" or any(term in message.lower() for term in
            ("out of memory", "paging file", "cannot allocate memory", "not enough memory"))):
        exception = ResourceError
    raise exception(f"Local worker {response.get('phase', 'generate')} failure: {name}: {message}")


class StreamedProcessBackend:
    """Persistent local inference worker with native-exit and timeout outcomes.

Construction is invoked lazily by GroundingModel during its load phase. Initial
checkpoint loading has a separate 120-second startup limit. Each generation has
the configured max_time_s deadline, enforced in the parent even if the child is
blocked inside native code. Prompts contain the already materialized image.
"""

    def __init__(self, config, *, startup_timeout_s=120.0, context=None):
        from dataclasses import asdict
        import multiprocessing

        self.config = config
        self._lock = threading.RLock()
        self._closed = False
        if not isinstance(startup_timeout_s, (float, int)) or not 0 < startup_timeout_s <= 300:
            raise ValueError("Worker startup timeout must be between 0 and 300 seconds")
        # Missing local checkpoints are diagnosed before starting another Python.
        if not config.model_dir:
            raise FileNotFoundError("Streamed runtime needs a local model_dir")
        checkpoint_identity(config.model_dir, verify_weights=False)
        context = context or multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe(duplex=True)
        self._process = context.Process(target=_streamed_worker, args=(child, asdict(config)), daemon=True)
        try:
            self._process.start()
            child.close()
            self._shutdown = _WorkerCleanup(self._process, self._connection)
            self._finalizer = weakref.finalize(self, self._shutdown)
            response = self._receive(float(startup_timeout_s))
            if response.get("kind") == "error":
                _raise_worker_error(response)
            if response.get("kind") != "ready":
                raise ResourceError("Invalid startup response from local model worker")
        except Exception:
            child.close()
            self.close()
            raise

    def _receive(self, timeout):
        if not self._connection.poll(timeout):
            if not self._process.is_alive():
                raise ResourceError(f"Local model worker exited natively (exit code {self._process.exitcode})")
            raise TimeoutError("Local model worker exceeded its parent-enforced deadline")
        try:
            response = self._connection.recv()
        except (EOFError, OSError) as error:
            raise ResourceError(f"Local model worker exited without a response (exit code {self._process.exitcode})") from error
        if not isinstance(response, dict):
            raise ResourceError("Local model worker returned an invalid transport response")
        return response

    def generate(self, messages, *, config=None, contract="interpret_v1"):
        with self._lock:
            if contract not in ("interpret_v1", "landmark_v1"):
                raise ValueError("Unknown worker output contract")
            if self._closed:
                raise ResourceError("Local model worker is closed; construct a fresh model")
            if config is not None and config != self.config:
                raise ArtifactError("Local worker configuration differs from its loaded model")
            timed_out = threading.Event()
            def expire():
                timed_out.set()
                self._shutdown()
            timer = threading.Timer(self.config.max_time_s, expire)
            timer.daemon = True
            timer.start()
            try:
                command = {"kind": "generate", "messages": messages}
                if contract != "interpret_v1":
                    command["contract"] = contract
                self._connection.send(command)
                response = self._receive(self.config.max_time_s)
                if timed_out.is_set():
                    raise TimeoutError("Local model worker exceeded its parent-enforced deadline")
                if response.get("kind") == "error":
                    _raise_worker_error(response)
                if response.get("kind") != "result":
                    raise ResourceError("Local model worker returned an invalid result envelope")
                return BackendResponse(response.get("text"), response.get("details"))
            except (TimeoutError, ResourceError) as error:
                self.close()
                if timed_out.is_set() and not isinstance(error, TimeoutError):
                    raise TimeoutError("Local model worker exceeded its parent-enforced deadline") from error
                raise
            except (EOFError, BrokenPipeError, OSError) as error:
                self.close()
                if timed_out.is_set():
                    raise TimeoutError("Local model worker exceeded its parent-enforced deadline") from error
                raise ResourceError("Local model worker transport closed without a usable result") from error
            finally:
                timer.cancel()
                if timer.is_alive() and threading.current_thread() is not timer:
                    timer.join(timeout=3.5)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if hasattr(self, "_finalizer"):
                self._finalizer()
            elif hasattr(self, "_process") and self._process.pid is not None:
                _terminate_worker(self._process, self._connection)
            elif hasattr(self, "_connection"):
                self._connection.close()
