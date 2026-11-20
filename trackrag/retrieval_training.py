"""Optional Task 4 LoRA training, with explicit synthetic smoke validation."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from time import perf_counter

from .retrieval_qwen import INPUT_VERSION, INSTRUCTION, PREFIX, SUFFIX, QwenConfig, QwenTextScorer, _query_text
from .retrieval_training_data import (
    canonical, digest, load_dataset, prepare_reviewed, synthetic_dataset, validate_dataset, write_json,
)


CHECKPOINT_SCHEMA = "trackrag.qwen-lora-checkpoint.v1"


@dataclass(frozen=True)
class TrainingConfig:
    """Small explicit defaults for proving machinery, not tuned hyperparameters."""

    max_steps: int = 2
    batch_size: int = 1
    learning_rate: float = 0.0001
    max_tokens: int = 512
    lora_rank: int = 4
    lora_alpha: int = 8
    last_layers: int = 1
    seed: int = 42
    max_grad_norm: float = 1.0

    def __post_init__(self):
        for name in ("max_steps", "batch_size", "lora_rank", "lora_alpha", "last_layers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_tokens) is not int or not 128 <= self.max_tokens <= 32768:
            raise ValueError("max_tokens must be an integer from 128 through 32768")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def input_policy():
    return {"input_version": INPUT_VERSION, "instruction": INSTRUCTION, "prefix": PREFIX, "suffix": SUFFIX}


def adapter_state_hash(model):
    """Hash only learned adapter tensors, independent of checkpoint serialization."""
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(model, save_embedding_layers=False)
    fingerprints = {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                           "sha256": sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
                    for name, tensor in sorted(state.items())}
    return digest(fingerprints)


def read_checkpoint(directory, *, allow_synthetic=False):
    """Verify a completed, hash-bound LoRA artifact before any base-model loading."""
    root = Path(directory)
    manifest = json.loads((root / "training.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA or manifest.get("status") != "completed":
        raise ValueError("Checkpoint is not a completed Task 4 LoRA run")
    kind = manifest.get("dataset_kind")
    if kind not in ("synthetic_smoke", "human_reviewed"):
        raise ValueError("Checkpoint has no valid dataset provenance")
    if kind == "synthetic_smoke" and not allow_synthetic:
        raise ValueError("Synthetic smoke checkpoint requires explicit allow_synthetic=True for testing")
    if manifest.get("input_policy") != input_policy():
        raise ValueError("Checkpoint input policy differs from the current reranker")
    config = QwenConfig(**manifest["base_config"])
    expected_files = {"adapter/adapter_model.safetensors", "adapter/adapter_config.json", "dataset.json", "training-log.jsonl"}
    if set(manifest.get("files_sha256", {})) != expected_files:
        raise ValueError("Checkpoint file inventory is incomplete")
    for name, expected in manifest["files_sha256"].items():
        if sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint file hash mismatch: {name}")
    dataset = load_dataset(root / "dataset.json")
    if dataset["kind"] != kind or dataset["dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("Checkpoint dataset binding mismatch")
    saved = json.loads((root / "adapter/adapter_config.json").read_text(encoding="utf-8"))
    if (saved.get("peft_type") != "LORA" or saved.get("base_model_name_or_path") != config.model_id
            or saved.get("revision") != config.revision or saved.get("bias") != "none"
            or set(saved.get("target_modules", [])) != {"q_proj", "v_proj"}
            or saved.get("layers_to_transform") != manifest["lora"]["layers"]):
        raise ValueError("LoRA config disagrees with the recorded base model and adapter policy")
    if not manifest.get("changed_tensors") or manifest["adapter_before_sha256"] == manifest["adapter_after_sha256"]:
        raise ValueError("Checkpoint does not demonstrate learned parameter updates")
    return manifest


def _encode(scorer, samples, limit):
    prompts = [f"<Instruct>: {INSTRUCTION}\n<Query>: {_query_text(sample['question'])}\n<Document>: {sample['document']}"
               for sample in samples]
    rows = scorer.tokenizer(prompts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    result = [scorer.prefix + row + scorer.suffix for row in rows]
    if any(len(row) > limit for row in result):
        raise ValueError("Training input exceeds the explicit token limit; no text was truncated")
    return result


def _probe_scores(scorer, samples):
    # These first two dev examples are only a save/reload probe, not a dev metric.
    return [scorer.score(sample["question"], [sample["document"]])[0] for sample in samples]


def train_adapter(bundle, output_dir, *, qwen_config=None, training_config=None,
                  _scorer_factory=QwenTextScorer):
    """Perform real LoRA updates, save safetensors, and verify fresh-base reload.

    The private factory permits tiny random Qwen models in regression tests.
    Normal CLI execution always uses the pinned downloaded pretrained weights.
    Test data is rejected before model loading. Loss uses only final no/yes logits,
    with binary relevance as its target, exactly matching inference normalization.
    """
    samples = validate_dataset(bundle)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Choose a new checkpoint directory")
    qwen_config = qwen_config if qwen_config is not None else QwenConfig()
    config = training_config if training_config is not None else TrainingConfig()
    if not isinstance(qwen_config, QwenConfig) or not isinstance(config, TrainingConfig):
        raise TypeError("Supply QwenConfig and TrainingConfig instances")
    try:
        import torch
        import peft
        from importlib.metadata import version
    except ImportError as error:
        raise RuntimeError("Install requirements-qwen-training.txt for LoRA training") from error
    torch.manual_seed(config.seed)
    scorer = _scorer_factory(qwen_config)
    scorer._load()
    # CPU float32 is the verified path; float32 is also explicit on CUDA rather
    # than silently applying an untested mixed-precision training policy.
    scorer.model.float()
    limit = min(config.max_tokens, qwen_config.max_tokens)
    encoded = _encode(scorer, samples, limit)
    train_positions = [number for number, sample in enumerate(samples) if sample["split"] == "train"]
    probes = [sample for sample in samples if sample["split"] == "dev"][:2]
    before_scores = _probe_scores(scorer, probes)
    layer_count = scorer.model.config.num_hidden_layers
    if config.last_layers > layer_count:
        raise ValueError("last_layers exceeds the model's layer count")
    layers = list(range(layer_count - config.last_layers, layer_count))
    lora = peft.LoraConfig(task_type=peft.TaskType.CAUSAL_LM, r=config.lora_rank,
                           lora_alpha=config.lora_alpha, lora_dropout=0.0, bias="none",
                           target_modules=["q_proj", "v_proj"], layers_to_transform=layers,
                           layers_pattern="layers", revision=qwen_config.revision)
    scorer.model = peft.get_peft_model(scorer.model, lora)
    scorer.model.peft_config["default"].base_model_name_or_path = qwen_config.model_id
    trainable = {name: parameter for name, parameter in scorer.model.named_parameters() if parameter.requires_grad}
    if not trainable or any(".lora_A." not in name and ".lora_B." not in name for name in trainable):
        raise RuntimeError("Only the declared LoRA tensors may be trainable")
    initial = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
    before_hash = adapter_state_hash(scorer.model)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=config.learning_rate, weight_decay=0.0)
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA, "status": "training", "dataset_kind": bundle["kind"],
        "training_state": "synthetic_smoke_adapter" if bundle["kind"] == "synthetic_smoke" else "human_reviewed_adapter",
        "dataset_sha256": bundle["dataset_sha256"], "split_manifest": bundle["split_manifest"],
        "base_config": asdict(qwen_config), "training_config": asdict(config), "input_policy": input_policy(),
        "objective": "Cross entropy over final [no, yes] logits; target is binary relevance",
        "training_dtype": "float32", "lora": {"rank": config.lora_rank, "alpha": config.lora_alpha,
                                                "target_modules": ["q_proj", "v_proj"], "layers": layers},
        "trainable_parameters": sum(parameter.numel() for parameter in trainable.values()),
        "trainable_tensor_names": sorted(trainable), "base_parameters_frozen": True,
        "train_examples": len(train_positions), "dev_examples": sum(sample["split"] == "dev" for sample in samples),
        "token_counts": [len(row) for row in encoded], "adapter_before_sha256": before_hash,
        "scorer_factory": _scorer_factory.__name__,
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
        "source_files_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("retrieval_qwen.py", "retrieval_training.py", "retrieval_training_data.py", "retrieval_qwen_trained.py")},
        "accuracy_metrics": None, "evaluation_status": "requires_independent_reviewed_evaluation",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix=".qwen-training-", dir=output.parent) as temporary:
        staging = Path(temporary) / "checkpoint"
        staging.mkdir()
        write_json(staging / "dataset.json", bundle)
        seen = set()
        try:
            order, cursor, generator = train_positions.copy(), 0, random.Random(config.seed)
            generator.shuffle(order)
            scorer.model.train()
            with (staging / "training-log.jsonl").open("x", encoding="utf-8") as log:
                for step in range(1, config.max_steps + 1):
                    chosen = []
                    for _ in range(config.batch_size):
                        if cursor == len(order):
                            generator.shuffle(order)
                            cursor = 0
                        chosen.append(order[cursor])
                        cursor += 1
                    seen.update(chosen)
                    inputs = scorer.tokenizer.pad({"input_ids": [encoded[number] for number in chosen]},
                                                  padding=True, return_tensors="pt")
                    inputs = {name: tensor.to(qwen_config.device) for name, tensor in inputs.items()}
                    targets = torch.tensor([samples[number]["relevance"] for number in chosen], device=qwen_config.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = scorer.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                    loss = torch.nn.functional.cross_entropy(logits[:, [scorer.no_id, scorer.yes_id]].float(), targets)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Training produced a nonfinite loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), config.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    row = {"step": step, "loss": loss.item(), "gradient_norm_before_clipping": norm.item(),
                           "sample_ids": [samples[number]["sample_id"] for number in chosen]}
                    log.write(canonical(row) + "\n")
                    log.flush()
                    print(f"LoRA step {step}/{config.max_steps}, loss={loss.item():.6f}", file=sys.stderr, flush=True)
            optimizer.zero_grad(set_to_none=True)
            after_hash = adapter_state_hash(scorer.model)
            changed = {name: (parameter.detach().cpu() - initial[name]).abs().max().item()
                       for name, parameter in trainable.items() if not torch.equal(parameter.detach().cpu(), initial[name])}
            if not changed or before_hash == after_hash:
                raise RuntimeError("Optimizer did not change any saved adapter tensors")
            scorer.model.eval().requires_grad_(False)
            after_scores = _probe_scores(scorer, probes)
            scorer.model.save_pretrained(staging / "adapter", safe_serialization=True, save_embedding_layers=False)
            manifest.update(status="completed", optimizer_steps=config.max_steps, unique_train_examples_seen=len(seen),
                            changed_tensors=changed, adapter_after_sha256=after_hash,
                            probe={"purpose": "Save/reload consistency only, not accuracy", "sample_ids": [sample["sample_id"] for sample in probes],
                                   "before_scores": before_scores, "after_scores": after_scores})
            manifest["files_sha256"] = {name: sha256((staging / name).read_bytes()).hexdigest() for name in (
                "adapter/adapter_model.safetensors", "adapter/adapter_config.json", "dataset.json", "training-log.jsonl")}
            write_json(staging / "training.json", manifest)
            from .retrieval_qwen_trained import TrainedQwenTextScorer

            reloaded = TrainedQwenTextScorer(staging, config=qwen_config, allow_synthetic=True,
                                            _base_factory=_scorer_factory)
            reloaded_scores = _probe_scores(reloaded, probes)
            difference = max(abs(left - right) for left, right in zip(after_scores, reloaded_scores))
            if difference > 1e-6:
                raise RuntimeError("Fresh checkpoint reload did not reproduce the saved model's scores")
            manifest["probe"].update(reloaded_scores=reloaded_scores, reload_max_absolute_difference=difference,
                                     reload_tolerance=1e-6, fresh_base_reload_verified=True)
        except Exception as error:
            manifest.update(status="error", error_type=type(error).__name__, error=str(error)[:2000])
            raise
        finally:
            manifest["elapsed_seconds_training_and_reload"] = perf_counter() - started
            write_json(staging / "training.json", manifest)
            if output.exists():
                raise FileExistsError("Checkpoint output was created concurrently")
            staging.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reviewed = commands.add_parser("prepare-reviewed", help="Prepare train/dev data using existing human-review checks")
    reviewed.add_argument("--review", type=Path, required=True)
    reviewed.add_argument("--labels", type=Path, required=True)
    reviewed.add_argument("--out", type=Path, required=True)
    reviewed.add_argument("--tracked-only", action="store_true", help="Exclude scene examples unsupported by this text model's track features")
    for name in ("smoke", "train"):
        command = commands.add_parser(name, help="Synthetic optimizer/save/reload proof" if name == "smoke" else "Train an explicitly identified dataset")
        if name == "train":
            command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--qwen-config", type=Path)
        command.add_argument("--training-config", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-reviewed":
            bundle = prepare_reviewed(args.review, args.labels, args.out, tracked_only=args.tracked_only)
            print(f"Saved reviewed training bundle with {len(bundle['examples'])} examples to {args.out}")
        else:
            bundle = synthetic_dataset() if args.command == "smoke" else load_dataset(args.dataset)
            qwen = QwenConfig(**json.loads(args.qwen_config.read_text(encoding="utf-8-sig"))) if args.qwen_config else QwenConfig()
            training = TrainingConfig(**json.loads(args.training_config.read_text(encoding="utf-8-sig"))) if args.training_config else TrainingConfig()
            result = train_adapter(bundle, args.out, qwen_config=qwen, training_config=training)
            print(f"Saved {result['training_state']} after {result['optimizer_steps']} steps to {args.out}")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
