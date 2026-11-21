"""Optimizer/reload regression tests use a TINY RANDOM Qwen, never downloaded weights."""

from contextlib import redirect_stderr
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from trackrag.retrieval_training import TrainingConfig, _encode, read_checkpoint, train_adapter
from trackrag.retrieval_training_data import _seal, synthetic_dataset, write_json
from trackrag.retrieval_qwen import MODEL_ID, MODEL_REVISION, QwenConfig, QwenTextScorer, rank_evidence
from trackrag.retrieval_qwen_trained import TrainedQwenTextScorer, build_index as trained_index


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft"))


class TinyRandomScorer(QwenTextScorer):
    """Inject a deterministic random two-layer model; not a pretrained result."""

    instances = []

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        class Tokenizer:
            def __call__(self, prompts, **kwargs):
                return {"input_ids": [[3 + sum(text.encode("utf-8")[start:start + 24]) % 59
                                       for start in range(0, len(text.encode("utf-8")), 24)] for text in prompts]}

            def pad(self, batch, **kwargs):
                width = max(len(row) for row in batch["input_ids"])
                return {"input_ids": torch.tensor([[0] * (width - len(row)) + row for row in batch["input_ids"]]),
                        "attention_mask": torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in batch["input_ids"]])}

        torch.set_num_threads(1)
        torch.manual_seed(123)
        config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                             pad_token_id=0, bos_token_id=3, eos_token_id=4, attention_dropout=0.0)
        config._commit_hash = MODEL_REVISION
        config._name_or_path = MODEL_ID
        self.model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        self.torch, self.tokenizer = torch, Tokenizer()
        self.prefix, self.suffix, self.no_id, self.yes_id = [5], [6], 1, 2
        self.base_snapshots = [(parameter, parameter.detach().clone()) for parameter in self.model.parameters()]
        self.instances.append(self)


class TrainingConfigurationTests(unittest.TestCase):
    def test_invalid_training_values_are_rejected(self):
        for key, value in (("max_steps", 0), ("batch_size", True), ("lora_rank", 0),
                           ("learning_rate", float("inf")), ("max_grad_norm", -1),
                           ("seed", -1), ("max_tokens", 100), ("last_layers", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                TrainingConfig(**{key: value})


@unittest.skipUnless(AVAILABLE, "Optional Qwen/PEFT dependencies are not installed")
class TrainingTensorTests(unittest.TestCase):
    def setUp(self):
        TinyRandomScorer.instances.clear()

    def train(self, output):
        with redirect_stderr(io.StringIO()):
            return train_adapter(synthetic_dataset(), output, qwen_config=QwenConfig(local_files_only=True),
                                 training_config=TrainingConfig(lora_rank=2, lora_alpha=4, learning_rate=0.01),
                                 _scorer_factory=TinyRandomScorer)

    def test_updates_only_lora_saves_safetensors_and_reloads_exact_scores(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "checkpoint"
            result = self.train(output)
            self.assertEqual(result["training_state"], "synthetic_smoke_adapter")
            self.assertEqual(result["scorer_factory"], "TinyRandomScorer")
            self.assertEqual(result["optimizer_steps"], 2)
            self.assertNotEqual(result["adapter_before_sha256"], result["adapter_after_sha256"])
            self.assertTrue(result["changed_tensors"])
            self.assertTrue(result["probe"]["fresh_base_reload_verified"])
            self.assertEqual(result["probe"]["reload_max_absolute_difference"], 0.0)
            self.assertTrue((output / "adapter/adapter_model.safetensors").is_file())
            self.assertFalse((output / "adapter/adapter_model.bin").exists())
            self.assertIsNone(result["accuracy_metrics"])
            for instance in TinyRandomScorer.instances:
                self.assertTrue(all(torch.equal(parameter.detach(), before) for parameter, before in instance.base_snapshots))
            saved = read_checkpoint(output, allow_synthetic=True)
            self.assertEqual(saved["adapter_after_sha256"], result["adapter_after_sha256"])
            with self.assertRaises(ValueError):
                TrainedQwenTextScorer(output)
            loaded = TrainedQwenTextScorer(output, allow_synthetic=True, _base_factory=TinyRandomScorer)
            bundle = synthetic_dataset()
            batch = bundle["batches"][0]
            class Recording:
                recording_id = "synthetic-train"
                def get_frame(self, number):
                    return {key: value for key, value in batch["candidates"][0]["images"][0].items() if key != "bbox_xyxy"}
            index = trained_index(Recording(), batch["histories"][0]["observations"] + batch["histories"][1]["observations"],
                                  source="annotated", checkpoint=output, allow_synthetic=True)
            index.scorer._base_factory = TinyRandomScorer
            intervals = rank_evidence(batch["question"], index)
            self.assertTrue(intervals)
            self.assertTrue(all(interval["end_frame"] <= 1 for interval in intervals))
            self.assertEqual(index.last_receipt["scorer"]["training_state"], "synthetic_smoke_adapter")
            loaded.score(batch["question"], [json.dumps({"text": "probe"})])
            self.assertTrue(all(not parameter.requires_grad for parameter in loaded.model.parameters()))

    def test_corrupt_checkpoint_and_manifest_changes_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "checkpoint"
            self.train(output)
            scorer = TrainedQwenTextScorer(output, allow_synthetic=True, _base_factory=TinyRandomScorer)
            path = output / "training.json"
            original = json.loads(path.read_text())
            changed = deepcopy(original)
            changed["accuracy_metrics"] = {"fake": 1}
            write_json(path, changed)
            with self.assertRaises(ValueError):
                scorer._load()
            write_json(path, original)
            weights = output / "adapter/adapter_model.safetensors"
            weights.write_bytes(weights.read_bytes() + b"changed")
            with self.assertRaises(ValueError):
                read_checkpoint(output, allow_synthetic=True)

    def test_split_leakage_and_existing_output_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "checkpoint"
            output.mkdir()
            def forbidden(config):
                raise AssertionError("Must not load a model")
            with self.assertRaises(FileExistsError):
                train_adapter(synthetic_dataset(), output, _scorer_factory=forbidden)
            bundle = synthetic_dataset()
            bundle["split_manifest"]["test"] = ["synthetic-train"]
            _seal(bundle)
            with self.assertRaises(ValueError):
                train_adapter(bundle, Path(directory) / "new", _scorer_factory=forbidden)

    def test_training_context_overflow_never_truncates(self):
        scorer = TinyRandomScorer(QwenConfig())
        scorer._load()
        sample = {"question": synthetic_dataset()["batches"][0]["question"], "document": "long document"}
        with self.assertRaises(ValueError):
            _encode(scorer, [sample], 1)

    def test_nonfinite_training_failure_remains_a_failed_checkpoint(self):
        import torch

        class NonfiniteScorer(TinyRandomScorer):
            def _load(self):
                super()._load()
                # Corrupt the final head for a deterministic nonfinite-loss repro.
                with torch.no_grad():
                    self.model.lm_head.weight.fill_(float("nan"))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            with redirect_stderr(io.StringIO()), self.assertRaises(RuntimeError):
                train_adapter(synthetic_dataset(), output, _scorer_factory=NonfiniteScorer)
            manifest = json.loads((output / "training.json").read_text())
            self.assertEqual(manifest["status"], "error")
            with self.assertRaises(ValueError):
                read_checkpoint(output, allow_synthetic=True)


if __name__ == "__main__":
    unittest.main()
