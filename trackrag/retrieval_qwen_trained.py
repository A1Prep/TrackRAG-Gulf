"""Explicit Task 4 adapter for a saved, verified LoRA relevance checkpoint."""

from hashlib import sha256
import os
from pathlib import Path

from .retrieval_qwen import (
    QwenConfig, QwenTextScorer, build_index as pretrained_index,
    config_from_environment, rank_evidence,
)
from .retrieval_training import adapter_state_hash, read_checkpoint


class TrainedQwenTextScorer(QwenTextScorer):
    def __init__(self, checkpoint, *, config=None, allow_synthetic=False, _base_factory=QwenTextScorer):
        self.checkpoint = Path(checkpoint)
        self.manifest = read_checkpoint(self.checkpoint, allow_synthetic=allow_synthetic)
        self.manifest_sha256 = sha256((self.checkpoint / "training.json").read_bytes()).hexdigest()
        self.allow_synthetic = allow_synthetic
        saved = QwenConfig(**self.manifest["base_config"])
        config = config if config is not None else saved
        if not isinstance(config, QwenConfig) or (config.model_id, config.revision) != (saved.model_id, saved.revision):
            raise ValueError("Checkpoint and runtime base model/revision must match")
        super().__init__(config)
        self._base_factory = _base_factory

    def _load(self):
        if self.model is not None:
            return
        # Recheck the local artifact immediately before attaching its parameters.
        read_checkpoint(self.checkpoint, allow_synthetic=self.allow_synthetic)
        if sha256((self.checkpoint / "training.json").read_bytes()).hexdigest() != self.manifest_sha256:
            raise ValueError("Checkpoint manifest changed after scorer construction")
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("Install requirements-qwen-training.txt to load a trained adapter") from error
        base = self._base_factory(self.config)
        base._load()
        model = PeftModel.from_pretrained(base.model.float(), self.checkpoint / "adapter", is_trainable=False)
        model.eval().requires_grad_(False)
        if adapter_state_hash(model) != self.manifest["adapter_after_sha256"]:
            raise ValueError("Loaded adapter tensors differ from the trained checkpoint")
        self.torch, self.tokenizer = base.torch, base.tokenizer
        self.prefix, self.suffix, self.no_id, self.yes_id = base.prefix, base.suffix, base.no_id, base.yes_id
        self.model = model

    def provenance(self):
        result = super().provenance()
        result.update(backend=type(self).__name__, training_state=self.manifest["training_state"],
                      training_dtype=self.manifest["training_dtype"], dataset_kind=self.manifest["dataset_kind"],
                      dataset_sha256=self.manifest["dataset_sha256"], adapter_after_sha256=self.manifest["adapter_after_sha256"],
                      checkpoint_manifest_sha256=self.manifest_sha256)
        return result


def build_index(recording, tracks, *, source=None, checkpoint=None, config=None,
                retrieval_config=None, allow_synthetic=False):
    """Keep Task 5's interface; trained checkpoints are selected explicitly."""
    checkpoint = checkpoint if checkpoint is not None else os.environ.get("TRACKRAG_QWEN_CHECKPOINT")
    if not checkpoint:
        raise ValueError("Supply checkpoint or set TRACKRAG_QWEN_CHECKPOINT")
    if config is None and os.environ.get("TRACKRAG_QWEN_CONFIG"):
        config = config_from_environment()
    allow_synthetic = allow_synthetic or os.environ.get("TRACKRAG_QWEN_ALLOW_SYNTHETIC") == "1"
    scorer = TrainedQwenTextScorer(checkpoint, config=config, allow_synthetic=allow_synthetic)
    return pretrained_index(recording, tracks, source=source, config=scorer.config,
                            retrieval_config=retrieval_config, scorer=scorer)
