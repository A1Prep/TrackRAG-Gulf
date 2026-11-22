# Task 4: LoRA training, saved adapters and verified reload

The [training command](../trackrag/retrieval_training.py) now performs actual
parameter updates, saves a LoRA adapter and reloads it onto a fresh base model.
The first completed run uses **synthetic examples only**. It proves that this
machinery works; it does not complete real-data fine-tuning or establish better
vehicle/time retrieval. Human review remains postponed, not silently replaced.

## Implementation

- [retrieval_training_data.py](../trackrag/retrieval_training_data.py) prepares
  validated training bundles. Every example binds a full candidate packet's
  metadata hash, a candidate ID, a binary relevance label and its provenance.
- [retrieval_training.py](../trackrag/retrieval_training.py) validates partitions,
  creates LoRA parameters, runs optimizer steps, logs loss/gradient norms, saves
  safetensors and verifies fresh-base reload consistency.
- [retrieval_qwen_trained.py](../trackrag/retrieval_qwen_trained.py) loads the saved
  parameters and exposes the existing `build_index` / `rank_evidence` interface.
- [requirements-qwen-training.txt](../requirements-qwen-training.txt) adds optional
  PEFT 0.18.1 and Accelerate 1.12.0 to the pretrained dependencies.

Inference, training and reloaded inference use the same pinned
`Qwen/Qwen3-Reranker-0.6B` base revision and `trackrag.qwen-text.v1` text formatter.
The base tokenizer is unchanged. No images, color descriptions or OCR are added.
Question IDs, file paths, gold answers, labels and reviewer notes are not model
prompt features. The full packet validator enforces original frame/source/track
identities and query cutoffs before formatting.

## Objective and trainable parameters

For each question/candidate pair, compute the model's final next-token logits for
`no` and `yes`. The pointwise loss is cross entropy over those two logits, with
irrelevant mapped to 0 and relevant mapped to 1. No target answer text is inserted
into the prompt. This matches inference's yes/no-normalized score.

LoRA adds low-rank trainable matrices to `q_proj` and `v_proj` in the selected last
attention layers. All original model weights stay frozen and only LoRA tensors
are supplied to AdamW. The verified CPU defaults are:

| Setting | Default |
| --- | --- |
| Optimizer steps | 2 |
| Batch size | 1 |
| Learning rate | 0.0001 |
| LoRA rank / alpha | 4 / 8 |
| Last layers adapted | 1 |
| Maximum input tokens | 512, also bounded by the inference configuration |
| Gradient clipping norm | 1.0 |
| Seed | 42 |
| Training dtype | float32 |

These small values are machinery-test defaults, not tuned settings. A finite loss
and finite gradient norm are required. Inputs beyond the token budget fail without
truncation. Training fails if no saved adapter tensor actually changes. Losses on
two different samples must not be described as a learning curve or improvement.
CUDA can be selected in the base configuration; this implementation explicitly
uses float32 and has only been exercised on CPU. GPU/mixed-precision optimization
and substantive training configuration selection remain future work.

## Run the synthetic smoke check

From the repository root in the existing isolated environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-qwen-training.txt
```

After the pretrained weights are cached, save a base configuration outside Git,
for example `runs/qwen-training-config.json`:

```json
{
  "cache_dir": "models/huggingface",
  "device": "cpu",
  "batch_size": 2,
  "max_tokens": 4096,
  "cpu_threads": 4,
  "local_files_only": true
}
```

Then choose a fresh output directory:

```powershell
.\.venv\Scripts\python.exe -B -m trackrag.retrieval_training smoke --qwen-config runs/qwen-training-config.json --out runs/qwen-training-smoke-new
```

The command generates two tiny recordings in memory: a training recording with a
bus question and a separate development recording with a car question. Each has
one bus and one car. Two relevant/irrelevant training examples and two development
examples are derived from their known generated classes. These are **synthetic
labels**, not inspected real footage or human judgments. No original recording or
answer model is needed. Development examples are not used for gradient updates.

`--training-config <JSON>` accepts `TrainingConfig` fields. Increasing the step
count changes how many updates run; it does not create additional training data.
The current loop cycles through a seeded shuffled training order. It has no
automatic hyperparameter search, early stopping, optimizer-resume support or
best-checkpoint selection.

## Saved artifacts and reload guarantees

A completed checkpoint directory contains:

- `adapter/adapter_model.safetensors`: learned LoRA tensors, not a full copy of
  Qwen's frozen weights.
- `adapter/adapter_config.json`: the base model/revision, layer selection and
  adapter structure; PEFT may also write an informational model card.
- `dataset.json`: the exact input bundle, including its synthetic/reviewed scope,
  split manifest, packet metadata, labels and provenance.
- `training-log.jsonl`: step losses, gradient norms and selected sample IDs.
- `training.json`: completion/error status, model/dependency versions, executed
  source hashes, configuration, trainable tensor names/counts, before/after adapter
  fingerprints, changed tensor magnitudes and file hashes.

Before success, the command saves the adapter, loads a **fresh base model**, attaches
the saved parameters and checks their exact tensor fingerprint. It compares scores
on the first two development examples against the in-memory trained model with an
absolute tolerance of 1e-6. This is a reload consistency probe, not development-set
accuracy, model selection or an independent relevance evaluation.

Checkpoint loading verifies the dataset, input policy, artifact hashes, base
model/revision and loaded adapter tensors. A manifest changed between scorer
construction and loading is rejected. Failed runs are retained with `status:error`
once checkpoint staging has begun; invalid preflight inputs may fail before an
output directory is created. Existing output directories are not overwritten.
Keep checkpoints and generated bundles under ignored paths or outside the repo.

## Use a saved adapter for ranking

```python
from trackrag.retrieval_qwen_trained import build_index, rank_evidence

index = build_index(
    recording, observations, source="annotated",
    checkpoint="runs/qwen-training-smoke-new",
    allow_synthetic=True,  # Required ONLY for deliberate smoke-check use.
)
intervals = rank_evidence(question, index, k=3)
receipt = index.last_receipt
```

The result still uses the shared seven-field `RankedInterval` records. All normal
prefix-only candidate generation, complete-pool scoring and deterministic tie
handling are reused. Scorer provenance identifies the checkpoint/dataset hashes
and reports `synthetic_smoke_adapter` or `human_reviewed_adapter` explicitly.

The existing reviewer can select the adapter without changes to Task 5:

```powershell
$env:TRACKRAG_QWEN_SOURCE = 'annotated'
$env:TRACKRAG_QWEN_CHECKPOINT = 'runs/qwen-training-smoke-new'
$env:TRACKRAG_QWEN_ALLOW_SYNTHETIC = '1' # Only for an explicitly labeled smoke run.
python -m trackrag check-adapters --source annotated --answerer none --retrieval-adapter trackrag.retrieval_qwen_trained
```

For a reviewed checkpoint, omit the synthetic opt-in. Optional
`TRACKRAG_QWEN_CONFIG` overrides runtime settings such as device/cache location,
but the saved base model and immutable revision must still match. Adapter
readiness checking validates interfaces; it is not a model-loading or quality test.

## Move to reviewed training later

The separate preparation command uses the existing review pack's image hashes,
candidate/label bindings and inspected-frame checks:

```powershell
.\.venv\Scripts\python.exe -B -m trackrag.retrieval_training prepare-reviewed --review runs/task4-relevance-review-01 --labels <REVIEWED_LABELS.jsonl> --out runs/qwen-reviewed-training.json
.\.venv\Scripts\python.exe -B -m trackrag.retrieval_training train --dataset runs/qwen-reviewed-training.json --qwen-config runs/qwen-training-config.json --out runs/qwen-reviewed-checkpoint-new
```

Only human relevant/irrelevant judgments from training and development recordings
are included. Pending, uncertain, AI-assisted and test-recording judgments are
excluded, not relabeled as negative. Both train classes and at least one separate
development example are required; these mechanical requirements do not certify
an adequate dataset. The current pending worksheet cannot pass this command.
Reviewer identity remains a declared assertion, not authentication.

`prepare-reviewed --tracked-only` explicitly excludes scene examples while keeping
the full candidate packets. The [focused real-data pilot](task4-realdata-next.md)
uses this option because current scene documents contain no observed vehicle
features. This is a training-selection policy, not an automatic relevance label.

For meaningful learning, expand the reviewed questions and include difficult
wrong-vehicle and wrong-time negatives whose supporting facts are present in the
model inputs. The synthetic class-only task does not train temporal reasoning or
visual appearance understanding. Preserve recording-level train/dev/test splits,
use development data for configuration choices and keep final test data separate.
Use identical candidate/input policies for frozen-versus-trained comparisons and
the existing Task 5 interfaces for downstream evidence/answer evaluation. Report
improvement only after independently reviewed evaluation supports it.

## Validation scope

[Data tests](../tests/test_retrieval_training_data.py) use generated images and
synthetic review assertions. [Training tests](../tests/test_retrieval_training.py)
use tiny randomly initialized two-layer Qwen models to verify optimizer updates,
frozen base tensors, saved safetensors, reload equality, corrupt-artifact rejection,
synthetic opt-in, split leakage rejection and nonfinite-loss handling. These tests
do not download pretrained weights or establish model quality.

Actual downloaded-weight smoke results are recorded separately in the experiment
summary. They are synthetic pipeline evidence, not real EMT training, held-out
ranking accuracy, image understanding or completion of the assessed fine-tuning.

The full local suite passed **386 tests and 397 subtests**, including 13 new
training/data tests. Training CLI help, existing reviewer/evaluator help and the
trained adapter's signature readiness check also passed. Readiness did not load
weights; the actual run below did.

The completed actual-weight run is `runs/task4-qwen-training-smoke-01`:

| Check | Observed result |
| --- | --- |
| Input data | Two synthetic train examples and two separate synthetic dev examples |
| Optimizer | Two AdamW steps on the actual pinned pretrained base plus LoRA |
| Trainable parameters | 20,480, in the last attention layer's q/v LoRA matrices |
| Changed saved tensors | Four LoRA tensors; before/after adapter fingerprints differ |
| Saved adapter weight file | 82,472 bytes |
| Fresh-base reload score difference | 0.0 on the two development probes |
| Shared retrieval interface | Actual saved adapter ranked both generated recordings |
| Future observation control | Added frame-2 observation left frame-1 inputs, scores and intervals exactly unchanged |

The [recorded training smoke summary](../experiments/task4-qwen-training-smoke-summary.json)
preserves configuration, versions, source/artifact hashes, losses, changed-tensor
magnitudes, reload checks and ranking receipts. The two step losses correspond to
different examples and are not evidence of a decreasing loss curve. No reviewed
real-data checkpoint or relevance improvement is claimed.

Sources: [official Qwen scoring example](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B),
[PEFT LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora),
[PEFT checkpoint format](https://huggingface.co/docs/peft/v0.18.0/en/developer_guides/checkpoint).
