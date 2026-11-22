# Local 8B grounding runtime

The explicit `streamed` runtime runs the native Qwen3-VL image-and-language model
with frozen base weights backed by local checkpoint files. It supports both
pretrained inference and a learned LoRA adapter. It does not call a hosted model
or supply coordinates when model generation fails.

## Identity and files

The quantized distribution is pinned separately from the original planning
checkpoint:

- Model: `unsloth/Qwen3-VL-8B-Instruct-bnb-4bit`
- Revision: `10d842936325905934ea24e64325d774693f166e`
- Processor: `Qwen/Qwen3-VL-8B-Instruct`
- Processor revision: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`

The two source weight files occupy approximately 6.73 GiB. The runtime verifies
both weight SHA256 values and the pinned `config.json` and
`model.safetensors.index.json` digests before native loading. This is a separately
pinned prequantized distribution; the runtime does not claim its bytes match the
original Qwen checkpoint revision.

Keep checkpoint and adapter files outside Git. `model_dir` points to the local
prequantized checkpoint directory, while `cache_dir` points to the Hugging Face
cache containing the original pinned processor. Changing computers requires
updating these paths and supplying the same files.

## Model and memory behavior

All 36 language decoder layers and the image encoder remain part of the native
model forward. LoRA training updates the `q_proj`, `k_proj`, `v_proj` and `o_proj`
adapters in the configured final decoder layers. The base weights are frozen.

The local training profile uses eight resident decoder layers. Other frozen
layers move to CUDA for their forward pass and then restore their original CPU
views. The embedding performs its frozen lookup on CPU. The complete BF16 output
projection is evaluated in 4,096-row chunks, including its final partial chunk;
its backward pass returns the hidden-state gradient with FP32 accumulation.

Inference can keep additional frozen decoder layers on CUDA to reduce transfers.
The default is 20 resident layers, with the adapter still attached to the last
eight. `inference_resident_layers` must cover the suffix and cannot exceed 24.
Use the same inference resident-layer count, image budget and generation settings
for pretrained and trained comparisons. These are runtime placements of the same
weights, not a change in parameter count.

The supported image-budget range is 65,536 to 262,144 pixels. The selected
configuration and real capacity check determine whether a particular combination
fits. PyTorch allocation measurements exclude some CUDA/driver overhead.

The tested stack uses Python 3.11, Torch 2.6 with CUDA 12.4, Transformers 5.1,
PEFT 0.18.1, Accelerate 1.12, bitsandbytes 0.49.1 and safetensors 0.8. The runtime
requires a BF16-capable CUDA device. It never substitutes CPU training or another
model when these dependencies are unavailable.

## Application callable

Use the normal shared `GroundingModel` interface. Its request retains original
frame IDs, original-image pixels, observed tracks and scene-bound dialogue. The
existing contract validation rejects future observations and stale dialogue
before model loading.

```python
from trackrag.dropoff.model import GroundingConfig, GroundingModel

def main(request, model_directory, processor_cache, adapter_directory=None):
    config = GroundingConfig(
        runtime="streamed",
        model_id="unsloth/Qwen3-VL-8B-Instruct-bnb-4bit",
        revision="10d842936325905934ea24e64325d774693f166e",
        model_dir=model_directory,
        cache_dir=processor_cache,
        adapter_dir=adapter_directory,
        device="cuda:0",
        dtype="bfloat16",
        quantization="nf4",
        suffix_layers=8,
        inference_resident_layers=20,
        min_pixels=65536,
        max_pixels=262144,
        max_new_tokens=192,
        max_time_s=300,
    )
    model = GroundingModel(config)
    try:
        outcome, receipt = model.interpret_with_receipt(request)
        return outcome, receipt
    finally:
        model.close()

# Call main(...) from a script protected by:
# if __name__ == "__main__":
```

Windows scripts using the application callable must use the usual `__main__`
guard because inference runs in a spawned local worker. The parent sends the
already verified in-memory image snapshot; the worker does not reopen an image
path or receive reference answers. One persistent child owns the model. The
application serializes initialization, interpretation and receipt updates.
`interpret_with_receipt` returns independent outcome and receipt snapshots under
that same lock. The older `last_receipt` property remains the most recent shared
record; reading it separately after `interpret` is appropriate for a single
caller only.

Worker startup has a separate 120-second limit. Generation uses the configured
deadline, with a parent watchdog that stops an unresponsive owned worker. Native
exit, missing checkpoints, resource limits, malformed output and timeout remain
operational failures. A failed worker is closed; construct a fresh model after
addressing its failure. `close()` releases the owned worker before another model
is loaded.

Successful model JSON is validated and converted from the internal 0..1000 image
scale to original-image pixels. Destination-only JSON is insufficient for the
full interpretation contract: the runtime does not invent `status`, ambiguity,
clarification or confirmation fields. Voice transcription supplies text to this
same interface; speech recognition and confirmation belong to the other task
components.

Application worker generation explicitly uses the `interpret_v1` JSON grammar.
All four fields are required, with status first: a proposal has both geometry
arrays and a null question; clarification has null geometry and a nonempty
question; insufficient evidence has null geometry and question. The native 8B
model chooses the branch, coordinates and question. This grammar supplies no
scene answers, coordinates, clarification labels or confirmation events. Bounds,
box ordering and point-inside-box checks still run after generation, so a
structurally complete answer can still fail validation or be visually wrong.

Install `requirements-grounding-format.txt` for this route. It pins upstream
lm-format-enforcer commit `817f944fcc8851917c40300a47f62d1defc3ffc3`, whose
Transformers 5-compatible integration retains package version 0.11.3. The
published version number alone does not identify that source; runtime checks the
installed archive origin. Missing or mismatched dependencies fail explicitly.
The tokenizer trie is cached once per runtime; each request gets a fresh parser
and the supported `prefix_allowed_tokens_fn` callback. Setup consumes the request
deadline. The allowed vocabulary is the smaller of the tokenizer and actual
output head, excluding padded output IDs. Receipts record the schema hash,
upstream commit, package version, vocabulary sizes and constraint mode.

Direct `StreamedRuntime.generate(...)` keeps `output_contract=None` by default.
The public destination-only pretrained/trained comparison uses that unconstrained
path with the same generation settings for both conditions. Its measurements
must not be presented as evaluation of this application-only JSON grammar.

## Direct training API

The training tool can use `StreamedRuntime` directly in its dedicated Python
process:

```python
import torch
from trackrag.dropoff.streamed import StreamedRuntime

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.90, 0)
runtime = StreamedRuntime(config, training_lora={"rank": 8, "alpha": 16, "dropout": 0.0})
# runtime.model, runtime.processor and runtime.device are available to the trainer.
# Optimize only parameters whose requires_grad is True.
```

Use a dedicated training process. If sharing a direct runtime, hold its
`execution_lock` across each training forward/backward/update and avoid concurrent
inference or adapter changes. Runtime generation and cleanup already hold this
lock. Only the isolated application worker changes process-wide thread and CUDA
allocation settings; a direct trainer configures its own process explicitly.

Before releasing a direct runtime, discard external optimizer, model and tensor
references, then call `runtime.close()`. The method removes offload callbacks,
restores frozen CPU views, and releases its owned model references.

Never merge adapters into this mapped base, initialize the loaded base in place,
call `prepare_model_for_kbit_training` on it, or make its mapped parameters
trainable. Torch's tensor interface cannot enforce the read-only mapping. The
supported operations train new CUDA LoRA tensors and save the adapter separately.

Saved adapters must include `grounding_adapter_manifest.json` binding their base
identity, preprocessing and both adapter files. The runtime also checks the saved
LoRA target expression and rejects layouts or initializers that could modify the
frozen base. Model quality, successful training and completion status come from
the run/evaluation records, not from the presence of an adapter directory.

The fit tool preserves `source_float_box_first` teacher targets by default.
An optional controlled experiment selects
`--target-encoding point_first_mask_preserving_integer_v1`. It starts a fresh
adapter on the same base and applies only to public destination-only labels.
It serializes the point before the box, rounds the point to an integer grid,
and rounds box edges outward. If the nearest point leaves the source mask or
encoded box, it selects the closest valid floor/ceil adjacent grid point by
distance in original pixels. A case without such a point fails preflight.
This is label preparation using training/development masks; inference never
receives those masks or moves generated points to fit them.

Before model construction, `target-encoding.json` binds every transformed target
to its original supervision, input, source image/mask and split-file hashes.
It records rounding rules, order, adjustments and maximum displacement. Training
settings and adapter manifests bind that receipt. Original cases and references
are retained. Pretrained and epoch teacher losses use the selected encoding.
For matching evaluation, explicitly pass
`--teacher-target-encoding point_first_mask_preserving_integer_v1 --teacher-all`.
Teacher summaries identify their encoding: token NLL across different codecs
must not be reported as a semantic accuracy improvement. Free generation still
uses its unchanged destination-only prompt and original spatial references.
This optional experiment is not evidence of improved quality until its actual
matched evaluations are complete; it supplies no missing status/clarification
supervision.

The release artifact's `training_source.zip` preserves the nine files actually
used during training. Apply that snapshot only in a separate copy when
reproducing training. Production inference uses the current source bundle with
its later interpretation and decoding repairs; overlaying the training snapshot
onto that production copy would remove those repairs.

## Verification scope

`tests/test_dropoff_streamed.py` exercises lazy imports, source integrity, loader
cleanup, suffix boundaries, original-pixel integration, unsupported outputs,
future-frame rejection, first-call concurrency and worker failures/deadlines.
The spawn tests include a child that exits abruptly while the parent survives.
Optional Torch checks compare the streamed output projection and gradients with
dense operations, check CPU-view lifetime, and exercise NF4 metadata restoration.

These are software and numerical checks. Real checkpoint training, generated
grounding quality and independent final evaluation require their separately
recorded real-data runs.
