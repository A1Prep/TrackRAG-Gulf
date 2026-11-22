# Reviewed local interpretation continuation

`tools.dropoff_adapt_local` is a separate continuation path for the existing
streamed 8B model. It leaves the public V1/V2 experiments and their runners
unchanged. No verified local training release has been established by this tool;
software tests do not supply human labels or prove 4 GB capacity on future inputs.

Use a completed **staged public handoff**, including its saved production/reload
evidence. A raw public run that still has reload pending is insufficient. The
configuration must select that handoff's adapter with the same pinned streamed
base, eight final decoder layers, LoRA rank 8/alpha 16/dropout 0, and the same
image and input budgets.
A relative `adapter_dir` resolves against the parent handoff; model/cache paths
retain their normal interpretation relative to the working directory.

Both child splits must contain genuinely reviewed full interpretations with
exactly `status`, `destination_point_xy`, `destination_box_xyxy` and
`clarification_question`. RegSeg examples, partial labels and optional anchor
labels are rejected; RegSeg remains in its separate public stage. Shared review
hashes/identities and scene/image split checks apply.
Child training and development must also use different recording IDs, even if
their scene-group strings, images or frame IDs differ.
Training must cover all three statuses; development coverage is reported.
`video_122233` stays reserved. Child training cannot use parent development
scenes/images, and child development cannot use parent training scenes/images.

Validate from the project root with the lightweight environment:

```powershell
python -m tools.dropoff_adapt_local validate --parent-run runs/verified-public-handoff --config runs/verified-public-handoff/inference_config.json --train data/local-reviewed/train.jsonl --development data/local-reviewed/development.jsonl
```

Only after supplying real reviewed data and arranging exclusive GPU use, run in
the documented [local GPU environment](dropoff-streamed-runtime.md):

```powershell
python -m tools.dropoff_adapt_local train --parent-run runs/verified-public-handoff --config runs/verified-public-handoff/inference_config.json --train data/local-reviewed/train.jsonl --development data/local-reviewed/development.jsonl --out runs/local-interpretation-continuation --epochs 2 --accumulation 2 --learning-rate 0.0001 --seed 42
```

The runner checks actual loaded parent parameter hashes before updates and uses
a fresh AdamW optimizer. It serializes reviewed teacher targets in production
order: status, point, box, question. It normalizes existing labels, adds no
statuses and performs no mask snapping. The complete child development split
provides per-epoch teacher NLL selection; this loss is not semantic accuracy.

The new run retains parent/data/target/source bindings, initialization hashes,
consumed case IDs, per-epoch adapters, the selected adapter, inference configs
and failures. Counters cover child updates only; parent history remains separate.
`reload-input/requests.jsonl` is a fixed reviewed **training-case** smoke input,
chosen before training. It contains no target labels and is not quality evidence.

Successful training ends at `local_interpretation_training_completed_reload_pending`.
Fresh-process parameter reload, actual full interpretation generation, independent
local evaluation and application acceptance remain pending. The runner writes no
`adaptation_completed` declaration or final completion file. The public-stage
packager is not a publication path for this new local stage.

All examples in `tests/test_dropoff_adapt_local.py` are temporary software
fixtures; injected training results explicitly deny real model training/quality.
Validation reports known fixture markers, and real training rejects those rows.
Keep human images, reviewed data, run snapshots and adapters outside Git.
