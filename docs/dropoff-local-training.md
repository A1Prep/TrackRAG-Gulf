# Local 8B grounding training

Task 5 now has a real local training route for Qwen3-VL-8B on the RTX 3050
laptop GPU. The base remains an 8B vision-language model. Parameter-efficient
fine-tuning changes attention LoRA adapters in its final eight language decoder
layers; it does not update every base parameter. The frozen visual encoder and
language prefix still process the actual image and instruction.

## Training scope

The current run uses 500 public training examples from 295 source scenes, with
125 development examples from 65 different scenes. It preserves the earlier
80 cases' source identities and scene assignments. All 58 source scenes reserved
by the concise-test audit are excluded. No final test annotations enter training
or checkpoint selection.

The labels supervise a destination point and coarse region only. They do not
label stopping legality, physical clearance, ambiguity, anchor identities,
clarification or conversation. Public navigation/region instructions are not
all taxi drop-off requests. This is a public destination-grounding training
stage, not evidence of complete local passenger-dialogue adaptation.

The new Task 2 ten-example public handoff contains reserved concise-test source
IDs and is deliberately excluded. The new EMT handoff's reviewed `video_122233`
cases remain reserved for independent evaluation. No reviewed local
clarification training release was present when this run started.

## Reproduce the environment

Use Python 3.11 and a separate environment on a volume with sufficient space.
From the repository root, an independent Windows installation can use:

```powershell
python -m venv .venv-grounding
.\.venv-grounding\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv-grounding\Scripts\python.exe -m pip install -r requirements-grounding-local.txt
.\.venv-grounding\Scripts\python.exe -m tools.dropoff_prepare_streamed_model
```

The pinned quantized base requires about 7.24 GB of downloaded weight files,
plus processor files and working space. The preparation command does not
download the original model's unquantized weights. It verifies all weight,
configuration and shard-index hashes before use. Keep weights outside Git.

The existing laptop training environment is `.venv-grounding-local`. Its CUDA
Torch installation is reused from an existing local environment; that path is
not a dependency of the new runtime or these reproduction instructions. A fresh
installation by another teammate remains a separate release acceptance check.

## Reproduce data preparation and fitting

The data preparation command downloads a bounded prefix of the pinned public
release, checks actual image/mask joins, rejects invalid masks and creates
scene-disjoint splits. It does not fabricate local labels.

```powershell
python -m tools.dropoff_expand_regseg --out data/dropoff-public-expanded-20261121 --limit 625 --image-prefix-mib 192 --previous-locks configs/dropoff/public-historical-split-locks.json
python -m tools.dropoff_fit_local train --config configs/dropoff/model-streamed-pretrained.json --train data/dropoff-public-expanded-20261121/splits/train.jsonl --development data/dropoff-public-expanded-20261121/splits/development.jsonl --out runs/dropoff-full-local-20261121 --epochs 2 --learning-rate 0.0001 --accumulation 2 --dev-limit 32 --generation-limit 20
```

Use a new output directory when reproducing a run. `cases.jsonl` is the combined
partitioned corpus; fitting uses **only** `splits/train.jsonl`. The earlier split
can be replaced by the committed portable lock registry shown above. The
registry contains only source-scene assignments and integrity hashes, not raw
images or answers. The expansion receipt binds its historical assignment input
and all prior case identities.

The frozen configuration uses NF4 language weights, BF16 compute, rank-8 LoRA
with alpha 16, no adapter dropout, batch size one with two-example accumulation,
seed 42 and a 65,536–262,144 pixel image budget. The learning rate warms up for
5% of updates and then decays with a cosine schedule. Two complete passes mean
1,000 example presentations and 500 optimizer updates.

Checkpoint selection compares token-weighted development NLL on a fixed,
target-independent 32-case subset after each epoch. The lower-NLL trained
checkpoint is saved in `adapter/`. A separately measured pretrained baseline
is retained. NLL is not a spatial accuracy score.

## Evaluation and evidence

Free generation uses a fixed 20-case development cohort, identical prompts,
image budgets, quantization, decoding and inference residency for pretrained
and trained conditions. Model prompts contain the image and instruction, not
the reference answer or mask. Scoring happens after generation.

```powershell
python -m tools.dropoff_fit_local evaluate --config runs/dropoff-full-local-20261121/pretrained_config.json --development data/dropoff-public-expanded-20261121/splits/development.jsonl --out runs/dropoff-full-local-20261121/evaluation/pretrained.jsonl --generation-limit 20 --teacher-all
python -m tools.dropoff_fit_local evaluate --config runs/dropoff-full-local-20261121/inference_config.json --development data/dropoff-public-expanded-20261121/splits/development.jsonl --out runs/dropoff-full-local-20261121/evaluation/trained.jsonl --generation-limit 20 --teacher-all
```

The primary public metric is whether the predicted point lies inside the native
destination mask. Coarse-box IoU and image-normalized point distance are
secondary. Malformed output, timeout and other operational failures remain in
the requested-case denominator. Coverage and valid-only point error are
separately labelled. This development comparison is not the final EMT test.

A second frozen protocol includes all 13 instructions in the six development
image groups with repeated image bytes. Their artificial public recording
identity is held constant, so scene metadata and image bytes are the same while
the human instruction changes. Original case IDs remain reference-scoring keys.
Run the same two model conditions with
`--protocol experiments/dropoff/public-contrast-protocol-20261121.json` and new
output paths. Joint reference hits and changed numerical proposals are reported
separately; different coordinates alone do not establish correct understanding.
Four groups have pairwise-disjoint native reference masks; two have overlapping
references. Only the disjoint groups establish incompatible reference locations.
The protocol uses committed LF bytes across Windows and Linux. Keep its final
hash fixed before inference; the handoff packager rejects an uncommitted change.

Complete the contrast generation and both paired comparison receipts with:

```powershell
python -m tools.dropoff_fit_local evaluate --config runs/dropoff-full-local-20261121/pretrained_config.json --development data/dropoff-public-expanded-20261121/splits/development.jsonl --out runs/dropoff-full-local-20261121/evaluation/contrast-pretrained.jsonl --protocol experiments/dropoff/public-contrast-protocol-20261121.json
python -m tools.dropoff_fit_local evaluate --config runs/dropoff-full-local-20261121/inference_config.json --development data/dropoff-public-expanded-20261121/splits/development.jsonl --out runs/dropoff-full-local-20261121/evaluation/contrast-trained.jsonl --protocol experiments/dropoff/public-contrast-protocol-20261121.json
python -m tools.dropoff_compare_local --pretrained runs/dropoff-full-local-20261121/evaluation/pretrained.jsonl --trained runs/dropoff-full-local-20261121/evaluation/trained.jsonl --out runs/dropoff-full-local-20261121/evaluation/comparison.json
python -m tools.dropoff_compare_local --pretrained runs/dropoff-full-local-20261121/evaluation/contrast-pretrained.jsonl --trained runs/dropoff-full-local-20261121/evaluation/contrast-trained.jsonl --protocol experiments/dropoff/public-contrast-protocol-20261121.json --out runs/dropoff-full-local-20261121/evaluation/contrast-comparison.json
```

Run artifacts record data/source hashes, actual consumed case IDs, settings,
losses, adapter hashes and observed PyTorch memory. PyTorch allocation is not the
entire device's driver memory. Fresh-process reload and real original-pixel
outputs must be checked after selecting the saved adapter.

The fitter also saves one portable public training image and annotation-free
request in `production-probe-input/`. This reload smoke input is not an accuracy
test. After completing both free-generation comparisons, run:

```powershell
python -m tools.dropoff_predict predict --requests runs/dropoff-full-local-20261121/production-probe-input/requests.jsonl --config runs/dropoff-full-local-20261121/inference_config.json --out runs/dropoff-full-local-20261121/production-probe
python -m tools.dropoff_package_local --run-dir runs/dropoff-full-local-20261121 --production-probe-run runs/dropoff-full-local-20261121/production-probe --probe-input runs/dropoff-full-local-20261121/production-probe-input --protocol experiments/dropoff/public-contrast-protocol-20261121.json --out runs/dropoff-handoff-local-20261121
python -m tools.dropoff_bundle artifacts --run-dir runs/dropoff-handoff-local-20261121 --out runs/dropoff-adapter-local-20261121.zip
python -m tools.dropoff_bundle source --out runs/dropoff-source-local-20261121.zip --include-tests
```

The staging directory must be separate from the original run. Packaging checks
selected-checkpoint tensor/file hashes against fresh reloads, recomputes the
matched comparisons, validates the actual isolated-worker production receipt,
and retains the exact training source. The original run is preserved. A full
two-epoch fitting schedule can select epoch one; the packaged manifest records
the selected checkpoint's counters separately from the full run.

For portable inference, extract the source and adapter bundles into **one new
directory**, open a terminal in that directory and run commands there. Paths in
`inference_config.json`, including `adapter` and `models/...`, resolve from the
working directory. Prepare the pinned base and processor with
`python -m tools.dropoff_prepare_streamed_model`, then use:

```powershell
python -m tools.dropoff_predict predict --requests probe-requests.jsonl --config inference_config.json --out runs/reload-check
```

The bundle includes `probe-requests.jsonl` and its public image. You can also
supply your own annotation-free request file. The model and processor downloads
are separate from the adapter bundle.

## Application handoff and remaining gates

Use the real callable and isolated worker described in
[the runtime handoff](dropoff-streamed-runtime.md). Both typed text and corrected
speech transcripts use the same request interface. Speech recognition and UI
dialogue/confirmation belong to the existing Task 3/4 owners.

Public destination-only generation and full interpretation are separate prompts.
Test the production interpretation prompt explicitly; public geometry scores
do not establish clarification quality. Preserve model errors and unresolved
responses rather than filling in missing fields or coordinates with rules.

The technical project is complete only after reviewed local interpretation
cases, final app/voice integration, independent reserved evaluation and another
teammate's fresh-start run are verified. The current training run cannot supply
missing human annotations or certify stopping safety.
