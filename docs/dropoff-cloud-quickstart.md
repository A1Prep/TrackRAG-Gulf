# Run the grounding pilot on a cloud GPU

The laptop's 4 GB GPU cannot run the planned 8B vision-language fine-tuning job. This handoff keeps the same planned model and moves installation, model downloads and training to a disposable cloud GPU session. No cloud GPU is provisioned or purchased automatically.

**Current verification limit:** the source/data pipeline and real processor/collator have local checks, but the actual 8B gradient-update/save/reload job has not yet been executed on a suitable GPU. The notebook is a runnable handoff, not a completed training result.

## 1. Build the small source ZIP on D:

From the project checkout, using the lightweight reviewer environment:

```powershell
.\.venv\Scripts\python.exe -m tools.dropoff_bundle source --out runs/dropoff-cloud-source.zip
.\.venv\Scripts\python.exe -m tools.dropoff_bundle verify --bundle runs/dropoff-cloud-source.zip
```

Choose a fresh output filename if it already exists. Keep the printed ZIP SHA256 to compare with the cloud upload receipt.

The allowlist contains the shared contracts, drop-off Python modules, dropoff tools/configurations and separate training requirements. Drop-off docs are included by default; tests are optional with --include-tests. The bundle excludes .env files, virtual environments, raw datasets, recordings, model caches, existing training runs and unrelated application files. Symlinks/reparse points and noncanonical archive paths are rejected.

BUNDLE_MANIFEST.json binds every included file's path, byte count and SHA256. These hashes check consistency; the ZIP still needs to be the one you built and trust.

## 2. Open the notebook in Google Colab

Download [dropoff_gpu_pilot.ipynb](../notebooks/dropoff_gpu_pilot.ipynb), then open it through Colab's **File -> Upload notebook**. Select **Runtime -> Change runtime type -> GPU**.

Run its cells in order and upload only the source ZIP when requested. No private GitHub token, repository clone or Hugging Face login is required for this public base model/data workflow.

Before installing packages or downloading weights, the notebook checks an actual NVIDIA GPU and free disk. Screening floors are at least 12 GiB total GPU memory, 10 GiB free GPU memory and 30 GiB free disk. They are not a promise that the configured model will fit. If the session fails these checks, stop and change the runtime; a small-model substitution is not hidden in the workflow.

The notebook verifies every source file before extracting it. Installation uses a separate cloud virtual environment and the supplied requirements, with --no-cache-dir to avoid duplicated wheel storage. A pinned virtualenv fallback handles cloud Python installations lacking standard venv/ensurepip support. Actual resolved dependency versions are saved with the run.

## 3. Run the genuine update/save/reload pilot

The notebook:

1. Bootstraps ten actual public training examples with pinned sources and bounded downloads, excluding all source scenes reused by the concise public test.
2. Validates labels, original-image geometry, image/mask hashes and source-scene guards.
3. Runs the training preflight without downloading model weights. Its preview path is separate from the training output.
4. Downloads the pinned base model on the cloud host and performs the configured two-step QLoRA pilot.
5. Requires verifiable parameter changes, saved adapter files, a fresh-base reload and a valid spatial generation probe.
6. Starts a fresh Python process to verify the saved adapter and portable probe again, reusing the cloud cache.
7. Downloads the small adapter/configuration/log bundle. Base weights, caches and the bulk dataset are excluded.

The saved spatial probe uses a training example. It establishes that the learned artifact can reload and produce a structured spatial output; it is not held-out accuracy. The ten public mask-only examples do not supervise ambiguity, anchor identities or conversational replies.

If training, dependencies, memory or generation fail, retain the failure receipt and use the notebook's final diagnostic-export cell. An incomplete bundle is labelled diagnostic and is not a successful trained artifact. Download available receipts before the cloud runtime disconnects.

The fresh-process attempt is recorded before launch, then updated with its actual exit code, stdout/stderr and verification result. A failed or interrupted fresh process cannot inherit successful flags from the earlier in-training reload. The exporter always permits labelled diagnostics, including when older completion files still exist.

## 4. Keep the handoff portable

A complete artifact ZIP contains the LoRA adapter, its hashes and pinned base revision, training/data/configuration records, reload verification, actual environment freeze, and exactly the public probe image needed for reload. It does not contain the 8B base weights.

Packaging checks schema/kind, matching base identities/revisions, positive optimizer-step/update records, file hashes and the cloud fresh-process outcome. These are consistency checks, not independent proof of an artifact's origin or model quality.

The saved configuration keeps original training-host paths as provenance. On another suitable GPU host, use explicit cache/adapter overrides instead of changing hashed historical files. Run normal preflight first, especially when the base cache is empty:

```text
python -m tools.dropoff_train preflight --config <RUN>/training_config.json --cache-dir <GPU_CACHE> --out <UNUSED_PREFLIGHT_PATH>
python -m tools.dropoff_train verify-reload --run-dir <RUN> --cache-dir <GPU_CACHE>
python -m tools.dropoff_predict predict --requests <REQUESTS.jsonl> --config <RUN>/inference_config.json --adapter-dir <RUN>/adapter --cache-dir <GPU_CACHE> --out <NEW_PREDICTION_DIRECTORY>
```

Use --allow-download with the prediction command only on the chosen GPU host if pinned base weights need fetching. The adapter ZIP alone does not make the 8B model runnable on the 4 GB laptop.

## What follows a successful pilot

Continue Task 5 with the reviewed full training corpus and a disjoint development partition, using the adaptation configuration. Then give the frozen pretrained/trained conditions to Task 1's independent evaluator and the callable model to Task 4's application.

The pilot does not finish the four-day project. Human-reviewed local examples, meaningful grounding/clarification behaviour, the original QA acceptance checks, voice evaluation, application integration and final held-out measurements remain separate completion gates in [Part 2](final-completion-part2.md).
