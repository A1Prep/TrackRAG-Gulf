# Task 5: grounding model and training handoff

Updated on 21 November 2026 for the Part 2 assignments. **Real local 8B LoRA training has completed on the 4 GiB RTX 3050 laptop GPU.** V1 processed all 500 public training examples twice, completing 1,000 example presentations and 500 optimizer updates. It updates attention adapters in the final eight language layers; the full 8B vision-language base remains frozen.

V1's fixed 20-case public development comparison produced 3 point-in-mask hits for the trained model and 0 for the matched pretrained model. All 20 outputs in each condition had valid geometry. This is limited development evidence, not final EMT accuracy or complete passenger-instruction performance. V2 is a controlled point-first/integer teacher-target experiment; its quality evaluation is pending. Neither run supplies the missing reviewed local clarification labels.

For the current laptop route, use [local training and portable handoff](dropoff-local-training.md) and [the streamed runtime interface](dropoff-streamed-runtime.md). The Linux/cloud runner below remains an alternative. Task 5 and the overall project must not be marked complete from training or valid JSON alone.

## What is available now

- A pinned local Qwen3-VL-8B runtime with lazy loading, an offline default, verified LoRA manifests, bounded image/token inputs and separate operational errors. The laptop uses the separately pinned prequantized distribution documented in the streamed-runtime handoff.
- Shared instruction/result records, original-image coordinates and strict model JSON parsing.
- Training-case loading, partial-label formatting and human-review bindings.
- A real local streamed-LoRA training path and a separate larger-GPU QLoRA runner. Saved-adapter reload, full interpretation and model quality have separate checks.
- A bounded public-data bootstrap tested from actual pinned HTTP sources through final exported cases.
- A cloud notebook and minimal source/artifact bundlers. The notebook is syntax/format validated but has not executed on a GPU.

The existing QA implementation, its retriever and its configured hosted answerer remain a separate route. Passenger grounding adds a new interface; its LoRA artifacts and results do not establish completion of a separate retrieval or embedding experiment.

## Task 4: application interface

Import GroundingConfig and GroundingModel from trackrag.dropoff.model. Constructing a model object does not download weights or start a GPU backend. The first interpretation call does. Use the selected trained artifact's `inference_config.json`; `configs/dropoff/model-streamed-pretrained.json` selects the pretrained baseline and does not contain a trained adapter.

The request format is defined in trackrag/dropoff/contracts.py:

| Field | Meaning |
| --- | --- |
| request_id | Unique identifier for this request |
| scene.frame | Existing Frame record: recording_id, original frame_id, timestamp_s, width, height and image_path |
| scene.source | predicted for the final route; annotated needs an explicit development override |
| scene.tracks | Existing TrackObservation records, all from the same recording/source and no later than the frame |
| scene.provenance | Optional receipt metadata; image_sha256 binds the expected original bytes. This object is excluded from the LM prompt. |
| text | Current typed or corrected recognized instruction |
| input_mode | typed or voice |
| dialogue | Up to 16 user/assistant text turns, each bound to this same recording and frame |
| original_transcript | Optional original recognized text, preserved separately from corrections |

Task 1 supplies the actual image and predicted observations. The public training pilot explicitly has no detector run and an empty observation list; it is not evidence of end-to-end 901 integration.

The [real-scene input command](dropoff-scene-inputs.md) now connects the existing EMT reader and validated Task 1 saved predictions to this format. It exports portable image/request snapshots without annotations. Task 4 can reuse `trackrag.dropoff.inputs.build_instruction_request` with its selected frame and validated predictions. This completes input preparation; actual language-model inference and trained-model acceptance remain separate.

Standalone smoke example after the current source and verified adapter bundles have been extracted into one working directory, dependencies installed and the pinned base prepared:

```python
import json
from pathlib import Path

from tools.dropoff_predict import read_requests
from trackrag.dropoff.model import GroundingConfig, GroundingModel


def main():
    config = GroundingConfig.from_mapping(
        json.loads(Path("inference_config.json").read_text(encoding="utf-8"))
    )
    request = read_requests("probe-requests.jsonl")[0]
    model = GroundingModel(config)
    try:
        outcome, receipt = model.interpret_with_receipt(request)
        print(json.dumps({"outcome": outcome, "receipt": receipt}, indent=2))
    finally:
        model.close()


if __name__ == "__main__":
    main()
```

The `__main__` guard is required for standalone Windows scripts using the spawned worker. Task 4 replaces the smoke request with the current image and predicted observations. Its application should reuse one model instance, call `interpret_with_receipt` for each request, and close it when the owning application session ends. The atomic method prevents another request from replacing the receipt before it is copied.

Successful semantic results contain status, destination_point_xy, destination_box_xyxy and clarification_question. Optional anchor_boxes_xyxy is available only when supplied by the model. All returned geometry uses original-image pixels. Unresolved results have null destination geometry. Missing checkpoints, timeouts, invalid JSON and other operational failures return ok=false; they are not semantic abstentions.

The model never creates a confirmation event. Task 4 owns proposal identity, correction, cancellation and confirmation bound to the current frame. A scene change invalidates the old dialogue/proposal. The predicted region expresses interpreted intent, not certified stopping clearance or legality.

Setting `input_mode="voice"` records the instruction's origin; it does not perform speech recognition. Task 3 supplies the transcript, Task 4 supports any correction, and the corrected text enters this same model interface. Preserve `original_transcript` separately when available. Neither recognized speech nor a generated proposal confirms a destination automatically.

The laptop configuration uses CUDA/NF4 with 65,536-262,144 image pixels, 2,048 input tokens and 192 generated tokens. Trained runs export a matching `inference_config.json`. Keep preprocessing and decoding matched in comparisons. The streamed worker is offline: prepare its pinned base and processor separately with `tools.dropoff_prepare_streamed_model`; `--allow-download` is not a shortcut for this runtime. The original `model-pretrained.json` belongs to the separate resident-model route.

Model, cache and adapter paths in a configuration resolve from the process's working directory, not the configuration file's folder. Request image paths loaded by `read_requests` resolve from the request JSONL folder. Follow the [verified extraction and reload steps](dropoff-local-training.md#portable-inference) before handing a bundle to another teammate. Keep the original ZIPs for verification. The captured `training_source.zip` belongs only in a separate training-reproduction copy; applying it to production would restore older interpretation/runtime code.

## Task 2/3: training data contract

Each JSONL case has exactly these top-level keys:

```text
schema_version, case_id, scene_group, partition, source,
request, target, supervised_fields, review
```

Partition is train, development or test. case_id matches request.request_id. Relative image and mask paths resolve from the JSONL file's directory. Every case declares source.dataset and the source image SHA256; a mask path also requires its SHA256. The loader checks actual hashes, dimensions, mask foreground, destination geometry and target availability.

Two supervision formats are supported:

1. **Destination only:** target and supervised_fields contain destination_point_xy and destination_box_xyxy. Public RegSeg masks support these two outputs. They do not supply status, anchor or clarification labels.
2. **Interpretation:** target includes status, both nullable destination fields and clarification_question. Optional anchor_boxes_xyxy is included only if actually labelled. Proposed targets contain valid geometry; unresolved targets contain null geometry.

Public RegSeg cases are checked against the pinned source-index/scene mapping and exact published command hash. A paraphrased or enriched command needs genuine human review rather than retaining public_annotation status. Known reserved scenes are rejected from both training and checkpoint selection.

Human-reviewed cases require reviewer identities, a timezone-aware reviewed_at, the inspected image SHA256 and two portable hashes. Ambiguous cases and all local final-test cases need at least two independent reviewers. Generate the bindings only after actual review:

```python
from trackrag.dropoff.data import inference_input_sha256, supervision_sha256

review["input_sha256"] = inference_input_sha256(
    case["request"], case["source"]["image_sha256"]
)
review["supervision_sha256"] = supervision_sha256(
    case["target"], case["supervised_fields"], case["request"]["scene"]["frame"]
)
```

These functions record bindings; they do not perform or certify human review. Changing the instruction, dialogue, image or targets invalidates the corresponding approval. Review text, source metadata and target fields never enter the model prompt.

The adaptation configuration refuses to call a tiny destination-only pilot the final trained system. It requires separate development data and full interpretation examples. Actual counts and model quality remain evaluation questions.

## Earlier public pilot source and leakage correction

The earlier ten-example bootstrap command is:

```text
python -m tools.dropoff_bootstrap_regseg --out data/pilot --limit 10 --image-prefix-mib 4 --max-download-mib 128
```

It fetches bounded portions of the original release rather than the full 1.49 GB archive. The completed local run transferred 97,222,464 bytes in 11 requests and exported ten training examples on ten verified distinct original scenes.

The release's concise test actually derives from 170 training and 338 validation commands. Source-scene mapping identified 58 affected training scenes, comprising 1,146 training rows; the pilot excludes all of them. This corrects the earlier paper-based assumption that the concise test came only from validation images.

See [source audit and hashes](dropoff-data-source.md). The ten examples are for a training infrastructure pilot, not held-out performance or a sufficient final corpus.

## Training and inference commands

The current laptop path uses `requirements-grounding-local.txt`, `configs/dropoff/model-streamed-pretrained.json` and `tools.dropoff_fit_local`. Follow the [local recipe](dropoff-local-training.md) for the 500/125 scene-disjoint data, fitting, matched evaluation, production probe and packaging. That recipe also distinguishes V1 float targets from the optional V2 teacher encoding.

The commands below are the alternative resident-model training path for a suitable Linux/cloud GPU host. Use a separate environment with `requirements-grounding-train.txt` and record resolved package versions. Its CUDA/disk preflight floors do not describe the streamed laptop route and do not guarantee capacity.

```text
python -m tools.dropoff_train validate --train data/pilot/cases.jsonl --config configs/dropoff/training-pilot.json
python -m tools.dropoff_train preflight --config configs/dropoff/training-pilot.json --cache-dir models/dropoff-cache --out runs/pilot-preflight
python -m tools.dropoff_train train --train data/pilot/cases.jsonl --config configs/dropoff/training-pilot.json --cache-dir models/dropoff-cache --out runs/pilot-run
python -m tools.dropoff_train verify-reload --run-dir runs/pilot-run --cache-dir models/dropoff-cache
```

Choose fresh output paths. The pilot performs two optimizer steps and verifies actual adapter changes. Save/reload success and a valid spatial smoke output remain separate from measured grounding accuracy. Failed generations and resource failures are retained, and cannot generate a successful pilot completion receipt.

For actual adaptation, use training-adaptation.json with the reviewed training/development corpus. Final evaluation data stays outside training, checkpoint selection and calibration.

Export public inference requests in a separate preparation step, then call the real model adapter:

```text
python -m tools.dropoff_predict export-requests --cases data/pilot/cases.jsonl --partition train --out runs/pilot-requests.jsonl
python -m tools.dropoff_predict predict --requests runs/pilot-requests.jsonl --config configs/dropoff/model-pretrained.json --out runs/pretrained-predictions
python -m tools.dropoff_predict predict --requests runs/pilot-requests.jsonl --config runs/pilot-run/inference_config.json --adapter-dir runs/pilot-run/adapter --cache-dir models/dropoff-cache --out runs/trained-predictions
```

Export strips all target/review/source-label fields and binds expected image bytes. Prediction snapshots its consumed request bytes before inference. It saves outcomes, original/resolved inputs and per-request receipts, including operational errors. The training-source example above is only an execution check; use reserved inputs for final scoring.

The tested streamed profile can train and infer on this 4 GiB laptop by keeping frozen weights outside VRAM and moving them as needed. Capacity does not establish live-response speed or destination quality. Task 4 must measure its own end-to-end latency and preserve the model, adapter and decoding identity when using a different GPU host.

## Verification limits and remaining acceptance

The real pinned Qwen3VLProcessor and training collator processed all ten genuine pilot examples using CPU tensors. Every target decoded to exactly the two supervised destination fields; image, user/system and padding tokens were excluded from the loss. Inputs used 565-585 tokens and 61-68 supervised tokens in this check.

That earlier check verified processor/template/collation compatibility only. The later V1 local run establishes actual parameter updates and the limited development result above. Software and exported-source tests still use labelled fixtures or injected backends where appropriate; they do not replace real-model evaluation.

Production interpretation has a separate structured-decoding route. A valid `ok=true` result may be unresolved; valid JSON does not establish that a proposed point represents the intended road area. Preserve the actual functional failures and evaluate the full application prompt separately from public destination-only scores.

Remaining acceptance includes the V2 results and model-selection decision, a verified selected-adapter handoff and another teammate's fresh-start run, reviewed local interpretation data, voice/application integration and independent reserved evaluation. Task 1 owns independent final scoring; Tasks 3/4 own speech and full application acceptance. The [cloud notebook](dropoff-cloud-quickstart.md) remains optional rather than the next required setup step.
