# Task 5: grounding model and training handoff

Prepared on 20 November 2026 for the Part 2 assignments. This implements the model/data interfaces and cloud training path. **No real 8B parameter-update run or grounding-quality result has been obtained yet.** The current laptop has 4 GiB VRAM; the actual GPU pilot is still required.

## What is available now

- A pinned Qwen3-VL-8B-Instruct adapter with lazy loading, an offline default, verified LoRA manifests, bounded image/token inputs and separate operational errors.
- Shared instruction/result records, original-image coordinates and strict model JSON parsing.
- Training-case loading, partial-label formatting and human-review bindings.
- A real QLoRA training/adapter-save/fresh-base-reload runner, with explicit CUDA/dependency/storage preflight.
- A bounded public-data bootstrap tested from actual pinned HTTP sources through final exported cases.
- A cloud notebook and minimal source/artifact bundlers. The notebook is syntax/format validated but has not executed on a GPU.

The existing QA implementation, its retriever and its configured hosted answerer are unchanged. This model is for the new passenger-instruction route; it is not a replacement label for the unfinished earlier embedding experiment.

## Task 4: application interface

Import GroundingConfig and GroundingModel from trackrag.dropoff.model. Constructing a model object does not download weights or start a GPU backend. The first interpret call does.

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

Example call, after the application has built the request:

```python
import json
from trackrag.dropoff.model import GroundingConfig, GroundingModel

config = GroundingConfig.from_mapping(
    json.loads(open("configs/dropoff/model-pretrained.json", encoding="utf-8").read())
)
model = GroundingModel(config)
outcome = model.interpret(request)
if not outcome["ok"]:
    show_operational_error(outcome["error"])
elif outcome["result"]["status"] == "proposed":
    show_destination_proposal(outcome["result"])
else:
    show_unresolved_interpretation(outcome["result"])
```

The show_* functions above belong to Task 4's UI and are illustrative names, not existing functions.

Successful semantic results contain status, destination_point_xy, destination_box_xyxy and clarification_question. Optional anchor_boxes_xyxy is available only when supplied by the model. All returned geometry uses original-image pixels. Unresolved results have null destination geometry. Missing checkpoints, timeouts, invalid JSON and other operational failures return ok=false; they are not semantic abstentions.

The model never creates a confirmation event. Task 4 owns proposal identity, correction, cancellation and confirmation bound to the current frame. A scene change invalidates the old dialogue/proposal. The predicted region expresses interpreted intent, not certified stopping clearance or legality.

The supplied pretrained configuration uses CUDA/NF4 with 65,536-262,144 image pixels, 2,048 input tokens and 192 generated tokens. Trained runs export a matching inference_config.json. Keep those budgets matched in comparisons. Downloads are disabled unless explicitly enabled on a suitable host.

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

## Public pilot source and leakage correction

The verified production command is:

```text
python -m tools.dropoff_bootstrap_regseg --out data/pilot --limit 10 --image-prefix-mib 4 --max-download-mib 128
```

It fetches bounded portions of the original release rather than the full 1.49 GB archive. The completed local run transferred 97,222,464 bytes in 11 requests and exported ten training examples on ten verified distinct original scenes.

The release's concise test actually derives from 170 training and 338 validation commands. Source-scene mapping identified 58 affected training scenes, comprising 1,146 training rows; the pilot excludes all of them. This corrects the earlier paper-based assumption that the concise test came only from validation images.

See [source audit and hashes](dropoff-data-source.md). The ten examples are for a training infrastructure pilot, not held-out performance or a sufficient final corpus.

## Training and inference commands

Use a separate Linux/cloud GPU environment with requirements-grounding-train.txt. Record the resolved package versions. Hardware/disk thresholds are only preflight floors, not a guarantee that the chosen model/input budget fits.

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

The 8B model also needs suitable resources for interactive inference. Cloud training does not make its base weights fit the 4 GiB laptop. Task 4 can run the application on the GPU host or connect its own model service while retaining the model/checkpoint identity.

## Verification limits

The real pinned Qwen3VLProcessor and training collator processed all ten genuine pilot examples using CPU tensors. Every target decoded to exactly the two supervised destination fields; image, user/system and padding tokens were excluded from the loss. Inputs used 565-585 tokens and 61-68 supervised tokens in this check.

This verifies actual processor/template/collation compatibility. It does not run the 8B neural model, update its parameters, measure prediction quality or prove CUDA memory fit. Software tests use clearly labelled generated fixtures and injected backends where appropriate.

The next required execution is the [GPU notebook pilot](dropoff-cloud-quickstart.md). Final adaptation additionally needs the reviewed local instruction data and development split from Tasks 2/3. Task 1 owns independent final scoring; Task 4 owns full application acceptance.
