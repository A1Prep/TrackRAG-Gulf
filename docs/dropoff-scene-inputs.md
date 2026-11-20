# Connect real vision results to the grounding model

The scene-preparation command turns an EMT recording, a selected original frame and validated Task 1 saved predictions into the existing Task 5 `InstructionRequest` format. Task 4 can use the same Python helper for its frame-selection screen. It does not run the language model or generate a destination.

The saved-prediction route works with the image-only dataset: human annotation files are unnecessary. It validates the explicit vision configuration and prediction hashes, preserves original coordinates, selects only observations at the requested frame, and copies the exact image bytes into a portable output folder. A frame with no current detections keeps an empty list; it does not inherit objects from earlier frames.

## Prepare one instruction

Run in the full project checkout with `requirements-integration.txt` installed:

```text
python -m tools.dropoff_prepare_scene --data-root <EMT_IMAGES> --predictions <TASK1_PREDICTIONS> --vision-config configs/vision/finetuned.json --recording video_054604 --frame 150 --request-id dropoff-a-150 --text "Please drop me beside the bus." --out runs/dropoff-input-a
```

`--predictions` is the folder containing the recording's `.tracks.jsonl` and `.provenance.json`. The explicit vision configuration must match the prediction identity; this command neither downloads nor reloads the detector checkpoint. Choose a fresh output folder.

Optional `--frame-rate` supplies a known recording rate. Otherwise the EMT reader uses its release-metadata/default policy; the actual rate is saved in the prediction receipt. `--split` may be `train` or `test`. Frame numbers remain original frame IDs.

## Prepare a bank across recordings

Put one instruction specification on each JSONL line:

```json
{"request_id":"a-150","recording_id":"video_054604","frame_id":150,"text":"Please drop me beside the bus."}
{"request_id":"b-150","recording_id":"video_054907","frame_id":150,"text":"I mean the gap between the two cars on the right."}
```

```text
python -m tools.dropoff_prepare_scene --data-root <EMT_IMAGES> --predictions <TASK1_PREDICTIONS> --vision-config configs/vision/finetuned.json --instructions instructions.jsonl --out runs/dropoff-input-bank
```

Specifications require `request_id`, `recording_id`, `frame_id` and `text`. Optional fields are `input_mode`, `dialogue` and `original_transcript`, following the [model handoff](dropoff-model-handoff.md). A corrected voice transcript belongs in `text`; preserve the original recognized wording in `original_transcript`. This preparation command does not perform speech recognition.

Reference destinations, masks and answer keys are rejected as specification fields. A malformed bank or duplicate request identity is rejected before output creation. An identifiable row with a missing frame, stale dialogue, wrong prediction source/identity or other preparation failure retains an explicit error outcome; later valid rows continue.

## Output and model handoff

| File | Purpose |
| --- | --- |
| `requests.jsonl` | Validated inputs for the existing grounding prediction command, with output-relative image paths. |
| `images/` | Exact image snapshots, named by content hash. Copy this with the request file. |
| `preparation-outcomes.jsonl` | One result per requested instruction, including preparation errors and the successful request's line/hash. |
| `prediction-receipts.jsonl` | Consumed prediction/provenance hashes, explicit vision identity and available-frame prefix. |
| `instructions.snapshot.jsonl` | Exact original instruction-bank snapshot for the preparation audit; it is not a model input. |
| `vision-config.snapshot.json`, `run.json` | Consumed configuration, output hashes, counts and clearly labelled execution conditions. |

Exit code 0 means every instruction was prepared, 1 means retained row-level errors, and 2 means a bank/configuration/output setup error. `requested_rows = prepared_rows + preparation_errors`; these preparation failures must remain visible alongside later model outcomes in final evaluation. `requests.jsonl` alone contains only the prepared subset.

Copy the output folder to the chosen inference host, then use the existing interface:

```text
python -m tools.dropoff_predict predict --requests <PREPARED>/requests.jsonl --config configs/dropoff/model-pretrained.json --out <NEW_PREDICTION_RUN>
```

For the trained condition, use the exported trained inference configuration and explicit adapter/cache locations described in the model handoff. The prediction loader resolves image paths relative to `requests.jsonl`; no laptop-specific image path is required on the inference host.

Scene preparation runs in the full checkout. It is intentionally excluded from the minimal cloud **training-source** ZIP because that ZIP omits the EMT reader, vision adapter and OpenCV dependencies. Prepare scene inputs locally, then transfer their small request/image folder separately when needed.

## Call from the application

```python
from trackrag.dropoff.inputs import build_instruction_request

request = build_instruction_request(
    recording, validated_predicted_observations,
    frame_id=selected_original_frame,
    request_id=request_id,
    text=instruction_text,
)
outcome = grounding_model.interpret(request)
```

The caller must supply validated prediction provenance; the CLI uses the existing strict Task 1 saved adapter. The helper verifies exact frame identity, actual image dimensions and SHA256, predicted track identities/boxes, dialogue binding and the shared request schema. It leaves caller records unchanged. Task 4 still owns clarification, proposal IDs and confirmation; Task 1 still owns independent evaluation.

## Verification and limits

The real local check used `video_054604` and `video_054907`, frames 75, 100 and 150, from the annotation-free image root and the verified fine-tuned saved predictions. Six input requests were prepared; three deliberate failures (missing frame, stale dialogue and an unexpected reference field) remained explicit errors. Two selected frames genuinely had no detector observations and kept empty lists. No language model, new detector inference, speech recognizer or NLP training was run in this check.

The actual pinned `Qwen3VLProcessor` also processed all six requests through the production inference-preparation method, offline on CPU. This exposed and fixed a Transformers 5.1 incompatibility with directly tokenizing mixed string/content-block messages. Inference now renders the chat and encodes the same verified in-memory image separately, matching the working training path. Inputs used 586–1,516 tokens and 258,048 processed image pixels, within the configured 2,048-token/262,144-pixel limits. This verifies preprocessing compatibility, not neural prediction quality or GPU fit. The original failed-preprocessing receipt is retained beside the successful rerun.

The new request hash binds the image bytes consumed now. Historical saved prediction artifacts lack per-image byte hashes, so the command cannot retrospectively prove their image-to-prediction binding. It checks their recorded identity and complete track-file integrity and labels this limitation in request provenance. A valid request package is a completed input handoff, not measured grounding accuracy.
