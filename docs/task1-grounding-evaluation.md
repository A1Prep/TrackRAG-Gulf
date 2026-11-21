# Task 1: scoring drop-off grounding

`evaluation/grounding.py` scores saved grounding outputs against separate reference files. It never runs or imports a model and never repairs an output. A reserved case is a case in the references file. A missing output, a request error and an invalid output all count as failures and stay in the denominator of every main rate, so declining to answer cannot raise a score.

Task 2's test cases and Task 5's prediction runs keep their own formats (`trackrag/dropoff/data.py`, `tools/dropoff_predict.py`). `evaluation/grounding_inputs.py` converts them. Task 4's clarification episodes use the evaluator's own format, described below.

## How the pieces fit

Task 5's `trackrag/dropoff/inputs.py` and `tools/dropoff_prepare_scene.py` build the model's input from an EMT frame and Task 1's saved predictions. Task 1's `trackrag/dropoff/scenes.py` records the same frame and observations for scoring, under a scene hash. When a reference or an output is converted, its frame and predicted observations are compared with the Task 1 scene of that case, and so are the image bytes when the request or outcome carries the image hash. A request built from another prediction run, such as a fresh run that numbered the tracks differently, is caught there.

The steps, from the repository root in the environment of `requirements-integration.txt`. `<EMT_IMAGES_ROOT>` is the folder that contains `images/`, and `<folder>` is where you unpacked the prediction zips that Task 1 shared.

```sh
# 1. Task 1: scenes for every test frame, from the shared fine-tuned predictions, with both manifests.
#    frames.json lists each test frame once: [{"recording_id": "video_204647", "frame_id": 1200}, ...]
python -m trackrag.dropoff.scenes export --data-root <EMT_IMAGES_ROOT> --frames frames.json \
    --profile configs/vision/finetuned.json --predictions <folder>/predictions-finetuned \
    --frame-manifest experiments/vision/emt_frame_sha256_manifest.json \
    --run-manifest experiments/vision/shared_prediction_runs.json --out-dir scenes

# 2. Task 2's test cases and Task 3's intended file -> references
python -m evaluation.grounding_inputs task5-cases --cases test-cases.jsonl --intended intended.jsonl \
    --scenes scenes --out-dir refs

# 3. Freeze the protocol before any final output exists (planned for 22 November 2026)
python -m evaluation.grounding_inputs protocol --references refs/references.jsonl --replies replies.jsonl \
    --voice voice.jsonl --evaluator-commit <commit> --responder-rules "<the rules below>" --out protocol.json

# 4. Each Task 5 run folder (one model, one transcript kind) -> evaluator outputs
python -m evaluation.grounding_inputs task5-run --run <run folder> --references refs/references.jsonl \
    --scenes scenes --transcript typed --out outputs-trained-direct-typed.jsonl

# 5. Score everything (list each outputs file; PowerShell does not expand wildcards here)
python -m evaluation.grounding --references refs/references.jsonl --scenes scenes \
    --outputs outputs-pretrained-direct-typed.jsonl outputs-trained-direct-typed.jsonl <episode files> \
    --replies replies.jsonl --voice voice.jsonl --protocol protocol.json \
    --train-manifests <Task 5's training and development case files> --out summary.json \
    --cases-out <outside Git>/cases.csv
```

Each `--out` file must be new and its folder must exist; each `--out-dir` folder must be new. Inputs that break the rules below stop a command with a message and exit status 2, and nothing is written. Otherwise the results are written. `status` becomes `not assessed:` with the reasons when any of these hold:

- no frozen protocol;
- no training manifests;
- a leakage hit;
- an unreviewed reference;
- a missing required condition (one with no accepted outputs);
- refused outputs in a required condition;
- a refused required comparison;
- scenes from the stand-in test detector.

The summary records the sha256 of every input and of `grounding.py` itself, and the command prints the headline numbers.

## Who supplies what

| File | From | Format |
|---|---|---|
| Test frames (`frames.json`) and scenes | Task 2 names the frames, Task 1 makes the scenes | Step 1. The scenes folder holds only the scene files and `index.json`. |
| Test cases | Task 2 | Task 5's case schema (`docs/dropoff-model-handoff.md`), below |
| Intended file (`intended.jsonl`) | Task 3, before any proposal is seen | Below |
| Passenger replies (`replies.jsonl`) | Task 3, before any proposal is seen | Below |
| Voice transcripts (`voice.jsonl`) | Task 3 | Below |
| Direct-mode outputs | Task 5 | The run folder of `tools.dropoff_predict predict`, below |
| Clarification episodes | Task 4's harness running Task 5's model | The episode format below |
| Training manifests | Task 5 | The training and development case files, and any other file listing what training or selection used. Any JSON or JSON Lines; every string and key is checked. |
| Protocol | Task 1 | Step 3 |

The references, replies and voice files must be final before step 3. A file changed or added after the freeze stops the run.

The four required conditions are the pretrained and the trained grounding model, each in direct and in clarification mode, with typed instructions. Human and ASR transcripts are extra conditions on the voice cases only. Task 5's Ollama route is a separate baseline with another base model, so `task5-run` refuses it rather than count it as the pretrained condition.

## What the converters check and map

### Task 2's test cases (`task5-cases`)

The cases are read with Task 5's own loader for partition `test`. That loader checks the image and mask hashes, two distinct reviewers with review bindings, the reference point inside the mask and the box against the mask extent. For scoring, each case also needs:

- a full interpretation target, with `status` among its supervised fields;
- `review.status` `human_reviewed`;
- for a `proposed` target, a destination mask as a PNG file (`source.mask_path` and `source.mask_sha256`);
- `request.scene` built from the shared fine-tuned predictions, so that it matches the Task 1 scene.

| Case field | Reference field |
|---|---|
| `case_id`, `scene_group` | `case_id`, `group_id` |
| `request.text`, `request.dialogue` | `instruction`, `dialogue` |
| `request.scene` frame and tracks, `source.image_sha256` | matched to the Task 1 scene, giving `scene_sha256` |
| `target.status` | `expected_status`; `proposed` gives category clear, `insufficient_visual_evidence` missing_reference, `needs_clarification` the category in the intended file |
| `source.mask_path`, `source.mask_sha256` | `mask`, `mask_sha256` (the file is copied into `refs/masks/`) |
| `target.destination_point_xy`, `target.destination_box_xyxy` | `reference_point`, `target_box` |
| `target.anchor_boxes_xyxy` | `anchors`, roles `anchor1`, `anchor2`, ... (unordered) |
| `review.reviewers` | `review.independent_reviews` (the number of distinct names) |
| `review.status` `human_reviewed` | `review.disagreements_resolved: true` |

The intended file has one line for every needs_clarification case:

```json
{"case_id": "A007", "category": "ambiguous", "intended_mask": "intended/A007.png", "intended_mask_sha256": "<sha256 of the PNG>"}
{"case_id": "M003", "category": "missing_reference", "intended_mask": null, "intended_mask_sha256": null}
```

An ambiguous case needs the intended mask, drawn before any proposal was seen. Paths are relative to the intended file.

### Task 5's prediction runs (`task5-run`)

One run folder holds `run.json`, `resolved-requests.jsonl`, `model-outcomes.jsonl` and `receipts.jsonl`. The three data files must match the hashes in `run.json`. One run is one model and one transcript kind (`--transcript typed`, `human` or `asr`); the mode is direct. The requests for a direct run come from `tools.dropoff_predict export-requests --partition test` on Task 2's case file, so each `request_id` is its `case_id`. In a human or ASR run, the request's `text` must be exactly that transcript from the voice file, without corrections; an ASR transcript of `null` counts as a failure.

| Task 5 | Evaluator output |
|---|---|
| `request_id` | `case_id` (it must be a reserved case) |
| `provenance.requested_condition` | `pretrained`, or `trained` for `fine_tuned` |
| request frame and tracks, and the image hash when present | `scene_sha256` of the case's Task 1 scene, or `null` with `scene_problems` listed |
| request `text` and `dialogue` | `instruction_sha256` |
| `ok: true` with `result` | `call` ok; `destination_point_xy`, `destination_box_xyxy`, `clarification_question`, `anchor_boxes_xyxy` become `point`, `box`, `clarification`, `landmark_boxes` |
| `ok: false`, `error.code` `invalid_model_output` | an invalid output: the call ran but the response could not be parsed |
| `ok: false`, any other `error.code` | a request error with that code |
| receipt `raw_text` | `raw_response` |
| `run.json` configuration and outcome provenance | `base_revision` (`model_id@revision`), `adapter_sha256` (the adapter manifest's sha256), `settings_sha256` (the configuration fields that change what the model does, without paths or devices), `image_budget` ([min_pixels, max_pixels]), `context_policy`, `coordinate_convention` original pixels, `substitute` (true for an injected test backend) |

Rows that failed scene preparation never reach the run, so they are scored as missing.

## The evaluator's own formats

These are what the converters write, and the format for Task 4's clarification episodes. References, outputs, replies and voice cases are JSON Lines, one object per line; the protocol is one JSON object. Paths inside a file are relative to that file's folder.

Coordinates follow `trackrag/dropoff/contracts.py`: original-image pixels, a point in [0, W) x [0, H), a box in [0, W] x [0, H] with positive area. Masks are PNG files, read the way `trackrag/dropoff/data.py` reads them: as greyscale, nonzero is inside, of any size. A point maps to mask pixel (floor(x · mask width / W), floor(y · mask height / H)), and the mask's box ("pixel_bbox") is its pixel extent [x0, y0, x1 + 1, y1 + 1] scaled to the image. Text files are hashed with CRLF line endings read as LF, so a Windows checkout gives the same hash; masks are hashed by their exact bytes. `instruction_sha256` is the sha256 of the JSON `{"dialogue": [...], "text": "..."}` with sorted keys, separators `,` and `:`, and non-ASCII characters written as UTF-8 rather than escaped; `evaluation.grounding.instruction_sha256` computes it.

A reference, one per category:

```json
{"case_id": "C001", "group_id": "G01", "partition": "final_evaluation", "scene_sha256": "<64 hex>",
 "recording_id": "video_204647", "frame_id": 1200, "image_sha256": "<64 hex>", "width": 2560, "height": 1440,
 "category": "clear", "expected_status": "proposed", "instruction": "Stop just after the white van", "dialogue": [],
 "mask": "masks/C001.png", "mask_sha256": "<64 hex>", "target_box": [812, 930, 1140, 1102],
 "box_rule": "pixel_bbox", "reference_point": [901, 1010],
 "anchors": [{"role": "after", "bbox_xyxy": [700, 880, 820, 960]}], "anchor_roles_ordered": false,
 "intended_mask": null, "intended_mask_sha256": null,
 "review": {"independent_reviews": 2, "disagreements_resolved": true}}
{"case_id": "A007", ..., "category": "ambiguous", "expected_status": "needs_clarification",
 "mask": null, "mask_sha256": null, "target_box": null, "box_rule": null, "reference_point": null,
 "intended_mask": "masks/A007.png", "intended_mask_sha256": "<64 hex>", ...}
{"case_id": "M003", ..., "category": "missing_reference", "expected_status": "insufficient_visual_evidence",
 "mask": null, "mask_sha256": null, "target_box": null, "box_rule": null, "reference_point": null,
 "intended_mask": null, "intended_mask_sha256": null, ...}
```

- `partition` must be exactly `final_evaluation`.
- `group_id` names cases that are not independent: the same or neighbouring frames, the same scene, or paraphrases and dialogue turns of one instruction. Intervals are resampled by group, and group IDs are part of the leakage check.
- `expected_status` is `proposed` for clear cases, `needs_clarification` for ambiguous ones, and either unresolved status for missing references. For a missing reference, asking and reporting insufficient evidence both count as correct; `expected_status` only changes the confusion table.
- The mask hash pins the file, so a mask cannot change after the references are frozen. The target box must equal the mask's box within 1e-6, and the reference point must be inside the mask; a centroid is not good enough, because it can fall outside a curved or split region.
- With `anchor_roles_ordered` true, each anchor role must be named with the right track.
- A case needs two reviews with disagreements resolved. `--allow-unreviewed` exists for development runs only.

An output in direct mode:

```json
{"case_id": "C001", "condition": {"model": "trained", "mode": "direct", "transcript": "typed"},
 "scene_sha256": "<64 hex>", "instruction_sha256": "<64 hex>", "call": {"outcome": "ok", "error_code": null},
 "result": {"status": "proposed", "point": [906.24, 1012.32], "box": [801.28, 924.48, 1149.44, 1100.16],
            "landmarks": [{"role": "after", "track_id": 6035}], "clarification": null},
 "raw_response": "{\"point_2d\": [354, 703], \"bbox_2d\": [313, 642, 449, 764]}",
 "provenance": {"base_revision": "<model>@<revision>", "adapter_sha256": "<64 hex>", "settings_sha256": "<64 hex>",
                "image_budget": [65536, 262144], "context_policy": "<text>", "coordinate_convention": "original_pixels",
                "turn_budget": null, "substitute": false}}
```

- `model` is `pretrained` or `trained`, `mode` is `direct` or `clarification`, and `transcript` is `typed`, `human` or `asr`.
- A request error is `"call": {"outcome": "error", "error_code": "timeout"}`. It counts as a request error even if a parsed result is attached. A `result` of `null` with `"invalid_reason"` is an invalid output with that reason.
- A proposal needs a valid point, a valid box and the point inside its own box. Numbers must be finite and not booleans.
- Landmarks are either `landmarks` (track IDs of the given scene, with roles) or `landmark_boxes` (boxes, as Task 5 returns them). `landmarks` is a list, `[]` when there are none; a missing key counts as naming none, but `null` is invalid. A track ID that is not in the scene counts as an invalid reference.
- A point or box attached to a non-proposal is counted but not scored.
- `raw_response` is optional. When it holds 0–1000 values (Task 5's `destination_point_xy` / `destination_box_xyxy`, or Qwen's `point_2d` / `bbox_2d`, also inside a code fence or a one-item list), the evaluator checks that they convert to the parsed pixels within 1 px.
- `substitute` must be `false`. `base_revision`, `settings_sha256`, `image_budget` and `context_policy` must be set; the last three are free values, only compared for equality. The trained model needs `adapter_sha256`, and the pretrained model must not have one. `coordinate_convention` must be `original_pixels`. Other outputs are listed as refused and scored as missing.
- One condition is one model setup: all its outputs must carry the same provenance values, or the run stops.

An episode is one clarification-mode output with all its turns. It keeps the top-level fields above: `call` is the envelope for the whole episode (an error when the harness could not run it at all). For `provenance`, copy it from the `task5-run` output of the same model and set `provenance.turn_budget` to an integer; this keeps the direct and clarification conditions comparable. `turns` replaces `result`:

```json
"turns": [{"turn": 0, "scene_sha256": "<64 hex>", "outcome": <Task 5 outcome for the first request>,
           "raw_response": null, "reply_id": null, "reply_sha256": null, "proposal_id": null,
           "confirmed_proposal_id": null},
          {"turn": 1, "scene_sha256": "<64 hex>", "outcome": <Task 5 outcome after the reply>,
           "raw_response": null, "reply_id": "A007-r1", "reply_sha256": "<sha256 of the reply text>",
           "proposal_id": "p-A007-1", "confirmed_proposal_id": "p-A007-1"}]
```

- A turn holds Task 5's outcome as `outcome`, or `call` and `result` in the direct-mode form. A Task 5 outcome must be for this case's request and come from the real model of the episode's condition (not an injected test backend or the Ollama baseline), with the same base model and adapter as `provenance`; otherwise the episode is refused.
- Every turn names the case's `scene_sha256`; a turn on another scene is a wrong scene.
- Turn 0 comes before any passenger input. Each later turn names the pre-registered reply it received and the sha256 of its UTF-8 text.
- `provenance.turn_budget` is the most model turns allowed, counting turn 0 and the turn that carries the confirmation.
- Proposal IDs and confirmations come from the application, not from the model. A proposal ID is never reused within an episode. A confirmation can only be on the last turn, and `confirmed_proposal_id` must equal that turn's own `proposal_id`. A passenger who accepts the first proposal confirms it on turn 0.

A reply: `{"case_id": "A007", "reply_id": "A007-r1", "kind": "answer", "text": "after the bus"}`. The responder rule is fixed: after a proposal the passenger's reply is a `rejection` carrying the correction; after anything else it is an `answer`. Every reply belongs to a reserved case, and one reply may be used more than once.

A voice case: `{"case_id": "C014", "human_transcript": "...", "asr_transcript": "..." or null, "recognizer": "<name and version>", "speech": "read", "noise": "real", "consent": true, "source": "<who and where>", "critical_errors": ["landmark dropped"]}`. `speech` is `read` or `spontaneous`, and `noise` is `none`, `real` or `added`. Every voice case must be a reserved case. Outputs from human or ASR transcripts need the voice file, must be for its cases, and must carry the instruction hash of that transcript.

The protocol, written by step 3: `{"version": 1, "frozen_utc", "evaluator_commit", "references_sha256", "replies_sha256", "voice_sha256", "responder_rules", "box_iou": 0.5, "landmark_iou": 0.5, "baseline_point": [0.5, 0.8], "near_constant_spread": 0.01, "bootstrap": {"resamples": 2000, "seed": 20261122}}`. Every field up to `responder_rules` is required; the replies and voice hashes are null when those files are not used. `evaluator_commit` is the commit of this repository the evaluator runs from. The run stops if the references, replies or voice file differ from the recorded hashes.

## How it scores

Each output gets one outcome, in this order: missing, request error, wrong scene, wrong instruction, invalid output, then the status the model gave.

- Clear cases: the clear-case point in mask is the main success. Box IoU with the target box is reported as a mean over all clear cases (non-proposals count as 0), as a mean over proposals, and as the share at IoU 0.5 or more. The distance of a missed point to the mask is reported too, measured at the mask's resolution.
- Ambiguous cases: whether the model asked or proposed without asking, and whether a proposal fell inside the intended mask.
- Missing references: any proposal is incorrect; asking or reporting insufficient evidence is correct.
- For each category and overall, the summary gives incorrect proposals and proposal coverage (the share of cases with a turn-0 proposal). It also gives a confusion table of expected status against outcome, clarification precision (asks on ambiguous or missing-reference cases, out of all asks) and clarification recall on the ambiguous cases.
- Landmarks: an anchor is correct when the named track's box, or the named box, has IoU 0.5 or more with it. All landmarks are correct only when the model names exactly one distinct item per anchor. Tracks or boxes and anchors are matched one to one, so a duplicate cannot fill two anchors. Ordered roles are checked for track IDs; boxes carry no roles, and those cases are counted. Unknown and extra IDs are counted. Anchors that no observation in the scene matches are reported as a vision limit.
- Clarification mode is scored twice. Initial is the turn-0 output. After intervention is the confirmed final proposal within the turn budget; for an ambiguous case it also needs a question before that proposal. The summary describes the replies as pre-registered replies from a simulated passenger. The first failed turn gives the episode its outcome. A turn without the right registered reply makes the episode invalid, but turn 0 is still scored. In a missing-reference episode, a proposal at any turn is incorrect.
- Comparisons, case by case:
  - pretrained against trained, within a mode and transcript, on correct behaviour, on the initial result and on clear-case point in mask (and confirmed clear-case point in mask in clarification mode);
  - direct against clarification, within a model, on the initial result and on clear cases, comparing the direct point in mask with the confirmed point in mask.
- A comparison is refused, with the field named, if any case was given a different scene or instruction in the two conditions (cases with a request error are skipped in this check), or if the setup differs. Across models, the setup is base revision, image budget, context policy, coordinate convention and turn budget; across modes, it is base revision, adapter, image budget, context policy and coordinate convention.
- Voice: corpus WER after NFKC, case folding, removing apostrophes and replacing other punctuation with a space. Empty or failed recognition counts as deleting every word and is also reported on its own, with a second WER over the recognised cases only. The same cases are scored with human and ASR transcripts.
- Rates come with a Wilson interval (which assumes independent cases) and, with two or more groups, a bootstrap interval over groups with the protocol's seed. Paired comparisons give discordant counts and an exact McNemar test.
- Diagnostics:
  - proposed boxes that match a predicted road-user box in the scene;
  - the spread of box sizes and of point-to-box offsets, flagged when both spreads of a kind are below 0.01 of the image once there are 5 or more proposals. A point always at the box centre is normal; with near-constant box sizes too, it suggests a fixed box drawn around a point;
  - how many raw responses were checked for coordinate conversion, and how many disagree (also a warning);
  - the share of the image each mask covers;
  - a fixed point (by default 50% across, 80% down) scored like a proposal on the clear cases, and the direct-mode score of a model that always asks or always reports insufficient evidence.
- Leakage: no reserved case ID or group ID (as a whole string, or a folder or file name in a path), scene hash or recording (anywhere in a string), or the recording next to it in time, may appear in the training manifests.

## Reading the summary

| Key | Meaning |
|---|---|
| `status` | `assessed`, or `not assessed:` with the reasons |
| `headline.conditions[<condition>]` | correct behaviour, clear-case point in mask, proposal coverage, incorrect proposals and failed-or-invalid share for each required condition, plus completion coverage in clarification mode |
| `headline.trained_vs_pretrained` | trained minus pretrained, in each mode, for correct behaviour and clear-case point in mask, with the McNemar p-value |
| `conditions[<condition>].correct_behaviour` | the initial result in direct mode, the after-intervention result in clarification mode |
| `conditions[<condition>].initial` | turn-0 outcomes, coverage, per-category results, landmarks, clarification precision and recall |
| `conditions[<condition>].after_intervention` | clarification mode only: completion coverage (confirmed proposals within the budget, right or wrong), results per category, incorrect proposals after replies, turns used |
| `comparisons` | every paired comparison; `difference_second_minus_first`, or `refused` / `not_run` with the reason |
| `fixed_point_baseline` | the fixed-point and constant-answer baselines and mask sizes |
| `warnings`, `refused_outputs`, `leakage`, `voice` | as named |

Read the pooled correct behaviour next to the clear-case point in mask and the constant-answer baselines, because asking counts as correct on ambiguous and missing-reference cases. The per-case CSV can reveal reference details, so it stays outside Git.

The tests (`tests/test_grounding_evaluation.py`) build scenes with the real scene adapter and a stand-in detector on small synthetic frames, and test these rules. The converter tests also run Task 5's own request builder, prediction command (with a stand-in model) and case loader; they skip where that code is not present.
