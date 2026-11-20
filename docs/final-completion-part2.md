# Part 2: final project completion

Agreed on 20 November 2026. Submission is Tuesday, 24 November 2026, Dubai time; the exact submission time has not been supplied. Aim to finish the technical release on Monday, 23 November. Prepare the reports in parallel.

These are continuations of the original five tasks, identified by their **previous owners**. They finish the whole technical project: the 901 vision pipeline, the existing footage-question application, the new trained passenger-instruction component, evaluation and a runnable demonstration. Voice is one input to the new component.

All five issues stay open until their completion checks pass. This document assigns work; it does not claim that the work or any new model training is already complete.

## What the finished application must do

1. **Existing route:** recording and question -> genuine predicted tracks -> retrieval -> original-frame evidence -> answer with citations, or a distinguishable insufficient-evidence/provider-error outcome.
2. **New route:** selected recorded scene and typed/spoken drop-off instruction -> trained language-grounding model -> proposed destination point and coarse region box -> clarification when needed -> explicit confirmation of the current proposal.

The new region represents the interpreted passenger destination. It does not establish physical clearance or legal stopping suitability. Use recorded scenes and paused frames for the final prototype; vehicle control is outside the approved scope.

The existing QA retriever can remain its accurately identified lexical baseline. **The adopted NLP fine-tuning objective is now passenger-instruction grounding, replacing the earlier separate Qwen-reranker training milestone.** Keep the useful retrieval work and existing QA route. Do not describe the lexical retriever, unfinished embedding experiment, hosted Gemini calls or frozen speech recognizer as a trained local NLP model.

For this deadline, predict a coarse destination box and an interior destination point. Evaluate each against its stated reference. A box around a vehicle is not a destination, and a fixed circle drawn around a point is not a learned region.

## Ownership and handoffs

| Previous owner / existing issue | Part 2 responsibility | Main handoff |
| --- | --- | --- |
| [Task 1](https://github.com/A1Prep/TrackRAG-Gulf/issues/1) | Vision handoff and independent evaluation | Scene inputs to Task 4/5; independently scored results |
| [Task 2](https://github.com/A1Prep/TrackRAG-Gulf/issues/2) | Data, destination annotations and grouped splits | Training/development data to Task 5; reserved references to Task 1 |
| [Task 3](https://github.com/A1Prep/TrackRAG-Gulf/issues/3) | Finish existing QA and speech input | Working cited answerer and transcripts to Task 4; reviewed references |
| [Task 4](https://github.com/A1Prep/TrackRAG-Gulf/issues/4) | Complete application, interaction and release | Both working routes, regression checks and startup instructions |
| [Task 5](https://github.com/A1Prep/TrackRAG-Gulf/issues/5) | Actual NLP grounding training and inference | Pretrained/trained model interface and reloadable learned parameters |

The main ownership change is deliberate: the previous Task 5 owner takes NLP training, and the previous Task 4 owner takes final application integration. Preserve the original authors' contributions when integrating existing code.

### Agree the shared records first

Task 4 should publish a small shared-contract commit first, agreed with Tasks 1, 2, 3 and 5. The following are **interfaces to implement**, not existing callable functions:

- **Scene input:** recording and original frame identifiers, timestamp, original image path/bytes and dimensions, genuine predicted observations available at that frame, and model/source provenance. Use the existing Frame and TrackObservation records where applicable.
- **Instruction request:** request/session identity, scene reference, typed or recognized text, input mode, and relevant prior dialogue. Preserve the recognized transcript separately from any user correction.
- **Grounding result:** status of proposed, needs_clarification or insufficient_visual_evidence; original-image destination point and coarse box when proposed; observed landmark references when available; clarification information; and model/configuration provenance.
- **Grounding call outcome:** carry a successful semantic result separately from an operational error. An unavailable checkpoint, timeout, memory failure or malformed response produces an error code/receipt with no proposal. It is not a valid abstention or clarification decision, and must remain in evaluation denominators.
- **Interaction state:** pending clarification, a proposal identifier, recording/frame binding, confirmation/cancellation and transcript corrections. The application creates proposal identifiers; the model does not invent confirmation events.

Use original-image pixel coordinates at module boundaries. Convert any model-internal normalized coordinates inside the model adapter. Validate recording/source/frame identities, coordinate bounds and finite values. Do not merge this new instruction record into the existing public Question schema.

No inference input may contain reference destinations, masks, answer keys, relevance labels or future observations. Gold labels belong only in training targets or scoring. Metadata alone does not supply vehicle colour; the grounding model must inspect the image.

## Task 1 - vision handoff and independent evaluation

**Owner: whoever completed the previous Task 1.** Continue from the verified detector/tracker; do not restart completed training.

### Build

1. Provide a scene-loading adapter that opens the exact original frame through the shared reader and supplies genuine predicted observations with their original coordinates and identities. Reject mismatched recording/source/frame metadata. The final predicted-input route must work with annotation files absent.
2. Make both vision profiles reproducible with explicit configuration, checkpoint paths/hashes and commands. Real saved predictions can be reused after strict validation; label them as saved predictions rather than fresh inference.
3. Recover the raw detection JSONL omitted from the supplied bundles and reproduce detection AP with the same evaluation rules. Reuse the completed full-test tracking verification. If original detections cannot be recovered, document the gap and the fresh-inference reproduction command; never calculate detection AP from track boxes.
4. Implement the independent grounding evaluator. Consume saved model outputs and separate references. Measure point-in-reference-mask success, coarse box overlap against the documented target, landmark/pair correctness on cases with those labels, invalid outputs and failed model requests.
5. On the reviewed interaction subset, compare pretrained/direct, pretrained/clarification, trained/direct and trained/clarification. Keep scenes, inputs, model family, confirmation opportunity and declared interaction budget matched. Report initial grounding separately from success after passenger intervention.
6. Compare human and ASR transcripts on the same voice cases. Report incorrect proposals and proposal/completion coverage, so abstaining on everything cannot look like success.

### Files and handoffs

Reuse trackrag/vision, the strict saved-prediction adapter and current vision configurations. Suggested new ownership: the scene adapter under trackrag/dropoff/scenes.py and independent scoring under evaluation/grounding.py. Record compact result summaries under experiments/dropoff; keep large artifacts outside Git.

Give the scene adapter and one real example to Tasks 4/5 early. Receive reserved evaluation references from Task 2, reviewed dialogue/voice targets from Task 3, and frozen model outputs from Task 5. Do not expose evaluation labels to inference.

### Verify and finish

- [ ] Exact frame lookup and original-coordinate overlays work on at least two real recordings.
- [ ] A wrong recording, a foreign track source, a future observation and a missing frame are handled explicitly.
- [ ] The scene adapter works without human annotations and does not invent colour or identity facts.
- [ ] Vision configurations, artifact locations and evaluation commands are supplied; archived versus independently reproduced measurements remain distinguishable.
- [ ] Grounding scorer fixtures include correct/wrong points, an interior point versus an invalid centroid, wrong landmark pairs, wrong scenes, empty proposals and invalid boxes.
- [ ] The final frozen conditions are independently scored; failures, abstentions, request errors and evaluation denominators remain visible.

**Complete when:** Task 4 can consume genuine scene inputs and the team has reproducible vision/grounding evaluation artifacts. Report any unrecovered detection-AP input as a visible limitation, not a reproduced result.

## Task 2 - complete data, destination annotations and splits

**Owner: whoever completed the previous Task 2.** The existing EMT reader is reusable; the missing work is instruction-to-destination supervision.

### Build

1. Validate a public grounding-data loader that joins each training image, instruction and destination mask correctly. Begin with genuine training records, not the concise public test set. Check actual image/mask dimensions and nonempty valid masks before exporting examples.
2. Use a suitable subset of Talk2Car-RegSeg training data. A scheduling target is 500-1,000 validated public examples; filter or represent complex instructions honestly rather than assigning every navigation command a false drop-off label.
3. Prepare approximately 80-120 reviewed local instruction cases where the images support the task. Include unique destinations, two-landmark references, multiple plausible references, missing/occluded landmarks, corrections and contrasting instructions on the same scene. Ask Task 3 to review references and disagreements.
4. Preserve masks. Derive the target point from an interior mask pixel, and document the mask-to-coarse-box rule. A naive centroid can be outside a non-convex destination mask.
5. Export explicit training, development and final-evaluation manifests. Reserve approximately 30-40 reviewed new cases for evaluation, including clear, ambiguous and missing-reference categories. These counts are scheduling targets, not rubric requirements or a guarantee of model quality.
6. Group related images, adjacent frames, paraphrases and dialogue turns together. Preserve EMT recording-level separation. Establish the public source-scene mapping before claiming scene-independent results. The concise RegSeg test reuses validation images: do not train/select/calibrate on them or pool overlapping evaluation views as independent scenes.
7. Record which fields are genuinely labelled. Public region masks do not automatically label anchor identities, ambiguity or dialogue responses. Keep model inputs and gold fields separate.

### Files and handoffs

Suggested ownership: tools/dropoff_prepare_data.py, annotations/dropoff schemas/manifests and splits/dropoff_manifest.json. Keep raw images/audio and bulk datasets outside Git, with source revisions, attribution and access instructions.

Give Task 5 a first batch of approximately ten validated training examples today, followed by the versioned training/development release. Give Task 1 reserved evaluation references after review. Coordinate the instruction and speech-case identifiers with Task 3.

### Verify and finish

- [ ] Image, instruction and mask joins are checked against actual pixels.
- [ ] Target points lie inside their masks; derived boxes and coordinate conversions have deterministic checks.
- [ ] Duplicate/source-group overlap checks pass across partitions; missing grouping information is reported.
- [ ] All held-out targets and ambiguous new cases have two independent human reviews with disagreements resolved.
- [ ] Exported training inputs exclude gold answers; targets contain only the supervision actually available.
- [ ] The loader handles corrupt/missing inputs and excludes invalid examples with an explicit receipt.

**Complete when:** Task 5 can load the versioned training/development corpus, and Task 1 has a reviewed, separately reserved evaluation set with usable references.

## Task 3 - finish existing QA and speech input

**Owner: whoever completed the previous Task 3.** Finish the original answerer as well as the voice input layer.

### Build

1. Run the normal integrated Task 3 answerer with genuine retrieved evidence from the final vision/QA route on at least two real recordings. Obtain successful, independently reference-checked cited answers on each recording and at least one valid insufficient-evidence outcome. Provider failures alone do not satisfy this gate. Earlier prototype live answers and SDK-mocked integration are different conditions; preserve that distinction.
2. Review the existing reference answers against original footage. Resolve the pending visual claims. Keep reference keys separate from public questions, and retain insufficient-evidence cases. Use natural-language references suitable for independently numbered predictions instead of requiring a user to know hidden annotation IDs.
3. Investigate actual provider/configuration, response-format, citation and exact-frame failures. Retain raw outcomes and distinguish network/quota/unavailable errors from an incorrect answer and a supported abstention. Verify that cited evidence opens the claimed frame and road user.
4. Add one fixed speech recognizer for microphone or uploaded audio. Preserve source audio, transcript and timing, expose correction, and supply the corrected text to the same model interface used for typed instructions. Speech recognition is not a second fine-tuning project.
5. Collect approximately 30 consented real recordings from at least three speakers for frozen scene/instruction cases. Save human transcripts and inspect errors in colours, directions, counts and negation. Tell Task 1 whether speech is read or spontaneous and whether noise is real or added.
6. Independently review Task 2's held-out and ambiguous references. For ambiguous interactions, record intended destinations/replies before seeing the model proposal. A participant must not manufacture an intention from the model's output.
7. Record a few brief rider/user observations about where landmark-based drop-off descriptions help or confuse. Keep the sample and limitations explicit; these are exploratory observations, not proven market-size or time-saving claims.

### Files and handoffs

Continue owning trackrag/questions.py, trackrag/answering.py, QA references and their tests. Add a speech adapter such as trackrag/speech.py with its own tests and recording manifests. Task 4 owns the interaction state/UI; provide transcripts and reviewed clarification scenarios to it.

Give the QA configuration and working evidence-to-answer call to Task 4. Give voice cases/references to Task 1 for matched text/ASR evaluation. Do not expose final reference answers to Task 5's model input.

### Verify and finish

- [ ] The final answerer produces successful live, reference-checked cited answers on each of two recordings and a valid insufficient-evidence outcome. Failed requests remain visible but do not satisfy this check.
- [ ] An answer key can be removed without breaking inference.
- [ ] Empty evidence, invented citations, unsupported exact frames and provider errors have distinct tested outcomes.
- [ ] Typed and recognized text reach the same grounding interface; transcript correction is preserved.
- [ ] Missing microphone/audio, empty or failed transcription and critical recognition errors do not silently create a confirmed proposal.
- [ ] Real recordings and reviewed references are available to the evaluator, with explicit consent and source information.

**Complete when:** the original QA route has successful, reference-checked final-path answers on both recordings, a valid abstention and retained failures, and the application receives working, editable speech transcripts for the new route.

## Task 4 - finish the complete application and release

**Owner: whoever completed the previous Task 4.** Continue supporting the retrieval baseline, and take final application integration while the previous Task 5 owner trains the new NLP model.

### Build

1. Publish the shared drop-off contracts first. Keep the existing Question/RankedInterval interfaces compatible. Put the new records in a separate module such as trackrag/dropoff/contracts.py, importing the current shared frame/track types.
2. Build one simple local application with recording/frame selection and both routes. Reuse the existing reviewer, evidence exporter and answerer for footage questions. Add instruction text/audio, an editable transcript, the proposed destination overlay, clarification replies, correction, confirmation and cancellation.
3. Connect Task 1's genuine scene adapter, Task 3's speech/QA modules and Task 5's model interface. A controlled substitute is acceptable in wiring tests while work proceeds; final acceptance must use the real artifacts. The new trained route must not silently substitute Gemini, static coordinates or question-ID lookups.
4. Own the dialogue controller. Ask a distinguishing clarification for unresolved references. Send the reply and relevant context back to the model. Bind confirmation to the exact proposal and recording/frame; invalidate it if the proposal or scene changes. Bare 'here' without a pointing/selection context must remain unresolved.
5. Validate all returned coordinates, identities, image references and cutoffs. Reject malformed outputs rather than drawing misleading proposals. Keep checkpoint/load failures, timeouts and invalid responses in the agreed operational-error envelope; never turn them into semantic abstentions. Do not invent model confidence or interpret a lexical score as a probability.
6. Preserve existing QA behaviour and measured failures. Run the final normal QA regression bank on genuine predictions with the real answerer. Preserve every requested row, including empty evidence, provider failures and any unfinished requests; do not report an incomplete cohort as a full quality result.
7. Package a reproducible startup command, dependency/configuration instructions, model/data locations and a demonstration walkthrough. Keep secrets and large weights outside Git. If using a self-hosted model endpoint, record its model/checkpoint identity and make it accessible to the intended teammates.
8. Rehearse after a fresh process start with another teammate. Distinguish fresh inference, validated cached vision predictions and saved historical QA outcomes in the interface and run receipts.

### Files and handoffs

Suggested new ownership: trackrag/dropoff/contracts.py, interaction.py and app.py, with integration tests. Coordinate changes to the existing reviewer/entry points and preserve original authorship. Supply one documented application entry point; paths above describe files to implement, not commands already available.

Task 5 supplies the actual model callable/checkpoint. Task 1 supplies scene inputs and independent scoring. Task 2 supplies data manifests; Task 3 supplies QA and speech. Merge small working interfaces early instead of waiting for a final all-at-once handoff.

### Verify and finish

- [ ] Both routes work through one application on at least two real recordings.
- [ ] The drop-off route works with typed input, voice, transcript correction, clarification, cancellation and explicit confirmation.
- [ ] Wrong recording/source, future frames, malformed coordinates and stale confirmations are rejected.
- [ ] The final drop-off route actually loads/calls the trained model and does not require the hosted QA provider.
- [ ] Existing QA has the successful two-recording acceptance results from Task 3; regressions retain errors and unanswerable outcomes, and cached outcomes are labelled if displayed.
- [ ] Automated checks and an independent review of boundaries, saved evidence and scoring are complete.
- [ ] Another teammate can restart and run the application using the supplied instructions and accessible artifacts.

**Complete when:** the integrated application and release package are usable by another teammate, with final run artifacts, known limitations and reproducible startup instructions.

## Task 5 - train and deliver the NLP grounding model

**Owner: whoever completed the previous Task 5.** This is now the actual learned NLP component, not another general planning or speech task.

### Build

1. Obtain a working GPU environment with adequate storage today. Use a separate training environment so changes do not break the verified vision/reviewer setup. Record the actual GPU, package versions, memory and run times.
2. Begin with the planned image-and-language model candidate, Qwen3-VL-8B-Instruct, subject to a real feasibility/quality pilot. Pin its revision. Do not call it the best or newest model without task-specific evidence.
3. Run a small genuine gradient-update pilot on training examples, save the adapter, reload it and produce a spatial prediction through the intended interface. Verify learned parameters actually changed. A toy substitute, successful model import or generated JSON does not satisfy this milestone.
4. Establish pretrained inference on development cases. Fine-tune a single grounding adapter on the validated training partition, including the reviewed local instruction cases. Use development data for selection; reserve final references for Task 1.
5. Use the image, instruction and relevant observed context to predict the intended location and unresolved status. Learn the language-conditioned interpretation; do not replace it with keyword-to-coordinate rules or a midpoint between detected vehicles.
6. Use separate target formats or masked losses for partial supervision. A public mask-only example can teach destination coordinates; it must not create false ambiguity, anchor or dialogue labels. Train real model parameters in the language-conditioned grounding component.
7. Deliver a callable accepting the agreed scene, utterance and dialogue context and returning the agreed structured result or operational-error envelope. Convert internal normalized coordinates to original pixels. Validate malformed/unobservable outputs and preserve failures; do not train or score a model-load failure as a correct semantic abstention.
8. Save the base revision, training/data manifest, seeds/settings, image processing, coordinate convention, trainable modules, logs and adapter/checkpoint hash/location. Make the artifact loadable by Task 4 without hidden owner-local files.
9. Freeze the trained checkpoint and inference settings before final evaluation. Give Task 1 matching pretrained/trained outputs under the same image budget, candidate/context policy and coordinate rules. Report non-improvement honestly.

### Files and handoffs

Suggested ownership: trackrag/dropoff/model.py, tools/dropoff_train.py, drop-off model configurations, training requirements and compact experiment records. Keep learned weight files outside Git with accessible locations and hashes.

Start with Task 2's first validated training batch; do not wait for its full annotation release. Coordinate the interface with Task 4 immediately. Give Task 1 frozen model configurations and unedited outputs for independent evaluation.

### Verify and finish

- [ ] A genuine training/update/save/reload pilot succeeds on the actual GPU.
- [ ] Training inputs exclude held-out targets, reference answers and future information.
- [ ] The learned artifact is nonempty, bound to the stated base model and reloads after a fresh process start.
- [ ] Unseen development instructions and meaning-changing contrasts are exercised; JSON validity alone is not counted as language understanding.
- [ ] Task 4 can call the real pretrained and trained interfaces, with explicit model provenance and a tested error outcome for unavailable checkpoints, timeouts and malformed responses.
- [ ] Task 1 receives the frozen conditions for final scoring; no evaluation case is silently removed.

**Complete when:** another teammate can load the trained parameters, the application uses them for new instructions, and the independent evaluator has real pretrained/trained outputs.

## Four-day delivery sequence

| Date, Dubai time | Required shared result |
| --- | --- |
| Friday, 20 November | Contracts agreed; a real scene handoff and first validated training examples available; real GPU update/save/reload pilot completed. |
| Saturday, 21 November | Data groups/splits frozen; adaptation running; the first complete typed interaction connected with an explicitly identified model condition. |
| Sunday, 22 November | Trained model reloaded in the application; voice and clarification/confirmation connected; development settings and evaluation protocol frozen. |
| Monday, 23 November | Independent comparisons and QA regressions completed; genuine integration defects repaired; final artifacts, startup instructions and demonstration package prepared. |
| Tuesday, 24 November | Submission. Avoid depending on last-minute training or interface changes. The exact deadline time still needs confirming. |

Reports and slides should progress alongside implementation. This sequence is tight and conditional on a usable GPU and reviewed labels arriving promptly. If the first-day training pilot cannot run, record the fine-tuned NLP deliverable as at risk; a pretrained/API-only demonstration is not equivalent.

## Shared project acceptance checklist

- [ ] **901:** trained visual artifacts, genuine predicted-input scene route, reproducible commands and honestly labelled evaluation evidence.
- [ ] **Existing QA:** successful live cited answers on each of two recordings through the normal predicted-evidence route, independently checked references/citations, a valid abstention and retained provider/answer failures.
- [ ] **New NLP:** actual parameter training, save/reload evidence, meaningful image-and-language grounding and original-coordinate destination outputs.
- [ ] **Interaction:** typed/spoken requests, visible transcript correction, useful clarification and confirmation bound to the current proposal.
- [ ] **Evaluation:** fixed pretrained/trained and direct/clarification conditions on reserved groups; separate initial versus interaction-assisted success, coverage, errors, voice effects and limitations.
- [ ] **Reproducibility:** a fresh-start run by another teammate, accessible model/data artifacts, pinned configurations, automated checks and independent integration review.
- [ ] **Release:** both routes in one runnable demonstration, no answer-key access during inference, no hidden stand-in in the assessed path, and known failures preserved.

After these checks pass, the remaining deliverables are the reports, presentation material and rehearsal. Defects discovered during acceptance belong to these same tasks; they do not justify declaring the technical project finished early.

## Existing evidence and sources

- [Verified Task 1 artifacts](task1-artifact-verification.md): actual checkpoint reload, two-recording real inference, full-test tracking recomputation and the remaining raw-detection/AP limitation.
- [Assessment alignment](assessment-alignment.md): written rubric coverage, reported lecturer preferences and current training milestones.
- [Task 3 integration](task3-integration.md): separate code/SDK checks, reference-review needs and final live-answer acceptance.
- [Task 4 preparation PR20](https://github.com/A1Prep/TrackRAG-Gulf/pull/20): useful candidate/review tools, with Qwen inference/training explicitly unfinished.
- [Earlier research plan PR22](https://github.com/A1Prep/TrackRAG-Gulf/pull/22): research and source audit. Its seven-day schedule and pre-verification Task 1 status are historical; this four-day continuation controls current assignments.
- [Talk2Car-RegSeg](https://rnr-t2c.github.io/) and [dataset/code release](https://github.com/kanji1011/Talk2car-Refseg): destination-region data; preserve source attribution and licensing.
- [Official Qwen training framework](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune): grounding formats and LoRA support. Official support is not proof that the selected checkpoint fits the available GPU or meets this task's quality needs.
