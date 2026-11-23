# Task 3 answerer and Task 5 integration

The contributor's answerer, privacy processor and nine starter questions are preserved from commit `4efa775`. The integration keeps the contributor commit and adds the normal Task 5 handoff, response validation and auditable provider inputs. The final README links the integrated application and the fresh 36-question release check.

## What is connected

`trackrag.answering.answer_question(question, evidence, run_dir)` now accepts the agreed three arguments. An optional injected model supports offline checks. The default client is created only after public-question, evidence-time, source, image-path and privacy checks; empty evidence makes no provider request.

The answerer reads the actual `tracks.jsonl` export instead of requiring a copied EMT test folder inside the results. Annotated and predicted observations retain their own IDs. Every processed image is saved in the question's hosted folder, and the model receives those exact files. Receipts preserve the question cutoff, source-labeled facts, frame inventory, sent-byte hashes, configured/provider model identifiers when available, raw response and errors. [Privacy policy and hosted setup](privacy.md) describe the actual masking and API configuration.

Valid answers, valid abstentions and provider/validation failures remain distinct. Unknown/duplicate citations, invented explicit track IDs, unsupported exact-frame claims and endpoints donated by unrelated tracked identities are rejected. Full scenes can still support current absence alongside historical identity evidence. These are consistency checks, not an automatic factual judge.

The reviewer exports explicitly requested observed frames as well as its beginning/middle/end representatives. A question about four named frames can therefore receive all four instead of silently losing intermediate observations. The auditor permits those additional requested images and checks each against the source observations and cutoff.

## Use the team's answerer before the trained retriever arrives

Install `requirements-hosted.txt` for hosted use. The readiness checker imports signatures without calling models or opening footage:

```sh
python -m trackrag check-adapters --source annotated --answerer gemini --retrieval-adapter tools.emt_trial_adapters
```

This selects the real Task 2 reader, Task 3 loader and Task 3 answerer, with only Task 4 replaced by the explicit TF-IDF development retriever. The saved configuration names each selected stage. Default readiness still reports the missing normal Task 1/4 implementations when those are requested.

After provider capacity is available, choose `GEMINI_MODEL` explicitly and run an authorized live check using a new output directory:

```sh
python -m trackrag review --data-root data/emt-official --recording video_054604 --questions experiments/temporal-development-questions.jsonl --source annotated --answerer gemini --retrieval-adapter tools.emt_trial_adapters --out runs/task3-live-first
```

Repeat with `video_054907` and `experiments/temporal-second-questions.jsonl`. `--answer-adapter` can override only answer generation when needed. Existing `--adapters` remains available for legacy development modules. These normal Task 3 runs use a different answerer from the earlier live trial adapters; their answer quality must be measured separately. The latest full live check of draft PR15 also remains pending after its quota stop.

## Starter questions and answer key

There are nine public questions for `video_204647`: seven intended answerable cases and two intended abstentions. All questions validate and join their separate key exactly once. The original key does not match the Task 5 scorer schema. Prepare a derived key and a blank review worksheet without changing the contributor files:

```sh
python -m evaluation prepare-annotations --questions annotations/questions.jsonl --key annotations/answer_key.jsonl --split-manifest splits/task2_manifest.json --out runs/task3-candidate-review
```

`retrieval-key.jsonl` is an **any-listed-target retrieval diagnostic**. Finding one listed pedestrian does not establish a complete pedestrian count or identity set, and interval overlap does not prove observations at all four requested timestamps. `qa-review.jsonl` and `worksheet.md` retain the expected answers, complete candidate ID lists and exact requested frames for separate review. Every review starts pending. No answer key is loaded by the inference path.

An independent AI-assisted check against the pinned original EMT annotation file agrees with the annotation claims in all seven answerable rows. Five reference answers are checkable from labels; two still need images for their visible/crossing/appearance claims. The labels do not establish personal names or intended destinations, consistent with the proposed abstentions. This is annotation-only agreement, not independent human validation. The [source review](../experiments/task3-key-source-review.json) records the checked frames and source hashes.

The Task 2 manifest assigns this recording to `test`. Keep it out of fine-tuning. Reading its development labels here does not certify an untouched final evaluation set; preserve separate evaluation data where required.

Questions naming human annotation IDs are source-specific diagnostics. Predicted tracks have independent IDs: use source-appropriate questions or a separately verified association for evaluation, rather than treating an annotated ID as a prediction ID. Include natural-language questions that do not depend on hidden annotation numbering in the final application evaluation.

## Measured integration and remaining work

The source passes 215 software tests and 70 subtests, including the actual installed Google SDK through a local HTTP test transport. Negative tests now reach the intended citation/timeout paths, and malformed provider JSON retains an error receipt. Tests also cover predicted IDs without human annotation access, direct future-evidence rejection, four-frame export, correct abstentions and valid scene-backed absence.

A completed real-data handoff uses the normal command on both staged EMT recordings and all 36 development questions. It exercises the real reader, evidence exporter, privacy processor and Task 3 client with the actual SDK and local mocked HTTP. All 255 processed-input byte hashes match the captured SDK images, and all 399 audited artifacts match their saved fingerprints. Its synthetic replies are not Gemini answers or quality measurements; no live provider requests were made. The [committed integration summary](../experiments/task3-handoff-summary.json) records source commit `6df710d`, captured-input hashes and exact local artifacts.

Still required: inspect the starter source images and obtain independent human review; run the normal Task 3 path live after quota permits; connect actual Task 1 detector/tracker outputs and the trained Task 4 Qwen model; compare fixed-generator baseline/trained retrieval on independently reviewed evaluation data. Task 3's hosted generator does not supply the assessed local NLP fine-tuning milestone.
