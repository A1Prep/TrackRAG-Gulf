# Checking retrieval before connecting the answerer

This command connects the real Task 2 EMT reader and Task 3 question loader to a selected retriever. It saves a report and ordinary Task 5 evidence folders, without calling an answerer or reading a gold key. It checks the handoffs and looks for changes caused by future inputs. It does not measure whether a retrieved vehicle or an answer is correct.

## Run the local baseline now

From the repository root, with integration dependencies installed:

```sh
python -m evaluation verify-retrieval --data-root data/emt-official --questions experiments/temporal-development-questions.jsonl experiments/temporal-second-questions.jsonl --source annotated --retrieval-module tools.emt_trial_adapters --out runs/retrieval-check-01
```

Choose a fresh output directory. This uses original EMT images and human annotations, with the explicitly selected local TF-IDF retriever. It does not use the final detector or fine-tuned Qwen model. Both recordings in these question files have been used for development; this is not held-out evaluation.

To check Task 4 when it arrives, omit `--retrieval-module`. The default imports `trackrag.retrieval.build_index` and `rank_evidence`. Model initialization and any service calls belong to the selected adapters; the runner does not make an arbitrary adapter offline.

## Connect saved Task 1 predictions

Task 1 can export its observations to JSONL rather than making everyone load its model:

```sh
python -m evaluation verify-retrieval --data-root <EMT_ROOT> --questions <QUESTIONS.jsonl> --source predicted --predicted-tracks <PREDICTIONS.jsonl> --out runs/predicted-retrieval-01
```

Each row follows `TrackObservation` in `trackrag/contracts.py`: `recording_id`, original `frame_id`, detector/tracker `track_id`, `class_name`, original-pixel `bbox_xyxy`, and `source: "predicted"`. Multiple files can follow `--predicted-tracks`. IDs are local to a recording and source. Keep predictions separate from human annotations; the runner does not rename annotated IDs or rescale boxes. It rejects mixed sources, foreign recordings, duplicate track/frame rows, unavailable frames and boxes outside the image.

The report hashes the input files and records that predictions were supplied. It cannot authenticate the model that produced them. A recording without saved rows is inconclusive: this format cannot distinguish a missing export from genuine zero detections. The automated predicted-handoff test uses synthetic fixtures, not real model predictions. Without `--predicted-tracks`, predicted mode calls `trackrag.vision.predict_tracks` and records that the adapter was called; that flag alone does not prove model provenance.

## What the checks do

The normal baseline sees the complete original recording and observations. Returned intervals must already satisfy the shared contract before evidence export: exactly the public ranked fields, at most three intervals, matching recording/source, ordered ranks, finite scores, available tracks or exact scene frames, and no interval after the query time. Invalid outputs are recorded as failures rather than clipped into valid ones.

For queries before the final available frame, the runner compares:

1. Repeated calls on the same index, then an index rebuilt from identical inputs.
2. A pass-through recording wrapper implementing the shared `Recording` protocol, using the same query sequence as each later variant.
3. A rebuilt index after changing only future class names/boxes and adding a future-only track.
4. A rebuilt index with only frames and observations at or before the query.
5. A rebuilt index with future image paths replaced by generated checkerboard images. Original footage is never edited.

Comparisons preserve raw IDs, source, frame spans, rank and scores. Only scores have a configurable finite tolerance, default `1e-6` applied as both absolute and relative tolerance. Stable empty outputs are reported separately from positive checks. A query at the last frame has no future-input test. Unstable repeat/rebuild controls or wrapper-dependent behavior make the affected comparison inconclusive.

The report separately counts zero-score scene contexts and their temporal comparisons. These can provide images when lexical search has no match, but they do not establish a relevant match or an answer. See the [five-question follow-up](scene-context-follow-up.md) for the current development-baseline behavior and regression protocol.

The tests include deliberately leaky adapters, so a failure to notice future observations or future pixels would break the software checks. Passing these perturbations is still not proof of no leakage: hidden caches or an untested dependency can mask it. Use isolated or correctly invalidated caches for real models. Frozen observations may themselves reflect future-dependent tracking; these checks do not establish Task 1's causality.

## Read the results

Open `report.md` for the summary or `report.json` for raw baseline rankings and failed comparisons. Invalid returned rankings retain a bounded diagnostic representation, including nonfinite values that cannot be serialized as valid intervals. A structurally valid baseline is exported into `runs/recording-01`, etc., using the ordinary evidence writer and saved-artifact audit. These exports replay the captured baseline without running retrieval again. Answers have status `not_requested`. Empty or failing runs stay visible, and failures in one recording do not prevent the other recording being checked. A mixed cohort with last-frame queries or stable-empty temporal results is `passed_with_gaps`; a cohort without any positive temporal comparison is inconclusive.

The command exits `0` for `passed` or `passed_with_gaps`, `1` for a failure, and `2` for an inconclusive suite. A passing status describes only the checks performed. It is not a model-quality gate. Retain the report and inspect gaps; then use the separate gold scorer and human answer reviews for accuracy.

## Recorded development check

The final run used code commit `ed1bec1fba9825d305b52df196fd9ac7a3c25bb0` and the command above with output `runs/task5-retrieval-acceptance-02`. The [committed summary](../experiments/retrieval-acceptance-summary.json) records the exact command, input/content hashes, gaps and local report hash. Original images and run folders stay outside Git.

| Recording | Original frames / labeled boxes | Questions | Nonempty rankings | Positive temporal comparisons | Stable-empty temporal comparisons |
| --- | ---: | ---: | ---: | ---: | ---: |
| video_054604 | 150 / 2,681 | 24 | 23 | 36 | 0 |
| video_054907 | 150 / 2,844 | 12 | 8 | 15 | 9 |

Both recordings returned `passed_with_gaps`, with no contract or temporal-perturbation failure. Five questions produced empty rankings. Sixteen final-frame queries had no future test. The 51 positive comparisons cover 17 earlier queries, with three perturbations each; they are not 51 independently reviewed questions. All 104 exported result, observation, metadata and image files were byte-identical to the first run. Representative marked images from both recordings were inspected for frame/box export.

These runs use human labels and the local TF-IDF development baseline. No answer model was called, and retrieval relevance/answer accuracy was not scored. The full software suite passes 171 tests and 58 subtests; independent review ran 49 focused tests and 16 subtests without a remaining blocking issue. Windows and Linux CI also passed. Genuine detector/tracker predictions, the fine-tuned Qwen retriever, Task 3 answerer and reviewed held-out questions are the next integration inputs.
