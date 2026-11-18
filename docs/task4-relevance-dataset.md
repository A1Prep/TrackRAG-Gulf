# Task 4 milestone 3: relevance review and dataset preparation

The review/export workflow is implemented. **The real relevance labels are still
pending human review.** Preparing a worksheet, passing tests or copying images does
not create a human-reviewed training dataset or implement Qwen.

[retrieval_dataset.py](../trackrag/retrieval_dataset.py) consumes the versioned
[candidate packets](task4-candidates.md). It creates a local worksheet and exports
only explicitly reviewed human labels for a selected recording split. It does not
read Task 3's answer key, turn existing AI-assisted reviews into human labels, or
modify the normal retrieval/answering interfaces.

## Prepared pilot

The local worksheet is `runs/task4-relevance-review-01/review.html`. Open it in a
browser alongside its packaged `images` folder. It contains:

| Question | Recording | Declared split | Candidates |
| --- | --- | --- | ---: |
| temporal-09 | video_054604 | train | 61 |
| second-07 | video_054907 | dev | 169 |

All **230 judgments are pending**. The pack contains 98 distinct original image
files, copied without pixel changes and bound by SHA256. No test recording was
included. These two questions are a workflow pilot, not a sufficiently diverse
training/evaluation corpus or an untouched held-out test.

The [preparation summary](../experiments/task4-relevance-preparation-summary.json)
records counts, source hashes, split assignments and checks. Images, candidate
snapshots, review drafts and future dataset exports remain in Git-ignored folders.

## How to review

1. Enter your name. Expand a candidate to inspect its question, source, vehicle,
   interval, observation gaps and images. Red rectangles mark the target vehicle;
   scene candidates have no target box.
2. Choose **relevant**, **irrelevant**, **uncertain**, or leave **pending**.
3. Choose a reason and explain the evidence in the rationale field. Mark that you
   inspected all shown frames only after doing so.
4. Before exporting new or changed judgments, confirm the named-human-reviewer
   assertion. Click **Download all labels** and save `labels-reviewed.jsonl`.
5. Keep the entire downloaded file, including pending rows. To continue later, use
   **Resume from labels JSONL**. Unsaved page edits are lost when the page closes.

Relevant means the displayed candidate materially supports the question. It need
not independently answer every part of a multi-evidence question. Correct vehicle
class or identity alone is insufficient if the interval shows the wrong time.
Use reasons such as `wrong_vehicle` and `wrong_time` for reviewed difficult negatives.
Do not mark an unseen or uncertain candidate irrelevant merely to fill the dataset.

A first annotation is not proof of entry. Missing labels are not proof of physical
absence. A broad history is not automatically better evidence, and scenes do not
establish hidden contents or physical speeds. If the selected views cannot establish
the claim, record uncertainty and the limitation; resolve ambiguous questions with
the question owner before using them for training.

The worksheet displays original, unmasked images locally. It has no remote assets,
uploads or model calls. Reviewer identity is a recorded assertion, not authentication.
Unchanged imported judgments retain their existing reviewer and provenance; importing
AI-assisted judgments does not turn them into human judgments.

## Commands

Run from the repository root using the isolated environment and existing integration
dependencies. Choose new output directories; neither command overwrites a pack.

Prepare a new pack from candidate JSON files:

```powershell
.\.venv\Scripts\python.exe -B -m trackrag.retrieval_dataset prepare --candidates runs/task4-candidates-verification-01/temporal-09-candidates.json runs/task4-candidates-verification-01/second-07-candidates.json --split-manifest splits/task2_manifest.json --out runs/task4-relevance-review-02
```

After a human saves reviewed labels, export each partition separately:

```powershell
.\.venv\Scripts\python.exe -B -m trackrag.retrieval_dataset export --review runs/task4-relevance-review-01 --labels <PATH_TO_LABELS_REVIEWED.jsonl> --split train --out runs/task4-relevance-train-01
.\.venv\Scripts\python.exe -B -m trackrag.retrieval_dataset export --review runs/task4-relevance-review-01 --labels <PATH_TO_LABELS_REVIEWED.jsonl> --split dev --out runs/task4-relevance-dev-01
```

Pending, uncertain and AI-assisted rows are excluded, never converted into negative
human labels. A completely pending pack fails export with an explanation and creates
no dataset directory. Partial human review can be exported, with coverage explicitly
reported. A small or one-class export is not proof that enough training data exists.

## Data bindings and export format

A review pack contains `pack.json`, copied candidate batches, content-addressed
images, a pending `labels.jsonl` template and `review.html`. Its hash binds the split
manifest snapshot, candidate metadata/files and original image bytes. Labels bind
that pack hash, case ID and candidate ID. Different track sources remain distinct.

Preparation validates the public four-field questions, complete candidate membership,
identity/source, observations, frame cutoff and image-sampling policy. It checks actual
image dimensions and verifies normalized endpoints/displacement against source boxes.
Center range is checked against sampled geometry and exactly when every observation
has image dimensions available; unsampled dimensions are not invented. These checks
do not establish the factual correctness of relevance judgments or source annotations.

Exports reject altered snapshots, stale labels, unknown or duplicate rows, missing
candidate judgments, inconsistent review reasons and incomplete inspected-frame
records. Every original candidate must remain represented in the labels file, even
when pending. Empty-pool questions also remain visible in summary accounting.

`examples.jsonl` contains pointwise examples with three separate fields:

- `model_input`: public question, source, candidate and associated observed history.
- `relevance`: `1` for reviewed relevant, `0` for reviewed irrelevant.
- `provenance`: pack/case/candidate bindings, split and the review record.

Only `model_input` belongs in a model prompt. The label and reviewer rationale are
training/evaluation metadata, not inference features. Required image assets are copied
beside the export. `summary.json` records positive/negative counts, all cohort judgments,
excluded categories, per-question coverage and byte hashes.

Cases with no reviewed positive are distinguished from fully human-labeled cases
without a positive. The latter still do not prove a pool miss: the question itself
may be ambiguous or unanswerable. Check those cases before assigning that diagnosis.

## Split discipline and remaining work

Partitions come from the exact supplied [recording manifest](../splits/task2_manifest.json),
not random candidate rows. The exporter uses its bound snapshot, so later edits to
the original manifest do not silently change an existing dataset. If split assignments
are corrected, create and review a new pack. Never edit a pack to move test data into
training. The Task 3 starter recording `video_204647` is assigned to test and must stay
out of fine-tuning.

Before training, obtain enough reviewed positive/negative examples across appropriate
training recordings, include paraphrases and difficult wrong-vehicle/wrong-time
examples, and resolve disagreements. Keep development and final evaluation recordings
separate. Prior development use must remain disclosed; a manifest alone does not
certify an untouched final test. Keep candidate and image policies fixed for later
frozen-versus-trained comparisons. Pairwise sampling and Qwen-specific formatting are
deferred until the training objective and model inputs are selected.

## Validation performed

The complete suite passed **334 tests, with 343 subtests**, including 28 dataset tests.
Tests use generated images and synthetic review records; they are not human labels.
Checks cover split separation, pending/uncertain/AI exclusions, altered metadata and
image bytes, future frames, output overwrite protection, HTML escaping and cohort
accounting. Existing retrieval and teammate components passed the same suite.

The real pilot was opened in the browser. Its first candidate's three original
2560-by-1440 images loaded, and its JavaScript export payload retained all 230 rows
as pending/unreviewed. That download was intercepted for verification without saving
or entering human judgments. The real command-line exporter correctly refused the
pending training pack with exit code 2 and no output dataset. All three command-line
help checks (dataset, reviewer and evaluator) succeeded. No Qwen or live answer model
was called.
