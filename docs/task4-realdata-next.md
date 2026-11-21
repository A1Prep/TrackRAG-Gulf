# Task 4: focused real-data pilot and remaining manual work

The local real EMT images and annotations are already available. The next stage
uses a [Task 4 development question cohort](../experiments/task4-text-realdata-questions.jsonl)
and an explicit [10-frame candidate policy](../experiments/task4-text-realdata-retrieval.json).
The questions were drafted for this experiment and remain subject to human
review; they are not a certified Task 3 question/answer set. No Task 3 module or
existing question bank is changed.

## Prepared worksheet

The worksheet is `runs/task4-realdata-review-01/review/review.html`. It contains
six questions, **313 pending candidate judgments** and **59 distinct original
images**, copied without pixel changes and bound by hashes.

| Question | Recording | Split | Query frame | Candidates |
| --- | --- | --- | ---: | ---: |
| task4-real-train-01: bus in frame 1 | video_054604 | train | 1 | 15 |
| task4-real-train-02: motorbike image-left motion | video_054604 | train | 13 | 43 |
| task4-real-train-03: car observed at frame 30 | video_054604 | train | 30 | 82 |
| task4-real-dev-01: medium vehicle in frame 1 | video_054907 | dev | 1 | 20 |
| task4-real-dev-02: large vehicle image-right motion | video_054907 | dev | 20 | 65 |
| task4-real-dev-03: car observed at frame 30 | video_054907 | dev | 30 | 88 |

The first two frame-1 cases are quick workflow starters. Later cases contain
separate windows for the same vehicle, so correct-vehicle/wrong-time judgments
are possible. Class categories come from EMT: for example, `Medium_vehicle`
is an annotated category rather than a claim about a physical size threshold.
Directions describe image-space motion, not a lane change or physical maneuver.

Queries focus on class, observed frame membership and movement that the current
text representation can describe. They do not ask the model to infer colors,
sign text, overpasses, hidden actions or physical speed from absent pixels.
All six queries are development work on previously used recordings. No final
held-out test evaluation is claimed.

## Regenerate the preparation

[retrieval_realdata.py](../trackrag/retrieval_realdata.py) uses the actual Task 2
reader and shared candidate/review interfaces. It assigns no relevance labels
and does not call Qwen, an answer model or a trainer.

```powershell
.\.venv\Scripts\python.exe -B -m trackrag.retrieval_realdata --data-root data/emt-task4-baseline --questions experiments/task4-text-realdata-questions.jsonl --split-manifest splits/task2_manifest.json --retrieval-config experiments/task4-text-realdata-retrieval.json --source annotated --out runs/task4-realdata-review-new
```

Choose a fresh output directory. The output contains full candidate packets,
the question/split/configuration snapshots, per-recording question files and a
packaged `review` directory. `preparation.json` gives the exact question-file
mapping, counts, hashes and `prepared_unreviewed` status. The review pack is
verified again against its saved metadata and image bytes before completion.
Only train/dev recordings are accepted. Saved prediction rows can be supplied
with `--source predicted --predicted-tracks <JSONL>`; that mode does not run a
detector or equate predicted IDs with annotated IDs.

The [recorded preparation summary](../experiments/task4-realdata-preparation-summary.json)
reports this actual two-recording run. The 10-frame window/stride is an explicit
pilot choice, not a development-tuned setting. It differs from the old 30-frame
baseline pool; keep this new policy fixed for subsequent model comparisons.

## What the human reviewer does

1. Open the worksheet, enter your name and inspect a question and its candidate
   frames. Red boxes identify the selected vehicle. Check both identity and time.
2. Mark a candidate relevant, irrelevant or uncertain, with a reason and a note.
   For frame-30 questions, an earlier interval showing the same car may still
   be unsuitable evidence. Do not infer entry/absence from missing observations.
3. Confirm all displayed frames were inspected only after viewing them. If the
   samples do not establish the claim, use uncertain rather than guessing.
4. Use the named-human-reviewer assertion and **Download all labels**. Preserve
   pending rows. Save the result as `labels-reviewed.jsonl` and provide its local
   path when continuing this task. Use **Resume from labels JSONL** to continue.

You can begin with a subset, including relevant and irrelevant training examples
and some reviewed development examples. A small partial review is sufficient for
a real-data machinery check; it is not sufficient evidence of training quality.
Source annotations are not automatic relevance judgments. Model/rule guesses
must not be recorded as human judgments.

The full review pool retains scenes, which can legitimately support a question
when inspected as images. For this text-only training pilot, use `--tracked-only`
below: scene documents contain no observed vehicle features, so scene examples
are explicitly excluded from training rather than assigned invented labels.

## After saving reviewed labels

Replace the label-file placeholder with the downloaded file's actual path:

```powershell
.\.venv\Scripts\python.exe -B -m trackrag.retrieval_training prepare-reviewed --review runs/task4-realdata-review-01/review --labels <REVIEWED_LABELS.jsonl> --tracked-only --out runs/task4-reviewed-realdata.json
.\.venv\Scripts\python.exe -B -m trackrag.retrieval_training train --dataset runs/task4-reviewed-realdata.json --qwen-config runs/task4-ranking-comparison-setup/qwen-cpu.json --out runs/task4-reviewed-realdata-checkpoint
```

The referenced CPU configuration is local to this workspace. To reproduce it
elsewhere, create the configuration documented in the [training guide](task4-qwen-training.md).
The default two steps are a short real-data pipeline check, not substantive
fine-tuning. Choose a deliberate training configuration and expand the dataset
before reporting a trained model's general usefulness. Pending/uncertain/AI rows
are excluded; training requires both label classes and separate dev examples.

## Frozen and trained ranking comparison

The [comparison command](task4-ranking-comparison.md) can now select a saved
adapter with `--checkpoint`. Frozen mode omits that option. For example, using
the per-recording development question file and the same explicit pool policy:

```powershell
.\.venv\Scripts\python.exe -B -m trackrag.retrieval_comparison --data-root data/emt-task4-baseline --recording video_054907 --questions runs/task4-realdata-review-01/questions-0002.jsonl --source annotated --retrieval-config experiments/task4-text-realdata-retrieval.json --qwen-config runs/task4-ranking-comparison-setup/qwen-cpu.json --out runs/task4-realdata-frozen-dev
.\.venv\Scripts\python.exe -B -m trackrag.retrieval_comparison --data-root data/emt-task4-baseline --recording video_054907 --questions runs/task4-realdata-review-01/questions-0002.jsonl --source annotated --retrieval-config experiments/task4-text-realdata-retrieval.json --qwen-config runs/task4-ranking-comparison-setup/qwen-cpu.json --checkpoint runs/task4-reviewed-realdata-checkpoint --out runs/task4-realdata-trained-dev
```

These commands score every candidate and save complete receipts. Do not use
`--allow-synthetic` for the reviewed training run. That option exists only for
explicit testing of synthetic checkpoints. Verify matching batch metadata hashes
and candidate IDs before comparing rankings. Scores from different rankers are
not calibrated against one another.

Ranking agreement is not accuracy. For relevance metrics, join independently
reviewed judgments to the exact batch/candidate identities. Unreviewed top-ranked
candidates have unknown correctness, not automatic failure or success. Final
held-out evaluation and downstream answer/evidence scoring remain separate work
through the existing Task 5 interfaces.

## Status at preparation

Software is available for real-data preparation, reviewed-example loading, LoRA
updates, checkpoint reload and frozen/trained ranking. **Real relevance review,
real-data training and independently measured improvement remain pending.**
Pushing this implementation does not close those experimental milestones.

The final local regression run passed **394 tests and 401 subtests**. CLI help
checks passed for preparation, reviewed training, comparison and the existing
reviewer/evaluator. The actual six-question pack was regenerated through the same
entry point and its complete pack hash matched exactly. A packaged original
image loaded at 2560 by 1440 in the browser. No browser relevance labels were
assigned by the agent. Tests of saved predictions use generated fixtures, not
genuine detector predictions.
