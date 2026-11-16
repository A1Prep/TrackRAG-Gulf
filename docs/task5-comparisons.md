# Offline experiment comparisons

Task 5 can compare saved experiments before or after the teammates' real models arrive. The command reads completed reviewer runs and makes no model/API calls. It produces `summary.json` and a readable `report.md` in a new output directory.

```sh
python -m evaluation compare --manifest experiments/comparisons/emt-development.json --out runs/comparison-01
```

The supplied historical manifest needs the local saved EMT runs named in that file. Raw footage, processed images and run folders are deliberately excluded from Git. The same command can use a new manifest pointing to the teammates' saved baseline and fine-tuned runs.

## What the comparison checks

- Every variant is compared against the same question IDs, exact question text, recording IDs and query frames. Questions contain the four public fields used by the reviewer; answer keys remain separate.
- Missing question results remain in the expected cohort. Provider errors and unrequested answers are recorded separately. Invalid or unexpected result rows cannot silently disappear.
- Evidence ranks, track source, exact scene frames, observed boxes, citations and image files must agree. Saved frames cannot exceed a query cutoff. Provider receipts, when present, must refer to evidence-ID/frame pairs that were supplied to that question. A known final receipt cannot contradict a successful delivered answer's status or citations. Draft responses may differ, and a completed provider response can legitimately be rejected as an application error.
- A content review is bound to the exact `results.jsonl` bytes using its SHA256. Exported review packets also support a snapshot of every audited run file, including image bytes and receipts. A review cannot mark a missing, failed or unrequested answer as successful. Review coverage remains explicit; incomplete coverage produces no clean-pass fraction.
- Manual diagnostics are declared once in the manifest and apply to every variant. A review file cannot add exclusions. The tool cannot establish when a diagnostic was originally declared.
- The report fingerprints question files, outputs, review files, evidence images and provider inputs/receipts. A legacy result-only binding does not bind earlier image bytes. An optional review snapshot detects changes to the packet's audited source files; neither form proves a reviewer actually inspected them or that an image authentically depicts its claimed source frame.

These checks establish stored-artifact consistency. Answer correctness still comes from the supplied reviewers, whose labels are shown. An AI-assisted review is not automatically human-validated ground truth.

## Export a review packet

```sh
python -m evaluation review-template --run runs/my-run --out runs/review-packets/my-run
```

This exports four files without calling a model:

- `worksheet.md`: the actual questions, delivered answers, citations, and original/model-input image links.
- `review.jsonl`: one row per question with a null grade. A human fills the grade, note and reviewer label.
- `artifacts.json`: fingerprints of all audited source files.
- `binding.json`: a run entry that can be pasted into a comparison manifest, with the exact result hash and artifact snapshot.

Blank grades stay unreviewed. Exporting the packet does not certify a human review or create ground-truth labels. For partial reviews, leave unfinished grades null. The packet does not run the model, modify its saved answers, or infer grades from its confidence.

The packet uses absolute paths on the current machine. Keep the source run available; update locations if the files are moved. A changed result/image/receipt makes the old snapshot invalid. Export and review the changed material rather than simply replacing the hash.

## Manifest format

Use version `1`, a comparison `name`, declared `purpose` (`development` or `evaluation`), a list of shared `questions` paths and at least two `variants`. Paths are relative to the manifest file, or absolute. Variant IDs must be unique.

Each variant has an `id`, optional `label`, and a `runs` list. Each run entry has:

| Field | Meaning |
| --- | --- |
| `directory` | Completed run containing `run.json`, `results.jsonl`, `tracks.jsonl` and referenced images |
| `review.path` | Optional JSONL content review file |
| `review.results_sha256` | Required with a review; SHA256 of the exact results file reviewed |
| `review.artifacts` | Optional exported `artifacts.json`; when present, its fingerprints must cover every audited run file and match current bytes |
| `annotated_tracks` | Reference observations for scoring genuine predicted track IDs against annotated identities |

Use one run per recording and one track source per variant. Compare annotated and predicted conditions as different variants. An absent recording/result is counted as missing rather than omitted from the shared cohort.

Optional top-level fields are `notes`, `manual_question_ids`, `reference_variant`, `gold`, `temporal_iou` and `box_iou`. Unknown fields are rejected so a misspelled option cannot silently change the comparison. The reference defaults to the first variant. Paired review changes show clean-pass gains and losses among questions reviewed in both conditions.

Review rows contain `question_id`, `grade`, `note` and `reviewer`. Completed grades are `pass`, `qualified`, `fail` or `manual`; `manual` is allowed only for the manifest's diagnostic IDs. A partial review can leave grades null or omit unfinished rows, which remain unreviewed. Duplicate rows are rejected even when their grades are null. A diagnostic answer is always outside the ordinary clean-pass denominator, while its availability and review stay visible.

After changing model outputs, review them again and update the binding only after that review. Compute the result fingerprint using the public helper:

```python
from evaluation.run_audit import fingerprint
print(fingerprint("runs/my-run/results.jsonl"))
```

Do not relabel a condition as trained or held-out based only on its manifest name. Verify the training artifacts and recording/query splits described in [assessment-alignment.md](assessment-alignment.md). The comparer records available adapter/provider provenance but cannot prove which weights actually ran, that a model was fine-tuned, or that its training data excluded the evaluation set.

## Retrieval scoring and content grading

Without a separate `gold` answer key, retrieval metrics are explicitly unscored. Content-review passes are not retrieval accuracy. With gold, provide a list of JSONL key paths covering every ordinary question exactly once, excluding manual diagnostics. Each key uses the format in [task5-integration.md](task5-integration.md). The comparer reuses the existing scorer and its temporal/box thresholds.

Predicted-ID scoring needs annotated reference observations in each applicable run entry. The scorer uses box/time association, not numeric equality between independently assigned detector and annotation IDs. Missing question results remain misses. A provider failure can still have correct retrieval; its answer failure is retained separately.

## Historical development check

The supplied manifest compares the earlier compact-context Gemini prototype with the later temporal reference on the same 36 development questions and two recordings. It preserves all original reviews, including three final failures and one ambiguous manual question. The conditions change several pipeline components and use human-labeled tracks plus AI-assisted content reviews. They do not isolate a training effect, establish final model accuracy, or measure the unfinished CV/Qwen models.

For final assessment comparisons, prepare human-reviewed evaluation questions and the separate answer key, load the real pretrained/trained components through the normal reviewer command, save their runs, and compare those outputs. Hold the generator, prompt, data, query cutoffs and scoring thresholds fixed when testing the reranker's effect. Report unsuccessful or regressed cases alongside improvements.
