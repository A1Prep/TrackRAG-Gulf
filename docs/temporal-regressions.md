# Grounding regression investigation

This work investigates the three regressions and three remaining failures in the 36-question annotated-track development cohort. The question files and separate rubrics are unchanged. No question IDs, reference answers, expected colours, sign text or vehicle IDs are used as inference rules.

## Causes and changes

| Case | Observed defect | General change |
| --- | --- | --- |
| `temporal-03` | Review removed a supported increase in visual clarity as if it were an unproven entry event. | Distinguish visible appearance comparisons from entry claims; preserve supported draft observations and keep answers concise. |
| `temporal-07` | Broad annotation class `Car` was treated as excluding an SUV, and visible upper-body colour was rejected because other regions were masked. | Keep annotation category separate from visual subtype; judge broad colour from visible panels. |
| `temporal-18` | The text stated visible absence while status/limitations withheld that same observation and supplied no citation. | Use the exact scene for visible presence/absence; distinguish this from hidden activity or absence inferred from missing labels. |
| `second-04` | Selection confused the pickup with an SUV; review then reversed that SUV's computed motion. | Add labeled identity galleries, resolve body shape before attaching motion, and preserve the relevant computed direction. |
| `second-09` | A correct English road name was accompanied by an unnecessary, incorrect Arabic transcription. | Answer at the requested language and granularity; avoid unsolicited OCR claims. |
| `second-11` | A real daily request limit stopped the answer, and retrieval selected the wrong vehicle. | Improve identity grounding; classify quota exhaustion explicitly as an availability error, with no fabricated evidence verdict. |

The selector now sees a separate tile for every track observed by the query cutoff, rather than relying only on labels on the 24 largest objects in two crowded scenes. Each tile uses the largest available observed box and labels its actual frame. Gallery times can precede the query, so the initial/current scenes remain the source for current visibility. The existing processor fully masks labeled people, cyclists and motorbikes and masks the lower 40 percent of other vehicle boxes before cropping. No future observations or pixels are used.

Sampling now uses the provider default and records that choice in the receipt. This follows [Google's current sampling guidance](https://aistudio.google.com/docs/troubleshooting). Requests with the old temperature override have different cache signatures. These pipeline and sampling changes do not isolate a single causal effect.

Transient provider limits retain bounded retries, respect a numeric `Retry-After` header and use exponential delays. Daily limits are not retried immediately or converted into successful abstentions. Receipts retain the HTTP code, classification, indicated delay and failed attempts. Daily availability still depends on the provider's quota; [Google documents the quota/reset policy](https://aistudio.google.com/docs/rate-limits).

## Verification status

- Full local software suite: 129 tests and 44 subtests passed. The added tests cover identity-gallery bounds/coverage/privacy, daily quota behavior, retry timing provider-default sampling and identity-token stability when unrelated tracks are inserted. They are software checks, not model-answer pass counts.
- Offline real-data input audit: all 36 questions passed at 11 distinct recording/query cutoffs. Every observed identity was represented, and all candidate/gallery/scene frames remained bounded.
- An intermediate same-model trial completed `temporal-03`, then hit the daily 500-request limit while selecting evidence for `temporal-07`. The multi-question run did not publish an incomplete result folder. Its successful first question was recovered from the exact validated request cache through the normal reviewer entry point with networking blocked; the recovery made zero network attempts.
- Independent inspection supports the recovered bus identity and clarity comparison. Its extra “fully visible” wording is unnecessary; the latest concise-answer instruction addresses that scope. This recovered output predates the final sampling/instruction changes and is not verification of the final code's full cohort.
- A separately labeled `gemini-3.8-flash` diagnostic with 16,384 output tokens and a 90-second request timeout failed in evidence selection after a transient 503 and a timeout. It produced no answer-quality result and does not replace the same-model regression comparison.
- The final five affected answers and all previous passes still need a fresh run and content review. Keep the change in draft until that is complete.

The current quota receipt indicates a reset around 2026-11-16 08:00 UTC, or noon Dubai time. This is a recorded provider indication, not an assurance of subsequent availability.

## Additional development controls

Twelve new questions were prepared through independent inspection of original and privacy-processed images, then checked by a second reviewer before the first complete 48-question rerun and before any control output. The question files are `experiments/temporal-control-questions.jsonl` and `experiments/temporal-control-second-questions.jsonl`; their separate `*-rubric.jsonl` files never enter inference. `experiments/temporal-control-protocol.json` records the pre-output file fingerprints and initial frozen inference commit. Later source revisions reuse these as development checks, not fresh held-out data.

Git preserves LF line endings for these frozen files so Windows checkout conversion does not invalidate their byte fingerprints.

The controls cover two distinct nearby objects, opposing horizontal movement, visible SUV/pickup body shape, current absence, hidden occupancy, English-only OCR, image growth and unknowable physical speed. Current-absence references were checked in the images rather than inferred from annotation endings. These are additional development controls on previously seen clips, not held-out assessment questions or proof of general model accuracy.

## Complete the same-model rerun

With the existing authorized dataset and configured API key, use fresh run directories:

```powershell
$env:TRACKRAG_TRIAL_MODEL='gemini-3.5-flash-lite'
$env:TRACKRAG_TRIAL_MAX_OUTPUT_TOKENS='3072'
$env:TRACKRAG_TRIAL_TIMEOUT_SECONDS='90'
python -m trackrag review --data-root data/emt-official --recording video_054604 --questions experiments/temporal-development-questions.jsonl --source annotated --answerer gemini --adapters tools.temporal_trial_adapters --out runs/temporal-regression-a-full-01
python -m trackrag review --data-root data/emt-official --recording video_054907 --questions experiments/temporal-second-questions.jsonl --source annotated --answerer gemini --adapters tools.temporal_trial_adapters --out runs/temporal-regression-b-full-01
python -m evaluation review-template --run runs/temporal-regression-a-full-01 --out runs/task5-regressions/full-first-review
python -m evaluation review-template --run runs/temporal-regression-b-full-01 --out runs/task5-regressions/full-second-review
```

Run the additional controls separately so the original paired-comparison denominator remains unchanged:

```sh
python -m trackrag review --data-root data/emt-official --recording video_054604 --questions experiments/temporal-control-questions.jsonl --source annotated --answerer gemini --adapters tools.temporal_trial_adapters --out runs/temporal-control-a-01
python -m trackrag review --data-root data/emt-official --recording video_054907 --questions experiments/temporal-control-second-questions.jsonl --source annotated --answerer gemini --adapters tools.temporal_trial_adapters --out runs/temporal-control-b-01
```

Review every answer against the unchanged separate rubric and actual cited images, including previous passes. Fill the blank review rows; do not copy old grades to changed answers. Paste the exported run bindings into the revised condition in `experiments/comparisons/emt-grounding-regressions.json`, then run:

```sh
python -m evaluation compare --manifest experiments/comparisons/emt-grounding-regressions.json --out runs/task5-regressions/full-comparison-01
```

That manifest references the planned full-run paths, which do not exist until the commands succeed. Its revised condition initially has no review binding and therefore cannot produce a content pass fraction. No retrieval answer key is supplied, so retrieval remains unscored. Report every gain, loss, abstention and availability failure on the same cohort. These tuned development questions still require separate held-out assessment later.

While live testing is unavailable, the team can review answer keys and supply real Task 1/3/4 modules and training artifacts. Task 5's final integration requires genuine detector predictions and the trained reranker; these annotated-track diagnostics do not measure those unfinished components.

## First complete rerun and follow-up

All 48 first-revision questions completed without a final provider error. Independent evidence inspection found five original problem cases sound, while pickup-motion retrieval still highlighted van 1 rather than pickup 6. Its correct leftward wording could be supported incidentally by the full scenes, illustrating why answer correctness alone cannot establish correct retrieval. Opening-car colour also became categorical despite glare/masking ambiguity. Raw outputs and judgments are retained.

The next revision uses candidate tokens containing the actual track and observed frame window. The first-revision selector returned `C6`, an ordinal candidate for van 1, while the gallery label `T6` denotes the pickup. Encoding identity in the token removes that numeric alias without supplying the desired answer. A separate general calibration instruction compares shaded/lit panels and preserves useful colour uncertainty. The same 48 questions will be rerun; completion is not claimed before their results are reviewed.
