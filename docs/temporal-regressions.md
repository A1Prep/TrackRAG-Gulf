# Temporal grounding regression results

The six originally flagged questions now have correct answers and relevant identities in the latest complete development run. The wider regression check still fails the release gate, so PR11 remains a draft. The main branch has not been replaced with this experimental revision.

## What changed

- A query-bounded identity gallery shows every observed track at its largest available view, with the actual frame and track label. The existing privacy masks are applied before cropping.
- Selection tokens now contain the actual identity and observed window, such as `T42-F1-F5`, rather than unrelated ordinal labels. In one earlier miss, `C6` selected van 1 while gallery label `T6` denoted the intended pickup. This removes an avoidable ambiguity without claiming to establish the model's reasoning.
- General instructions distinguish broad annotation classes from visible subtypes, current visible absence from hidden states, image observations from world motion, and requested OCR from extra transcription. They preserve supported draft claims, allow sensible colour uncertainty and give explicit answer formats priority.
- Daily quota exhaustion remains an availability error. Transient limits receive bounded retries, numeric Retry-After handling and accurate receipts. Provider-default sampling is recorded following [Google's current guidance](https://aistudio.google.com/docs/troubleshooting).

No reference answers, expected colours, sign names or question-ID branches were added to inference. Public question files contain only their four interface fields. References remain separate scoring inputs.

## Complete small-model runs

All conditions use real EMT images, human-labeled tracks, the same question records and the requested/reported Gemini 3.5 Flash-Lite model. The references were independently reviewed against original and processed images. The later rounds randomized A/B labels per question and hid model/revision identity. Core question/reference records are unchanged; Windows line endings can differ. These are previously seen development recordings, not held-out accuracy or trained-detector/reranker results.

| Complete condition | Clean | Qualified | Failed | Manual diagnostic |
| --- | ---: | ---: | ---: | ---: |
| Earlier reference |29|3|3|1|
| First gallery revision |34|1|0|1|
| Latest explicit-identity revision |31|2|2|1|

The first revision's high QA count concealed a wrong pickup target: incidental full-scene pixels supported correct wording while the highlighted vehicle was van 1. It was not accepted as a complete fix. The latest revision corrects that identity, but loses four previous clean passes: two qualifications and two factual failures. All original and revised outputs are retained; favourable answers were not combined into a synthetic cohort.

Both paired rounds assign the same reference grade to every question. Grades remain AI-assisted with human validation pending. Multiple pipeline and sampling changes prevent attributing differences to one isolated cause. There is no separate formal retrieval key here, so these counts must not be called ranking accuracy.

## Original six cases

| Question | Latest finding |
| --- | --- |
| `temporal-03` | Correct bus 46 and cited70-150 history showing increased clarity. |
| `temporal-07` | Correct grey SUV 8 at exact frame 30, supported by visible panels. |
| `temporal-18` | Correct current absence statement with scene 30 citation. |
| `second-04` | Correct leftward image motion and pickup 6 evidence over frames 1-100. |
| `second-09` | Correct Al Fay Rd, with no incorrect added Arabic transcription. |
| `second-11` | Correct closed truck 14 at 150; contents appropriately remain unknowable. |

## Remaining regressions and grounding defects

| Question | Remaining issue |
| --- | --- |
| `temporal-06` | Identifies bus 2 but unnecessarily withholds the supported image-edge movement as if it required a proven road manoeuvre. Qualified. |
| `temporal-10` | Categorical white overstates confidence against the silver/light-grey reference under glare and masking. Qualified; human colour-reference validation is needed. |
| `temporal-11` | Falsely abstains on which car appears largest despite usable scene/car 7 evidence. Failed. |
| `second-08` | Invents a remaining portion of the opening cage truck at 100, where the original truck is absent. Failed. |
| `second-02` | Correct direction words, but selected E1 highlights pickup 6 rather than cage truck 17. Incidental scene pixels support the words. This is a separate retrieval defect, not evidence of correct target selection. |

The predeclared manual bus-entry diagnostic remains outside the 35 ordinary-question denominator. Exact first visibility/edge entry is unproven, and the latest answer also adds an incorrect net-motion claim. Its exclusion does not turn that answer into a pass; the raw output and review remain visible.

## Twelve additional controls

The new question/reference files are `experiments/temporal-control-questions.jsonl`, `temporal-control-rubric.jsonl`, `temporal-control-second-questions.jsonl` and `temporal-control-second-rubric.jsonl`. The protocol records their hashes before their first model outputs and the initial inference commit. Two reviewers checked the reference imagery, including actual visible absence rather than annotation endings. Git preserves LF for these files so their byte fingerprints survive Windows/Linux checkout.

| Control condition | Clean | Qualified | Failed |
| --- | ---: | ---: | ---: |
| First gallery revision |9|2|1|
| Latest revision |12|0|0|

They cover opposing vehicle movements, body types, relative position, visible absence, hidden occupancy, unknown physical speed, English-only OCR and image growth. Later code revisions used them as development checks. They are not fresh held-out assessment evidence. Human review packets retain blank grades separately from the AI-assisted judgments.

## Stronger-reviewer check and confirmed quota limit

A separate diagnostic retained the latest selected evidence and used Lite drafting with Gemini 3.6 Flash final review. It corrected largest-car identity and truck absence on the four targeted cases. Image-exit qualification and colour confidence remained; the largest-car answer also added white as a descriptor. This does not establish a causal model ranking or a clean full-cohort result.

The full 48-question stronger-review attempt retained 19 answered rows and 29 unavailable error rows. The terminal receipt first reported the 5-request minute limit, then the 20-request daily limit. After that terminal failure, 28 subsequent review requests were suppressed locally; exact cached requests could still return. No unavailable result became an abstention or a favourable older answer. Provider requests and completed questions are different counts. See `experiments/strong-review-availability.json` and [Google's quota policy](https://aistudio.google.com/docs/rate-limits).

Full-cohort quality of that stronger configuration is therefore unverified. The four diagnostic outputs are separate in `experiments/strong-review-diagnostic.jsonl`; they were not substituted into either complete small-model cohort. Their exploratory orchestration and raw run folders remain local.

## Software and artifact checks

The local suite passed 129 tests and 44 subtests; Windows and Linux CI passed. Checks cover future-frame isolation, complete identity-gallery coverage, masks before crops, identity-token stability under unrelated-track insertion, valid duplicate rejection, exact requested endpoints, citation validation, retry/quota behavior and saved-run comparison integrity. Offline input checks covered all 48 question inputs. Independent audit validated 10 run folders and reproduced the comparison JSON/Markdown from the saved inputs.

These checks verify program behavior and stored consistency. They do not certify model-answer correctness. No detector or Qwen fine-tuning was performed by these trials.

## Reproduce the latest complete condition

Set the existing authorized key and pin the requested model. Use new output directories; these names are already occupied on the development machine.

```powershell
$env:TRACKRAG_TRIAL_MODEL='gemini-3.5-flash-lite'
python -m trackrag review --data-root data/emt-official --recording video_054604 --questions experiments/temporal-development-questions.jsonl --source annotated --answerer gemini --adapters tools.temporal_trial_adapters --out runs/my-grounding-first
python -m trackrag review --data-root data/emt-official --recording video_054907 --questions experiments/temporal-second-questions.jsonl --source annotated --answerer gemini --adapters tools.temporal_trial_adapters --out runs/my-grounding-second
```

Run the two control banks separately through the same command. Export ungraded human packets with `python -m evaluation review-template --run <RUN> --out <NEW_PACKET>`. Inspect every new answer against the unchanged reference and actual cited images; do not copy older grades to changed outputs.

The supplied retained-cohort comparison needs its local run files, which are excluded from Git:

```sh
python -m evaluation compare --manifest experiments/comparisons/emt-grounding-regressions.json --out runs/my-grounding-comparison
```

The experiment source commits and local run-source snapshots distinguish the retained conditions. The latest complete generation used `ec747a3`; the first complete gallery revision used the same inference source as `cdf0bb6` before identity-token/calibration changes.

## Next integration gate

1. Obtain enough availability for a complete evaluation of the chosen stronger answer backend. Keep the same cases, preserve all failures and check highlighted identities separately from answer wording before promoting a configuration.
2. Have Tasks 1, 3 and 4 implementations supplied as PRs, including the trained detector and trained Qwen artifacts/configuration. The current repository readiness check still lacks those normal adapters; private teammate work is not assessed here.
3. Human-review the answer references, especially colour and edge-entry ambiguity, and reserve a separate recording excluded from training and this debugging. Then run the real trained pipeline with both annotated and genuine predicted tracks.

Task 5's framework and comparison preparation are built; final model integration, held-out evaluation and the requested no-regression release gate remain open. The assessment training roles are defined in [assessment-alignment.md](assessment-alignment.md).
