# Temporal evidence investigation

## Try the current footage reviewer

Choose **Answer from recorded tracks** to query the saved Task 1 detections in a particular original frame. In recording 054604 at frame 126, “Which types of road users are detected in this frame?” returned one bus, seven cars and three large vehicles. Its citation E1 opens frame 126. In recording 054907 at frame 150, “How many vehicles are detected in this frame?” returned 11 vehicles and cited E2 from frame 150. These counts describe the detector output. [Open recording 054604 at frame 126](http://127.0.0.1:8775/footage?recording=video_054604&frame=126) · [Open recording 054907 at frame 150](http://127.0.0.1:8775/footage?recording=video_054907&frame=150)

Choose **Ask Gemini about the privacy-processed frames** for visual questions. At 054604 frame 126, “Is the sandy shoulder on the same side as the tall pointed tower?” returned yes, both on the right. At 054907 frame 150, “Which side of the picture has the group of blue overhead direction boards?” returned the right-hand side. Each answer cited a single privacy-processed view of the requested original. The two responses took 10.1 and 14.3 seconds on the prepared laptop. The latter included one retry after an HTTP 503. [Questions, answers, source frames and request receipts](../experiments/footage-selected-scene-review-20261124.json)

An independent check compared five present-scene layout questions on these two recordings. Previously, all five either retrieved earlier track images or found no image. All five now retrieve the full paused frame, including scenes where the detector offers no matching track for a landmark. Requests about a future bend and the architect’s intent returned no evidence. The archived 36 development questions retained their interval ranks and scores. The five-question check measures retrieval, with hosted answer content checked on two of the five questions. [Independent footage comparison](../experiments/footage-selected-scene-review-20261124.json)

## Earlier bus-entry investigation

The bus-entry failure involved both evidence selection and an ambiguous premise. The first TF-IDF run ranked opening bus 2 first and gave the later bus only short windows, missing its clearest final view. Bus 2 is already visible at frame 1. Yellow bus 46 is first annotated at frame 70, already inside the image and partly hidden by vegetation. It has later annotation gaps. Those labels do not establish a clean first boundary crossing or a physical road-entry maneuver.

A useful response identifies the later yellow school bus and explains that its exact entry cannot be established. The system must preserve that distinction even when the question presupposes an entry event.

## Development adapter

Select `tools.temporal_trial_adapters` explicitly to reproduce the earlier reference experiment. It supplies annotated observations and uses the same configured Gemini model for evidence selection, drafting and a separate grounding review.

1. Arjun's EMT reader loads original frames and labeled observations.
2. Candidate construction removes observations after the query before computing track histories, gaps, boxes, sizes or windows. It offers full histories, overlapping local windows, and initial/current scene snapshots.
3. Gemini selects up to three candidates from structured histories and masked initial/current images. It also extracts exact requested frame endpoints. A tracked interval uses those endpoints only when the track is observed at each one; an absent observation is never invented or substituted. Prominent visible objects receive track-ID labels. Window-specific boxes remain available alongside shared history. Textual selector box summaries are rounded to 0.1 pixel; source observations and exported evidence retain their exact coordinates.
4. The reviewer exports exact evidence images and original boxes. A null track ID represents an exact full-scene frame without a fake box.
5. The answerer receives cited interval facts, not the entire later history of that vehicle. Box size, net image direction and dominant displacement axis are computed from the observed boxes, so the language model need not derive them from coordinate notation. These are image-space measurements, not physical velocity or a complete trajectory description. It sees privacy-masked full views and targeted crops produced before whole-frame downscaling. Queries about earlier motion may use earlier observations even when that track ends before the query.
6. A separate model review checks every draft against the same facts and images, including unsupported event timing and mixed identities. It limits qualifications to uncertainties relevant to the question. Structured output separates supported observations from unverified details. Invalid response shapes or citations are rejected before caching. At most one output-validation repair can reuse the original evidence; the rejected response and repair provenance remain recorded. It can change selected candidate or citation IDs within the supplied evidence. Neither validation nor a model review proves factual correctness.

No question-specific answers, bus IDs or benchmark keys enter this adapter. Box widths, heights and displacements are general image-space measurements. They do not supply physical speed, driver intent or unseen cabin activity. Ordinal scores (`1/rank`) describe selected order, not probabilities.

Both selection images and answer images apply the existing masks to labeled people and the lower 40% of labeled vehicles. Crops use that masked source. Coverage remains partial because unlabeled details can remain visible. The selected public EMT inputs were authorized for the Google trial. Keep footage, processed images, caches and credentials outside Git.

## Reproduce the trial

Use `requirements-integration.txt` and the commands in [task5-integration.md](task5-integration.md). Set the model explicitly so a changed provider alias is visible.

```powershell
$env:TRACKRAG_TRIAL_MODEL='gemini-3.5-flash-lite'
python -m trackrag review --data-root data/emt-official --recording video_054604 --questions experiments/temporal-development-questions.jsonl --source annotated --answerer gemini --adapters tools.temporal_trial_adapters --out runs/temporal-a
python -m tools.review_report --run runs/temporal-a
python -m trackrag review --data-root data/emt-official --recording video_054907 --questions experiments/temporal-second-questions.jsonl --source annotated --answerer gemini --adapters tools.temporal_trial_adapters --out runs/temporal-b
python -m tools.review_report --run runs/temporal-b
```

The official-layout data root must contain both recordings' original frames and labels. The prepared-manifest layout from `tools.prepare_emt_clip` is also accepted. These local runs used the first 150 frames of each recording from pinned EMT revision `4418cec01b03c6f797cadb06c9405843a30988e2`. Reader outputs matched an independent parse for all 2,681 and 2,844 observations respectively. The split manifest assigns `video_054604` to training and `video_054907` to development. Both are development inputs for this investigation; neither run is held-out evaluation.

`GEMINI_API_KEY` is required. Optional settings are `TRACKRAG_TRIAL_CACHE` (default `runs/temporal-request-cache`), `TRACKRAG_TRIAL_TIMEOUT_SECONDS` (default 90) and `TRACKRAG_TRIAL_MAX_OUTPUT_TOKENS` (default 3072). A model that spends more output tokens on reasoning may need a larger budget. Receipts record requested and provider-reported models, request hashes, token usage, failures, repairs and cache hits. Only validated results are cached. Transient HTTP failures receive bounded retries. Daily quota failures remain errors, and changing models is explicit.

`--answerer none` disables drafting and answer review, while this semantic adapter still sends selection requests and masked scene images to Google. Use `tools.emt_trial_adapters` with `--answerer none` for a fully offline TF-IDF comparison.

## Question bank and review

The 24 questions in `experiments/temporal-development-questions.jsonl` and 12 in `experiments/temporal-second-questions.jsonl` cover temporal identity, exits, growth, colours, relative movement, sign text, current-frame absence, historical identity, future information and unknowable details. The reference rubrics are separate `*-rubric.jsonl` files and never enter inference. They were prepared through independent AI-assisted inspection of images and annotations before the new runs. They still require student review before use as an assessment benchmark.

There are 26 expected-answerable cases, 9 expected abstentions and one manually judged ambiguous bus-entry question. Bright white/silver colour distinctions can be uncertain after masking and resizing. Content and cited support must be checked alongside status. A response with the correct status can still name the wrong vehicle or cite an unrelated track.

An earlier compact-context Gemini 3.5 Flash-Lite pilot delivered all 36 result rows, including four protocol errors. Independent review graded the 35 ordinary cases as 23 clean passes, 3 qualified and 9 failures. The bus-entry question was reviewed separately. Failures included a missed grey SUV, a wrong green-truck ID, confusion between the opening cage truck and a later pickup, fabricated box dimensions and rejection of usable historical motorbike observations. This prompted the richer visual selector, explicit geometric facts and output recovery described above. These pilot grades are development observations, not a general accuracy estimate.

## Model comparison limits

The complete development cohort uses explicitly requested and provider-reported `gemini-3.5-flash-lite`. The larger-endpoint attempts requested `gemini-3.5-flash` and `gemini-3.6-flash`; completed receipts reported `gemini-3.6-flash`. Nine answers completed before the account's free daily limit of 20 requests for that model stopped the comparison. These counts refer to provider requests, not questions; selection, drafting and review each use a request before retries or repairs.

Two `gemini-3.8-flash` answer attempts did not deliver usable answers here. One returned malformed JSON and another timed out after increasing the output budget and request timeout. This is an availability/response-format observation for these calls, not a model-quality ranking. The uneven model runs do not support declaring a best model.

A targeted six-question `gemini-3.7-flash` diagnostic was also attempted. Its first evidence-selection request timed out at 120 seconds, so that comparison produced no completed answer cohort. No quality score is assigned to it.

The smaller model remains a development baseline. A correct output schema, a model critique and passing integration tests do not guarantee factual answers. Use the saved per-question review notes to see remaining model limitations, and repeat evaluation with the completed NLP pipeline, including the trained Qwen reranker and configured answer generator, rather than treating this trial as final deployment validation. The assessed training roles and completion evidence are defined in [assessment-alignment.md](assessment-alignment.md).

## Recorded final cohort

The fixed code in commit `5676635` was tested in `runs/temporal-final-a-05` and `runs/temporal-final-b-05`. The separate reviews are saved in `experiments/reviews/temporal-final-first.jsonl` and `temporal-final-second.jsonl`, with the recording run identified on every row.

| Independent development review | Cases |
| --- | ---: |
| Clean content-and-evidence pass | 29 |
| Qualified or incomplete answer | 3 |
| Failure | 3 |
| Ambiguous bus-entry diagnostic, judged manually | 1 |

The 35 ordinary cases include a false colour abstention (`temporal-07`), a wrong pickup identity and direction (`second-04`), and one HTTP 429 provider failure (`second-11`). Qualified cases retain an incomplete clearer-view comparison (`temporal-03`), inconsistent uncertainty/citations for a correct absence statement (`temporal-18`), and an unnecessary Arabic sign transcription error alongside the correct English name (`second-09`). These weaknesses are not fixed answers or hidden scoring exclusions.

Exact frame comparison 117–150 and the historical motorbike's dominant leftward movement now receive the correct bounded evidence and derived measurements. The original bus-entry response gives track 46 and explicitly withholds a proven edge crossing. Its first annotation at 70 must not be read as proof of first appearance.

The final first-pass cohort contains 35 usable model responses and one provider error. A later same-code availability replay of the second recording (`runs/temporal-final-b-06`) did not recover the failure and produced three quota errors. Provider receipts identify the account's 500-request daily free limit for Flash-Lite. The first-pass cohort is retained consistently rather than selecting favourable answers from different runs.

All 99 local automated tests passed, including 31 subtests. Windows and Linux CI also passed on the same code. These checks verify adapter boundaries, exact scene evidence, temporal cutoffs, image coordinates, scoring and recovery behavior. They do not certify language-model correctness. The independent grades are AI-assisted and require student verification.

This original trial used human-annotated observations. The later application evaluation used saved predictions from the team's fine-tuned detector and checked the footage pipeline on both recordings. The release report records each answer, abstention, citation and source check. [Integrated release evaluation](project-release.md)
