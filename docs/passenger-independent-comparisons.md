# Passenger grounding on reviewed Gulf images and public directions

The team compared Qwen3-VL-8B-Instruct before and after LoRA fine-tuning. The trained condition uses an adapter trained on 500 public examples with annotated destinations and continued on four Gulf training images. Each model received the same images and used the same generation and decoding settings. These runs measure the combined effect of public and local adaptation. Isolating the contribution of the four Gulf images would require running a third, public-only model on the same inputs. The released passenger interface also offers landmark selection, point correction and confirmation. [Registered model pair](../experiments/dropoff/passenger-final-model-conditions-20261122.json) · [Passenger application](passenger-release.md)

A rider can know the right street address and still prefer a different place to get out. Waymo already lets riders choose another available pullover on a map or request the next eligible spot after arrival. Our question is whether a passenger can express that preference through visible landmarks in speech or text, see the proposed area and confirm it. A confirmed selection records the point and area on the image. [Waymo rider guidance](https://support.google.com/waymo/answer/9696059?hl=en)

## Gulf decisions and stopping locations

Two reviewers approved AI-assisted annotations for the original typed requests on 44 EMT frames from five recordings. They approved a location on 33, a clarification on four and insufficient visual evidence on seven. Each of the 33 locations has a point and rectangle. Both models saw the original photographs, language and empty observation lists those reviewers inspected. Each case contributed one attempt per model, including malformed replies and runtime errors. The four approved questions use the same wording. [Input selection](../experiments/dropoff/passenger-gulf-input-choice-20261122.json) · [Frozen scoring rules](../experiments/dropoff/passenger-reviewed-original-eligibility-20261122.json)

| Measure | Pretrained 8B | Fine-tuned 8B |
|---|---:|---:|
| Decision matched both reviewers, all cases | 19/44 | 24/44 |
| Usable model response | 36/44 | 44/44 |
| Decision accuracy with the three reference statuses weighted equally | 26.5% | 31.6% |
| Returned a point on a location-labelled case | 18/33 | 23/33 |
| Point inside the reviewed rectangle, all marked locations | 0/33 | 1/33 |
| Average predicted vs reviewed rectangle overlap, missing = 0 | 0.007 | 0.033 |
| Rectangles with at least 50% overlap | 0/33 | 0/33 |
| Proposed a point when reviewers wanted clarification or insufficient evidence | 2/11 | 3/11 |
| Asked on a clarification-labelled case | 1/4 | 1/4 |
| Matched the accepted question literally | 0/4 | 0/4 |
| Invalid model reply / other runtime failure | 1 / 7 | 0 / 0 |

The fine-tuned model asked a question on each of the seven insufficient-evidence cases. On the four cases where the reviewers wanted a question, it asked once and proposed a location three times. This separates two different dialogue failures that the overall decision accuracy combines.

Across the same 44 decisions, both models were correct on 18, only the pretrained model on 1, only the trained model on 6, and neither on 19. Saying “propose a location” on every case achieves 33/44 decisions and produces a false proposal on all 11 other cases. Equal weighting of the three statuses gives that simple baseline 33.3%. Task 1’s fixed point halfway across and 80% down each image hit 0/33 rectangles. The coarse rectangles provide a location check. The 44 frames represent five recordings, and the reviews contain no passenger confirmation of a particular model output. [Full scores and confusion tables](../experiments/dropoff/passenger-reviewed-original-results-20261122.json) · [Fixed baselines](../experiments/dropoff/passenger-reviewed-original-baselines-20261122.json)

On the laptop’s 4 GB RTX 3050, the median native call took 120 seconds pretrained and 73 seconds fine-tuned. The batches ran sequentially. During the pretrained batch a lighter monitoring process freed approximately 505 MiB of Windows memory. Interpret latency and timeouts in that context. [Measured timing and conditions](../experiments/dropoff/passenger-reviewed-original-latency-20261122.json)

A follow-up diagnostic examined the 23 location-labelled cases where the fine-tuned model returned a point. Two were inside the reviewed horizontal range and seven were inside the vertical range. Only one met both. On these same 23 cases, the median predicted rectangle covered 1.64% of the frame, compared with 0.52% for the reviewed rectangles. The 23 predicted points were all distinct. The miss pattern is visible even among correctly formatted outputs. [Post-hoc geometry diagnostic](../experiments/dropoff/passenger-reviewed-original-geometry-20261122.json)

Two reviewers accepted each candidate ground area. The retained files record agreement on the candidate and contain no individual drawing traces. The 33 rectangles all have different normalized positions and have a median area of 0.275% of the frame. None equals an annotated road-user rectangle in the original EMT data. A separate calculation converted every raw model proposal using the original photograph’s dimensions and recovered every saved coordinate exactly. Scoring in the model’s normalized coordinates also gave zero and one hit. A passenger might accept a nearby spot outside an annotated rectangle. [Geometry and annotation provenance](../experiments/dropoff/passenger-reviewed-original-geometry-audit-20261122.json) · [Original review instructions](https://github.com/A1Prep/TrackRAG-Gulf/blob/2d96dcb6afdffb46c5d94d299768fe6a92acc094/Task2_56_Case_Human_Review_Corrected/REVIEW_INSTRUCTIONS.md#L16-L29)

One of the five recordings accounts for the single fine-tuned point hit. With recordings weighted equally, decision agreement was 42.5% pretrained and 54.2% fine-tuned. Task 1’s fixed point hit zero on every recording. An independent implementation matched all primary counts, all confusion cells and the five recording totals. [Per-recording comparison](../experiments/dropoff/passenger-reviewed-original-recordings-20261122.json) · [Independent verification](../experiments/dropoff/passenger-reviewed-original-agreement-20261122.json)

The application has a separate decision rule. In an earlier test on twelve other public scenes, it offered a point on four, and three fell inside the annotated region. That is three hits across twelve requests and three hits among four proposals. The passenger can inspect, correct and confirm a location in the app. [Previously measured application coverage and accuracy](../experiments/dropoff/passenger-spatial-confidence-results-20261122.json)

## Does a second description improve the point?

The twelve public Talk2Car photographs come from twelve different source scenes excluded from earlier project fitting and evaluation. Each model received the published concise instruction. After a usable first answer it received its own exact previous response and the longer published instruction on the same frame. The script supplied the restatement after every usable first response, including a proposed location. The source commands include yielding, turning and following as well as stopping. All four output sets were sealed before scoring against the human-drawn navigable regions. Figure 5 of Rufus et al. already illustrates an instruction to park between black cars. The Gulf comparison studies local adaptation, and the public comparison tests what a second description changes. [Source paper, Figure 5](https://arxiv.org/pdf/2112.13031#page=7) · [Dataset](https://github.com/kanji1011/Talk2car-Refseg) · [Frozen public cohort](../experiments/dropoff/passenger-public-dialogue-cohort-20261122.json)

| Model and instruction | Point inside drawn region | Mean rectangle overlap | Additional model calls |
|---|---:|---:|---:|
| Pretrained, concise | 0/12 | 0.011 | 0 |
| Pretrained, after restatement | 0/12 | 0.011 | 12 |
| Fine-tuned, concise | 1/12 | 0.023 | 0 |
| Fine-tuned, after restatement | 2/12 | 0.030 | 12 |

The point must land on an actual labeled mask pixel. Rectangle overlap is intersection over union with the smallest rectangle enclosing the mask. Missing or failed proposals score zero in both main measures.

The pretrained model increased its proposals from seven to eleven after the restatement and missed the marked region on every scene. The fine-tuned model increased its proposals from seven to ten, kept its initial hit and gained one. Each model’s two conditions share the same genuine first response. The models returned responses to all 48 distinct registered requests. [Four-condition results, paired comparisons and uncertainty intervals](../experiments/dropoff/passenger-public-dialogue-results-20261122.json)

The fine-tuned model already placed its point inside the human-drawn area for the passenger asking to get out near a person at the curb. The only additional hit came on a lane-change instruction. The model initially asked for clarification. With the longer description of a car slowing for pedestrians, it proposed a point inside the region. On these twelve scenes the longer wording often elicited a proposal, while precise placement remained uncommon. The two public dataset distributions contained byte-identical reference masks for all twelve cases. [Scene-level outcomes](../experiments/dropoff/passenger-public-dialogue-results-20261122.json)

An external process interruption stopped the final batch after nine saved answers. Those nine records were preserved byte for byte. The remaining three exact inputs were completed in their original order before the reference masks were opened. One of the three inputs may have started earlier without returning an answer, so the number of physical attempts is uncertain by one. Assigning zero to that interrupted input also gives two fine-tuned restatement hits and average rectangle overlap of 0.030. The downloadable archive contains the original records, recovery records, both scores and independent verification. [Reproduction and interruption check](../experiments/dropoff/passenger-independent-reproduction-20261122.json)

## Speech and the participant study

Sixteen real direction-givers in the HCRC Map Task spoke one instruction each. Whisper retained 20 of 21 spatial words and 18 of 23 named landmark mentions. There were 38 word edits in 269 reference words (14.1%). Every original transcript was editable before sending. These recordings test how the system hears landmarks and spatial words. [Speech results and reproduction](passenger-whisper.md#spoken-instructions-that-describe-a-place)

Task 3 also logged 31 spoken footage questions under three speaker codes, covering ten scripts per speaker and one extra take. All 31 final transcripts match the prompt. Measuring recognition on these particular recordings requires the original audio and verified spoken references. The drop-off study uses a separate set of ten passenger instructions. [Team recording evidence](passenger-speech-study.md)

For the separate Gulf interactive study, use the saved detector inputs that match Task 1 on 44/44 frames and contain 494 observations. Archive genuine participant-drawn regions, replies and explicit confirmations. Record that the same 44 source images were already evaluated on their original, image-only inputs. Measuring the proportion of riders who would confirm a proposed stop requires that separate study. [Evaluator handoff](passenger-evaluator-handoff.md) · [Detector-context reviewer packet](passenger-context-review.md) · [Participant form](passenger-participant-study.md) · [Voice recording form](passenger-speech-study.md)

## Using the team’s current application work

For a passenger demonstration, open the release application. It includes the road-area estimate, the two-vehicle selector, point correction and confirmation. The footage tab retrieves saved-prediction evidence from both attached recordings. [Launch instructions and example interaction](passenger-release.md)

In a fresh launch after integrating Task 4, the two-car request opened 23 selectable vehicle outlines. The operator cancelled that request. The footage tab retrieved three evidence intervals on each of two recordings, and both evidence images opened. [Application checks](../experiments/dropoff/passenger-evaluator-unified-app-20261122.json)

Task 4’s unified app also called the locally fine-tuned 8B directly on two development recordings. One input came from genuine public in-car audio transcribed by Vosk and edited by the test operator. The other was typed. Both calls returned a proposed area, verified the intended adapter and image, and took about 107 seconds each on the 4 GB laptop. The public speaker had recorded their words on another road, and neither proposal has a passenger-confirmed target. A separate Edge browser check passed a real audio upload through Whisper Small and retained the original and edited text. [Two-recording integration evidence](../experiments/dropoff/passenger-evaluator-unified-app-20261122.json) · [Run the exact Task 4 revision](https://github.com/A1Prep/TrackRAG-Gulf/blob/09e73559817f072b40ca0b500fa2c8cae04de5a1/docs/task4-application.md#check-real-module-handoffs-before-a-live-trial) · [GPU compatibility fix](https://github.com/A1Prep/TrackRAG-Gulf/pull/39)

The Task 3 branch now preserves the first transcript and the original audio format on new footage-question recordings. Its starter QA command uses the same evidence preparation and citation checks as the released footage application. The two fixes passed the team’s hosted Windows and Linux test suites. [Audio capture fix](https://github.com/A1Prep/TrackRAG-Gulf/pull/40) · [Footage and citation fix](https://github.com/A1Prep/TrackRAG-Gulf/pull/41)

## Reproduce the comparisons

The model and scoring files were frozen before the team merged updated application contracts. For an exact replay, start a separate checkout at the verified study revision shown below. This leaves the current application available in the regular checkout. Both aggregate scores were independently reproduced from this revision on the same Windows laptop. [Reproduction record](../experiments/dropoff/passenger-independent-reproduction-20261122.json)

```powershell
git clone --filter=blob:none --no-checkout https://github.com/A1Prep/TrackRAG-Gulf.git TrackRAG-study
Set-Location TrackRAG-study
git switch --detach ae68cf7d3eeb19e7fc5528c49729f393b32957fc
```

In that checkout, restore the three evaluation archives and original EMT JPEGs by following the evaluator handoff. For a private Gulf re-score, place the coordinator’s `passenger-reviewed-gulf-private-20261122.zip` in `runs/evaluator-downloads`. The following commands extract the original responses and compare them against the exact inputs the reviewers approved. The individual reviewer records and model responses stay in private storage. [Input restore instructions](passenger-evaluator-handoff.md#restore-the-published-inputs-on-another-checkout)

```powershell
$privateRuns = "runs/private-gulf-comparison"
Expand-Archive -LiteralPath runs/evaluator-downloads/passenger-reviewed-gulf-private-20261122.zip -DestinationPath $privateRuns
python -m tools.passenger_reviewed_baseline score --requests runs/evaluator-inputs/inputs/requests-as-human-reviewed.jsonl --eligibility experiments/dropoff/passenger-reviewed-original-eligibility-20261122.json --execution "$privateRuns/execution.json" --pretrained "$privateRuns/pretrained" --trained "$privateRuns/trained" --out runs/replayed-gulf-score.json
```

To repeat the native Gulf inference in a fresh private checkout, install the streamed 8B base model and the three evaluation archives. Restore the archives to the exact input path used by the registered runner. Preflight verifies the two configurations and all 44 original image bindings. The run uses one GPU for both conditions and preserves each case as it finishes. [Local GPU setup](dropoff-local-training.md)

```powershell
python -m tools.dropoff_restore_evaluator --archives runs/evaluator-downloads --out runs/task5-evaluator-delivery-20261122/installed
$gpu = ".venv-grounding-local/Scripts/python.exe"
& $gpu -m tools.dropoff_prepare_streamed_model --offline
& $gpu -m tools.passenger_reviewed_runs preflight
& $gpu -m tools.passenger_reviewed_runs run
& $gpu -m tools.passenger_reviewed_runs verify
```

The separate public evidence archive contains the registered cohort, both published wordings and the four native run directories. After restoring the two registered model configurations, the scorer can verify the saved calls on CPU. It downloads the original annotation archive member only after validating every run. [Public-scene evidence ZIP](https://github.com/A1Prep/TrackRAG-Gulf/releases/download/passenger-feature-20261122/passenger-public-dialogue-evidence-20261122.zip)

```powershell
gh release download passenger-feature-20261122 --repo A1Prep/TrackRAG-Gulf --pattern passenger-public-dialogue-evidence-20261122.zip --dir runs/evaluator-downloads --skip-existing
Expand-Archive -LiteralPath runs/evaluator-downloads/passenger-public-dialogue-evidence-20261122.zip -DestinationPath runs/public-language-evidence
python -m tools.passenger_public_dialogue_score --work runs/public-language-evidence --fetch-masks --out runs/replayed-public-score.json
```
