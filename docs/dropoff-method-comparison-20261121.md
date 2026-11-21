# Passenger grounding: bounded method comparison, 21 November 2026

The subsequently authorized [quick verification](dropoff-quick-verification-20261121.md) ran the previously prepared diagnostics, tested clearer marks and recovered the crowded-scene memory failure. The sections below preserve this earlier window's results and then-unrun status.

**Decision: do not adopt either tested local candidate selector.** Finding visible road patches is feasible, but the tested language models did not select destinations on the six-case public development cohort. Preserve the existing QA/RAG route and trained adapter; these experiments do not establish a working passenger feature.

## What actually ran

Three agents assisted with research, image-only candidate generation, independent scoring and source review. One local GPU process ran at a time. No new account, hosted GPU setup, paid model purchase or additional training was started.

The fixed public cohort was `1194, 1348, 1566, 2394, 2650, 2945` from the existing public development partition. These scene groups had no previously inspected free-generation results. Their partition had already been used for teacher-forced development loss, so this is development work, not an untouched final test. Reserved EMT final recordings were excluded.

| Local condition | Point inside original destination mask | Location proposals | Clarifications | Request errors | Mean request time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected Qwen3-VL-8B adapter, direct full interpretation | 0/6 | 2/6 | 4/6 | 0 | 55.29s |
| Selected adapter, candidate selection | 0/6 | 0/6 | 6/6 | 0 | 52.77s |
| Original pretrained 8B, candidate selection | 0/6 | 0/6 | 6/6 | 0 | 57.15s |

Every requested case remains in the denominator. A clarification receives no localization credit. **Public masks do not label the correctness of status or clarification decisions:** these results cannot establish that all six clarifications were wrong, or quantify semantic status accuracy.

Independent checks confirmed identical requests, marked-image bytes and prompts for pretrained versus adapter candidate selection, apart from local file paths. Their configuration differed only in adapter presence. Direct interpretation uses the existing constrained four-field decoder; selection uses strict validation of an unconstrained three-field JSON response. This is a whole-method comparison, not a pure decoding ablation.

## Candidate generation

The new CPU prototype uses pinned `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`, revision `21b3847fae21ddee674abd31129307b6a1235bd9`. Its 14.96 MB checkpoint hash is verified before loading with `weights_only=True` and no remote code.

It predicts road pixels from the image, excludes current-frame predicted road-user boxes, splits the predicted road into a fixed 6×6 spatial grid, and assigns up to 36 numbered interior patches. Neither the passenger instruction nor destination labels influence candidate generation. The grid enumerates candidates; it does not encode answers to particular questions.

Generation took **19.41 seconds on CPU for ten requests/eight images**, including model loading. After freezing generation, independent mask scoring found at least one candidate representative point inside the reference in **5/6 public cases**. That is a coverage ceiling, not achieved grounding accuracy. Both local selectors achieved **0/5** on the coverable cases. Case1194 had no correct representative point and remains a failure in the six-case denominator.

Road predictions have false positives and exclude some possible off-road destinations. A predicted road patch does not establish occupancy, physical clearance, stopping legality or safety. Candidate patches and direct coarse boxes have different sizes, so point agreement is the primary comparison.

## Hosted and real EMT checks

The planned Gemini3.8Flash trial stopped after two HTTP503 responses in approximately six seconds. This establishes an access/service failure for that attempt, not model quality. The previously working `gemini-3.5-flash-lite` fallback completed eight requests: direct and candidate interpretation of four instructions on two EMT development frames.

Only the established privacy-processed EMT images were sent to Google: observed people/riders masked, lower40% of observed vehicles masked, then resized. No public RegSeg images or unprocessed EMT originals were uploaded by these trials. The segmenter used originals locally; hosted/local pipelines therefore do not have identical image information.

An independent, method-anonymized AI visual review examined the saved outputs. It is provisional diagnostic evidence, not human ground truth:

- The fallback's direct bus and truck answers copied vehicle boxes rather than identifying ground beside them.
- Its candidate truck answer selected a plausible visible asphalt patch near the truck. This is one narrow positive example, not validated general performance.
- Its bare “here” clarification was useful. The request involving two parked black cars did not establish the required pair in the image; no reliable between-cars destination was demonstrated.
- The selected local adapter's historical direct EMT proposals remained unsupported vehicle/occupied-region placements.
- The local candidate route produced two clarifications and two input-budget errors on the four EMT requests. The second review phase disclosed its method identity and was not fully blinded.

The crowded-frame requests exceeded2048 input tokens. A retry configuration with3072 tokens was prepared, preserving every observation and the original prompt. **It was not run and is not a verified fix.** Original failures remain in the evidence. Voice-tagged requests used supplied transcripts; no microphone/ASR evaluation was performed here.

## Training and next steps

Both downloaded reviewer CSVs still contain56 unfilled review rows and no completed decisions. The published Task2 targets specify vehicle identities/landmarks, not reviewed ground points/boxes plus full interpretation statuses and clarification targets. Those files cannot support the planned local full-interface adaptation yet. Task1's latest handoff adds scene/provenance/evaluation support, not ground segmentation.

The earlier 8B LoRA training is real and remains preserved. Today's runs were inference comparisons, not another fine-tuning run. Public destination-mask supervision does not teach all passenger-interface decisions, and the current coordinate adapter was not trained to select numbered candidate IDs.

The next necessary work is:

1. Deliver reviewed local training/development cases with actual image files and hashes, whole-recording split separation, all four target fields, and reviewer provenance. A vehicle's center/box is not a stopping-ground label.
2. In a new bounded test window, verify the two crowded-frame requests with the larger input limit; retain original and retry outcomes separately.
3. If testing candidate selection further, use the prepared **geometry-only forced-choice diagnostic** to separate location selection from the full interface's status decision. It requires one ID on the same six public cases, including the uncovered case. It is **prepared but unrun**, emits no app status, and must not be deployed as a passenger decision policy.
4. Select a training objective only after those results: direct full interpretation needs genuine full targets; a candidate selector needs ID supervision derived from image-only candidates and training-only destination labels. Re-evaluate localization, proposal coverage, clarification usefulness and unseen recordings through the actual app before adoption.

No new checkpoint or final route was selected from these comparisons. Asking for clarification on every request is not sufficient evidence that the feature works. A larger/newer model has also not been demonstrated to solve this task automatically.

## Reproduction and evidence

The frozen protocols are in `experiments/dropoff/method-comparison-protocol-20261121.json` and `experiments/dropoff/pretrained-candidate-followup-protocol-20261121.json`. The durable summary is `experiments/dropoff/method-comparison-results-20261121.json`. Local raw results, snapshots, hashes, reviews and process receipts are under `runs/dropoff-methods-20261121/`.

The new tools are isolated development entry points: `tools.dropoff_ground_candidates`, `tools.dropoff_method_trial`, `tools.dropoff_score_methods` and the unrun `tools.dropoff_candidate_choice`. They do not replace the production grounding or cited-QA adapters. Scoring loads no model weights, makes no model calls, and verifies input/candidate snapshots, original image/mask hashes, raw-to-saved output agreement and whole-cohort denominators. It rejects wrong-instruction and malformed-output results.

Final software verification: **1217 passed,28 skipped,401 subtests passed**. These are software checks, separate from the model-quality results above. The local inference runs completed within their process deadlines; the temporary awake requests ended and no inference worker remained.

## Research informing the experiment

[LLM-RG](https://arxiv.org/abs/2509.25528) supports candidate/object grounding with spatial language in outdoor scenes. [Set-of-Mark](https://arxiv.org/abs/2310.11441) and [Graph-of-Mark](https://arxiv.org/abs/2603.06663v2) motivate marked visual regions and spatial relations. They do not validate our road patches or passenger feature. [NVIDIA's model card](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024) documents the pretrained segmentation source.

Google documents image/structured-output support for [Gemini3.8Flash](https://aistudio.google.com/docs/models/gemini-3.8-flash) and newer [Gemini4Carbon](https://aistudio.google.com/docs/models/gemini-4-carbon). Carbon has [no free tier](https://aistudio.google.com/docs/pricing#gemini-4-carbon), so it was not purchased for this bounded comparison. Model recency is not a passenger-grounding benchmark.
