# Quick passenger-grounding verification, 21 November 2026

**Decision: the passenger feature is not cleared for completion.** We verified and recovered the crowded-scene execution failure, but correct destination selection remains unreliable in the tested pipeline. No training or checkpoint changes occurred in this diagnostic window.

This follows the [earlier method comparison](dropoff-method-comparison-20261121.md). The new user request authorized another bounded verification window. All GPU processes completed before the new17:37:20UTC cutoff, with one process at a time and no downloads or cloud setup.

## What the quick tests established

The same six public development cases and image-only road candidates were reused. At least one candidate representative point lies inside the original reference mask in five cases. The sixth remains in every denominator.

| Geometry-only condition | Valid choices | Reference-mask hits | Mean request time |
| --- | ---: | ---: | ---: |
| Selected coordinate-trained 8B adapter, original marks | 6/6 | 0/6 | 14.05s |
| Pretrained 8B, original marks | 6/6 | 1/6 | 13.50s |
| Selected adapter, larger final-size marks | 6/6 | 1/6 | 12.14s |

These tests require one candidate ID and produce only two geometry fields. They do not produce an application status, clarification or confirmation event. All18 answers were valid IDs; the reference-mask misses were not JSON or time-limit failures. The candidate ceiling remains5/6, with0/5,1/5 and1/5 among coverable cases respectively.

The trained/pretrained comparison used identical images, requests, prompts and candidates; adapter presence was the only saved configuration difference. Removing the clarification decision did not recover reliable localization. This small, reused development cohort does not establish that fine-tuning damaged the base model, that8B is inherently too small, or that a particular untested training method would fix it.

## Actual model-input inspection

The pinned processor turns both1600×900 public images and2560×1440 EMT images into672×384 inputs, with252 merged vision tokens. In the original overlays, text glyphs shrink to approximately5.1–5.5pixels high and one-pixel patch borders become thinner than one pixel. An independent processor-only audit reconstructed the actual model inputs and verified their saved image hashes.

No crop, RGB, axis-order or normalized-coordinate mismatch was identified. The0–1000 coordinate convention matches the official Qwen3-VL grounding convention.

A separate reference-only AI inspection found no obvious image/mask indexing or spatial-alignment error in the six cases. It also found an important task-definition limitation: the public masks are coarse destination regions, and some overlap lower vehicle silhouettes. They are not strict empty-ground labels. Several commands omit pointing/counting/prior context; one rainy scene does not clearly establish the named red truck. Mask agreement therefore cannot certify an empty stopping area or prove that every clarification was wrong. This is provisional visual inspection, not human re-annotation; no reference was changed.

The new optional `--legible-marks` variant first resizes to the configured model budget, then draws20-pixel labels, two-pixel borders and visible point crosses. Candidate identities and original-pixel geometry remain fixed; no instruction or gold mask enters rendering. This changes both marking and PIL pre-resampling, so the0/6→1/6 difference cannot be attributed solely to OCR readability. It remains inadequate for adoption.

## Crowded-scene execution recovery

Two instructions on one crowded EMT frame originally exceeded2048 input tokens. Retaining all observations required2274–2275 tokens.

| Runtime profile | Outcome for the two requests |
| --- | --- |
| 2048 input limit,20 resident language layers | Two input-budget errors in the preceding comparison |
| 3072 input limit,20 resident layers | Two CUDA out-of-memory errors |
| 3072 input limit,8 resident layers | Both requests completed; no operational errors |

The successful profile still executes the complete8B model with the same frozen weights and selected LoRA adapter. It streams more layers instead of keeping20 in GPU memory. No observations were removed or input text truncated.

**This is an execution fix for the two tested requests, not a successful grounding result.** Both returned clarification and no stopping-area proposal. Their request times were82.48s and107.92s. Independent provisional AI review found weak questions: the black-car question repeated parked/between information, while the truck question offered generic alternatives rather than identifying visible ground. Human semantic labels remain unavailable. The voice-tagged request used supplied text, not microphone/ASR input.

The validated settings can be applied to an explicitly selected experimental inference config: `max_input_tokens:3072`, `inference_resident_layers:8`, retaining the pinned model, adapter and262144-pixel image limit. They were not silently installed as a production default.

## Higher image-detail limitation

The planned larger-image condition did not perform inference. The selected adapter's manifest binds its image budget, and the streamed backend itself also rejects budgets above262144pixels, including with pretrained weights. The attempted pretrained589824-pixel configuration stopped during initialization with zero calls.

Its model accuracy is **unmeasured**. This is an unsupported runtime configuration, not evidence that the model failed at higher resolution. The guard and adapter manifest were preserved. Testing more visual detail would require a separately validated backend/profile or another image representation; this window did not establish that result.

## Implication for finishing the project

There are separate issues:

1. **Execution capacity:** recovered on the two crowded-frame requests using a lower resident-layer profile.
2. **Input readability/detail:** a verified weakness; the clearer marking variant did not produce reliable selections, and higher-resolution inference remains untested.
3. **Learned grounding and full-interface supervision:** unresolved. The existing adapter learned public destination-coordinate supervision, not this candidate-ID task or the full local status/clarification policy. The pretrained model also missed the reference regions on five of six forced choices.

Do not continue a long run of the same objective and assume the feature will be fixed. The next training decision needs reviewed local ground targets and status/clarification labels, a matching objective and a held-out application test. Candidate-ID supervision can be derived from training-only destination masks, but it must not invent full status labels or be presented as the finished passenger application.

This evidence supports a **no-go for the current passenger-instruction grounding implementation**. Preserve the working QA/RAG work, the real8B training artifact and these diagnostics. Do not turn a forced ID into an automatic `proposed` status or call repeated clarification a working location feature.

## Verification and evidence

Final software checks: **1228 passed,28 skipped,401 subtests passed**. The rendering helper has independent boundary/geometry tests and source review. These checks do not change the model-quality conclusions.

`experiments/dropoff/quick-verification-results-20261121.json` records the durable summary and source hashes. Raw outputs, protocol addenda, process receipts, independent scores and qualitative reviews are under `runs/dropoff-methods-20261121/quick-verify/`. The preceding window and its original failures remain unchanged.
