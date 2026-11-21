# Local 8B training and handoff — 21 November 2026

**The public-destination training milestone is complete. Task 5 and the passenger
feature are not complete: current stopping-area quality is inadequate.**

## Actual training

Both V1 and V2 ran on the laptop's RTX 3050 4 GB GPU. Each completed two epochs,
500 optimizer updates and 1,000 presentations of 500 unique public training
cases. The 125 development cases use separate source scenes. The 58 reserved
concise-test source scenes were excluded from training and selection.

The complete native 8B vision-language model runs with frozen, streamed base
weights. Training updates 1,703,936 LoRA parameters in the final eight language
layers. All 64 adapter tensors changed and remained finite. Both runs selected
epoch two by their own fixed 32-case development NLL. Peak PyTorch allocation
was 2.35 GiB for V1 and 2.25 GiB for V2; these figures exclude driver overhead.

The quantized base is `unsloth/Qwen3-VL-8B-Instruct-bnb-4bit`, revision
`10d842936325905934ea24e64325d774693f166e`. Its original Qwen processor is pinned
separately as documented in [the runtime guide](dropoff-streamed-runtime.md).

## Measured development results

All conditions use the same original images, reference masks and decoding.
Failures remain in the requested-case denominator.

| Model condition | Original-mask point hits | Mean coarse-box IoU |
| --- | ---: | ---: |
| Pretrained | 0/20 | 0.02221 |
| V1, float/box-first teacher targets | 3/20 | 0.09954 |
| V2, integer/point-first teacher targets | 7/20 | 0.19053 |

Each produced valid geometry for all 20 requests. The frozen development
selection rule chose V2, retaining every V1 hit and adding four. Independent
scoring and saved-tensor audits confirmed these results. V1 and V2 started with
identical adapter tensors and used the same case order and update count, but
supervised token counts were 65,238 and 35,842 respectively. The experiment
changes ordering and rounding together; it cannot establish which change caused
the improvement. NLL across different teacher encodings is not an accuracy metric.

On the harder same-image language contrasts, pretrained scored 0/13 and V2
scored 3/13. Neither got every reference correct in any of the six image groups.
Two trained groups produced effectively identical proposals for different
instructions. This does not establish robust language-conditioned grounding.

The full application interface was also compared on the same six predeclared
public development requests. Pretrained proposed four locations and V2 proposed
three; neither condition had a point inside the reference mask. All six outputs
in each condition were structurally valid. Public region masks do not label
clarification or insufficient-evidence correctness, so this is a localization
diagnostic rather than a six-case semantic-accuracy score.

Five further functional requests used two genuine EMT recordings with Task 1
saved predictions and one public parking image. Clarification for an unpointed
“here” and a gray-sedan reference was useful. Three location proposals still
landed on or at vehicle bodies/edges rather than convincingly identifying ground
areas. A model-label-masked AI visual review confirmed these observations; it is
not human benchmark ground truth. The voice flag used a manually supplied
transcript, not microphone/ASR evidence.

Those V2 functional calls took approximately 43–94 seconds each on the streamed
4 GB runtime. This establishes local execution, not live passenger-response
latency acceptance.

One bounded, generic landmark-versus-ground prompt trial retained those failures
and was rejected. Its outputs are preserved. Production prompts, trained weights
and geometry are unchanged; predictions are never moved using reference masks.

## Software and artifact verification

- Repository suite: 1,122 passed, 28 optional skips, 401 subtests passed.
- Exported source: 597 passed, 17 optional skips, nine CLI help checks, with
  project imports resolved inside its separate extraction.
- Unchanged core runtime: a separate 87-test Torch CPU/numerical suite passed.
- Actual selected-adapter reloads matched all 64 saved tensor hashes.
- The public handoff packager verified the full schedule, data/source bindings,
  both actual generated comparisons and an isolated-worker public spatial probe.
- The local-continuation CLI accepted that real staged parent without loading
  Torch. Its reviewed-local training workflow has software checks but has not
  been run on genuine local interpretation labels.

The portable GPU trial uses a new extracted project, the bundled public probe,
the saved adapter and explicit locations for the shared pinned base/processor.
It reuses the existing CUDA environment. Archive hashes and the actual reload
outcome are recorded in the release receipt. Another teammate's independent
installation remains an acceptance step.

The private [public-training milestone release](https://github.com/A1Prep/TrackRAG-Gulf/releases/tag/task5-local-8b-20261121)
provides the selected adapter, current minimal source, diagnostic evidence and
`release_receipt.json`. Base weights and bulk datasets remain separate. Follow
[portable extraction and startup](dropoff-local-training.md#portable-inference).
The exact training snapshot inside the adapter is for a separate reproduction
copy; do not overlay it onto the current production source.

## What the team needs next

1. **Task 2: finish actual local stopping-area labels and review.** The checked
   release at commit `186a80e` contains 45 train, 20 development and 50 evaluation
   drafts, all pending. Reviewer forms have zero completed decisions. Local
   targets contain landmark track IDs, not the required destination point,
   ground-area box, status and clarification question. Supply those fields,
   real review identities/bindings and portable image files/hashes. Do not turn
   a vehicle's centre or box into a stopping target. The declared train/dev paths
   currently refer to another computer; only two frame-name candidates are
   available in this checkout.
2. **Task 2: keep the audited split policy.** The new combined public release
   includes 338 source-validation rows and 170 IDs covered by our reserved
   concise-train guard. They must not replace the clean 500-case training split.
   Keep local recording splits disjoint and `video_122233` reserved. The added
   `trackrag/dropoff.py` also conflicts with the existing package; integrate
   selectively after fixing its namespace and target contract.
3. **Task 5: run reviewed-local continuation and evaluate the actual interface.**
   The [4 GB continuation command](dropoff-local-adaptation.md) is prepared and
   independently reviewed. It requires genuine full-interpretation train/dev
   data and all three training statuses. Perform real updates, fresh reload and
   semantic/location checks once that release exists; software fixtures do not
   complete this step.
4. **Task 3: complete speech and QA acceptance.** The reviewed branch `474713a`
   needs its WAV-header/sample-rate handling, environment loading and audio
   dependencies corrected. Preserve existing QA privacy and factual-scoring
   checks when integrating. Supply actual scene-linked recordings and human
   transcripts for a real ASR-to-grounding trial.
5. **Task 4: connect the application and confirmation flow.** Use
   `GroundingModel.interpret_with_receipt`, display the proposed area or useful
   unresolved response, and preserve original/corrected transcripts. Confirmation,
   correction, cancellation and invalidation on a scene change belong to the
   application. A proposal does not control a vehicle or certify a stopping area.
6. **Task 1: freeze the final conditions and perform independent evaluation.**
   After local adaptation and integration, compare actual pretrained/trained
   outputs under the same settings on reviewed reserved cases. Keep final
   references outside training and model selection, and retain every failure.

Issue #5 remains open. The successful local fine-tuning and artifact checks do
not remove these data, quality, integration and independent-evaluation gates.
