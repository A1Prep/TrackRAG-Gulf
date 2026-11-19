# Task 1 trained artifact verification

Verified on 19 November 2026 against Task 1 commit `6ea0e126ef77aa1e9b9c0f1746896a0b2b238e29`. The compact machine-readable record is [the verification summary](../experiments/vision/task5_artifact_verification_20261119.json).

The supplied trained checkpoint loads through the real Task 1 adapter and produces road-user predictions on original EMT images. Its outputs connect to the normal reviewer. The supplied test tracking results have been independently recomputed from the official labels. Full-test detection AP remains an archived result because the prediction ZIPs do not include raw detection files.

## Supplied artifacts

| Artifact | SHA256 |
| --- | --- |
| `checkpoint_best_total.pth.zip` | `9ea96f1050f1d3375c5cf0f25a086cd5bb9c6a290fa5fdc009804d9a31181d95` |
| `task1-predictions-finetuned.zip` | `572b0322050f23d918e5a68fb12c8ecde8d1a241788f17b9c485d6af1d71a510` |
| `task1-predictions-pretrained.zip` | `08ebd92fc4fef5b1abc194360e420d3be19c05364f3d93ff9929631052baf3fe` |

The checkpoint's 134,900,699 bytes are already in PyTorch's native ZIP-based checkpoint format. The extra `.zip` filename suffix does not indicate an outer archive to unpack. It was copied as `models/task1/checkpoint_best_total.pth`, with its original hash preserved. Prediction folders were extracted under ignored `data/task1-shared-20261119/`. Archive README text was not executed.

Both prediction archives pass ZIP integrity checks. Each contains eight recordings' track and provenance files, plus development/test metrics. All 16 track hashes and all 14 development/test metric links to those track/provenance bytes match.

## Checkpoint inspection and actual inference

Restricted `torch.load(weights_only=True)` succeeds without custom unsafe globals. The checkpoint records RFDETRLarge, rfdetr 1.12.1, EMA selection and global step 952, consistent with the recorded epoch-4 selection at 238 steps per epoch. It contains the expected nine EMT class names in order. Compared with the pinned base checkpoint, 480 same-shaped tensors differ, including backbone tensors; classification heads change from 91 to 10 outputs, including background. This supports actual adaptation beyond replacing metadata.

Both the pretrained and fine-tuned profiles processed the first 150 frames of `video_054604` and `video_054907`: 300 original images per profile, 600 model-frame executions in total. Images were hard-linked into an image-only recording root without annotations. Inference used CPU torch 2.15.0 and the normal `trackrag.vision.predict` path; package versions and source hashes are recorded locally.

Every fine-tuned exported observation matches a supplied observation after geometric association, with no unmatched observations or class mismatches:

| Recording | Observations | Minimum matched box IoU | Largest coordinate difference |
| --- | ---: | ---: | ---: |
| `video_054604` | 476 | 0.999963 | 0.0013 pixels |
| `video_054907` | 1,840 | 0.991538 | 2.4175 pixels |

Numeric track IDs are unchanged in the first recording. In the second, 58 identity numbers differ while preserving consistent one-to-one correspondence throughout the compared observations. The serialized outputs are not byte-identical; the claim is measured geometry, class and identity-structure agreement. No ground-truth identities were used for this fresh-versus-supplied comparison.

A large wall-clock delay occurred during the fine-tuned second recording alongside a tool-host reconnect. These verification runs are not a controlled latency comparison.

## Independent test tracking recomputation

The official annotation archives were fetched from EMT revision `4418cec01b03c6f797cadb06c9405843a30988e2`. Their file SHA256 values match the revision's LFS object IDs: test `a5d6bb56995738acd1251ffcdfea11432a2f47d96e1da5e3c34897debde45372`, train `58ac63c71f1b6c12dfc8b31dde20afdfbcac9af3066929526053968438fcfb44`.

Tracking metrics were recomputed over all five supplied test recordings, 8,530 frames and 107,894 ground-truth road-user observations. Every available numeric result across the road-user and class-group views matches the supplied metric files within `1e-9`. The actual track, provenance and official-label hashes also match the archived evaluation inputs.

| Class-agnostic road-user metric | Pretrained profile | Fine-tuned profile |
| --- | ---: | ---: |
| IDF1 | 0.399853 | 0.442683 |
| MOTA | 0.230847 | 0.340705 |
| False positives | 26,628 | 18,456 |
| Misses | 54,952 | 50,817 |
| ID switches | 1,407 | 1,861 |
| Fragmentations | 2,561 | 2,830 |

Aggregate identity score and MOTA improve, while switches and fragmentations increase. The detector/tracker profiles use different development-selected tracking settings, so this comparison evaluates the combined profiles rather than isolating detector fine-tuning.

The archived full-test road-user detection AP is 0.303455 versus 0.354616. These numbers agree with the handoff but were not independently reproduced: neither ZIP includes the corresponding detection JSONL. Track boxes must not be substituted for scored detections to manufacture an AP result.

## Fresh development-prefix results

The two new 150-frame inference runs per profile include raw detections, allowing a separate local detection/tracking comparison:

| Metric on these 300 development frames | Pretrained profile | Fine-tuned profile |
| --- | ---: | ---: |
| Road-user AP | 0.286363 | 0.303659 |
| Road-user AP50 | 0.512 approximately | 0.534 approximately |
| IDF1 | 0.365813 | 0.377758 |
| MOTA | 0.221 approximately | 0.229 approximately |

These are development-prefix results, with class-agnostic road-user detection scoring. They are not the five-recording test benchmark or full nine-class accuracy. Detailed values remain in the machine-readable summary and local metric reports.

## Reviewer integration and checks

The normal `trackrag review` command loaded the newly generated fine-tuned tracks through the explicitly selected saved-prediction adapter on both image-only recordings. All 36 public development questions completed evidence export. The existing lexical retriever returned evidence for 26 and empty results for 10; this is output coverage, not relevance accuracy. All answers are `not_requested` and no provider was called.

The saved-run auditor verified question text, recording/source identity, cutoffs, evidence geometry, file references and readable images for both runs. There are 60 exported evidence image files. This verifies handoff consistency, not factual answers or semantic retrieval success.

Three concrete integrity issues were addressed:

1. Saved track loading now requires a valid track SHA256, validates complete artifact identities, frames, classes and geometry, and verifies the available recording prefix before returning it.
2. Full detection evaluation now requires exactly one record per declared frame, including empty detections. Missing/duplicate frames, foreign sources and invalid scores/boxes are rejected.
3. `--tracking-only` no longer opens absent detection files. Its result omits AP, detector coverage and detection hashes rather than implying those measurements were computed.

The primary suite passes 388 tests and 285 subtests, with four optional-dependency skips. The optional metric fixtures pass 54 checks in the model/evaluation environment, and the three real-tracker/training-config checks pass there separately. Two independent AI-assisted source reviews found no remaining actionable issue.

## Reuse and reproduction

Use the provided prediction folders without a GPU, with the model profile selected explicitly. For example, in PowerShell:

```powershell
$env:TRACKRAG_VISION_CONFIG = 'configs/vision/finetuned.json'
$env:TRACKRAG_SAVED_PREDICTIONS = 'data/task1-shared-20261119/predictions-finetuned'
python -m trackrag review --adapters tools.vision_saved_adapters --source predicted --data-root <EMT_ROOT> --recording video_054604 --questions experiments/temporal-development-questions.jsonl --answerer none --out <NEW_RUN_DIRECTORY>
```

For fresh inference, use the Task 1 model environment and a new destination:

```text
python -m trackrag.vision predict --data-root <EMT_IMAGES_ONLY_ROOT> --recordings video_054604 video_054907 --config configs/vision/finetuned.json --weights-dir models/task1 --device cpu --max-frames 150 --out <NEW_PREDICTION_DIRECTORY>
```

Recompute the supplied test tracking without downloading test images or fabricating detections:

```text
python -m tools.vision_evaluate --pred-dir <PREDICTION_FOLDER> --labels-root <OFFICIAL_LABELS_ROOT> --split test --recordings video_122233 video_125233 video_155425 video_160325 video_204647 --tracking-only --out <NEW_METRIC_FILE>
```

Raw footage, checkpoints and prediction bundles remain outside Git. Full-test AP reproduction still needs the original detection files. This verification resolves the trained vision handoff; it does not complete the separate NLP training or passenger-instruction work.
