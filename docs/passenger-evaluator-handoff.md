# Passenger grounding handoff to Task 1

## Preparing compatible scenes

Use the same original images, prediction folder and vision profile for the model requests and the evaluator scenes. Keep the source folder in the run record. On the daytime development recording, all 19 requests from the team’s shared prediction ZIP matched the 19 scenes exported from that ZIP. All 19 earlier development requests matched scenes exported from the fresh development detector run. The cross-source comparison matched zero of 19. Later frames had different track identities as well as small differences in coordinates. [Measured source comparisons](../experiments/dropoff/passenger-task1-evaluator-handoff-20261121.json)

For the participant study, Task 2 supplies the case IDs, instructions and reviewed destination masks. Task 3 supplies the intended regions and replies for ambiguous cases before model responses are collected. Task 1 freezes the protocol, reviewer records and source groups. The five recordings are excluded from Task 5 model fitting and selection. The separate original-input comparison uses them for evaluation. [Task 1 evaluation instructions](task1-grounding-evaluation.md)

The corrected Task 2 package records two reviewer confirmations of AI-assisted labels for four training images and two development images. All six requests pair an empty observation list with “Stop beside the car directly ahead.” Both forms record acceptance of “Which visible car ahead should I stop beside?” for each image. The original JPEGs and the six model-visible inputs match their review hashes. Task 1 has saved predictions for three of these frames. Adding those detections produces a new input that would need its own review. [Independent review audit](../experiments/dropoff/passenger-human-six-independent-audit-20261122.json) · [Original EMT pixel verification](../experiments/dropoff/passenger-human-six-source-pixels-20261122.json)

## Native direct interpretation

The native route runs the full image, predicted observations and passenger dialogue through Qwen, then converts the four-field response into original-frame pixels. Keep the complete output directory from `python -m tools.dropoff_predict predict`. Task 1’s `python -m evaluation.grounding_inputs task5-run` reads its `run.json`, `resolved-requests.jsonl`, `model-outcomes.jsonl` and `receipts.jsonl`. Give Task 1 the byte-identical configurations, requests and results for both the pretrained and trained conditions.

The original native compatibility run contained four EMT instructions and one public RegSeg control. Task 1’s unmodified field converter consumed all five native envelopes. Its scene checker matched all four EMT inputs against their actual source. The public control had no EMT scene. No final reference file or assessment score was produced during this exercise. The upstream software suite also passed 80 evaluator and converter tests. [Recorded handoff](../experiments/dropoff/passenger-task1-evaluator-handoff-20261121.json)

The registered pretrained and locally fine-tuned 8B models also processed the same two development requests on different EMT recordings. Both runs produced two proposed areas, and Task 1’s conversion functions read all four native responses. On the request for the road just behind the white SUV on the left, the pretrained model proposed pixel `(460.8, 1008)` and the trained model proposed `(460.8, 1080)` in the 2560 × 1440 frame. There is no passenger-marked target for this comparison. For this recording the team could independently verify the saved tracks and their provenance. The original detections file named by that provenance was unavailable for rechecking. [Measured runs and source details](../experiments/dropoff/passenger-evaluator-paired-delivery-20261122.json) · [Download the original development responses and verifier](https://github.com/A1Prep/TrackRAG-Gulf/releases/download/passenger-feature-20261122/passenger-evaluator-paired-delivery-20261122.zip)

The experimental native child adapter is separately packaged. Extract `native-direct-8b-v3-verified.zip` into `models/passenger-ai-direct/` under the project. Its `inference_config.json` specifies two resident decoder layers for the 4 GB laptop. The package includes the checksum of each member and evidence from a fresh GPU inference. The pinned 8B base and processor cache use the project’s existing local model setup. [Verified adapter and reload](../experiments/dropoff/passenger-native-child-reload-20261121.json)

```sh
python -m tools.dropoff_predict predict --config models/passenger-ai-direct/inference_config.json --requests requests.jsonl --out native-results
```

### Native trial with the Task 2 reviews

The local 8B clarification adapter took four optimizer steps on the four approved training photographs. On the two development photographs, both the public parent and the new adapter proposed a location when the recorded decision called for a question. On two separately labelled, unreviewed paraphrases, the new adapter asked one relevant question and the parent asked none. Both models handled the clear control and proposed incorrectly on the missing-reference control. All 12 native calls returned valid four-field responses. [Per-request results and exact questions](../experiments/dropoff/passenger-human-reviewed-trial-20261122.json)

A matched pretrained run added six native responses on the same inputs. Across all three conditions the models missed both reviewer-confirmed requests for a question. On the two unreviewed paraphrases the pretrained model matched the expected response type once, the public adapter zero times and the locally reviewed adapter once. Each matched one of the two AI-authored controls. All 18 raw replies reproduced their saved four-field responses using the original image dimensions. [Three-model comparison](../experiments/dropoff/passenger-native-reviewed-comparison-20261122.json)

Extract the direct-inference archive into `models/passenger-reviewed-clarification/` at the repository root. It includes the trained weights, a configuration for the 4 GB laptop and the raw native receipts from both conditions. It uses the project's existing pinned 8B base and processor. The two reviewed inputs can be rebuilt from the corrected Task 2 Git commit. In a shallow clone, fetch `Task-2-arjun-part-2` first. [Download the native clarification adapter](https://github.com/A1Prep/TrackRAG-Gulf/releases/download/passenger-feature-20261122/passenger-human-native-clarification.zip) · [Archive checksum and verified tensor changes](../experiments/dropoff/passenger-human-native-package-20261122.json)

```sh
python -m tools.dropoff_import_task2_review --out reviewed-emt
python -m tools.dropoff_predict export-requests --cases reviewed-emt/development.jsonl --partition development --out reviewed-emt/requests.jsonl
python -m tools.dropoff_predict predict --config models/passenger-reviewed-clarification/inference_config.json --requests reviewed-emt/requests.jsonl --out reviewed-native-results
```

## Composite location proposals and passenger corrections

On the same six original inputs, the full passenger application behaved identically with both adapters. It withheld a location on the two human-confirmed requests. Both called for a distinguishing question. It asked a distinguishing question on both unreviewed paraphrases and matched both provisional AI-labelled controls. [All twelve original application responses](../experiments/dropoff/passenger-human-application6-20261122.json)

Adding the 23 and five genuine Task 1 detections to the two development frames changed what the application could ask. Both adapters asked which car the passenger meant on the original wording. Both proposed a spot on the two paraphrases. The passenger application switches to its track-aware landmark prompt when it receives detections. None of these four changed requests has a new human rating. [Frozen input comparison](../experiments/dropoff/passenger-operational-context-protocol-20261122.json) · [Eight live tracked responses](../experiments/dropoff/passenger-operational-context-20261122.json)

An additional clarification check handles phrases relative to the passenger, such as “straight ahead of us.” On 24 separately written, AI-labelled language cases using real detections and the learned referent ranker, it matched 22 expected actions. The earlier policy matched 13. It preserved all ten controls for specific landmarks, ordinals, attributes and static destinations. Replaying the new check on the eight saved tracked responses converted the four paraphrase proposals into questions grounded in actual detected alternatives. All twelve original application responses stayed the same. [Independent language study](../experiments/dropoff/passenger-egocentric-language-results-20261122.json) · [Saved-output policy replay](../experiments/dropoff/passenger-egocentric-application-replay-20261122.json) · [Raw receipts and archived policies](https://github.com/A1Prep/TrackRAG-Gulf/releases/download/passenger-feature-20261122/passenger-language-context-evidence.zip)

To reproduce the language comparison on Windows or Linux, the study runner verifies the three Python modules against their registered SHA-256 values after converting CRLF line endings to LF. It checks the exact file bytes for the registered protocol, weights, recordings, images and saved rankings.

The interactive route uses Qwen to identify a landmark, a separately learned estimator to locate a potential destination, and a road model to check the visible surface. A passenger can clarify, mark a different area or confirm the current proposal. Its audit export labels those events separately and preserves the original model output, selected source, geometry, road evidence and file hashes. The current Task 1 protocol lists native direct and clarification conditions. A composite study needs its own declared condition and selection procedure.

```sh
python -m tools.dropoff_passenger_handoff batch --run batch-run --requests requests.jsonl --replay raw-landmarks.jsonl --out composite-evidence.json
python -m tools.dropoff_passenger_handoff session --journal session.jsonl --out interaction-evidence.json
```

Omit `--replay` for a live batch. For a saved-landmark batch, supply the original raw decision file. For a voice session, keep the original WAV in the session’s `audio/` directory. These examples write separate evidence files. The exported JSON can include utterances, local paths, model prompts and timestamps.

## Preparing the 44 corrected final inputs

### Restore the published inputs on another checkout

From the repository root, use a Python environment with the project’s integration and grounding-format dependencies. The three ZIPs total about 6.2 MiB. Task 2’s Git branch supplies the 44 original photographs. The restore command reads only the named JPEGs from that revision and needs no GPU. [GitHub release](https://github.com/A1Prep/TrackRAG-Gulf/releases/tag/passenger-feature-20261122)

```powershell
git fetch origin Task-2-arjun-part-2
gh release download passenger-feature-20261122 --repo A1Prep/TrackRAG-Gulf --pattern passenger-44-inputs-only-20261122.zip --pattern passenger-44-canonical-scenes-20261122.zip --pattern passenger-human-native-clarification.zip --dir runs/evaluator-downloads --skip-existing
python -m tools.dropoff_restore_evaluator --archives runs/evaluator-downloads --out runs/evaluator-inputs
```

This installs the registered trained adapter under `models/passenger-reviewed-clarification` and verifies its configuration against `configs/dropoff/passenger-final-pretrained.json`. It restores the images and checks all 44 cases, five recordings and 494 detector observations through the Task 1 scene reader. Run the command again to verify an existing installation. Keep the generated images under the private output directory.

Use `inputs/requests-with-saved-tracks.jsonl` with `canonical/scenes` for the detector-assisted passenger study. Task 1’s scene reader matched this input on all 44 frames. The original review input, `inputs/requests-as-human-reviewed.jsonl`, has the same pixels, timestamps, instructions and dialogue. Its empty observation lists differ on all 44 frames. The previous reviewer decisions bind to those original inputs. Reviewing the additional detector information is still necessary before applying a human label to the changed input. Keep references, replies and model outputs in separate directories. [Measured input comparison and technical selection](../experiments/dropoff/passenger-gulf-input-choice-20261122.json)

On the GPU machine, run `python -m tools.dropoff_prepare_streamed_model` from the repository root to install the pinned base model and processor. Use `python -m tools.dropoff_prepare_streamed_model --offline` to verify an existing installation. The input restore reports the absolute paths to both model configurations and to the scene directory that the evaluator reads. [GPU environment and base installation](dropoff-local-training.md)

A clean Windows checkout restored all 44 original photographs from Task 2 and loaded all 44 scenes with Task 1’s reader. Rechecking preserved all 106 installed files. A changed photograph or an extra reference file made the verification fail, and it succeeded again after the test changes were removed. This cold installation used the same laptop and downloaded the three archives afresh. [Measured restore and retry results](../experiments/dropoff/passenger-evaluator-restore-20261122.json)

### What the existing Gulf reviews can measure

Two reviewers accepted AI-assisted annotations on each of the 44 original images. They approved proposing a location on 33, asking for clarification on four and reporting insufficient visual evidence on seven. The 33 proposed locations each have an approved point and rectangle. The four approved questions share one wording. These records support a direct measurement of the model’s decision, predicted point and rectangle. The source files contain no separately drawn destination masks, passenger replies or confirmations. [Frozen original-input scoring rules and supervision counts](../experiments/dropoff/passenger-reviewed-original-eligibility-20261122.json)

The project chose to use these previously reserved photographs for an offline comparison of the two frozen 8B models. A later passenger study on the same recordings must disclose that use. Keep later participant intentions and model-specific confirmations under the separate interaction protocol.

### Rebuild the scene evidence from source

Task 2’s corrected package contains 44 confirmed final requests across five recordings and six other final cases awaiting confirmation. All 44 confirmed requests were reviewed with an empty track list. Task 1’s saved fine-tuned detector supplies 494 observations across the same 44 frames. Keep the original reviewed request file and the new detector-bound request file as separate conditions. A participant can record an intended area from the original image and words. Approval of a changed detector context is a separate Task 2 decision.

The input-only workflow reads original images and saved detector outputs. The native Task 1 exporter checks the full run prefix through the selected frame, even when scene history in the model input is empty. The following commands use fresh output locations and the extracted Task 1 prediction folders. The source worker streams the official compressed archive and keeps one genuine recording prefix on disk at a time. The evaluator receives one canonical JSON file per frame. Keep the original images in the private source location.

The November 22 run verified the full 8,720,080,037-byte official archive and independently checked each of the 7,235 required JPEGs after writing it to disk. All 44 native scene exports matched the beforehand-sealed inputs and detector provenance. A separate process extracted the private `passenger-44-canonical-scenes-20261122.zip`, verified every listed member, and loaded all 44 scenes against both Task 1 manifests. The ZIP contains JSON only. [Recorded source hashes and remaining review requirements](../experiments/dropoff/passenger-final-input-readiness-20261122.json)

```sh
python -m tools.dropoff_final_input_inventory --predictions data/task1-shared-20261119 --out runs/final44-inputs --verify-archive
python -m tools.dropoff_final_handoff_checks --bank runs/final44-inputs --predictions data/task1-shared-20261119 --out runs/final44-inputs/coverage.json --bundle runs/final44-sparse.zip
python -m tools.dropoff_official_final_scenes --bank runs/final44-inputs --predictions data/task1-shared-20261119 --out runs/final44-native
python -m tools.dropoff_package_final_scenes --bank runs/final44-inputs --native runs/final44-native --sparse-archive runs/final44-sparse.zip --archive runs/final44-canonical.zip --summary runs/final44-source-summary.json
```

The scene identifiers cover the model inputs and detector provenance. The ZIP additionally records extraction timestamps and elapsed times. Task 1 should freeze the reviewed source condition, any required masks and the passenger reply rules before running the four final prediction conditions. The source commands above produce zero final model answers.

## Collecting native clarification episodes

Task 1 can run the typed clarification condition with `tools.dropoff_native_episodes`. The proposed pair uses the pretrained Qwen and the adapter continued on four locally reviewed training photographs. Both use the same image, prompt and generation settings. The interactive application has a separate model selection. [Pinned model pair](../experiments/dropoff/passenger-final-model-conditions-20261122.json)

Before any final model output, finish each passenger’s independent intention and registered replies, Task 2’s review of the exact source condition and Task 1’s references. The study owner supplies the actual responder instructions in `responder-rules.txt`, the agreed model turn budget, the full evaluator Git commit, and one assigned participant code per case in a private JSON object. A small example is `{"C001":"P01","A007":"P02"}`. The full assignment file must cover every case. The protocol command calls Task 1’s original freezer and also binds the assignment file and budget. Include `--voice voice.jsonl` at this freeze if the official study includes voice.

```sh
python -m tools.dropoff_native_episodes freeze --references refs/references.jsonl --replies replies.jsonl --participants participants.json --responder-rules responder-rules.txt --turn-budget <agreed-budget> --evaluator-commit <full-git-commit> --model-conditions experiments/dropoff/passenger-final-model-conditions-20261122.json --out protocol.json
python -m tools.dropoff_native_episodes prepare --protocol protocol.json --references refs/references.jsonl --replies replies.jsonl --participants participants.json --scenes runs/evaluator-inputs/canonical/scenes --requests runs/evaluator-inputs/inputs/requests-with-saved-tracks.jsonl --config configs/dropoff/passenger-final-pretrained.json --out native-pretrained-clarify
python -m tools.dropoff_native_episodes prepare --protocol protocol.json --references refs/references.jsonl --replies replies.jsonl --participants participants.json --scenes runs/evaluator-inputs/canonical/scenes --requests runs/evaluator-inputs/inputs/requests-with-saved-tracks.jsonl --config models/passenger-reviewed-clarification/inference_config.json --out native-trained-clarify
```

`prepare` performs no inference. It verifies the original image, predicted observations, frame, initial typed instruction and same-frame dialogue. The freeze command defaults to the linked Task 5 model pair. Production preparation checks each exact registered configuration and the trained adapter manifest before collection. The runner reads the permitted identifiers and wording from the frozen references.

1. Run `python -m tools.dropoff_native_episodes collect --session native-trained-clarify`. This calls the normal prediction adapter for the active cases and saves its full receipts. The first round contains the entire cohort. The collector reconstructs the model prompt, reparses successful raw replies into original-image pixels and checks every receipt against its saved outcome or error before opening review.
2. Open `native-trained-clarify/round-00/review/review.html` to locate each assigned participant’s separate HTML file. Give each participant only their own file. It embeds their original photographs and displays the model’s actual question or its proposed point and rectangle. Participants can confirm that particular proposal, use one of their own eligible preregistered replies, or end the case without confirming it. A participant saves their choices and explicitly consents before downloading their JSON return.
3. Run `python -m tools.dropoff_native_episodes advance --session native-trained-clarify --return P01.json --return P02.json`, supplying every applicable return for this round. A question can receive a registered answer. A proposed area can receive a registered rejection. The next native model call sees the original conversation, the model’s actual previous response, and the exact selected reply on the same frame. If another round is prepared, repeat from step 1. Model errors stay in their original turn. Each completed case keeps its initial result, and every later proposal keeps its own confirmation identifier.
4. Repeat for `native-pretrained-clarify` with the same protocol. Give Task 1 each session’s `final/outputs.jsonl`, resulting receipt, and the two separately converted native direct outputs for its existing `python -m evaluation.grounding` command. Retain the private source sessions for audit. Review forms contain original footage and passenger choices.

The tool identifies participants by their assigned code and self-attestation. It records no confirmation until an eligible person explicitly selects the displayed proposal. Controlled software tests cover a clarification followed by confirmation, a rejected proposal, a case finished without confirmation, and a failing model call. The unchanged evaluator scores the turn semantics and rejects the flagged software fixtures as formal evidence. No official final answer or participant return was generated during this software test.
