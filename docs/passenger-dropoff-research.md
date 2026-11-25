# Passenger drop off research and delivery plan

Research and repository audit: 19 November 2026. Delivery horizon: next week; the exact submission date still needs confirming.

Keep the existing TrackRAG footage-question workflow and add a passenger-instruction workflow. The approved addition interprets a requested nearby stopping area, shows its interpretation and asks for confirmation. Voice applies to this interaction. Vehicle control is outside the approved scope.

The central NLP task should be **grounding a passenger's language in the current scene**, with clarification when the intended reference is unresolved. A microphone, larger model or vector database alone would leave the original weakness in problem definition and task evaluation unresolved.

## Users and the problem

The primary user is a passenger communicating a nearby intended drop-off location to an AI interface. The prototype demonstrates that the system understood the passenger; a driver or a separate driving system retains responsibility for execution. Fleet and interface developers are a possible downstream audience, rather than validated customers of this student prototype.

The practical problem is more specific than navigation to an address: an address or pin may not identify the desired side, entrance or nearby visual location. Uber documents wrong-side pickup pins and the need for descriptive pickup labels. Waymo permits changing among eligible spots and requesting another pullover location when the arrival spot is unsuitable. These sources establish a communication problem, not proof that this particular voice interface improves it. [Uber pickup research and product changes, 2021](https://www.uber.com/us/en/newsroom/smooth-pickup-experience/), [Waymo pickup and drop-off help](https://support.google.com/waymo/answer/9696059?hl=en).

For ordinary human-driven taxis, another screen or confirmation step could add effort to an instruction the driver already understands. Measure that benefit rather than assuming it. The strongest immediate framing is passenger-to-machine interpretation with an inspectable visual result. The need for a UAE-specific deployment remains a hypothesis to validate with local users.

Existing products already address part of this space. Google Maps introduced conversational voice and landmark guidance in November 2025, and expanded destination-arrival guidance in March 2026. Uber announced driver voice features in September 2026. Their published features do not establish that they solve the exact camera-grounded two-car instruction, but they prevent a broad claim that voice or landmark navigation is new. [Google landmark guidance](https://blog.google/products-and-platforms/products/maps/gemini-navigation-features-landmark-lens/), [Google Maps March 2026 update](https://blog.google/products-and-platforms/products/maps/ask-maps-immersive-navigation/), [Uber driver voice](https://www.uber.com/us/en/blog/voice-in-the-driver-app/).

Before making a usefulness claim, conduct a few short interviews with actual riders and, where available, drivers. Ask for their most recent difficult local pickup/drop-off, the words they used, whether the driver and passenger could see the same landmarks, and whether visual confirmation would have helped or distracted them. Record the sample and responses; do not invent market-size or time-saving figures. A small convenience sample is exploratory evidence.

## Research that changes the design

| Work | What it establishes | Consequence for this project |
| --- | --- | --- |
| [Talk2Car, EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1215/) | Passenger language can identify objects in driving scenes. Its original target is a referred object. | Finding the car is one part of finding the intended destination. |
| [Grounding Linguistic Commands to Navigable Regions, IROS 2021](https://arxiv.org/abs/2112.13031) | Talk2Car-RegSeg supplies destination-region masks. The introduction explicitly discusses parking between two differently coloured cars. | Neither two-car instructions nor road-region grounding can be claimed as new. |
| [Talk2Car-Destination, AAAI 2022](https://cdn.aaai.org/ojs/19952/19952-13-23965-1-2-20220628.pdf) | Physical destinations are different targets from referred object boxes. | Do not present a box around a car as the location where the passenger wants to stop. |
| [Giving Commands to a Self-Driving Car, 2021](https://arxiv.org/abs/2106.04232) | Uncertain references and clarifying questions were already studied with Talk2Car. Its associated extension is Talk2Car-Expr. | Evaluate the clarification policy; do not claim that asking a question is itself novel. |
| [Audio-Visual Understanding of Passenger Intents, 2020](https://aclanthology.org/2020.challengehml-1.7/) | Passenger-intent research already combines spoken interaction and visual context. | Adding ASR is an input layer, not the main research contribution. |
| [doScenes, ITSC 2025](https://ieeexplore.ieee.org/abstract/document/11423583) | Real driving scenes have passenger-style instruction annotations. The preprint dates to December 2024; the publication was added to Xplore in March 2026. | Relevant background, but scene instructions do not automatically provide image-space drop-off labels. |
| [ClarifyVC, ICLR 2026](https://openreview.net/pdf?id=afO3vnSNsS) | Hybrid data and evaluation address ambiguous in-vehicle commands and multi-turn clarification. Its core task concerns vehicle functions rather than road-region prediction. | Useful methods and prior art; not a substitute for our spatial labels. |
| [doPlan, September 2026 preprint](https://arxiv.org/abs/2609.38028) | 5,154 human-written instructions cover persistent, deferred and multistage intentions. | Avoid claiming that retaining passenger intent over time is new. Long-horizon planning is unnecessary for this deadline. |

A February 2026 [doScenes/OpenEMMA study](https://arxiv.org/abs/2602.04184) also examines instruction-conditioned trajectories. Its headline improvement is strongly affected by extreme baseline outliers. This illustrates why task compliance and failure analysis matter alongside aggregate scores; it does not justify operational reliability claims here.

The proposed research question is: **Does a locally adapted grounding model, combined with targeted clarification, reduce incorrect interpretations of passenger drop-off instructions under a fixed interaction budget?** This is a testable project contribution, not a claim of inventing visual grounding.

## Required output and interaction

The full target is a proposed image region with supporting landmarks and confirmation. For the one-week build, start with an explicitly labelled **destination point** inside the intended region. If a coarse region is promised, add a mask-derived region-box target and evaluate that box separately. A fixed-radius circle drawn around a point is only a display marker; it is not a predicted area or segmentation mask.

The RegSeg paper's Pointing Game counts whether a predicted point lies inside a human-annotated destination mask. This supports point grounding as a legitimate first experiment. It does not measure metric clearance or legal stopping suitability. [Evaluation definition](https://arxiv.org/html/2112.13031v1).

The new output record needs a separate contract from a footage question:

- Recording, original frame, utterance timestamp and permitted history cutoff.
- Passenger intent and spatial relation.
- Resolved landmark references, scoped to their recording, source and current frame.
- A proposed original-image point; optionally a coarse region box if actually supervised and evaluated.
- Status: proposed, needs clarification, or insufficient visual evidence.
- A clarification question when needed.
- A proposal identifier and confirmation state tied to that proposal and frame.

Examples to include in the annotation rubric:

| Situation | Required behaviour |
| --- | --- |
| A unique identifiable pair of black cars and a visible intended intervening region | Ground the references and propose a point, then request confirmation. |
| Several plausible pairs of black cars | Ask a question that distinguishes the plausible pairs. Do not select one arbitrarily and merely ask for yes/no confirmation. |
| Bare "drop me here" without a selected location or spatial context | Ask which visible place is intended, or request a selection. Speech alone does not supply a pointing gesture. |
| A required landmark is absent, occluded or unobservable | Report insufficient evidence or request a different description. |
| A previous reference has moved or the scene changed | Revalidate against the current frame and ask for renewed confirmation if the proposal changed. |
| Speech changes a colour, direction, number or negation | Expose or confirm the interpreted instruction before accepting a location. |

The midpoint of two vehicle boxes is not a physical free-space estimate. Colour must be observed from image pixels; the nine-class EMT road-user taxonomy does not contain black-car attributes. A proposed point expresses interpreted intent, not a certified place to drive or stop.

## Data that is available

| Source | Verified availability and label scope | Planned use |
| --- | --- | --- |
| [RegSeg full release](https://github.com/kanji1011/Talk2car-Refseg) | The public ZIP was reachable and advertises 1,485,242,059 bytes. The project reports 8,349 training and 1,163 validation image-command pairs. Full training-archive contents have not been loaded in this audit. | Image-command destination supervision after joining and validating images and masks. |
| [RegSeg concise test commands](https://raw.githubusercontent.com/kanji1011/Talk2car-Refseg/main/dataloader/talk2car_test.json) | The actual 44,243-byte JSON contains a `test` dictionary of 508 records with `command` and `img`. Its SHA256 is `91b3b5f169b1d22c0569fe60b0aa35dee0c89e7a5e945558616e4420152127a2`. Range inspection of the linked 75,039,632-byte ZIP found 508 JPGs and 508 PNG masks. | A small, accessible evaluation source. The paper's nominal count is 500; report the actual release used. |
| [Talk2CarSlim](https://github.com/talk2car/Talk2Car) | Official frontal-image release is about 1.77 GB; the full 300 GB+ sensor collection is unnecessary for this task. Original command metadata also supplies referred-object annotations. | Optional object-reference supervision and source-scene mapping. Avoid downloading redundant image archives. |
| Existing EMT recordings | Frames and road-user boxes/tracks are available. Passenger commands, destination regions and ambiguity labels are not supplied by those boxes. | Existing 901 evaluation, QA, and a modest local grounding subset only where the scene supports the instruction. |
| New reviewed examples | No new two-car, ambiguity or spoken examples have yet been completed for this addition. | Supply the missing supervision and realistic evaluation cases. |

Only one of the 508 concise test commands contains the word "between", and it concerns distance to another vehicle. This release is not a ready benchmark for the exact two-black-car interaction. Additional labels are required rather than fabricated from box geometry.

RegSeg masks supervise destination location. They do not automatically supervise every anchor identity, ambiguity status or conversational reply. Use each source only for outputs it actually labels. Missing labels must be excluded from their training loss or handled using a separate task format, not converted into false "no anchor" or "unambiguous" gold labels.

Derive any point training target from an interior mask pixel. A naive centroid can lie outside a non-convex mask. If box targets are added, document the exact mask-to-box rule and preserve the underlying mask for evaluation.

The concise test comes from original validation images with rewritten commands. Create development groups from the official training data; keep the original validation images and their concise-test derivatives out of gradient training, checkpoint selection and threshold calibration. Report overlap between the two public evaluation views rather than pooling them as independent samples. Keep all commands, paraphrases, dialogue turns and adjacent frames from a source scene in the same split. Establish scene mapping before claiming scene-independent evaluation.

The [Talk2Car data licence](https://raw.githubusercontent.com/talk2car/Talk2Car/master/LICENSE) covers the dataset and extensions under CC BY-NC-SA 4.0. Distinguish it from repository software licensing and preserve attribution. A third-party mirror's label does not replace the source data terms.

## Model and training plan

Use one image-and-language instruction model for learned grounding. [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) is a first candidate because its official card explicitly supports spatial perception and grounding, and the [official training repository](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune) supports LoRA adaptation. This is a candidate choice requiring a short feasibility test, not a claim that it is the best or newest model for every task.

The previously tested Qwen3 text embedding checkpoint is a different component. Preserve it as a retrieval baseline; it cannot directly inspect the pixels needed for colour and road-location grounding.

The learned model should consume the current image, the passenger utterance and relevant observed landmarks/history. It should learn attributes, spatial relations and destinations. Structured boxes or track features can help connect the output to the existing vision pipeline, but they must be distinguished from ground truth. Missing detections remain measurable failures. Evaluate whether supplying detector evidence helps instead of assuming that it does.

Fine-tune a single adapter rather than starting separate new ASR and answer-generation training projects. Begin with a documented subset of the appropriate public training commands and add reviewed examples for two anchors, ambiguity, missing references and conversational corrections. A training budget of roughly 1,000-2,000 public examples plus 150-250 new instruction cases is a provisional scheduling target; actual counts depend on image suitability, label review and the first measured training run. It is not an assessment requirement or a claim that these data are already prepared.

Allocate the new cases before training: approximately 100-140 for training, 25-40 for development and 25-70 for held-out evaluation. Each partition must include clear two-anchor, ambiguous and missing-reference cases on different scene groups. Have two people independently review ambiguous cases and all held-out targets, then reconcile disagreements. Multiple ratings of one case do not create extra independent examples. Keep final evaluation labels inaccessible to model development.

Fine-tuning must perform real parameter updates and save/reload the adapter. Use distinct supervision formats or masked losses for partially labelled sources. Save the base revision, image preprocessing, coordinate convention, grouping manifest, training settings and logs. Select checkpoints using the development partition. Report non-improvement honestly.

Template-based clarification wording is sufficient this week if it describes real candidates and resolves the missing distinction. Do not spend the deadline creating an elaborate agent architecture or a second clarification-training corpus at the expense of grounding. Evaluate the model's interpretation on unseen scenes and contrasts such as left/right, front/behind, colour changes, negation and alternative anchor pairs; JSON validity alone is not language understanding.

Before committing to the training schedule, run a short gradient-update pilot on the actual available GPU, save and reload the adapter, and generate spatial outputs through the intended interface. Measure time and memory. Use a suitable GPU or adjust the input budget if necessary, checking that the relevant landmarks remain visible. If an adequate GPU is unavailable within the schedule, use an explicitly identified smaller instruction VLM under the same contract. Its pretrained version becomes the matching baseline; do not attribute a size change to fine-tuning. No smaller model's adequacy should be assumed or its substitution hidden.

Task 1 artifact recovery can proceed alongside this pilot. Begin independent language-grounding development on the public labelled images with a clearly identified pretrained visual baseline, or an explicitly labelled oracle-anchor development condition. Ground-truth anchors are not permitted as the final predicted-input result.

## Preserve the existing application

```text
Shared recording reader, frame IDs, detector/tracker, evidence and provenance
  |
  +-- Existing route: footage question -> retrieval -> cited QA answer
  |
  +-- New route: voice/text instruction -> grounded intent and location
                  -> clarify when needed -> confirmed proposal
```

Keep the existing QA behaviour and its known failures visible. The new command route uses its own schema and evaluation; an imperative drop-off instruction is not forced into a question-answer record.

The existing track memory can support references to earlier observations when those observations really exist. History ablations are optional this week. A new vector database, legal-rule RAG system and trajectory planner are outside the critical path. The core trained command interpreter can run independently of the hosted QA provider.

## Voice for the drop off interaction

Use a fixed speech recognizer to turn microphone input into text, preserving the original audio, transcript, timing and correction. Pass that transcript into the same command interpreter used for typed input. A spoken clarification response can update the same dialogue state. Speech synthesis is a presentation feature and can use a fixed engine.

The reviewed public command datasets provide text instructions, not a ready matched speech-and-destination corpus. Plan 30-60 consented recordings from 3-5 speakers for frozen evaluation instructions. Mark whether these are read or spontaneous speech and whether noise is real or added. Compare the human transcript and ASR transcript on exactly the same scene/command. Report critical errors in colour, direction, count and negation, as well as word error rate and downstream grounding success. Synthetic speech is development augmentation, not a replacement for the real-voice evaluation slice.

## What the submitted tasks actually contain

This source/artifact audit inspected main `1885051`, Task 1 `6ea0e12` and Task 4 `057072d`. It did not reload external weights or reproduce the reported fine-tuned test scores.

| Task | Verified contribution | Remaining dependency |
| --- | --- | --- |
| 1 | Open [PR21](https://github.com/A1Prep/TrackRAG-Gulf/pull/21) contains the model adapters, configuration, 30-epoch training record and checkpoint hashes. Committed logs substantiate the reported epoch-4 selection. | Obtain the external checkpoint and prediction/metric bundles, verify hashes and independently reload. On five official test recordings the PR reports road-user AP .303 to .355 and IDF1 .400 to .443, but also more ID switches; configurations include different selected tracker settings. These remain reported combined-system results until reproduced. |
| 2 | The merged EMT reader and original-coordinate frame/label interfaces are reusable. | Prepare the new instruction/location data; existing EMT boxes do not contain it. |
| 3 | The merged answerer validates evidence, performs privacy processing and records requests. | Human reference review remains incomplete. The latest five normal-path requests yielded no usable answers: two timeouts, one provider 503, one disconnect and one DNS failure. Earlier prototype live answers are a different condition. |
| 4 | The lexical baseline is merged. Open [PR20](https://github.com/A1Prep/TrackRAG-Gulf/pull/20) adds candidate histories, scenes and relevance-review/export tools. | Its 230 candidate judgements are pending, with zero reviewed exports. Its own Qwen inference, fine-tuning and reload milestones are unfinished. Those tooling records are not a completed learned NLP model. |
| 5 | Reviewer, evidence export, scoring and real pretrained experiments exist. The separate [benchmark PR19](https://github.com/A1Prep/TrackRAG-Gulf/pull/19) preserves the unfinished embedding comparison. | Integrate validated final artifacts, add the new command route and evaluate both routes. The new learning and label work requires team contributions, not only Task 5 integration. |

Two integrity fixes are needed in Task 1's saved-artifact route before relying on final measurements: require the saved-track hash rather than accepting an omitted hash, and verify that detection frame records are complete and unique against declared provenance. These are source-review findings; no benchmark result was reclassified or recomputed here.

## Evaluation that answers the research question

Freeze model inputs, coordinate conventions, groups and metric definitions before inspecting final results. On the interaction subset with the necessary human labels, compare four conditions on the same held-out cases: pretrained/direct, pretrained/clarification, tuned/direct and tuned/clarification. Use the same base checkpoint, visual inputs and final confirmation opportunity. This separates learning gains from dialogue gains. Public mask-only examples can support a separate pretrained/tuned destination comparison; they do not automatically supply dialogue gold.

Compare targeted clarification with direct prediction under a stated turn budget. Record how often the system proposes a location, not just how rarely it makes a wrong proposal. Report initial grounding separately from completion after passenger intervention; a corrected location is not an initially correct model prediction.

| Outcome | Measurement |
| --- | --- |
| Correct referred vehicle or pair | Anchor identity/box agreement; both anchors must be correct for a pair. |
| Correct destination point | Predicted point inside a human-labelled acceptable mask. Report outside-mask distance separately if used. |
| Coarse region prediction, if implemented | Box overlap against the stated mask-derived target, reported separately from point accuracy and full-mask segmentation. |
| Ambiguity handling | Clarification precision/recall and the incorrect-proposal rate on ambiguous/missing-reference cases. |
| Dialogue usefulness | Correct intended destination after clarification, completion rate and turns required. Compare at matched proposal coverage or report risk versus coverage. |
| Voice robustness | Transcript errors and the change in grounding/dialogue results from human text to ASR text. |
| Practical performance | Inference latency, failed requests and invalid spatial outputs. |
| Existing functionality | QA regression results with provider errors separated from factual answer failures. |

Use independently recorded passenger intent for dialogue evaluation. A simulated responder may know that intent as the user would, but must not derive its answers from the model's proposed location or expose hidden labels to the model. Label AI-assisted reviews distinctly from human review.

The supplied 902 outline gives 15% to problem definition, 20% to architecture, 20% to implementation/performance and 15% to evaluation, with further credit for practical/ethical issues, reporting and reproducibility. The new task needs evidence in all of these areas; voice or a model name alone does not establish coverage. Actual local NLP fine-tuning remains a project milestone based on the reported lecturer preference. No architecture guarantees a mark.

## Seven day delivery sequence

| Day | Required result |
| --- | --- |
| 1 | Freeze the point/optional-box contract, ambiguity rubric and grouped data manifest. Recover Task 1 artifacts in parallel with an independent grounding pilot. Inspect suitable training scenes. Complete a small pretrained inference and save/reload training pilot on the real GPU; invoke the explicit checkpoint/input fallback above if needed. |
| 2 | Prepare the public training subset and reviewed new cases, including reserved held-out interaction targets. Freeze development/evaluation groups. Obtain a small development baseline score and record exploratory rider/driver feedback. |
| 3 | Run the chosen grounding adaptation, save configuration and logs, reload its parameters and verify the interface. Annotation and integration can continue in parallel. |
| 4 | Connect the command route, targeted clarification and proposal confirmation. Tie confirmation to the reference frame and proposal identity. Keep the existing QA route. |
| 5 | Add voice to this route and record the real speech slice. Freeze final inference settings using development data. |
| 6 | Run held-out pretrained/tuned comparisons and the dialogue/voice evaluation. Check failures, stale references and the existing QA regression slice. |
| 7 | Finish both reports, figures, contribution evidence, reproducible instructions and the recorded-scene demo. Reserve time for integration repairs. |

Original task owners still have clear contributions: Task 1 supplies and validates the trained visual artifacts; Task 2 prepares and groups the new data; Task 3 coordinates reviewed instructions/dialogue targets and speech examples; Task 4 owns the actual learned grounding model and training evidence; Task 5 connects the routes and runs independent evaluation. Agree the field names and first-day pilot immediately so these contributions can proceed in parallel.

## Evidence limits and access

This is targeted research and a source/artifact audit, not a completed implementation or an exhaustive systematic review. Google Scholar search pages were inaccessible through the available search tool; papers were verified through original publisher/arXiv records and official project repositories. Some OpenReview pages also presented a browser verification barrier, so their indexed primary conference PDFs were used. Published work is distinguished above from preprints.

The first practical dependency is the shared Task 1 checkpoint/prediction location. The largest schedule uncertainty is the first real grounding-training pilot and the availability of valid new labels. Keep those conditions visible rather than calling the plan fully verified before they are resolved.
