# Assessment 3: training and completion evidence

Checked against the supplied MAAI 901 and 902 subject outlines on 16 November 2026, plus Omar's report that the lecturers prefer fine-tuning and locally run models. The current Gemini trials are development work. They do not complete the assessed model-training work.

Updated on 20 November 2026 for the approved passenger drop-off addition and Tuesday, 24 November submission. [Part 2: final project completion](final-completion-part2.md) controls current assignments and the four-day delivery plan. The adopted local NLP training objective is now language-conditioned destination grounding, replacing the earlier separate Qwen-reranker objective. Both application routes remain in scope.

## Written requirements and lecturer feedback

MAAI 901 Assessment 3 explicitly includes model training, evaluation, preprocessing and augmentation. It measures LO3, which concerns fine-tuning/customizing pretrained models. See the 901 subject outline, pages 2 and 7–8.

MAAI 902 Assessment 3 requires an implemented end-to-end NLP application. It lists transformers, fine-tuned LLMs, RAG and embeddings as examples, requires training/fine-tuning details in the report, and measures LO4. See the 902 subject outline, pages 2 and 8–9.

Neither supplied outline explicitly says every model or the final answer generator must run locally. They do not prescribe training every component. In response to the reported lecturer feedback, however, **actual CV fine-tuning and a substantive self-hosted, fine-tuned NLP component are project completion milestones**. Loading a pretrained model, changing prompts, adjusting retrieval rules or calling Gemini does not demonstrate weight training.

This interpretation does not guarantee a mark. The adopted trained NLP contribution learns to interpret passenger language in a road scene, predicts an intended destination and supports clarification. It must actually control the new application's interpretation. The existing QA route can retain its documented lexical retrieval and configured Gemini generator. The written outlines do not establish that this separate answer generator must itself run locally; calling Gemini alone does not complete the local training milestone.

## Assessed model roles

| Component | Owner | Completion evidence |
| --- | --- | --- |
| RF-DETR Large detector, adapted to EMT | Previous Task 1 owner | Actual fine-tuning, explicit class mapping, saved trained weights/adapter, training configuration and before/after detection evaluation |
| BoT-SORT tracking and genuine scene inputs | Previous Task 1 owner | Real predicted detections linked over time, identity metrics, failure analysis and exact-frame handoff; tracker fine-tuning is not automatically required |
| EMT/grounding data and grouped splits | Previous Task 2 owner | Original frame IDs/coordinates, reviewed destination annotations and separate training/development/evaluation groups |
| Reviewed questions, cited QA and speech input | Previous Task 3 owner | Human-reviewed references, actual answer-provider outcomes, citations, abstention/error handling and fixed-ASR evaluation |
| Application, clarification and confirmation | Previous Task 4 owner | Both routes connected, current-frame proposal binding, regression checks and a reproducible release |
| Fine-tuned language-grounding model | Previous Task 5 owner | A local/self-hosted model that actually learns passenger-instruction-to-destination interpretation, with saved/reloaded parameters used by the application |
| Independent evaluation | Previous Task 1 owner, with integration support from Task 4 | Fixed pretrained/trained conditions, reserved references, real run outputs, coverage and retained failures |

Here, self-hosted means the team can load and run its own model weights in the chosen GPU environment. Training may use a cloud GPU; a closed hosted inference API is a different component. Do not imply that training must occur on a particular laptop.

Gemini can remain the answer generator for the original QA route. It is not evidence that the team fine-tuned a model. The trained grounding contribution must be central to the drop-off route: it must interpret new language and image inputs rather than decorate a decision already made by a hosted model or a fixed lookup.

## Required training milestones

### Task 1: visual model

1. Establish a pretrained detector baseline on EMT with documented class coverage and identical evaluation rules.
2. Fine-tune the chosen detector on training recordings, with justified preprocessing and augmentation. Use development recordings for model selection.
3. Save the training configuration, base revision, seed, logs/learning curves and trained checkpoint location. Keep large weights outside Git.
4. Reload those weights and run `predict_tracks(recording)` on real images with annotation files unavailable to prediction.
5. Compare pretrained and tuned detection results, plus downstream tracking quality and representative failures. Use appropriate detection and identity metrics.

### Previous Task 5 owner: NLP grounding model

1. Build validated image/instruction/destination examples from appropriate public data and reviewed local cases. Include spatial relations, two-landmark instructions, ambiguity, missing references and meaning-changing paraphrases. Do not invent labels absent from a source.
2. Keep source-scene groups, neighbouring frames and related utterances together in a partition. Separate training, model selection and final evaluation. Resolve human annotation disagreements before scoring.
3. Complete a real gradient-update/save/reload pilot on the actual GPU, then adapt the selected image-and-language model. Parameter-efficient training is acceptable when justified and documented; it must update and save real learned parameters.
4. Save the base revision, input format, trainable modules, image processing, coordinate convention, dataset version/splits, settings, logs and adapter/checkpoint location.
5. Reload the trained model locally/self-hosted through the agreed grounding interface and connect it to the application. Do not substitute Gemini, static destination coordinates or question-ID lookups.
6. Compare pretrained/direct, pretrained/clarification, trained/direct and trained/clarification on matching reviewed interaction cases with a fixed input and turn budget. Score destination points and coarse regions separately; report proposal coverage, ambiguity outcomes and failures.
7. Compare human versus ASR transcripts on the same voice cases, and retain a regression evaluation of the original QA route. Report non-improvement honestly.

The exact training method, number of examples and evaluation metrics need technical justification. The outline does not specify a universal dataset size, require every component to be trained, or guarantee improvement from fine-tuning. Task 4's earlier candidate/relevance-review preparation remains useful QA work but is not a completed trained model. A separate reranker-training project is not part of the four-day completion plan.

## Rubric coverage

| MAAI 901 criterion | Weight | Project evidence to supply |
| --- | ---: | --- |
| Problem definition and relevance | 10% | A specific footage-review problem, intended users and useful outcomes |
| Model architecture and implementation | 25% | Suitable detector/tracker choices, implemented training and justified decisions |
| Data processing and pipeline design | 15% | Class mapping, preprocessing, augmentation and integrated predictions |
| Evaluation and results interpretation | 20% | Held-out detection/tracking results, visual examples, comparisons and limitations |
| Team collaboration and contribution | 10% | Attributed work, commits/reviews and integration contributions |
| Report clarity and documentation | 10% | The required technical report, references and reproducible instructions |
| Presentation/demo | 10% | A clear final walkthrough; the rubric labels this optional while the overview also lists a walkthrough, so the team plans to provide it |

| MAAI 902 criterion | Weight | Project evidence to supply |
| --- | ---: | --- |
| Problem definition and relevance | 15% | Why natural-language retrieval/reasoning is useful for the intended reviewer |
| Solution architecture and design | 20% | Existing evidence-retrieval QA plus a meaningful trained language-grounding route, clear model roles and clarification/confirmation |
| Model implementation and performance | 20% | A working trained NLP component with saved/reloaded weights and measured results |
| Evaluation and interpretation | 15% | Pretrained/trained grounding and dialogue comparisons, matched text/voice cases, QA regressions and supported error analysis |
| Ethical and practical considerations | 10% | Privacy handling, data/annotation limits, bias, hallucination and provider/compute constraints |
| Report quality and presentation | 10% | The required technical report and clear demonstration |
| Code quality and reproducibility | 10% | Modular code, pinned configuration, setup/run instructions and a submission README |

The README was deliberately cleared during initial setup. It must be completed for submission; this note does not change that file now.

## Current verified status

The integrated code contains the EMT reader, Task 3 question loader and cited answerer, reviewer, scoring, explicit development adapters, question banks and small real-data trials. Automated checks cover application boundaries and reproducibility, including the Task 3 SDK through local mocked HTTP. The 36 Gemini development cases provide error analysis using human-labeled tracks, with known failures and qualifications retained; the newly integrated normal Task 3 answerer still requires its own live quality check. The [Task 3 handoff](task3-integration.md) separates integration checks, source-annotation agreement and pending human review. The [retrieval verification command](retrieval-verification.md) adds offline contract and future-input checks; those are not answer-accuracy scores. Zero-score scene-context fallbacks remain separate from semantic retrieval performance.

On 19 November 2026, the supplied Task 1 trained checkpoint was verified against its pinned hash, safely loaded and exercised through the real vision interface on two 150-frame image-only recordings. The supplied full-test tracking metrics were independently reproduced on five recordings. [The artifact verification](task1-artifact-verification.md) distinguishes these completed checks from archived full-test detection AP, whose raw detection files were not included in the handoff.

No completed Qwen fine-tuning run has been verified here. Actual local grounding training, saved/reloaded parameters and final evaluation remain incomplete. The current development question bank is not final held-out assessment evidence. The new voice layer, application route and reserved grounding labels are also unfinished; the [Part 2 checklist](final-completion-part2.md#shared-project-acceptance-checklist) defines the remaining technical release gates.

On 20 November, Task 5's [model/training handoff](dropoff-model-handoff.md) added the actual training/inference code, strict partial-label and source-scene guards, a verified ten-example public training bootstrap and a cloud pilot notebook. The real pinned Qwen processor/collator passed ten-case CPU checks; the full software suite passed 590 tests and 285 subtests, with 12 optional/platform skips. These are implementation and input-pipeline checks, not 8B parameter updates or model accuracy. The laptop preflight correctly remains blocked, and the [preparation record](../experiments/dropoff/task5_preparation_20261120.json) keeps actual neural training/inference explicitly false.
