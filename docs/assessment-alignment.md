# Assessment 3: training and completion evidence

Checked against the supplied MAAI 901 and 902 subject outlines on 16 November 2026, plus Omar's report that the lecturers prefer fine-tuning and locally run models. The current Gemini trials are development work. They do not complete the assessed model-training work.

## Passenger instruction addition agreed on 19 November 2026

Omar reports that the lecturers found the current NLP contribution too basic and approved an additional passenger drop-off interaction: identify a proposed stopping location and ask for confirmation, with voice for this interaction and no vehicle control. The existing footage-question workflow remains part of the application.

The [passenger research and delivery plan](passenger-dropoff-research.md) defines a grounded-language task, its published precedents, available location labels, reviewed ambiguity cases and a seven-day evaluation sequence. It proposes prioritising actual local image-and-language grounding adaptation for the new interaction while preserving the existing retrieval baselines. The older relevance-reranking milestones below describe the original training route; this addition does not imply that two separate NLP models must be fine-tuned. The assessed contribution must still be central, trained, reloaded and evaluated rather than an unused adapter or speech-only wrapper.

As of the 19 November source audit, Task 1's open PR21 now includes substantive training logs and reported fine-tuned results, with weights and prediction bundles shared externally. Independent artifact reload remains pending. Task 4's open PR20 provides candidate and relevance-data tooling but explicitly leaves reviewed human labels, Qwen inference and fine-tuning unfinished. See the linked plan for the current contribution audit. The historical status at the end of this file records the earlier integrated version.

## Written requirements and lecturer feedback

MAAI 901 Assessment 3 explicitly includes model training, evaluation, preprocessing and augmentation. It measures LO3, which concerns fine-tuning/customizing pretrained models. See the 901 subject outline, pages 2 and 7–8.

MAAI 902 Assessment 3 requires an implemented end-to-end NLP application. It lists transformers, fine-tuned LLMs, RAG and embeddings as examples, requires training/fine-tuning details in the report, and measures LO4. See the 902 subject outline, pages 2 and 8–9.

Neither supplied outline explicitly says every model or the final answer generator must run locally. They do not prescribe training every component. In response to the reported lecturer feedback, however, **actual CV fine-tuning and a substantive self-hosted, fine-tuned NLP component are project completion milestones**. Loading a pretrained model, changing prompts, adjusting retrieval rules or calling Gemini does not demonstrate weight training.

This interpretation does not guarantee a mark. A trained reranker is a defensible NLP contribution when it learns the question-to-evidence task and controls the evidence used by the application. The written outlines do not establish whether the lecturer's verbal preference specifically concerns the answer-generating model; this remains a grading uncertainty, not permission to present a Gemini-only prototype as the completed project.

## Assessed model roles

| Component | Owner | Completion evidence |
| --- | --- | --- |
| RF-DETR Large detector, adapted to EMT | Task 1 | Actual fine-tuning, explicit class mapping, saved trained weights/adapter, training configuration and before/after detection evaluation |
| BoT-SORT tracking | Task 1 | Real predicted detections linked over time, identity metrics and failure analysis; tracker fine-tuning is not automatically required |
| EMT data and recording splits | Task 2 | Original frame IDs/coordinates, verified labels, recording-level train/development/test separation |
| Reviewed questions, answer key and cited answer generation | Task 3 | Human-reviewed labels, answer-provider adapter, citations and abstention/error handling; Gemini is the planned configured generator |
| Fine-tuned Qwen relevance/reranking model | Task 4 | A local/self-hosted trained checkpoint that learns which vehicle/time evidence supports a natural-language question and actually determines the evidence passed to Task 3 |
| Integration and independent evaluation | Task 5 | The same reviewer/evaluator runs the baselines and trained components, retains errors and provides reproducible comparison results |

Here, self-hosted means the team can load and run its own model weights in the chosen GPU environment. Training may use a cloud GPU; a closed hosted inference API is a different component. Do not imply that training must occur on a particular laptop.

Gemini can remain the answer generator in this hybrid plan. It is not the evidence that the team fine-tuned a model. The Qwen contribution must be central: avoid having Gemini first make all the decisive evidence choices and then placing a trained but unused reranker afterward.

## Required training milestones

### Task 1: visual model

1. Establish a pretrained detector baseline on EMT with documented class coverage and identical evaluation rules.
2. Fine-tune the chosen detector on training recordings, with justified preprocessing and augmentation. Use development recordings for model selection.
3. Save the training configuration, base revision, seed, logs/learning curves and trained checkpoint location. Keep large weights outside Git.
4. Reload those weights and run `predict_tracks(recording)` on real images with annotation files unavailable to prediction.
5. Compare pretrained and tuned detection results, plus downstream tracking quality and representative failures. Use appropriate detection and identity metrics.

### Task 4: NLP model

1. Build reviewed training examples pairing a natural-language question with relevant and irrelevant vehicle/time evidence. Include paraphrases, similar-looking vehicles and wrong-time evidence as difficult negatives. The language/visual inputs must contain the information needed for the question.
2. Keep training, model-selection and final evaluation queries/recordings separate. Resolve ambiguous labels rather than treating model-generated answers as unquestioned truth.
3. Run actual parameter updates on the selected Qwen relevance/reranking model. Parameter-efficient training is acceptable as a project method if justified and documented; it must save trained parameters.
4. Save the base model/revision, input format, trainable parameters, dataset version/splits, settings, logs and adapter/checkpoint location.
5. Reload and run the trained model locally/self-hosted through `rank_evidence`. Do not silently substitute Gemini or a static rule-based result for this assessed run.
6. Compare simple retrieval, frozen Qwen reranking and fine-tuned Qwen reranking on identical held-out candidate sets and reviewed questions. Use relevant ranking/vehicle-and-time metrics and analyze failures.
7. Keep the answer generator and prompt fixed for a downstream comparison. Check whether changed evidence improves answer support, citations and appropriate abstention. Report a lack of improvement honestly.

The exact training method, number of examples and evaluation metrics need technical justification. The outline does not specify a universal dataset size, require every component to be trained, or guarantee improvement from fine-tuning.

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
| Solution architecture and design | 20% | Track-memory RAG, embeddings/retrieval, meaningful local Qwen integration and generator roles |
| Model implementation and performance | 20% | A working trained NLP component with saved/reloaded weights and measured results |
| Evaluation and interpretation | 15% | Frozen/tuned comparisons, downstream ablations and supported error analysis |
| Ethical and practical considerations | 10% | Privacy handling, data/annotation limits, bias, hallucination and provider/compute constraints |
| Report quality and presentation | 10% | The required technical report and clear demonstration |
| Code quality and reproducibility | 10% | Modular code, pinned configuration, setup/run instructions and a submission README |

The README was deliberately cleared during initial setup. It must be completed for submission; this note does not change that file now.

## Current verified status

The integrated code contains the EMT reader, Task 3 question loader and cited answerer, reviewer, scoring, explicit development adapters, question banks and small real-data trials. Automated checks cover application boundaries and reproducibility, including the Task 3 SDK through local mocked HTTP. The 36 Gemini development cases provide error analysis using human-labeled tracks, with known failures and qualifications retained; the newly integrated normal Task 3 answerer still requires its own live quality check. The [Task 3 handoff](task3-integration.md) separates integration checks, source-annotation agreement and pending human review. The [retrieval verification command](retrieval-verification.md) adds offline contract and future-input checks; those are not answer-accuracy scores. Zero-score scene-context fallbacks remain separate from semantic retrieval performance.

No CV or Qwen fine-tuning run has been integrated and verified here. Do not claim a teammate has or has not trained privately without checking their artifacts. The training milestones above remain incomplete until the actual configuration, trained weights, logs and evaluation are supplied and exercised through the project interfaces. The current development bank is not final held-out assessment evidence.
