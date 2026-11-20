# Task 4 milestone 4: frozen pretrained Qwen text reranking

[retrieval_qwen.py](../trackrag/retrieval_qwen.py) adds an explicitly selected
pretrained reranker. The original lexical ranker remains the default. No human
labels are needed just to run pretrained inference; labels are still needed for
measured relevance evaluation and future supervised fine-tuning.

## Model and current scope

The first backend is **Qwen/Qwen3-Reranker-0.6B**, pinned to model commit
`e61197ed45024b0ed8a2d74b80b4d909f1255473`. It is a text model. The local PC has
31.6 GiB RAM and Intel integrated graphics, with no CUDA GPU; the verified setup
therefore uses CPU float32 inference with four CPU threads.

This model sees the question text/cutoff and each candidate's observed class,
original frame bounds/boxes, normalized displacement/range, observation gaps and image-frame
inventory. It does **not** receive image pixels, filenames, color descriptions,
OCR, reviewer judgments, gold answers, question IDs or recording IDs as prompt
features. The scene document explicitly says visual information is unavailable.

Consequently this backend cannot determine a bus's color, read an exit sign or
establish an overpass from an image path. A vision-language backend or independently
validated visual representation is separate future work, not an implemented
capability of this text adapter. Its two real development questions remain useful
integration controls even though appearance/scene content cannot be established
from their supplied text metadata.

The compact text representation groups only consecutive observed frame IDs into
inclusive ranges and rounds floating features to six decimal places. Missing
ranges remain explicit; no observations are invented. Original candidate metadata
and output coordinates retain their original precision. This input policy must
remain fixed in controlled frozen-versus-trained comparisons.

Weights are frozen in evaluation mode with gradients disabled. There is no
parameter training, answer generation, Gemini call or rule-based fallback. Importing
the adapter and building an index do not download or initialize the model. The
first nonempty scoring call loads weights and tokenizer from the configured cache.

## Installation

From the repository root, using the existing isolated Python environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-qwen.txt
```

[requirements-qwen.txt](../requirements-qwen.txt) keeps optional PyTorch/Transformers
dependencies separate from normal reviewer requirements. The verified versions are
PyTorch 2.15.0 CPU and Transformers 4.57.6 on Python 3.14.7/Windows. GPU runtime setup
needs an appropriate CUDA-enabled PyTorch build; it was not exercised here.

The first online model load downloads approximately 1.2 GB of pretrained weights,
plus tokenizer/config files, from the official repository into `models/huggingface`.
This is a download; footage and questions are not uploaded for hosted inference.
Weights and caches remain ignored by Git. After downloading, set `local_files_only`
to true to require cached files and prohibit model-file network requests.

## How ranking works

1. Build the unchanged observation index with an explicitly declared run source.
2. Generate the complete milestone-2 candidate pool from observations at or before
   the query cutoff. Include exact scenes, histories and local windows.
3. Validate candidate membership, bounds and metadata, then format one versioned
   text document per candidate (`trackrag.qwen-text.v1`). No lexical shortlist or
   arbitrary candidate-count cap removes evidence before scoring.
4. Apply a fixed road-evidence instruction and the model's documented yes/no
   token-logit method to every query/document pair.
5. Sort by descending score, then start/end frame, tracked-before-scene, track ID
   and candidate ID. Project at most three selected candidates into the existing
   seven-field `RankedInterval` contract.

Scores are the softmax yes probability **conditioned on the two yes/no logits**.
They are not calibrated answer correctness, comparable to the old lexical scores,
or proof that the selected evidence is sufficient. Every scored candidate remains
eligible: no relevance or abstention threshold has been calibrated. Even low-scoring
candidates can therefore appear in the top three. Human review and development-set
evaluation are needed before introducing a justified threshold.

Inputs longer than the configured token limit fail explicitly. There is no silent
truncation that discards the question or later evidence. Model/scorer failures
propagate rather than substituting successful-looking lexical results. Empty pools
and `k=0` return no intervals without invoking the model.

Whole histories can have less precise temporal bounds than short windows. A broad
candidate's high score must not automatically count as a correct event interval.
The model can only reorder existing candidates; it cannot invent a missing interval
or supply information that the candidate representation omitted.

## Direct Python use

```python
from trackrag.emt import open_recording, load_annotated_tracks
from trackrag.questions import load_questions
from trackrag.retrieval_qwen import QwenConfig, build_index, rank_evidence

recording = open_recording("data/emt-task4-baseline", "video_054604")
tracks = load_annotated_tracks(recording)
question = load_questions("experiments/temporal-development-questions.jsonl")[0]
index = build_index(recording, tracks, source="annotated",
                    config=QwenConfig(local_files_only=True))
intervals = rank_evidence(question, index, k=3)
receipt = index.last_receipt
```

Predicted observations use the same API with `source="predicted"`, preserving their
independent IDs. Source is a fixed run condition rather than something inferred
from future-only observations. An empty predicted history does not authenticate a
detector run or prove visual absence.

Receipts retain candidate IDs, all scores, formatted documents, model revision,
token counts, configuration, instruction/template, metadata/input fingerprints and
elapsed time. Programmatic callers can persist `index.last_receipt`; a failed new
question cannot reuse a previous success receipt. The adapter has no global score
cache or shared history cache that could leak across questions.

## Command that saves rankings and receipts

Create a public-question JSONL file containing only questions for the chosen
recording. For a short pilot, select one unchanged question from the existing bank.
Then choose a new output directory:

```powershell
.\.venv\Scripts\python.exe -B -m trackrag.retrieval_qwen --data-root data/emt-task4-baseline --recording video_054604 --questions <ONE_RECORDING_QUESTIONS.jsonl> --source annotated --out runs/qwen-text-new
```

The command writes `rankings.jsonl` and per-question `receipt-*.json`. These are
retrieval outputs, not Task 5 answer results. If inference fails after staging starts,
the failed run remains visible, with `failure.json` when a model-call receipt exists;
do not treat it as a completed scored cohort. The command does not overwrite runs.

For saved detector predictions, use `--source predicted --predicted-tracks <FILE>`.
Annotated mode rejects prediction files. Predictions must match the shared observation
fields and selected recording/source. This command does not run a detector.

`--config <JSON>` accepts `QwenConfig` fields. For example, after the initial download:

```json
{
  "cache_dir": "models/huggingface",
  "device": "cpu",
  "batch_size": 2,
  "max_tokens": 4096,
  "cpu_threads": 4,
  "local_files_only": true
}
```

## Select it in the existing reviewer

Task 5's adapter interface remains unchanged. Its two-argument builder call obtains
the explicit source and optional config file from environment variables:

```powershell
$env:TRACKRAG_QWEN_SOURCE = 'annotated'
# Optional: $env:TRACKRAG_QWEN_CONFIG = 'path/to/qwen-config.json'
.\.venv\Scripts\python.exe -B -m trackrag check-adapters --source annotated --answerer none --retrieval-adapter trackrag.retrieval_qwen
.\.venv\Scripts\python.exe -B -m trackrag review --data-root data/emt-task4-baseline --recording video_054604 --questions <ONE_RECORDING_QUESTIONS.jsonl> --source annotated --answerer none --retrieval-adapter trackrag.retrieval_qwen --out runs/qwen-review-new
```

Set both source selections to `predicted` for real prediction-adapter runs. The
readiness checker only checks imports/signatures; it does not load weights or prove
model readiness. The normal reviewer saves selected evidence; use the direct Qwen
command or persist the index receipt when full model-input provenance is required.

## Evaluation and next steps

The human relevance worksheet remains pending. Running Qwen requires no relevance
labels, but comparing its ordering to gold evidence does. Do not report top-1/top-3
accuracy or training improvement from an unlabeled inference check.

Compare frozen and tuned scoring on identical candidate sets, input versions and
image policies. The original lexical ranker searches a narrower pool; compare a
matched-pool baseline separately or explicitly attribute candidate-coverage effects.
Keep Task 3's generator/prompt fixed for downstream answer comparisons.

The [fixed-pool ranking comparison](task4-ranking-comparison.md) now provides that
separate lexical-versus-pretrained experiment. It records both complete score
vectors and rankings over one saved pool, retaining zero scores for both variants.
Relevance accuracy remains unavailable until human labels are completed.

Next work is to finish reviewed labels and establish a relevance baseline, choose
the required text/visual scope and GPU environment, then fine-tune and reload actual
learned parameters. This first text backend is not a completed multimodal system.

The [LoRA training/save/reload pipeline](task4-qwen-training.md) now supplies the
machinery for that later training. Its first completed run updates the downloaded
model using explicitly synthetic examples, with a separately identified adapter.
Reviewed real-data fine-tuning and relevance evaluation remain pending.

## Validation performed

The complete offline suite passed **354 tests, with 362 subtests**, including 20
Qwen-adapter tests. Contract tests inject scorers; tensor tests use constructed
logits. They cover complete-pool scoring, deterministic ties, finite score/count
validation, source separation, empty inputs, query validation, future-input
invariance, compact gap preservation, lazy loading and failure propagation. These
tests are distinct from the actual pretrained-weight checks below.

The downloaded model first passed a synthetic bus-versus-car description check.
Its yes/no scores were approximately 0.9997 for the bus description and 0.0279 for
the car description. This is a small smoke control, not a relevance accuracy result.

Then the actual frozen weights scored complete opening-frame pools from the two
original EMT recordings, using the real reader and public questions:

| Question | Recording | Candidates scored | CPU scoring time |
| --- | --- | ---: | ---: |
| temporal-11 | video_054604 | 15 | 114.19 seconds, including initial model loading |
| second-01 | video_054907 | 20 | 122.20 seconds, reusing the model |

For the first question, rebuilding the index from prefix-only frames/observations
left all input documents, scores and selected intervals exactly unchanged. Both
queries were at frame 1; this does not replace testing more challenging crossing
windows and later cutoffs, which the injected-scorer tests cover separately.

Captured Qwen intervals were passed through the ordinary evidence exporter and
artifact auditor on both recordings, with six audited files each. Answers were
`not_requested`. The export replay did not perform a second Qwen call. Adapter
signature checking, Qwen CLI help and the actual `k=0` CLI output path also passed;
`k=0` was verified not to call a model.

The larger temporal-09/second-07 CPU pilot was stopped before completion because
of runtime. No completed result is claimed for that pilot. CPU performance here
is sufficient for small smoke runs but expensive for larger candidate pools;
use an appropriate GPU for broader experiments and visual reranking.

The [recorded pretrained summary](../experiments/task4-qwen-pretrained-summary.json)
contains source hashes, pinned revision, input/score hashes, timings, raw intervals
and local receipt locations. These were two previously used development questions
without relevance labels. The second question mentions color, which the text model
was not given. Scores—even high ones—do not establish that the evidence supports
that visual claim. The listed lexical results use a narrower pool and are not a
controlled same-pool accuracy comparison. No model parameters were updated.

Sources: [official model card and scoring example](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B),
[pinned model revision](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/tree/e61197ed45024b0ed8a2d74b80b4d909f1255473),
[separate Qwen vision-language rerankers](https://github.com/QwenLM/Qwen3-VL-Embedding).
