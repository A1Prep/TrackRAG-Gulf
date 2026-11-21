# Task 4: compare lexical and pretrained Qwen ranking on one candidate pool

[retrieval_comparison.py](../trackrag/retrieval_comparison.py) implements the next
experiment after [pretrained Qwen inference](task4-qwen-pretrained.md). It generates
one complete query-bounded candidate batch and scores every candidate with both
the lexical feature scorer and the frozen text Qwen adapter. This provides a
ranking comparison before human relevance labels are available. It does not
measure accuracy, train a model, generate answers or replace Task 5 evaluation.

## What is controlled

Both rankers receive the same question, recording/source identity, inclusive query
cutoff, candidate IDs and candidate bounds. The shared pool contains local track
windows, whole permitted track histories and available exact scene frames. There
is no lexical shortlist, candidate-count cap or use of observations after the
cutoff. Changing `--k` changes the number selected, not the number scored.

Both variants retain zero scores, apply no relevance threshold and break score
ties by ascending start frame, end frame, tracked-before-scene, track ID and
candidate ID. Up to three selected intervals use the unchanged shared
`RankedInterval` fields. Complete rankings retain all candidate IDs and positions.
Rankers cannot introduce candidates outside the saved pool.

This is a comparison of two ranking pipelines on one pool, not a model-only
ablation with identical input representations:

| Variant | Inputs and scoring | Limitations |
| --- | --- | --- |
| `lexical_shared_pool` | Reuses the original class/location/movement descriptors and weighted-overlap function on each candidate's own observations. Reads original dimensions for every observed frame, including unsampled interior frames. | No scene descriptor: scenes score zero. Whole-history descriptors can span explicit observation gaps. Image-space entry/exit rules remain heuristics. |
| `qwen_text_shared_pool` | Reuses the existing versioned compact text documents, fixed instruction and pretrained yes/no token-logit scoring. | Does not read pixels, color or OCR. Scores are not calibrated correctness. |

The default [production lexical ranker](../trackrag/retrieval.py) is unchanged: it
still searches short windows and omits zero-score results. The comparison's lexical
variant expands its pool and retains zero scores deliberately. A result from this
variant must not be presented as the unchanged production baseline's output.
Numerical lexical and Qwen scores have different meanings; do not subtract them
or interpret a larger number from one ranker as greater accuracy.

## Run locally

Use the optional dependencies and downloaded weights described in the
[Qwen setup guide](task4-qwen-pretrained.md). Run from the repository root. The
question JSONL must contain only the four public question fields and belong to
the selected recording. Gold answers and review labels are not ranking inputs.

For offline use after the weights have been downloaded, create a configuration
outside version control, for example `runs/qwen-comparison-config.json`:

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

Create a question file for one recording and choose a fresh output directory:

```powershell
.\.venv\Scripts\python.exe -B -m trackrag.retrieval_comparison --data-root data/emt-task4-baseline --recording video_054604 --questions <ONE_RECORDING_QUESTIONS.jsonl> --source annotated --qwen-config runs/qwen-comparison-config.json --out runs/ranking-comparison-new
```

`--k` accepts 1, 2 or 3. `--retrieval-config <JSON>` accepts `RetrievalConfig` fields
for the shared window policy and lexical feature weights; omitted values keep
the documented defaults. The comparison never silently changes its candidate
policy to fit the model's token budget. Model context-limit errors remain errors.

For saved predictions use `--source predicted --predicted-tracks <JSONL>`.
This reads predictions in the shared observation format; it does not run Task 1.
Recording/source/track identities remain separate. Never equate annotated and
predicted IDs directly. Run each recording/source condition separately.

To compare a saved trained adapter on that same pool, add `--checkpoint <DIR>`.
Its base configuration is loaded from the checkpoint unless `--qwen-config`
explicitly overrides runtime settings. Model identity/revision must match.
`--allow-synthetic` requires a checkpoint and is only an explicit smoke-test opt-in.
Scorer provenance distinguishes a frozen model, a synthetic adapter and a reviewed
adapter; the variant label alone does not establish a real-data training run.

CPU scoring can take minutes even for small pools. Begin with one early-frame
question. This command does not require a hosted model, API key or cloud service.

## Outputs and interpretation

Each fresh run directory contains:

- `run.json`: cohort completion/error status, expected and completed question
  counts, input-question file hash, executed source-file hashes and configuration.
- `candidates-0001.json`, etc.: the exact candidate batches, original frame/image
  references and metadata hashes for each question.
- `comparison-0001.json`, etc.: every candidate's lexical and Qwen score, full
  rankings, selected intervals, descriptor documents, input hashes, timing and
  scorer provenance. Actual Qwen provenance includes the loaded model revision.
- `comparisons.jsonl`: compact successful per-question rows containing both
  interval lists and ranking agreement. These are comparison records, not Task 5
  answer results. A failed run's file may contain only a completed prefix.
- `failure.json`, when a model scoring failure occurred: partial receipt with the
  failure and completed lexical side, without substituted Qwen results.

Output directories are not overwritten. A failed cohort is marked `error` in
`run.json`, even if earlier questions completed. An empty pool returns empty
rankings without loading/calling Qwen. An empty track list can still have a scene
candidate and therefore can still invoke Qwen.

Input hashes bind text/metadata, not image bytes. No image pixels are model
features in this experiment; the recording reader may decode images to obtain
frame metadata. The separate relevance-review pack binds the images inspected
by a human. It remains the appropriate route for reviewed training/evaluation
labels.

Agreement reports whether the rankers chose the same candidate first and how
many selected candidates overlap. Agreement is not a relevance metric: two
rankers can agree on unsuitable evidence. `accuracy_metrics` is explicitly null
and `evaluation_status` remains `requires_reviewed_relevance_labels`.

Once human review is complete, join labels using the exact candidate IDs AND
batch metadata identity; candidate IDs alone do not authenticate their contents.
Pending or uncertain judgments must not be converted into negatives. Keep
train/dev/test recordings separate and use the existing Task 5 scoring interfaces
for downstream evidence/answer evaluation. Distinguishing a missing correct
candidate from a low-ranked one requires reviewed evidence, not ranker agreement.

## Checks

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_retrieval_comparison.py -q
.\.venv\Scripts\python.exe -B -m trackrag.retrieval_comparison --help
```

[Contract tests](../tests/test_retrieval_comparison.py) inject scorers and synthetic
recordings. They exercise full-pool scoring, shared ties, zero scores, original
lexical score reuse, original dimensions, source/recording isolation, crossing
cutoffs, future changes, immutable caller inputs, invalid scores, preserved
failures and saved CLI outputs. These checks do not establish pretrained quality.
Real-weight checks are recorded separately in the experiment summary.

## Recorded local validation

The complete suite passed **373 tests and 376 subtests**, including 19 focused
comparison tests. These checks use injected scorers; they are separate from the
actual downloaded-weight checks below.

The new command completed on both real EMT recordings with annotated tracks and
the cached frozen `Qwen/Qwen3-Reranker-0.6B` revision. Both questions use frame 1:

| Question | Recording | Complete pool | Lexical top-three track IDs | Qwen top-three track IDs | Qwen CPU time including loading |
| --- | --- | ---: | --- | --- | ---: |
| temporal-11 | video_054604 | 15 | 1, 3, 4 | 8, 9, 13 | 60.65 s |
| second-01 | video_054907 | 20 | 2, 3, 9 | 14, 19, 13 | 80.23 s |

Every candidate was scored by both rankers. Saved Qwen documents, input hashes,
all scores and selected intervals exactly match the earlier actual pretrained
pilot for those questions. Readback checks validated batch hashes, membership,
complete rankings, source/time bounds and the loaded model revision. Timing is a
pilot observation, not a benchmark. Neither question has a completed relevance
review, so these different rankings do not establish an improvement.

The first pool has 14 track histories and one scene; the second has 19 histories
and one scene. At frame 1, history/window bounds coincide and are deduplicated, so
these actual-weight checks do not exercise later local windows. Synthetic tests
cover later cutoffs, crossing windows and changed future observations separately.
No genuine prediction run, answerer or Task 5 exporter is invoked by this step.

The [recorded comparison summary](../experiments/task4-ranking-comparison-summary.json)
contains the exact intervals, configuration/provenance, input/output hashes and
local run paths. Human labels, relevance metrics and supervised training remain
pending.
