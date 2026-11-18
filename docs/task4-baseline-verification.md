# Task 4 milestone 1: production baseline integration

On 17 November 2026, the unchanged Task 4 retriever was exercised through the real
Task 2 reader, Task 3 question loader and Task 5 evidence exporter on two EMT
recordings. The verification result is **passed_with_gaps**, with no contract,
repeatability, wrapper-control or future-input comparison failures. This establishes
the tested integration and temporal stability; it does not measure retrieval accuracy.

The executed source was commit `1885051d8187321946978b1e4dae8c3ed93f828a`.
The [machine-readable summary](../experiments/task4-baseline-verification-summary.json)
records commands, configuration, dependency versions, input/report hashes and gaps.
Production retrieval, shared contracts and teammates' implementation files were not
changed for this milestone.

## Inputs and environment

- Python 3.14.7 in an isolated `.venv`, with [hosted requirements](../requirements-hosted.txt).
  The Google SDK was needed by existing offline mock-transport tests; no live model
  was requested by this verification.
- Original EMT JPEGs for frames 1–150 of `video_054604` and `video_054907`, with
  unmodified original annotation files, from release revision
  `4418cec01b03c6f797cadb06c9405843a30988e2` of `KuAvLab/EMT`.
- 24 questions from [the first development bank](../experiments/temporal-development-questions.jsonl)
  and 12 from [the second bank](../experiments/temporal-second-questions.jsonl).
- The normal `trackrag.retrieval` module and default `RetrievalConfig` were used,
  with annotated observations. No development retrieval adapter was substituted.

The data is staged locally at `data/emt-task4-baseline` in Task 2's original layout:
`images/train/images/<recording_id>/<frame_id>.jpg` and
`annotations/train/<recording_id>.txt`. Its `task4-input-provenance.json` records
original-file hashes and bounded download provenance. Both combined image/observation
hashes match the previously recorded development clips. These are previously used
development inputs, not an untouched final evaluation set. The project split manifest
places the first recording in train and the second in development.

Raw images, input labels, the environment and full run directories remain Git-ignored.
The temporary input preparation and summary scripts are local setup artifacts under
`runs/task4-baseline-setup`, not new project adapters.

## Results and their limits

| Recording | Frames | Human observations | Questions | Nonempty rankings | Exact nonempty temporal comparisons | Stable-empty temporal comparisons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| video_054604 | 150 | 2,681 | 24 | 21 | 30 | 6 |
| video_054907 | 150 | 2,844 | 12 | 7 | 12 | 12 |

Score tolerance was **zero**. The 42 nonempty comparisons cover 14 earlier queries
under three variants: changed future observations, prefix-only inputs, and replaced
future images. Rankings, frame bounds, identities and scores matched exactly. These
are 14 questions tested three ways, not 42 independently reviewed questions.

Six earlier queries stayed empty under all three variants, giving 18 stable-empty
comparisons. Sixteen questions use frame 150, so no future-frame perturbation is
possible for them. The eight empty baseline cases are:

- `temporal-09`, `temporal-12`, `temporal-17`
- `second-03`, `second-05`, `second-07`, `second-10`, `second-11`

An empty ranking does not establish that a question is unanswerable. Likewise, the
28 nonempty rankings are coverage, not 28 correct retrievals or answers. Relevance
labels, candidate-coverage analysis and held-out accuracy remain separate work.

The verifier exported and audited 93 saved files across both recordings. Every answer
status is `not_requested`. Visual spot checks of the marked bus at frame 149 in the
first recording and SUV at frame 1 in the second confirmed visible box placement;
they do not validate an entry event or the full question cohort.

The ordinary `trackrag review` command was also run without adapter overrides on
`temporal-01` and `second-01`, one question per recording. Both saved configurations
name `task_modules` and `trackrag.retrieval`, with `answerer: none`. The two result
records exactly match their verifier exports. Independent saved-artifact audits passed
for those runs, covering 10 and 6 files respectively.

The complete software suite passed **282 tests, with 285 subtests**. These tests include
generated images and mocked provider responses. They are distinct from the two
real-image retrieval runs above. No real detector predictions, Qwen inference,
parameter training, live Gemini answering or gold relevance scoring occurred.

## Reproduce the checks

From the repository root, after installing the requirements in an isolated environment
and supplying the same two original clips in the layout described above:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B -m trackrag check-adapters --source annotated --answerer none
.\.venv\Scripts\python.exe -B -m evaluation verify-retrieval --data-root data/emt-task4-baseline --questions experiments/temporal-development-questions.jsonl experiments/temporal-second-questions.jsonl --source annotated --retrieval-module trackrag.retrieval --score-tolerance 0 --out runs/task4-baseline-verification-02
```

Choose a fresh output directory each time. The completed first run is in
`runs/task4-baseline-verification-01`; its report JSON hash is
`77b2692d16f69feeabc1e813d16e8d1b0c8b1d26fba527b69ef7d3a2d5c81e71`.
The verifier replays its captured baseline rankings through the normal exporter.
For the separate default-command checks, select the unchanged four-field `temporal-01`
and `second-01` records into one-question JSONL files, then run:

```powershell
.\.venv\Scripts\python.exe -B -m trackrag review --data-root data/emt-task4-baseline --recording video_054604 --questions <FIRST_QUESTION.jsonl> --source annotated --answerer none --out runs/task4-review-first-new
.\.venv\Scripts\python.exe -B -m trackrag review --data-root data/emt-task4-baseline --recording video_054907 --questions <SECOND_QUESTION.jsonl> --source annotated --answerer none --out runs/task4-review-second-new
```

Keep this measured baseline for the next milestone: define query-bounded candidate
evidence and features for frozen and trained Qwen reranking, including the scene and
temporal-context cases the current lexical baseline cannot adequately represent.
