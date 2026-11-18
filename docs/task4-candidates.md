# Task 4 milestone 2: evidence candidates for learned reranking

[retrieval_candidates.py](../trackrag/retrieval_candidates.py) defines a model-neutral
candidate pool and its evidence features. The existing lexical retriever remains
the normal application default. This milestone does not load Qwen, score candidates,
produce answers, invent relevance labels or change Task 5's shared output records.

## Build the pool

```python
from trackrag.emt import open_recording, load_annotated_tracks
from trackrag.questions import load_questions
from trackrag.retrieval import build_index
from trackrag.retrieval_candidates import build_candidate_batch

recording = open_recording("data/emt-task4-baseline", "video_054604")
tracks = load_annotated_tracks(recording)
question = load_questions("experiments/temporal-development-questions.jsonl")[0]
index = build_index(recording, tracks)
batch = build_candidate_batch(question, index, source="annotated")
```

Predictions use the same function with `source="predicted"` and actual predicted
observations. Source is mandatory: it describes the selected run condition even
when there are no permitted track observations. The generator validates eligible
rows against it; it does not use future-only observations to decide scene source.
An empty predicted history does not prove that a detector ran or that no vehicle
was visible.

The batch uses schema `trackrag.candidates.v1` and has three candidate kinds:

| Kind | Evidence represented | Time policy |
| --- | --- | --- |
| `track_window` | One short interval of one road user | Reuses the baseline window, stride and gap configuration after truncation |
| `track_history` | The complete supplied observation history of one road user up to the cutoff | May span gaps, which are explicitly recorded |
| `scene` | One exact full-scene frame, without a target vehicle box | Query frame, explicitly named legal frames, and earliest permitted observation frame |

All candidate observations and image references are at or before `query_frame`.
No full-recording frame scan discovers an opening frame. The earliest observation
frame is not necessarily the beginning of the recording or the first visual
appearance of the vehicle.

Identical history/window intervals are emitted once, as a history. Different
intervals of the same vehicle remain separate. Candidates are ordered by histories
in track-ID order, then local windows in track/frame order, then scenes in frame
order. This is deterministic enumeration, not relevance ranking. The pool has no
top-three cap or lexical filter, so potentially relevant evidence is not removed
before the learned ranker sees it.

## Batch and candidate fields

| Batch field | Meaning |
| --- | --- |
| `schema_version` | Version of this internal candidate representation |
| `question`, `source` | Copied four-field public question and explicit track source |
| `policy` | Window settings and scene/image selection policies |
| `histories` | Shared permitted observations and features for each recording/source/track identity |
| `candidates` | Local windows, whole histories and exact scenes |
| `unavailable_scene_frames` | Requested legal scene IDs that the reader could not open exactly |
| `excluded_future_frame_requests` | Explicitly named frames beyond the query cutoff; never looked up |
| `metadata_sha256` | Portable fingerprint of this batch's metadata, excluding image paths |

Each candidate contains its stable `candidate_id`, `kind`, recording/source/track
identity, inclusive original start/end frames, `history_id`, `features` and `images`.
Scene candidates have null track/history IDs and null tracked features. Their single
image has a null bounding box. Every image reference retains the shared frame fields
and the target's original-pixel box when applicable.

Tracked features retain observed frame IDs, observation count, original class-label
counts, first/last boxes, normalized first/last centers, net center displacement and
center range. They also list every missing-observation interval and distinguish
gaps exceeding the configured continuity allowance. Shared histories retain all
original per-frame class labels and boxes, so summaries do not erase class changes.

A single observation has null displacement. A zero net displacement does not mean
stationary: the range can reveal an out-and-back trajectory. Across gaps, geometry
does not establish continuous movement. No feature claims physical speed, intent,
entry time, or absence based only on missing annotations.

## Image selection and visual limits

For each tracked interval, image references cover the first, lower middle
(`(n-1)//2`) and last observed frames, plus explicitly requested frames actually
observed inside the interval. This matches the current reviewer export policy.
The shared frame-reference parser extracts written numbers; a textual range supplies
its written endpoints, not every integer between them.

This follows the existing English `frame`/`frames` reference syntax. It does not add
general natural-language time interpretation or convert seconds to frame IDs.

Missing exact scene frames are reported and omitted. They are never replaced by a
neighbor. A missing frame referenced by an existing observation is an error. Frame
metadata is cached within a call, with no lookups beyond the cutoff. The module does
not open image pixels itself; the actual EMT reader can decode images internally
when supplying metadata.

Image references make visual evidence available for a later adapter. They are not
visual embeddings, crops, captions or OCR. A text-only Qwen model cannot determine
color, sign text or appearance from class/box metadata alone. The model-selection
milestone must choose a suitable visual/text input approach and specify processing,
privacy and image-budget policies. This module neither uploads images nor creates
processed evidence files.

## Identity, fingerprints and future experiments

History IDs identify a recording/source/track. Candidate IDs identify the same
scope plus candidate kind and interval bounds. They are deterministic SHA256-based
identifiers, not relevance scores. IDs stay the same when identity is unchanged,
even when underlying permitted observations change; use the batch fingerprint to
bind the actual versioned metadata and question.

The fingerprint excludes image paths so moving the same recording does not change
it. It does **not** hash image bytes, authenticate a detector or validate relevance.
Before training, bind reviewed labels to the question, candidate ID, schema/policy,
metadata fingerprint and actual image-content hashes. Keep gold labels outside
inference batches. The exported records are mutable snapshots; modifying a batch
invalidates its stored fingerprint and does not alter the index or future calls.

Freeze candidate generation, available features and image policy when comparing
simple, frozen-Qwen and fine-tuned-Qwen ranking. Broad history candidates have
different temporal precision from local windows; measure that tradeoff rather than
treating long intervals as automatically correct. Scene availability is context
coverage, not proof of answerability or a successful semantic retrieval.

## Preserve the downstream contract

Once a later ranker has selected a candidate from the current batch, it can project
it into Task 5's unchanged public record:

```python
from trackrag.retrieval_candidates import candidate_to_interval

# selected is an existing member of this question's batch, chosen by the later ranker.
interval = candidate_to_interval(selected, rank=1, score=model_score)
```

The helper returns only `recording_id`, `source`, `track_id`, `start_frame`,
`end_frame`, `rank` and `score`. It accepts ranks 1–3 and finite scores, including
logits; it does not choose candidates or validate a model's returned candidate IDs.
The future ranker must enforce current-batch membership, ordering and top-k selection.

Run the [focused candidate tests](../tests/test_retrieval_candidates.py) with:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_retrieval_candidates.py -q -p no:cacheprovider
```

These tests use synthetic observations and metadata, with no model or image downloads.

## Milestone verification

The complete local suite passed **306 tests, with 321 subtests**, including 24 focused
candidate tests. An independent code review found no blocking issue; its suggested
checks for missing observed frames, malformed metadata and invalid output scores
were added before the final suite run.

Two additional checks used the original EMT clips prepared in
[milestone 1](task4-baseline-verification.md), through `trackrag.emt.EMTRecording`:

| Development question | Query frame | Whole histories | Local windows | Exact scenes | Total candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| `temporal-09`, `video_054604` | 30 | 31 | 28 | 2 | 61 |
| `second-07`, `video_054907` | 75 | 40 | 127 | 2 | 169 |

Every projected candidate passed the existing Task 5 interval-contract checker.
The complete batches, including metadata fingerprints, were exactly unchanged under
a pass-through wrapper, changed future observations, prefix-only inputs, and changed
future image references. No requested frame was missing in these two cases.

Both questions were selected because their measured lexical rankings were empty.
Those baseline rankings remain empty and unchanged: generating a broader candidate
pool does not select an answer or establish a relevance improvement. These are two
development representation checks, not held-out evaluation. Image references were
checked against actual local files; no model read pixels or generated visual features.

The [recorded summary](../experiments/task4-candidates-verification-summary.json)
contains question inputs, candidate counts, comparison outcomes, packet hashes and
hashes of the new source/test files (which were not yet committed during execution).
The full local batches are under `runs/task4-candidates-verification-01`, outside Git.
Image-path changes were the future-image control; no visual model was tested.

The next [relevance-review workflow](task4-relevance-dataset.md) snapshots these
packets and actual image bytes for human labeling, with recording-level split checks
and a separate export of reviewed examples. Candidate availability alone never
assigns a relevance label.
