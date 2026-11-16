# Task 4: search for a vehicle and its evidence interval

Task 4 takes a public question and one recording's track observations, then returns
up to three ranked intervals. Task 5 opens the selected frames and prepares evidence
images; Task 3 uses that evidence to answer. Task 4 itself does not generate answers.

The implementation is in [retrieval.py](../trackrag/retrieval.py), using the shared
[contracts](../trackrag/contracts.py). The baseline and demo use only the Python
standard library. They have been tested locally on Python 3.14.6; no GPU, model,
download or package installation is needed for the demo.

## Run the demo

Open a terminal in the repository root (the folder containing `trackrag` and
`tests`), with Python available as `python`:

```powershell
python -m trackrag.retrieval_demo --question "Which bus entered the scene?" --query-frame 100
```

The [demo](../trackrag/retrieval_demo.py) builds the real retrieval index using
synthetic observations for a bus (track 7) and a car (track 42) across frames 90–130.
The bus moves inward from the left edge; the car moves left inside the image.
Both stop moving after frame 110. Frame metadata describes a 100 × 80 image and
a synthetic 10 fps timeline. There are **no actual images or video files**: the
metadata's image paths are placeholders and cannot be opened by Task 5.

Expected default table:

```text
Rank  Recording  Source     Track  Start  End  Score
1     demo       annotated  7      90     100  0.7500
2     demo       annotated  7      100    100  0.5000
```

Even though the index includes observations through frame 130, this question only
uses frames up to 100. Each row is an interval, so the same vehicle can appear more
than once. Scores express lexical relevance, not confidence or probability.

Try changing the question or cutoff:

```powershell
python -m trackrag.retrieval_demo --question "Which car is visible?" --query-frame 100
python -m trackrag.retrieval_demo --question "Which bus is stationary?" --query-frame 130
python -m trackrag.retrieval_demo --question "bus" --query-frame 89
python -m trackrag.retrieval_demo --help
```

The car question ranks track 42 first. The stationary-bus question prefers a later
interval. The cutoff at 89 returns no results because neither vehicle has appeared.
The command defaults to the bus-entry question, cutoff 100 and at most three results.

## Output and configuration

Use JSON to print exactly the shared `RankedInterval` fields:

```powershell
python -m trackrag.retrieval_demo --question "car" --query-frame 100 --k 1 --format json
```

JSON goes to stdout; the synthetic-data notice goes to stderr. An empty JSON result
is `[]`. Valid searches, including empty ones, exit with status 0. Invalid arguments
exit with status 2 and an explanation. `--k` accepts 0 through 3.

`--source annotated` is the default. `--source predicted` exercises the same search
with a different source label on the same synthetic tracks; it does **not** run
Task 1's detector or load Task 2's real labels.

The configuration flags forward directly to `RetrievalConfig`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--window-frames` | 30 | Inclusive window length in original frame-number units |
| `--stride-frames` | 10 | Advance between window starts; cannot exceed window length |
| `--max-gap-frames` | 5 | Allowed missing frame numbers before starting a new segment |
| `--class-weight` | 2 | Class-match weight |
| `--location-weight` | 1 | Image-position-match weight |
| `--movement-weight` | 1 | Image-motion-match weight |

Weights must be finite and nonnegative, with a finite positive total. Setting a
weight to zero disables that feature. For example:

```powershell
python -m trackrag.retrieval_demo --question "bus" --window-frames 10 --stride-frames 5 --class-weight 1 --location-weight 0 --movement-weight 0
```

## How the baseline ranks evidence

Observations are grouped by recording, source and track ID. A large gap splits a
history; windows restart at the first observation of each segment. Partial windows
remain eligible. Bounds are inclusive IDs of actual observations, not invented
frames. Before descriptions or scoring, histories are truncated at `query_frame`
and windows are rebuilt, including windows that would cross the cutoff.

Each window has text features for its majority class, mean image location in thirds,
and image-space movement. More than 2% center movement distinguishes motion from
stationarity; one observation cannot establish motion. Entry/exit cues require an
observed crossing of a 5% image-edge margin with inward/outward movement. First
appearance alone does not establish entry. These thresholds are provisional.

The score is `(class_weight*C + location_weight*L + movement_weight*M) / sum(weights)`.
`C` measures matched dominant-class tokens; `L` and `M` measure matched requested
position and motion terms. Unrequested groups contribute zero. A class-only match
therefore scores 0.5 with defaults. Zero-score candidates are omitted; ties prefer
earlier start, earlier end, then smaller track ID. The English lexical baseline
does not understand negation, intent, color or physical speed. Full details and
vocabulary are documented in [the implementation](../trackrag/retrieval.py).

## Use real inputs when available

Task 2 supplies a `Recording` and annotated `TrackObservation` rows. Task 1 supplies
predicted rows with the same fields. Build a separate index for each source, then
use public questions from Task 3:

```python
from trackrag.retrieval import build_index, rank_evidence

# recording, tracks and question come from the upstream tasks.
index = build_index(recording, tracks)
intervals = rank_evidence(question, index, k=3)
```

The question contains only `question_id`, `recording_id`, `query_frame` and `question`;
gold labels are not search inputs. Results contain `recording_id`, `source`,
`track_id`, `start_frame`, `end_frame`, `rank` and `score`, ready for Task 5.

The demo is not a real-data importer or a retrieval accuracy evaluation. Real-input
evaluation remains pending: use reviewed Task 3 questions against Task 2's labels
and Task 1's predictions. For a miss, distinguish a missing correct candidate from
a candidate that was generated but ranked too low. Later model comparisons must
keep the same output contract.

## Run checks

From the repository root:

```powershell
python -B -m unittest discover -s tests -p "test_retrieval*.py"
```

[Retrieval tests](../tests/test_retrieval.py) check windowing, ranking and invariance
to future observations. [Demo tests](../tests/test_retrieval_demo.py) run the actual
module command, checking table/JSON output, both source labels, configuration,
empty results and invalid arguments.
