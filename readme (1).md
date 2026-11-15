README.md — for the Task 2 deliverable
Save this as README.md in the repo root (sibling to trackrag/, tests/, splits/).

markdown
# Task 2 — EMT Recording Reader

A small, tested reader for the **EMT** driving-recording dataset. Given a local EMT
folder and a recording ID, it opens the recording and exposes:

- its **original frames** in numerical order, with exact-path lookup and
  per-frame timestamps and dimensions;
- its **human-labeled tracks** as shared `TrackObservation` dicts carrying the
  original-pixel box, class name, and recording-local track ID.

It is the single reliable way for the rest of the project to resolve an image
or a box back to an exact original frame — so a reviewer can open the exact
image cited in an answer, and Tasks 1, 4, 5 (and the later 901 trajectory
experiment) all consume the same record.

The reader never substitutes a neighboring frame, never invents a label, and
never lets a recording leak between dataset splits.

---

## Repository layout

Pushable artifacts (these go in Git):

```text
<repo>/
├── trackrag/
│   ├── contracts.py          # shared Frame / Recording / TrackObservation dicts
│   └── emt.py                # the reader (open_recording, load_annotated_tracks, ...)
├── tests/
│   └── test_emt.py           # pytest suite
├── splits/
│   └── task2_manifest.json   # whole-recording train/dev/test manifest
├── .gitignore                # excludes the downloaded footage
└── README.md
Downloaded footage (kept outside Git):

text
<EMT>/
├── images/
│   ├── train/
│   │   └── images/
│   │       ├── video_054604/
│   │       │   ├── 00001.jpg
│   │       │   ├── 00002.jpg
│   │       │   └── ...
│   │       └── video_220047/
│   └── test/
│       └── images/
└── annotations/
    ├── train/
    │   ├── video_054604.txt
    │   └── video_220047.txt
    └── test/
The reader also accepts the shorter <split>/<recording_id>/ layout, so an
alternate extraction without the inner images/ level still works.

Setup
The reader needs Python 3.10+, opencv-python-headless, numpy, and (for the
tests) pytest. To install into the current interpreter:

bash
python -m pip install -q opencv-python-headless numpy matplotlib pytest
The shared contract definitions in trackrag/contracts.py must be present
before importing trackrag.emt. They are supplied by the project and should
not be redefined here.

Quick start
Open one recording and inspect its first and last frame plus one labeled
object:

bash
python -m trackrag.emt --data-root /path/to/EMT --recording-id video_054604
Example output:

text
FIRST: recording=video_054604 split=train frame=1 timestamp_s=0.000 \
       dimensions=1280x720 image_path=/path/to/EMT/images/train/images/video_054604/00001.jpg
LAST:  recording=video_054604 split=train frame=612 timestamp_s=61.100 \
       dimensions=1280x720 image_path=/path/to/EMT/images/train/images/video_054604/00612.jpg
LABELED OBJECT: recording=video_054604 frame=3 track=42 class=Car \
       bbox_xyxy=(612.0, 331.0, 704.0, 412.0) source=annotated
If the recording's label file is missing, the command still prints the first
and last frame and then reports IMAGES ONLY: …; the recording opens fine in
images-only mode.

Python API
python
from trackrag.emt import open_recording, load_annotated_tracks

recording = open_recording("/path/to/EMT", "video_054604")

# Iterate frames in numerical order.
for frame in recording.iter_frames():
    print(frame["frame_id"], frame["timestamp_s"], frame["image_path"])

# Or open one exact frame; this never returns a different frame.
frame = recording.get_frame(17)
assert frame["recording_id"] == "video_054604"
assert frame["frame_id"] == 17

# Load the human labels as shared TrackObservation dicts.
observations = load_annotated_tracks(recording)
for obs in observations[:3]:
    print(obs["frame_id"], obs["track_id"], obs["class_name"], obs["bbox_xyxy"])
What a Frame contains
key	type	meaning
recording_id	str	the recording this frame belongs to
frame_id	int	numeric stem of the image filename
timestamp_s	float	seconds from the first available image frame
width	int	original image width in pixels
height	int	original image height in pixels
image_path	str	local path to the exact original image file
What a TrackObservation contains
key	type	meaning
recording_id	str	the recording this label belongs to
frame_id	int	frame the box was annotated on
track_id	int	recording-local track identifier
class_name	str	road-user class (e.g. Car, Bus, Pedestrian)
bbox_xyxy	tuple	(left, top, right, bottom) in original pixels
source	str	always "annotated" for labels loaded from EMT
Behavioral guarantees
Exact frame lookup. get_frame(frame_id) returns the exact requested
image, or raises FrameNotFoundError (a KeyError subclass). It never
falls back to a neighbor.

Numerical iteration. iter_frames() yields frames sorted by numeric
filename stem, not by creation order or lexicographic name.

Clock anchor. timestamp_s == 0.0 on the first available image
frame (recording.first_frame_id). Every timestamp is
(frame_id - first_frame_id) / frame_rate.

Frame rate. open_recording reads the release frame rate from
annotations/metadata.txt when present; otherwise it uses the documented
DEFAULT_FRAME_RATE = 10.0 and callers can pass frame_rate= to override.

Images-only mode. A recording opens and decodes frames without labels.
Only load_annotated_tracks() raises MissingLabelsError when the label
file is absent.

Recording-local track IDs. The same numeric track_id in two
recordings refers to two distinct vehicles. Loading one recording never
returns the other's observations.

Split safety. A recording found in more than one split is rejected
rather than silently double-assigned.

Annotation layout. EMT GMOT columns:
frame_id track_id class_name <x> <y> <z> left top right bottom.
Columns 6..9 are original-pixel xyxy. AV ego-vehicle rows and labels
for unavailable image frames are skipped.

Splits
splits/task2_manifest.json records whole-recording assignments to
train / dev / test:

The official EMT test recordings are preserved unchanged.

Development recordings are selected only from the official train set.

Every available recording belongs to exactly one split.

A recording found in more than one split is an error.

Generate or regenerate the manifest from the footage tree:

python
from trackrag.emt import make_split_manifest
manifest = make_split_manifest("/path/to/EMT", dev_count=2, seed=42)
By default the manifest is written to splits/task2_manifest.json (repo
relative) so it can be committed. It is not written inside the downloaded
footage tree.

Tests
Run the full suite:

bash
python -m pytest tests/test_emt.py -v
The suite covers:

numerical frame ordering regardless of file creation order;

exact frame lookup returning the requested original image;

frame-rate–dependent timestamp conversion and clock anchoring;

FrameNotFoundError on a missing frame ID (and KeyError compatibility);

original-pixel bbox_xyxy from the annotation file;

zero-label frames (timestamps still correct, no observations);

AV ego-vehicle rows being skipped;

images-only mode when the label file is missing;

recording-local track IDs (same numeric ID across two recordings);

split disjointness and preservation of the official test set;

frame-rate validation and metadata fallback.

No real EMT data is required — the suite builds a small synthetic fixture in
tmp_path that mirrors the on-disk layout.

Keeping footage out of Git
The downloaded footage is deliberately excluded. A .gitignore at the repo
root should contain at minimum:

gitignore
/workspace/EMT/
demo_emt/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
What this enables next
Task 1 replaces its temporary frame reader with open_recording /
get_frame.

Task 4 searches load_annotated_tracks output while the vision model is
being developed.

Task 5 resolves Task 4's evidence back to real images via image_path.

901's trajectory experiment uses the same reader for its
constant-velocity vs. small-LSTM forecast comparison.