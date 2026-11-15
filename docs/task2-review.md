# Task 2 integration review

The reader and its original tests were extracted from Arjun's [pushed notebook](https://github.com/A1Prep/TrackRAG-Gulf/blob/0d1f12eeb8ad73f5efe145e5fdfd20ebf2715e48/task%202%20arjun.ipynb). The branch contained the notebook and readme. The generated Python module, test file and split manifest are materialized in this integration change so the reviewer can import them directly. The reader implementation is Arjun's contribution.

The supplied HTML records checks on two full training recordings, `video_054604` and `video_220047`, each with 1,800 frames. It also records 12 passing pytest tests and a 13/2/5 train/development/test split. Those are the notebook's recorded results.

The local verification independently reproduced all 12 original tests, then opened 150 original images from `video_054604` through `trackrag.emt.EMTRecording`. The returned frame numbers, 0–14.9 second clock, 2560 by 1440 dimensions and 2,681 original label boxes agree with the prepared interval. The real reviewer run records the actual reader class in `run.json`.

Additional regression cases exposed validation gaps. Nonfinite frame rates, fractional frame or track IDs, nonfinite box coordinates and duplicate track observations in one frame are now rejected. An invalid FPS declaration in release metadata also raises an error instead of silently using 10 FPS. A valid declaration is used for timestamps, and a genuinely absent declaration retains the documented default. All 23 Task 2 tests pass. Existing valid annotation geometry and IDs retain their original values.

`splits/task2_manifest.json` was reconstructed from the recording names in the pinned official annotation archives using the notebook's sorted inventory, seed 42 and two development recordings. It contains 13 training recordings, development recordings `video_054907` and `video_174340`, and the five official test recordings. It describes the full release. The local trial contains only a training interval, so its available images form a subset of that inventory. Test images were not used in the trial.

The trial search originally assumed its own recording wrapper exposed a `frames` attribute. It now uses the shared `iter_frames()`/`get_frame()` interface and accepts Arjun's reader for the original split-first layout. An automated regression test verifies that connection.

Install `requirements-integration.txt` and run `python -m pytest tests -q` for all available local checks. The HTML's full-recording checks can be repeated wherever the complete EMT cache is available.
