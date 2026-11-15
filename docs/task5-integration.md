# Running and connecting Task 5

The reviewer joins the other tasks through the records in `trackrag/contracts.py`. It writes a completed run containing `results.jsonl`, the observations used in `tracks.jsonl`, and marked evidence images. Image paths are relative to the run folder, so the folder can be moved intact.

## Set up and test

Use Python 3.11 or newer. From the repository root on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-review.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_review*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_score*.py" -v
```

On Linux, create the environment with `python3 -m venv .venv` and use `.venv/bin/python` for the remaining commands. GitHub runs the same Task 5 tests on Windows and Linux. The process tests generate temporary images and small substitute modules. They exercise the production commands and saved files. Real EMT accuracy and live-model behavior need separate runs.

## Check what can connect

```sh
python -m trackrag check-adapters --source annotated --answerer none
python -m trackrag check-adapters --source predicted --answerer gemini
```

The checker imports the selected modules and tests their function signatures without calling their adapter functions. It exits with status 1 and lists each unavailable function when modules or dependencies are missing. Module imports should keep expensive model initialization inside the callable function.

| Task | Function to supply | Returned value |
| --- | --- | --- |
| 2 | `trackrag.emt.open_recording(data_root, recording_id)` | A `Recording` with `iter_frames()` and `get_frame(frame_id)` |
| 2 | `trackrag.emt.load_annotated_tracks(recording)` | An iterable of `TrackObservation` records |
| 1 | `trackrag.vision.predict_tracks(recording)` | An iterable of observations with source `predicted` |
| 3 | `trackrag.questions.load_questions(path)` | An iterable of the four-field `Question` records |
| 3 | `trackrag.answering.answer_question(question, evidence, run_dir)` | An `Answer` using the supplied evidence IDs |
| 4 | `trackrag.retrieval.build_index(recording, tracks)` | The search index for one recording and source |
| 4 | `trackrag.retrieval.rank_evidence(question, index, k=3)` | Up to three `RankedInterval` records ordered from rank 1 |

All image boxes use original-image coordinates. Intervals include their first and last frame. A track ID belongs to a recording and source. The answerer receives image paths relative to `run_dir`. An answered result needs text and citations to supplied evidence. A missing observation, an unavailable model and a valid finding of insufficient evidence have distinct outcomes.

## Run on actual footage

After the selected adapters are available:

```sh
python -m trackrag review --data-root <EMT_ROOT> --recording <ID> --questions <QUESTIONS.jsonl> --source annotated --answerer none --out runs/annotated-01
python -m trackrag review --data-root <EMT_ROOT> --recording <ID> --questions <QUESTIONS.jsonl> --source predicted --answerer gemini --out runs/predicted-01
```

Choose a new output directory per run. If track loading or retrieval fails, the incomplete run is cleaned up. An answer-provider failure is saved as an error for that question and later questions continue. Open the marked images and verify the recording, time and road user. Repeat on a second recording using the same commands.

Task 4 must also test that changing observations after a question's query frame leaves its earlier ranking unchanged. Task 5 rejects returned intervals past the query and prevents adapters from rewriting the saved question or evidence. This check on outputs cannot establish how the search index calculated its ranking.

## Score saved evidence

The separate gold JSONL file has `question_id`, `recording_id`, `query_frame`, `answerable` and `acceptable_evidence`. Each acceptable alternative contains `track_id`, `start_frame` and `end_frame` using annotated identities. An answerable question needs at least one alternative. An unanswerable question uses an empty list. Task 3 should use this shape when exporting the scoring key.

```sh
python -m evaluation --results runs/annotated-01/results.jsonl --gold <GOLD.jsonl>
python -m evaluation --results runs/predicted-01/results.jsonl --gold <GOLD.jsonl> --predicted-tracks runs/predicted-01/tracks.jsonl --annotated-tracks runs/annotated-01/tracks.jsonl
```

The scorer counts answerable questions with a correct first result and a correct result among the first three. One acceptable alternative is sufficient. Missing question results count as misses. Unanswerable cases have their own rows and are excluded from the retrieval denominator. This metric assesses retrieval. A human or separately validated answer metric must check the answer's content.

Predicted IDs are associated against all competing annotated tracks from the same recording and retrieved interval. The scorer maximizes total mean box overlap with a one-to-one assignment. Missing observations contribute zero and pairs below `--box-iou` are ineligible. If excluding a selected pair gives an equal global optimum within `1e-9`, the identity is ambiguous. An unmatched prediction is also ambiguous if it can take a positive match in another equal optimum. Matched, unmatched and ambiguous counts remain visible. Temporal overlap uses inclusive frame IoU. The default thresholds are 0.5; select thresholds on development recordings before final evaluation.

The remaining real integration checks are runs with genuine EMT readers, detector output, reviewed questions and retrieval, followed by live answering with approved inputs. Preserve those outcomes separately from automated wiring checks.
