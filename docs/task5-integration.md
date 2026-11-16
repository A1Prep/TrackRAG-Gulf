# Running and connecting Task 5

The reviewer joins the other tasks through the records in `trackrag/contracts.py`. It writes a completed run containing `run.json`, `results.jsonl`, the observations used in `tracks.jsonl`, and marked evidence images. Image paths are relative to the run folder, so the folder can be moved intact.

## Set up and test

Use Python 3.11 or newer. From the repository root on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-integration.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_review*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_score*.py" -v
.\.venv\Scripts\python.exe -m pytest tests -q
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

Scene evidence uses `track_id: null`, an exact frame (`start_frame == end_frame`), and an image with `bbox_xyxy: null`. This supports signs, lighting and current-frame presence or absence without inventing a vehicle identity. Existing tracked intervals retain integer IDs and boxes. Task 3 answerers and Task 4 retrievers must accept this optional scene representation when connecting their modules. A scene in a predicted-source run does not require vehicle-ID association for scoring.

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

## Try real EMT footage while the task modules are in progress

An explicit development adapter is available as `tools.emt_trial_adapters`. It reads EMT's human labels and ranks descriptions of observed track histories with TF-IDF text similarity. Selecting this adapter lets the same reviewer command process original footage now. The default command continues to load the task modules listed above. The adapter checker displays the selected module so its readiness is clear.

Prepare a contiguous 15-second training interval once:

```sh
python -m tools.prepare_emt_clip --data-root data/emt-development --recording video_054604 --frames 150
```

The downloader streams a bounded prefix of the pinned official release. It writes original JPEGs, a recording manifest and human observations. It keeps both the HTTP range request and a client-side byte limit. Choose another data-root directory if an interval is already prepared. The source is [KuAvLab/EMT](https://huggingface.co/datasets/KuAvLab/EMT).

The four development questions used in the real trial are in `experiments/emt-development-questions.jsonl`. Copy that file and change its question text to try another question. For this prepared interval, query frame 150 is 14.9 seconds after the annotated recording origin. Run:

```sh
python -m trackrag review --data-root data/emt-development --recording video_054604 --questions experiments/emt-development-questions.jsonl --source annotated --answerer none --adapters tools.emt_trial_adapters --out runs/my-emt-trial
python -m tools.review_report --run runs/my-emt-trial
```

Open `report.html` in the run folder. It shows the question, answer status, cited evidence and marked original-size images. The run's `run.json` records its source and selected adapter module. Change the question file and select a new output folder to try another run.

For an authorized hosted trial, set `GEMINI_API_KEY`, select `--answerer gemini`, and optionally set `TRACKRAG_TRIAL_MODEL`. The default is `gemini-3.8-flash`. The hosted adapter masks labeled people and lower portions of labeled vehicles, then resizes images to at most 1024 by 576. This is partial masking and can miss unlabeled details. Inspect the processed inputs for the chosen recording and obtain authorization before sending footage to the provider. The `hosted` folder preserves processed images and a receipt with the actual model ID, frame IDs, token usage and response. Keys are never written to the receipt.

The completed live trial selected `TRACKRAG_TRIAL_MODEL=gemini-3.5-flash-lite` after the preferred endpoint timed out. On PowerShell, set `$env:TRACKRAG_TRIAL_MODEL='gemini-3.5-flash-lite'` to reproduce that model choice.

The development adapter loads annotated tracks. Its index also accepts supplied predicted observations, preserving their IDs and source; it does not supply a detector. It delegates to Arjun's `trackrag.emt` reader for the original split-first EMT layout, and also accepts the prepared manifest layout. Its retrieval ranking and live answer behavior are development results. Fine-tuned detector, tracker and Qwen comparisons still use the task implementations and held-out experiments.

## Recorded trial

A real run used 150 original training frames from `video_054604`, Arjun's reader, 2,681 human-labeled boxes and the TF-IDF baseline. Gemini 3.8 Flash timed out on the first live request. Gemini 3.5 Flash-Lite completed all four subsequent development questions. It answered that the yellow bus, track 46, grows in the image and cited the selected frames. It returned insufficient evidence for the entry question, driver phone use and speed in kilometres per hour. The entry question remains unresolved and is retained for review of the retrieved context.

These four development questions provide a working application check. They are not an accuracy estimate or a substitute for independent annotation and held-out evaluation. The local run folder preserves the exact processed images and model receipts.

The later bus investigation and 36-question development bank are documented in [temporal-evidence.md](temporal-evidence.md). The newer explicit trial adapter is `tools.temporal_trial_adapters`. It adds visual evidence selection, full-history candidates, exact scene snapshots and privacy-masked object crops. The original TF-IDF adapter remains available for comparisons.

## Compare saved experiments

Use `python -m evaluation compare --manifest <COMPARISON.json> --out <NEW_DIRECTORY>` to compare multiple recordings and conditions without calling a model. It audits saved evidence and review bindings, preserves missing/error cases, and writes JSON and Markdown summaries. See [task5-comparisons.md](task5-comparisons.md) for the manifest and review formats, historical development comparison and final-model evaluation protocol.

## Verify incoming retrieval and saved predictions

Aditi's Task 3 question loader is integrated. It accepts only the four public question fields, rejects duplicate IDs and answer-key fields, and accepts Windows UTF-8 BOM files. The Task 3 answerer is still separate.

Use `python -m evaluation verify-retrieval --help` for the integration checks in [retrieval-verification.md](retrieval-verification.md). The runner checks original frame/box coordinates, raw retrieval outputs, repeated runs, and changes to future observations and images. It also accepts saved detector predictions so integration can run without loading Task 1's model. The explicit TF-IDF adapter allows an offline check now; genuine detector and Qwen results must be tested when those tasks arrive.
