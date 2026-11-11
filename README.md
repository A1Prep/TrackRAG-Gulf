# TrackRAG-Gulf

A shared project for MAAI 901 and 902 using annotated UAE driving footage. MAAI 901 detects and tracks road users. MAAI 902 retrieves visual evidence for driving-scene questions and grounds answers in specific frames and tracks. The NLP team can start with EMT's annotated tracks while the vision team develops its tracker.

## First run on the university lab PC

Open a terminal **inside the remote desktop session**. Use a project folder that persists between lab sessions. Clone the private repo using a GitHub account with access.

```sh
git clone https://github.com/A1Prep/TrackRAG-Gulf.git
cd TrackRAG-Gulf
```

On Windows, from PowerShell with Python 3.11 installed:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe tools\check_environment.py
.\.venv\Scripts\python.exe tools\emt_sample.py
```

On Linux, with Python 3.11 available:

```sh
python3.11 -m venv .venv
.venv/bin/python tools/check_environment.py
.venv/bin/python tools/emt_sample.py
```

If the lab already provides a managed Python environment with PyTorch, run `python tools/check_environment.py` from that environment as well. Record the OS, Python version, GPU name, total VRAM, driver, free disk and any PyTorch/CUDA result. These determine the training package versions and batch sizes. The two setup scripts use the Python standard library. Only the first sample download needs network access.

## Shared EMT example

The downloader reads a bounded portion of the official training archives at a fixed revision and writes three consecutive JPEGs and `tracks.json` under `data/emt-sample/`. Images and weights stay outside version control. [EMT dataset and terms](https://huggingface.co/datasets/KuAvLab/EMT) · [Published annotation loader](https://huggingface.co/datasets/KuAvLab/EMT/blob/4418cec01b03c6f797cadb06c9405843a30988e2/EMT.py)

| Field | Meaning |
| --- | --- |
| `recording_id`, `frame_id` | EMT recording and original integer frame number. JPEG filenames preserve zero padding. |
| `fps`, `time_origin_frame_id`, `timestamp_s` | Annotated frames at 10 Hz. Time in seconds is relative to the first sample frame. |
| `track_id`, `class_name` | Recording-local object identity and the original EMT class label. |
| `bbox_xyxy` | Left, top, right and bottom in source-image pixels. |
| `source` | `emt_ground_truth` for this example. Record predicted tracks separately when 901 produces them. |

To check the handoff, 901 can overlay a box on a JPEG. 902 can follow the same track ID into the next frame and cite both frame numbers. This three-frame training example is for integration. Keep full experimental splits at recording level for evaluation.
