"""Write notebooks/task1_colab.ipynb (the Colab driver for Task 1 data preparation and training).

The notebook only calls repository scripts; all logic lives in tested Python modules.
Shell work uses Colab's %shell / %%shell magics: they stream output live and raise on a
non-zero exit (``!`` and ``%%bash`` do neither), so a failed step stops "Run all".
Regenerate after editing: python -m tools.build_task1_notebook
"""

import json
from pathlib import Path

REVISION = "4418cec01b03c6f797cadb06c9405843a30988e2"
TRAIN_IMAGES_SHA256 = "d1b8d4221d9f2460d4243a5c1f2ed3133dc11231a256035c19c71a00165b3f1e"
TRAIN_IMAGES_BYTES = 21848135924
# Downloaded and hashed locally on 2026-11-18 (contains exactly the 15 manifest train + dev recordings).
TRAIN_ANNOTATIONS_SHA256 = "58ac63c71f1b6c12dfc8b31dde20afdfbcac9af3066929526053968438fcfb44"
# Expected dataset (computed from the real label archive: every labelled frame 1..N, every 5th frame).
EXPECTED = {"train_images": 3799, "valid_images": 720, "train_boxes": 66763, "valid_boxes": 15441}
DRIVE = "/content/drive/MyDrive/task1"
ZIP = f"{DRIVE}/code/trackrag-task1-code.zip"
WEIGHTS = "/content/models/rf-detr-large-2026.pth"

CELLS = [
    ("markdown", f"""# Task 1 (TrackRAG-Gulf): EMT data preparation and RF-DETR Large fine-tuning

Run the sections in order; each says which runtime it needs.

* **Never paste tokens or passwords into cells.** Code comes from the zip you upload to `{DRIVE}/code/`.
* **Section A** (data) needs **no GPU**: Runtime > Change runtime type > CPU (High-RAM if offered).
* **Section B** (training) needs a GPU: **L4** preferred, T4 fallback.
* Colab Pro does not train in the background: keep this tab open and the laptop awake during training.
* Cells with a `MICRO_CHECKED` or `RESUME` flag run only after you set the flag on purpose."""),
    ("markdown", "## 0. Every session: environment, Drive, code"),
    ("code", """!nvidia-smi || echo "No GPU in this runtime"
!df -h /content | tail -1
!free -h | head -2
!nproc"""),
    ("code", f"""from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '{DRIVE}'
os.makedirs(DRIVE, exist_ok=True)
print(DRIVE)"""),
    ("code", f"""ZIP = '{ZIP}'
assert os.path.isfile(ZIP), f'Upload trackrag-task1-code.zip to {{ZIP}} first (one copy only, no "(1)" duplicates)'
_r = %shell sha256sum "{{ZIP}}"
print('Compare this hash with trackrag-task1-code.zip.sha256 on the laptop.')
_r = %shell rm -rf /content/repo && mkdir -p /content/repo && unzip -q "{{ZIP}}" -d /content/repo
%cd /content/repo
_r = %shell ls /content/repo"""),
    ("markdown", "## A. Data preparation (CPU runtime, once)\n\nDownload the pinned EMT release (resumable, verified), build the COCO dataset, save one tar to Drive."),
    ("code", f"""%%shell
set -euo pipefail
if pgrep -x curl >/dev/null; then echo "An old download is still running. Stop it with: pkill -x curl"; exit 1; fi
BASE=https://huggingface.co/datasets/KuAvLab/EMT/resolve/{REVISION}
EXPECTED_BYTES={TRAIN_IMAGES_BYTES}
mkdir -p /content/emt && cd /content/emt
size() {{ stat -c%s "$1" 2>/dev/null || echo 0; }}
# Resumable: safe to re-run. No curl --retry (it would truncate the partial file); this loop resumes instead.
attempt=0
while [ "$(size train_images.tar.gz)" -lt "$EXPECTED_BYTES" ]; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt 40 ]; then echo "Download keeps failing: check the network, then re-run this cell"; exit 1; fi
  echo "attempt $attempt: have $(size train_images.tar.gz) of $EXPECTED_BYTES bytes"
  curl -L --fail --speed-limit 1048576 --speed-time 120 -C - -o train_images.tar.gz "$BASE/train_images.tar.gz" || sleep 5
done
if [ "$(size train_images.tar.gz)" -ne "$EXPECTED_BYTES" ]; then echo "Unexpected size: delete train_images.tar.gz and re-run"; exit 1; fi
echo "{TRAIN_ANNOTATIONS_SHA256}  train_annotation.tar.gz" | sha256sum -c --status - || {{ rm -f train_annotation.tar.gz; curl -L --fail --retry 5 -o train_annotation.tar.gz "$BASE/train_annotation.tar.gz"; }}
ls -l
echo "{TRAIN_IMAGES_SHA256}  train_images.tar.gz" | sha256sum -c -
echo "{TRAIN_ANNOTATIONS_SHA256}  train_annotation.tar.gz" | sha256sum -c -"""),
    ("code", """%%shell
set -euo pipefail
if pgrep -f '^python3? -m tools[.]vision_prepare_dataset' >/dev/null; then echo "An old dataset prep is still running"; exit 1; fi
# The script re-verifies the archive hash itself (about 2 minutes) and records it in the manifest.
cd /content/repo
python -m tools.vision_prepare_dataset \\
    --images-tar /content/emt/train_images.tar.gz \\
    --annotations-tar /content/emt/train_annotation.tar.gz \\
    --out /content/ds_v1 --pack /content/ds_v1.tar"""),
    ("code", f"""import json
summary = json.load(open('/content/ds_v1/dataset_manifest.json'))
counts = summary['counts']
print(json.dumps(counts, indent=1))
print('dropped/adjusted:', summary['dropped_or_adjusted_label_rows'])
print('other archive members:', summary['other_archive_members'])
for name, info in summary['recordings'].items():
    print(f"{{name:22s}} {{info['role']:5s}} frames={{info['frames_in_archive']:4d}} size={{info['image_size']}} "
          f"gaps={{info['frame_gap_count']}} unlabelled_images={{info['images_without_labels']}}")
EXPECTED = {EXPECTED!r}
roles = [info['role'] for info in summary['recordings'].values()]
assert roles.count('train') == 11 and roles.count('valid') == 2, roles
assert 'video_054604' not in summary['recordings'] and 'video_204347' not in summary['recordings']
assert summary['inputs']['images_tar']['sha256'] == '{TRAIN_IMAGES_SHA256}'
assert counts['train']['images'] == EXPECTED['train_images'], counts['train']['images']
assert counts['valid']['images'] == EXPECTED['valid_images'], counts['valid']['images']
assert abs(counts['train']['annotations'] - EXPECTED['train_boxes']) <= 0.005 * EXPECTED['train_boxes']
assert abs(counts['valid']['annotations'] - EXPECTED['valid_boxes']) <= 0.005 * EXPECTED['valid_boxes']
# video_214547 has 1,800 images but labels only up to frame 1,629: its last 171 frames are not used.
unlabelled = {{name: info['images_without_labels'] for name, info in summary['recordings'].items()
              if info['images_without_labels']}}
assert unlabelled == {{'video_214547': 171}}, unlabelled
assert summary['sampled_images_skipped_without_labels'] == 34
assert summary['other_archive_members'] == {{'train recording without a role': 3600}}  # the two excluded recordings
assert all(counts['valid']['by_class'][c] > 0 for c in ('Car', 'Bus', 'Medium_vehicle', 'Large_vehicle'))
print('ALL DATASET CHECKS PASSED')"""),
    ("code", """from IPython.display import Image, display
import glob
for path in sorted(glob.glob('/content/ds_v1/sanity/*.jpg'))[:10]:
    display(Image(filename=path, width=640))"""),
    ("code", f"""%%shell
set -euo pipefail
df -h /content/drive | tail -1
mkdir -p {DRIVE}/data
cp /content/ds_v1.tar /content/ds_v1.tar.sha256 {DRIVE}/data/
cp /content/ds_v1/dataset_manifest.json {DRIVE}/data/ds_v1.dataset_manifest.json
ls -l {DRIVE}/data"""),
    ("code", f"""# Push the copy to Google's servers, then check it. Keep this CPU runtime until the
# restore cell in section B passes on the GPU runtime (that is the real proof of the Drive copy).
drive.flush_and_unmount(timeout_ms=30 * 60 * 1000)
drive.mount('/content/drive')
_r = %shell cd {DRIVE}/data && sha256sum -c ds_v1.tar.sha256"""),
    ("markdown", "## B. Training (GPU runtime)\n\nRun section 0 first. Then install, restore the dataset, run the micro gate (train + Drive sync + resume), then the full run."),
    ("code", f"""!pip install -q "rfdetr[train]==1.12.1" "ultralytics==8.4.215" "lap>=0.5.12"
_r = %shell python -c "import rfdetr, ultralytics, lap, torch; print('rfdetr OK', ultralytics.__version__, torch.__version__, torch.cuda.is_available())"
!mkdir -p {DRIVE}/env && pip freeze > {DRIVE}/env/pip-freeze-train.txt"""),
    ("code", f"""%%shell
set -euo pipefail
mkdir -p /content/models
cp {DRIVE}/data/ds_v1.tar {DRIVE}/data/ds_v1.tar.sha256 /content/
cd /content && sha256sum -c ds_v1.tar.sha256 && rm -rf /content/ds_v1 && tar -xf ds_v1.tar -C /content
ls /content/ds_v1
df -h /content/drive | tail -1
echo "Drive needs about 3 GB free for training state. Empty the Drive Trash now if space is tight." """),
    ("code", f"""# MICRO GATE part 1 (about 10 minutes): 1 epoch on 100/40 images, with Drive sync every minute.
!rm -rf /content/runs/micro /content/runs/micro-dataset {DRIVE}/runs/micro
_r = %shell cd /content/repo && python -m tools.vision_train --dataset /content/ds_v1 --output /content/runs/micro --weights {WEIGHTS} --micro 100 40 --epochs 1 --batch-size auto --device cuda --sync-dir {DRIVE}/runs/micro --sync-every-min 1 --code-zip "{{ZIP}}" """),
    ("code", f"""import json
micro = json.load(open('/content/runs/micro/task1_training_summary.json'))
print('class names :', micro['selected_checkpoint']['class_names'])
print('checkpoint  :', micro['selected_checkpoint']['kind'], micro['selected_checkpoint']['sha256'][:12])
print('auto batch  :', micro['resolved_train_config'])
print('last.ckpt   :', micro['has_last_ckpt'], micro['last_ckpt_progress'])
print('drive sync  :', micro['drive_sync'])
print('seconds     :', micro['seconds'], '| gpu:', micro['environment']['gpu'])
assert micro['finished_normally'] and micro['has_last_ckpt']
assert micro['selected_checkpoint']['class_names'] == ['Pedestrian', 'Cyclist', 'Motorbike', 'Car',
    'Small_motorised_vehicle', 'Medium_vehicle', 'Large_vehicle', 'Bus', 'Emergency_vehicle']
assert micro['drive_sync'] and not micro['drive_sync']['errors']
assert os.path.isfile('{DRIVE}/runs/micro/last.ckpt')
print('MICRO PART 1 PASSED')"""),
    ("code", f"""# MICRO GATE part 2: delete the local run, then resume from Drive to epoch 2 (proves disconnect recovery).
!rm -rf /content/runs/micro
_r = %shell cd /content/repo && python -m tools.vision_train --dataset /content/ds_v1 --output /content/runs/micro --weights {WEIGHTS} --micro 100 40 --epochs 2 --batch-size auto --device cuda --sync-dir {DRIVE}/runs/micro --sync-every-min 1 --resume
resumed = json.load(open('/content/runs/micro/task1_training_summary.json'))
print('after resume:', resumed['last_ckpt_progress'], resumed['selected_checkpoint']['kind'])
assert resumed['finished_normally'] and resumed['resume']
print('MICRO PART 2 PASSED: training, Drive sync, resume and class order all work on Colab')"""),
    ("code", f"""# FULL RUN. Choose EPOCHS from the micro run's speed (aim to finish within about 4 hours) and BATCH from
# 'auto batch' above (L4 typically 8, T4 4; grad-accum defaults to 16 // BATCH). Then set MICRO_CHECKED.
MICRO_CHECKED = False
EPOCHS = 12
BATCH = 8
assert MICRO_CHECKED, 'Read the micro results, set EPOCHS and BATCH, then set MICRO_CHECKED = True'
_r = %shell cd /content/repo && python -m tools.vision_train --dataset /content/ds_v1 --output /content/runs/ft01 --weights {WEIGHTS} --epochs {{EPOCHS}} --batch-size {{BATCH}} --device cuda --num-workers 4 --sync-dir {DRIVE}/runs/ft01 --code-zip "{{ZIP}}" """),
    ("code", f"""# ONLY after a disconnect: run section 0, the install cell and the dataset-restore cell, then set RESUME = True.
RESUME = False
if RESUME:
    import glob, json
    os.makedirs('/content/models', exist_ok=True)
    first = sorted(glob.glob('{DRIVE}/runs/ft01/task1_train_request_*.json'))[0]
    original = json.load(open(first))['train_options']
    EPOCHS, BATCH = original['epochs'], original['batch_size']
    print('Resuming with the original settings:', EPOCHS, 'epochs, batch', BATCH)
    _r = %shell cd /content/repo && python -m tools.vision_train --dataset /content/ds_v1 --output /content/runs/ft01 --weights {WEIGHTS} --epochs {{EPOCHS}} --batch-size {{BATCH}} --device cuda --num-workers 4 --sync-dir {DRIVE}/runs/ft01 --resume --code-zip "{{ZIP}}"
else:
    print('RESUME is False: nothing to do')"""),
    ("code", f"""import json
final = json.load(open('/content/runs/ft01/task1_training_summary.json'))
print(json.dumps({{k: final[k] for k in ('selected_checkpoint', 'resolved_train_config', 'last_ckpt_progress', 'drive_sync', 'seconds')}}, indent=1))
!ls -la {DRIVE}/runs/ft01"""),
]


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {"cells": cells, "metadata": {"colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "notebooks" / "task1_colab.ipynb"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
