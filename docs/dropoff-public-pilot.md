# Public destination pilot splits

The public pilot supplies genuine Talk2Car-RegSeg training-source images,
instructions and destination masks. It does not supply reviewed EMT drop-off,
ambiguity, clarification, dialogue or voice labels. It is not the final test set.

Prepare a bounded subset and separate whole source scenes for development:

```powershell
python -m tools.dropoff_bootstrap_regseg --out data/regseg-public --limit 80 --image-prefix-mib 32 --max-download-mib 144
python -m tools.dropoff_split_public --cases data/regseg-public/cases.jsonl --out data/regseg-public/splits --development-cases 20 --seed 42 --keep-training-cases data/dropoff-regseg-cli-pilot-20261120/cases.jsonl
```

The optional `--keep-training-cases` prevents scenes and images already used in
an earlier pilot from entering development. Omit it on a fresh project with no
previous pilot. Every input must be a validated public training record; the
existing concise-test scene exclusions and image/mask hashes remain enforced.

The output contains `train.jsonl`, `development.jsonl` and a manifest binding the
source files, split files, case identities and source-scene groups. An identical
image declared under different scenes is rejected. Groups stay together, so the
actual development count can differ from the requested count; both counts are
recorded. No original annotations are rewritten and no status label is inferred.

Image and mask paths are relative references to the original data directory.
When moving the splits, preserve the dataset directory and its `splits`, `images`
and `masks` layout together. Keep the earlier pilot source referenced by the
manifest in a reproducibility package as well. Large data and run outputs remain
outside Git.

The preparation run on 20 November 2026 produced **80 valid public examples**
from **74 source scenes**. One empty destination mask, source record `5361`, was
excluded with its reason and image/mask hashes in the bootstrap receipt. With
the previous ten pilot cases kept in training, the grouped split contains
**60 training cases from 57 scenes** and **20 development cases from 17 scenes**.
The unsplit `cases.jsonl` SHA256 is
`3142a3cca0a44ab61deb7ec5fc5c5e0b440693d94b447eb27f57954b2afae3ed`.
All 80 cases passed the actual pinned Qwen3-VL processor and training-collator
checks. These checks verify preprocessing; model training remains pending.

Validate the complete split locally:

```powershell
python -m tools.dropoff_train validate --config configs/dropoff/training-public-pilot.json --train data/regseg-public/splits/train.jsonl --dev data/regseg-public/splits/development.jsonl
```

On a larger CUDA GPU, install the separate environment described in
[the GPU quickstart](dropoff-cloud-quickstart.md), then run:

```powershell
python -m tools.dropoff_train preflight --config configs/dropoff/training-public-pilot.json --cache-dir models/grounding-cache --out runs/dropoff-public-pilot
python -m tools.dropoff_train train --config configs/dropoff/training-public-pilot.json --train data/regseg-public/splits/train.jsonl --dev data/regseg-public/splits/development.jsonl --cache-dir models/grounding-cache --out runs/dropoff-public-pilot
```

The public pilot keeps the selected **Qwen3-VL-8B-Instruct** model and loads all
60 training and 20 development cases. Its configuration requests 20 optimizer
steps, with evaluation and checkpoint saving every five steps. Twenty steps do
not imply that every training row has been consumed. The existing preflight
floors remain **12 GiB total GPU memory, 10 GiB free GPU memory and 30 GiB free
disk space**; passing them does not guarantee a fit. These floors describe this
resident-model runner. The laptop's 4 GiB GPU uses the separately verified
[streamed local route](dropoff-local-training.md), whose completed public runs
retain their own pinned distribution, configuration and results.

This is a public destination-geometry pilot. It cannot complete the final 8B
adaptation: reviewed interpretation/status cases and independently reserved final
evaluation are still required before final project training and quality claims.
