# Public training-data handoff

This local review ZIP intentionally contains **no combined public training export**. It therefore adds none of the 338 source-validation rows and none of the 170 concise-test training IDs to the local corpus.

Use the existing audited public recipe from the repository root; do not substitute the rejected ten-example Task 2 public handoff or any source-validation rows:

```powershell
python -m tools.dropoff_expand_regseg --out data/dropoff-public-expanded-20261121 --limit 625 --image-prefix-mib 192 --previous-locks configs/dropoff/public-historical-split-locks.json
```

The accepted corpus is 500 training examples from 295 source scenes and 125 development examples from 65 different source scenes. All 58 reserved concise-test source scenes are excluded. Fit only from `data/dropoff-public-expanded-20261121/splits/train.jsonl`; `cases.jsonl` is the partitioned master, not a train-only file.

The public source download could not be repeated in the correction environment because its HTTPS connections were terminated. No replacement public labels or counts have been fabricated here. The repository's pinned expansion command and committed lock registry remain the authoritative route.

Local `video_122233` and every other test recording stay reserved for independent evaluation. Local recording splits are disjoint, as recorded in `split_audit.json`.
