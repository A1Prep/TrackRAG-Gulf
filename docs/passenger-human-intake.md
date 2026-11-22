# Passenger review and supervised-data handoff

## Arjun's corrected Task 2 handoff

The corrected package records two reviewer confirmations on 50 cases. Four belong to training, two to development and 44 to the reserved evaluation recordings. Another six reserved cases await reconfirmation. All six admitted training/development examples say “Stop beside the car directly ahead.” Their approved response is “Which visible car ahead should I stop beside?” The photographs span six recordings. Each signed model input contains the image and zero detector observations. Reviewers supplied no stopping point or area on these six cases. [Corrected Task 2 package](https://github.com/A1Prep/TrackRAG-Gulf/tree/2d96dcb6afdffb46c5d94d299768fe6a92acc094/Task2_56_Case_Human_Review_Corrected) · [Independent source audit](../experiments/dropoff/passenger-human-six-independent-audit-20261122.json)

To reproduce the clarification experiment, first fetch the pinned commit, then extract the six signed nonfinal rows and their original photographs. These commands run from the repository root with the training environment active and a verified public parent handoff available.

```powershell
git fetch origin Task-2-arjun-part-2
$data = "runs/passenger-task2-reviewed/data"
$parent = "runs/dropoff-handoff-v2-local-20261121"
python -m tools.dropoff_import_task2_review --out $data
python -m tools.dropoff_adapt_local validate --parent-run $parent --config "$parent/inference_config.json" --train "$data/train.jsonl" --development "$data/development.jsonl" --coverage clarification-only
python -m tools.dropoff_adapt_local train --parent-run $parent --config "$parent/inference_config.json" --train "$data/train.jsonl" --development "$data/development.jsonl" --coverage clarification-only --tiled-attention --epochs 2 --accumulation 2 --learning-rate 0.00005 --seed 42 --out runs/passenger-task2-reviewed/child
```

The separate blank forms below cover 36 additional, more varied examples. Two reviewers can use those to add language variation and local passenger-selected stopping areas.

## For the two reviewers

1. Open a blank HTML for each daytime recording in your desktop browser. The training form has 17 cases from `video_054604`. The development form has 19 from `video_054907`. The working laptop has both under `runs/passenger-human-intake-20261122/blank-*.html`.
2. Work independently from the original photo, request and dialogue. For a visible destination, draw the ground region and click a point inside it. For ambiguity, write your own clarification question. Select missing visual evidence if the named place cannot be located.
3. Enter your name or team ID. Choose **Save this review** on each case. Tick the declaration that the answers are your own and choose **Download reviewed cases** before closing. Keep the exported JSON for each recording. A partial export records only the answers you saved. Send your files privately to the data owner.

## For the data owner

To regenerate the two offline forms from prepared requests, run the following from the project root. Each destination must be a new file. You can also supply the EMT directory containing `images/train/` with `--image-root`. The original dataset calls this directory `train` for both recordings.

```powershell
$base = "runs/passenger-references"
python -m tools.dropoff_passenger_references_pack --requests "$base/train-prepared/requests.jsonl" --out blank-train.html
python -m tools.dropoff_passenger_references_pack --partition development --requests "$base/development-prepared/requests.jsonl" --out blank-development.html
```

Create a folder containing only the individual JSON exports you received. Run the intake into a new directory each time. It checks each signed row against the original image, instruction, dialogue and observations. It preserves the exact returned files and every individual answer. Names and notes stay in the local snapshot.

```powershell
$base = "runs/passenger-references"
$intake = @("--train-requests", "$base/train-prepared/requests.jsonl", "--development-requests", "$base/development-prepared/requests.jsonl", "--returns", "runs/passenger-review-returns")
python -m tools.dropoff_passenger_review_intake @intake --out runs/passenger-intake-01
```

Read `receipt.json` for outstanding cases. A single ordinary review and exact agreement between independent reviewers are identified separately. Clarification needs two distinct original reviewers and two people supporting the exact selected answer. Any different status, point, box or question appears in `adjudication-template.json`. A named adjudicator inspects the original case and the individual choices, copies the selected answer's `supervision_sha256` into `chosen_supervision_sha256`, and writes their reason. They enter their name, a timezone-bearing time, and set `human_attested` to `true` themselves. Unresolved rows can stay blank. Keep the original independent returns alongside this file.

```powershell
python -m tools.dropoff_passenger_review_intake @intake --decisions runs/passenger-adjudication.json --out runs/passenger-intake-02
```

The original version 1 training form had no personal-review declaration. Its rows stay pending until the named reviewer confirms their own exact export. The intake supplies a `legacy-confirmations-template.json` identifying the export by its hash. The reviewer enters their timestamp and declaration. Rerun the intake with `--legacy-confirmations <signed-file>`. For several reviewers, the data owner can combine their independently confirmed entries into one sheet and retain each original confirmation.

An available `train.jsonl` or `development.jsonl` contains only the accepted cases from that partition. Compare eligible and expected counts in the receipt before describing coverage. Once the receipt shows `training_loader_validated: true`, training covers all three statuses and the separate development file has eligible labels. Validate the reviewed files against the actual staged parent before deciding whether to train a new adapter.

```powershell
$parentRun = "runs/dropoff-handoff-v2-local-20261121"
python -m tools.dropoff_adapt_local validate --parent-run $parentRun --config "$parentRun/inference_config.json" --train runs/passenger-intake-02/train.jsonl --development runs/passenger-intake-02/development.jsonl
```

Reviewer identity is self-attested. Human annotation concerns the target in these AI-authored requests on real EMT images. The two recordings are three seconds apart. Report continuation and any later evaluation separately from the frozen passenger release. [Continuation validation and training](dropoff-local-adaptation.md)
