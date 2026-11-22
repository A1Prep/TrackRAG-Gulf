# Passenger instruction development evidence

The first bank contains 36 AI-authored passenger requests on original EMT images. It provides concrete examples of choosing a place, resolving a reference, handling a missing landmark and correcting an earlier instruction. The instructions and reference answers were frozen before the new models produced outputs. A second AI reviewer worked from the same originals, instructions and prior dialogue, without access to either the first answers or model predictions.

| Bank | Recording | Original frames | Requests | Intended proposals | Clarifications | Missing visual evidence |
|---|---|---:|---:|---:|---:|---:|
| Train diagnostic | video_054604 | 4 | 17 | 10 | 3 | 4 |
| Daytime development | video_054907 | 4 | 19 | 11 | 4 | 4 |

These recordings are three seconds apart. They have distinct original frames and their shared route limits conclusions about generalization. Each proposed reference has a point and a small plausible rectangle in original 2560 × 1440 coordinates. The two AI readings agreed on status in 36 of 36 cases. The second reviewer’s point landed in the first rectangle in 12 of 21 proposed cases. The median separation was 51.61 original pixels. A narrow rectangle is one interpretation of where the passenger meant. These recorded highway examples assess image interpretation. They establish neither a legal stopping location nor physical clearance.

`*-instructions.jsonl` contains the utterance and any previously scripted dialogue. Keep `*-references.jsonl`, later scripted replies and the two reviews outside model inference. Each portable hash in `input-bindings.jsonl` binds the full instruction, dialogue, exact current frame and its selected observations. Original image bytes were checked against the Task 1 frame manifest at commit `709e969260866a5476d8c199609c59de580589f7`. The daytime source observations were independently reconstructed from the complete fine-tuned Task 1 track artifacts. Human review of the local EMT cases is still pending.

## Complete an independent human review

Open the blank review HTML in a desktop browser. Enter your name or team ID, read the dialogue and choose a response for each frame. For a proposal, drag a box on a plausible ground patch and click its intended point. For ambiguity, write the question you would ask. Choose **Save this review** for each case. Check the personal-review declaration and use **Download reviewed cases** before closing the page. Each reviewer starts with a blank copy and returns their own JSON. The export captures their original-frame geometry, inputs, timestamps and notes.

To build the training form from a local copy of the EMT images, run this command from the repository root. The folder passed to `--image-root` contains `images/train/` with `video_054604`. The optional `--partition development` selects the 19 cases from `video_054907`.

```sh
python -m tools.dropoff_passenger_references_pack --image-root data/task1-image-only-20261119 --out train-human-review.html
```

Each offline form embeds four verified originals and its own frozen instructions. On the project laptop both forms are available in `runs/passenger-human-intake-20261122`. The intake archives each return and records exactly which answers a named person selected when reviewers disagree. [Teammate instructions and supervised-data handoff](../../../docs/passenger-human-intake.md)

## Reproduce scoring

Score the **entire** daytime request bank, retaining missing, invalid, failed, misbound and unresolved outputs in the relevant denominators.

```sh
python -m tools.dropoff_passenger_references --bank annotations/dropoff/passenger-development --partition development --requests runs/passenger-references/development-prepared/requests.jsonl --outcomes MODEL_OUTCOMES.jsonl --out daytime-scores.json
```

The separate public diagnostic contains 16 source-disjoint Talk2Car-RegSeg examples with externally annotated destination masks. Each case comes from a different original source scene. The instructions have no detector boxes or state labels. The model receives only its image and original linguistic command. Use `public-mask-evidence/` for the predeclared selection, provenance and scorer checks. The local data package is `runs/passenger-references/public-independent/public-mask-diagnostic.zip`. After extracting it, score a saved run independently.

```sh
python -m tools.dropoff_passenger_references_public --corpus UNPACKED_DIAGNOSTIC --outcomes MODEL_OUTCOMES.jsonl --out public-mask-scores.json
```

The public scorer reports original point-in-mask agreement and coarse box IoU, with errors and missing proposals retained as zero geometry. The source authors drew their masks manually and used a separate student-review process. Their task labels a navigable region and supplies no gold confirmation or clarification decisions. [Rufus et al., IROS 2021, section III.A](https://arxiv.org/abs/2112.13031)
