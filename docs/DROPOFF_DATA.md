# Task 2 drop-off data release

## Release status

The local release contains **115 draft cases**: 45 train, 20 dev, and 50 eval.
It is not release-final until the required independent human reviews are complete.
Self-authoring and AI-assisted visual inspection do not count as independent review.

The small public RegSeg annotation files are included for reproducibility. Actual
public image/mask validation requires the separately downloaded dataset archive;
bulk images and masks must remain outside Git.

## Public source audit and discrepancy

Source: https://github.com/kanji95/Talk2car-Refseg

Pinned repository revision: `cebce1943ce373857cc9be5def68ded4008c29e1`.

The supplied availability audit reports 170 train + 179 val lines (349 total).
The pinned public files currently contain **170 train + 338 val lines (508 total)**.
`public_source/source_receipt.json` records both counts and file hashes. The loader
never pads, duplicates, or silently truncates examples to match an expected count.

RegSeg masks are navigable-region supervision. They do not automatically label a
drop-off point, anchor identity, ambiguity, or dialogue response. Every public case
is tagged `rnr_region`; consumers can filter on that tag when necessary.

## Public split and scene policy

All public examples are assigned to **train** only. Local EMT cases alone provide
dev and eval. RegSeg has no public `image_id -> scene_id` mapping in this release,
so the reserved 58-scene assertion is reported as `not_enforceable`; it is not a
passing scene-disjointness check. Obtain the original Talk2Car commands mapping or
the team's pinned scene audit before making source-scene-independent claims.

Public data has one instruction per image. Paraphrase grouping applies to local
cases only. Local `source_group_id` is the recording ID, keeping adjacent frames,
paraphrases, and dialogue variants from one recording in the same split.

## Mask and coordinate rules

For every public example, the loader decodes the actual image and mask, requires
identical dimensions, and requires a nonempty foreground mask.

- Target: foreground pixel with maximum OpenCV L2 distance-transform clearance.
- Ties: first NumPy flattened `argmax` index, making selection deterministic.
- Coarse box: `[xmin, ymin, xmax + 1, ymax + 1]`, using half-open pixel bounds.
- Tests include a non-convex U-shaped mask whose naive centroid is not a valid target.

Invalid images, masks, joins, dimensions, and empty masks are excluded with explicit
JSON receipts. A receipt is not a valid training example.

## Build commands

Run from the repository root. Activate the project environment first.

Audit the small public annotation files:

```bash
PYTHONPATH=. python tools/dropoff_prepare_data.py source-receipt \
  --annotation-dir annotations/dropoff/public_source \
  --commit cebce1943ce373857cc9be5def68ded4008c29e1 \
  --output annotations/dropoff/public_source/source_receipt.json
```

Build the local draft release without image-path validation:

```bash
PYTHONPATH=. python tools/dropoff_prepare_data.py local \
  --annotation-root annotations/dropoff/local \
  --output-dir release_task2
```

Add `--emt-root "C:\path\to\EMT"` to decode and validate every local image.

After downloading and extracting RegSeg, build the combined release:

```bash
PYTHONPATH=. python tools/dropoff_prepare_data.py release \
  --data-root "C:\path\to\Talk2Car-RefSeg" \
  --public-annotation-dir annotations/dropoff/public_source \
  --local-annotation-root annotations/dropoff/local \
  --emt-root "C:\path\to\EMT" \
  --output-dir release_task2
```

The RegSeg root must contain `imgs/img_<source_split>_<image_id>.jpg` and
`val_masks_new/gt_img_ann_<source_split>_<image_id>.png`.

## Outputs

- `splits/dropoff_manifest.json`: cases, counts, limitations, and review status.
- `annotations/dropoff/model_inputs/<split>.json`: case ID, image, instruction only.
- `annotations/dropoff/targets/<split>.json`: separately stored gold supervision.
- `annotations/dropoff/*_exclusions.json`: explicit invalid-input receipts.
- `reviews/reviewer_1.csv` and `reviewer_2.csv`: independent human review forms.
- `reviews/review_status.json`: initially `pending`; never pre-marked as reviewed.

All eval cases and all ambiguous local cases require two independent human reviews.
Reviewers must not be the same person. Merge completed forms with:

```bash
PYTHONPATH=. python tools/dropoff_resolve_reviews.py \
  --reviewer-1 reviews/reviewer_1.csv \
  --reviewer-2 reviews/reviewer_2.csv \
  --reviewer-1-id "Reviewer A" \
  --reviewer-2-id "Reviewer B" \
  --output-dir reviews/merged
```

Each reviewer chooses `accept`, `reject`, or `correct`. A correction must supply
`track:<track_id>`. Different outcomes are written to `disagreements.csv` for
resolution rather than being silently averaged or overwritten.

## Tests

```bash
python -m unittest tests.test_dropoff -v
```
