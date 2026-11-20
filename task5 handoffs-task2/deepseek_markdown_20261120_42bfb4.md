# Field provenance — Task 2 grounding dataset

Each `GroundingCase` records per-field provenance in its `field_provenance` dict.
Two dicts are defined in `trackrag/grounding.py`:

- `PROVENANCE_PUBLIC` — for Talk2Car-RegSeg cases
- `PROVENANCE_LOCAL` — for EMT-authored cases

## Public source: Talk2Car-RegSeg

| Field | Provenance | Note |
|---|---|---|
| `instruction` | **gold** | Written by Talk2Car annotators |
| `destination.mask` | **gold** | Human-segmented navigable region (RegSeg release) |
| `destination.point` | derived | Interior pixel via `cv2.distanceTransform` — not the mask centroid |
| `destination.box` | derived | Tight bbox of the mask; exists for consumers that cannot take a mask |
| `landmarks` | **unlabelled** | RegSeg does NOT label anchor identity or relations — downstream must not treat `landmarks: []` as "there are none" |
| `difficulty_tags` | derived | Rule-based: `rnr_region` for all; no ambiguity/missing-reference labels in RegSeg |
| `paraphrases` | **absent** | RegSeg ships one instruction per image; no paraphrase grouping |
| `scene_id` | **unavailable** | RegSeg does not ship `scene_token`; `scene_id=None` on all 508 cases |

## Local source: EMT-authored cases

| Field | Provenance | Note |
|---|---|---|
| `instruction` | authored | By `author_case.py` / `author_case_batch.py`; reviewer recorded in `reviewed_by` |
| `destination.mask` | authored | Circle mask (radius chosen by author) around the chosen point |
| `destination.point` | authored | Interior-corrected via `interior_point()` |
| `destination.box` | derived | Tight bbox of the authored mask |
| `landmarks` | authored | Explicit `track_id` + `relation` + `role` |
| `difficulty_tags` | authored | From the controlled vocabulary in the authoring tool |
| `paraphrases` | authored | Pipe-separated list; grouped via `parent_case_id` |

## Model inputs vs gold supervision

Training exports strip gold fields:

- Public: keep `instruction`, `image_path`, `destination.mask`, `destination.point`, `destination.box`. Drop `landmarks` (unlabelled) and `field_provenance` internals.
- Local: keep `instruction`, `image_path`, `landmarks`, `destination.mask`, `destination.point`, `destination.box`. Drop `reviewed_by`, `review_date`, `field_provenance` internals.

Eval exports are the same shape — labels are separate from inputs.

## Known limitations

1. No scene map for public data. All 508 public cases have `scene_id=None`; `unique_scenes` in summary reports `1`. **Consequence**: no scene-disjointness claims on public dev. Mitigation: public cases all go to `train`; dev/eval use local EMT exclusively.
2. No paraphrase grouping in public data. `paraphrases: []` for all public cases.
3. The 58-reserved-training-scenes audit referenced in the task spec is **not available on disk** — see `docs/data_access.md` for how this affects the dataset.
4. Non-navigation commands (e.g. "slow down behind the truck") are kept with `difficulty_tags: ["rnr_region"]`. They are honest supervision for navigable-region tasks but should not be evaluated as drop-off/destination predictions.

## Generation

- Provenance dicts are set by `trackrag/talk2car_adapter.py` (public) and `scripts/author_case.py` (local).
- To check any case's provenance: see `docs/data_access.md#checking-provenance` or run the one-liner in this file's "How to verify".