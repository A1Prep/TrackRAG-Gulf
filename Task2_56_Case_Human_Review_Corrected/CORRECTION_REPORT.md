# Correction report

## Applied fixes

- Rechecked all 39 proposed destinations against the actual images.
- Moved four vehicle-overlapping boxes onto visible ground: `local-eval-video_155425-00871`, `local-eval-video_155425-01171`, `local-eval-video_160325-00781`, and `local-eval-video_160325-00961`.
- Rewrote the primary passenger requests for `local-eval-video_204647-00961` and `local-eval-video_122233-00301`.
- Rewrote ten non-passenger paraphrase sets and aligned correction-case landmark roles with the final requested landmark. Landmark IDs remain separate from destination geometry.
- Preserved every original recording ID, frame ID, image, partition, outcome, and confirmed historical review.

## Review state

Fifty cases retain unchanged instruction and supervision hashes and therefore retain their confirmed two-person approvals. Six cases changed the instruction or target after review. They are isolated in `reconfirmation_cases.jsonl`; their old approvals are preserved in `history/`, but are not misrepresented as approvals of the changed content.

Run the merger after both reviewers complete the six blank rows. It verifies every current binding and writes mutually approved cases to a portable JSONL file.

## Public data and repository

No public combined export is bundled with this local review package. `PUBLIC_DATA_HANDOFF.md` pins the existing audited 500/125 public recipe, which excludes the 338 source-validation rows, the 170 concise-test training IDs, and all 58 reserved source scenes. The public source could not be downloaded again in this environment, so no substitute public labels were created.

Current repository `main` already uses `trackrag/dropoff/` without a competing `trackrag/dropoff.py`. Local recording partitions are disjoint, and all test recordings remain reserved.

## Verification

`validation_report.json` records structural, image, hash, review-form, manifest, and split checks. To repeat them from the extracted package:

```powershell
python tools/validate_package.py
```
