# Task 2 — corrected 56-case human review

The original two-person human reviews are confirmed and preserved. Corrections made afterward changed the approval bindings on six cases, so those six—and only those six—need fresh independent confirmation.

## Contents

- `review_cases.jsonl`: 50 unchanged, fully human-reviewed cases with valid current bindings.
- `reconfirmation_cases.jsonl`: six corrected candidates pending two current reviews.
- `reviewer_1.csv` and `reviewer_2.csv`: all 56 current instructions, targets and bindings. The 50 unchanged rows retain their confirmed accepts; six changed rows have blank decisions for reconfirmation.
- `history/`: original confirmed JSONL and reviewer forms.
- `images/`: all 56 unchanged source images with portable relative paths.
- `correction_log.json`: exact before/after instructions, geometry and hashes.
- `split_audit.json`: recording-disjoint partition and reserved-evaluation audit.
- `PUBLIC_DATA_HANDOFF.md`: authoritative clean public-data recipe and exclusion scope.

## Review rules

Inspect the actual image and the passenger's intended **ground stopping area**, not the landmark vehicle's extent. Landmark IDs remain separate metadata.

For each pending row, enter one decision: `accept`, `reject`, `correct`, or `unavailable` (only for a missing image). Supply a real reviewer identity, a timezone-aware ISO-8601 timestamp and useful notes. The two reviewers must be distinct.

For `decision=correct`:

- `proposed`: corrected point and ground-area box, empty corrected question.
- `needs_clarification`: empty corrected point and box, nonempty useful question.
- `insufficient_visual_evidence`: all corrected target fields empty.

Coordinates use original-image pixels. A proposed point must lie inside its box. Do not edit current target columns or SHA256 binding columns.

## Merge returned forms

From the extracted package directory:

```powershell
python tools/dropoff_resolve_reviews.py --reviewer-1 reviewer_1.csv --reviewer-2 reviewer_2.csv --output-dir resolved
```

The merger verifies current bindings, reports pending/disagreement/unavailable rows, and writes any mutually approved cases to `resolved/approved_cases.jsonl`. Six pending rows prevent a complete 56-case release until returned.
