# Review passenger instructions with the saved detections

## For each reviewer

1. Extract the whole private ZIP into a folder of your own. Open **review.html** and choose your assigned form, **Reviewer 1** or **Reviewer 2**. Work independently. Each case contains the complete passenger instruction and dialogue, the original photograph, and every detection from the saved model on that frame. Toggle the boxes and IDs, or expand the detection table. The amber rectangle and dot show the candidate inherited from the earlier image-only review. Some candidates instead contain a question or record insufficient evidence.
2. Choose **Accept**, **Correct**, or **Reject**, enter your own reviewer ID, and press **Save this case**. When correcting a proposal, enter a point and rectangle in original-image pixels. Click the photo to read coordinates. When correcting a question, enter the words you would ask. **Export CSV** downloads the whole form, including any pending cases. Send this file privately to the coordinator. Keep a copy before closing or moving the folder. The blank reviewer CSV is also editable directly if necessary. Leave the first 14 columns and all three legacy columns unchanged.

The earlier two reviewers saw an empty detection list. The model's saved detections and reconstructed timestamp are now included in each input. The box list and input hashes identify exactly what you reviewed. All 44 new decisions start pending.

## For the coordinator

Keep the original ZIP. Store each returned CSV privately outside its extracted folder. Use one reviewer ID per form. The two people must have different IDs even after accounting for letter case, equivalent Unicode characters and extra spaces. From the extracted folder, use Python 3.10 or newer to check the original images and every return binding, and then merge the two forms with the bundled, unchanged Task 2 resolver.

```text
python resolve_context.py --reviewer-1 "path/to/reviewer_1_return.csv" --reviewer-2 "path/to/reviewer_2_return.csv" --output "../context-resolution-01"
```

Choose a fresh output folder for every attempt. Read **review_status.json** for the counts. Matching accepts and identical valid corrections from two different reviewers produce **approved_cases.jsonl**. Rejected cases, disagreements, pending rows and unavailable cases have separate resolver outputs. Keep the extracted original image folder in place because approved records use paths relative to the resolution folder. To transfer approved records, preserve the relative layout of these folders.

The signed coarse rectangle lets the team measure whether a predicted point lies inside it and how much a predicted rectangle overlaps it. Formal clear-case scoring requires a separately reviewed human-drawn PNG mask tied to the original photograph. For ambiguous cases, collect the passenger’s own intended destination and reply separately. This packet contains no masks or passenger returns. It covers the 44 cases with an earlier completed review.

## Rebuild

Follow the [evaluator restore instructions](passenger-evaluator-handoff.md#restore-the-published-inputs-on-another-checkout) to download the published input archives and restore the original photographs into `runs/evaluator-inputs`. From the project root, build the new private packet in a fresh Git-ignored directory. The command checks the Task 2 source, both earlier signatures, all saved Task 1 scenes and 494 current-frame observations. It runs the unchanged resolver on the two blank forms, then writes **packet/review.html**, **passenger-context-review-private.zip**, an aggregate **readiness.json**, and a private blank-resolution check.

```text
python -m tools.passenger_context_reviews --installed runs/evaluator-inputs --output runs/passenger-context-review
```

Only the aggregate readiness receipt belongs in the project evidence. The candidate JSONL, images, per-case labels, working forms, returns and resolved outputs stay private.
