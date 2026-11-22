# Collect passenger intentions and replies before showing a model proposal

Task 3 needs the passenger’s own clarification and correction. Task 1 needs their intended region for eligible ambiguous cases. A participant sees the original photograph, instruction and any earlier dialogue. They draw the place they mean or indicate a missing landmark, then enter both replies in their own words. They can leave an undecidable case blank. Their submission is a named or coded self-attestation. Formal Task 2 review and the final scoring protocol remain separate. [Independent evaluation contract](task1-grounding-evaluation.md)

## Open the private form

On the project laptop, open `runs/passenger-participant-study-20261122/final-form/review.html`. Keep the `images` folder and `blank-receipt.json` beside it. Another participant can open a private copy offline. Give model developers the aggregate return count while the evaluation remains blind. Keep photos, individual participant exports, replies and masks in private evaluation storage.

The private team handoff is `runs/passenger-participant-study-20261122/TrackRAG-private-passenger-form.zip`. Extract the whole ZIP into one folder, then open `review.html`. The public readiness record contains its SHA-256 and the results of checking all 44 extracted originals. [Image-free readiness record](../experiments/dropoff/passenger-participant-readiness-20261122.json)

1. Enter your own study code and click **Start or resume**.
2. Read the words and look at the picture. If you have a particular place in mind, choose **I can show my intended area**. Click around the boundary, click **Finish outline**, then click your chosen stopping point inside it. If the landmark is absent, use **I cannot see the landmark I mean**.
3. Write what you would say when asked to clarify, and what you would say if the assistant selected an incorrect place.
4. Click **Save this case**. The scene menu shows which cases you have saved. Tick the declaration and click **Download reviewed cases** to keep a partial or complete return. The export contains your code and answers.
5. Resume with the same code in the same browser, or use **Open a previous export** on another device. Return the downloaded JSON privately to the independent evaluation owner.

## Build or verify a copy

Use Python 3.11 and the project’s reviewer/integration dependencies. Follow the [evaluator restore instructions](passenger-evaluator-handoff.md#restore-the-published-inputs-on-another-checkout) to download the input files and restore the original JPEGs from Task 2’s exact Git revision. The resulting `inputs` directory contains `requests-with-saved-tracks.jsonl` beside `source/images/test`. The pinned input covers 44 cases on five reserved recordings. Six cases awaiting reconfirmation are outside this form. If your team already has a private form folder, the `verify` command checks the 44 original JPEG hashes and dimensions, the display inputs and the HTML against this checkout. Verification is possible without access to private case targets.

~~~~powershell
$py = ".venv/Scripts/python.exe"
& $py -m tools.passenger_participant_study prepare --requests runs/evaluator-inputs/inputs/requests-with-saved-tracks.jsonl --out runs/my-private-passenger-form
& $py -m tools.passenger_participant_study verify --form-dir runs/my-private-passenger-form
~~~~

The independent owner can reconstruct the earlier input-only handoff with `python -m tools.dropoff_final_input_inventory --out <new-private-input-folder>` after installing the required private detector artifacts and original-image sources for that existing tool. The participant collector itself opens only the pinned saved-tracks input request file and original images. Participants receive the text and photographs. Their responses establish their intentions. Task 2 handles separate reviewer confirmation of any changed detector information.

## Import responses for the evaluator

Choose a new output directory for each snapshot. Pass each supplied export with `--return`. Identical copies of a saved case are deduplicated across partial exports. Conflicting responses are refused. Preserve both originals and have the independent evaluator apply the participant assignment recorded before they saw model results.

~~~~powershell
& $py -m tools.passenger_participant_study intake --return "private/participant-a.json" --return "private/participant-b.json" --out runs/private-passenger-returns
~~~~

The import preserves both the original self-attestation and the individual case responses, including source, review time, original image, instruction and request hashes. Each exported PNG is a filled polygon at original resolution. Pixel coordinates start in the top-left. Its bounding box is the exact painted pixel extent with exclusive right and bottom edges. The point comes directly from the participant and must lie inside the PNG. The importer uses the evaluator’s mask reader and box, point and reply checks.

| File | Use |
|---|---|
| `replies.jsonl` | One `answer` and one `rejection` per completed case, using the participant’s exact submitted wording. The evaluator can freeze them before model trials. |
| `participant-responses.jsonl` and `participant-attestations.jsonl` | Private original words, polygon vertices, point, participant code, times, declaration and SHA-256 links back to the supplied files. |
| `masks/` | Hash-bound PNGs made from actual drawn vertices. A missing landmark contributes no region. |
| `task2-region-candidates.jsonl` | Candidate boxes, points and masks for Task 2. Two independent reviewers must assess these references and resolve disagreements. The importer records zero approvals. |
| `receipt.json` | How many participants, cases, replies, masks and eligible intentions were received, plus hashes. |

Only the independent evaluator should supply `--unresolved-ids`. The file has one JSON object per line with the single key `case_id`, listing the cases that evaluator has already classified as needing clarification. Leave this argument out while that list is unavailable. The following example uses a filename that must be supplied by the evaluator.

~~~~powershell
& $py -m tools.passenger_participant_study intake --return "private/participant-a.json" --unresolved-ids "private/evaluator-unresolved-ids.jsonl" --out runs/private-passenger-returns-with-intentions
~~~~

Only listed cases with genuine participant returns appear in `intended.jsonl`. A participant-drawn region supplies the ambiguous case’s intended-mask path and SHA-256. An explicitly missing landmark supplies the `missing_reference` category and null mask fields. Pending designated cases appear in the intake receipt. Task 1 can use the resulting `intended.jsonl` with `evaluation.grounding_inputs task5-cases` and `replies.jsonl` with its protocol freezer once the independently reviewed Task 2 cases and scenes are ready. [Evaluator input formats](../evaluation/grounding_inputs.py)
