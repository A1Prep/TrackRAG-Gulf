# Night passenger instructions

These 20 commands refer to six original frames from EMT recording `video_174340`. The recording is separate from both daytime development recordings. One AI reviewer wrote the commands, provisional destination rectangles and reference decisions after reading the original scenes. A second AI reviewer received the same images and instructions without access to the first references or model predictions. Both reviewers agreed on all 20 decisions. Eleven of the second reviewer’s 12 proposed points landed inside the first reviewer’s rectangle. No human passenger supplied these intentions. [Reference agreement](independent-reference-agreement.json)

The original references, frames and saved Task 1 predictions were bound by hash before model evaluation. The input-only requests were released after the model settings were frozen. Outcome scoring began after all 72 night and public prediction attempts had been recorded. [Original freeze](freeze.json)

Run the independent scorer with the frozen requests and a completed prediction file. The original images and saved predictions referenced by the requests stay in the local project.

```sh
python -m tools.dropoff_passenger_references --bank annotations/dropoff/passenger-development/night --partition night --requests runs/passenger-references/night-inputs/requests.jsonl --outcomes MODEL_OUTCOMES.jsonl --out night-scores.json
```
