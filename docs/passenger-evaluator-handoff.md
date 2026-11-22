# Passenger grounding handoff to Task 1

## Preparing compatible scenes

Use the same original images, prediction folder and vision profile for the model requests and the evaluator scenes. Keep the source folder in the run record. On the daytime development recording, all 19 requests from the team’s shared prediction ZIP matched the 19 scenes exported from that ZIP. All 19 earlier development requests matched scenes exported from the fresh development detector run. The cross-source comparison matched zero of 19. Later frames had different track identities as well as small differences in coordinates. [Measured source comparisons](../experiments/dropoff/passenger-task1-evaluator-handoff-20261121.json)

Task 2 supplies the final case IDs, instructions and reviewed destination masks. Task 3 supplies the intended regions and scripted replies for ambiguous cases before model responses are collected. Task 1 freezes the protocol, reviewer records and source groups. The final five recordings stay outside Task 5 development. [Task 1 evaluation instructions](task1-grounding-evaluation.md)

## Native direct interpretation

The native route runs the full image, predicted observations and passenger dialogue through Qwen, then converts the four-field response into original-frame pixels. Keep the complete output directory from `python -m tools.dropoff_predict predict`. Task 1’s `python -m evaluation.grounding_inputs task5-run` reads its `run.json`, `resolved-requests.jsonl`, `model-outcomes.jsonl` and `receipts.jsonl`. Give Task 1 the byte-identical configurations, requests and results for both the pretrained and trained conditions.

The original native compatibility run contained four EMT instructions and one public RegSeg control. Task 1’s unmodified field converter consumed all five native envelopes. Its scene checker matched all four EMT inputs against their actual source. The public control had no EMT scene. No final reference file or assessment score was produced during this exercise. The upstream software suite also passed 80 evaluator and converter tests. [Recorded handoff](../experiments/dropoff/passenger-task1-evaluator-handoff-20261121.json)

The experimental native child adapter is separately packaged. Extract `native-direct-8b-v3-verified.zip` into `models/passenger-ai-direct/` under the project. Its `inference_config.json` specifies two resident decoder layers for the 4 GB laptop. The package includes the checksum of each member and evidence from a fresh GPU inference. The pinned 8B base and processor cache use the project’s existing local model setup. [Verified adapter and reload](../experiments/dropoff/passenger-native-child-reload-20261121.json)

```sh
python -m tools.dropoff_predict predict --config models/passenger-ai-direct/inference_config.json --requests requests.jsonl --out native-results
```

## Composite location proposals and passenger corrections

The interactive route uses Qwen to identify a landmark, a separately learned estimator to locate a potential destination, and a road model to check the visible surface. A passenger can clarify, mark a different area or confirm the current proposal. Its audit export labels those events separately and preserves the original model output, selected source, geometry, road evidence and file hashes. The current Task 1 protocol lists native direct and clarification conditions. A composite study needs its own declared condition and selection procedure.

```sh
python -m tools.dropoff_passenger_handoff batch --run batch-run --requests requests.jsonl --replay raw-landmarks.jsonl --out composite-evidence.json
python -m tools.dropoff_passenger_handoff session --journal session.jsonl --out interaction-evidence.json
```

Omit `--replay` for a live batch. For a saved-landmark batch, supply the original raw decision file. For a voice session, keep the original WAV in the session’s `audio/` directory. These examples write separate evidence files. The exported JSON can include utterances, local paths, model prompts and timestamps.
