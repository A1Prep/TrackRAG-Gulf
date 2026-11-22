# Road behind two neighboring vehicles

Two vehicles can be visibly distinct even when their detection rectangles overlap slightly. The released pair resolver asks for clarification whenever the rectangles overlap. An optional experiment handles a passenger who explicitly describes each vehicle and asks for the road just behind the gap. The original three-turn black-car study uses the released resolver. [Development protocol](../experiments/dropoff/passenger-occluded-pair-protocol-20261122.json)

The first complete Qwen answer and two separate Qwen answers about the passenger’s quoted descriptions must identify the same two current tracks. The experimental route reopens the original photograph, checks the detector boxes, and verifies the recorded model, adapter and worker identities. It considers only slight horizontal overlap between objects at compatible image heights. A third vehicle between them at comparable height makes the route abstain.

The original fitted two-vehicle head ranks road locations. For overlapping boxes, its input expresses the interval between their projected centers while preserving the observed vehicle sizes and bottoms. This extrapolates beyond the 110 separated vehicle pairs used to fit the head. Every cell of the proposed rectangle must lie on predicted road within the learned region, below both vehicles, between their centers, and outside the current detection rectangles. “Behind” here is the passenger-facing side of the vehicles in the forward camera view.

A further Qwen pass sees the original scene with enlarged crops of the selected objects. It must return `MATCH` for the exact instruction and rendered crops before a new location can be offered. The same model can still misidentify the vehicles on multiple passes. Only the passenger can confirm the proposed area. Neither the road segmentation nor the confirmation measures legal stopping, traffic clearance or driving safety.

## Development checks

All 31 original answers from two previously recorded local runs kept their semantic outcomes. Independently enumerating every current motor-vehicle pair across five other daytime source frames from two EMT recordings tested 1,122 potential pairs. Four met the geometry conditions. Three also contained a contiguous road patch within the learned region. None of the enumerated pairs supplied a passenger intention or a new Qwen identification. [Complete input hashes, outcomes and geometry checks](../experiments/dropoff/passenger-occluded-pair-controls-20261122.json)

## Running the optional check

Use the existing public-trained application configuration with its installed adapter path and the matching release geometry. The road cache must contain the segmentation for that original photograph. The CPU command below preserves the original model answer and reports whether a fresh crop check is warranted.

```sh
python -m tools.dropoff_projected_pair --config <installed-grounding-config.json> --geometry models/passenger-feature/geometry.json --road-cache <segformer-cache> --out <new-result.json> turn --journal <session.jsonl> --request-id <submitted-request-id>
```

Add `--verify-with-local-model` after the request ID to run the eligible crop check on the local 8B model. Give each invocation a new output path. The original journal stays available and the new record contains the original outcome digest. An application experiment can explicitly wrap its normal model in `ProjectedPairGrounder(model)` from `trackrag.dropoff.projected_pair` and use the existing `PassengerSession`. In that route a proposal from a matching crop still needs a separate passenger confirmation. The default web application and the sealed study retain their original grounders.

## When the passenger identifies the vehicles

The passenger app lets the rider select two current detected vehicles on the original photograph. They then choose **Between** or **Just behind**. The learned two-vehicle model estimates a road patch and the rider can confirm, change the pair, or cancel. For overlapping vehicles, **Between** asks for another description. **Just behind** can offer a road patch when the vehicles meet the projected-depth checks. Changing a selection invalidates the previous confirmation button. The journal preserves the original automatic answer, the selected source vehicles, the calculated road area and the explicit confirmation as separate events.

On the previously registered frame 781 example, the automatic system selected the wrong vehicles. A scripted operator then chose the two intended sedans. **Between** correctly asked for another choice because their boxes overlap. **Just behind** produced (1097.5, 1132.5) inside the registered road polygon. A second browser case in the same recording produced (1037.5, 1292.5) after the operator selected its vehicles. A further CPU test included a different recording and cross-recording track IDs. Across its eight choices the system made one offer, asked six times and rejected one invalid pair. Both studies used scripted inputs. No rider supplied or confirmed a destination in either study. [Browser protocol](../experiments/dropoff/passenger-two-vehicle-ui-protocol-20261122.json) · [Two-recording CPU results](../experiments/dropoff/passenger-assisted-pair-controls-20261122.json)

The following command reopens the original image, verifies the detected vehicles, loads the fitted two-vehicle coefficients and checks the computed point and complete rectangle against the current saved segmentation. It also replays any automatic pair-identification question using the matching referring-object ranker. A journal-only export explicitly identifies which mask evidence was copied from the original event.

```sh
python -m tools.dropoff_passenger_handoff session --journal <session.jsonl> --referent models/passenger-feature/referent.json --geometry models/passenger-feature/geometry.json --road-cache <segformer-cache> --out assisted-handoff.json
```
