# Passenger drop-off results on UAE driver-view photos

The default passenger app asked for clarification on all 15 requests about a gap between vehicles. Seven scenes warranted that response. Eight had identifiable pavement. A separate Qwen visual selector identified three of those eight gaps and also proposed points on two of the seven negative scenes. The airport photograph was excluded.

## What the app checks

The app takes the original frame, the current RF-DETR/BoT-SORT observations and the passenger's words. It checks that the two referenced vehicles are distinguishable. A potentially admissible region needs a patch of predicted road between the two vehicle ground contacts, a visible road-edge cue and no overlapping current road-user detection. The result is a proposed 2-D location and a request for confirmation. Ambiguous vehicle identities or insufficient evidence lead to clarification. Selecting the two outlined vehicles is a separate passenger action.

The default mode is enabled by the normal release command. The optional parking-row experiment can also inspect class-1 sidewalk predictions from the same pinned SegFormer checkpoint and visible parking paint. A unique qualifying pair has a distinct decision record and can be reproduced from the original scene. To reproduce the experiment in the app, add the following option to the usual launch command.

~~~powershell
python -m tools.passenger_release start --between-road-mode edge-or-bay
~~~

The parking-row mode remains experimental. After two simulated passenger selections it offered a point on a scene containing a raised separator. A painted boundary can belong to an island. The primary app uses the more conservative edge check.

## What was measured

The trial used 17 original photographs from 16 UAE camera recordings or street-image sequences. Five were EMT frames and twelve came from KartaView. Fifteen requests described a gap between vehicles. Two additional requests referred to a bin or a traffic-island sign. Each model received one current frame. Real Task 1 predictions and SegFormer probabilities were computed from that photo. [EMT dataset](https://huggingface.co/datasets/KuAvLab/EMT) · [EMT collection paper](https://arxiv.org/html/2502.19260v1#S3.SS1) · [KartaView](https://kartaview.org/)

Independent AI reviewers marked a conservative polygon of visible pavement or recorded that the request should trigger clarification. These references were frozen separately from the model inputs. The scores below use a point inside the reference polygon and a valid clarification on a negative scene. A blocked bay, an operational error and an incorrect point are separately recorded. One positive was a narrow visible strip. Physical vehicle fit, curb access and permission to stop were outside the evaluation.

| First response on the 15 vehicle-gap requests | Correct points among 8 visible gaps | Correct clarifications among 7 negative scenes | False points on negative scenes |
|---|---:|---:|---:|
| Previous passenger app | 0 | 5 | 0 |
| Updated passenger app, default road-edge check | 0 | 7 | 0 |
| Optional edge and parking-bay check | 0 | 7 | 0 |
| Fine-tuned Qwen3-VL-8B directly predicting coordinates | 0 | 2 | 5 |
| Qwen3.5-9B directly predicting coordinates | 1 | 3 | 4 |
| Qwen3.5-9B choosing from visible road points or asking | 3 | 5 | 2 |

The previous app returned four errors across the 15 requests. The seven correct clarifications from the updated app were genuine first-turn outcomes. The final row uses actual unassisted visual model choices. The generator enumerates plausible vehicle pairs, excludes detector-occupied pixels and puts one candidate at the centre of the available road area within each retained pair. The visual model can choose a letter or ask. This is an offline comparison. The passenger app retains its default road check.

To reproduce the candidate points, use the annotation-free current-frame requests and the cache containing the segmentation masks for those exact images. The command exports the marked photos, coordinates and selector prompt. For pixel-identical labels, pass the original font with `--font-file`.

~~~powershell
python -m tools.passenger_road_points --requests requests.jsonl --road-cache path/to/cache --out path/to/new-output
~~~

Thinking mode was tested on nine photos using the same Qwen3.5-9B revision. Six responses produced no usable final answer within the 1,800-token budget. The three returned coordinates fell outside their reference areas.

Four confirmation photographs came from additional street-image sequences. Three contained a reference gap, and one contained a raised island. The visible-road-point selector located one of the three gaps and requested clarification on the island. Developers knew the broad composition of this small cohort and had heard one positive described before the final run. Its reference regions were kept out of model inference. The two separate EMT landmark controls both received an app point outside their reference polygons.

## What failed

On five development street positives the detector found both target vehicles and SegFormer classified about 97–100% of the reference pixels as road. The initial lateral-edge test still rejected all five, including after simulated passenger selections. The orientation of parked vehicles, occluded curbs and faint paint matter even when the target pixels are correctly classified as asphalt.

The less restrictive candidate generator put at least one point inside the reference on four of eight visible gaps. The 9B selector chose that point on three of those four. Changing the selector alone could recover at most one additional hit with that candidate bank. Evaluation should measure both whether a correct candidate exists and whether the language model selects it. The sample was deliberately chosen to exercise these cases. It measures neither general UAE driving accuracy nor stopping safety.

Each record lists the original source, image hash, frame ID, passenger wording, reference polygon, model outcomes and original-pixel points. The five EMT frames come from the dataset's training partition and may previously have been seen by the fine-tuned vision detector. The requests are new passenger-grounding evaluations. [Per-photo results](passenger-uae-road-results.json)
