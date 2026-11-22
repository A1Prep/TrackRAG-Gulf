# Passenger destination density experiment

An 88,473-parameter network learns to place a destination point and predict four distances to the destination-region edges. The input is a road image and a passenger instruction. A frozen SegFormer-B0 encodes 19 Cityscapes class probabilities. The network combines them with RGB, image position and trainable instruction features at three scales. The selected point must lie on predicted road and outside any supplied current-frame object boxes. The predicted coarse box can include other surfaces. Passenger confirmation and dialogue are separate application stages.

The training run used the 500 public Talk2Car-RegSeg commands on 295 training scenes. Each network ran for 20 fixed epochs and 640 AdamW updates. The image-dependent region and the vocabulary were learned from the training partition. Training used the native destination masks, mirrored images and mirrored left/right terms. There was no development-based checkpoint selection. The previously used public development set contains 125 commands across 65 other scenes.

| Development condition | Point in original region | Mean coarse-box IoU | Same-image point hits | Entire contrast groups correct |
| --- | ---: | ---: | ---: | ---: |
| Image and actual instruction | 28/125 | 0.105 | 4/13 | 1/6 |
| Independently trained image-only control | 20/125 | 0.079 | 2/13 | 1/6 |
| Frozen first network, instructions permuted across scenes | 11/125 | 0.050 | 0/13 | 0/6 |

Every case yielded geometry under each condition. Relative to the shuffled-text control, 21 cases succeeded only with the original text and four succeeded only with the shuffled text. This shows sensitivity to the instruction. The public masks alone cannot establish whether a clarification would be appropriate. The standalone spatial model therefore remains an experimental comparison.

A fresh process loaded the saved checkpoint and produced four geometry outputs across EMT recordings `video_054604` and `video_054907`. The four requests include the ambiguous utterance “here.” All four produced geometry because this component has no status or dialogue supervision. The test demonstrates real model execution. It supplies no passenger-verified location accuracy.

The pinned semantic encoder produced 609 unique image features. A second CPU extraction compared every quantized feature tensor to the cache. A second scoring implementation checked all 375 saved point predictions against the original mask pixels. The public data source, scene exclusions and non-commercial conditions are listed in the [data handoff](dropoff-data-source.md). The complete counts, hashes and paired comparisons are in the [run summary](../experiments/dropoff/passenger-spatial-results-20261121.json).

## Reproduce the experiment

From the repository root use the optional CPU packages in `requirements-spatial.txt` and a fresh output directory. The vision cache stores the pinned SegFormer configuration and weights. The feature cache is separate.

```text
python -m tools.dropoff_spatial_cache --train data/dropoff-public-expanded-20261121/splits/train.jsonl --development data/dropoff-public-expanded-20261121/splits/development.jsonl --vision-cache <segformer-cache> --output <feature-cache> --verify-existing
python -m tools.dropoff_spatial_fit --train data/dropoff-public-expanded-20261121/splits/train.jsonl --development data/dropoff-public-expanded-20261121/splits/development.jsonl --cache <feature-cache> --output <run>
python -m tools.dropoff_spatial_controls --development data/dropoff-public-expanded-20261121/splits/development.jsonl --cache <feature-cache> --fit <run> --output <control-run>
```

For inference use `SpatialGrounder(checkpoint_directory, vision_cache=..., expected_manifest_sha256=...)` from `trackrag.dropoff.spatial`. Call `predict(rgb_image, instruction, exclusion_boxes=...)`. The return contains an original-pixel point, box, uncalibrated peak mass and predicted-road probability. A scene with no eligible point yields `NoVisibleGround`. The published experimental checkpoint uses manifest `3f186895b9c80208a00d9332053846fc396c1986cc26af2d9f6e8d31a36e09be` and weights `7300ee2861e4d7a5e140b7b86c983699f93a10086817e7df399e2bca2b53d74e`.

```text
python -m tools.dropoff_spatial_probe --requests <hash-bound-predicted-requests.jsonl> --checkpoint <run>/instruction_conditioned --vision-cache <segformer-cache> --expected-manifest-sha256 3f186895b9c80208a00d9332053846fc396c1986cc26af2d9f6e8d31a36e09be --output <new-probe-run>
```

## Selecting between learned destination points

The two frozen spatial models sometimes put their points in different parts of the road. A separate logistic classifier learned to choose between those two points from their proposed locations, extent, agreement and visual support. Its training target was whether each point landed inside the participant's original mask. Every development prediction used a classifier fitted without any examples from that prediction's source scene. The final classifier then used the full 125-case development partition for calibration. [Frozen study protocol](../experiments/dropoff/passenger-spatial-confidence-protocol-20261122.json)

On the 125 development cases, the classifier chose the instruction-conditioned model 85 times and the image-only model 40 times. Switching gained two correct regions and lost seven. The classifier placed 23 points inside a participant region, compared with 28 for always using the instruction-conditioned model. Requiring a predicted success probability of at least 0.30 accepted 41 points, of which seven landed inside the region. These development results support retaining the deployed configuration.

The separate study used twelve previously unused public scenes with original passenger instructions and human-marked destination masks. Each condition saw the same image and instruction, with no detector observations. The deployed Qwen application offered four points, three inside the passenger's region. The instruction-conditioned spatial model and the learned selector each offered twelve points with four hits. The image-only model had three hits in twelve. The selector with the predeclared confidence threshold offered four points with three hits. All cases stayed in the denominator. These twelve images measure destination placement. They have no human labels for when to clarify. [Full matched results](../experiments/dropoff/passenger-spatial-confidence-results-20261122.json)

```text
python -m tools.dropoff_spatial_confidence fit --cases data/dropoff-public-expanded-20261121/splits/development.jsonl --heads <two-trained-heads> --out <new-calibration>
python -m tools.dropoff_spatial_confidence predict --requests <fresh-bank>/inputs/requests.jsonl --heads <two-trained-heads> --vision-cache <segformer-cache> --model <new-calibration>/calibrator.json --out <new-cpu-predictions.json>
python -m tools.dropoff_spatial_confidence score --predictions <new-cpu-predictions.json> --references <fresh-bank>/corpus/references.jsonl --bank <fresh-bank>/manifest.json --deployed-run <sealed-release-run> --vision-cache <segformer-cache> --out <new-scores.json>
```
