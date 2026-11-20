# Real training-data bootstrap for passenger grounding

The Task 5 pilot can start without waiting for the full Task 2 corpus. The bootstrap joins original public training images, destination masks and commands from Talk2Car-RegSeg. It is a training-only subset, not a final evaluation set or a demonstration that the grounding model works.

## Run on a machine with space

After installing the project's lightweight dependencies, including Pillow:

```text
python -m tools.dropoff_bootstrap_regseg --out data/dropoff-regseg-pilot --limit 10 --image-prefix-mib 4 --max-download-mib 128
```

Use a new output directory. The tool refuses to overwrite an existing directory. It writes cases.jsonl, untouched selected JPEGs, native PNG masks and a source/download receipt. Image and mask paths are relative to cases.jsonl, so the directory can be moved to the training GPU.

The release ZIP contains two deflated TAR members, not separately accessible JPEG ZIP members. The tool fetches a bounded compressed prefix of images.tar and the complete small masks.tar member. It takes only fully decoded training JPEGs actually present in that prefix and matching masks/commands. It never downloads the complete 1.49 GB ZIP. Random TAR ordering means a small prefix is a convenience pilot sample, not a representative random sample; fewer records than requested remain explicit in the receipt.

An ordinary fresh invocation transfers approximately 97 MB of public metadata, compressed mask bytes and a 4 MiB image prefix, below the default 128 MiB cumulative ceiling. Full metadata is parsed in memory; only selected examples and receipts are saved. Servers must return the requested HTTP ranges with the expected source lengths. Range, decompression, CRC and source-hash failures stop the operation.

## Important release discrepancy and exclusions

The paper describes the concise test using validation images. However, the [pinned official create_dataset.py](https://github.com/kanji1011/Talk2car-Refseg/blob/cebce1943ce373857cc9be5def68ded4008c29e1/dataloader/create_dataset.py) loops over **both train and val**. Its associated files enumerate 170 training-source commands and 338 validation-source commands, totalling the actual 508-record concise test.

Excluding only those 170 renamed training images is insufficient: other commands and nearby frames can belong to the same source scene. The bootstrap therefore joins all 8,349 RegSeg training entries to [original Talk2Car train commands](https://github.com/talk2car/Talk2Car/blob/3fde52aee3341312196ab9b2f999337eff0693a0/data/commands/train_commands.json) using command_token, t2c_img and exact command text. That file supplies scene_token and sample_token.

The verified training release spans 587 source scenes. The 170 concise-test-derived training IDs touch 58 scenes, containing 1,146 training rows. **All 58 scenes are excluded from the bootstrap**, not just the 170 exact rows. All validation/original-test images are excluded by the training-only member selection. The 338 validation-source concise-test records never enter training.

This makes source grouping auditable, but the pilot still contains only training examples. Task 2 must choose disjoint development/evaluation scene groups for the final experiment and preserve this exclusion rule. Do not call the pilot a held-out result.

## Pinned source receipts

| Source | Pin / measured SHA256 |
| --- | --- |
| RegSeg repository | cebce1943ce373857cc9be5def68ded4008c29e1 |
| Full RegSeg metadata, 78,543,821 bytes | c94b741178b0cf8955014e43229fd0eb9afb384c29269b7c071e9858b402a3f8 |
| Metadata counts | train 8,349; val 1,163; original test 3,610. The original test is different from the 508-record concise release. |
| Concise-test train-source list | 45fa1baa6f1a48d2b300ac3604016e4cc1c8bc29ec7b8723068998ea3fd10653 |
| Concise-test val-source list | 052c34427f228f06094605f2133a28073a7908b4445a49747fef7ae29a5d7a15 |
| Talk2Car repository | 3fde52aee3341312196ab9b2f999337eff0693a0 |
| Talk2Car training metadata, 4,891,514 bytes | 5a90c471284685bd8dd8bfa6e992cfefce69e58a392c934d2f9d54ea119a42d2 |
| Release archive size | 1,485,242,059 bytes |
| ZIP central-directory SHA256 | 32fe81c35bed2cac8b675fdeed7421bd39a2c8f40dd8e148c6cd4a2567283134 |
| Complete compressed masks.tar member SHA256 | cd1b6320199a09020b1a10047e3d6e44e0b27328ec6d8f54063c3ac9ac2b1092 |

The full outer ZIP was not downloaded and its complete SHA256 is not verified. The tool pins its size/central directory and complete mask-member SHA, checks the mask TAR's CRC/length, and records every selected image/mask hash and requested byte range. A transport ETag is not substituted for a content SHA256.

## Supervision and coordinate rules

The original loader defines mask foreground as grayscale greater than zero. Preserve the native mask; do not bilinearly interpolate it to derive labels.

The destination box is the foreground extent with exclusive right/bottom edges, mapped from native mask coordinates to original image pixels. The point is the centre of a foreground pixel nearest the foreground-box centre. This is guaranteed to be inside the mask even when a non-convex mask's naive centroid would be outside it.

These public rows supervise only destination_point_xy and destination_box_xyxy. They do not invent ambiguity, landmark identities, clarification replies or a universal drop-off intent. The training formatter must use its destination-only task format or appropriate loss masking. Reference masks and targets stay outside the inference request.

The image-only request has no detector observations and explicitly records detector_run:false. Its source value describes the non-annotated input channel; it is not evidence that RF-DETR ran on these public images.

Keep the dataset/extension attribution and the [source Talk2Car data licence](https://github.com/talk2car/Talk2Car/blob/3fde52aee3341312196ab9b2f999337eff0693a0/LICENSE), including the non-commercial/share-alike conditions. Source repository code licensing is a separate matter.

## Local verification on 20 November 2026

Bounded source reads and the same extraction/export functions produced ten real training cases on ten verified source scenes: indices 5535, 1385, 4834, 3934, 3662, 6119, 4260, 1164, 5347 and 3674. Each joined image and mask is 1600 by 900.

A subsequent complete CLI invocation, using fresh bounded HTTP reads and the pinned hashes without overrides, also succeeded. It fetched exactly 97,222,464 bytes in 11 requests and exported the same ten case identities, source scenes, image/mask hashes, commands and targets. Its receipt and execution record are under ignored data/dropoff-regseg-cli-pilot-20261120. This was a real source-download/export check, not a transport replay or model execution.

The compact configs/dropoff/regseg-scene-guards.json binds all 8,349 training indices to 587 source scenes and lists the 58 reserved scenes and 170 exact concise-test train indices. It also contains a SHA256 of each exact official training command so edited utterances cannot silently retain public-annotation status. Its measured SHA256 is 27a218f0f71c8bae59f9c31f603e4a531e687c7ee42596d359e2e466d87fadfb. It contains source indices, hashes and provenance, not command text.

An earlier pre-scene-audit pilot was quarantined after two cases were found to share concise-test source scenes. Its old manifest is labelled DO-NOT-TRAIN. The corrected corpus is under ignored data/dropoff-regseg-scene-pilot-20261120 locally.

The downloader tests use generated archive/image fixtures to check transport bounds, directory identity, safe paths, incomplete prefixes, source grouping, partial supervision and non-convex masks. Those tests are software checks. No model inference, gradient update, trained-checkpoint reload or model-quality measurement is claimed by this bootstrap.
