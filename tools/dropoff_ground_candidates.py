"""Experimental CPU semantic-ground candidates; never consumes destination labels.

The road mask is a prediction, not evidence of legal/safe stopping. Fixed spatial
tiling deliberately makes candidate selection independent of the instruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw

MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
MODEL_REVISION = "21b3847fae21ddee674abd31129307b6a1235bd9"
WEIGHTS_SHA256 = "027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c"
RESERVED_RECORDINGS = {"video_122233", "video_125233", "video_155425", "video_160325", "video_204647"}
RULE = {
    "road_class_id": 0, "sidewalk_class_id": 1, "minimum_road_probability": 0.5,
    "grid_rows": 6, "grid_columns": 6, "minimum_tile_pixels": 25,
    "patch_maximum_half_width_fraction": 0.025,
    "patch_maximum_half_height_fraction": 0.025,
    "selection": "maximum distance from non-road inside each fixed spatial tile",
    "labels_used": False, "instruction_used": False,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_candidates(classes, road_probability, exclusion_boxes=()):
    """Use only predicted classes/confidence and optional predicted object boxes.

    Boxes are half-open original-image coordinates. Every returned patch pixel
    must belong to the retained predicted-road mask. Tiling is not semantic
    destination reasoning; its coverage must be measured before adoption.
    """
    classes = np.asarray(classes)
    road_probability = np.asarray(road_probability)
    if classes.ndim != 2 or classes.shape != road_probability.shape:
        raise ValueError("matching two-dimensional class and probability arrays required")
    if not np.isfinite(road_probability).all() or np.any((road_probability < 0) | (road_probability > 1)):
        raise ValueError("probabilities must be finite and in [0,1]")
    height, width = classes.shape
    if not height or not width:
        raise ValueError("empty prediction")
    ground = ((classes == 0) & (road_probability >= RULE["minimum_road_probability"]))
    for raw in exclusion_boxes:
        if len(raw) != 4 or not all(math.isfinite(float(value)) for value in raw):
            raise ValueError("invalid exclusion box")
        x1, y1, x2, y2 = map(float, raw)
        if x1 > x2 or y1 > y2:
            raise ValueError("reversed exclusion box")
        ground[max(0, min(height, math.floor(y1))):max(0, min(height, math.ceil(y2))),
               max(0, min(width, math.floor(x1))):max(0, min(width, math.ceil(x2)))] = False
    ys, xs = np.nonzero(ground)
    if not len(xs):
        return [], ground
    edges_x = np.linspace(xs.min(), xs.max() + 1, 7).astype(int)
    edges_y = np.linspace(ys.min(), ys.max() + 1, 7).astype(int)
    result = []
    for gy in range(6):
        for gx in range(6):
            left, right = int(edges_x[gx]), int(edges_x[gx + 1])
            top, bottom = int(edges_y[gy]), int(edges_y[gy + 1])
            tile = ground[top:bottom, left:right].astype(np.uint8)
            if int(tile.sum()) < RULE["minimum_tile_pixels"]:
                continue
            # Explicit zero padding prevents OpenCV treating all-ones borders
            # as infinite distance. No destination mask participates here.
            distance = cv2.distanceTransform(np.pad(tile, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            ty, tx = np.unravel_index(np.argmax(distance), distance.shape)
            x, y = left + int(tx), top + int(ty)
            half_w = max(0, min(int(width * 0.025), int(distance[ty, tx] / math.sqrt(2)) - 1))
            half_h = max(0, min(int(height * 0.025), int(distance[ty, tx] / math.sqrt(2)) - 1))
            while not ground[y-half_h:y+half_h+1, x-half_w:x+half_w+1].all():
                half_w = max(0, half_w - 1)
                half_h = max(0, half_h - 1)
            box = [x-half_w, y-half_h, x+half_w+1, y+half_h+1]
            point = [x + 0.5, y + 0.5]
            result.append({
                "candidate_id": f"R{gy*6+gx+1:02d}",
                "point_original_px": point, "bbox_original_px": box,
                "point_normalized": [point[0]/width, point[1]/height],
                "bbox_normalized": [box[0]/width, box[1]/height, box[2]/width, box[3]/height],
                "semantic_class": "road", "predicted_ground_fraction": 1.0,
                "point_road_probability": float(road_probability[y, x]),
                "tile": [gx, gy],
            })
    return result, ground


def prepare_model(cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        output = cache_dir / filename
        if not output.exists():
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{filename}"
            with urllib.request.urlopen(url, timeout=45) as source, output.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
    if sha256(cache_dir / "pytorch_model.bin") != WEIGHTS_SHA256:
        raise ValueError("downloaded model weight digest does not match pinned upstream digest")
    # weights_only=True prevents arbitrary pickle execution in this old .bin
    # model. No remote code is enabled.
    import torch
    from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor
    config = SegformerConfig.from_pretrained(cache_dir, local_files_only=True)
    model = SegformerForSemanticSegmentation(config)
    state = torch.load(cache_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    processor = SegformerImageProcessor.from_pretrained(cache_dir, local_files_only=True, do_reduce_labels=False)
    return model, processor


def run(requests_path, output_dir, cache_dir):
    import torch
    import torch.nn.functional as functional
    torch.set_num_threads(4)
    requests_path = Path(requests_path).resolve()
    rows = [json.loads(line) for line in requests_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows or len(rows) > 32:
        raise ValueError("exploratory trial requires 1..32 requests")
    if any(row["scene"]["frame"]["recording_id"] in RESERVED_RECORDINGS for row in rows):
        raise ValueError("reserved final recording cannot enter exploratory trial")
    # Strong structural boundary: this executable accepts annotation-free
    # requests, never the full training/development case objects.
    from tools.dropoff_predict import read_requests
    rows = read_requests(requests_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol.json").write_text(json.dumps({
        "kind": "exploratory_predicted_ground_candidates_not_stopping_safety",
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "weight_sha256": WEIGHTS_SHA256, "device": "cpu", "rule": RULE,
        "requests_sha256": sha256(requests_path), "request_ids": [r["request_id"] for r in rows],
        "targets_read": False, "reserved_final_data_used": False,
    }, indent=2) + "\n", encoding="utf-8")
    start = time.perf_counter()
    model, processor = prepare_model(cache_dir)
    model_load_s = time.perf_counter() - start
    cached = {}
    outputs = []
    for row in rows:
        frame = row["scene"]["frame"]
        image_path = Path(frame["image_path"])
        digest = sha256(image_path)
        boxes = [track["bbox_xyxy"] for track in row["scene"]["tracks"]
                 if track["frame_id"] == frame["frame_id"] and track["recording_id"] == frame["recording_id"]]
        boxes_digest = hashlib.sha256(json.dumps(boxes, sort_keys=True).encode()).hexdigest()
        cache_key = (digest, boxes_digest)
        if cache_key not in cached:
            image = Image.open(image_path).convert("RGB")
            if image.size != (frame["width"], frame["height"]):
                raise ValueError("request dimensions do not match original image")
            started = time.perf_counter()
            inputs = processor(images=image, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**inputs).logits
                logits = functional.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
                probabilities = logits.softmax(dim=1)
                classes = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
                road_probability = probabilities[0, 0].numpy()
            # Only genuine predicted boxes at the current frame are excluded.
            candidates, ground = build_candidates(classes, road_probability, boxes)
            stem = digest[:16] + "-" + boxes_digest[:8]
            paths = {key: str(output_dir / f"{stem}-{name}.png") for key, name in (
                ("overlay_path", "overlay"), ("class_mask_path", "classes"),
                ("road_mask_path", "road"), ("sidewalk_mask_path", "sidewalk"))}
            Image.fromarray(classes).save(paths["class_mask_path"])
            Image.fromarray(ground.astype(np.uint8)*255).save(paths["road_mask_path"])
            Image.fromarray((classes == 1).astype(np.uint8)*255).save(paths["sidewalk_mask_path"])
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            for candidate in candidates:
                x, y = candidate["point_original_px"]
                draw.rectangle(candidate["bbox_original_px"], outline="cyan", width=max(2, image.width//700))
                r = max(4, image.width//250)
                draw.ellipse([x-r,y-r,x+r,y+r], fill="yellow", outline="black", width=2)
                draw.text((x+r,y-r), candidate["candidate_id"], fill="yellow", stroke_width=2, stroke_fill="black", font_size=max(14,image.width//90))
            overlay.save(paths["overlay_path"])
            np.savez_compressed(output_dir / f"{stem}-probabilities.npz", road_probability=road_probability.astype(np.float16))
            cached[cache_key] = {"image_path": str(image_path), "image_sha256": digest, **paths,
                "candidates": candidates, "elapsed_s": time.perf_counter()-started,
                "predicted_road_pixels": int(ground.sum()), "predicted_sidewalk_pixels": int((classes==1).sum()),
                "exclusion_boxes_sha256": boxes_digest,
                "excluded_predicted_road_pixels": int(((classes==0) & (road_probability>=0.5) & ~ground).sum())}
        outputs.append({"request_id": row["request_id"], **cached[cache_key]})
    (output_dir / "candidates.jsonl").write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in outputs), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps({
        "requests": len(rows), "unique_images": len(cached), "model_load_s": model_load_s,
        "total_elapsed_s": time.perf_counter()-start,
        "candidate_counts": {row["request_id"]: len(row["candidates"]) for row in outputs},
    }, indent=2)+"\n", encoding="utf-8")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    args = parser.parse_args()
    run(args.requests, args.output, args.cache_dir)


if __name__ == "__main__":
    main()
