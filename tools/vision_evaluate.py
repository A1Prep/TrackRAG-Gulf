"""Evaluate Task 1 predictions against EMT labels: detection (pycocotools) and tracking (motmetrics).

Run in the evaluation environment (numpy 1.26.4, pandas 2.2.3, motmetrics 1.4.0, pycocotools 2.0.11):

    python -m tools.vision_evaluate --pred-dir RUN --labels-root LABELS --split test \
        --recordings video_122233 ... --out RUN/metrics.json [--tracking-only]

Inputs per recording in RUN: <rec>.tracks.jsonl (TrackObservation rows), <rec>.detections.jsonl
(per-frame pipeline detections with scores, written by ``python -m trackrag.vision predict``) and
<rec>.provenance.json (evaluated frame ids and image size). Labels are read from
LABELS/annotations/<split>/<rec>.txt; rows whose class is not one of the 9 EMT classes are skipped
and boxes are clipped to the image. With --tracking-only, detections are never opened and
detection metrics, detector-to-track coverage and detection hashes are omitted.

Rules (identical for every model):
* Frames: exactly the frames listed in the prediction provenance.
* Detection: COCO AP from the pipeline detections ("mapped": after NMS, class mapping and the
  camera-car filter) at three label granularities: class-agnostic road user, the 4 EMT groups
  (Pedestrian, Cyclist, Motorbike, Vehicle) used by the dataset authors, and the 9 EMT classes.
  Boxes are scaled to a 1280-pixel long side before scoring, so COCO small/medium/large buckets
  mean the same thing for 2560x1440 and 3840x2160 recordings (IoU is unchanged by scaling).
  Default COCO maxDets (1, 10, 100); EMT frames hold at most 31 labelled boxes.
* Operating point: precision/recall/F1 at IoU >= 0.5 for detections with score >= 0.4, greedy by score.
* Tracking: motmetrics CLEAR + identity metrics on the exported tracks, IoU >= 0.5, class-agnostic and
  per group. MOTP here is mean (1 - IoU) of matches: lower is better.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tools.vision_prepare_dataset import parse_label_rows
from tools.vision_saved_adapters import load_track_artifact, validated_box
from trackrag.vision.classes import EMT_CLASSES, GROUP_OF, GROUPS
from trackrag.vision.geometry import clip_box

LONG_SIDE = 1280
OPERATING_SCORE = 0.4
MATCH_IOU = 0.5
TRACKING_METRICS = ["num_frames", "num_unique_objects", "num_objects", "num_predictions", "num_matches",
                    "num_misses", "num_false_positives", "num_switches", "num_fragmentations", "mota", "motp",
                    "idf1", "idp", "idr", "precision", "recall", "mostly_tracked", "partially_tracked", "mostly_lost"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_recording(pred_dir: Path, labels_root: Path, split: str, recording: str,
                   tracking_only: bool = False) -> dict:
    provenance = json.loads((pred_dir / f"{recording}.provenance.json").read_text(encoding="utf-8"))
    track_rows, frames, size, tracks_sha = load_track_artifact(
        pred_dir / f"{recording}.tracks.jsonl", provenance, recording)
    width, height = size
    label_path = labels_root / "annotations" / split / f"{recording}.txt"
    gt: dict[int, list] = defaultdict(list)
    skipped = Counter()
    wanted = set(frames)
    for frame, track, name, box in parse_label_rows(label_path.read_text(encoding="utf-8-sig"), str(label_path)):
        if frame not in wanted:
            continue
        if name not in EMT_CLASSES:
            skipped[name] += 1
            continue
        clipped = clip_box(box, width, height)
        if clipped is not None:
            gt[frame].append((track, name, clipped))
    detections: dict[int, list] = defaultdict(list)
    files = {"tracks.jsonl": tracks_sha, "provenance.json": sha256(pred_dir / f"{recording}.provenance.json"),
             "labels": sha256(label_path)}
    if not tracking_only:
        detection_path = pred_dir / f"{recording}.detections.jsonl"
        payload = detection_path.read_bytes()
        seen_frames = set()
        for row in [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]:
            if not isinstance(row, dict):
                raise ValueError("Detection frame must be an object")
            frame_id = row.get("frame_id")
            if (row.get("recording_id") != recording or row.get("source", "predicted") != "predicted"
                    or type(frame_id) is not int or frame_id not in wanted):
                raise ValueError("Detection row has a foreign recording, source or frame")
            if frame_id in seen_frames:
                raise ValueError("Duplicate detection frame")
            seen_frames.add(frame_id)
            if (type(row.get("width")) is not int or type(row.get("height")) is not int
                    or (row["width"], row["height"]) != (width, height)):
                raise ValueError(f"{recording}: frame size changed inside the recording")
            if not isinstance(row.get("mapped"), list):
                raise ValueError("Detection frame requires a mapped detections list (possibly empty)")
            for detection in row["mapped"]:
                if not isinstance(detection, dict):
                    raise ValueError("Mapped detection must be an object")
                score = detection.get("score")
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Detection score must be finite and within [0, 1]")
                if detection.get("class_name") not in EMT_CLASSES:
                    raise ValueError("Mapped detection uses an unknown EMT class")
                box = validated_box(detection.get("bbox_xyxy"), width, height)
                detections[frame_id].append((box, float(score), detection["class_name"]))
        if seen_frames != wanted:
            raise ValueError("Detection frame coverage differs from provenance, including empty frames")
        files["detections.jsonl"] = hashlib.sha256(payload).hexdigest()
    tracks: dict[int, list] = defaultdict(list)
    for row in track_rows:
        tracks[row["frame_id"]].append((row["track_id"], row["class_name"], row["bbox_xyxy"]))
    return {"recording": recording, "frames": frames, "size": (width, height), "gt": gt,
            "detections": detections, "tracks": tracks, "skipped_gt": dict(skipped), "provenance": provenance,
            "files": files}



def _label(name: str, granularity: str) -> str:
    if granularity == "agnostic":
        return "road_user"
    if granularity == "group":
        return GROUP_OF[name]
    return name


def coco_detection(recordings: list[dict], granularity: str) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    names = {"agnostic": ["road_user"], "group": list(GROUPS), "class": list(EMT_CLASSES)}[granularity]
    category = {name: index + 1 for index, name in enumerate(names)}
    images, annotations, results, gt_counts = [], [], [], Counter()
    for data in recordings:
        width, height = data["size"]
        scale = LONG_SIDE / max(width, height)
        for frame in data["frames"]:
            image_id = len(images) + 1
            images.append({"id": image_id, "width": round(width * scale), "height": round(height * scale)})
            for _, name, (l, t, r, b) in data["gt"].get(frame, []):
                label = _label(name, granularity)
                w, h = (r - l) * scale, (b - t) * scale
                annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": category[label],
                                    "bbox": [l * scale, t * scale, w, h], "area": w * h, "iscrowd": 0})
                gt_counts[label] += 1
            for (l, t, r, b), score, name in data["detections"].get(frame, []):
                if name not in GROUP_OF:
                    continue
                results.append({"image_id": image_id, "category_id": category[_label(name, granularity)],
                                "bbox": [l * scale, t * scale, (r - l) * scale, (b - t) * scale], "score": score})
    ground_truth = COCO()
    ground_truth.dataset = {"images": images, "annotations": annotations,
                            "categories": [{"id": i, "name": n} for n, i in category.items()]}
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth.createIndex()
        if results:
            predictions = ground_truth.loadRes(results)
        else:  # pycocotools.loadRes([]) raises; an empty prediction set must score as all misses
            predictions = COCO()
            predictions.dataset = {**ground_truth.dataset, "annotations": []}
            predictions.createIndex()
        evaluator = COCOeval(ground_truth, predictions, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            evaluator.summarize()
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100",
            "AR_small", "AR_medium", "AR_large"]
    summary = {key: float(value) for key, value in zip(keys, evaluator.stats)}
    precision = evaluator.eval["precision"]  # [IoU thresholds, recall points, classes, areas, maxDets]
    per_class = {}
    for name, cat_id in category.items():
        k = evaluator.params.catIds.index(cat_id)
        all_iou = precision[:, :, k, 0, -1]
        at_50 = precision[0, :, k, 0, -1]
        per_class[name] = {
            "gt_boxes": gt_counts[name],
            "AP": float(all_iou[all_iou > -1].mean()) if (all_iou > -1).any() else None,
            "AP50": float(at_50[at_50 > -1].mean()) if (at_50 > -1).any() else None,
        }
    return {"summary": summary, "per_class": per_class, "predictions": len(results), "gt_boxes": sum(gt_counts.values()),
            "note": "classes without ground truth are excluded by pycocotools (their false positives are not penalised)"}


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def operating_point(recordings: list[dict], score: float = OPERATING_SCORE) -> dict:
    """Greedy (by score) one-to-one matching per frame and group at IoU >= 0.5."""
    counts = {group: Counter() for group in (*GROUPS, "road_user")}
    for data in recordings:
        for frame in data["frames"]:
            for group in (*GROUPS, "road_user"):
                gts = [box for _, name, box in data["gt"].get(frame, [])
                       if group == "road_user" or GROUP_OF[name] == group]
                preds = sorted(((s, box) for box, s, name in data["detections"].get(frame, [])
                                if s >= score and name in GROUP_OF and (group == "road_user" or GROUP_OF[name] == group)),
                               key=lambda item: -item[0])
                used = set()
                tp = 0
                for _, box in preds:
                    best, best_iou = None, MATCH_IOU
                    for index, gt in enumerate(gts):
                        if index not in used:
                            overlap = _iou(box, gt)
                            if overlap >= best_iou:
                                best, best_iou = index, overlap
                    if best is not None:
                        used.add(best)
                        tp += 1
                counts[group].update(tp=tp, fp=len(preds) - tp, fn=len(gts) - tp)
    result = {}
    for group, c in counts.items():
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if precision is not None and recall is not None else None)
        result[group] = {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "precision": precision, "recall": recall, "f1": f1}
    return {"score_threshold": score, "iou": MATCH_IOU, "groups": result}


def tracking(recordings: list[dict], *, groups: tuple[str, ...] = ("road_user", *GROUPS)) -> dict:
    import motmetrics as mm
    import numpy as np

    def xywh(box):
        return [box[0], box[1], box[2] - box[0], box[3] - box[1]]

    results = {}
    for group in groups:
        accumulators, names = [], []
        for data in recordings:
            accumulator = mm.MOTAccumulator(auto_id=False)
            for frame in data["frames"]:
                gts = [(track, box) for track, name, box in data["gt"].get(frame, [])
                       if group == "road_user" or GROUP_OF[name] == group]
                preds = [(track, box) for track, name, box in data["tracks"].get(frame, [])
                         if group == "road_user" or GROUP_OF.get(name) == group]
                distances = mm.distances.iou_matrix(np.asarray([xywh(b) for _, b in gts], dtype=float).reshape(-1, 4),
                                                    np.asarray([xywh(b) for _, b in preds], dtype=float).reshape(-1, 4),
                                                    max_iou=1 - MATCH_IOU)
                accumulator.update([t for t, _ in gts], [t for t, _ in preds], distances, frameid=frame)
            accumulators.append(accumulator)
            names.append(data["recording"])
        if not any(len(a.events) for a in accumulators):
            continue
        summary = mm.metrics.create().compute_many(accumulators, names=names, metrics=TRACKING_METRICS,
                                                   generate_overall=True)
        results[group] = json.loads(summary.to_json(orient="index"))
    return results


def track_coverage(recordings: list[dict], score: float = OPERATING_SCORE) -> dict:
    """Share of detections with score >= 0.4 that the tracker exported in the same frame (exact box)."""
    total = kept = 0
    for data in recordings:
        for frame in data["frames"]:
            exported = {tuple(round(v, 4) for v in box) for _, _, box in data["tracks"].get(frame, [])}
            for box, s, _ in data["detections"].get(frame, []):
                if s >= score:
                    total += 1
                    kept += tuple(round(v, 4) for v in box) in exported
    return {"detections": total, "exported": kept, "coverage": kept / total if total else None}


def evaluate(pred_dir: Path, labels_root: Path, split: str, recordings: list[str], tracking_only: bool = False) -> dict:
    if not recordings or len(set(recordings)) != len(recordings):
        raise ValueError("Evaluation requires unique recording IDs")
    data = [load_recording(pred_dir, labels_root, split, recording, tracking_only) for recording in recordings]
    result = {
        "version": 1, "pred_dir": str(pred_dir), "split": split, "recordings": recordings,
        "tracking_only": tracking_only,
        "identity_hashes": sorted({d["provenance"].get("identity_hash") for d in data}),
        "frames": {d["recording"]: len(d["frames"]) for d in data},
        "skipped_gt_rows": {d["recording"]: d["skipped_gt"] for d in data},
        "inputs_sha256": {d["recording"]: d["files"] for d in data},
        "rules": {"match_iou": MATCH_IOU, "operating_score": OPERATING_SCORE, "coco_scale_long_side": LONG_SIDE,
                  "coco_max_dets": [1, 10, 100], "motp": "mean (1 - IoU), lower is better"},
        "tracking": tracking(data),
    }
    if not tracking_only:
        result["track_coverage"] = track_coverage(data)
        result["detection"] = {granularity: coco_detection(data, granularity) for granularity in ("agnostic", "group", "class")}
        result["detection_per_recording"] = {d["recording"]: coco_detection([d], "agnostic")["summary"] for d in data}
        result["operating_point"] = operating_point(data)
    return result


def brief(result: dict) -> str:
    lines = []
    overall = result["tracking"].get("road_user", {}).get("OVERALL", {})
    if overall:
        lines.append("tracking (road users): IDF1 {idf1:.3f}  MOTA {mota:.3f}  recall {recall:.3f}  precision {precision:.3f}  "
                     "switches {num_switches:.0f}".format(**overall))
    if "detection" in result:
        for granularity in ("agnostic", "group"):
            s = result["detection"][granularity]["summary"]
            lines.append(f"detection ({granularity}): AP {s['AP']:.3f}  AP50 {s['AP50']:.3f}  "
                         f"AP_small {s['AP_small']:.3f}  AP_medium {s['AP_medium']:.3f}  AP_large {s['AP_large']:.3f}")
    if "track_coverage" in result:
        lines.append(f"track coverage of score>=0.4 detections: {result['track_coverage']['coverage']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--recordings", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tracking-only", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new file")
    result = evaluate(args.pred_dir, args.labels_root, args.split, args.recordings, args.tracking_only)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False, default=float) + "\n", encoding="utf-8")
    print(brief(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
