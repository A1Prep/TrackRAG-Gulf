"""Reproduce selection of the frozen short-gap tracker on the two EMT development drives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.vision_evaluate import load_recording, tracking
from trackrag.vision.config import PROJECT_ROOT, load_config
from trackrag.vision.continuity import TrackContinuity

DEVELOPMENT = ("video_054907", "video_174340")
GRID = ((3, 0.45), (6, 0.35), (10, 0.25), (16, 0.20), (25, 0.15))
AMBIGUITY_MARGIN = 0.08


def _evaluate(items):
    metrics = tracking(items, groups=("road_user",))["road_user"]["OVERALL"]
    return {key: metrics[key] for key in ("idf1", "num_switches", "num_predictions", "num_fragmentations",
                                          "precision", "recall", "mota")}


def select(pred_dir: Path, labels_root: Path) -> dict:
    inputs = [load_recording(pred_dir, labels_root, "train", recording, tracking_only=True)
              for recording in DEVELOPMENT]
    expected = load_config(PROJECT_ROOT / "configs/vision/finetuned.json")
    if any(item["provenance"].get("identity_hash") != expected.identity_hash()
           or item["provenance"].get("identity") != expected.identity() for item in inputs):
        raise ValueError("Selection requires the original fine-tuned detector and BoT-SORT profile")
    baseline = _evaluate(inputs)
    candidates = []
    for gap, min_iou in GRID:
        relabelled = []
        for item in inputs:
            profile = TrackContinuity(gap, min_iou, AMBIGUITY_MARGIN)
            frames = {}
            for frame in item["frames"]:
                rows = item["tracks"].get(frame, [])
                ids = profile.update(frame, rows)
                if len(set(ids)) != len(rows):
                    raise ValueError("Relabelling lost a track or assigned an ID twice")
                frames[frame] = [(canonical, name, box) for canonical, (_, name, box) in zip(ids, rows)]
            relabelled.append(dict(item, tracks=frames))
        candidates.append({"max_gap": gap, "min_iou": min_iou, "metrics": _evaluate(relabelled)})
    eligible = [row for row in candidates if row["metrics"]["num_switches"] < baseline["num_switches"]
                and row["metrics"]["idf1"] >= baseline["idf1"]]
    winner = min(eligible, default=None, key=lambda row: (-row["metrics"]["idf1"],
                                                         row["metrics"]["num_switches"], row["max_gap"]))
    return {"recordings": list(DEVELOPMENT), "inputs": {item["recording"]: item["files"] for item in inputs},
            "fixed_ambiguity_margin": AMBIGUITY_MARGIN, "baseline": baseline,
            "candidates": candidates, "winner": winner}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Choose a new result file")
    report = select(args.pred_dir, args.labels_root)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Development selection: {report['winner']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
